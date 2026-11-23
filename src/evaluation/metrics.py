"""Classification metrics for DR grading / detection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)


def _safe_auc(y, prob, n):
    try:
        if n == 2:
            return float(roc_auc_score(y, prob[:, 1]))
        present = np.unique(y)
        if len(present) < 2:
            return float("nan")
        return float(roc_auc_score(y, prob, multi_class="ovr", average="macro",
                                   labels=list(range(n))))
    except ValueError:
        return float("nan")


def compute_metrics(y: np.ndarray, prob: np.ndarray, class_names: list[str],
                    pred: np.ndarray | None = None) -> dict:
    """All metrics required by the study. ``prob`` is (N, C) class probabilities."""
    y = np.asarray(y).astype(int)
    n = prob.shape[1]
    pred = prob.argmax(1) if pred is None else np.asarray(pred)
    labels = list(range(n))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    # per-class specificity (one-vs-rest)
    spec = []
    for k in labels:
        tn = cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k]
        fp = cm[:, k].sum() - cm[k, k]
        spec.append(tn / (tn + fp) if tn + fp else float("nan"))
    out = dict(
        n=int(len(y)),
        accuracy=float(accuracy_score(y, pred)),
        macro_precision=float(p.mean()),
        macro_recall=float(r.mean()),
        macro_f1=float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        weighted_f1=float(f1_score(y, pred, average="weighted", labels=labels, zero_division=0)),
        macro_specificity=float(np.nanmean(spec)),
        roc_auc=_safe_auc(y, prob, n),
        qwk=float(cohen_kappa_score(y, pred, weights="quadratic")) if n > 2 else
            float(cohen_kappa_score(y, pred)),
        confusion_matrix=cm.tolist(),
        per_class={class_names[k]: dict(precision=float(p[k]), recall=float(r[k]), f1=float(f[k]),
                                        specificity=float(spec[k]), support=int(s[k]))
                   for k in labels},
        majority_baseline_acc=float(np.bincount(y, minlength=n).max() / len(y)),
    )
    # "screening" view for multi-class: any DR (grade>=1) and referable (>=2)
    if n == 5:
        for name, thr in (("anyDR", 1), ("referable", 2)):
            yt, yp = (y >= thr).astype(int), (pred >= thr).astype(int)
            tp = int(((yt == 1) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
            score = prob[:, thr:].sum(1)
            out[f"{name}_sensitivity"] = tp / (tp + fn) if tp + fn else float("nan")
            out[f"{name}_specificity"] = tn / (tn + fp) if tn + fp else float("nan")
            out[f"{name}_auc"] = _safe_auc(yt, np.stack([1 - score, score], 1), 2)
            out[f"{name}_false_negatives"] = fn
    if n == 2:
        cmf = cm
        out["sensitivity"] = float(cmf[1, 1] / cmf[1].sum()) if cmf[1].sum() else float("nan")
        out["specificity"] = float(cmf[0, 0] / cmf[0].sum()) if cmf[0].sum() else float("nan")
    else:
        out["sensitivity"] = out["macro_recall"]
        out["specificity"] = out["macro_specificity"]
    return out


def bootstrap_ci(y, prob, metric_fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for a scalar metric_fn(y, prob)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        try:
            vals.append(metric_fn(y[i], prob[i]))
        except ValueError:
            continue
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def qwk_from_prob(y, prob):
    return cohen_kappa_score(y, prob.argmax(1), weights="quadratic")


def macro_f1_from_prob(y, prob):
    return f1_score(y, prob.argmax(1), average="macro", zero_division=0)
