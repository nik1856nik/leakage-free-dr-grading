"""Ensembling of saved per-model probabilities. Weights/stackers are fit on VALIDATION only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def load_preds(run_dir: Path, tag: str):
    d = np.load(Path(run_dir) / f"{tag}_predictions.npz", allow_pickle=True)
    order = np.argsort(d["uid"])
    return d["y"][order], d["prob"][order], d["uid"][order]


def align(runs: list[Path], tag: str):
    ys, ps, uids = zip(*(load_preds(r, tag) for r in runs))
    for u in uids[1:]:
        assert (u == uids[0]).all(), f"{tag} predictions are not aligned across runs"
    return ys[0], np.stack(ps)  # (M, N, C)


def clinical_score(y, prob):
    pred = prob.argmax(1)
    return 0.5 * (cohen_kappa_score(y, pred, weights="quadratic") +
                  f1_score(y, pred, average="macro", zero_division=0))


def soft_vote(P):
    return P.mean(0)


def optimise_weights(y, P, n_iter: int = 3000, seed: int = 0):
    """Random search on the simplex (Dirichlet) maximising the clinical score on validation."""
    rng = np.random.default_rng(seed)
    M = P.shape[0]
    best_w, best = np.ones(M) / M, clinical_score(y, P.mean(0))
    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(M))
        s = clinical_score(y, np.tensordot(w, P, 1))
        if s > best:
            best, best_w = s, w
    return best_w, best


def weighted(P, w):
    return np.tensordot(w, P, 1)


def stacking(yv, Pv, Pt, seed: int = 0):
    """Multinomial LR on concatenated validation probabilities.
    Returns (cross-validated val probs for honest val scoring, test probs from full-val fit)."""
    Xv = Pv.transpose(1, 0, 2).reshape(Pv.shape[1], -1)
    Xt = Pt.transpose(1, 0, 2).reshape(Pt.shape[1], -1)
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pv = cross_val_predict(lr, Xv, yv, cv=cv, method="predict_proba")
    pt = lr.fit(Xv, yv).predict_proba(Xt)
    return pv, pt
