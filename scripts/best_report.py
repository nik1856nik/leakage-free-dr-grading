"""Detailed results for the frozen best ensemble (after `best_model.py --final`).

Adds bootstrap 95% CIs, per-class metrics, referable-DR accuracy, confusion matrices, plots, and a
context table of the individual candidate models on the same test split. The context table is
REPORTING ONLY: the ensemble was frozen on validation before any test image was read, and nothing
here feeds back into the selection.

    uv run python scripts/best_report.py --dataset aptos --external idrid messidor2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from best_model import NAMES, decide, member_preds, sel_score  # noqa: E402
from src.datasets.sources import ROOT  # noqa: E402
from src.evaluation import plots  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score  # noqa: E402


def referable(y, pred, prob):
    yt, yp = (y >= 2).astype(int), (pred >= 2).astype(int)
    return dict(accuracy=float((yt == yp).mean()), auc=float(roc_auc_score(yt, prob[:, 2:].sum(1))),
                sensitivity=float(yp[yt == 1].mean()), specificity=float(1 - yp[yt == 0].mean()),
                prevalence=float(yt.mean()))


def boot(y, pred, prob, n=2000, seed=0):
    r = np.random.default_rng(seed)
    fns = dict(accuracy=lambda a, b, c: (a == b).mean(),
               qwk=lambda a, b, c: cohen_kappa_score(a, b, weights="quadratic"),
               macro_f1=lambda a, b, c: f1_score(a, b, average="macro", labels=range(5), zero_division=0),
               referable_accuracy=lambda a, b, c: ((a >= 2) == (b >= 2)).mean(),
               referable_sensitivity=lambda a, b, c: (b[a >= 2] >= 2).mean(),
               referable_specificity=lambda a, b, c: (b[a < 2] < 2).mean(),
               referable_auc=lambda a, b, c: roc_auc_score(a >= 2, c[:, 2:].sum(1)))
    vals = {k: [] for k in fns}
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        for k, f in fns.items():
            try:
                vals[k].append(f(y[i], pred[i], prob[i]))
            except ValueError:
                pass
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aptos")
    ap.add_argument("--external", nargs="*", default=[])
    a = ap.parse_args()
    out = ROOT / "experiments" / f"best_{a.dataset}"
    plot_dir = ROOT / "results" / "plots" / "best_model"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sel = json.loads((out / "selection.json").read_text())
    mode, thr = sel["mode"], np.array(sel["thresholds"])
    members = [Path(m) for m in sel["members"]]

    report = dict(members=[m.stem for m in members], decision=mode, thresholds=thr.tolist(),
                  validation_cross_fitted=sel["val_cv"], evaluations={})
    for ds, split in [(a.dataset, "test")] + [(e, "all") for e in a.external]:
        P = [member_preds(c, ds, split, out) for c in members]
        y = P[0][0].astype(int)
        s = np.mean([p[2] for p in P], 0); prob = np.mean([p[1] for p in P], 0)
        pred = decide(prob, s, thr, mode)
        m = compute_metrics(y, prob, NAMES, pred=pred)
        m["referable"] = referable(y, pred, prob)
        m["ci95"] = boot(y, pred, prob)
        key = f"{ds}_{split}"
        report["evaluations"][key] = m
        plots.confusion(np.array(m["confusion_matrix"]), NAMES, plot_dir / f"confusion_{key}.png",
                        title=f"Best ensemble · {ds} {split} (n={len(y)})")
        rs = prob[:, 2:].sum(1)
        plots.roc_curves((y >= 2).astype(int), np.stack([1 - rs, rs], 1), ["non-referable", "referable DR"],
                         plot_dir / f"roc_referable_{key}.png", title=f"Referable DR · {ds} {split}")
        plots.roc_curves(y, prob, NAMES, plot_dir / f"roc_5class_{key}.png", title=f"Five-class (one-vs-rest) · {ds} {split}")

    # context: every candidate single model on the same test split (reporting only, not selection)
    ctx = []
    cands = sorted({c for cand in sel["candidates"] for c in cand["members"]})
    ck = {p.stem: p for p in (ROOT / "checkpoints").glob("*.pt")}
    for name in cands:
        yv, pv, rv, _ = member_preds(ck[name], a.dataset, "val", out)
        yt, pt, rt, _ = member_preds(ck[name], a.dataset, "test", out)
        pred = pt.argmax(1)
        ctx.append(dict(model=name, val_score=sel_score(yv.astype(int), pv.argmax(1)),
                        test_accuracy=float((pred == yt).mean()),
                        test_qwk=float(cohen_kappa_score(yt, pred, weights="quadratic")),
                        test_macro_f1=float(f1_score(yt, pred, average="macro", labels=range(5), zero_division=0)),
                        test_referable_auc=float(roc_auc_score(yt >= 2, pt[:, 2:].sum(1)))))
    e = report["evaluations"][f"{a.dataset}_test"]
    ctx.append(dict(model="BEST ENSEMBLE (frozen on validation)", val_score=sel["cv_val_score"],
                    test_accuracy=e["accuracy"], test_qwk=e["qwk"], test_macro_f1=e["macro_f1"],
                    test_referable_auc=e["referable"]["auc"]))
    df = pd.DataFrame(ctx).sort_values("val_score", ascending=False)
    df.to_csv(out / "test_context_single_models.csv", index=False)
    report["test_context_single_models"] = df.to_dict("records")
    (out / "final_metrics.json").write_text(json.dumps(report, indent=2))

    for key, m in report["evaluations"].items():
        c, r = m["ci95"], m["referable"]
        print(f"\n== {key}  (n={m['n']})")
        print(f"  five-class: acc {m['accuracy']:.3f} [{c['accuracy'][0]:.3f}-{c['accuracy'][1]:.3f}]  "
              f"QWK {m['qwk']:.3f} [{c['qwk'][0]:.3f}-{c['qwk'][1]:.3f}]  "
              f"macro-F1 {m['macro_f1']:.3f} [{c['macro_f1'][0]:.3f}-{c['macro_f1'][1]:.3f}]  AUC {m['roc_auc']:.3f}")
        print(f"  referable : acc {r['accuracy']:.3f} [{c['referable_accuracy'][0]:.3f}-{c['referable_accuracy'][1]:.3f}]  "
              f"sens {r['sensitivity']:.3f} [{c['referable_sensitivity'][0]:.3f}-{c['referable_sensitivity'][1]:.3f}]  "
              f"spec {r['specificity']:.3f} [{c['referable_specificity'][0]:.3f}-{c['referable_specificity'][1]:.3f}]  "
              f"AUC {r['auc']:.3f} [{c['referable_auc'][0]:.3f}-{c['referable_auc'][1]:.3f}]  prevalence {r['prevalence']:.2f}")
        print("  per-class recall: " + "  ".join(f"{k} {v['recall']:.2f} (n={v['support']})" for k, v in m["per_class"].items()))
        print("  confusion (rows true, cols pred):")
        print(pd.DataFrame(m["confusion_matrix"], index=NAMES, columns=NAMES).to_string())
    print("\ncontext, single models on APTOS test (argmax; NOT used for selection):")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
