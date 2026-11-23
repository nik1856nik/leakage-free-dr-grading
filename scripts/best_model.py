"""Build and select the final "best model": an ensemble of trained checkpoints on one dataset.

Every member gives one continuous grade score per image (regression output, or the expected grade
sum_k k*p_k of a classifier), averaged over 4 flip TTA views. A subset of members is averaged and
cut into grades 0-4 by four thresholds.

Selection is validation-only and guarded against optimism:
  * thresholds are fitted on validation,
  * the member subset is chosen by 5-fold *cross-fitted* validation score (thresholds fitted on
    4/5 of val, scored on the held-out 1/5), so threshold fitting cannot inflate the choice,
  * the test split is read only with --final, once, after the choice is frozen in selection.json.

    uv run python scripts/best_model.py --dataset aptos --members 'checkpoints/ft__*__aptos*.pt' ...
    uv run python scripts/best_model.py --dataset aptos --final --external idrid messidor2
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.fundus import CLASS_NAMES, FundusDataset, load_split  # noqa: E402
from src.datasets.sources import ROOT  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.zoo import build  # noqa: E402
from src.training.engine import predict  # noqa: E402

NAMES = CLASS_NAMES["5class"]


def member_preds(ckpt: Path, dataset: str, split: str, out_dir: Path, tta: int = 4):
    """(y, prob, raw, uid) for one checkpoint on one split; cached."""
    f = out_dir / "preds" / f"{ckpt.stem}__{dataset}_{split}_tta{tta}.npz"
    if not f.exists():
        ck = torch.load(ckpt, map_location="cpu")
        cfg = ck["config"]
        model = build(cfg["model"], 1 if cfg.get("head") == "reg" else 5)
        model.load_state_dict(ck["model"])
        model.cuda()
        df = load_split(dataset, "5class", None if split == "all" else split)
        dl = DataLoader(FundusDataset(df, cfg["size"], False, cfg["enhancement"]),
                        batch_size=32, num_workers=8, pin_memory=True)
        y, p, r = predict(model, dl, torch.device("cuda"), tta=tta, return_raw=True)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, y=y, prob=p, raw=r, uid=df.uid.values)
        del model; torch.cuda.empty_cache()
    d = np.load(f, allow_pickle=True)
    return d["y"], d["prob"], d["raw"], d["uid"]


def cut(s, thr):
    return np.digitize(s, np.sort(thr))


_W = (np.subtract.outer(np.arange(5), np.arange(5)) ** 2) / 16.0


def sel_score(y, pred):
    """0.5*(QWK + macro-F1) from one 5x5 confusion matrix (fast; equals the sklearn values)."""
    cm = np.bincount(y * 5 + pred, minlength=25).reshape(5, 5).astype(float)
    n = cm.sum()
    exp = np.outer(cm.sum(1), cm.sum(0)) / n
    qwk = 1 - (_W * cm).sum() / max((_W * exp).sum(), 1e-12)
    tp = np.diag(cm); prec = tp / np.maximum(cm.sum(0), 1e-12); rec = tp / np.maximum(cm.sum(1), 1e-12)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return 0.5 * (qwk + f1.mean())


def fit_thresholds(y, s, sweeps: int = 3):
    """Coordinate search over the 4 cut points maximising the selection score."""
    thr = np.array([0.5, 1.5, 2.5, 3.5])
    for _ in range(sweeps):
        for k in range(4):
            lo = thr[k - 1] + 0.05 if k else -0.5
            hi = thr[k + 1] - 0.05 if k < 3 else 4.5
            grid = np.arange(lo, hi + 1e-9, 0.02)
            best = max(grid, key=lambda g: sel_score(y, cut(s, np.r_[thr[:k], g, thr[k + 1:]])))
            thr[k] = best
    return thr


def cv_score(y, s, seeds=(0, 1, 2)):
    """Mean cross-fitted score over repeated 5-fold splits; oof predictions of the first split."""
    scores, first = [], None
    for seed in seeds:
        oof = np.zeros_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(s, y):
            oof[te] = cut(s[te], fit_thresholds(y[tr], s[tr]))
        scores.append(sel_score(y, oof))
        first = oof if first is None else first
    return float(np.mean(scores)), first


def decide(prob, s, thr, mode):
    """Grades from an ensemble: 'thr' = cut the mean grade score; 'argmax' = most likely class of mean probs."""
    return prob.argmax(1) if mode == "argmax" else cut(s, thr)


def full_metrics(y, prob, s, thr, mode="thr"):
    return compute_metrics(y, prob, NAMES, pred=decide(prob, s, thr, mode))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aptos")
    ap.add_argument("--members", nargs="+", help="checkpoint paths / globs (selection step)")
    ap.add_argument("--max_size", type=int, default=4, help="largest ensemble considered")
    ap.add_argument("--final", action="store_true", help="evaluate the frozen selection on the test split")
    ap.add_argument("--external", nargs="*", default=[], help="datasets evaluated on all images")
    a = ap.parse_args()
    out = ROOT / "experiments" / f"best_{a.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    sel_file = out / "selection.json"

    if not a.final:
        ckpts = sorted({Path(p) for g in a.members for p in glob.glob(g)})
        assert ckpts, "no checkpoints matched"
        V = {c.stem: member_preds(c, a.dataset, "val", out) for c in ckpts}
        y = next(iter(V.values()))[0]
        for k, v in V.items():
            assert (v[0] == y).all(), f"{k}: val order mismatch"
        rows = []
        for n in range(1, min(a.max_size, len(V)) + 1):
            for combo in itertools.combinations(sorted(V), n):
                s = np.mean([V[c][2] for c in combo], 0)
                prob = np.mean([V[c][1] for c in combo], 0)
                cv, _ = cv_score(y, s)                       # thresholds: cross-fitted (fitted -> CV)
                rows.append((cv, combo, "thr"))
                rows.append((sel_score(y, prob.argmax(1)), combo, "argmax"))   # nothing fitted -> direct
        rows.sort(key=lambda r: -r[0])
        print("top candidates by validation score 0.5*(QWK+macroF1)  [thr = cross-fitted thresholds, argmax = mean probs]:")
        for cv, combo, mode in rows[:15]:
            print(f"  {cv:.4f}  {mode:6s} {' + '.join(combo)}")
        print("single models:")
        for cv, combo, mode in rows:
            if len(combo) == 1:
                print(f"  {cv:.4f}  {mode:6s} {combo[0]}")
        cv, combo, mode = rows[0]
        s = np.mean([V[c][2] for c in combo], 0)
        prob = np.mean([V[c][1] for c in combo], 0)
        thr = fit_thresholds(y, s)
        oof = cv_score(y, s)[1] if mode == "thr" else prob.argmax(1)
        sel = dict(dataset=a.dataset, members=[str(c) for c in ckpts if c.stem in combo], mode=mode,
                   thresholds=thr.tolist(), cv_val_score=cv, val_cv=compute_metrics(y, prob, NAMES, pred=oof),
                   val_fitted=full_metrics(y, prob, s, thr, mode),
                   candidates=[dict(score=r[0], members=list(r[1]), mode=r[2]) for r in rows])
        sel_file.write_text(json.dumps(sel, indent=2))
        v = sel["val_cv"]
        print(f"\nSELECTED ({len(combo)} members, decision: {mode}): {' + '.join(combo)}"
              + (f"\n  thresholds {np.round(thr, 2).tolist()}" if mode == "thr" else ""))
        print(f"  val (cross-fitted): acc {v['accuracy']:.4f}  QWK {v['qwk']:.4f}  macro-F1 {v['macro_f1']:.4f}  "
              f"AUC {v['roc_auc']:.4f}  referable sens {v['referable_sensitivity']:.3f} spec {v['referable_specificity']:.3f}")
        return

    sel = json.loads(sel_file.read_text())
    ckpts = [Path(p) for p in sel["members"]]
    thr = np.array(sel["thresholds"])
    res = {}
    for ds, split in [(a.dataset, "test")] + [(e, "all") for e in a.external]:
        P = [member_preds(c, ds, split, out) for c in ckpts]
        y = P[0][0]
        s = np.mean([p[2] for p in P], 0)
        prob = np.mean([p[1] for p in P], 0)
        res[f"{ds}_{split}"] = m = full_metrics(y, prob, s, thr, sel.get("mode", "thr"))
        print(f"{ds:10s} {split:4s} n={m['n']:5d}  acc {m['accuracy']:.4f}  QWK {m['qwk']:.4f}  "
              f"macro-F1 {m['macro_f1']:.4f}  AUC {m['roc_auc']:.4f}  | referable: sens "
              f"{m['referable_sensitivity']:.3f} spec {m['referable_specificity']:.3f} AUC {m['referable_auc']:.4f}")
    (out / "final_results.json").write_text(json.dumps(dict(selection=sel["members"], mode=sel.get("mode", "thr"), thresholds=sel["thresholds"],
                                                            results=res), indent=2))


if __name__ == "__main__":
    main()
