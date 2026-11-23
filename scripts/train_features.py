"""Deep-feature + classical-classifier models.

P6  hybrid_svm : frozen GoogLeNet + ResNet-18 1000-d FC outputs -> 2000-d -> SVM
P11 resnet50_rf: ResNet-50 fine-tuned by scripts/train.py (model resnet50_rf) -> GAP 2048-d -> Random Forest

Classifier hyper-parameters are selected on the validation split only.

    uv run python scripts/train_features.py --model hybrid_svm --dataset aptos --task 5class
    uv run python scripts/train_features.py --model hybrid_svm --dataset aptos --task 3class_p6 --undersample
    uv run python scripts/train_features.py --model resnet50_rf --dataset aptos --cnn_ckpt checkpoints/<run>.pt
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.fundus import CLASS_NAMES, TASKS, FundusDataset, load_split  # noqa: E402
from src.datasets.sources import ROOT  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.zoo import REGISTRY, build, n_params  # noqa: E402


@torch.no_grad()
def extract(model, df, size, enh, bs=64):
    ds = FundusDataset(df, size, train=False, enhancement=enh)
    dl = DataLoader(ds, batch_size=bs, num_workers=8, pin_memory=True)
    feats = []
    model.eval().cuda()
    t0 = time.time()
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            feats.append(model.features(x.cuda()).float().cpu().numpy())
    return np.concatenate(feats), ds.targets, (time.time() - t0) / len(ds)


def undersample(X, y, seed=42):
    rng = np.random.default_rng(seed)
    m = np.bincount(y).min()
    idx = np.concatenate([rng.choice(np.where(y == k)[0], m, replace=False) for k in np.unique(y)])
    return X[idx], y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["hybrid_svm", "resnet50_rf"])
    ap.add_argument("--dataset", default="aptos")
    ap.add_argument("--task", default="5class")
    ap.add_argument("--cnn_ckpt", default=None)
    ap.add_argument("--undersample", action="store_true", help="P6 original: balance by undersampling")
    ap.add_argument("--final", action="store_true", help="evaluate on test split")
    a = ap.parse_args()

    n_cls = TASKS[a.task][0]
    names = CLASS_NAMES[a.task]
    _, size, enh, paper = REGISTRY[a.model]
    tag = f"{a.model}__{a.dataset}__{a.task}{'__us' if a.undersample else ''}"
    run_dir = ROOT / "experiments" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    if a.model == "hybrid_svm":
        net = build("hybrid_svm", n_cls)
    else:
        assert a.cnn_ckpt, "--cnn_ckpt (fine-tuned resnet50_rf checkpoint) required"
        ck = torch.load(a.cnn_ckpt, map_location="cpu")
        size, enh = ck["config"]["size"], ck["config"]["enhancement"]
        net = build("resnet50_rf", TASKS[ck["config"]["task"]][0])
        net.load_state_dict(ck["model"])

    splits = ("train", "val", "test") if a.final else ("train", "val")
    data = {}
    for sp in splits:
        df = load_split(a.dataset, a.task, sp)
        X, y, sec = extract(net, df, size, enh)
        data[sp] = (X, y, df.uid.values)
        print(f"{sp}: X {X.shape}  ({1000*sec:.1f} ms/img feature extraction)")
    Xtr, ytr, _ = data["train"]
    if a.undersample:
        Xtr, ytr = undersample(Xtr, ytr)
        print("undersampled train:", np.bincount(ytr))

    # ---- classifier search on validation
    if a.model == "hybrid_svm":
        grid = [dict(kernel=k, C=c, class_weight=w) for k, c, w in
                itertools.product(["linear", "rbf"], [0.01, 0.1, 1, 10], [None, "balanced"])]
        make = lambda p: make_pipeline(StandardScaler(), SVC(probability=True, random_state=42, **p))
    else:
        # P11: criterion entropy, min_samples_leaf 1, min_samples_split 2, random_state 1
        grid = [dict(n_estimators=n, class_weight=w) for n, w in
                itertools.product([100, 300, 500], [None, "balanced_subsample"])]
        make = lambda p: RandomForestClassifier(criterion="entropy", min_samples_leaf=1,
                                                min_samples_split=2, random_state=1, n_jobs=-1, **p)
    Xv, yv, uv = data["val"]
    results = []
    for p in grid:
        clf = make(p).fit(Xtr, ytr)
        m = compute_metrics(yv, clf.predict_proba(Xv), names)
        score = m["roc_auc"] if n_cls == 2 else 0.5 * (m["qwk"] + m["macro_f1"])
        results.append((score, p, m))
        print(f"  {p}  val acc {m['accuracy']:.4f} mF1 {m['macro_f1']:.4f} qwk {m['qwk']:.4f} -> {score:.4f}")
    best_score, best_p, mv = max(results, key=lambda r: r[0])
    print("best:", best_p)
    t0 = time.time()
    clf = make(best_p).fit(Xtr, ytr)
    fit_time = time.time() - t0
    summary = dict(run_name=tag, model=a.model, paper=paper, dataset=a.dataset, task=a.task,
                   undersample=a.undersample, classifier_params=best_p, best_val_score=best_score,
                   train_time_s=fit_time, params=n_params(net), val=compute_metrics(yv, clf.predict_proba(Xv), names))
    np.savez(run_dir / "val_predictions.npz", y=yv, prob=clf.predict_proba(Xv), uid=uv)
    if a.final:
        Xt, yt, ut = data["test"]
        t0 = time.time(); pt = clf.predict_proba(Xt)
        summary["test"] = compute_metrics(yt, pt, names)
        summary["classifier_ms_per_image"] = 1000 * (time.time() - t0) / len(yt)
        np.savez(run_dir / "test_predictions.npz", y=yt, prob=pt, uid=ut)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("run_name", "classifier_params")}, default=str))
    for sp in ("val", "test"):
        if sp in summary:
            print(sp, {k: round(summary[sp][k], 4) for k in ("accuracy", "macro_f1", "qwk", "roc_auc")})


if __name__ == "__main__":
    main()
