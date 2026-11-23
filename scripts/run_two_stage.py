"""P10 two-stage framework (leak-free): stage 1 binary DR detection, stage 2 severity
(4 classes, DR-positive images only) by retraining the head of the frozen stage-1 network.
Stages are combined into a five-class probability:
    P(0) = p1(noDR) ;  P(k) = p1(DR) * p2(k-1),  k = 1..4

    uv run python scripts/run_two_stage.py --dataset aptos [--final]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.fundus import CLASS_NAMES, FundusDataset, load_split  # noqa: E402
from src.datasets.sources import ROOT  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.zoo import build  # noqa: E402
from src.training.engine import TrainConfig, predict, train  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def probs(ckpt_path, n_cls, dataset, split, size, enh):
    m = build("light2stage", n_cls)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    m.cuda()
    df = load_split(dataset, "5class", split)  # ALL images of the split, five-class targets
    dl = DataLoader(FundusDataset(df, size, False, enh), batch_size=128, num_workers=8)
    y, p = predict(m, dl, torch.device("cuda"))
    return y, p, df.uid.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aptos")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--epochs1", type=int, default=100)
    ap.add_argument("--epochs2", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    base = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["light2stage"]["train"]

    c1 = TrainConfig(**{**base, "dataset": a.dataset, "task": "binary", "epochs": a.epochs1, "seed": a.seed})
    s1 = train(c1, evaluate_test=a.final)
    ck1 = ROOT / "checkpoints" / f"{c1.run_name}.pt"
    # stage 2: body frozen (P10: "weights are frozen and the last three layers retrained")
    c2 = TrainConfig(**{**base, "dataset": a.dataset, "task": "severity4", "epochs": a.epochs2,
                        "seed": a.seed, "init_from": str(ck1), "freeze_body": True,
                        "lr": 1e-2})
    s2 = train(c2, evaluate_test=a.final)
    ck2 = ROOT / "checkpoints" / f"{c2.run_name}.pt"

    out = dict(model="light2stage", paper="P10", dataset=a.dataset, task="5class (combined)",
               stage1=dict(val=s1["val"], test=s1.get("test")),
               stage2=dict(val=s2["val"], test=s2.get("test")),
               params=s1["params"], train_time_s=s1["train_time_s"] + s2["train_time_s"])
    run_dir = ROOT / "experiments" / f"light2stage__{a.dataset}__5class_combined_s{a.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for sp in (["val", "test"] if a.final else ["val"]):
        y, p1, uid = probs(ck1, 2, a.dataset, sp, c1.size, c1.enhancement)
        _, p2, _ = probs(ck2, 4, a.dataset, sp, c1.size, c1.enhancement)
        p5 = np.concatenate([p1[:, :1], p1[:, 1:2] * p2], 1)
        out[sp] = compute_metrics(y, p5, CLASS_NAMES["5class"])
        np.savez(run_dir / f"{sp}_predictions.npz", y=y, prob=p5, uid=uid)
        print(sp, {k: round(out[sp][k], 4) for k in ("accuracy", "macro_f1", "qwk", "roc_auc")})
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
