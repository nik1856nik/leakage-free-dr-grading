"""One-time TEST evaluation of the 10 paper-default APTOS baselines (configs frozen on validation).

End-to-end networks: the saved best-validation checkpoint is evaluated as-is (no TTA, the same
protocol as their validation numbers). P10 light two-stage: stage-1 x stage-2 combination from its
saved checkpoints. Feature-classifier models (P11 RF, P6 SVM) are run via train_features.py --final.
Results are merged into each run's summary.json under "test", plus an external IDRiD evaluation
(all 508 gradable images) for the APTOS -> IDRiD generalisation test.

    uv run python scripts/final_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_two_stage import probs  # noqa: E402
from src.datasets.fundus import CLASS_NAMES, FundusDataset, load_split  # noqa: E402
from src.datasets.sources import ROOT  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.zoo import build  # noqa: E402
from src.training.engine import evaluate_split  # noqa: E402

RUNS = ["effb3_se__aptos__5class__ben_graham_384_s42", "inception_v3__aptos__5class__raw_384_s42",
        "densenet169__aptos__5class__raw_384_s42", "resnet50_rf__aptos__5class__raw_384_s42",
        "mscnn_lstm__aptos__5class__clahe_224_s42", "vit_b16__aptos__5class__raw_224_s42",
        "p8_vgg16__aptos__5class__agc_p8_384_s42", "mil_resnet18__aptos__5class__graham_rim_512_s42"]
NAMES = CLASS_NAMES["5class"]


def main():
    dev = torch.device("cuda")
    for run in RUNS:
        ck = torch.load(ROOT / "checkpoints" / f"{run}.pt", map_location="cpu")
        cfg = ck["config"]
        model = build(cfg["model"], 5)
        model.load_state_dict(ck["model"]); model.to(dev)
        run_dir = ROOT / "experiments" / run
        summ = json.loads((run_dir / "summary.json").read_text())
        for ds, split, key in (("aptos", "test", "test"), ("idrid", None, "external_idrid_all")):
            df = load_split(ds, "5class", split)
            dl = DataLoader(FundusDataset(df, cfg["size"], False, cfg["enhancement"]),
                            batch_size=16, num_workers=8, pin_memory=True)
            m = evaluate_split(model, dl, dev, NAMES, run_dir, key)
            ms = m.pop("_ms_per_image")
            summ[key] = m
            if key == "test":
                summ["inference_ms_per_image"] = ms
            print(f"{run:52s} {key:20s} acc {m['accuracy']:.4f}  mF1 {m['macro_f1']:.4f}  QWK {m['qwk']:.4f}  "
                  f"AUC {m['roc_auc']:.4f}  ({ms:.1f} ms/img)", flush=True)
        (run_dir / "summary.json").write_text(json.dumps(summ, indent=2))
        del model; torch.cuda.empty_cache()

    # P10 light two-stage CNN: P(0) = p1(0); P(k) = p1(1) * p2(k-1)
    ck1 = ROOT / "checkpoints" / "light2stage__aptos__binary__raw_227_s42.pt"
    ck2 = ROOT / "checkpoints" / "light2stage__aptos__severity4__raw_227_s42.pt"
    run_dir = ROOT / "experiments" / "light2stage__aptos__5class_combined_s42"
    summ = json.loads((run_dir / "summary.json").read_text())
    for ds, sp, key in (("aptos", "test", "test"), ("idrid", None, "external_idrid_all")):
        y, p1, uid = probs(ck1, 2, ds, sp, 227, "raw")
        _, p2, _ = probs(ck2, 4, ds, sp, 227, "raw")
        p5 = np.concatenate([p1[:, :1], p1[:, 1:2] * p2], 1)
        summ[key] = m = compute_metrics(y, p5, NAMES)
        np.savez(run_dir / f"{key}_predictions.npz", y=y, prob=p5, uid=uid)
        print(f"{'light2stage (P10, combined)':52s} {key:20s} acc {m['accuracy']:.4f}  mF1 {m['macro_f1']:.4f}  "
              f"QWK {m['qwk']:.4f}  AUC {m['roc_auc']:.4f}")
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
