"""Evaluate a trained end-to-end checkpoint on a split of any dataset.

Within-dataset final test (optionally with flip TTA):
    uv run python scripts/evaluate.py --ckpt checkpoints/<run>.pt --dataset aptos --split test --tta 4
Cross-dataset (external) validation, e.g. EyePACS-trained -> all gradable Messidor-2 images:
    uv run python scripts/evaluate.py --ckpt checkpoints/<eyepacs run>.pt --dataset messidor2 --split all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.fundus import CLASS_NAMES, TASKS, FundusDataset, load_split  # noqa: E402
from src.datasets.sources import ROOT  # noqa: E402
from src.models.zoo import build  # noqa: E402
from src.training.engine import evaluate_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test", help="train|val|test|all")
    ap.add_argument("--task", default=None, help="defaults to the checkpoint's task")
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = ck["config"]
    task = a.task or cfg["task"]
    assert TASKS[task][0] == TASKS[cfg["task"]][0], "task class count must match the checkpoint"
    model = build(cfg["model"], TASKS[task][0])
    model.load_state_dict(ck["model"])
    model.cuda()

    df = load_split(a.dataset, task, None if a.split == "all" else a.split)
    dl = DataLoader(FundusDataset(df, cfg["size"], False, cfg["enhancement"]),
                    batch_size=a.batch_size, num_workers=8, pin_memory=True)
    same = a.dataset == cfg["dataset"]
    tag = f"{a.split}" if same else f"external_{a.dataset}_{a.split}"
    tag += f"_tta{a.tta}" if a.tta else ""
    run_dir = ROOT / "experiments" / cfg["run_name"]
    m = evaluate_split(model, dl, torch.device("cuda"), CLASS_NAMES[task], run_dir, tag, tta=a.tta)
    m["inference_ms_per_image"] = m.pop("_ms_per_image")
    m.update(train_dataset=cfg["dataset"], eval_dataset=a.dataset, eval_split=a.split, tta=a.tta,
             model=cfg["model"], run_name=cfg["run_name"], task=task)
    (run_dir / f"eval_{tag}.json").write_text(json.dumps(m, indent=2))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                      if k not in ("confusion_matrix", "per_class")}, indent=1))
    print("confusion matrix:\n", pd.DataFrame(m["confusion_matrix"], index=CLASS_NAMES[task],
                                              columns=CLASS_NAMES[task]))


if __name__ == "__main__":
    main()
