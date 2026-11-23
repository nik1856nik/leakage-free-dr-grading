"""Train an end-to-end model.

Examples:
    uv run python scripts/train.py --model effb3_se --dataset aptos --epochs 30
    uv run python scripts/train.py --model mil_resnet18 --dataset eyepacs --task rdr --lr 3e-4 --optimizer adam
    # final run only (after tuning on validation): also evaluate the untouched test split
    uv run python scripts/train.py --model effb3_se --dataset aptos --final

Any TrainConfig field can be set with --<field> <value>. Defaults per model come from
configs/models.yaml (key "train"), overridden by command-line flags.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.engine import TrainConfig, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="evaluate on the held-out test split")
    for f in fields(TrainConfig):
        if f.name == "extra":
            continue
        t = f.type if isinstance(f.type, type) else str
        if f.type in ("bool", bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"))
        elif "int" in str(f.type):
            ap.add_argument(f"--{f.name}", type=int)
        elif "float" in str(f.type):
            ap.add_argument(f"--{f.name}", type=float)
        else:
            ap.add_argument(f"--{f.name}", type=str)
    return ap.parse_args()


def main():
    a = parse()
    defaults = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    kw = dict(defaults.get(a.model, {}).get("train", {}))
    kw.update({k: v for k, v in vars(a).items() if v is not None and k != "final"})
    cfg = TrainConfig(**kw)
    s = train(cfg, evaluate_test=a.final)
    short = {k: s[k] for k in ("run_name", "best_epoch", "train_time_s", "params")}
    short["val"] = {k: round(s["val"][k], 4) for k in ("accuracy", "macro_f1", "qwk", "roc_auc")}
    if "test" in s:
        short["test"] = {k: round(s["test"][k], 4) for k in ("accuracy", "macro_f1", "qwk", "roc_auc")}
    print(json.dumps(short, indent=2))


if __name__ == "__main__":
    main()
