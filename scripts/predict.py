"""Grade new fundus photographs with the packaged best model (best_model/).

    uv run python scripts/predict.py path/to/image.jpg [more images or folders ...] [--csv out.csv]

Applies exactly the training-time pipeline: retina crop + square pad -> 512 px -> JPEG q95 round-trip
(the cache format) -> each member's enhancement -> resize/normalise -> 4 flip views. Members are
averaged and the grade is decided as frozen on validation (model_card.json).
Research use only; not a medical device.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.fundus import build_transforms  # noqa: E402
from src.models.zoo import build  # noqa: E402
from src.preprocessing.retina import ENHANCEMENTS, base_preprocess  # noqa: E402
from src.training.engine import _out_to_prob  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def cached_form(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    rgb = base_preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--model_dir", default=str(ROOT / "best_model"))
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    md = Path(a.model_dir)
    card = json.loads((md / "model_card.json").read_text())
    names, thr = card["classes"], np.sort(card["thresholds"])
    files = []
    for s in a.inputs:
        p = Path(s)
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {f: cached_form(f, card["cache_size"]) for f in files}

    probs, scores = np.zeros((len(files), 5)), np.zeros(len(files))
    flips = [lambda x: x, lambda x: x.flip(3), lambda x: x.flip(2), lambda x: x.flip(2).flip(3)][:card["tta_flips"]]
    for m in card["members"]:
        ck = torch.load(md / m["checkpoint"], map_location="cpu")
        net = build(m["model"], 1 if m["head"] == "reg" else 5)
        net.load_state_dict(ck["model"]); net.to(dev).eval()
        enh, tf = ENHANCEMENTS[m["enhancement"]], build_transforms(m["size"], False)
        with torch.no_grad():
            for i, f in enumerate(files):
                x = torch.from_numpy(tf(image=enh(cache[f]))["image"].transpose(2, 0, 1))[None].to(dev)
                outs = [net(fl(x)).float() for fl in flips]
                p = torch.stack([_out_to_prob(o) for o in outs]).mean(0)[0].cpu().numpy()
                r = (torch.stack(outs).mean().item() if outs[0].shape[1] == 1
                     else float((p * np.arange(5)).sum()))
                probs[i] += p / len(card["members"]); scores[i] += r / len(card["members"])
        del net
    grade = probs.argmax(1) if card["decision"] == "argmax" else np.digitize(scores, thr)
    df = pd.DataFrame(dict(image=[str(f) for f in files], grade=grade, grade_name=[names[g] for g in grade],
                           grade_score=scores.round(3), p_referable=probs[:, 2:].sum(1).round(4),
                           referable=np.where(grade >= 2, "REFER", "no")))
    for k in range(5):
        df[f"p_{names[k].replace(' ', '_')}"] = probs[:, k].round(4)
    print(df[["image", "grade", "grade_name", "grade_score", "p_referable", "referable"]].to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"saved {a.csv}")


if __name__ == "__main__":
    main()
