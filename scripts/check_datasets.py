"""Report which datasets are present and summarise their prepared metadata.

    uv run python scripts/check_datasets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.sources import RAW, ROOT, available  # noqa: E402


def main():
    av = available()
    print("raw availability:")
    for k, v in av.items():
        print(f"  {k:10s} {'OK' if v else 'MISSING'}")
    eye = sorted((RAW / "eyepacs").glob("train.zip.0*"))
    if eye:
        print(f"  eyepacs parts present: {[p.name for p in eye]}")
    for name in av:
        meta = ROOT / "data" / "metadata" / f"{name}.csv"
        sp = ROOT / "data" / "splits" / f"{name}_5class.csv"
        if not meta.exists():
            continue
        m = pd.read_csv(meta)
        print(f"\n== {name}: {len(m)} images, unreadable {int((~m.ok.astype(bool)).sum())}")
        print("  label:", m.label.value_counts(dropna=False).sort_index().to_dict())
        print("  original sizes (top 5):",
              m.groupby(["orig_w", "orig_h"]).size().sort_values(ascending=False).head(5).to_dict())
        for c in ("brightness", "contrast", "sharpness", "retina_frac"):
            print(f"  {c:12s} median {m[c].median():.2f}  p5 {m[c].quantile(.05):.2f}  p95 {m[c].quantile(.95):.2f}")
        dup = ROOT / "data" / "metadata" / f"{name}_duplicates.csv"
        if dup.exists():
            d = pd.read_csv(dup)
            print(f"  near-dup clusters {d.dup_cluster.nunique()} ({len(d)} imgs), "
                  f"label-conflict clusters {d[d.label_conflict].dup_cluster.nunique()}")
        if sp.exists():
            s = pd.read_csv(sp)
            print("  splits:\n" + pd.crosstab(s.split, s.label, margins=True).to_string())


if __name__ == "__main__":
    main()
