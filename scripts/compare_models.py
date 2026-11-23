"""Collect experiment summaries into reports/model_comparison.csv and draw comparison plots.

    uv run python scripts/compare_models.py [--split test]
Only runs that were evaluated on the requested split are included.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.fundus import CLASS_NAMES  # noqa: E402
from src.datasets.sources import ROOT  # noqa: E402
from src.evaluation import plots  # noqa: E402

KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "roc_auc", "qwk",
        "sensitivity", "specificity", "anyDR_sensitivity", "anyDR_specificity", "referable_sensitivity",
        "referable_specificity", "majority_baseline_acc", "n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    rows = []
    for f in sorted((ROOT / "experiments").glob("*/summary.json")):
        s = json.loads(f.read_text())
        m = s.get(a.split)
        if not m:
            continue
        task = s["task"].split(" ")[0]
        r = dict(run=f.parent.name, model=s["model"], paper=s.get("paper"), dataset=s["dataset"], task=task,
                 params=s.get("params"), train_time_s=s.get("train_time_s"),
                 inference_ms=s.get("inference_ms_per_image", s.get("classifier_ms_per_image")))
        r.update({k: m.get(k) for k in KEYS})
        # external (cross-dataset) results, when evaluated: APTOS -> IDRiD, EyePACS -> Messidor-2
        ext = {"idrid": s.get("external_idrid_all")}
        ej = f.parent / "eval_external_messidor2_all.json"
        ext["messidor2"] = json.loads(ej.read_text()) if ej.exists() else None
        for ds_, em in ext.items():
            if em:
                for k in ("accuracy", "macro_f1", "qwk", "roc_auc", "referable_sensitivity",
                          "referable_specificity", "referable_auc"):
                    r[f"ext_{ds_}_{k}"] = em.get(k)
        for c, v in m.get("per_class", {}).items():
            for mm in ("precision", "recall", "f1"):
                r[f"{c}_{mm}"] = v[mm]
        rows.append(r)
        # per-run figures
        pdir = ROOT / "results" / "plots" / f.parent.name
        names = CLASS_NAMES.get(task, list(m["per_class"]))
        plots.confusion(m["confusion_matrix"], names, pdir / f"confusion_{a.split}.png",
                        f"{s['model']} · {s['dataset']} · {a.split}")
        pf = f.parent / f"{a.split}_predictions.npz"
        if pf.exists():
            d = np.load(pf, allow_pickle=True)
            plots.roc_curves(d["y"], d["prob"], names, pdir / f"roc_{a.split}.png",
                             f"{s['model']} · {s['dataset']} · ROC ({a.split})")
        hf = f.parent / "history.csv"
        if hf.exists():
            plots.training_curves(pd.read_csv(hf), pdir / "training_curves.png", f"{s['model']} · {s['dataset']}")
    if not rows:
        print("no runs with", a.split, "metrics yet")
        return
    df = pd.DataFrame(rows)
    out = ROOT / "reports" / ("model_comparison.csv" if a.split == "test" else f"model_comparison_{a.split}.csv")
    df.to_csv(out, index=False)
    print(df[["run", "accuracy", "macro_f1", "qwk", "roc_auc"]].round(4).to_string(index=False))
    # cross-model figures per (dataset, task)
    # cross-model figures: one bar per model = the paper-default baseline runs only
    # (tuning / fine-tune variants share model names and are compared in their own views)
    base = df[~df.run.str.startswith(("tune__", "ft__", "smoke"))]
    for (ds, task), g in base.groupby(["dataset", "task"]):
        nm = np.where((g.model == "resnet50_rf") & (g.run.str.count("__") > 2), "resnet50 CNN stage",
                      np.where(g.model == "resnet50_rf", "resnet50 + RF", g.model))
        g = g.assign(label=pd.Series(nm, index=g.index) + " (" + g.paper.fillna("") + ")")
        tag = f"{ds}_{task}_{a.split}"
        for metric in ("accuracy", "macro_f1", "qwk", "roc_auc"):
            plots.metric_bars(g, metric, ROOT / "results/plots/comparison" / f"{tag}_{metric}.png",
                              f"{metric} · {ds} · {task} ({a.split})")
        if task == "5class":
            plots.per_class_grouped(g, CLASS_NAMES["5class"], ROOT / "results/plots/comparison" /
                                    f"{tag}_per_class_recall.png", f"Per-class recall · {ds} ({a.split})")
    print("wrote", out)


if __name__ == "__main__":
    main()
