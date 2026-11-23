"""Additional paper figures (photos, architecture diagrams, data and result plots).

    uv run python paper/make_figures_extra.py
Every number is read from project files (data/, experiments/, reports/).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import INK, INK2, GRID, SERIES, NAMES, save, ensemble_preds  # noqa: E402  (also sets rcParams)
from src.preprocessing.retina import ENHANCEMENTS  # noqa: E402

BLUES5 = ["#cde2fb", "#9ec5f4", "#5c9be6", "#256abf", "#0d366b"]   # ordinal grades 0..4


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ photos: one image per grade
def fig_grades():
    s = pd.read_csv(ROOT / "data/splits/aptos_5class.csv")
    tr = s[s.split == "train"]
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 1.75))
    for k, a in enumerate(axes):
        row = tr[tr.label == k].iloc[rng.integers(0, (tr.label == k).sum())]
        a.imshow(load_rgb(row.cache)); a.axis("off")
        a.set_title(f"Grade {k}: {NAMES[k]}", fontsize=7)
    plt.tight_layout(pad=0.3)
    save(fig, "fig_grades")


# ------------------------------------------------------------------ photos: preprocessing variants
def fig_preproc():
    s = pd.read_csv(ROOT / "data/splits/aptos_5class.csv")
    row = s[(s.split == "train") & (s.label == 2)].iloc[5]
    img = load_rgb(row.cache)
    variants = [("raw", "Cropped, 512 px\n(all models)"), ("ben_graham", "Graham subtraction\n(EffNet-B3+SE)"),
                ("graham_rim", "Graham + rim zeroing\n(attention-MIL)"), ("clahe", "CLAHE on L channel\n(MS-CNN+LSTM)"),
                ("agc_p8", "Adaptive gamma + HE\n(AGC+VGG16)")]
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 1.85))
    for a, (k, t) in zip(axes, variants):
        a.imshow(ENHANCEMENTS[k](img.copy())); a.axis("off"); a.set_title(t, fontsize=6.4)
    plt.tight_layout(pad=0.3)
    save(fig, "fig_preproc")


# ------------------------------------------------------------------ class distribution of the four datasets
def fig_class_dist():
    dist = {"APTOS (3,662)": [1805, 370, 999, 193, 295], "EyePACS (35,126)": [25810, 2443, 5292, 873, 708],
            "Messidor-2 (1,740)": [1017, 268, 345, 75, 35], "IDRiD (508)": [167, 23, 165, 92, 61]}
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    names = list(dist)[::-1]
    for i, n in enumerate(names):
        v = np.array(dist[n], float); v = v / v.sum(); left = 0
        for k in range(5):
            ax.barh(i, v[k], left=left, color=BLUES5[k], height=0.62, edgecolor="white", linewidth=0.8,
                    label=NAMES[k] if i == 0 else None)
            if v[k] > 0.07:
                ax.text(left + v[k] / 2, i, f"{100 * v[k]:.0f}%", ha="center", va="center", fontsize=5.8,
                        color="white" if k >= 2 else INK)
            left += v[k]
    ax.set_yticks(range(len(names)), names, fontsize=6.5); ax.set_xlim(0, 1)
    ax.set_xlabel("Share of images"); ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.45, 1.22), handlelength=1, columnspacing=0.8)
    save(fig, "fig_class_dist")


# ------------------------------------------------------------------ duplicate-audit correlation distribution
def _hp(path):
    from prepare_data import _highpass_vec
    return _highpass_vec(path)


def fig_dup_corr():
    import torch
    out = {}
    for ds in ("aptos", "idrid", "messidor2"):
        m = pd.read_csv(ROOT / f"data/metadata/{ds}.csv")
        m = m[m.ok.astype(bool)]
        with ProcessPoolExecutor(12) as ex:
            H = np.stack(list(ex.map(_hp, m.cache.values, chunksize=64)))
        Ht = torch.from_numpy(H).cuda().float() / np.sqrt(H.shape[1])
        C = Ht @ Ht.T
        C.fill_diagonal_(-1)
        out[ds] = C.max(1).values.cpu().numpy()
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    bins = np.linspace(0.5, 1.0, 101)
    for k, (ds, lab) in enumerate((("aptos", "APTOS"), ("messidor2", "Messidor-2"), ("idrid", "IDRiD"))):
        ax.hist(out[ds], bins=bins, histtype="step", lw=1.2, color=SERIES[k], label=f"{lab} (n={len(out[ds])})")
    ax.axvline(0.99, color=SERIES[1] if False else "#b3261e", lw=0.9, ls="--")
    ax.text(0.985, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] else 1, "threshold 0.99", rotation=90,
            ha="right", va="top", fontsize=6, color="#b3261e")
    ax.set_yscale("log"); ax.set_xlabel("Nearest-neighbour high-pass correlation"); ax.set_ylabel("Images (log)")
    ax.legend(fontsize=6, loc="upper left")
    save(fig, "fig_dup_corr")
    for ds, v in out.items():
        print(ds, "images with nn-corr >= 0.99:", int((v >= 0.99).sum()), " max below thr:", float(v[v < 0.99].max()))


# ------------------------------------------------------------------ architecture diagrams
def _box(ax, x, y, w, h, t, fc="#e8f1fc", fs=5.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec="#7a7975", lw=0.6))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)


def _arr(ax, a, b, rad=0.0):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.7, shrinkA=0, shrinkB=0,
                                          connectionstyle=f"arc3,rad={rad}"))


def fig_arch():
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.6), gridspec_kw=dict(hspace=0.25))
    for a in axes:
        a.set_xlim(0, 100); a.set_ylim(0, 20); a.axis("off")
    # (a) two-stage CNN
    a = axes[0]
    a.text(0, 19.3, "(a) Lightweight two-stage CNN (P10, leakage-free protocol)", fontsize=7, weight="bold")
    _box(a, 0, 6, 11, 8, "Fundus\n227 px")
    _box(a, 14, 6, 16, 8, "Conv-BN-ReLU trunk\n32-64-128 ch.\n+ max-pool")
    _box(a, 33, 10.5, 14, 5, "3x3 path")
    _box(a, 33, 3.5, 14, 5, "1x1 -> 5x5 path")
    _box(a, 50, 6, 13, 8, "concat\nconv 192\nGAP", fc="#fdf0e8")
    _box(a, 67, 11, 15, 6, "Stage 1 head\nDR vs no DR  p1", fc="#e6f6ef")
    _box(a, 67, 2, 15, 6, "Stage 2 head\n(frozen body)\nseverity 1-4  p2", fc="#e6f6ef")
    _box(a, 85, 6, 15, 8, "P(0)=p1(0)\nP(k)=p1(1) p2(k-1)", fc="#fff4d6")
    _arr(a, (11, 10), (14, 10)); _arr(a, (30, 10), (33, 13)); _arr(a, (30, 10), (33, 6))
    _arr(a, (47, 13), (50, 11)); _arr(a, (47, 6), (50, 9)); _arr(a, (63, 10), (67, 14)); _arr(a, (63, 10), (67, 5))
    _arr(a, (82, 14), (85, 11)); _arr(a, (82, 5), (85, 9))
    # (b) attention-MIL
    a = axes[1]
    a.text(0, 19.3, "(b) Attention-based multiple-instance learning (P12)", fontsize=7, weight="bold")
    _box(a, 0, 5, 12, 10, "Fundus 512 px\nGraham +\nrim zeroing", fs=5.4)
    _box(a, 15, 5, 17, 10, "64x64 patches\ntrain: K=50 random\ntest: grid,\n0.75 overlap\n(>=50% retina)", fs=5.4)
    _box(a, 35, 5, 14, 10, "shared\nResNet-18\nencoder\nh_k, M=128", fs=5.4)
    _box(a, 52, 5, 17, 10, "attention, L=32\na_k = softmax(\nw^T tanh(V h_k))", fc="#fdf0e8", fs=5.4)
    _box(a, 72, 5, 12, 10, "pooling\nz = sum a_k h_k", fc="#fdf0e8", fs=5.4)
    _box(a, 87, 5, 13, 10, "FC -> grade\n(+ attention\nheat-map)", fc="#e6f6ef", fs=5.4)
    for x1, x2 in ((12, 15), (32, 35), (49, 52), (69, 72), (84, 87)):
        _arr(a, (x1, 10), (x2, 10))
    # (c) final ensemble
    a = axes[2]
    a.text(0, 19.3, "(c) Proposed ensemble with validation-fitted ordinal thresholds", fontsize=7, weight="bold")
    _box(a, 0, 6, 11, 8, "Fundus\n(4 flip views)")
    mem = ["EffNet-B3+SE classifier\n512 px, Graham (EyePACS->APTOS)", "EffNet-B3+SE regressor\n512 px, Graham (EyePACS->APTOS)",
           "Inception-v3 classifier\n384 px, raw (APTOS)", "DenseNet-169 classifier\n384 px, raw, focal (APTOS)"]
    for i, t in enumerate(mem):
        _box(a, 15, 15 - 4.6 * i, 27, 4, t, fs=5.3)
        _arr(a, (11, 10), (15, 17 - 4.6 * i))
        _box(a, 45, 15 - 4.6 * i, 14, 4, "regression output" if i == 1 else "expected grade", fc="#fdf0e8", fs=5.3)
        _arr(a, (42, 17 - 4.6 * i), (45, 17 - 4.6 * i)); _arr(a, (59, 17 - 4.6 * i), (62, 10))
    _box(a, 62, 6, 13, 8, "mean grade\nscore s", fc="#fdf0e8")
    _box(a, 78, 6, 22, 8, "thresholds t=(0.64, 1.51,\n2.74, 3.09)\ngrade 0-4, refer if >= 2", fc="#e6f6ef")
    _arr(a, (75, 10), (78, 10))
    save(fig, "fig_arch")


# ------------------------------------------------------------------ training curves
def fig_training():
    runs = [("resnet50_rf__aptos__5class__raw_384_s42", "ResNet-50"), ("densenet169__aptos__5class__raw_384_s42", "DenseNet-169"),
            ("effb3_se__aptos__5class__ben_graham_384_s42", "EffNet-B3+SE"), ("inception_v3__aptos__5class__raw_384_s42", "Inception-v3")]
    eruns = [("effb3_se__eyepacs__5class__ben_graham_512_s42", "EffNet-B3+SE"), ("densenet169__eyepacs__5class__raw_384_s42", "DenseNet-169"),
             ("resnet50_rf__eyepacs__5class__raw_384_s42", "ResNet-50")]
    col = {"ResNet-50": SERIES[0], "DenseNet-169": SERIES[1], "EffNet-B3+SE": SERIES[2], "Inception-v3": SERIES[3]}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2))
    for ax, rr, title in ((axes[0], runs, "APTOS (ImageNet init.)"), (axes[1], eruns, "EyePACS pretraining")):
        for r, n in rr:
            h = pd.read_csv(ROOT / f"experiments/{r}/history.csv")
            ax.plot(h.epoch, h.score, color=col[n], lw=1.2, label=n)
            b = h.score.idxmax(); ax.plot(h.epoch[b], h.score[b], "o", ms=3.5, color=col[n], mec="white", mew=0.6)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(fontsize=6, loc="lower right")
    axes[0].set_ylabel("Validation S = (QWK+mF1)/2")
    plt.tight_layout(pad=0.4)
    save(fig, "fig_training")


# ------------------------------------------------------------------ ablation
def fig_ablation():
    factors = ["512 px", "Other preproc.", "Focal loss", "Sqrt sampling"]
    d = {"ResNet-50": [-0.004, -0.016, -0.002, -0.008], "DenseNet-169": [0.003, -0.017, 0.005, -0.007],
         "EffNet-B3+SE": [0.013, -0.010, -0.015, -0.008]}
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(4); w = 0.26
    ax.axhspan(-0.005, 0.005, color="#efeeea", zorder=0, label="seed noise (+-0.005)")
    for i, (n, v) in enumerate(d.items()):
        ax.bar(x + (i - 1) * w, v, w * 0.92, color=SERIES[i], label=n, zorder=2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x, factors, fontsize=6.5); ax.set_ylabel("Change in validation S")
    ax.legend(fontsize=5.8, ncol=2, loc="lower left"); ax.grid(axis="x", visible=False)
    save(fig, "fig_ablation")


# ------------------------------------------------------------------ ensemble score distributions + thresholds
def fig_scores():
    sel = json.loads((ROOT / "experiments/best_aptos/selection.json").read_text())
    thr = np.sort(sel["thresholds"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2), sharey=True)
    for ax, split, title in ((axes[0], "val", "Validation (thresholds fitted here)"), (axes[1], "test", "Test (single evaluation)")):
        P = [np.load(ROOT / f"experiments/best_aptos/preds/{Path(m).stem}__aptos_{split}_tta4.npz", allow_pickle=True)
             for m in sel["members"]]
        y = P[0]["y"].astype(int); s = np.mean([p["raw"] for p in P], 0)
        rng = np.random.default_rng(0)
        for k in range(5):
            v = s[y == k]
            ax.scatter(k + rng.uniform(-0.28, 0.28, len(v)), v, s=3, color=SERIES[0],
                       alpha=0.55, linewidths=0)
        for t in thr:
            ax.axhline(t, color="#b3261e", lw=0.7, ls="--")
        ax.set_xticks(range(5), NAMES, fontsize=6.5); ax.set_title(title); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Ensemble grade score s")
    axes[1].text(4.45, thr[-1] + 0.05, "t4", fontsize=6, color="#b3261e", ha="right")
    axes[1].text(4.45, thr[0] + 0.05, "t1", fontsize=6, color="#b3261e", ha="right")
    plt.tight_layout(pad=0.4)
    save(fig, "fig_scores")


# ------------------------------------------------------------------ selection landscape
def fig_selection():
    sel = json.loads((ROOT / "experiments/best_aptos/selection.json").read_text())
    c = pd.DataFrame(sel["candidates"]); c["size"] = c.members.map(len)
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for i, n in enumerate((1, 2, 3, 4)):
        v = c[c["size"] == n].score
        ax.hist(v, bins=np.linspace(0.74, 0.885, 45), histtype="step", lw=1.1, color=SERIES[i], label=f"{n} member{'s' if n > 1 else ''}")
    ax.axvline(sel["cv_val_score"], color="#b3261e", lw=0.9, ls="--")
    ax.text(sel["cv_val_score"] - 0.002, ax.get_ylim()[1] * 0.95, "selected\n0.8776", fontsize=6, color="#b3261e", ha="right", va="top")
    ax.set_xlabel("Validation score (cross-fitted for thr)"); ax.set_ylabel("Candidates")
    ax.legend(fontsize=6, loc="upper left")
    save(fig, "fig_selection")


# ------------------------------------------------------------------ external confusion matrices
def fig_external_cm():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for ax, (ds, lab) in zip(axes, (("idrid", "APTOS -> IDRiD (n=508)"), ("messidor2", "APTOS -> Messidor-2 (n=1,740)"))):
        y, prob, s, pred, _ = ensemble_preds(ds, "all")
        cm = np.zeros((5, 5), int)
        for a_, b_ in zip(y, pred):
            cm[a_, b_] += 1
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.0f}%", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(5), NAMES, rotation=30, ha="right", fontsize=6.5); ax.set_yticks(range(5), NAMES, fontsize=6.5)
        ax.set_title(lab); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for sp in ax.spines.values():
            sp.set_visible(False)
    plt.tight_layout(pad=0.4)
    save(fig, "fig_external_cm")


# ------------------------------------------------------------------ per-class recall on test
def fig_perclass():
    fm = json.loads((ROOT / "experiments/best_aptos/final_metrics.json").read_text())
    rows = [("Proposed ensemble", fm["evaluations"]["aptos_test"])]
    for r, n in (("resnet50_rf__aptos__5class__raw_384_s42", "ResNet-50"), ("densenet169__aptos__5class__raw_384_s42", "DenseNet-169"),
                 ("effb3_se__aptos__5class__ben_graham_384_s42", "EffNet-B3+SE"), ("inception_v3__aptos__5class__raw_384_s42", "Inception-v3")):
        rows.append((n, json.loads((ROOT / f"experiments/{r}/summary.json").read_text())["test"]))
    fig, ax = plt.subplots(figsize=(7.1, 2.1))
    x = np.arange(5); w = 0.16
    for i, (n, m) in enumerate(rows):
        v = [m["per_class"][c]["recall"] for c in NAMES]
        ax.bar(x + (i - 2) * w, v, w * 0.92, color=SERIES[i], label=n)
    ax.set_xticks(x, [f"{c} (n={rows[0][1]['per_class'][c]['support']})" for c in NAMES], fontsize=6.8)
    ax.set_ylabel("Test recall"); ax.set_ylim(0, 1.05); ax.legend(fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.grid(axis="x", visible=False)
    save(fig, "fig_perclass")


if __name__ == "__main__":
    fig_grades(); fig_preproc(); fig_class_dist(); fig_arch(); fig_training(); fig_ablation()
    fig_scores(); fig_selection(); fig_external_cm(); fig_perclass(); fig_dup_corr()
