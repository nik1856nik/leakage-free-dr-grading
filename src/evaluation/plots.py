"""Static report figures (matplotlib) using the validated reference palette.

Rules applied: categorical hues in fixed slot order (never cycled), one sequential hue
(blue) for magnitude, single axis per chart, thin marks, recessive grid, text in ink
colours (never series colours), legends whenever >=2 series.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 130,
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def confusion(cm, names, path, title):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.grid(False)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color=SURFACE if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (row-normalised %)")
    ax.set_title(title)
    _save(fig, path)


def roc_curves(y, prob, names, path, title):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    n = prob.shape[1]
    ks = [1] if n == 2 else range(n)
    for k in ks:
        yt = (y == k).astype(int)
        if yt.min() == yt.max():
            continue
        fpr, tpr, _ = roc_curve(yt, prob[:, k])
        from sklearn.metrics import auc
        ax.plot(fpr, tpr, color=SERIES[k if n > 2 else 0], label=f"{names[k]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def training_curves(hist: pd.DataFrame, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (tr, va, lab) in zip(axes, [("train_loss", "val_loss", "Loss"), ("train_acc", "val_acc", "Accuracy")]):
        ax.plot(hist.epoch, hist[tr], color=SERIES[0], label="Train")
        ax.plot(hist.epoch, hist[va], color=SERIES[1], label="Validation")
        ax.set_xlabel("Epoch"); ax.set_title(lab); ax.legend()
    fig.suptitle(title, fontsize=11, fontweight="bold")
    _save(fig, path)


def metric_bars(df: pd.DataFrame, metric: str, path, title, label_col="label"):
    """Single-series horizontal bars, sorted; one hue, value labels at bar ends."""
    d = df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(d) + 1.2))
    ax.barh(d[label_col], d[metric], color=SERIES[0], height=0.62)
    for yv, v in enumerate(d[metric]):
        ax.text(v + 0.005, yv, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, min(1.0, d[metric].max() * 1.12 + 0.02))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(metric); ax.set_title(title)
    _save(fig, path)


def per_class_grouped(df: pd.DataFrame, classes, path, title, metric="recall", label_col="label"):
    """Models on x, one bar per class (classes = categorical slots 1..5, fixed order)."""
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(df) + 2), 3.8))
    w = 0.8 / len(classes)
    x = np.arange(len(df))
    for k, c in enumerate(classes):
        ax.bar(x + (k - (len(classes) - 1) / 2) * w, df[f"{c}_{metric}"], width=w * 0.92,
               color=SERIES[k], label=c)
    ax.set_xticks(x, df[label_col], rotation=30, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel(f"Per-class {metric}")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(classes), loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)
    ax.set_title(title, pad=24)
    _save(fig, path)
