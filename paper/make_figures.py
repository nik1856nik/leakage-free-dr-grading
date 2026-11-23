"""Figures for the IEEE paper (paper/figures/*.pdf|png). Every number is read from project files.

    uv run python paper/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 8, "axes.titlesize": 8.5,
    "axes.titleweight": "bold", "legend.frameon": False, "font.family": "serif",
})
NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote", name)


def ensemble_preds(ds, split):
    sel = json.loads((ROOT / "experiments/best_aptos/selection.json").read_text())
    P = [np.load(ROOT / f"experiments/best_aptos/preds/{Path(m).stem}__{ds}_{split}_tta4.npz", allow_pickle=True)
         for m in sel["members"]]
    y = P[0]["y"].astype(int)
    s = np.mean([p["raw"] for p in P], 0)
    prob = np.mean([p["prob"] for p in P], 0)
    return y, prob, s, np.digitize(s, np.sort(sel["thresholds"])), P[0]["uid"]


# ------------------------------------------------------------------ 1. pipeline diagram
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")
    boxes = [
        (1, 18, 15, 12, "Public datasets\nAPTOS, EyePACS,\nMessidor-2, IDRiD\n(genuine images,\noriginal grades)"),
        (19, 18, 16, 12, "Duplicate audit\nhigh-pass corr. ≥ 0.99\nconflicting grades\nexcluded"),
        (38, 18, 16, 12, "Grouped splits\npatient / exam /\nduplicate cluster\n70 / 15 / 15"),
        (57, 18, 19, 12, "Preprocessing\nretina crop, 512 px\nGraham / CLAHE / AGC\n(per paper)"),
        (79, 18, 20, 12, "10 models (Track A)\nAPTOS baselines\n→ top-3 tuning"),
        (1, 1, 22, 12, "EyePACS pretraining\n(top 3, 24.6k images)"),
        (26, 1, 22, 12, "APTOS fine-tuning\nclassifier + regressor\nheads"),
        (51, 1, 23, 12, "Ensemble selection\n770 candidates, 3×5-fold\ncross-fitted thresholds\n(validation only)"),
        (77, 1, 22, 12, "One-time test\nAPTOS test + external\nIDRiD, Messidor-2"),
    ]
    for i, (x, y, w, h, t) in enumerate(boxes):
        fc = "#e8f1fc" if i < 5 else "#fdf0e8"
        if i == 8:
            fc = "#e6f6ef"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc=fc, ec="#7a7975", lw=0.6))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=5.6, color=INK, linespacing=1.15)
    arrow = dict(arrowstyle="-|>", color="#52514e", lw=0.8, shrinkA=0, shrinkB=0)
    for (x1, x2) in ((16.3, 18.7), (35.3, 37.7), (54.3, 56.7), (76.3, 78.7)):
        ax.annotate("", (x2, 24), (x1, 24), arrowprops=arrow)
    for (x1, x2) in ((23.3, 25.7), (48.3, 50.7), (74.3, 76.7)):
        ax.annotate("", (x2, 7), (x1, 7), arrowprops=arrow)
    ax.annotate("", (12, 13.4), (89, 17.6), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.8,
                                                             connectionstyle="arc3,rad=-0.08"))
    ax.text(50, 15.4, "top-3 architectures", fontsize=6, color=INK2, ha="center")
    save(fig, "fig_pipeline")


# ------------------------------------------------------------------ 2. confusion matrix (APTOS test)
def fig_confusion():
    y, prob, s, pred, _ = ensemble_preds("aptos", "test")
    cm = np.zeros((5, 5), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.0f}%", ha="center", va="center", fontsize=6.5,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(5), NAMES, rotation=30, ha="right"); ax.set_yticks(range(5), NAMES)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "fig_confusion_test")


# ------------------------------------------------------------------ 3. referable-DR ROC, 3 datasets
def fig_roc():
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    for k, (ds, split, lab) in enumerate((("aptos", "test", "APTOS test"), ("idrid", "all", "IDRiD (external)"),
                                          ("messidor2", "all", "Messidor-2 (external)"))):
        y, prob, s, pred, _ = ensemble_preds(ds, split)
        yt, sc = (y >= 2).astype(int), prob[:, 2:].sum(1)
        fpr, tpr, _ = roc_curve(yt, sc)
        auc = roc_auc_score(yt, sc)
        ax.plot(fpr, tpr, color=SERIES[k], lw=1.4, label=f"{lab}  AUC {auc:.3f}")
        op_fpr, op_tpr = ((pred >= 2) & (yt == 0)).sum() / (yt == 0).sum(), ((pred >= 2) & (yt == 1)).sum() / yt.sum()
        ax.plot([op_fpr], [op_tpr], "o", ms=4, color=SERIES[k], mec="white", mew=0.8)
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, ls="--")
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=6.5)
    ax.text(0.40, 0.42, "dots: operating point at the\nfrozen APTOS thresholds", fontsize=6, color=INK2)
    save(fig, "fig_roc_referable")


# ------------------------------------------------------------------ 4. Grad-CAM grid
def fig_gradcam():
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from src.datasets.fundus import FundusDataset, load_split
    from src.models.zoo import build
    ck = torch.load(ROOT / "best_model/ft__effb3_cls__aptos.pt", map_location="cpu")
    net = build("effb3_se", 5); net.load_state_dict(ck["model"]); net.eval().cuda()
    df = load_split("aptos", "5class", "test")
    ds, raw = FundusDataset(df, 512, False, "ben_graham"), FundusDataset(df, 512, False, "raw")
    y, prob, s, pred, _ = ensemble_preds("aptos", "test")
    rng = np.random.default_rng(3)
    picks = []
    for k in (1, 2, 3, 4):
        ok = np.where((y == k) & (pred == k))[0]
        picks.append((rng.choice(ok), "correct"))
    for t in (1, 3, 4):
        bad = np.where((y == t) & (pred == 2))[0]
        picks.append((rng.choice(bad), "error"))
    ok0 = np.where((y == 0) & (pred == 0))[0]
    picks.insert(0, (rng.choice(ok0), "correct"))
    cam = GradCAM(model=net, target_layers=[net.backbone.conv_head])
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.1), gridspec_kw=dict(hspace=0.28, wspace=0.04))
    for a, (i, kind) in zip(axes.ravel(), picks):
        x, _ = ds[i]
        g = cam(input_tensor=x[None].cuda(), targets=[ClassifierOutputTarget(int(pred[i]))])[0]
        img = cv2.resize(raw.load_rgb(i), (512, 512))
        heat = cv2.applyColorMap(np.uint8(255 * g), cv2.COLORMAP_JET)[..., ::-1]
        over = np.uint8(0.6 * img + 0.4 * heat) * (img.sum(2, keepdims=True) > 30)
        a.imshow(over); a.axis("off")
        a.set_title(f"true {NAMES[y[i]]} → pred {NAMES[pred[i]]}", fontsize=7,
                    color=INK if kind == "correct" else "#b3261e")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    save(fig, "fig_gradcam")


# ------------------------------------------------------------------ 5. conflicting duplicates (APTOS)
def fig_duplicates():
    d = pd.read_csv(ROOT / "data/metadata/aptos_duplicate_pairs.csv")
    m = pd.read_csv(ROOT / "data/metadata/aptos.csv").set_index("uid")
    conf = d[d.label_a != d.label_b].sort_values("corr", ascending=False)
    pairs = [conf.iloc[0], conf.iloc[3]]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.0))
    for c, r in enumerate(pairs):
        for j, (u, lab) in enumerate(((r.uid_a, r.label_a), (r.uid_b, r.label_b))):
            img = cv2.cvtColor(cv2.imread(str(m.loc[u, "cache"])), cv2.COLOR_BGR2RGB)
            a = axes[2 * c + j]; a.imshow(img); a.axis("off")
            a.set_title(f"{u.split('/')[1][:8]}…  grade {lab} ({NAMES[lab]})", fontsize=6.5)
    fig.text(0.27, 0.01, f"pair 1: correlation {pairs[0]['corr']:.3f}", ha="center", fontsize=6.5, color=INK2)
    fig.text(0.75, 0.01, f"pair 2: correlation {pairs[1]['corr']:.3f}", ha="center", fontsize=6.5, color=INK2)
    plt.tight_layout(pad=0.4)
    save(fig, "fig_duplicates")


# ------------------------------------------------------------------ 6. test QWK of all evaluated models
def fig_test_bars():
    mc = pd.read_csv(ROOT / "reports/model_comparison.csv")
    lab = {"effb3_se": "EffB3+SE (P4)", "inception_v3": "Inception-v3 (P7)", "densenet169": "DenseNet-169 (P14)",
           "hybrid_svm": "GoogLeNet+ResNet18+SVM (P6)", "light2stage": "Two-stage CNN (P10)",
           "mil_resnet18": "Attention-MIL (P12)", "mscnn_lstm": "MS-CNN+LSTM (P15)", "p8_vgg16": "AGC+VGG16 (P8)",
           "vit_b16": "ViT-B/16 (P14)"}
    rows = []
    for _, r in mc.iterrows():
        if r.run.startswith("BEST"):
            name = "Proposed ensemble"
        elif r.model == "resnet50_rf":
            name = "ResNet50 (P11)" if r.run.count("__") > 2 else "ResNet50+RF (P11)"
        else:
            name = lab[r.model]
        rows.append((name, r.qwk, r.accuracy))
    rows.sort(key=lambda t: t[1])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    yy = np.arange(len(rows))
    for a, idx, title in ((axes[0], 1, "Quadratic weighted kappa"), (axes[1], 2, "Accuracy")):
        vals = [r[idx] for r in rows]
        cols = [SERIES[1] if r[0] == "Proposed ensemble" else SERIES[0] for r in rows]
        a.barh(yy, vals, color=cols, height=0.62)
        for k, v in enumerate(vals):
            a.text(v + 0.004, k, f"{v:.3f}", va="center", fontsize=6.2, color=INK)
        a.set_xlim(0.6 if idx == 1 else 0.6, 1.0); a.set_title(title)
        a.grid(axis="y", visible=False)
    axes[0].set_yticks(yy, [r[0] for r in rows], fontsize=6.8)
    plt.tight_layout(pad=0.4)
    save(fig, "fig_test_bars")


if __name__ == "__main__":
    fig_pipeline(); fig_confusion(); fig_roc(); fig_duplicates(); fig_test_bars(); fig_gradcam()
