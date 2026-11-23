"""Black-and-white, print-friendly versions of every paper figure -> paper/figures_bw/.

Same data and layout as make_figures.py / make_figures_extra.py, with:
  * series told apart by gray level + line style (lines/histograms) or hatching (bars), not colour;
  * fundus photographs converted to grayscale with CLAHE + gamma brightening (no dark prints);
  * Grad-CAM drawn as contour outlines on the brightened grayscale image instead of a colour map.

    uv run python paper/make_figures_bw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.axes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "scripts"))
import make_figures as mf  # noqa: E402
import make_figures_extra as mx  # noqa: E402

OUT = ROOT / "paper" / "figures_bw"
OUT.mkdir(parents=True, exist_ok=True)

GRAYS = ["#000000", "#5a5a5a", "#8c8c8c", "#2e2e2e", "#b4b4b4"]
STYLE = {g: s for g, s in zip(GRAYS, ["-", "--", ":", "-.", (0, (5, 1, 1, 1))])}
HATCH = {g: h for g, h in zip(GRAYS, ["", "////", "....", "xxxx", "\\\\\\\\"])}

# ---- swap the palette in both figure modules
for mod in (mf, mx):
    mod.SERIES = list(GRAYS)
mf.OUT = OUT
mx.BLUES5 = ["#f0f0f0", "#c8c8c8", "#959595", "#5f5f5f", "#262626"]


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", name)


mf.save = _save
mx.save = _save

# ---- line style / hatch by gray level, applied transparently to every plot call
_plot, _hist, _bar, _barh, _imshow = (matplotlib.axes.Axes.plot, matplotlib.axes.Axes.hist,
                                      matplotlib.axes.Axes.bar, matplotlib.axes.Axes.barh,
                                      matplotlib.axes.Axes.imshow)


def _c(kw):
    c = kw.get("color")
    return c if isinstance(c, str) and c in STYLE else None


def plot(self, *a, **kw):
    c = _c(kw)
    if c and "ls" not in kw and "linestyle" not in kw and not (len(a) >= 3 and isinstance(a[2], str)):
        kw["linestyle"] = STYLE[c]
    return _plot(self, *a, **kw)


def hist(self, *a, **kw):
    c = _c(kw)
    if c:
        kw.setdefault("linestyle", STYLE[c])
    return _hist(self, *a, **kw)


def _bars(orig):
    def f(self, *a, **kw):
        col = kw.get("color")
        cols = col if isinstance(col, list) else [col]
        r = orig(self, *a, **{**kw, "edgecolor": "black", "linewidth": 0.4})
        for i, patch in enumerate(r.patches):
            c = cols[i % len(cols)]
            if isinstance(c, str) and c in HATCH:
                patch.set_hatch(HATCH[c])
        return r
    return f


def bright_gray(rgb):
    """Fundus RGB -> brightened, contrast-enhanced grayscale (prints without going dark)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    m = (rgb.max(2) > 15).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(m)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 1, thickness=-1)   # filled retina disc
    mask = mask.astype(bool)
    g = np.clip(255 * (L / 255.0) ** 0.7, 0, 255).astype(np.uint8)
    return np.where(mask, g, 0).astype(np.uint8)


def imshow(self, X, *a, **kw):
    if isinstance(X, np.ndarray) and X.ndim == 3 and X.shape[2] == 3 and X.dtype == np.uint8:
        return _imshow(self, bright_gray(X), *a, cmap="gray", vmin=0, vmax=255, **kw)
    if kw.get("cmap") == "Blues":
        kw["cmap"] = "Greys"
    return _imshow(self, X, *a, **kw)


matplotlib.axes.Axes.plot, matplotlib.axes.Axes.hist = plot, hist
matplotlib.axes.Axes.bar, matplotlib.axes.Axes.barh = _bars(_bar), _bars(_barh)
matplotlib.axes.Axes.imshow = imshow


# ---- Grad-CAM in black and white: contours on the brightened grayscale photo
def fig_gradcam_bw():
    import matplotlib.pyplot as plt
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from src.datasets.fundus import FundusDataset, load_split
    from src.models.zoo import build
    ck = torch.load(ROOT / "best_model/ft__effb3_cls__aptos.pt", map_location="cpu")
    net = build("effb3_se", 5); net.load_state_dict(ck["model"]); net.eval().cuda()
    df = load_split("aptos", "5class", "test")
    ds, raw = FundusDataset(df, 512, False, "ben_graham"), FundusDataset(df, 512, False, "raw")
    y, prob, s, pred, _ = mf.ensemble_preds("aptos", "test")
    rng = np.random.default_rng(3)          # same examples as the colour figure
    picks = []
    for k in (1, 2, 3, 4):
        picks.append((rng.choice(np.where((y == k) & (pred == k))[0]), "correct"))
    for t in (1, 3, 4):
        picks.append((rng.choice(np.where((y == t) & (pred == 2))[0]), "error"))
    picks.insert(0, (rng.choice(np.where((y == 0) & (pred == 0))[0]), "correct"))
    cam = GradCAM(model=net, target_layers=[net.backbone.conv_head])
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.1), gridspec_kw=dict(hspace=0.28, wspace=0.04))
    for a, (i, kind) in zip(axes.ravel(), picks):
        x, _ = ds[i]
        g = cam(input_tensor=x[None].cuda(), targets=[ClassifierOutputTarget(int(pred[i]))])[0]
        img = cv2.resize(raw.load_rgb(i), (512, 512))
        bg = bright_gray(img)
        g = g * (cv2.resize((bg > 0).astype(np.uint8), g.shape[::-1]) > 0)     # heat only inside the retina
        _imshow(a, bg, cmap="gray", vmin=0, vmax=255)
        a.contourf(g, levels=[0.6, 1.01], colors=["black"], alpha=0.28)
        a.contour(g, levels=[0.6], colors=["black"], linewidths=1.1)
        a.contour(g, levels=[0.6], colors=["white"], linewidths=0.4, linestyles="--")
        a.axis("off")
        a.set_title(f"{mf.NAMES[y[i]]} -> {mf.NAMES[pred[i]]}" + ("  (error)" if kind == "error" else "  (correct)"),
                    fontsize=6.8, color="black", weight="bold" if kind == "error" else "normal")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    _save(fig, "fig_gradcam")


if __name__ == "__main__":
    mf.fig_pipeline(); mf.fig_confusion(); mf.fig_roc(); mf.fig_duplicates(); mf.fig_test_bars()
    fig_gradcam_bw()
    mx.fig_grades(); mx.fig_preproc(); mx.fig_class_dist(); mx.fig_arch(); mx.fig_training(); mx.fig_ablation()
    mx.fig_scores(); mx.fig_selection(); mx.fig_external_cm(); mx.fig_perclass(); mx.fig_dup_corr()
