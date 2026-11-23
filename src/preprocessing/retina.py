"""Fundus image preprocessing operations drawn from the surveyed literature.

All functions take/return uint8 RGB numpy arrays (H, W, 3) unless noted.

Provenance
----------
crop_retina / circular mask : P4 ("crop dark background"), P12 (disk detection + crop)
ben_graham                  : P4 (Gaussian blur), P12 ("subtract local colour average", Graham 2015)
clahe                       : P15
agc_p8                      : P8, ported from the authors' MATLAB (agc.m, alpha=0.4, gamma=(1-cdf)/0.7)
cdf_equalize_v_p8           : P8, ported from cdf.m (CDF matching of HSV V channel to uniform)
green_channel               : P5
zero_outer_rim              : P12 (zero out outer 5% of retina disk)
"""
from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------------------- geometry
def retina_mask(img: np.ndarray, thresh: int = 10) -> np.ndarray:
    """Binary mask of the retina (non-black) region."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > thresh).astype(np.uint8)
    # remove small specks / text overlays
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def crop_retina(img: np.ndarray, thresh: int = 10, square: bool = True) -> np.ndarray:
    """Remove black borders by cropping to the retina bounding box.

    If ``square`` the crop is padded (with black) to a square centred on the
    retina, so that subsequent resizing does not distort the disk.
    """
    mask = retina_mask(img, thresh)
    ys, xs = np.where(mask > 0)
    if len(ys) < 0.01 * mask.size:  # degenerate / very dark image: keep as is
        crop = img
    else:
        crop = img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    if not square:
        return crop
    h, w = crop.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 3), dtype=img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0: y0 + h, x0: x0 + w] = crop
    return out


def resize(img: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def circular_mask(size: int, scale: float = 1.0) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    r = scale * size / 2
    return (((yy - c) ** 2 + (xx - c) ** 2) <= r * r).astype(np.uint8)


def zero_outer_rim(img: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """P12: zero out the outer ``frac`` of the (centred, square-cropped) retina disk."""
    m = circular_mask(img.shape[0], 1.0 - frac)
    return img * m[..., None]


# --------------------------------------------------------------------------- photometric
def ben_graham(img: np.ndarray, sigma_frac: float = 1 / 30, weight: float = 4.0) -> np.ndarray:
    """Local-colour-average subtraction (Graham 2015; used by P4 & P12).

    out = weight*img - weight*GaussianBlur(img) + 128, background re-masked.
    """
    sigma = max(1.0, img.shape[0] * sigma_frac)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, weight, blur, -weight, 128)
    m = retina_mask(img)
    return out * m[..., None]


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB (P15 uses CLAHE; colour-preserving variant)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def green_channel(img: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """P5: green channel (highest vessel/lesion contrast), replicated to 3 channels."""
    g = img[..., 1]
    if use_clahe:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return np.repeat(g[..., None], 3, axis=2)


def _masked_hist(channel: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    vals = channel[mask > 0] if mask is not None else channel.ravel()
    return np.bincount(vals.ravel(), minlength=256).astype(np.float64)


def agc_p8(img: np.ndarray, alpha: float = 0.4, use_mask: bool = True) -> np.ndarray:
    """P8 adaptive gamma correction, per channel (port of authors' agc.m).

    modified_hist = alpha*hist + (1-alpha)*uniform ; gamma(l) = (1 - cdf(l)) / 0.7
    out(l) = 255 * (l/255) ** gamma(l)

    Adaptation (documented): histogram computed over retina pixels only when
    ``use_mask`` so the black background does not dominate the CDF.
    """
    mask = retina_mask(img) if use_mask else None
    out = np.empty_like(img)
    levels = np.arange(256, dtype=np.float64)
    for c in range(3):
        hist = _masked_hist(img[..., c], mask)
        n = hist.sum()
        mod = alpha * hist + (1 - alpha) * (n / 256.0)
        cdf = np.cumsum(mod / n)
        gamma = (1.0 - cdf) / 0.7
        lut = 255.0 * np.power(levels / 255.0, gamma)
        out[..., c] = np.clip(lut, 0, 255).astype(np.uint8)[img[..., c]]
    if mask is not None:
        out = out * mask[..., None]
    return out


def cdf_equalize_v_p8(img: np.ndarray, use_mask: bool = True) -> np.ndarray:
    """P8: equalise the HSV V channel so its CDF matches a uniform CDF.

    Note on the authors' cdf.m: ``[~,ind]=min(abs(input_cdf - target_cdf'),[],1)``
    followed by ``ind(img*255+1)-1`` maps level l to argmin_i |cdf(i) - l/255|,
    which is the *inverse* of histogram equalisation. We implement the
    equalisation the paper describes (l -> 255*cdf(l)) and document the
    discrepancy rather than reproduce the bug.
    """
    mask = retina_mask(img) if use_mask else None
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    hist = _masked_hist(v, mask)
    cdf = np.cumsum(hist) / hist.sum()
    lut = np.clip(np.round(255.0 * cdf), 0, 255).astype(np.uint8)
    hsv[..., 2] = lut[v]
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if mask is not None:
        out = out * mask[..., None]
    return out


def p8_enhance(img: np.ndarray) -> np.ndarray:
    """Full P8 enhancement (GAN restoration excluded): AGC then CDF equalisation."""
    return cdf_equalize_v_p8(agc_p8(img))


# --------------------------------------------------------------------------- registry
ENHANCEMENTS = {
    "raw": lambda x: x,
    "ben_graham": ben_graham,
    "clahe": clahe,
    "agc_p8": p8_enhance,
    "green": green_channel,
    "graham_rim": lambda x: zero_outer_rim(ben_graham(x)),  # P12 pipeline
}


def base_preprocess(img: np.ndarray, size: int) -> np.ndarray:
    """Deterministic geometric normalisation used to build the on-disk cache."""
    return resize(crop_retina(img), size)
