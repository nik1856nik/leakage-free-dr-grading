"""Torch dataset over the cached, retina-cropped fundus images."""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.sources import ROOT
from src.preprocessing.retina import ENHANCEMENTS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- task label maps
TASKS = {
    # name: (num_classes, mapping from ICDR grade -> class or None to drop)
    "5class": (5, lambda g: g),
    "binary": (2, lambda g: int(g >= 1)),            # any DR (P6, P10 stage 1)
    "rdr": (2, lambda g: int(g >= 2)),               # referable DR (P7, P12)
    "3class_p6": (3, lambda g: 0 if g == 0 else (1 if g <= 2 else 2)),  # P6 original
    "severity4": (4, lambda g: g - 1 if g >= 1 else None),              # P10 stage 2
}
CLASS_NAMES = {
    "5class": ["No DR", "Mild", "Moderate", "Severe", "PDR"],
    "binary": ["No DR", "DR"],
    "rdr": ["Non-referable", "Referable"],
    "3class_p6": ["NDR", "MDR(1-2)", "PDR(3-4)"],
    "severity4": ["Mild", "Moderate", "Severe", "PDR"],
}


def load_split(dataset: str, task: str = "5class", split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "splits" / f"{dataset}_5class.csv")
    df["grade"] = df.label
    df["target"] = df.grade.map(TASKS[task][1])
    df = df[df.target.notna()].copy()
    df["target"] = df.target.astype(int)
    if split is not None:
        df = df[df.split == split]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- transforms
def build_transforms(size: int, train: bool, aug: str = "standard",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD) -> A.Compose:
    ops = [A.Resize(size, size, interpolation=cv2.INTER_AREA)]
    if train and aug != "none":
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(rotate=(-180, 180), scale=(0.9, 1.1), translate_percent=(-0.03, 0.03),
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.9),
        ]
        if aug in ("standard", "strong"):
            ops += [A.RandomBrightnessContrast(0.15, 0.15, p=0.5)]
        if aug == "strong":
            ops += [A.HueSaturationValue(5, 10, 5, p=0.3),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
                    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.03, 0.08),
                                    hole_width_range=(0.03, 0.08), p=0.2)]
    ops += [A.Normalize(mean=mean, std=std)]
    return A.Compose(ops)


class FundusDataset(Dataset):
    """Reads cached images, applies a literature enhancement, then augmentation."""

    def __init__(self, df: pd.DataFrame, size: int = 384, train: bool = False,
                 enhancement: str = "raw", aug: str = "standard",
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.df = df.reset_index(drop=True)
        self.enh = ENHANCEMENTS[enhancement]
        self.tf = build_transforms(size, train, aug, mean, std)
        self.paths = self.df.cache.tolist()
        self.targets = self.df.target.values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def load_rgb(self, i: int) -> np.ndarray:
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.paths[i])
        return self.enh(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __getitem__(self, i):
        x = self.tf(image=self.load_rgb(i))["image"]
        return torch.from_numpy(x.transpose(2, 0, 1)), self.targets[i]


def class_weights(targets: np.ndarray, n: int, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency^power class weights, normalised to mean 1."""
    cnt = np.bincount(targets, minlength=n).astype(float)
    w = (cnt.sum() / np.maximum(cnt, 1)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def balanced_sampler_weights(targets: np.ndarray, power: float = 0.5) -> np.ndarray:
    cnt = np.bincount(targets).astype(float)
    return (1.0 / cnt[targets]) ** power
