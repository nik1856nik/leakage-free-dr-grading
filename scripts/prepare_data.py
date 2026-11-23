"""Build metadata, a retina-cropped image cache, quality stats, duplicate groups and splits.

Usage:
    uv run python scripts/prepare_data.py --datasets idrid messidor2 aptos eyepacs [--size 512]

Outputs:
    data/processed/<dataset>/<size>/<image>.jpg     cropped+resized cache (JPEG q95, 4:4:4)
    data/metadata/<dataset>.csv                     metadata + quality + duplicate groups
    data/metadata/<dataset>_duplicates.csv          near-duplicate clusters (with label conflicts)
    data/splits/<dataset>_5class.csv                uid, label, group, split (train/val/test)
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.sources import LOADERS, ROOT, available, eyepacs_zip  # noqa: E402
from src.preprocessing.retina import base_preprocess, retina_mask  # noqa: E402

SEED = 42
_ZIP = None


def _read_raw(src: str) -> np.ndarray | None:
    global _ZIP
    if src.startswith("zip:"):
        if _ZIP is None:
            _ZIP = eyepacs_zip()
        buf = np.frombuffer(_ZIP.read(src[4:]), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _process(args):
    uid, src, out_path, size = args
    rec = dict(uid=uid, ok=False)
    try:
        if src in ("", "zip:"):
            rec["error"] = "missing"
            return rec
        img = _read_raw(src)
        if img is None:
            rec["error"] = "decode_failed"
            return rec
        rec.update(orig_h=img.shape[0], orig_w=img.shape[1])
        if not Path(out_path).exists():
            out = base_preprocess(img, size)
            cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                         cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
        out = cv2.cvtColor(cv2.imread(out_path), cv2.COLOR_BGR2RGB)
        m = retina_mask(out)
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        inside = gray[m > 0]
        pil = Image.fromarray(out)
        rec.update(
            ok=True,
            retina_frac=float(m.mean()),
            brightness=float(inside.mean()) if inside.size else 0.0,
            contrast=float(inside.std()) if inside.size else 0.0,
            sharpness=float(cv2.Laplacian(gray, cv2.CV_64F)[m > 0].var()) if inside.size else 0.0,
            phash=str(imagehash.phash(pil, hash_size=8)),
            dhash=str(imagehash.dhash(pil, hash_size=8)),
        )
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


# --------------------------------------------------------------------------- duplicates
DUP_THR = 0.99  # high-pass corr. Audit: exact copies = 1.000; max distinct pair 0.945 (IDRiD), 0.983 (Messidor-2, same-camera/session images)


_YY, _XX = np.mgrid[:96, :96]
_INNER = np.hypot(_YY - 47.5, _XX - 47.5) / 48 <= 0.80  # retina interior only


def _highpass_vec(path: str, s: int = 96) -> np.ndarray:
    """Green-channel high-pass structure inside the central 80% radius of the retina disc.

    The rim is excluded: EyePACS cameras imprint the same notch/edge pattern on every image,
    which made distinct eyes correlate >= 0.99 when the full square was used (16-image false
    cluster). Excluding it leaves the IDRiD/Messidor-2/APTOS duplicate pairs exactly unchanged.
    """
    g = cv2.imread(path)
    g = cv2.resize(g, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = (g - cv2.GaussianBlur(g, (0, 0), s / 30))[..., 1][_INNER]
    return (g - g.mean()) / (g.std() + 1e-6)


def near_duplicate_clusters(df: pd.DataFrame, thr: float = DUP_THR):
    """Union-find clusters of images whose high-pass (vessel/lesion) structure correlates >= thr.

    Perceptual 64-bit hashes were rejected: every fundus image looks alike at 8x8 and
    they produced mass false-positive clusters (IDRiD: 181/516 images flagged).
    Returns (cluster_root_index per row, list of (i, j, corr) pairs).
    """
    import torch
    idx = np.where(df.ok.values.astype(bool))[0]
    with ProcessPoolExecutor(12) as ex:
        H = np.stack(list(ex.map(_highpass_vec, df.cache.values[idx], chunksize=64)))
    Ht = torch.from_numpy(H).cuda().half() / np.sqrt(H.shape[1])
    parent = np.arange(len(df))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = []
    for s in range(0, len(idx), 4096):
        C = (Ht[s: s + 4096] @ Ht.T).float()
        ii, jj = torch.nonzero(C >= thr, as_tuple=True)
        for i, j, c in zip((ii + s).tolist(), jj.tolist(), C[ii, jj].tolist()):
            if i < j:
                pairs.append((idx[i], idx[j], c))
                ri, rj = find(idx[i]), find(idx[j])
                if ri != rj:
                    parent[rj] = ri
    return np.array([find(i) for i in range(len(df))]), pairs


# --------------------------------------------------------------------------- splits
def make_splits(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Stratified, group-aware ~70/15/15 split (20 folds -> 14/3/3)."""
    split = pd.Series("", index=df.index)
    lab = df.label.astype(int).values
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), int)
    for k, (_, te) in enumerate(sgkf.split(df, lab, df.group.values)):
        fold[te] = k
    split[:] = np.where(fold < 3, "test", np.where(fold < 6, "val", "train"))
    return split


def official_holdout_splits(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Keep dataset's official test set; carve ~15% of total out of official train for val."""
    split = pd.Series("test", index=df.index)
    tr = df[df.official_split == "train"]
    frac = 0.15 * len(df) / len(tr)
    k = max(2, int(round(1 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    _, va = next(sgkf.split(tr, tr.label.astype(int), tr.group))
    split[tr.index] = "train"
    split[tr.index[va]] = "val"
    return split


# --------------------------------------------------------------------------- main
def prepare(name: str, size: int, workers: int):
    print(f"\n=== {name}")
    meta = LOADERS[name]()
    out_dir = ROOT / "data" / "processed" / name / str(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r.uid, r.src, str(out_dir / f"{r.image if name != 'idrid' else r.uid.split('/')[1]}.jpg"), size)
            for r in meta.itertuples()]
    meta["cache"] = [j[2] for j in jobs]
    # EyePACS: members are read sequentially per worker from one zip; sort by src helps locality
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = list(tqdm(ex.map(_process, jobs, chunksize=16), total=len(jobs), desc=name))
    q = pd.DataFrame(recs)
    meta = meta.merge(q, on="uid", how="left")

    bad = meta[~meta.ok.astype(bool)]
    print(f"  images: {len(meta)} | unreadable/missing: {len(bad)}")
    if len(bad):
        print(bad[["uid", "error"]].head(10).to_string())

    # near-duplicates (high-pass structure correlation)
    roots, pairs = near_duplicate_clusters(meta)
    meta["dup_cluster"] = meta.uid.values[roots]
    pd.DataFrame([(meta.uid[i], meta.uid[j], round(c, 4), meta.label[i], meta.label[j])
                  for i, j, c in pairs], columns=["uid_a", "uid_b", "corr", "label_a", "label_b"]) \
        .to_csv(ROOT / "data" / "metadata" / f"{name}_duplicate_pairs.csv", index=False)
    sizes = meta.groupby("dup_cluster").uid.transform("size")
    dups = meta[sizes > 1].copy()
    meta["exclude_reason"] = ""
    if len(dups):
        conf = dups.groupby("dup_cluster").label.nunique()
        dups["label_conflict"] = dups.dup_cluster.map(conf > 1)
        dups.sort_values("dup_cluster")[["dup_cluster", "uid", "label", "official_split", "group",
                                         "label_conflict"]] \
            .to_csv(ROOT / "data" / "metadata" / f"{name}_duplicates.csv", index=False)
        print(f"  near-duplicate clusters (corr>={DUP_THR}): {dups.dup_cluster.nunique()} "
              f"({len(dups)} images), with conflicting labels: {int(conf.gt(1).sum())}")
        # ---- de-duplication policy (documented in reports/dataset_status.md)
        for cl, g in dups.groupby("dup_cluster"):
            official_test = g[g.official_split == "test"]
            if len(official_test):
                # never alter an official test set: drop the training-side copies instead
                for u in g[g.official_split != "test"].uid:
                    meta.loc[meta.uid == u, "exclude_reason"] = "duplicate_of_official_test_image"
            elif g.label.nunique() > 1:
                for u in g.uid:
                    meta.loc[meta.uid == u, "exclude_reason"] = "duplicate_with_conflicting_labels"
            else:
                for u in g.uid.iloc[1:]:
                    meta.loc[meta.uid == u, "exclude_reason"] = "redundant_duplicate_copy"
        print("  excluded:", meta.exclude_reason[meta.exclude_reason != ""].value_counts().to_dict())
    # merge leakage groups: patient/exam group  ∪  duplicate cluster (union-find)
    parent = {}

    def f(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(meta.group, meta.dup_cluster):
        ra, rb = f(a), f(b)
        if ra != rb:
            parent[rb] = ra
    meta["leak_group"] = [f(a) for a in meta.group]

    # class distribution
    print("  label distribution:", meta.label.value_counts(dropna=False).sort_index().to_dict())
    meta.to_csv(ROOT / "data" / "metadata" / f"{name}.csv", index=False)

    # splits (5-class): only gradable, readable, labelled images
    usable = meta[meta.ok.astype(bool) & meta.label.notna() & (meta.exclude_reason == "")].copy()
    if "gradable" in usable:
        usable = usable[usable.gradable.astype(float) == 1]
    usable["label"] = usable.label.astype(int)
    usable["group"] = usable.leak_group
    if (usable.official_split == "test").any():
        usable["split"] = official_holdout_splits(usable)
    else:
        usable["split"] = make_splits(usable)
    cols = ["uid", "dataset", "label", "group", "split", "cache"] + \
           [c for c in ("dme", "eye") if c in usable]
    usable[cols].to_csv(ROOT / "data" / "splits" / f"{name}_5class.csv", index=False)
    # leakage audit
    gs = usable.groupby("group").split.nunique()
    assert (gs == 1).all(), "group spans multiple splits!"
    print("  split sizes:", usable.split.value_counts().to_dict())
    print("  per-split label dist:\n" +
          pd.crosstab(usable.split, usable.label).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["idrid", "messidor2", "aptos", "eyepacs"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    for d in ("metadata", "splits"):
        (ROOT / "data" / d).mkdir(parents=True, exist_ok=True)
    av = available()
    for name in a.datasets:
        if not av.get(name):
            print(f"[skip] {name}: raw data not available yet")
            continue
        prepare(name, a.size, a.workers)


if __name__ == "__main__":
    main()
