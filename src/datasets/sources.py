"""Raw-dataset readers: build a uniform metadata table for each dataset.

Every table has columns:
    uid        unique id "<dataset>/<image stem>"
    dataset    aptos | eyepacs | messidor2 | idrid
    image      original image stem
    label      ICDR grade 0..4 (NaN if ungradable/missing)
    group      leakage group (patient / exam / near-duplicate cluster)
    src        how to read the raw image (path, or "zip:<member>")
    official_split  train/test where the dataset defines one, else ""
    + dataset-specific extras (dme, gradable, eye)
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


# --------------------------------------------------------------------------- split-zip reader
class MultiPartFile(io.RawIOBase):
    """Read-only, seekable view over byte-split archive parts (file.zip.001, .002, ...)."""

    def __init__(self, parts: list[Path]):
        self.parts = [Path(p) for p in parts]
        self.sizes = [p.stat().st_size for p in self.parts]
        self.offsets = np.cumsum([0] + self.sizes)
        self.total = int(self.offsets[-1])
        self.pos = 0
        self._fh: dict[int, io.BufferedReader] = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.total}[whence]
        self.pos = max(0, base + off)
        return self.pos

    def _handle(self, i):
        if i not in self._fh:
            self._fh[i] = open(self.parts[i], "rb")
        return self._fh[i]

    def readinto(self, b):
        n, done = len(b), 0
        while done < n and self.pos < self.total:
            i = int(np.searchsorted(self.offsets, self.pos, side="right") - 1)
            local = self.pos - int(self.offsets[i])
            take = min(n - done, self.sizes[i] - local)
            fh = self._handle(i)
            fh.seek(local)
            chunk = fh.read(take)
            b[done: done + len(chunk)] = chunk
            done += len(chunk)
            self.pos += len(chunk)
            if not chunk:
                break
        return done

    def close(self):
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()
        super().close()


def eyepacs_zip() -> zipfile.ZipFile:
    parts = sorted((RAW / "eyepacs").glob("train.zip.0*"))
    if not parts:
        raise FileNotFoundError("EyePACS train.zip.00* parts not found")
    return zipfile.ZipFile(MultiPartFile(parts))


# --------------------------------------------------------------------------- per dataset
def load_idrid() -> pd.DataFrame:
    base = RAW / "idrid" / "B. Disease Grading"
    rows = []
    for split, sub, csv in [
        ("train", "a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv"),
        ("test", "b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv"),
    ]:
        d = pd.read_csv(base / "2. Groundtruths" / csv).iloc[:, :3].dropna()
        d.columns = ["image", "label", "dme"]
        for r in d.itertuples():
            p = base / "1. Original Images" / sub / f"{r.image}.jpg"
            rows.append(dict(image=r.image, label=int(r.label), dme=int(r.dme),
                             src=str(p), official_split=split,
                             # train and test reuse the same names (IDRiD_001...): make unique
                             uid=f"idrid/{split}_{r.image}"))
    df = pd.DataFrame(rows)
    df["dataset"] = "idrid"
    df["group"] = df["uid"]  # no patient ids; refined by near-duplicate clustering
    return df


def load_messidor2() -> pd.DataFrame:
    base = RAW / "messidor2"
    grades = pd.read_csv(base / "messidor_data.csv")
    pairs = pd.read_csv(base / "messidor-2.csv", sep=";")
    pairs.columns = [c.strip() for c in pairs.columns]
    # grade file uses ".jpg" while ADCIS images are ".JPG": match case-insensitively
    on_disk = {f.lower(): f for f in os.listdir(base / "IMAGES")}
    exam_of, eye_of = {}, {}
    for i, (left, right) in enumerate(zip(pairs.iloc[:, 0], pairs.iloc[:, 1])):
        for eye, name in (("left", left), ("right", right)):
            exam_of[name.strip().lower()] = f"messidor2/exam{i:04d}"
            eye_of[name.strip().lower()] = eye
    rows = []
    for r in grades.itertuples():
        key = r.image_id.strip().lower()
        name = on_disk.get(key, r.image_id.strip())
        rows.append(dict(
            uid=f"messidor2/{Path(name).stem}", image=Path(name).stem,
            label=r.adjudicated_dr_grade, dme=r.adjudicated_dme,
            gradable=r.adjudicated_gradable, src=str(base / "IMAGES" / name),
            group=exam_of.get(key, f"messidor2/{name}"), eye=eye_of.get(key, ""),
            official_split=""))
    df = pd.DataFrame(rows)
    df["dataset"] = "messidor2"
    return df


def load_aptos() -> pd.DataFrame:
    base = RAW / "aptos2019"
    d = pd.read_csv(base / "train.csv")
    df = pd.DataFrame(dict(
        uid="aptos/" + d.id_code, image=d.id_code, label=d.diagnosis.astype(int),
        src=[str(base / "train_images" / f"{i}.png") for i in d.id_code],
        official_split=""))
    df["dataset"] = "aptos"
    df["group"] = df["uid"]  # refined by near-duplicate clustering
    return df


def load_eyepacs() -> pd.DataFrame:
    base = RAW / "eyepacs"
    d = pd.read_csv(base / "trainLabels.csv")
    with eyepacs_zip() as z:
        members = {Path(n).stem: n for n in z.namelist() if n.lower().endswith((".jpeg", ".jpg"))}
    df = pd.DataFrame(dict(
        uid="eyepacs/" + d.image, image=d.image, label=d.level.astype(int),
        src=["zip:" + members.get(i, "") for i in d.image], official_split="train"))
    df["eye"] = d.image.str.split("_").str[1]
    df["group"] = "eyepacs/patient" + d.image.str.split("_").str[0]  # both eyes together
    df["dataset"] = "eyepacs"
    missing = (df.src == "zip:").sum()
    if missing:
        print(f"[eyepacs] WARNING {missing} labelled images not found in archive")
    return df


LOADERS = {"idrid": load_idrid, "messidor2": load_messidor2,
           "aptos": load_aptos, "eyepacs": load_eyepacs}


def available() -> dict[str, bool]:
    return {
        "idrid": (RAW / "idrid" / "B. Disease Grading").exists(),
        "messidor2": (RAW / "messidor2" / "messidor_data.csv").exists()
                     and (RAW / "messidor2" / "IMAGES").exists(),
        "aptos": (RAW / "aptos2019" / "train.csv").exists(),
        # raw parts are deleted after caching (disk); the 512 px cache + split file then suffice
        "eyepacs": (RAW / "eyepacs" / "trainLabels.csv").exists()
                   and (len(list((RAW / "eyepacs").glob("train.zip.0*"))) == 5
                        or (ROOT / "data" / "splits" / "eyepacs_5class.csv").exists()),
    }
