# Dataset Status

*Last updated 2026-11-19 (final). Only genuine fundus images with authentic labels are used.*

## Summary

| Dataset | Status | Location | Images | Labels |
|---|---|---|---|---|
| **IDRiD** (Disease Grading) | ✅ **Available, validated** | `data/raw/idrid` → symlink to `~/Desktop/idrid` | 516 (413 train / 103 test), 4288×2848 JPG | DR grade 0–4 + DME risk 0–2, CC-BY-4.0 |
| **APTOS 2019** | ✅ **Available, validated** (Kaggle; archive integrity-checked) | `data/raw/aptos2019` | 3,662 labelled (test labels private) | 5-class ICDR: {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295} |
| **EyePACS (Kaggle DR 2015)** | ✅ **Available, validated** (official Kaggle train archive, 5 byte-split parts read in place; all images decodable) | `data/raw/eyepacs` → cached 512 px in `data/processed/eyepacs` | 35,126 train (17,563 patients × 2 eyes); Kaggle test set not downloaded | 5-class ICDR: {0: 25810, 1: 2443, 2: 5292, 3: 873, 4: 708} |
| **Messidor-2** | ✅ **Available, validated** (images from ADCIS, supplied by the user; grades from Kaggle `google-brain/messidor2-dr-grades`) | `data/raw/messidor2/IMAGES` | 1,748 (874 exams); sizes 2240×1488 (616), 2304×1536 (604), 1440×960 (528) | Krause et al. 2018 adjudicated ICDR grade + rDME + gradability; 4 ungradable excluded |

Storage after preparation: all four datasets are cached as 512 px retina-cropped JPEGs in `data/processed/` (2.7 GB).
The 35 GB of raw EyePACS zip parts were **deleted after caching** (verified: 35,126/35,126 cached files present);
`trainLabels.csv` is kept. The APTOS archive was extracted (train only) and deleted. Re-running
`prepare_data.py` for EyePACS needs the raw parts to be downloaded again (`scripts/download_datasets.sh`).

## IDRiD validation
- Train DR: {0: 134, 1: 20, 2: 136, 3: 74, 4: 49}; Test DR: {0: 34, 1: 5, 2: 32, 3: 19, 4: 13}.
- **Mild is extremely rare** (25 images total, 5 in test), so per-class Mild metrics on IDRiD have very wide confidence intervals.
- The label CSV has trailing empty columns (handled in the loader). The official train/test split is kept as-is. There are no patient IDs.

## Manual download instructions

### A. Kaggle API token (unblocks APTOS, EyePACS and Messidor-2 grades)
1. Log in at https://www.kaggle.com → *Settings* → *API* → **Create New Token** (downloads `kaggle.json`).
2. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
3. Accept the competition rules (click **"Join competition" / "I understand and accept"** on the *Data* tab) for **both**:
   - https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
   - https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data
4. Tell me when that's done and I'll run the downloads (`scripts/check_datasets.py --download`).

### B. Messidor-2 images (manual only)
1. Go to https://www.adcis.net/en/third-party/messidor2/ and complete the **licence agreement / download request form** (academic use).
2. Download the image archive(s) you receive the link for (`IMAGES.zip`, ~2.2 GB, 1,748 images `IM00xxxx.JPG/.jpg`), plus the `messidor-2.csv` pairing file if it's provided.
3. Place them at:
```
DR_Detection/data/raw/messidor2/
├── IMAGES/            # all 1,748 images, unchanged filenames
└── messidor-2.csv     # (if provided) left/right image pairing per exam
```
4. Grades come from Kaggle `google-brain/messidor2-dr-grades` (`messidor_data.csv`: `image_id, adjudicated_dr_grade, adjudicated_dme, adjudicated_gradable`). I'll download that once the Kaggle token exists.

### Expected layout after download
```
data/raw/
├── idrid/            (present)
├── aptos2019/        train_images/*.png, train.csv (id_code, diagnosis)
├── eyepacs/          train/*.jpeg, trainLabels.csv (image, level),
│                     test/*.jpeg, retinopathy_solution.csv (image, level, Usage)
└── messidor2/        IMAGES/*, messidor_data.csv
```

## Storage estimate

| Item | Size |
|---|---|
| APTOS raw | ~10 GB |
| EyePACS train (zip + extracted) | ~35 GB zipped; extract one part at a time, then preprocess to 512 px and **delete the raw files** (~5 GB cached) |
| EyePACS test (optional, for a larger held-out set) | ~53 GB zipped. **Recommended to skip initially** because of disk space |
| Messidor-2 | ~2.2 GB |
| Preprocessed caches (512 px PNG/JPG, all datasets) | ~8–10 GB |
| Checkpoints (10 models × few configs) | ~5–10 GB |

**Recommendation:** download EyePACS *train* only (35,126 labelled images). That's enough for a patient-level 70/15/15 split, and it fits on disk if we preprocess and discard raw images part by part.

## Leakage controls planned
- **EyePACS:** group by patient ID (`{id}_left/right`). Both eyes always go in the same split.
- **APTOS:** no patient IDs. Group by perceptual-hash (pHash + dHash) near-duplicate clusters, because APTOS contains known duplicate images, some with conflicting labels. Conflicting-label clusters are reported, not silently dropped.
- **Messidor-2:** group by exam (two images per patient).
- **IDRiD:** official split; check pHash duplicates across train/test.
- All splits are stratified by grade within groups, seed 42, and saved to `data/splits/*.csv`.

## Duplicate audit (2026-11-19)

Method: green-channel high-pass (vessel/lesion structure) correlation on 96 px thumbnails, restricted to the central 80 % of the retina disc, threshold 0.99.
Perceptual 64-bit hashes were tried first and rejected: all fundus images look alike at 8×8, so they flagged
181 of 516 IDRiD images (mostly false positives). Calibration: genuine copies score exactly 1.000; the most
similar *distinct* pairs scored 0.945 (IDRiD) and 0.983 (Messidor-2 same-session images) with the first,
full-square measure. With the final interior-only measure they score at most 0.873 (APTOS), 0.875
(Messidor-2), 0.770 (IDRiD) and 0.988 (EyePACS).

| Dataset | Duplicate pairs | Finding | Action |
|---|---|---|---|
| IDRiD | 6 byte-identical pairs (MD5 equal) | 3 pairs carry **different grades** (IDRiD_021/391: 1 vs 2; 028/327: 4 vs 3; train_118 vs **test_064: 0 vs 3**, crossing the official train/test split) | Conflicting pairs inside train excluded (4 images); training copy of the test duplicate excluded (official test set untouched); 3 redundant same-label copies dropped |
| APTOS 2019 | 148 pairs in 131 clusters (270 images); all corr ≥ 0.998, identical original resolution | **43 pairs disagree on grade** (33 by 1 grade, 9 by 2, 1 by 3), confirming APTOS's known duplicate label noise | 79 images in 37 conflicting clusters excluded; 97 redundant same-label copies dropped; 3,486 images used |
| EyePACS | 0 at corr ≥ 0.99 (closest distinct pair 0.988: two washed-out, near-featureless images) | First pass (full square) produced a false 16-image cluster across 14 patients: the camera's rim notch dominated the correlation. Fixed by restricting the correlation to the central 80 % of the retina disc; re-running the fixed detector on IDRiD, Messidor-2 and APTOS gives **identical** pair sets, so their splits are unchanged | Nothing excluded |
| Messidor-2 | 4 exact pairs | Each pair is listed under two *different* exams in `messidor-2.csv`; labels agree | One copy dropped; the two exams merged into one leakage group |

Policy: (1) official test sets are never modified; (2) identical images with contradictory grades have no
consistent ground truth and are excluded (counted above, never silently); (3) same-label copies are reduced
to one; (4) all remaining copies share a leakage group so they cannot straddle splits.

## Prepared splits

| Dataset | Train | Val | Test | Notes |
|---|---|---|---|---|
| IDRiD | 324 | 81 | 103 (official) | val carved from official train; test Mild = 5 images |
| APTOS 2019 | 2,437 | 524 | 525 | duplicate-cluster grouped, stratified, seed 42 |
| EyePACS | 24,586 | 5,268 | 5,272 | patient-grouped (both eyes in one split, verified), stratified, seed 42 |
| Messidor-2 | 1,213 | 263 | 264 | exam-grouped, stratified, seed 42; primarily used as **external** test set |
