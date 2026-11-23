# Replication Guide: what we did, why we did it, and how to repeat it

This guide walks through the whole project in the order it was actually done, from an empty machine to
the packaged best model. For every step it gives:
- **Why:** the reason the step exists or the decision taken;
- **Do:** the exact commands;
- **Check:** what you should see, so you know the step worked;
- **Pitfalls:** problems we hit and how they were solved.

Results and their interpretation are in `reports/final_report.md`; this file is about *process*.

> **Golden rules used throughout** (from the project brief):
> 1. Genuine fundus images and their original labels only. No synthetic, GAN or generated data.
> 2. The test split of each dataset is read **once**, after every choice is frozen on validation.
> 3. No leakage: grouped splits, augmentation only after splitting and only on training images.
> 4. Report everything honestly, including failures, and label adapted models as adapted.

---

## Contents
- [0. Hardware and time budget](#0-hardware-and-time-budget)
- [1. Environment setup (uv)](#1-environment-setup-uv)
- [2. Literature audit](#2-literature-audit)
- [3. Dataset access and download](#3-dataset-access-and-download)
- [4. Data preparation: cache, duplicate audit, splits](#4-data-preparation-cache-duplicate-audit-splits)
- [5. Implementation overview](#5-implementation-overview)
- [6. Smoke test](#6-smoke-test)
- [7. Stage A: APTOS baselines for all 10 models](#7-stage-a-aptos-baselines-for-all-10-models)
- [8. Stage B: tuning the top 3](#8-stage-b-tuning-the-top-3)
- [9. Stage C: scope decision, best-model focus](#9-stage-c-scope-decision-best-model-focus)
- [10. Stage D: EyePACS pretraining](#10-stage-d-eyepacs-pretraining)
- [11. Stage E: fine-tuning on APTOS (classifier and regressor)](#11-stage-e-fine-tuning-on-aptos-classifier-and-regressor)
- [12. Stage F: ensemble selection (validation only)](#12-stage-f-ensemble-selection-validation-only)
- [13. Stage G: one-time test and external evaluation](#13-stage-g-one-time-test-and-external-evaluation)
- [14. Stage H: baselines on test, cross-dataset, statistics, plots](#14-stage-h-baselines-on-test-cross-dataset-statistics-plots)
- [15. Stage I: packaging and inference](#15-stage-i-packaging-and-inference)
- [16. Monitoring tools](#16-monitoring-tools)
- [17. Troubleshooting: every problem we hit](#17-troubleshooting-every-problem-we-hit)
- [18. Expected results checklist](#18-expected-results-checklist)
- [19. Complete command sequence](#19-complete-command-sequence)

---

## 0. Hardware and time budget

| Item | What we used | Notes |
|---|---|---|
| GPU | NVIDIA RTX 4060 Laptop, 8 GB | Every model fits in 8 GB (peak 0.9–5.6 GB). Use the batch sizes in `configs/models.yaml` |
| CPU / RAM | 16 cores / 23 GB | 8 data-loader workers per run |
| Disk | ~45 GB free during download, ~10 GB at the end | EyePACS raw parts are 35 GB; they can be deleted after caching |
| OS | Linux (Ubuntu), zsh | |
| Power | AC power, **performance** power profile (`powerprofilesctl set performance`), sleep disabled | The laptop GPU throttled to 930–1545 MHz at 86–89 °C; cooling directly affects speed |

| Stage | Wall-clock |
|---|---|
| Downloads (APTOS 10 GB, EyePACS 35 GB) | ~3 h, network-dependent |
| Data preparation (EyePACS 35k images) | ~35 min |
| APTOS baselines (10 models) | 2 h 33 m |
| Tuning rounds 1 and 2 | ~4 h 20 m |
| EyePACS pretraining (3 models) | 4 h 18 m |
| APTOS fine-tunes (6 runs) | 57 m |
| Selection, test evaluation, reports | ~30 min |

---

## 1. Environment setup (uv)

**Why.** `uv` pins the Python version and every package (with `uv.lock`), so the environment can be
rebuilt exactly. PyTorch comes from the CUDA 12.8 wheel index declared in `pyproject.toml`.

**Do:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # if uv is not installed
cd DR_Detection
uv sync                                                 # Python 3.12 + locked dependencies
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Check.** It prints `2.11.0+cu128 True`. Key versions: timm 1.0.31, albumentations 2.0.8, scikit-learn,
opencv-python-headless, grad-cam.

**Rules.**
- Always run Python through `uv run python ...`.
- Add packages with `uv add <pkg>`, never `pip install`.
- `pyproject.toml` sets `[tool.uv] package = false` (this is a project, not a library) and sources torch
  and torchvision from the explicit `pytorch-cu128` index.

---

## 2. Literature audit

**Why.** The brief's 95–98% accuracy target came from the survey's headline numbers. Before
reproducing anything, we had to know what task each number measured. If a "95%" is binary
detection, comparing it to five-class grading is meaningless.

**Do.**
1. Read `main.pdf`.
2. For each selected paper, find and read the original wherever possible:
   - P6, P10, P11 and P12 via Europe PMC open access;
   - P7 via JAMA;
   - P8's MATLAB code on GitHub.
3. Record for each paper: task, classes, data, split, preprocessing, architecture, hyperparameters, and
   anything missing.

**Output.** `reports/literature_analysis.md` and `reports/model_selection.md`. The essential findings:
- **P6** reports 3-class, not 5-class.
- **P7 and P12** are referable-DR (binary) papers.
- **P10** augmented its data *before* splitting, so its test set contains copies of training images.
  Its own clean five-class result is 80.08%.
- **P11**'s numbers sit barely above majority-class baselines.
- **P15**'s results are implausible.
- **P4, P8, P14 and P15**'s full papers were unavailable, so their settings are partly ours and marked
  as such.

**Decisions taken here** (kept to the end):
- **Track A:** five-class grading for all 10 models. **Track B:** referable DR, defined as grade ≥ 2
  because public labels lack DME.
- **Fidelity labels:** every model is labelled *R-faithful*, *R-adapted* or *Baseline*, and every
  hyperparameter is marked `paper` or `ours` in `configs/models.yaml`.
- **P8's GAN restoration is excluded** (synthetic data). Only its AGC + histogram-equalisation
  enhancement is used, ported from the authors' `agc.m` and `cdf.m`.
- **P10 uses a leak-free protocol:** augmentation only on the fly, on training images, after the split.
- **The selection metric is fixed in advance** as 0.5·(QWK + macro-F1), not accuracy, because the
  classes are imbalanced (EyePACS is 73.5% No DR).

---

## 3. Dataset access and download

**Why these datasets.** They are the four public, genuinely labelled datasets the brief names:
- **APTOS 2019:** the main five-class benchmark;
- **EyePACS:** large and multi-site, used for pretraining and generalisation;
- **Messidor-2:** adjudicated grades, used as an external test;
- **IDRiD:** used as an external test.

### 3.1 Credentials (done by the user, never shared in chat)
1. **Kaggle:**
   - Go to *Settings → API → Create New Token* and save the token to `~/.kaggle/access_token`.
   - Accept the competition rules ("Join competition") for *aptos2019-blindness-detection* and
     *diabetic-retinopathy-detection*.
2. **Messidor-2 images:** fill in the ADCIS licence form at
   https://www.adcis.net/en/third-party/messidor2/. Download the 4-part zip and `messidor-2.csv` into
   `data/raw/messidor2/`, then extract with 7-Zip (`7z x` on the first part).
3. **IDRiD:** download "B. Disease Grading" from https://idrid.grand-challenge.org/. We symlinked it:
   `data/raw/idrid -> ~/Desktop/idrid`.

**Why the brief's "stop at authentication" rule applied:** Messidor-2 can't be fetched automatically
(it needs a licence form), so we stopped and wrote instructions (`reports/dataset_status.md`), and the
user supplied the files.

### 3.2 Download
```bash
bash scripts/download_datasets.sh >> data/raw/download.log 2>&1
uv run python scripts/download_status.py --watch       # progress bars and ETA in a second terminal
```

**What it does:**
- Fetches the Messidor-2 adjudicated grades (`google-brain/messidor2-dr-grades`).
- Fetches the EyePACS labels.
- Fetches the full APTOS archive, verifies it with `unzip -t`, extracts **only** `train.csv` and
  `train_images/` (the test labels are private), then deletes the zip.
- Fetches the 5 EyePACS train parts (4 × 8.39 GB + 1.43 GB).

**Why the official EyePACS archive and not a pre-resized mirror:**
- **Provenance:** we know exactly which images and labels we have.
- **Control:** we do the resizing ourselves.

**Pitfalls** (all handled in the script; see §17):
- **Kaggle CLI version:** the old CLI can't use `access_token`, so the script runs Kaggle CLI ≥ 1.8 via
  `uvx --python 3.12 --from 'kaggle>=1.8' --with six kaggle`, wrapped in a shell function.
- **Silent hangs:** the CLI has no network timeout. A **watchdog** restarts any download whose file
  hasn't grown for 120 s; Kaggle resumes from `.kaggle-partial`.
- **Wrapped EyePACS parts:** Kaggle serves each part wrapped as `train.zip.00N.zip`. The watchdog must
  watch the wrapper, which is then unwrapped part by part.

**Check:**
- `data/raw/aptos2019/train_images` has 3,662 PNGs.
- `data/raw/eyepacs/` has the 5 parts plus `trainLabels.csv`.
- `data/raw/messidor2/IMAGES` has 1,748 files.
- IDRiD has 516 images.

Validate with `uv run python scripts/check_datasets.py`.

---

## 4. Data preparation: cache, duplicate audit, splits

```bash
uv run python scripts/prepare_data.py --datasets idrid messidor2 aptos eyepacs --workers 12
```

### 4.1 Cache (why: speed and identical geometry for every model)
- **Cropping:** each image is cropped to the retina's bounding box (threshold 10 plus a morphological
  opening to remove text specks) and padded to a square.
- **Resizing:** to 512 px, saved as JPEG quality 95 with 4:4:4 chroma (no colour subsampling) in
  `data/processed/<dataset>/512/`. Re-runs reuse existing cache files.
- **EyePACS:** read straight from the byte-split zip parts through `MultiPartFile`, a seekable view over
  the 5 parts, without concatenating them, which avoids another 35 GB on disk.
- **Image statistics:** quality stats (size, brightness, retina fraction) are recorded in
  `data/metadata/<ds>.csv`.

### 4.2 Duplicate audit (why: duplicates cause leakage, and conflicting copies have no valid label)

**Method** (`near_duplicate_clusters` in `prepare_data.py`):
1. Make a 96 px thumbnail, take the green channel minus its Gaussian blur, which keeps vessel and
   lesion structure.
2. Keep **only the central 80% of the disc radius**.
3. Standardise, then compute all-pairs correlation on the GPU (fp16 matrix multiply in chunks of
   4,096).
4. Link pairs with correlation ≥ **0.99** and merge them into clusters (union-find).

**Why this method:**
- *Perceptual hashes* failed: 181 of 516 IDRiD images were flagged, because every fundus image looks
  the same at 8×8.
- A *0.97 threshold* chained same-camera Messidor-2 images into a 20-image cluster.
- *Full-square correlation* (before the 80% restriction) linked 16 distinct EyePACS eyes, because the
  camera's rim notch is identical on every image.
- **Calibration** (final interior-only measure): genuine copies score 1.000; the closest distinct images
  score at most 0.873 (APTOS), 0.875 (Messidor-2), 0.770 (IDRiD) and 0.988 (EyePACS). Regenerate the
  distribution with `paper/make_figures_extra.py` (`fig_dup_corr`).
- **Always visually inspect a sample of flagged clusters,** which is how we caught the EyePACS false
  cluster.

**Policy**, decided before training and applied by the script:
1. Official test sets are never modified. If a test image duplicates a training image, the *training*
   copy is removed.
2. Copies with **contradictory grades** are excluded. Counts are logged in
   `data/metadata/<ds>_duplicates.csv` and `<ds>_duplicate_pairs.csv`.
3. Same-label copies are reduced to one.
4. All copies share one leakage group.

**Expected findings:**

| Dataset | Expected |
|---|---|
| IDRiD | 6 identical pairs; 3 with conflicting grades (one crossing train/test: train_118 = test_064) |
| APTOS | 148 pairs, 43 with conflicting grades → 79 excluded + 97 redundant → **3,486** used |
| Messidor-2 | 4 pairs listed under different exams |
| EyePACS | 0 pairs |

### 4.3 Splits (why: grouped, stratified, reproducible)
- **Method:** `StratifiedGroupKFold(n_splits = 20, seed 42)`, with 14 folds → train, 3 → val,
  3 → test (70/15/15).
- **Groups:**
  - EyePACS: patient ID (both eyes together);
  - Messidor-2: exam from `messidor-2.csv`, merged with duplicate clusters;
  - APTOS: duplicate cluster (no patient IDs exist);
  - IDRiD: keeps its official 103-image test set, with validation carved from its train set.
- **Safety check:** an assertion fails the script if any group crosses splits.

**Check:** `data/splits/<ds>_5class.csv` exists with the sizes below, and
`groups crossing splits: 0` for every dataset.

| Dataset | Train | Val | Test |
|---|---|---|---|
| APTOS | 2,437 | 524 | 525 |
| EyePACS | 24,586 | 5,268 | 5,272 |
| Messidor-2 | 1,213 | 263 | 264 |
| IDRiD | 324 | 81 | 103 |

**Storage.** After caching (2.3 GB for EyePACS), we verified all 35,126 cache files exist and deleted
the 35 GB of raw parts. `trainLabels.csv` is kept. `download_datasets.sh` skips EyePACS once the split
file exists, and `sources.available()` treats the cache as sufficient.

---

## 5. Implementation overview

| File | Role |
|---|---|
| `src/preprocessing/retina.py` | Crop and pad, resize, retina mask, `ben_graham`, `graham_rim` (P12), `clahe` (P15), `agc_p8` + `cdf_equalize_v_p8` (P8), `green` (P5); registry `ENHANCEMENTS` |
| `src/datasets/sources.py` | Raw-data loaders per dataset, `MultiPartFile`, `available()` |
| `src/datasets/fundus.py` | Tasks (`5class`, `binary` ≥ 1, `rdr` ≥ 2, `3class_p6`, `severity4`), `load_split`, augmentation (`standard`, `geometric`, `strong`), `FundusDataset`, class weights (N/n_k)^0.5 |
| `src/models/zoo.py` | The 10 architectures and `REGISTRY` (name → builder, size, enhancement, paper) |
| `src/training/engine.py` | `TrainConfig`, training loop (AMP, grad-clip 5, cosine/warm-up, early stopping, focal loss, √ sampler, regression head `head=reg`, channels-last), `predict` (with flip TTA and raw grade scores), `evaluate_split` |
| `src/evaluation/metrics.py` | All metrics, including per-class specificity and any-DR / referable views |
| `src/evaluation/plots.py` | Confusion matrix, ROC, training curves, metric bars, per-class recall |
| `src/ensemble/ensemble.py` | Soft voting, Dirichlet weight search, stacking utilities |
| `configs/models.yaml` | Per-model defaults; every setting marked paper or ours |

**Outputs of every run:**
- `experiments/<run>/`: `config.json`, `history.csv` (per epoch: losses, val metrics, score, lr, epoch
  time), `summary.json` (val, plus test when final), and `val_predictions.npz`
  (`y`, `prob`, `raw`, `uid`).
- `checkpoints/<run>.pt`: the best-validation weights plus config.

The default run name is `{model}__{dataset}__{task}__{enhancement}_{size}_s{seed}`.

---

## 6. Smoke test

**Why.** Before spending hours, confirm all 10 models train, fit in 8 GB and produce metrics.

**Do:** run each model for 1 epoch on IDRiD train/val (never test), then delete the smoke artefacts.
```bash
uv run python scripts/train.py --model effb3_se --dataset idrid --epochs 1 --run_name smoke
rm -rf experiments/smoke checkpoints/smoke.pt
```

**Check.** Each run finishes with a val line. Peak GPU memory is between 0.9 and 4.5 GB. Parameter
counts:

| Model | Params |
|---|---|
| VGG16-BN | 134 M |
| ViT-B/16 | 86 M |
| P10 CNN | 1.56 M |

---

## 7. Stage A: APTOS baselines for all 10 models

**Why.** Compare all approaches under identical data and protocol, using paper-derived settings, on
validation only.

```bash
bash scripts/run_queue.sh > experiments/queue_aptos.log 2>&1
uv run python scripts/model_status.py --watch           # live table
```

**What runs:**
- **End-to-end networks** via `train.py`.
- **P11:** fine-tune the ResNet50 CNN (20 epochs as in the paper), then `train_features.py` extracts
  2048-d features and fits a Random Forest (paper settings; n_estimators and class weight picked on
  validation).
- **P6:** `train_features.py --model hybrid_svm` (frozen GoogLeNet + ResNet18 FC outputs → SVM grid on
  validation).
- **P10:** `run_two_stage.py`:
  1. stage 1 is binary;
  2. stage 2 is `severity4`, using the frozen stage-1 body, a new head and LR 1e-2;
  3. the stages combine as P(0) = p₁(0), P(k) = p₁(1)·p₂(k−1).

**Check** (`model_status.py` table; validation scores, 0.5·(QWK + mF1)):

| Model | Score |
|---|---|
| ResNet50 | 0.855 |
| DenseNet-169 | 0.849 |
| EffB3+SE | 0.831 |
| Inception-v3 | 0.825 |
| ViT-B/16 | 0.819 |
| VGG16 | 0.815 |
| MIL | 0.792 |
| MS-CNN+LSTM | 0.635 |

**No `--final` here:** the test split stays untouched.

---

## 8. Stage B: tuning the top 3

**Why.**
- **Where the effort goes:** tuning effort goes where it can matter most. The top 3 by validation
  score were ResNet50, DenseNet-169 and EffB3+SE.
- **One factor at a time:** each variant changes one thing, so any change in score can be attributed
  to it.

**Round 1** (12 runs, about 3 h):
```bash
bash scripts/tune_queue.sh                                # experiments/tune_aptos.log
uv run python scripts/model_status.py --tuning --watch
```

The four factors per model:
1. **512 px:** same effective batch via `--batch_size 8 --accum 2` (DenseNet 6 × 2).
2. **The other preprocessing:** Ben Graham for ResNet and DenseNet, raw for EffB3.
3. **Focal loss** (γ = 2).
4. **√ class-balanced sampler + unweighted CE.**

**Round 2** (4 runs):
```bash
bash scripts/tune_round2.sh
```
1. **ResNet50 for 40 epochs (patience 10).** Its baseline's best epoch was its last (20/20), so we
   checked whether more epochs help. They don't: Δ = 0.000.
2. **DenseNet with focal + 512 px**, combining its two positive factors: +0.001.
3. **ResNet50 baseline re-run with seeds 1 and 2:** 0.850 and 0.850 vs 0.855. This **measures noise
   (±0.005)**, so smaller changes are not real.

**Outcome:** only EffB3 at 512 px (+0.013) is clearly better. All runs are named `tune__*` so they
don't mix with the baselines. Full table: final report Appendix B.

---

## 9. Stage C: scope decision, best-model focus

**What happened.** The plan was EyePACS five-class training for all 10 models (~15 h), then Track B. The
goal was then set to **"get the single best model"**. The other 7 models were clearly weaker on APTOS
(validation 0.63–0.82 vs 0.83–0.86), so their EyePACS runs could not contribute.

**Decision:**
- **EyePACS:** pretrain only the top 3.
- **Track B:** cancel the dedicated models; referable DR is derived from the five-class models.
- **The limits are documented** in `final_report.md` §13 and `model_selection.md`.

**Mechanics.** The EyePACS queue (`scripts/run_eyepacs.sh`) was stopped right after its ResNet50
stages. See §17 for the `pkill` pitfall.

---

## 10. Stage D: EyePACS pretraining

**Why.**
- **Data:** EyePACS has 10× more training images than APTOS, from many sites.
- **Recipe:** pretraining on it and fine-tuning on APTOS is a known strong recipe, and gives a
  model trained on many sites to test generalisation with.

**Do:** the first part of `scripts/run_eyepacs.sh`, or individually:
```bash
E="--dataset eyepacs --epochs 12 --patience 4"
uv run python scripts/train.py --model effb3_se    $E --size 512 --batch_size 8 --accum 2
uv run python scripts/train.py --model densenet169 $E --loss focal
uv run python scripts/train.py --model resnet50_rf $E
uv run python scripts/train_features.py --model resnet50_rf --dataset eyepacs \
    --cnn_ckpt checkpoints/resnet50_rf__eyepacs__5class__raw_384_s42.pt
```

**Why 12 epochs with patience 4:** one EyePACS epoch has 10× the updates of an APTOS epoch, and the
APTOS best epochs were 7–23.

**Check** (EyePACS validation):

| Model | QWK | Time |
|---|---|---|
| EffB3+SE | 0.815 | ~10 min/epoch, 2 h total |
| ResNet50 | 0.772 | 54 min |
| DenseNet-169 | 0.770 | 1 h 14 m |

---

## 11. Stage E: fine-tuning on APTOS (classifier and regressor)

**Why two heads.** DR grades are *ordered*. A classifier treats them as unrelated categories, which
suits accuracy and macro-F1. A **regressor** predicts one continuous grade, which makes errors smaller
in grade distance and suits QWK. Ensembles of both often combine the advantages.

**Regression head** (`--head reg` in `TrainConfig`):
- One output, SmoothL1 loss against the grade as a number.
- `init_from` keeps every weight whose shape matches, so only the 5-class head is dropped.
- For metrics, the score is converted to soft class probabilities (Gaussian, σ = 0.5).
- The raw score is saved in `.npz` as `raw`.

**Do:**
```bash
bash scripts/run_best.sh          # 6 fine-tunes + automatic selection -> experiments/best_aptos.log
uv run python scripts/model_status.py --best --watch
```

**Fine-tune settings:** `--epochs 15 --patience 6 --lr 1e-4 --warmup_epochs 1 --scheduler cosine`,
init from the EyePACS checkpoint. EffB3 runs at 512 px with batch 8 × 2.

**Check** (validation):
- Classifiers score 0.825–0.838.
- Regressors have QWK 0.922–0.938 but lower accuracy at fixed cuts.
- On validation, pretraining didn't beat the APTOS-only models. Selection decides anyway.

---

## 12. Stage F: ensemble selection (validation only)

**Why.** Single models of similar quality make different errors. Averaging them, plus fitting
grade cut-off thresholds, usually helps. But choosing among many ensembles and fitting thresholds on
the same validation set **over-fits**, so the procedure guards against that.

**Do:** run automatically at the end of `run_best.sh`. The **reported** selection was this rerun, which
adds Inception-v3:
```bash
E=checkpoints
uv run python scripts/best_model.py --dataset aptos --max_size 4 --members \
  "$E/ft__*__aptos.pt" $E/inception_v3__aptos__5class__raw_384_s42.pt \
  $E/resnet50_rf__aptos__5class__raw_384_s42.pt $E/tune__densenet169__focal__aptos.pt \
  $E/tune__effb3_se__res512__aptos.pt | tee experiments/best_aptos/selection_log.txt
```

**Algorithm:**
1. Each candidate predicts validation with 4-flip TTA (cached in `experiments/best_aptos/preds/`).
   Its grade score is the regressor output, or Σ k·p_k for classifiers.
2. Form all 1–4-member subsets (385), each under two rules (770 candidates):
   - `thr`: average the scores and cut them at 4 thresholds found by coordinate search over a 0.02
     grid, 3 sweeps;
   - `argmax`: average the probabilities and take the most likely class.
3. Scoring:
   - `thr` candidates by **3 × repeated 5-fold cross-fitting**: thresholds are fitted on 4/5 of
     validation and scored on the other 1/5;
   - `argmax` candidates directly, since nothing is fitted.
4. Save the winner, with thresholds refitted on all of validation, to `selection.json`.

**The fast scorer:** the selection score is computed from a 5×5 confusion matrix (0.05 ms per call). It
was verified to equal scikit-learn's QWK and macro-F1 exactly.

**Why the `argmax` rule exists:** on validation, threshold-cutting a classifier's expected grade lowered
its accuracy (0.868 → 0.845).

**Check:**
- **Selected:** `ft__effb3_cls + ft__effb3_reg + inception_v3 + tune__densenet169__focal`, rule `thr`,
  thresholds `[0.64, 1.51, 2.74, 3.09]`.
- **Score:** 0.8776 (cross-fitted validation: accuracy 0.903, QWK 0.953).

---

## 13. Stage G: one-time test and external evaluation

**Why.** This is the only unbiased estimate. Nothing may change after it.

```bash
uv run python scripts/best_model.py  --dataset aptos --final --external idrid messidor2
uv run python scripts/best_report.py --dataset aptos --external idrid messidor2
```

**What they produce:**
- **`--final`** reads `selection.json`, predicts APTOS test plus all of IDRiD and Messidor-2 (4-flip
  TTA), and writes `final_results.json`.
- **`best_report.py`** adds:
  - 2,000-sample bootstrap CIs, per-class tables, referable and any-DR views;
  - confusion and ROC plots;
  - a **reporting-only** table of every candidate single model on the same test split.

**Check** (APTOS test):

| Accuracy | QWK | Macro-F1 | Referable accuracy | Referable AUC |
|---|---|---|---|---|
| 0.846 | 0.931 | 0.683 | 0.935 | 0.984 |

IDRiD QWK 0.819; Messidor-2 QWK 0.725, with referable sensitivity 0.451.

**Rule:** we did **not** switch to ResNet50 after seeing that it had higher test accuracy. That would be
model selection on the test set.

---

## 14. Stage H: baselines on test, cross-dataset, statistics, plots

**Why.** The brief requires full test metrics per model and both cross-dataset directions. These runs
are pure evaluation of configurations that were already frozen.

```bash
# 1) 8 end-to-end baselines + P10 on APTOS test, plus APTOS -> IDRiD (all 508)
uv run python scripts/final_baselines.py
# 2) feature models: refit the validation-chosen classifier (fixed seeds), score test once
uv run python scripts/train_features.py --model resnet50_rf --dataset aptos \
    --cnn_ckpt checkpoints/resnet50_rf__aptos__5class__raw_384_s42.pt --final
uv run python scripts/train_features.py --model hybrid_svm --dataset aptos --task 5class --final
# 3) EyePACS -> Messidor-2 (all 1,740)
for c in effb3_se__eyepacs__5class__ben_graham_512_s42 densenet169__eyepacs__5class__raw_384_s42 \
         resnet50_rf__eyepacs__5class__raw_384_s42; do
  uv run python scripts/evaluate.py --ckpt checkpoints/$c.pt --dataset messidor2 --split all; done
# 4) tables and plots
uv run python scripts/compare_models.py --split test      # reports/model_comparison.csv
uv run python scripts/compare_models.py --split val       # reports/model_comparison_val.csv
```

**Check:**
- The feature models' **validation** numbers after the refit equal the original runs (RF 0.876 / 0.745;
  SVM 0.824 / 0.593). This proves the frozen configuration was reproduced.
- The test table appears in the final report §7.2.

**Post-processing done inline:**
- the ensemble row was appended to `model_comparison.csv`;
- `reports/cross_dataset.csv` was assembled from the `external_idrid_all` summaries and
  `eval_external_messidor2_all.json` files;
- bootstrap CIs were computed for every baseline (`experiments/best_aptos/baseline_test_ci95.json`);
- the paired bootstrap and McNemar test of ensemble vs ResNet50 gave ΔQWK +0.019, ΔAcc −0.016,
  p = 0.38;
- the ensemble's cost was measured: 56.3 M params, ≈174 ms/img (`ensemble_cost.json`);
- a Grad-CAM figure was made for the EffB3 classifier member
  (`results/plots/best_model/gradcam_effb3_test_examples.png`).

**Plot fix.** Comparison bar charts include **baseline runs only**, one bar per model. Including the
tuning and fine-tune runs produced overlapping bars, since they share model names. ResNet50's CNN and
RF runs are labelled separately.

---

## 15. Stage I: packaging and inference

**Why.** A "best model" must be usable outside the training code.

**`best_model/`** holds the 4 checkpoints plus `model_card.json`: members, heads, sizes, enhancements,
decision rule, thresholds, selection method, validation and test results.

**Predict:**
```bash
uv run python scripts/predict.py path/to/image.jpg some/folder/ --csv predictions.csv
```
For each image it outputs:
- the grade (0–4) and its name;
- the continuous grade score;
- P(referable) and REFER / no;
- the 5 class probabilities.

It repeats the exact pipeline: crop, 512 px, JPEG q95 round-trip (as in the cache), each member's
enhancement and resize, 4-flip TTA, averaging, thresholds.

**Check.** Running it on 40 raw APTOS test images reproduced the evaluated ensemble grades **40/40**
(maximum score difference 0.002, from fp16).

---

## 16. Monitoring tools

| Command | Shows |
|---|---|
| `uv run python scripts/download_status.py --watch` | Download progress bars, speed, ETA |
| `uv run python scripts/model_status.py --watch` | APTOS baseline queue: running model (epoch bar, ETA range, last and best metrics, score sparkline, stall warning) + table of finished models |
| `uv run python scripts/model_status.py --tuning --watch` | Tuning variants vs each baseline, with Δ score and ★ for the best |
| `uv run python scripts/model_status.py --dataset eyepacs --watch` | EyePACS queue |
| `uv run python scripts/model_status.py --best --watch` | Best-model pipeline: pretraining → fine-tunes → selection → test |
| `tail -f experiments/best_aptos.log`, `experiments/logs/<run>.log` | Raw logs |

---

## 17. Troubleshooting: every problem we hit

| Symptom | Cause | Fix |
|---|---|---|
| Kaggle 401/404 with `~/.kaggle/access_token` | Old kaggle CLI (1.7) doesn't support access tokens | Use Kaggle CLI ≥ 1.8 via `uvx --python 3.12 --from 'kaggle>=1.8' --with six kaggle` |
| `ModuleNotFoundError: six` | Missing dependency of the CLI | `--with six` (and `uv add six`) |
| `kg: command not found` in zsh when stored in a variable | zsh doesn't word-split variables | Use a shell function `kg() { uvx ... kaggle "$@"; }` |
| APTOS download stuck at 3.59 GB forever | Kaggle CLI has no network timeout | Watchdog in `download_datasets.sh` (restart after 120 s without growth; resumes) |
| EyePACS watchdog restarting endlessly or watching nothing | Parts arrive wrapped as `train.zip.00N.zip` | Watch `$f.zip`, then unwrap and delete the wrapper per part |
| Perceptual-hash duplicate detector flags a third of IDRiD | All fundus images look alike at 8×8 | High-pass correlation at threshold 0.99 |
| Leakage assertion failed on IDRiD | Duplicate pair across the official train/test split | Remove the **training** copy; never touch the official test set |
| 16-image "duplicate" cluster across EyePACS patients | Identical camera rim notch dominates the correlation | Correlate only inside the central 80% radius (other datasets unchanged) |
| YAML parse error in `models.yaml` | `->` and commas inside flow maps | Quote the `source` strings |
| P10 model 2.0 M params, too big | Initial widths | Widths reduced to give 1.56 M |
| Matplotlib "semibold" font warning | Not available | `fontweight="bold"` |
| `RuntimeError: Can't call numpy() on Tensor that requires grad` in `predict` | A helper was inserted between `@torch.no_grad()` and `predict`, so the decorator moved | Keep the decorator directly above `predict` |
| Stop command killed itself (exit 143/144) and never wrote its marker | `pkill -f "train.py --model inception_v3"` matched its *own* shell command line | Use `pgrep -f "[t]rain.py"` (the bracket trick), or kill by PID |
| Edited a running bash queue script | Bash reads scripts incrementally; `sed -i` replaces the file, so the running process keeps the old version | Never edit a running queue; rerun the step afterwards (we reran selection with Inception) |
| Training slower than expected; channels-last gave no speed-up | The laptop GPU was thermally throttled (930–1545 MHz at 86–89 °C) | Performance power profile, AC power, cooling; the speed changes don't alter results |
| `uv cache prune` hangs | Cache lock held by running uv processes | Run it when no uv job is active |
| Comparison bar charts with dozens of overlapping bars | Tuning and fine-tune runs share model labels | Plot baseline runs only; label ResNet50 CNN vs RF separately |
| AWS GPU instance refused ("vCPU limit") | New or free-tier accounts have a GPU quota of 0, and the free tier has no GPU | Stayed on the laptop |

---

## 18. Expected results checklist

Small differences (±0.005–0.01) are expected from cuDNN non-determinism. Larger deviations mean
something differs.

**Data:**
- [ ] APTOS used 3,486 (2,437 / 524 / 525).
- [ ] IDRiD external 508.
- [ ] Messidor-2 1,740.
- [ ] EyePACS 24,586 / 5,268 / 5,272.
- [ ] 0 groups crossing splits.

**Baselines and tuning (validation):**
- [ ] APTOS baselines: ResNet50 0.887 / 0.773 / 0.937 (acc / mF1 / QWK), DenseNet 0.876 / 0.764 / 0.933.
- [ ] Tuning: only EffB3 512 px clearly above noise (+0.013).
- [ ] Seed noise about ±0.005.
- [ ] EyePACS validation QWK: EffB3 0.815, ResNet50 0.772, DenseNet 0.770.

**Selection:**
- [ ] The 4-member ensemble, `thr` rule, thresholds about [0.64, 1.51, 2.74, 3.09], score 0.8776.

**Test:**
- [ ] APTOS test: acc 0.846, QWK 0.931, mF1 0.683, referable sens 0.961, spec 0.919, AUC 0.984.
- [ ] External: IDRiD QWK 0.819; Messidor-2 QWK 0.725.
- [ ] EyePACS → Messidor-2: EffB3 referable sens 0.923, AUC 0.967.
- [ ] `predict.py` reproduces the test grades 40/40.

---

## 19. Complete command sequence

```bash
# setup
uv sync
# data (credentials: ~/.kaggle/access_token; Messidor-2 + IDRiD placed manually)
bash scripts/download_datasets.sh >> data/raw/download.log 2>&1
uv run python scripts/check_datasets.py
uv run python scripts/prepare_data.py --datasets idrid messidor2 aptos eyepacs --workers 12
# Stage A: APTOS baselines (validation)
bash scripts/run_queue.sh > experiments/queue_aptos.log 2>&1
# Stage B: tuning (validation)
bash scripts/tune_queue.sh
bash scripts/tune_round2.sh
# Stage D: EyePACS pretraining of the top 3 (first 4 lines of run_eyepacs.sh; stop the queue after resnet50_rf_eyepacs)
bash scripts/run_eyepacs.sh > experiments/queue_eyepacs.log 2>&1
# Stage E+F: APTOS fine-tunes + automatic selection, then the reported selection rerun incl. Inception-v3
bash scripts/run_best.sh
E=checkpoints; uv run python scripts/best_model.py --dataset aptos --max_size 4 --members \
  "$E/ft__*__aptos.pt" $E/inception_v3__aptos__5class__raw_384_s42.pt \
  $E/resnet50_rf__aptos__5class__raw_384_s42.pt $E/tune__densenet169__focal__aptos.pt \
  $E/tune__effb3_se__res512__aptos.pt | tee experiments/best_aptos/selection_log.txt
# Stage G: ONE test evaluation + details
uv run python scripts/best_model.py  --dataset aptos --final --external idrid messidor2
uv run python scripts/best_report.py --dataset aptos --external idrid messidor2
# Stage H: baselines on test, cross-dataset, tables
uv run python scripts/final_baselines.py
uv run python scripts/train_features.py --model resnet50_rf --dataset aptos \
    --cnn_ckpt checkpoints/resnet50_rf__aptos__5class__raw_384_s42.pt --final
uv run python scripts/train_features.py --model hybrid_svm --dataset aptos --task 5class --final
for c in effb3_se__eyepacs__5class__ben_graham_512_s42 densenet169__eyepacs__5class__raw_384_s42 \
         resnet50_rf__eyepacs__5class__raw_384_s42; do
  uv run python scripts/evaluate.py --ckpt checkpoints/$c.pt --dataset messidor2 --split all; done
uv run python scripts/compare_models.py --split test
uv run python scripts/compare_models.py --split val
# Stage I: use the model
uv run python scripts/predict.py path/to/fundus.jpg --csv predictions.csv
```

`run_eyepacs.sh` still lists the full original 10-model EyePACS queue plus Track B. To reproduce
exactly what was done, stop it after `resnet50_rf_eyepacs`, or run only its first four `run` lines (as
in §10).
