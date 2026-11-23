# Automated Diabetic Retinopathy Detection and Five-Class Severity Grading Using Deep Learning

**Final report** · DR_Detection project · completed 2026-11-19

---

## Abstract

We reproduced, adapted and compared **10 deep-learning approaches** from the team's 15-paper literature
survey (P4, P6, P7, P8, P10, P11, P12, P14 ×2, P15) for diabetic retinopathy (DR) grading on the
five-class ICDR scale (0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative DR). We used four public
datasets of **genuine fundus photographs with their original grades**: APTOS 2019, EyePACS (Kaggle 2015),
Messidor-2 and IDRiD. No synthetic, GAN-generated or artificially labelled data were used.

Every split is grouped by patient, exam or duplicate cluster. **Every model, hyperparameter, ensemble
member and grade threshold was chosen on validation data only.** Each test split was evaluated once,
after the choices were frozen. Auditing the data exposed real duplicate images with *contradictory
grades* in IDRiD (3 pairs, one crossing the official train/test split) and in APTOS (43 pairs). These
were excluded and documented.

**Final model.** A four-member ensemble:
- EfficientNet-B3+SE pretrained on EyePACS and fine-tuned on APTOS, as a classifier and as a regressor;
- Inception-v3;
- DenseNet-169 trained with focal loss.

Each member makes four predictions per image (original plus three flipped copies, i.e. 4-flip test-time
augmentation, TTA). The ensemble averages their grade scores and cuts the average at four thresholds
fitted on validation. It was selected from 770 candidate ensembles by repeated 5-fold cross-fitted
validation score.

**Results on the untouched APTOS test set (n = 525, evaluated once, 95% bootstrap CIs):**

| Task | Result |
|---|---|
| Five-class grading | **accuracy 0.846** [0.815–0.874] · **QWK 0.931** [0.913–0.947] · macro-F1 0.683 [0.626–0.734] · AUC 0.960 |
| Referable DR (grade ≥ 2) | **accuracy 0.935** [0.914–0.956] · **sensitivity 0.961** [0.932–0.985] · specificity 0.919 [0.889–0.948] · **AUC 0.984** [0.976–0.992] |
| Any DR (grade ≥ 1) | sensitivity 0.996 · specificity 0.982 · AUC 0.9997 |

The ensemble has the highest test QWK of all 19 single models evaluated on the test split (best single
model 0.926), and the highest five-class AUC among the 11 baseline runs. Its differences from
the best single model (ResNet50, P11 configuration) are **not statistically significant**. The ensemble is
+0.019 QWK and −0.016 accuracy, with McNemar p = 0.38.

**Generalisation:**
- On IDRiD (never seen in training) it scores QWK 0.819 and referable AUC 0.976.
- On Messidor-2 it under-grades badly: referable sensitivity 0.45, although its AUC is 0.948.
- Models trained on the larger, multi-site **EyePACS transfer much better to Messidor-2**. EfficientNet-B3+SE
  reaches referable sensitivity 0.92, specificity 0.85, AUC 0.967.

**Against the brief's targets:**
- The 95% (minimum) / 98% (preferred) accuracy target was **not reached** for five-class grading (84.6%).
- It was **nearly reached** for referable DR (93.5%, CI up to 95.6%).
- The literature audit shows that no verified paper in the survey achieves ≥ 95% on honest five-class
  grading. The higher numbers in the survey come from binary or referable tasks, or from protocols with
  leakage (verified for P10).

---

## Table of contents
1. [Objectives and constraints](#1-objectives-and-constraints)
2. [Literature audit: what the surveyed papers actually did](#2-literature-audit)
3. [Datasets, duplicate audit and splits](#3-datasets-duplicate-audit-and-splits)
4. [Preprocessing](#4-preprocessing)
5. [The 10 models](#5-the-10-models)
6. [Training and evaluation protocol](#6-training-and-evaluation-protocol)
7. [Experiments and results](#7-experiments-and-results)
8. [The final best model](#8-the-final-best-model)
9. [Cross-dataset generalisation](#9-cross-dataset-generalisation)
10. [Interpretability (Grad-CAM)](#10-interpretability-grad-cam)
11. [Computational cost](#11-computational-cost)
12. [Discussion](#12-discussion)
13. [Limitations and deviations from the original plan](#13-limitations-and-deviations-from-the-original-plan)
14. [Research-integrity statement](#14-research-integrity-statement)
15. [Conclusions and future work](#15-conclusions-and-future-work)
16. [Deliverables and file index](#16-deliverables-and-file-index)
- [Appendix A: hyperparameters of every model](#appendix-a-hyperparameters-of-every-model)
- [Appendix B: tuning runs (validation)](#appendix-b-tuning-runs-validation)
- [Appendix C: full per-class tables of the final model](#appendix-c-full-per-class-tables-of-the-final-model)
- [Appendix D: development history of the selection procedure](#appendix-d-development-history-of-the-selection-procedure)

---

## 1. Objectives and constraints

**Goal.** Detect diabetic retinopathy and grade its severity into the five ICDR classes from colour
fundus photographs, by reproducing, optimising and comparing deep-learning models from the literature
survey.

**Work requested by the brief:**
- **Models:** reproduce and optimise 10 selected models:
  - P4 EfficientNet-B3+SE;
  - P7 Inception-v3;
  - P14 DenseNet-169 and ViT-B/16;
  - P11 ResNet50+Random Forest;
  - P6 GoogLeNet+ResNet18+SVM;
  - P10 lightweight two-stage CNN;
  - P12 attention-MIL ResNet-18;
  - P8 adaptive enhancement + CNN;
  - P15 multi-scale CNN+LSTM+attention.
- **Two tasks:** **Track A**, five-class ICDR grading for all models, and **Track B**, binary /
  referable-DR detection for the papers published on that task (P7, P12, P6, P10 stage 1).
- **Metrics:**
  - accuracy, macro precision, recall and F1, and weighted F1;
  - per-class precision, recall, F1 and specificity;
  - sensitivity, specificity and ROC-AUC;
  - quadratic weighted kappa (QWK) and the confusion matrix;
  - inference time and parameter count.
- **Other work:** cross-dataset tests (EyePACS → Messidor-2, APTOS → IDRiD), ensembles, plots and reports.

**Hard constraints (all met, see §14):**
1. **Genuine data only:**
   - genuine retinal fundus images with authentic dataset annotations;
   - no synthetic or GAN images, artificial samples or artificially generated labels;
   - real-image augmentation during training is allowed.
2. **Test-set integrity:** never train on test images, never optimise on test results, and keep a
   completely untouched final test set.
3. **No manipulation:** no tampering with test data, no removal of difficult samples to inflate
   accuracy, and no data leakage.
4. **Honest reporting:** no fabricated results, and adapted implementations are never claimed as
   exact reproductions.
5. **Authentication blocks:** if authentication blocks a download, stop and give instructions.

**Accuracy target:** 98% preferred, 95% minimum. The brief states these are targets; actual results
must be reported honestly.

**Hardware and software.** All work ran on one laptop:
- NVIDIA RTX 4060 Laptop GPU with 8 GB, 16 CPU cores, 23 GB RAM.
- Python 3.12 managed with `uv`, PyTorch 2.11.0 + CUDA 12.8, timm 1.0.31, albumentations 2.0.8,
  scikit-learn.
- The GPU ran at 86–89 °C under sustained load and thermally throttled (SM clock 930–1545 MHz instead
  of ~1890 MHz). This affected run times, not results.

---

## 2. Literature audit

Before reproducing anything, we read the original papers wherever they were obtainable (P6, P7, P10,
P11, P12 in full; P8's authors' code). The main finding: **many headline numbers in the survey are not
five-class grading at all.** Details are in `reports/literature_analysis.md`.

| Paper | Survey headline | Actual task (verified where marked ✅) |
|---|---|---|
| P4 EffB3+SE | 88.44% APTOS, κ 0.88 | Multiclass APTOS; full text paywalled, class count unverified |
| P6 GoogLeNet+ResNet18+SVM | 89.29% "multiclass" | ✅ **3-class** (0 / 1–2 / 3–4) on an undersampled balanced subset; the number is *average class accuracy*. Binary 97.8% |
| P7 Inception-v3 | AUC 0.991 / 0.990 | ✅ **Referable DR** (moderate+ or referable DME), private 128k multiply-graded images, 10-network ensemble, no code |
| P8 enhancement + CNN | 95.88% | Probably binary (Messidor); paper not located; enhancement code verified. Includes a GAN restoration step, **excluded** here |
| P10 two-stage CNN | 99.06% / 90.75% | ✅ Binary + 4-class severity. **Augmentation done before the random 80/20 split**, so augmented copies of training images are in the test set (leakage). Its own leakage-free five-class result is **80.08%, κ 0.70** |
| P11 ResNet50 → RF | 96% / 75.09% | ✅ Messidor-2 is **referable DME** (majority baseline 91.1%); EyePACS five-class 75.09% against a 73.5% majority baseline |
| P12 attention-MIL | AUC 0.961 / 0.976 | ✅ **Referable DR**; the 0.961 is a 5-model ensemble |
| P14 DenseNet-169, ViT-B/16 | "highest accuracy" | Review paper; no traceable experiment, so we treat these as baselines with our own configuration |
| P15 MS-CNN+LSTM | 94–97%, per-class F1 > 0.99 | Claimed five-class EyePACS. **Implausible** (F1 > 0.99 on EyePACS Mild/Severe; a dataset size matching no release) |

**Consequence for the targets.** Honest five-class results in the literature are about 80–88% accuracy
(QWK ≈ 0.85–0.93) on APTOS and 75–85% on EyePACS, where predicting "No DR" for everything already gives
73.5%. A 95–98% accuracy is realistic only for binary or referable-DR tasks. The project therefore
reports both views (five-class and referable DR) for every result.

---

## 3. Datasets, duplicate audit and splits

### 3.1 Sources

| Dataset | Source | Images used | Labels |
|---|---|---|---|
| **APTOS 2019** | Kaggle competition (train set; test labels are private) | 3,662 labelled → **3,486** after duplicate policy | five-class ICDR |
| **EyePACS** | Kaggle "Diabetic Retinopathy Detection" 2015, official train archive (5 split zip parts, 35 GB) | **35,126** (17,563 patients × 2 eyes) | five-class ICDR |
| **Messidor-2** | Images from ADCIS (licence form, supplied by the user); adjudicated grades from Krause et al. 2018 (Kaggle `google-brain/messidor2-dr-grades`) | 1,748 → **1,740** (4 ungradable + 4 duplicate copies removed) | adjudicated ICDR, rDME, gradability |
| **IDRiD** (Disease Grading) | idrid.grand-challenge.org, CC-BY-4.0 | 516 → **508** (8 removed by duplicate policy; official test set untouched) | ICDR + DME risk |

Class distributions:

| Dataset | No DR | Mild | Moderate | Severe | PDR |
|---|---|---|---|---|---|
| APTOS (all 3,662) | 1,805 | 370 | 999 | 193 | 295 |
| EyePACS (35,126) | 25,810 | 2,443 | 5,292 | 873 | 708 |
| IDRiD train / test | 134 / 34 | 20 / 5 | 136 / 32 | 74 / 19 | 49 / 13 |
| Messidor-2 used (1,740) | 1,017 | 268 | 345 | 75 | 35 |

Every dataset is heavily imbalanced. **Mild and Severe are minority classes everywhere**, which is why
macro-F1 and per-class recall matter more than accuracy.

### 3.2 Duplicate audit

Public fundus datasets contain duplicated images. If they are left in, a copy can appear in both train
and test (leakage), and copies with *different* grades mean there is no consistent ground truth.

**Method** (after two rejected attempts):
- Compare green-channel high-pass "vessel and lesion structure" on 96 px thumbnails, using GPU
  correlation, **restricted to the central 80% radius of the retina disc**.
- Image pairs with correlation ≥ 0.99 are linked, and linked pairs are merged into clusters (union-find).

**Calibration.** True copies score 1.000. With the final interior-only measure, the most similar *distinct*
images score at most 0.873 (APTOS), 0.875 (Messidor-2), 0.770 (IDRiD) and 0.988 (EyePACS, two washed-out,
almost featureless images that were checked visually). The full-square first version had given 0.945
(IDRiD) and 0.983 (Messidor-2).

**Rejected attempts:**
1. **64-bit perceptual hashes** flagged 181 of 516 IDRiD images: at 8×8 resolution all fundus images
   look alike.
2. **Full-square correlation** chained 16 different EyePACS eyes from 14 patients into one false
   cluster. The camera stamps the same notch and edge pattern on every image's rim. Excluding the rim
   fixed this. Re-running the fixed detector on IDRiD, Messidor-2 and APTOS gave **exactly the same
   pairs** as before, so their splits did not change.

**Findings:**

| Dataset | Duplicates found | Problem | Action |
|---|---|---|---|
| IDRiD | 6 byte-identical pairs | 3 pairs carry **different grades** (1 vs 2; 4 vs 3; train_118 grade 0 = test_064 grade 3, crossing the official train/test split) | Conflicting pairs inside train excluded (4 images); the *training* copy of the test duplicate excluded (official test set untouched); 3 redundant same-label copies dropped |
| APTOS | 148 pairs in 131 clusters (270 images), corr ≥ 0.998 | **43 pairs disagree on grade** (33 by 1 grade, 9 by 2, 1 by 3) | 79 images in 37 conflicting clusters excluded; 97 redundant copies dropped → 3,486 used |
| Messidor-2 | 4 exact pairs | Each pair listed under two different exams | One copy dropped; the two exams merged into one group |
| EyePACS | none | — | — |

**Policy:**
1. Official test sets are never modified.
2. Identical images with contradictory grades have no valid label and are excluded. They are counted,
   never removed silently.
3. Same-label copies are reduced to one.
4. All remaining copies share one leakage group.

This is not "removing difficult samples". The excluded images are *duplicates whose grades contradict
each other*, decided before any model was trained, and the counts are reported.

### 3.3 Splits (`data/splits/*.csv`)
- **Method:** 70 / 15 / 15 train / validation / test, via `StratifiedGroupKFold(20)` → 14 / 3 / 3 folds,
  stratified by grade, seed 42.
- **Groups:** patient (EyePACS, both eyes together), exam (Messidor-2), or duplicate cluster (APTOS; no
  patient IDs exist). An assertion in `prepare_data.py` verifies that no group crosses splits.
- **IDRiD** keeps its official test set (103), and validation is carved from the official train set.

| Dataset | Train | Val | Test |
|---|---|---|---|
| APTOS | 2,437 | 524 | 525 (No DR 270, Mild 51, Moderate 138, Severe 24, PDR 42) |
| EyePACS | 24,586 | 5,268 | 5,272 |
| Messidor-2 | 1,213 | 263 | 264 |
| IDRiD | 324 | 81 | 103 (official) |

Messidor-2 and IDRiD were used **in full** (all gradable, non-duplicate images) as **external** test
sets. No model was ever trained or tuned on them.

---

## 4. Preprocessing

Every image is first normalised geometrically and cached:
- crop to the retina's bounding box and pad to a square (black), so resizing doesn't distort the disc;
- resize to 512 px;
- save as JPEG quality 95 with full 4:4:4 colour.

Model-specific enhancements from the literature are then applied on the fly
(`src/preprocessing/retina.py`):

| Enhancement | From | Implementation |
|---|---|---|
| `raw` | — | none |
| `ben_graham` | P4, P12 (Graham 2015) | 4·img − 4·GaussianBlur(img, σ = size/30) + 128, background re-masked |
| `graham_rim` | P12 | `ben_graham`, then zero the outer 5% of the retina disc |
| `clahe` | P15 | CLAHE (clip 2, 8×8 tiles) on the LAB L channel |
| `agc_p8` | P8 | Port of the authors' `agc.m`: per-channel adaptive gamma, modified hist = 0.4·hist + 0.6·uniform, γ(l) = (1 − cdf(l))/0.7, followed by HSV-V CDF equalisation. Histograms are computed over retina pixels only (documented adaptation). **Bug note:** the authors' `cdf.m` actually implements the *inverse* of equalisation; we implement the equalisation the paper describes and document the discrepancy. P8's GAN cataract-restoration step is **excluded** (synthetic data). |

**Augmentation:** on the fly, training images only, after splitting.
- *Standard:* horizontal and vertical flips, rotation ±180°, scale 0.9–1.1, translation 3%,
  brightness/contrast ±0.15.
- *Geometric* (for MIL): the same without the photometric changes.
- Inputs are ImageNet-normalised.

---

## 5. The 10 models

**Fidelity labels:**
- **R-faithful:** architecture and protocol follow the verified paper; only the data differ.
- **R-adapted:** core idea kept; missing details filled with documented choices.
- **Baseline:** standard architecture cited in the survey, with our own configuration.

No model is claimed to be an exact reproduction.

| # | Model | Paper | Implementation (`src/models/zoo.py`) | Status | Params |
|---|---|---|---|---|---|
| 1 | EfficientNet-B3 + SE | P4 | timm `efficientnet_b3.ra2_in1k` → extra SE block (r = 16, our choice) on the final map → GAP → dropout 0.5 → FC | R-adapted (training settings not in survey) | 11.0 M |
| 2 | Inception-v3 | P7 | timm `inception_v3.tv_in1k`, ImageNet init, dropout 0.2 | R-adapted (public data, five-class head; P7 used private data and binary heads) | 21.8 M |
| 3 | DenseNet-169 | P14 | timm `densenet169.tv_in1k` | Baseline | 12.5 M |
| 4 | ResNet50 → Random Forest | P11 | ResNet-50 fine-tuned 20 epochs (paper) → 2048-d GAP features → RF (entropy, min_samples_leaf 1, min_samples_split 2, random_state 1, as in the paper; n_estimators and class weight tuned on validation) | R-faithful (code exists) | 23.5 M + RF |
| 5 | GoogLeNet + ResNet18 → SVM | P6 | Frozen torchvision ImageNet GoogLeNet and ResNet-18; 1000-d FC outputs concatenated (2000-d) → standardised → SVM (kernel and C tuned on validation) | R-faithful pipeline, five-class adaptation | 18.3 M (frozen) |
| 6 | Lightweight two-stage CNN | P10 | Custom conv/BN/ReLU trunk with one parallel branch (3×3 path and 1×1→5×5 path), 227 px. Stage 1 binary (DR vs no DR); stage 2 freezes the body and retrains the head for 4-class severity. Combined: P(0) = p₁(0), P(k) = p₁(1)·p₂(k−1) | R-adapted (layer table only in a figure; widths matched to the parameter budget) and **leakage-free protocol** | 1.56 M (paper 1.3 M) |
| 7 | Attention-MIL ResNet-18 | P12 | 512 px `graham_rim` input; 64×64 patches (≥ 50% retina); K = 50 random patches in training, grid with 0.75 overlap at test; ResNet-18 → M = 128 → additive attention (L = 32) → FC | R-faithful hyperparameters (patching, M, L, Adam 3e-4, 60 epochs); five-class head is an adaptation | 11.2 M |
| 8 | AGC enhancement + VGG16-BN | P8 | `agc_p8` enhancement + timm `vgg16_bn.tv_in1k` (closest available relative of P8's VGG-S) | R-adapted, GAN excluded | 134.3 M |
| 9 | Multi-scale CNN + LSTM + attention | P15 | Three parallel conv paths (3×3, 5×5, 7×7) → concat + maxpool → 7×7 map read in raster order as 49 time steps (our assumption: P15 doesn't say how an image becomes a sequence) → LSTM 256 → LSTM 128 → softmax attention → FC | R-adapted | 1.9 M |
| 10 | ViT-B/16 | P14 | timm `vit_base_patch16_224.augreg2_in21k_ft_in1k`, 224 px | Baseline | 85.8 M |

---

## 6. Training and evaluation protocol

**Training engine** (`src/training/engine.py`):
- Mixed precision (fp16 autocast + GradScaler), gradient clipping at 5.0, seed 42.
- Optimisers AdamW, Adam or SGD (Nesterov, momentum 0.9), per paper.
- Cosine schedule with optional warm-up, or a constant rate.
- Early stopping with per-model patience.
- From the tuning stage onward: channels-last memory layout for the plain CNNs (a speed-only change),
  and loss totals accumulated on the GPU without a per-step sync.

**Class imbalance:**
- Default: weighted cross-entropy with weights (N/n_k)^0.5, normalised to mean 1.
- Alternatives tested: focal loss (γ = 2), and square-root class-balanced sampling.

**Regression head** (added for the best-model stage):
- A single output trained with SmoothL1 against the grade as a number.
- For metrics that need probabilities, the score is converted to soft class probabilities (Gaussian,
  σ = 0.5, around the score).
- Grade decisions use thresholds fitted on validation.

**Selection criterion (fixed before any experiment):**
- Multiclass: validation score = **0.5·(QWK + macro-F1)**, which rewards both agreement with the
  ordered grades and minority-class performance.
- Binary: validation ROC-AUC.
- The best-validation checkpoint is kept.

**Metrics** (`src/evaluation/metrics.py`):
- Overall: accuracy, macro and weighted precision/recall/F1, macro specificity, ROC-AUC (one-vs-rest
  macro), QWK, the confusion matrix and the majority-class baseline.
- Per class: precision, recall, F1 and specificity.
- For five-class models, two derived screening views: **any DR** (grade ≥ 1) and **referable DR**
  (grade ≥ 2), each with sensitivity, specificity, AUC (from P(grade ≥ k)) and false-negative count.
- Final results get 2,000-sample bootstrap 95% CIs (`scripts/best_report.py`).

**Test-set policy:** a test split is read only by a `--final` / `final_*` step, once per frozen
configuration, after all selection on validation is complete.

---

## 7. Experiments and results

The project ran in stages:
1. APTOS baselines for all 10 models.
2. Tuning of the top 3.
3. EyePACS pretraining of the top 3.
4. Fine-tuning on APTOS.
5. Ensemble selection.
6. One-time test and external evaluation.

All numbers below up to §7.4 are **validation** unless labelled *test*.

### 7.1 APTOS five-class baselines (paper-default settings)

| Model (paper) | Val acc | Val mF1 | Val QWK | Val AUC | Mild / Severe / PDR recall | Best epoch | Train time |
|---|---|---|---|---|---|---|---|
| ResNet50 CNN stage (P11) | **0.887** | **0.773** | **0.937** | **0.967** | 0.75 / 0.52 / 0.72 | 20/20 | 10 m 54 s |
| DenseNet-169 (P14) | 0.876 | 0.764 | 0.933 | 0.959 | 0.61 / 0.64 / 0.75 | 23 | 21 m 38 s |
| ResNet50 + RF (P11) | 0.876 | 0.745 | 0.916 | 0.960 | 0.59 / 0.48 / 0.70 | — | RF fit 1 s |
| EfficientNet-B3+SE (P4) | 0.861 | 0.737 | 0.926 | 0.963 | 0.51 / 0.68 / 0.75 | 10 | 15 m 02 s |
| Inception-v3 (P7) | 0.870 | 0.732 | 0.917 | 0.958 | 0.59 / 0.44 / 0.70 | 7 | 7 m 47 s |
| ViT-B/16 (P14) | 0.855 | 0.727 | 0.910 | 0.939 | 0.41 / 0.64 / 0.65 | 20 | 13 m 57 s |
| AGC + VGG16 (P8) | 0.853 | 0.714 | 0.916 | 0.955 | 0.61 / 0.48 / 0.68 | 7 | 25 m 03 s |
| Attention-MIL (P12) | 0.855 | 0.695 | 0.890 | 0.956 | 0.71 / 0.44 / 0.38 | 18 | 39 m 42 s |
| GoogLeNet+ResNet18+SVM (P6) | 0.824 | 0.593 | 0.864 | 0.936 | 0.47 / 0.16 / 0.30 | — | SVM 11 s |
| MS-CNN+LSTM (P15) | 0.767 | 0.450 | 0.820 | 0.894 | 0.33 / 0.16 / 0.00 | 47 | 5 m 31 s |
| Light two-stage CNN (P10) | 0.773 | 0.422 | 0.754 | 0.900 | 0.43 / 0.00 / 0.00 | — | 7 m 18 s |

**P10 stages on validation:**
- **Stage 1 (DR vs no DR)** reaches accuracy 0.975 and AUC 0.998, so the leakage-free binary detector
  works well.
- **Stage 2 (4-class severity on the frozen body)** fails: accuracy 0.579, Severe/PDR recall 0. This
  makes the combined five-class model the weakest.

### 7.2 One-time APTOS test evaluation of the 10 baselines, plus APTOS → IDRiD

These configurations were frozen on validation before the test was read (`scripts/final_baselines.py`;
`train_features.py --final` for RF and SVM). The APTOS test majority-class baseline is 0.514.

| Model (paper) | Test acc [95% CI] | Test QWK [95% CI] | Test mF1 [95% CI] | Test AUC | Referable sens / spec / AUC | IDRiD acc / QWK |
|---|---|---|---|---|---|---|
| ResNet50 CNN (P11) | **0.861** [0.830–0.891] | 0.911 [0.879–0.939] | **0.743** [0.688–0.796] | 0.952 | 0.887 / 0.956 / 0.984 | 0.520 / 0.746 |
| ResNet50 + RF (P11) | 0.844 [0.813–0.874] | 0.902 [0.870–0.928] | 0.671 [0.611–0.726] | 0.949 | 0.946 / 0.928 / 0.983 | — |
| DenseNet-169 (P14) | 0.836 [0.804–0.867] | 0.904 [0.877–0.928] | 0.654 [0.598–0.712] | 0.944 | 0.902 / 0.944 / 0.982 | 0.520 / 0.744 |
| Inception-v3 (P7) | 0.829 [0.796–0.861] | 0.898 [0.872–0.922] | 0.687 [0.633–0.739] | 0.942 | 0.926 / 0.925 / 0.977 | 0.575 / **0.790** |
| EfficientNet-B3+SE (P4) | 0.821 [0.789–0.853] | 0.903 [0.877–0.926] | 0.675 [0.619–0.726] | 0.950 | 0.936 / 0.919 / 0.982 | 0.577 / 0.748 |
| GoogLeNet+ResNet18+SVM (P6) | 0.819 [0.785–0.851] | 0.850 [0.812–0.883] | 0.594 [0.531–0.651] | 0.922 | 0.912 / 0.941 / 0.972 | — |
| AGC + VGG16 (P8) | 0.815 [0.779–0.848] | 0.886 [0.857–0.913] | 0.646 [0.587–0.702] | 0.933 | 0.912 / 0.916 / 0.974 | **0.596** / 0.769 |
| Attention-MIL (P12) | 0.815 [0.781–0.848] | 0.874 [0.841–0.903] | 0.631 [0.567–0.689] | 0.940 | 0.848 / 0.950 / 0.980 | 0.470 / 0.688 |
| ViT-B/16 (P14) | 0.813 [0.779–0.846] | 0.894 [0.867–0.918] | 0.625 [0.566–0.683] | 0.907 | 0.936 / 0.919 / 0.974 | 0.400 / 0.640 |
| Light two-stage CNN (P10) | 0.750 [0.714–0.785] | 0.733 [0.688–0.776] | 0.398 [0.370–0.425] | 0.868 | 0.779 / 0.925 / 0.949 | 0.376 / 0.482 |
| MS-CNN+LSTM (P15) | 0.741 [0.701–0.777] | 0.788 [0.755–0.819] | 0.405 [0.360–0.450] | 0.881 | 0.882 / 0.882 / 0.957 | 0.453 / 0.512 |
| **Final ensemble (§8)** | 0.846 [0.815–0.874] | **0.931** [0.913–0.947] | 0.683 [0.626–0.734] | **0.960** | **0.961** / 0.919 / **0.984** | 0.593 / **0.819** |

**Observations:**
- **Test is below validation for every model,** by about 2–5 points of accuracy. The validation set is
  small (524) and was used for early stopping and selection, which makes its numbers slightly
  optimistic.
- **The 9 strongest single models overlap,** with test accuracy between 0.813 and 0.861 and overlapping
  confidence intervals. P10 and P15 are clearly last, by about 7–12 points of accuracy and 0.10–0.18 QWK.
- **Referable-DR AUC is 0.95–0.98 for every model,** so detecting *whether* a patient needs referral is
  much easier than assigning the exact grade.

### 7.3 Reproduced results compared with the papers' claims

| Paper | Claim | Same task? | Our result | Assessment |
|---|---|---|---|---|
| P4 EffB3+SE | APTOS 88.44%, κ 0.88 | Probably (five-class APTOS) | Test 0.821, QWK 0.903 (val 0.861) | Close to the claim; differences in split and protocol are unknown (paper paywalled) |
| P7 Inception-v3 | rDR AUC 0.991 / 0.990 | No (private multi-graded data, binary ensemble) | APTOS test rDR AUC 0.977 (from five-class) | Not comparable; high rDR AUC is confirmed on public data |
| P6 hybrid SVM | Binary 97.8%, 3-class 89.29% | No (we ran five-class) | Five-class test 0.819, QWK 0.850 | Original binary/3-class tasks not run (scope change, §13) |
| P8 AGC + CNN | 95.88% (likely binary) | No | Five-class test 0.815, QWK 0.886 | The enhancement gave no advantage over raw inputs for comparable backbones |
| P10 two-stage | 99.06% binary / 90.75% 4-class (leaky) | Partly | Stage 1 binary **val** acc 0.975 (leak-free); combined five-class test 0.750, κ_w 0.733 | Binary stage reproduces well; the five-class result is below the paper's own leak-free 80.08% (κ 0.70), and its high numbers are not reproducible without the leakage |
| P11 ResNet50 + RF | EyePACS five-class 75.09% | Yes (EyePACS five-class) | EyePACS RF **val** 0.841 (majority 0.735); APTOS test 0.844 | Reproduced and exceeded, but accuracy alone is misleading here. The RF never beat its own CNN's softmax head on the selection score (EyePACS val 0.664 vs 0.678; APTOS test acc 0.844 vs 0.861) |
| P12 attention-MIL | rDR AUC 0.961 (5-model ensemble) | No (we trained five-class) | APTOS test rDR AUC 0.980; IDRiD 0.959 | rDR discrimination comparable; five-class grading weaker than standard CNNs |
| P14 DenseNet-169 / ViT | no traceable numbers | — | 0.836 / 0.813 test | DenseNet strong; ViT-B/16 is the weakest pretrained model on this small dataset and transfers worst to IDRiD |
| P15 MS-CNN+LSTM | 94–97%, F1 > 0.99 | Claimed five-class | Test 0.741, QWK 0.788 | **Not reproducible**; consistent with the audit's implausibility flag |

### 7.4 Tuning of the top 3 (validation only)

The top 3 by validation score were ResNet50 (0.855), DenseNet-169 (0.849) and EffB3+SE (0.831), with
Inception-v3 close behind at 0.825. **Round 1** changed one factor at a time against each baseline:

| Factor | ResNet50 Δscore | DenseNet-169 Δscore | EffB3+SE Δscore |
|---|---|---|---|
| 512 px input (same effective batch via gradient accumulation) | −0.004 | +0.003 | **+0.013** |
| Other preprocessing (Ben Graham ↔ raw) | −0.016 | −0.017 | −0.010 (raw) |
| Focal loss γ = 2 | −0.002 | **+0.005** | −0.015 |
| √ class-balanced sampling + unweighted CE | −0.008 | −0.007 | −0.008 |

**Round 2:**
- **ResNet50, 40-epoch schedule:** Δ 0.000. Its baseline's best epoch was its last, so this was
  checked; a longer schedule didn't help.
- **DenseNet-169, focal + 512 px:** +0.001.
- **Run-to-run noise:** the ResNet50 baseline was re-run with seeds 1 and 2, scoring 0.850 and 0.850
  against 0.855. So the **noise is about ±0.005**.

**Conclusion:** only EffB3+SE at 512 px (+0.013) is clearly above noise. Focal loss and balanced
sampling trade overall score for rare-class recall: DenseNet with focal loss reaches Mild recall 0.82
(vs 0.61), and DenseNet with balanced sampling reaches Severe recall 0.76 (vs 0.64). APTOS five-class
levels off at about 0.88 accuracy and 0.93 QWK. Full table in Appendix B.

### 7.5 EyePACS pretraining of the top 3 (EyePACS validation, n = 5,268)

The aim was to learn from 10× more data, then fine-tune on APTOS. Settings: 12-epoch cosine schedule,
patience 4.

| Model | Acc | mF1 | QWK | AUC | Mild / Severe / PDR recall | Time |
|---|---|---|---|---|---|---|
| EffB3+SE, 512 px | 0.830 | **0.616** | **0.815** | **0.911** | 0.38 / 0.48 / 0.65 | 2 h 02 m |
| ResNet50 | 0.838 | 0.585 | 0.772 | 0.892 | 0.17 / 0.46 / 0.61 | 54 m |
| DenseNet-169 + focal | 0.822 | 0.579 | 0.770 | 0.880 | 0.20 / 0.60 / 0.64 | 1 h 14 m |
| ResNet50 + RF (P11 on EyePACS) | 0.841 | 0.561 | 0.767 | 0.892 | 0.12 / 0.34 / 0.51 | RF 43 s |

The EyePACS majority-class baseline is 0.735. EyePACS is harder than APTOS: its single grades are
noisier and image quality varies more. Mild DR (often a few microaneurysms) is nearly invisible at
384–512 px.

### 7.6 Fine-tuning on APTOS from EyePACS weights (APTOS validation)

Settings: learning rate 1e-4, 15 epochs, cosine with 1 warm-up epoch, patience 6. Each model was trained
with a classifier head and with a regression head.

| Run | Acc | mF1 | QWK | AUC | Score | vs APTOS-only |
|---|---|---|---|---|---|---|
| EffB3+SE, classifier | 0.876 | 0.748 | 0.929 | 0.960 | 0.838 | +0.007 vs EffB3 baseline (−0.006 vs EffB3 512 px APTOS-only) |
| EffB3+SE, regressor | 0.823 | 0.660 | 0.933 | 0.926 | 0.797* | |
| DenseNet-169, classifier | 0.868 | 0.736 | 0.926 | 0.959 | 0.831 | −0.018 |
| DenseNet-169, regressor | 0.849 | 0.696 | **0.938** | 0.929 | 0.817* | |
| ResNet50, classifier | 0.866 | 0.732 | 0.918 | 0.955 | 0.825 | −0.030 |
| ResNet50, regressor | 0.834 | 0.678 | 0.922 | 0.932 | 0.800* | |

\* Regressor scores here use fixed cuts at 0.5, 1.5, 2.5 and 3.5. Fitted thresholds raise them (e.g.
DenseNet regressor accuracy 0.849 → 0.861, cross-fitted).

**Classifier vs regressor:**
- **Regressors** have the best QWK (0.938 for DenseNet). They treat grades as ordered, so their errors
  are usually one grade off.
- **Classifiers** have higher exact accuracy and macro-F1.
- **On validation, EyePACS pretraining did not improve single models.** On the test set (observed
  after freezing, for reporting only) the pretrained regressors were the best single models by QWK
  (§8.5).

---

## 8. The final best model

### 8.1 Selection procedure (`scripts/best_model.py`, validation only)

**Candidates:** 10 checkpoints:
- the 6 EyePACS → APTOS fine-tunes;
- APTOS-only ResNet50 (baseline);
- DenseNet-169 + focal;
- EffB3+SE at 512 px;
- Inception-v3.

**Steps:**
1. Each candidate predicts every validation image with **4-flip TTA** (identity, horizontal, vertical,
   both). It outputs class probabilities and one continuous grade score: the regression output, or the
   expected grade Σ k·p_k for a classifier.
2. **All 385 subsets of 1–4 members** are formed, each with **two decision rules**, giving 770
   candidates:
   - `thr`: average the grade scores and cut them at 4 thresholds fitted by coordinate search to
     maximise the validation score;
   - `argmax`: average the probabilities and take the most likely grade.
3. Because the thresholds are fitted, a `thr` candidate is scored by **repeated cross-fitting**. Over 3
   random 5-fold partitions of validation, thresholds are fitted on 4/5 and applied to the held-out 1/5;
   the out-of-fold predictions are scored and the 3 repeats averaged. `argmax` candidates have nothing
   fitted and are scored directly.
4. The highest-scoring candidate is **frozen** in `experiments/best_aptos/selection.json`, with
   thresholds refitted on all of validation. **Only then** is the test split read, once
   (`best_model.py --final`).

### 8.2 Selected model

The selected candidate scored 0.8776. The top candidates were within about 0.001 of each other, so the
choice among them is a tie in practice.

| Member | Init | Input | Head |
|---|---|---|---|
| EfficientNet-B3+SE | EyePACS → APTOS fine-tune | 512 px, Ben Graham | classifier |
| EfficientNet-B3+SE | EyePACS → APTOS fine-tune | 512 px, Ben Graham | regressor |
| Inception-v3 | ImageNet → APTOS | 384 px, raw | classifier |
| DenseNet-169 (focal loss) | ImageNet → APTOS | 384 px, raw | classifier |

- **Decision:** `thr`, i.e. mean grade score cut at **[0.64, 1.51, 2.74, 3.09]**.
- **Size:** 56.3 M parameters in total.
- **Packaged in** `best_model/` with `model_card.json`.

### 8.3 Validation vs test

| | Accuracy | QWK | Macro-F1 | AUC | Referable sens / spec / AUC | Referable acc |
|---|---|---|---|---|---|---|
| Validation, cross-fitted | 0.903 | 0.953 | 0.802 | 0.967 | 0.975 / 0.935 / 0.985 | — |
| **Test, run once** (n = 525) | **0.846** | **0.931** | **0.683** | **0.960** | **0.961 / 0.919 / 0.984** | **0.935** |
| Test 95% CI | 0.815–0.874 | 0.913–0.947 | 0.626–0.734 | — | 0.932–0.985 / 0.889–0.948 / 0.976–0.992 | 0.914–0.956 |

The validation-to-test drop (−5.7 points of accuracy, −0.022 QWK, −0.12 macro-F1) is mainly
**selection optimism**: the best of 770 candidates was picked on 524 images. Cross-fitting removes the
bias from fitting thresholds, but not the bias from picking the maximum. Macro-F1 drops most because
Severe has only 24 test images, where one image is 4 points of recall. **The test numbers are the
result to report.**

### 8.4 Test-set behaviour per class

| True \ Predicted | No DR | Mild | Moderate | Severe | PDR | Recall |
|---|---|---|---|---|---|---|
| **No DR** (270) | **265** | 5 | 0 | 0 | 0 | 0.981 |
| **Mild** (51) | 1 | **24** | 26 | 0 | 0 | 0.471 |
| **Moderate** (138) | 0 | 8 | **119** | 9 | 2 | 0.862 |
| **Severe** (24) | 0 | 0 | 13 | **8** | 3 | 0.333 |
| **PDR** (42) | 0 | 0 | 10 | 4 | **28** | 0.667 |

| Class | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|
| No DR | 0.996 | 0.981 | 0.989 | 0.996 |
| Mild | 0.649 | 0.471 | 0.545 | 0.973 |
| Moderate | 0.708 | 0.862 | 0.778 | 0.873 |
| Severe | 0.381 | 0.333 | 0.356 | 0.974 |
| PDR | 0.848 | 0.667 | 0.747 | 0.990 |

Weighted F1 is 0.842, macro precision 0.717 and macro recall 0.663.

- **Almost every error is one grade off.** Only 12 of 81 errors are two grades off (10 PDR predicted
  as Moderate, 2 Moderate predicted as PDR), and none are further. This is why QWK is high while accuracy is moderate.
- **The model separates healthy from diseased almost perfectly:** any-DR sensitivity 0.996, specificity
  0.982, AUC 0.9997. On APTOS, No DR images also differ systematically in appearance, which makes this
  view optimistic.
- **The hard boundaries are Mild/Moderate and Severe/Moderate.** These are the boundaries where human
  graders also disagree most.
- **Referable DR:** 8 referable eyes were missed on test (false negatives). Every one of them was
  predicted Mild; none was called No DR.

### 8.5 Context: every candidate on the same test split (reporting only)

These rows were produced **after** the selection was frozen and were not used for any decision.
Single models use argmax with TTA (`experiments/best_aptos/test_context_single_models.csv`).

| Model | Test acc | Test QWK | Test mF1 | Test rDR AUC |
|---|---|---|---|---|
| **Best ensemble** | 0.846 | **0.931** | 0.683 | 0.984 |
| DenseNet-169 regressor (EyePACS → APTOS) | 0.819 | 0.926 | 0.641 | 0.982 |
| EffB3+SE regressor (EyePACS → APTOS) | 0.813 | 0.924 | 0.622 | 0.980 |
| EffB3+SE classifier (EyePACS → APTOS) | 0.834 | 0.919 | 0.691 | 0.983 |
| ResNet50 classifier (EyePACS → APTOS) | 0.844 | 0.912 | 0.703 | 0.982 |
| ResNet50 (APTOS only, TTA) | **0.857** | 0.911 | **0.730** | 0.986 |
| DenseNet-169 focal (APTOS only) | 0.832 | 0.910 | 0.685 | **0.986** |
| ResNet50 regressor (EyePACS → APTOS) | 0.804 | 0.910 | 0.620 | 0.977 |
| EffB3+SE 512 px (APTOS only) | 0.840 | 0.902 | 0.695 | 0.981 |
| Inception-v3 (APTOS only) | 0.840 | 0.900 | 0.703 | 0.978 |
| DenseNet-169 classifier (EyePACS → APTOS) | 0.825 | 0.894 | 0.672 | 0.982 |

**Paired comparison, ensemble vs best single model** (ResNet50 CNN without TTA, 2,000 paired bootstrap
resamples):

| Δ (ensemble − ResNet50) | Mean | 95% CI | P(Δ > 0) |
|---|---|---|---|
| Accuracy | −0.016 | [−0.048, +0.013] | 0.14 |
| QWK | +0.019 | [−0.001, +0.046] | 0.97 |
| Macro-F1 | −0.061 | [−0.124, +0.002] | — |

McNemar on per-image correctness: the ensemble alone was right on 28 images and ResNet50 alone on 36;
p = 0.38. **Neither model is significantly better.** The ensemble is (marginally) better on the ordinal
metric QWK, and ResNet50 on exact accuracy and macro-F1. The ensemble remains the final model because
it was chosen by the pre-declared validation rule; switching after seeing test results would be
selection on the test set.

---

## 9. Cross-dataset generalisation

All gradable, non-duplicate images of the external datasets were used, never trained or tuned on.
Thresholds and decision rules are carried over unchanged (`reports/cross_dataset.csv`).

### 9.1 APTOS → IDRiD (508 images)

| Model | Acc | QWK | mF1 | Referable sens / spec / AUC |
|---|---|---|---|---|
| **Best ensemble** | 0.593 [0.549–0.636] | **0.819** [0.787–0.849] | 0.462 | 0.808 / **0.995** / **0.976** |
| Inception-v3 | 0.575 | 0.790 | 0.445 | 0.802 / 0.974 / 0.966 |
| AGC + VGG16 | **0.596** | 0.769 | 0.469 | 0.818 / 0.963 / 0.966 |
| EffB3+SE | 0.577 | 0.748 | **0.481** | **0.874** / 0.826 / 0.942 |
| ResNet50 CNN | 0.520 | 0.746 | 0.430 | 0.811 / 0.947 / 0.954 |
| DenseNet-169 | 0.520 | 0.744 | 0.444 | 0.818 / 0.937 / 0.956 |
| Attention-MIL | 0.470 | 0.688 | 0.385 | 0.755 / 0.979 / 0.959 |
| ViT-B/16 | 0.400 | 0.640 | 0.332 | 0.780 / 0.963 / 0.919 |
| MS-CNN+LSTM | 0.453 | 0.512 | 0.259 | 0.582 / 1.000 / 0.928 |
| Light two-stage CNN | 0.376 | 0.482 | 0.245 | 0.594 / 0.989 / 0.908 |

IDRiD's majority baseline is 0.329. Five-class accuracy falls by about 25 points because IDRiD's graders
and cameras differ. Severe is the most affected class (ensemble recall 0.16): most IDRiD Severe images
are predicted Moderate. Referable-DR detection transfers well (AUC 0.98).

### 9.2 EyePACS → Messidor-2 (1,740 images, adjudicated grades)

| Model (trained on) | Acc | QWK | mF1 | AUC | Referable sens / spec / AUC |
|---|---|---|---|---|---|
| ResNet50 (EyePACS) | **0.763** | **0.784** | 0.650 | 0.904 | 0.884 / 0.907 / 0.963 |
| DenseNet-169 focal (EyePACS) | 0.742 | **0.785** | 0.615 | 0.876 | 0.868 / 0.906 / 0.953 |
| EffB3+SE 512 px (EyePACS) | 0.719 | 0.756 | **0.651** | **0.912** | **0.923** / 0.847 / **0.967** |
| *Best ensemble (APTOS)* | 0.661 | 0.725 | 0.434 | 0.880 | 0.451 / 0.995 / 0.948 |

Messidor-2's majority baseline is 0.585.

**The training data matter more than the architecture for generalisation.** On Messidor-2,
EyePACS-trained single models beat the APTOS ensemble on five-class accuracy, QWK and macro-F1 (five-class
AUC is mixed: 0.876–0.912 vs 0.880). They also catch about
twice as many referable cases (0.87–0.92 vs 0.45). The ensemble's referable AUC is still 0.948, so it
*ranks* images well. Its failure is **calibration**: it systematically under-grades Messidor-2's images,
calling Moderate "Mild" (187 of 345) and Mild "No DR" (207 of 268). Its thresholds were fitted to APTOS's
appearance.

EyePACS comes from many clinics and cameras, so its models are more robust. A deployment on a new site
would need site-specific threshold calibration, or an EyePACS-trained model.

---

## 10. Interpretability (Grad-CAM)

Figure: `results/plots/best_model/gradcam_effb3_test_examples.png`. It shows Grad-CAM on the
EffB3+SE classifier member for 2 correct test examples per class and 3 typical errors.

- **Moderate and Severe:** the heat-maps concentrate on clusters of hard exudates and haemorrhages, the
  clinically relevant lesions.
- **Mild:** the maps are diffuse and patchy. That matches the weakest class: microaneurysms are tiny at
  512 px.
- **PDR:** in one example the map centres on the optic disc. This fits neovascularisation of the disc,
  but it could also be a shortcut on disc appearance. A lesion-level check, e.g. against IDRiD lesion
  masks, would be needed to tell.
- **Errors** (Mild → Moderate, Severe → Moderate, PDR → Moderate): the model fires on lesions but grades
  their extent wrongly. That matches the "one grade off" error pattern.

Grad-CAM shows where the model looks, not why; it is qualitative evidence only.

---

## 11. Computational cost

| Model | Params | Inference (test, batch 16, no TTA) |
|---|---|---|
| Light two-stage CNN (P10) | 1.6 M | not timed separately (smallest model, 227 px) |
| MS-CNN+LSTM (P15) | 1.9 M | 2.4 ms/img |
| EffB3+SE (P4) | 11.0 M | 22.8 ms/img* |
| Attention-MIL (P12) | 11.2 M | 46.7 ms/img (hundreds of patches per image) |
| DenseNet-169 | 12.5 M | 5.6 ms/img |
| GoogLeNet+ResNet18 (P6) | 18.3 M frozen | 2.3 ms features + 1.3 ms SVM |
| Inception-v3 (P7) | 21.8 M | 3.8 ms/img |
| ResNet50 (P11) | 23.5 M | 4.3 ms/img (+ 0.08 ms RF) |
| ViT-B/16 | 85.8 M | 3.2 ms/img |
| VGG16-BN (P8) | 134.3 M | 10.6 ms/img |
| **Final ensemble** | **56.3 M** | **≈ 174 ms/img** at batch 1 with 4-flip TTA, including image loading (41 + 42 + 33 + 58 ms per member) |

\* The first model evaluated includes cuDNN warm-up.

**Wall-clock compute on the laptop:**

| Stage | Time |
|---|---|
| APTOS baselines | 2 h 33 m |
| Tuning round 1 | 3 h 10 m |
| Tuning round 2 | ≈ 1 h 10 m |
| EyePACS pretraining | 4 h 18 m |
| APTOS fine-tunes | 57 m |
| Selection, test and reports | under 30 min |

The ensemble is affordable for offline screening (about 6 images/s on a laptop GPU). If speed matters
more, a single ResNet50 runs at about 4 ms/image and is statistically as accurate (§8.5).

---

## 12. Discussion

**Was the target met?**
- **Five-class grading: no.** Accuracy is 84.6% against the 95% minimum.
- **Referable-DR detection: nearly.** Accuracy is 93.5%, sensitivity 96.1% and AUC 0.984. The CI's upper
  end (95.6%) includes the target.
- **Context from the audit (§2):** the surveyed five-class claims ≥ 95% are either different tasks or
  leaky protocols (P10 verified), or implausible (P15). The best leak-free published five-class APTOS
  results are in the same range as ours.
- **QWK 0.931** is "almost perfect agreement" on the usual kappa scale, and close to the agreement
  between human graders. The remaining errors are almost all one grade off at clinically ambiguous
  boundaries.

**Why five-class accuracy is limited:**
1. **Label noise:** APTOS contained 43 identical-image pairs graded differently, so the ground truth
   itself is noisy at the Mild/Moderate and Severe/Moderate boundaries.
2. **Tiny minority classes:** 24 Severe test images, where one image is 4% of recall.
3. **Resolution:** Mild DR is defined by microaneurysms only a few pixels wide at 512 px.
4. **Small data:** 2,437 APTOS training images.

**What helped and what didn't:**
- **Helped:**
  - ensembling diverse models with validation-fitted ordinal thresholds (best QWK and five-class AUC on
    test);
  - 512 px input for EfficientNet;
  - the regression head (best single-model QWK);
  - EyePACS pretraining, for generalisation (Messidor-2) and for test QWK.
- **Didn't help:**
  - the alternative preprocessing (Ben Graham on ResNet/DenseNet; raw on EffB3);
  - balanced sampling;
  - a longer schedule;
  - P8's AGC enhancement (no gain over raw inputs);
  - replacing the softmax head with RF or SVM classifiers (P11, P6), which never beat the network's own
    head on the selection score.
- **Mixed:** focal loss, which trades overall score for minority recall.

**Reproducibility of the surveyed claims:**
- **P10:** its binary stage is excellent without leakage (val 97.5%), but its 99% / 90.75% headline
  numbers depend on the augmentation leakage. Its five-class pipeline was the weakest model.
- **P15:** not reproducible; its result is consistent with the implausibility flag.
- **P11:** its EyePACS claim is matched, but its 75.09% sits just above a 73.5% majority baseline,
  which shows why accuracy alone is misleading on imbalanced data.

**Clinical reading.** On APTOS-like images the model is a capable screening tool:
- 96% of referable eyes caught at 92% specificity;
- the 8 missed referable eyes were all called Mild, not No DR.

Across sites it needs recalibration: Messidor-2 sensitivity is 45% at APTOS thresholds. It is a research
artefact, **not a medical device**.

---

## 13. Limitations and deviations from the original plan

| # | Planned | What was done | Why / effect |
|---|---|---|---|
| 1 | EyePACS five-class training for all 10 models | Only the top 3 (EffB3+SE, DenseNet-169, ResNet50, plus ResNet50 + RF features) | Scope narrowed to the best model; the other 7 were clearly weaker on APTOS. EyePACS-level comparison of all 10 is missing |
| 2 | Track B: dedicated referable-DR models (P7, P12, P6 binary and 3-class, P10 stage 1) | Referable DR **derived from the five-class models**; P10 stage 1 binary reported on validation only | Same scope decision. P6's original 3-class and binary tasks and P7/P12 dedicated binary training were not run |
| 3 | EyePACS → Messidor-2 for P7 / P12 | Done for the 3 EyePACS-trained models | Uses the models that existed |
| 4 | Feature models (P11 RF, P6 SVM) on IDRiD | Not evaluated externally | Their scripts have no external mode; the CNN version of P11 was evaluated |
| 5 | P4 / P8 / P14 / P15 full papers | Survey-only (paywalled or not found) | Their settings are partly ours (`configs/models.yaml` marks every setting as paper or ours) |
| 6 | P7 / P12 referable definitions include DME | Referable = grade ≥ 2 only | Public grade labels don't carry DME for APTOS/EyePACS |
| 7 | P8 GAN restoration | Excluded | Synthetic images are forbidden by the brief |
| 8 | EyePACS official test set (53k images) | Not downloaded | Disk space; the patient-grouped split of the train set (5,272 test images) is used instead (and wasn't needed for the final model) |
| 9 | Single seed | Seed 42 throughout; seed noise measured once (±0.005 score on ResNet50) | Differences below about 0.01 score are not meaningful |
| 10 | Validation-set size | 524 APTOS images for selecting among 770 ensembles | Causes optimistic validation numbers (−5.7 points of accuracy on test); a larger validation set or nested CV would reduce this |
| 11 | Determinism | cuDNN benchmark mode was on, so re-runs can differ slightly | Measured noise is within ±0.005 |
| 12 | Interpretability | Qualitative Grad-CAM on one member only | No lesion-level quantitative validation |
| 13 | Calibration | Thresholds fitted on APTOS | Under-grading on Messidor-2; site-specific calibration would be needed |

---

## 14. Research-integrity statement

| Constraint | How it was met |
|---|---|
| Genuine images and labels only | All images come from APTOS, EyePACS, Messidor-2 and IDRiD, with their official or adjudicated grades. No synthetic, GAN or generated images or labels. P8's GAN step was excluded. Augmentation is on-the-fly geometric/photometric changes of real training images only. |
| No training on test images | The splits are files fixed before training (`data/splits/`). Training loaders read only `split == "train"`. |
| No tuning on test results | Every choice used validation only: hyperparameters, tuning factors, early stopping, the scope decision, ensemble members, decision rule and thresholds. The selection was frozen in `selection.json` before `--final` was run. The 10 baselines' test runs happened after their configurations were final. Test-set context tables were produced after freezing and changed nothing. |
| Untouched test set | Official test sets (IDRiD) were never modified; for duplicates, the *training* copy was removed. No test image was dropped for being hard. |
| No leakage | Patient-, exam- and duplicate-grouped splits are verified by assertion. Augmentation happens after splitting. P10's leaky protocol was **not** used. |
| No fabricated results | Every number in this report is read from files in `experiments/` and `reports/`. Failures and negative results are reported (P10 stage 2, P15, EyePACS pretraining on validation, Messidor-2 sensitivity). |
| Adapted ≠ exact | Every model carries a fidelity label (§5), and every setting is marked "paper" or "ours" in `configs/models.yaml`. |
| Authentication | Kaggle downloads used the user's token. Messidor-2 needed the ADCIS licence form, so download instructions were given and the user supplied the images. |

**Transparency notes:**
- **The selection code changed during the project, on validation evidence, before any test read.**
  1. Selection was first tried on the APTOS-only checkpoints.
  2. The `argmax` decision rule was added after seeing that threshold cuts lowered classifier accuracy
     on validation.
  3. Inception-v3 was added to the candidates after the automatic run.
  4. The automatic run (without Inception) picked another ensemble, scoring 0.8766. The rerun's winner
     (0.8776) is reported.

  All of this happened before the test split was read (Appendix D).
- **One rerun of `prepare_data.py` for EyePACS** after fixing the duplicate detector. It changed
  EyePACS's splits before any EyePACS training.

---

## 15. Conclusions and future work

1. **Final model:** a validation-selected 4-model ensemble.
   - **Five-class (APTOS test):** QWK 0.931 and accuracy 0.846.
   - **Referable-DR:** AUC 0.984, sensitivity 0.961, specificity 0.919, accuracy 0.935.

   Packaged with a verified inference script.
2. **The 95% five-class target is not achievable honestly on these data.** The survey's higher numbers
   come from other tasks, leakage or implausible claims. Referable DR comes within the confidence
   interval of the target.
3. **Standard pretrained CNNs dominate:** ResNet50, DenseNet-169, EfficientNet-B3+SE and Inception-v3.
   The lightweight (P10) and sequence (P15) models are much weaker, and ViT-B/16 generalises worst.
4. **Generalisation is decided by the training data.** EyePACS-trained models transfer to Messidor-2
   far better than APTOS-trained ones.
5. **Data auditing matters.** Real contradictory duplicates existed, including one crossing IDRiD's
   official split.

**Future work:**
- **Messidor-2 generalisation:** train the ensemble's members on EyePACS (+ APTOS) jointly, and
  calibrate thresholds per site.
- **Validation design:** a larger validation set or nested cross-validation, to shrink selection
  optimism.
- **Mild DR:** higher resolution (768–1024 px) or lesion-aware supervision (e.g. IDRiD lesion masks) for
  microaneurysms.
- **Track B:** dedicated referable-DR training for P7 and P12, and P6's original tasks.
- **Uncertainty:** multiple seeds per configuration; uncertainty-based referral of borderline cases.
- **Interpretability:** quantitative checks against lesion annotations.

---

## 16. Deliverables and file index

| Path | Content |
|---|---|
| `best_model/` | The 4 member checkpoints and `model_card.json` (members, preprocessing, thresholds, results) |
| `scripts/predict.py` | Grade new images: `uv run python scripts/predict.py img.jpg folder/ --csv out.csv` (verified 40/40 identical to the evaluated predictions) |
| `reports/literature_analysis.md` | Paper-by-paper audit and task-type table |
| `reports/model_selection.md` | Selection matrix, fidelity labels, tuning and scope decisions |
| `reports/dataset_status.md` | Sources, validation, duplicate audit, splits |
| `reports/model_comparison.csv` | **Test** metrics of all 10 baselines and the ensemble (per class, referable, external IDRiD / Messidor-2 columns) |
| `reports/model_comparison_val.csv` | Validation metrics of every run (baselines, tuning, EyePACS, fine-tunes) |
| `reports/cross_dataset.csv` | APTOS → IDRiD and EyePACS → Messidor-2 results |
| `reports/replication_guide.md` | Step-by-step guide to reproduce everything, with the reasoning behind each step |
| `experiments/best_aptos/` | `selection.json`, `selection_log.txt` (ranked candidates), `final_results.json`, `final_metrics.json` (CIs, confusion matrices), `test_context_single_models.csv`, `baseline_test_ci95.json`, `ensemble_cost.json`, cached member predictions |
| `experiments/<run>/` | Per run: `config.json`, `history.csv`, `summary.json`, prediction `.npz` files |
| `results/plots/best_model/` | Confusion matrices (APTOS test, IDRiD, Messidor-2), five-class and referable ROC curves, ensemble-vs-single-model bars, Grad-CAM |
| `results/plots/comparison/` | Cross-model bar charts (validation and test) |
| `results/plots/<run>/` | Per-run confusion, ROC and training curves |
| `configs/` | `models.yaml` (every setting marked paper or ours), `datasets.yaml`, `training.yaml` |

---

## Appendix A: hyperparameters of every model

All models use seed 42, mixed precision, gradient clipping at 5, the selection score
0.5·(QWK + macro-F1), and weighted CE with weights (N/n_k)^0.5 unless stated otherwise.

| Model | Size | Enhancement | Optimiser | LR | Weight decay | Batch | Epochs / patience | Scheduler | Aug | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| EffB3+SE (P4) | 384 | ben_graham | AdamW | 3e-4 | 1e-4 | 16 | 30 / 8 | cosine | standard | SE r = 16, dropout 0.5 |
| Inception-v3 (P7) | 384 | raw | AdamW | 3e-4 | 1e-4 | 16 | 30 / 8 | cosine | standard | dropout 0.2 |
| DenseNet-169 (P14) | 384 | raw | AdamW | 3e-4 | 1e-4 | 12 | 30 / 8 | cosine | standard | |
| ResNet50 CNN (P11) | 384 | raw | AdamW | 3e-4 | 1e-4 | 16 | 20 / 20 | cosine | standard | 20 epochs as in paper; RF grid n_estimators {100, 300, 500} × class_weight {None, balanced_subsample}, selected 300 / balanced_subsample |
| GoogLeNet+ResNet18+SVM (P6) | 224 | raw | — | — | — | — | — | — | — | SVM grid kernel {linear, rbf} × C {0.01, 0.1, 1, 10} × class_weight {None, balanced}, selected rbf, C = 1, None |
| Light two-stage CNN (P10) | 227 | raw | SGD (Nesterov, 0.9) | 1e-3 (stage 2: 1e-2) | 1e-4 | 128 | 100 / 15 | none | standard | stage 2 freezes the body |
| Attention-MIL (P12) | 512 | graham_rim | Adam | 3e-4 | 0 | 8 | 60 / 12 | none | geometric | K = 50, 64 px patches, M = 128, L = 32 |
| AGC + VGG16-BN (P8) | 384 | agc_p8 | AdamW | 1e-4 | 1e-4 | 12 | 30 / 8 | cosine | standard | dropout 0.5 |
| MS-CNN+LSTM (P15) | 224 | clahe | Adam | 1e-3 | 1e-2 | 32 | 50 / 12 | cosine | standard | width 48, 49-step sequence |
| ViT-B/16 (P14) | 224 | raw | AdamW | 5e-5 | 0.05 | 16 | 30 / 8 | cosine, 2 warm-up epochs | standard | dropout 0.1 |
| EyePACS pretraining (top 3) | as tuned (EffB3 512, others 384) | as above | as above | as above | as above | EffB3 8×2 accumulation | 12 / 4 | cosine | standard | DenseNet with focal loss |
| APTOS fine-tunes (6) | as above | as above | as above | **1e-4** | as above | as above | 15 / 6 | cosine, 1 warm-up epoch | standard | init from EyePACS; regressor = SmoothL1, 1 output |

---

## Appendix B: tuning runs (validation)

Score = 0.5·(QWK + macro-F1).

| Run | Acc | mF1 | QWK | AUC | Mild / Severe / PDR recall | Score | Δ baseline |
|---|---|---|---|---|---|---|---|
| ResNet50 baseline | 0.887 | 0.773 | 0.937 | 0.967 | 0.75 / 0.52 / 0.72 | 0.855 | — |
| ResNet50 · 512 px | 0.884 | 0.771 | 0.932 | 0.963 | 0.73 / 0.52 / 0.78 | 0.851 | −0.004 |
| ResNet50 · Ben Graham | 0.874 | 0.768 | 0.910 | 0.963 | 0.76 / 0.64 / 0.68 | 0.839 | −0.016 |
| ResNet50 · focal | 0.887 | 0.774 | 0.932 | 0.964 | 0.78 / 0.64 / 0.62 | 0.853 | −0.002 |
| ResNet50 · √ sampling | 0.872 | 0.765 | 0.928 | 0.963 | 0.71 / 0.68 / 0.75 | 0.847 | −0.008 |
| ResNet50 · 40 epochs | 0.885 | 0.786 | 0.925 | 0.964 | 0.71 / 0.68 / 0.70 | 0.855 | 0.000 |
| ResNet50 · seed 1 | 0.885 | 0.774 | 0.926 | 0.965 | 0.69 / 0.56 / 0.72 | 0.850 | −0.005 |
| ResNet50 · seed 2 | 0.882 | 0.770 | 0.929 | 0.964 | 0.63 / 0.64 / 0.70 | 0.850 | −0.005 |
| DenseNet-169 baseline | 0.876 | 0.764 | 0.933 | 0.959 | 0.61 / 0.64 / 0.75 | 0.849 | — |
| DenseNet · 512 px | 0.880 | 0.770 | 0.933 | 0.967 | 0.75 / 0.60 / 0.68 | 0.851 | +0.003 |
| DenseNet · Ben Graham | 0.874 | 0.741 | 0.921 | 0.960 | 0.71 / 0.44 / 0.65 | 0.831 | −0.017 |
| DenseNet · focal | 0.880 | 0.776 | 0.930 | 0.961 | 0.82 / 0.56 / 0.70 | 0.853 | +0.005 |
| DenseNet · √ sampling | 0.855 | 0.754 | 0.929 | 0.962 | 0.76 / 0.76 / 0.75 | 0.841 | −0.007 |
| DenseNet · focal + 512 px | 0.874 | 0.777 | 0.923 | 0.964 | 0.75 / 0.68 / 0.78 | 0.850 | +0.001 |
| EffB3+SE baseline | 0.861 | 0.737 | 0.926 | 0.963 | 0.51 / 0.68 / 0.75 | 0.831 | — |
| EffB3 · 512 px | 0.876 | 0.766 | 0.921 | 0.964 | 0.71 / 0.68 / 0.68 | 0.844 | +0.013 |
| EffB3 · raw | 0.861 | 0.719 | 0.923 | 0.954 | 0.39 / 0.48 / 0.68 | 0.821 | −0.010 |
| EffB3 · focal | 0.857 | 0.720 | 0.913 | 0.958 | 0.65 / 0.68 / 0.50 | 0.816 | −0.015 |
| EffB3 · √ sampling | 0.866 | 0.725 | 0.920 | 0.959 | 0.55 / 0.48 / 0.60 | 0.823 | −0.008 |

---

## Appendix C: full per-class tables of the final model

**APTOS validation (cross-fitted predictions, n = 524)**

| True \ Predicted | No DR | Mild | Moderate | Severe | PDR |
|---|---|---|---|---|---|
| No DR (270) | 269 | 1 | 0 | 0 | 0 |
| Mild (51) | 0 | 30 | 20 | 0 | 1 |
| Moderate (138) | 0 | 5 | 127 | 6 | 0 |
| Severe (25) | 0 | 0 | 6 | 15 | 4 |
| PDR (40) | 0 | 0 | 7 | 1 | 32 |

Per class (precision / recall / F1):

| No DR | Mild | Moderate | Severe | PDR |
|---|---|---|---|---|
| 1.000 / 0.996 / 0.998 | 0.833 / 0.588 / 0.690 | 0.794 / 0.920 / 0.852 | 0.682 / 0.600 / 0.638 | 0.865 / 0.800 / 0.831 |

**IDRiD, all 508 (external)**

| True \ Predicted | No DR | Mild | Moderate | Severe | PDR | Recall |
|---|---|---|---|---|---|---|
| No DR (167) | 152 | 14 | 1 | 0 | 0 | 0.91 |
| Mild (23) | 16 | 7 | 0 | 0 | 0 | 0.30 |
| Moderate (165) | 11 | 45 | 106 | 3 | 0 | 0.64 |
| Severe (92) | 2 | 1 | 64 | 15 | 10 | 0.16 |
| PDR (61) | 0 | 2 | 16 | 22 | 21 | 0.34 |

Any-DR sensitivity 0.915, specificity 0.910, AUC 0.971. Referable false negatives: 61 of 318.

**Messidor-2, all 1,740 (external)**

| True \ Predicted | No DR | Mild | Moderate | Severe | PDR | Recall |
|---|---|---|---|---|---|---|
| No DR (1,017) | 967 | 44 | 5 | 0 | 1 | 0.95 |
| Mild (268) | 207 | 60 | 1 | 0 | 0 | 0.22 |
| Moderate (345) | 57 | 187 | 100 | 0 | 1 | 0.29 |
| Severe (75) | 0 | 5 | 58 | 7 | 5 | 0.09 |
| PDR (35) | 1 | 0 | 16 | 2 | 16 | 0.46 |

Any-DR sensitivity 0.634, specificity 0.951, AUC 0.895. Referable false negatives: 250 of 455. This shows
systematic under-grading of one grade at APTOS thresholds.

---

## Appendix D: development history of the selection procedure

Every step below used validation data only and happened before the APTOS test split was read.

1. **Pilot on 4 APTOS-only checkpoints**, with the `thr` rule and single-seed cross-fitting. Best: two
   DenseNet-169 variants, score 0.866.
2. **Faster exact scorer and repeated cross-fitting added.** A confusion-matrix implementation of
   0.5·(QWK + macro-F1) was checked to match scikit-learn exactly, and cross-fitting was averaged over
   3 partitions, so the pick isn't luck on one partition.
3. **`argmax` decision rule added.** Validation showed that cutting a classifier's expected grade at
   thresholds *lowered* its accuracy (DenseNet classifier 0.868 → 0.845), so each candidate is now
   scored under both rules.
4. **Automatic run after the fine-tunes, 9 candidates (Inception-v3 not yet included).** Selected
   EffB3 classifier (EyePACS) + ResNet50 regressor (EyePACS) + ResNet50 (APTOS) with `argmax`, score
   0.8766.
5. **Rerun with Inception-v3 added (10 candidates, 770 ensembles).** Selected the reported 4-member
   ensemble with `thr`, score 0.8776. The top three candidates were within 0.0011 of each other.
6. **Frozen**, then a single `--final` test evaluation, then the external evaluations.
