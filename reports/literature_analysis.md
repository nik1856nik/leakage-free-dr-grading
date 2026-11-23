# Literature Analysis — Automated DR Detection & Five-Class Grading

*Source: `main.pdf` (Team 10 literature survey, 15 papers). Generated 2026-11-19.*

## 0. How claims were verified

| Paper | Original full text read? | How |
|---|---|---|
| P6 Butt et al. 2022 (Diagnostics) | **Yes** | Europe PMC open-access XML (PMC9324358) |
| P7 Gulshan et al. 2016 (JAMA) | **Yes** (main text; supplement not read) | jamanetwork.com |
| P10 Zafar et al. 2025 (Front. Med.) | **Yes** | Europe PMC (PMC12000039) |
| P11 Yaqoob et al. 2021 (Sensors) | **Yes** | Europe PMC (PMC8200077) |
| P12 Papadopoulos et al. 2021 (Sci Rep) | **Yes** | Europe PMC (PMC8275626) |
| P4 Dixit & Jha 2025 | No: paywalled. Located as *Med. Eng. Phys.* 2025, doi:10.1016/j.medengphy.2025.104350 | Europe PMC metadata |
| P8 Abbasi et al. 2025 | No: paper not located. **Code repo verified**: `github.com/farhanamin/Diabetic-Retinopahty` (MATLAB: `agc.m`, `cdf.m`, `Mn_code.m`) | GitHub API |
| P14 Banerjee et al. 2025 | No: not located in indexed sources | web search |
| P15 Sushith et al. 2025 | No: only citation metadata found (attributed to a Nature Portfolio journal) | web search |
| P1–P3, P5, P9, P13 | Survey text only (reviews or not selected) | — |

Where the original paper wasn't read, everything below comes from the survey only and is marked **[survey-only]**.

## 1. Task-type audit (the most important finding)

Many of the high accuracies in the survey **are not five-class grading**. The table gives the actual task behind each headline number.

| Paper | Headline number in survey | Actual task | Classes | Verified? |
|---|---|---|---|---|
| P1 | 99.8% (Xception, APTOS) | Binary DR vs no-DR | 2 | survey-only |
| P3 | 99.89% (Gayathri) | Binary DR vs no-DR | 2 | survey-only |
| P4 | 88.44% APTOS, κ 0.88 | Multiclass. Survey says "multiclass"; assumed 5-class ICDR on APTOS. **Unverified.** | 5? | survey-only |
| P5 | exudate-area ranking | Feature ranking, 40 images; no classifier | — | survey-only |
| **P6** | "89.29% multiclass" | **3-class**: NDR (0), MDR (1–2), PDR (3–4), on an **undersampled balanced subset**. 89.29% is *average class accuracy*. The survey implies 5-class, which is **wrong**. | **3** | ✅ verified |
| **P7** | AUC 0.991 / 0.990 | **Referable DR** (moderate-or-worse DR **or** referable DME). Binary. Private 128k-image dev set; patent pending; no code or weights. | 2 | ✅ verified |
| P8 | 95.88% acc, AUC 0.98 | Likely binary DR detection on Messidor (sensitivity/specificity/AUC reported). Class count **unverified**. | 2? | survey-only |
| P9 | 75% acc, sens 30% | 5-class EyePACS | 5 | survey-only |
| **P10** | 99.06% / 90.75% | **Binary** detection (99.06%) + **4-class** severity among DR-positive images (90.75%). **Augmentation was done before the 80/20 random split** (Table 1: Severe 193 → 1,000; the test set is exactly 20% of the augmented pools), so the reported test sets very likely contain augmented copies of training images. Its **leakage-free 5-class baseline is 80.08% (κ 0.70)** (Table 3). | 2 + 4 | ✅ verified |
| **P11** | 96% M2 / 75.09% EyePACS | Messidor-2: **referable DME** (binary; 1,593 vs 151, so the majority baseline is 91.1%). EyePACS: **5-class** DR (majority baseline 25,810/35,126 = **73.5%**, so 75.09% is only +1.6 pp). 10-fold CV. Code: `github.com/kashifyy/ResRF`. | 2 / 5 | ✅ verified |
| **P12** | AUC 0.961 / 0.976 | **Referable DR** (≥ moderate). Binary. The 0.961 is from an **ensemble of 5 models**. IDRiD is used only for lesion localization. | 2 | ✅ verified |
| P13 | AUC 0.955 avg | Multi-task: grading + lesion seg + quality. Private 666k-image data. | 5 + seg | survey-only |
| P14 | "DenseNet-169 highest" | Review paper; the DenseNet-169 / ViT-B/16 numbers can't be traced to a specific experiment | ? | survey-only |
| P15 | 94–97% acc; per-class F1 > 0.99 | Claimed 5-class on Kaggle/EyePACS. **Implausible**: per-class F1 > 0.99 on EyePACS Mild/Severe is far beyond anything published, the EyePACS "104,000 images" count doesn't match any official release, and treating one image as an LSTM sequence is conceptually odd. | 5 (claimed) | survey-only, **flagged** |

**Implication.** No verified paper in this survey reaches ≥ 95% accuracy on honest five-class ICDR grading. Realistic published five-class APTOS results are about 80–88% accuracy with QWK about 0.85–0.93. For EyePACS, five-class accuracy is about 75–85%, and the majority-class baseline alone is 73.5%. The project's 95–98% target is realistic for **binary / referable DR** tasks, **not** for five-class grading on these datasets without leakage.

## 2. Per-paper extraction (deep-learning methods selected for reproduction)

### P4 — EfficientNetB3 + SE [survey-only]
- **Architecture:** EfficientNetB3 backbone → SE block (GAP → FC-ReLU → FC-sigmoid → channel rescale) → GAP → dropout 0.5 → softmax.
- **Preprocessing:** crop dark background, resize 256×256, BGR→RGB, Gaussian blur (Ben Graham-style).
- **Augmentation:** rotation, zoom, shifts, shear, horizontal flip.
- **Data/split:** APTOS 3,662, IDRiD 516, Messidor-2 1,748; 80/10/10 each.
- **Reported:** APTOS 88.44% acc, κ 0.88, sens 83%, spec 98%; IDRiD 81.74% (AUC 0.83); Messidor-2 73.28% (AUC 0.89).
- **Missing:** optimizer, LR, epochs, batch size, SE reduction ratio, where the SE block sits.
- **Limitations:** small-dataset overfitting; imbalance on IDRiD/Messidor-2.

### P6 — GoogleNet + ResNet18 features → SVM ✅
- **Architecture:** frozen ImageNet GoogleNet and ResNet-18; each **1000-d FC output** (pre-softmax logits) is concatenated into 2000-d → SVM / RF / RBF / NB (MATLAB).
- **Preprocessing:** resize to 224×224×3, normalize.
- **Tasks:** binary (0 vs 1–4) and 3-class (0 / 1–2 / 3–4) on a **balanced random undersample** (smallest class size).
- **Reported:** binary 97.80% (SVM, hybrid); 3-class 89.29% average class accuracy (NDR 96.66%, MDR 81.64%, PDR 90.07%).
- **Missing:** SVM kernel and C, exact split / CV protocol.
- **Limitations:** undersampling discards data; the 1000-d ImageNet-logit features are class-specific to ImageNet rather than generic.

### P7 — Inception-v3 (Gulshan) ✅
- **Architecture:** Inception-v3, ImageNet init, multiple binary sigmoid heads (moderate+, severe+, rDME, gradability); 10-model ensemble; early stopping on the tuning set.
- **Data:** 128,175 private images (EyePACS + 3 Indian hospitals), 3–7 grades per image. Validation: EyePACS-1 (9,963), Messidor-2 (1,748).
- **Reported:** rDR AUC 0.991 (EyePACS-1), 0.990 (Messidor-2); high-sensitivity point 97.5% / 93.4% (EyePACS-1).
- **Not reproducible exactly:** private, multiply-graded data; no code or weights; input resolution is in the supplement (not read).
- **Our plan:** Track B = referable DR (grade ≥ 2) on public EyePACS, external test on Messidor-2 (adjudicated grades). Track A = five-class adaptation (labelled as adapted).

### P8 — Adaptive enhancement + CNN [survey-only; code verified]
- **Pipeline:** luminance gain-matrix colour normalization (keeps r:g:b ratios) → adaptive gamma correction → quantile-weighted histogram equalization → fine-tuned CNN (best: VGG-S).
- **Excluded component:** cataract restoration uses **synthetic degradation + U-Net GAN generators**. This conflicts with our no-synthetic-images rule, so **only the AGC + histogram-equalization enhancement is reproduced** (ported from `agc.m` / `cdf.m`).
- **Reported:** 95.88% acc, sens 86.90%, spec 97.77%, AUC 0.982 (likely binary, Messidor).

### P10 — Lightweight two-stage CNN ✅
- **Architecture:** custom 37-layer CNN with one parallel branch, 227×227 input, ~1.3M parameters. The exact layer list is only in Fig. 2 (image), not text.
- **Training:** MATLAB, SGDM, momentum 0.9, LR 1e-3, batch 128, 100 epochs, zero-centre normalization.
- **Stage 2:** freeze, retrain the last 3 layers for 4-class severity.
- **Reported:** 99.06% binary, 90.75% 4-class (both affected by the augmentation-before-split leakage); clean 5-class 80.08%, κ 0.70.
- **Our plan:** adapted lightweight two-stage CNN (≈1–2M params). Augmentation happens **only on the fly on training images**. Stage outputs are combined into a five-class prediction.

### P11 — ResNet50 features → Random Forest ✅
- **Architecture:** ResNet-50 fine-tuned for 20 epochs → 2048-d GAP features → RF (entropy, min_samples_leaf 1, min_samples_split 2, random_state 1; n_estimators not given in the extracted text). Dropout 0.2; 9:1 train/val for fine-tuning.
- **Evaluation:** 10-fold CV, all images (no quality filtering).
- **Reported:** Messidor-2 rDME 96.00%; EyePACS 5-class 75.09%.
- **Caveat:** both numbers are only slightly above majority-class baselines (91.1% / 73.5%), so accuracy alone overstates performance.

### P12 — Attention MIL + ResNet-18 ✅
- **Preprocessing:** detect retina disk, crop square, resize 512×512, subtract local colour average (Graham), zero the outer 5% of the disk.
- **MIL:** 64×64 patches (drop if < 50% retina); K = 50 random patches/train image; test uses a grid with 0.75 overlap; ResNet-18 → M = 128-d embedding; Ilse et al. attention pooling (2-layer FC, params V, w) → FC → rDR.
- **Training:** Adam, LR 3e-4, 60 epochs, BCE; ImageNet init; 3,000 Kaggle-train images held out for validation; trained on gradable images; best-val checkpoint.
- **Reported:** AUC 0.961 Kaggle test (5-model ensemble), 0.976 Messidor-2.
- **Our plan:** Track B reproduction (rDR = grade ≥ 2) on EyePACS → Messidor-2; Track A five-class adaptation.

### P14 — DenseNet-169 and ViT-B/16 [survey-only, review paper]
- Survey says DenseNet-169 had the highest accuracy and that ViT-B/16 captures long-range dependencies. The original experimental configuration couldn't be traced.
- **Our plan:** standard ImageNet-pretrained DenseNet-169 and ViT-B/16 fine-tuning baselines with our own documented configuration, **not** attributed to P14.

### P15 — Multi-scale CNN + LSTM + Attention (TAHDL) [survey-only, flagged]
- **Architecture (per survey):** 224×224, [0,1] scaling + CLAHE → three parallel conv paths (3×3, 5×5, 7×7) → concat + maxpool → 2-layer LSTM (256, 128) with softmax attention → FC → softmax.
- **Training:** Adam 1e-3, batch 32, dropout 0.5, L2 0.01, 50 epochs, 80/20 split.
- **Concerns:** results are implausible (see §1); the spatial-to-sequence mapping is unspecified.
- **Our plan:** implement as described. We'll treat the feature map rows as a sequence (documented assumption) and expect realistic, not reported, numbers.

## 3. Reviews and classical work (not reproduced)
- **P1, P2, P3:** reviews. Their main messages: there's a gap between reported and clinical performance, datasets are imbalanced and inconsistently graded, and compute cost matters.
- **P5:** exudate area is the most discriminative hand-crafted feature (40 images only). This motivates checking whether attention / Grad-CAM focuses on exudates.
- **P9:** 5-class EyePACS baseline (75% acc, 30% sensitivity). This is the historical reference point for "honest" five-class EyePACS.
- **P13:** DeepDR (private 666k images). A lesion-aware multi-task design and the strongest evidence for quality control plus lesion supervision. Not reproducible (private data).

## 4. Cross-cutting preprocessing found in the literature

| Technique | Papers |
|---|---|
| Retina crop / black-border removal | P4, P5, P12 |
| Ben Graham local-colour subtraction (Gaussian blur) | P4, P12 |
| CLAHE | P15 (P8 compares against it) |
| Adaptive gamma + quantile HE | P8 |
| Green channel | P5 |
| Zero-out outer retina rim | P12 |
| Zero-centre / [0,1] normalisation | P10, P15 |

## 5. Common limitations and future directions in the literature
1. Task conflation: binary, referable and five-class results get compared directly with each other.
2. Augmentation or duplicates leak across splits (P10 verified; APTOS has known near-duplicates).
3. Accuracy is reported without majority-class baselines (P11).
4. Severe imbalance: Mild and Severe are minority classes everywhere.
5. Little external validation. P7, P12 and P13 are the exceptions.
6. Interpretability: only P12 and P13 provide region-level evidence.
7. Compute and deployment: P10 is the only one that focuses on parameter count.
