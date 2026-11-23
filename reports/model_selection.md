# Model Selection — 10 Models

*Generated 2026-11-19 after verifying papers (see `literature_analysis.md`).*

All 10 candidates are kept. Verification changed **how** each is reproduced and labelled; no candidate was dropped. No model is called an "exact reproduction" unless its architecture, data and protocol all match the original.

## Reproduction status legend
- **R-faithful:** architecture and training follow the verified paper. Only the data differ (public data instead of private).
- **R-adapted:** core idea preserved; missing or ambiguous details filled in with documented choices.
- **Baseline:** standard architecture cited in the survey; the configuration is ours.

## Selection matrix

| # | Model | Paper | Original task (verified?) | Our Track A (5-class) | Our Track B (binary/rDR) | Status | Params (approx.) |
|---|---|---|---|---|---|---|---|
| 1 | EfficientNetB3 + SE | P4 | multiclass APTOS (unverified) | ✅ primary | — | R-adapted (training hyperparams missing) | 11M |
| 2 | Inception-v3 | P7 | **referable DR** ✅ | ✅ adapted | ✅ rDR, EyePACS → Messidor-2 | R-adapted (private data, no code) | 22M |
| 3 | DenseNet-169 | P14 | unclear (review) | ✅ | — | Baseline | 13M |
| 4 | ResNet50 → Random Forest | P11 | rDME (M2) + **5-class** (EyePACS) ✅ | ✅ | optional rDME on Messidor-2 | R-faithful (code exists) | 24M + RF |
| 5 | GoogleNet + ResNet18 → SVM | P6 | binary + **3-class** ✅ | ✅ adapted to 5-class | ✅ binary, + 3-class original replication | R-faithful for 2/3-class; adapted for 5-class | 18M frozen + SVM |
| 6 | Lightweight two-stage CNN | P10 | binary + 4-class ✅ (leaky protocol) | ✅ stage-combined 5-class | ✅ stage 1 binary | R-adapted (layer table only in figure); **leak-free protocol** | ~1.3M |
| 7 | Attention-MIL + ResNet-18 | P12 | **referable DR** ✅ | ✅ adapted | ✅ rDR, EyePACS → Messidor-2 | R-faithful (hyperparams verified) | 11M |
| 8 | Adaptive enhancement + CNN | P8 | binary? (unverified) | ✅ | ✅ | R-adapted: AGC + quantile-HE ported from authors' MATLAB; **GAN restoration excluded** (synthetic) | depends on backbone |
| 9 | Multi-scale CNN + LSTM + Attention | P15 | claimed 5-class (implausible) | ✅ | — | R-adapted (sequence mapping unspecified) | ~3–8M |
| 10 | ViT-B/16 | P14 | unclear (review) | ✅ | — | Baseline | 86M |

## Why each was selected (criteria the brief prioritises)

| Model | Feature extraction | Attention | Multi-scale | Imbalance handling | Efficiency | Interpretability | Generalisation evidence |
|---|---|---|---|---|---|---|---|
| EffB3+SE | strong compound-scaled CNN | channel SE | MBConv receptive fields | — (we add) | good (11M) | Grad-CAM | P4 tested 3 datasets |
| Inception-v3 | strong; multi-branch | — | ✔ inception modules | — | medium | Grad-CAM | **best external evidence** (P7) |
| DenseNet-169 | dense reuse of features, good for small lesions | — | partial | — | medium | Grad-CAM | widely reported |
| ResNet50+RF | deep features + non-parametric classifier | — | — | RF class weights | fast classifier | feature importance | P11 cross-dataset |
| GoogleNet+ResNet18+SVM | hybrid frozen features, no fine-tuning | — | ✔ (inception) | P6 undersampling | **very cheap to train** | low | — |
| Two-stage light CNN | custom | — | parallel branch | stage separation | **best (≈1.3M)** | Grad-CAM | P10 tested 3 datasets |
| Attention-MIL | patch-level, local lesions | **instance attention** | patch vs global | — | heavy at test (many patches) | **native heatmaps** | Kaggle → Messidor-2 0.976 |
| Enhancement + CNN | depends on backbone | — | — | — | preprocessing cost | — | targets illumination shift, **relevant to domain shift** |
| MS-CNN+LSTM+Attn | multi-scale (3/5/7) | sequence attention | ✔ explicit | — | medium | attention weights | unverified |
| ViT-B/16 | global self-attention | **self-attention** | patch tokens | — | heaviest (86M) | attention rollout | data-hungry; tests whether transformers help on small DR data |

Accuracy was **not** a selection criterion on its own. P10 and P15 have the highest reported numbers, and they were kept *despite* their protocol concerns because they add diversity (lightweight, sequence modelling). Their reproduced results are expected to come out much lower.

## Specific decisions
1. **P6:** we replicate the original **3-class + binary** tasks on APTOS (Track B) **and** run the same pipeline on 5-class ICDR (Track A). We also test full-data vs undersampled training.
2. **P10:** augmentation applied **only on the fly to training images after splitting**. We report what the original protocol would have produced only as an explicit leakage-demonstration ablation, if time allows.
3. **P8:** the backbone is VGG16-BN, the closest available pretrained relative of VGG-S, plus an ablation applying the P8 enhancement to the best Track A model.
4. **P7 / P12 Track B:** referable DR = ICDR grade ≥ 2. Public EyePACS/Messidor-2 grade labels don't support the DME part of the original definition, so this difference from P7 is documented.
5. **ViT-B/16:** expected to need 224 px (or 384 with gradient checkpointing) on an 8 GB GPU.
6. A **majority-class baseline** and a **"predict-0 vs rest" baseline** are reported next to every result.

## Tuning focus (decided 2026-11-19, after the APTOS baselines)

All 10 models keep their APTOS baseline, one-time APTOS test evaluation and APTOS → IDRiD external
evaluation. Hyperparameter tuning effort goes to the **top 3 by the validation selection score
0.5·(QWK + macro-F1)** on APTOS five-class:

| Rank | Model | Score | QWK | macro-F1 | Acc |
|---|---|---|---|---|---|
| 1 | ResNet50 CNN stage (P11) | 0.855 | 0.937 | 0.773 | 0.887 |
| 2 | DenseNet-169 (P14) | 0.849 | 0.933 | 0.764 | 0.876 |
| 3 | EfficientNetB3+SE (P4) | 0.831 | 0.926 | 0.737 | 0.861 |
| (4) | Inception-v3 (P7) | 0.825 | 0.917 | 0.732 | 0.870 |

Round 1 (`scripts/tune_queue.sh`) changes one factor at a time against each baseline: 512 px input
(same effective batch via gradient accumulation), the alternative preprocessing (raw ↔ Ben Graham),
focal loss (γ = 2), and sqrt class-balanced sampling (with unweighted CE). Round 2 combines the factors
that improved the validation score. Selection uses validation only; the test split is not read.

## Scope decision: best-model focus (2026-11-19, after tuning)

Once the first EyePACS run was underway, the project goal was narrowed to **producing the best
possible final model**. The full 10-model EyePACS queue and the dedicated Track B training runs were
therefore cancelled:

| Planned | Done | Reason |
|---|---|---|
| EyePACS five-class for all 10 models | EyePACS pretraining for the **top 3 only** (EffB3+SE 512 px, DenseNet-169 focal, ResNet50) | The other 7 were clearly weaker on APTOS (validation score 0.63–0.82 vs 0.83–0.86), so their EyePACS runs (~12 h) could not contribute to the best model |
| Track B: dedicated referable-DR models (P7 Inception-v3, P12 MIL, P6 binary/3-class) | Referable-DR performance **derived from the five-class models** (grade ≥ 2 and P(grade ≥ 2)) | Same reason; the five-class models already give rDR sensitivity/specificity/AUC |
| EyePACS → Messidor-2 for P7 / P12 | EyePACS → Messidor-2 for the 3 EyePACS-trained models | Uses the models that exist |

Then: fine-tune the top 3 on APTOS from their EyePACS weights (classifier and regressor heads), and
select an ensemble with grade thresholds on validation (`scripts/run_best.sh`, `scripts/best_model.py`).
These are scope limits, stated as such in `final_report.md`.
