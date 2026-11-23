# Leakage-Free Diabetic Retinopathy Grading

Reproduction of 10 diabetic-retinopathy (DR) models from a 15-paper literature survey, plus a validation-selected
ensemble for **five-class ICDR severity grading**. Everything is in PyTorch and uses only genuine fundus photographs
with their original grades: no synthetic or GAN images and no generated labels.

Hyperparameters, ensemble members and grade thresholds were all fixed on validation data. Each test split was then
evaluated **exactly once**.

**Authors:** Akshai Krishna KP, Mohammed Azhar Sait, Karthik Krishna K, Rohit A.

| Read this | What it is |
|---|---|
| [`paper/main.pdf`](paper/main.pdf) | IEEE-format paper (14 pages); B/W print copy: [`paper/main(bw).pdf`](paper/main(bw).pdf) |
| [`reports/final_report.md`](reports/final_report.md) | Complete results, discussion and limitations |
| [`reports/replication_guide.md`](reports/replication_guide.md) | Step-by-step reproduction, with the reason for each decision |
| [`docs/literature_survey.pdf`](docs/literature_survey.pdf) | The 15-paper survey the models were taken from |

---

## Results

### Final ensemble: APTOS 2019, five classes

| Member | Initialisation | Input |
|---|---|---|
| EfficientNet-B3 + SE, classifier | EyePACS-pretrained, fine-tuned on APTOS | 512 px, Ben Graham |
| EfficientNet-B3 + SE, regressor | EyePACS-pretrained, fine-tuned on APTOS | 512 px, Ben Graham |
| Inception-v3, classifier | ImageNet → APTOS | 384 px, raw |
| DenseNet-169, classifier, focal loss | ImageNet → APTOS | 384 px, raw |

How a prediction is made:
- Each member gives a grade score, averaged over 4 flipped views.
- The four scores are averaged.
- The average is cut at 4 thresholds fitted on validation: 0.64 / 1.51 / 2.74 / 3.09.

The ensemble was chosen from 770 candidates by a repeated, cross-fitted validation score, S = ½(QWK + macro-F1).

| APTOS test (n = 525, evaluated once) | Value [95% bootstrap CI] |
|---|---|
| Accuracy | **0.846** [0.815–0.874] |
| Quadratic weighted kappa | **0.931** [0.913–0.947] |
| Macro-F1 | 0.683 [0.626–0.734] |
| Referable DR (grade ≥ 2) | sensitivity 0.961 · specificity 0.919 · AUC 0.984 |

### External datasets (never used for training or selection)

| Dataset | Accuracy | QWK | Referable-DR AUC (sens / spec) |
|---|---|---|---|
| IDRiD (508) | 0.593 | 0.819 | 0.976 (0.81 / 0.995) |
| Messidor-2 (1,740) | 0.661 | 0.725 | 0.948 (0.45 / 0.995) |

The APTOS ensemble is over-conservative on Messidor-2 (low sensitivity there). EyePACS-trained models transfer much
better: EfficientNet-B3+SE trained on EyePACS reaches AUC 0.967 on Messidor-2, with sensitivity 0.92 and
specificity 0.85.

For context:
- The best single model on the same test split is ResNet-50, at accuracy 0.861.
- The ensemble's QWK gain over it is +0.019, which is not significant (McNemar p = 0.38).

All 10 baselines are in [`reports/model_comparison.csv`](reports/model_comparison.csv). Full metrics are in
[`experiments/best_aptos/final_metrics.json`](experiments/best_aptos/final_metrics.json).

---

## Repository structure

```
.
├── configs/                 # per-paper model defaults, dataset notes, training protocol
│   ├── models.yaml
│   ├── datasets.yaml
│   └── training.yaml
├── src/                     # library code
│   ├── preprocessing/retina.py   # retina crop, Ben Graham, CLAHE
│   ├── datasets/            # dataset sources + FundusDataset
│   ├── models/zoo.py        # the 10 reproduced architectures
│   ├── training/engine.py   # training loop (AMP, CE / focal / regression heads)
│   ├── evaluation/          # metrics, bootstrap CIs, plots
│   └── ensemble/            # score averaging, thresholds
├── scripts/                 # entry points
│   ├── download_datasets.sh, prepare_data.py     # data download, 512 px cache, duplicate audit, grouped splits
│   ├── train.py, train_features.py, run_two_stage.py
│   ├── run_queue.sh, tune_queue.sh, tune_round2.sh, run_eyepacs.sh, run_best.sh
│   ├── best_model.py, best_report.py            # ensemble selection (validation) + one-time test report
│   ├── final_baselines.py, evaluate.py, compare_models.py
│   ├── predict.py           # grade new images with the packaged ensemble
│   └── model_status.py      # live progress views
├── data/
│   ├── splits/              # train/val/test assignment per image (IDs + original grades)
│   └── metadata/            # duplicate-audit clusters and pairs
├── experiments/             # per-run config, history, summary, validation predictions
│   └── best_aptos/          # selection.json, final_metrics.json, cached member predictions
├── results/plots/           # per-run, comparison and best-model figures
├── reports/                 # final report, replication guide, literature analysis, CSV tables
├── best_model/model_card.json   # ensemble definition (weights: see below)
├── paper/                   # LaTeX source, figures, compiled PDFs
├── docs/literature_survey.pdf
├── pyproject.toml, uv.lock  # uv-managed environment
└── README.md
```

These are **not in git**:
- `data/raw/` and `data/processed/`: the datasets are licensed by their providers.
- `checkpoints/` and `*.pt`: the weights are 44–88 MB each.

---

## Setup

Requires Python 3.12 and an NVIDIA GPU with CUDA 12.8. The work was done on an RTX 4060 Laptop with 8 GB.

```bash
git clone <this repo> && cd <repo>
uv sync                     # installs torch 2.11 (cu128), timm, scikit-learn, grad-cam, …
```

### Datasets
| Dataset | Source | Use |
|---|---|---|
| APTOS 2019 | Kaggle (`aptos2019-blindness-detection`) | train / val / test |
| EyePACS | Kaggle (`diabetic-retinopathy-detection`) | pretraining, cross-dataset |
| IDRiD | IEEE DataPort | external test |
| Messidor-2 | ADCIS (manual request) + Kaggle grades | external test |

Put your Kaggle API token in `~/.kaggle/`, then run `bash scripts/download_datasets.sh`. See
[`reports/dataset_status.md`](reports/dataset_status.md) for the manual steps.

### Model weights
Download the four ensemble checkpoints from the GitHub **Releases** page into `best_model/`:
- `ft__effb3_cls__aptos.pt`
- `ft__effb3_reg__aptos.pt`
- `inception_v3__aptos__5class__raw_384_s42.pt`
- `tune__densenet169__focal__aptos.pt`

## Grade new images
```bash
uv run python scripts/predict.py path/to/fundus.jpg more_images/ --csv predictions.csv
```
For each image the output gives:
- the grade (0–4) and its name;
- the continuous grade score;
- P(referable), with a REFER flag;
- the five class probabilities.

It applies the exact training pipeline: retina crop, each member's enhancement, and 4-flip TTA.

**Research use only. This is not a medical device.**

## Reproduce from scratch
```bash
uv run python scripts/prepare_data.py --datasets idrid messidor2 aptos eyepacs   # cache, duplicate audit, splits
bash scripts/run_queue.sh                                # 10 baselines (validation)
bash scripts/tune_queue.sh && bash scripts/tune_round2.sh   # top-3 tuning (validation)
bash scripts/run_eyepacs.sh                              # EyePACS pretraining
bash scripts/run_best.sh                                 # APTOS fine-tunes + ensemble selection (validation)
uv run python scripts/best_model.py --dataset aptos --final --external idrid messidor2   # ONE test evaluation
uv run python scripts/best_report.py --dataset aptos --external idrid messidor2          # CIs, plots
uv run python scripts/final_baselines.py                 # ONE test evaluation of the baselines
uv run python scripts/compare_models.py --split test     # comparison table + plots
```
The replication guide gives every command, including the feature-based RF/SVM models and the EyePACS →
Messidor-2 evaluation.

## Data protocol
- **Splits:** 70/15/15, stratified, seed 42, and grouped:
  - by patient for EyePACS;
  - by exam for Messidor-2;
  - by near-duplicate cluster for APTOS.

  IDRiD keeps its official test set.
- **Duplicate audit:** green-channel high-pass correlation inside the retina, at a threshold of 0.99.
  - Duplicates with conflicting grades were excluded (79 in APTOS, 4 in IDRiD).
  - Same-grade copies were collapsed to one.
- **No leakage:** test images were never used for training, tuning, model selection or threshold fitting.

## Citation
```bibtex
@misc{krishna2026leakagefree,
  title  = {Leakage-Free Reproduction and Validation-Selected Ensembling of Deep Learning Models for Five-Class Diabetic Retinopathy Grading},
  author = {Krishna KP, Akshai and Sait, Mohammed Azhar and Krishna K, Karthik and A, Rohit},
  year   = {2026},
  note   = {Preprint}
}
```

## Licence and data terms
The code is released under the licence in `LICENSE`, if one is added. The datasets and their grades remain under
their providers' terms:
- APTOS and EyePACS: Kaggle competition rules.
- IDRiD: CC BY 4.0.
- Messidor-2: ADCIS licence.
