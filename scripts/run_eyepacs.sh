#!/usr/bin/env bash
# Stage 5: EyePACS (24,586 train / 5,268 val images, patient-grouped). VALIDATION ONLY, no --final.
#  Track A: five-class, all 10 models. Top-3 use their APTOS-tuned settings (tuning rounds 1-2):
#    ResNet50 baseline · DenseNet-169 + focal loss · EffB3+SE at 512 px.
#  Track B: referable DR (grade >= 2): P7 Inception-v3, P12 attention-MIL, P6 hybrid SVM.
# EyePACS has 10x the images per epoch of APTOS, so the deep models get a 12-epoch cosine schedule
# with early stopping after 4 epochs without validation improvement (APTOS best epochs were 7-23
# at 1/10 of the updates per epoch).
cd "$(dirname "$0")/.."
mkdir -p experiments/logs
DS=eyepacs
E="--epochs 12 --patience 4"
run() {
  local name="$1"; shift
  echo "[$(date +%T)] START $name"
  if uv run python "$@" > "experiments/logs/$name.log" 2>&1; then
    echo "[$(date +%T)] DONE  $name  $(grep -E '"(qwk|macro_f1|roc_auc)"' "experiments/logs/$name.log" | head -3 | tr -d ' \n')"
  else
    echo "[$(date +%T)] FAIL  $name (see experiments/logs/$name.log)"
  fi
}
# ---- Track A: five-class
run effb3_se_$DS      scripts/train.py --model effb3_se     --dataset $DS $E --size 512 --batch_size 8 --accum 2
run densenet169_$DS   scripts/train.py --model densenet169  --dataset $DS $E --loss focal
run resnet50_cnn_$DS  scripts/train.py --model resnet50_rf  --dataset $DS $E
run resnet50_rf_$DS   scripts/train_features.py --model resnet50_rf --dataset $DS \
    --cnn_ckpt "checkpoints/resnet50_rf__${DS}__5class__raw_384_s42.pt"
run inception_v3_$DS  scripts/train.py --model inception_v3 --dataset $DS $E
run vit_b16_$DS       scripts/train.py --model vit_b16      --dataset $DS $E
run hybrid_svm_$DS    scripts/train_features.py --model hybrid_svm --dataset $DS --task 5class
run light2stage_$DS   scripts/run_two_stage.py --dataset $DS
run mscnn_lstm_$DS    scripts/train.py --model mscnn_lstm   --dataset $DS --epochs 30 --patience 6
run p8_vgg16_$DS      scripts/train.py --model p8_vgg16     --dataset $DS $E
run mil_resnet18_$DS  scripts/train.py --model mil_resnet18 --dataset $DS $E
echo "[$(date +%T)] TRACK A FINISHED"
# ---- Track B: referable DR (grade >= 2), the task P7 / P12 were published on
run rdr_inception_v3_$DS scripts/train.py --model inception_v3 --dataset $DS --task rdr $E
run rdr_mil_resnet18_$DS scripts/train.py --model mil_resnet18 --dataset $DS --task rdr $E
run rdr_hybrid_svm_$DS   scripts/train_features.py --model hybrid_svm --dataset $DS --task rdr
run p6_3class_svm_$DS    scripts/train_features.py --model hybrid_svm --dataset $DS --task 3class_p6 --undersample
echo "[$(date +%T)] QUEUE FINISHED"
