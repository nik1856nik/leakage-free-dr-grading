#!/usr/bin/env bash
# Stage 4a: baseline five-class runs on APTOS for all 10 models with paper-derived defaults
# (configs/models.yaml). VALIDATION ONLY - no --final, the test split is not touched.
# Each run logs to experiments/logs/<name>.log; failures do not stop the queue.
cd "$(dirname "$0")/.."
mkdir -p experiments/logs
DS=${DS:-aptos}
run() {
  local name="$1"; shift
  echo "[$(date +%T)] START $name"
  if uv run python "$@" > "experiments/logs/$name.log" 2>&1; then
    echo "[$(date +%T)] DONE  $name  $(grep -E '"(qwk|macro_f1)"' -A0 "experiments/logs/$name.log" | tr -d ' \n')"
  else
    echo "[$(date +%T)] FAIL  $name (see experiments/logs/$name.log)"
  fi
}
run effb3_se_$DS      scripts/train.py --model effb3_se     --dataset "$DS"
run inception_v3_$DS  scripts/train.py --model inception_v3 --dataset "$DS"
run densenet169_$DS   scripts/train.py --model densenet169  --dataset "$DS"
run resnet50_cnn_$DS  scripts/train.py --model resnet50_rf  --dataset "$DS"
run resnet50_rf_$DS   scripts/train_features.py --model resnet50_rf --dataset "$DS" \
    --cnn_ckpt "checkpoints/resnet50_rf__${DS}__5class__raw_384_s42.pt"
run hybrid_svm_$DS    scripts/train_features.py --model hybrid_svm --dataset "$DS" --task 5class
run light2stage_$DS   scripts/run_two_stage.py --dataset "$DS"
run mscnn_lstm_$DS    scripts/train.py --model mscnn_lstm   --dataset "$DS"
run vit_b16_$DS       scripts/train.py --model vit_b16      --dataset "$DS"
run p8_vgg16_$DS      scripts/train.py --model p8_vgg16     --dataset "$DS"
run mil_resnet18_$DS  scripts/train.py --model mil_resnet18 --dataset "$DS"
echo "[$(date +%T)] QUEUE FINISHED"
