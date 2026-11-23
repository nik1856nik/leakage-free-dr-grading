#!/usr/bin/env bash
# Best-model pipeline (APTOS five-class), VALIDATION ONLY until the final step is run by hand.
#  1. Fine-tune the top-3 architectures on APTOS starting from their EyePACS-trained weights,
#     each as a classifier (softmax) and as a grade regressor (SmoothL1, one output).
#  2. Select the ensemble + grade thresholds by repeated 5-fold cross-fitted validation score
#     (scripts/best_model.py); APTOS-only tuned models are candidates too.
# The test split is evaluated only afterwards, once: scripts/best_model.py --final ...
cd "$(dirname "$0")/.."
DS=aptos
LOG=experiments/best_${DS}.log
FT="--dataset $DS --epochs 15 --patience 6 --lr 1e-4 --warmup_epochs 1 --scheduler cosine"
run() {
  local name="$1"; shift
  echo "[$(date +%T)] START $name" >> "$LOG"
  if uv run python scripts/train.py --run_name "ft__${name}__${DS}" $FT "$@" \
       > "experiments/logs/ft__${name}__${DS}.log" 2>&1; then
    echo "[$(date +%T)] DONE  $name $(grep -E '"(qwk|macro_f1)"' experiments/logs/ft__${name}__${DS}.log | tr -d ' \n')" >> "$LOG"
  else
    echo "[$(date +%T)] FAIL  $name" >> "$LOG"
  fi
}
E=checkpoints
run resnet50_cls   --model resnet50_rf --init_from $E/resnet50_rf__eyepacs__5class__raw_384_s42.pt
run resnet50_reg   --model resnet50_rf --init_from $E/resnet50_rf__eyepacs__5class__raw_384_s42.pt --head reg
run densenet_cls   --model densenet169 --init_from $E/densenet169__eyepacs__5class__raw_384_s42.pt --loss focal
run densenet_reg   --model densenet169 --init_from $E/densenet169__eyepacs__5class__raw_384_s42.pt --head reg
run effb3_cls      --model effb3_se --init_from $E/effb3_se__eyepacs__5class__ben_graham_512_s42.pt \
                   --size 512 --batch_size 8 --accum 2
run effb3_reg      --model effb3_se --init_from $E/effb3_se__eyepacs__5class__ben_graham_512_s42.pt \
                   --size 512 --batch_size 8 --accum 2 --head reg
echo "[$(date +%T)] SELECTION" >> "$LOG"
uv run python scripts/best_model.py --dataset $DS --max_size 4 --members \
  "$E/ft__*__${DS}.pt" $E/inception_v3__aptos__5class__raw_384_s42.pt \
  $E/resnet50_rf__aptos__5class__raw_384_s42.pt \
  $E/tune__densenet169__focal__aptos.pt \
  $E/tune__effb3_se__res512__aptos.pt >> "$LOG" 2>&1
echo "[$(date +%T)] BEST PIPELINE FINISHED (validation only; run --final once to test)" >> "$LOG"
