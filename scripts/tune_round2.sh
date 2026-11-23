#!/usr/bin/env bash
# Tuning round 2 (APTOS, VALIDATION ONLY). Round 1 deltas were all within about ±0.017 score, so
# (a) re-run the ResNet50 baseline with 2 more seeds to measure run-to-run noise, and
# (b) test the only changes that helped, plus the longer ResNet50 schedule (its best epoch was the last).
cd "$(dirname "$0")/.."
DS=${DS:-aptos}; LOG=experiments/tune_${DS}.log
run() {
  local name="$1"; shift
  echo "[$(date +%T)] START $name" >> "$LOG"
  if uv run python scripts/train.py --dataset "$DS" --run_name "tune__${name}__${DS}" "$@" \
       > "experiments/logs/tune__${name}__${DS}.log" 2>&1; then
    echo "[$(date +%T)] DONE  $name" >> "$LOG"
  else echo "[$(date +%T)] FAIL  $name" >> "$LOG"; fi
}
run resnet50_rf__ep40         --model resnet50_rf --epochs 40 --patience 10
run densenet169__focal_res512 --model densenet169 --loss focal --size 512 --batch_size 6 --accum 2
run resnet50_rf__seed1        --model resnet50_rf --seed 1
run resnet50_rf__seed2        --model resnet50_rf --seed 2
echo "[$(date +%T)] ROUND2 FINISHED" >> "$LOG"
