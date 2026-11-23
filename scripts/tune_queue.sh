#!/usr/bin/env bash
# Stage 4b, tuning round 1: top-3 APTOS baselines by validation score 0.5*(QWK + macro-F1):
#   ResNet50 (P11, CNN stage) 0.855 · DenseNet-169 (P14) 0.849 · EffB3+SE (P4) 0.831
# One factor changed at a time vs. each model's baseline (configs/models.yaml), so every
# delta is attributable. VALIDATION ONLY: no --final, the test split is never read.
# Runs are named tune__<model>__<ds>__<variant> (kept apart from the baseline runs).
# Factors: resolution 512 (same effective batch via grad accumulation), the other
# preprocessing (raw <-> Ben Graham), focal loss, class-balanced (sqrt) sampling.
cd "$(dirname "$0")/.."
mkdir -p experiments/logs
DS=${DS:-aptos}
LOG=experiments/tune_${DS}.log
run() {
  local name="$1"; shift
  echo "[$(date +%T)] START $name" >> "$LOG"
  if uv run python scripts/train.py --dataset "$DS" --run_name "tune__${name}__${DS}" "$@" \
       > "experiments/logs/tune__${name}__${DS}.log" 2>&1; then
    echo "[$(date +%T)] DONE  $name" >> "$LOG"
  else
    echo "[$(date +%T)] FAIL  $name (see experiments/logs/tune__${name}__${DS}.log)" >> "$LOG"
  fi
}
# model         variant         overrides
for m in resnet50_rf densenet169 effb3_se; do
  case $m in
    resnet50_rf) other=ben_graham; bs=8; acc=2 ;;
    densenet169) other=ben_graham; bs=6; acc=2 ;;
    effb3_se)    other=raw;        bs=8; acc=2 ;;
  esac
  run "${m}__res512"     --model $m --size 512 --batch_size $bs --accum $acc
  run "${m}__enh_$other" --model $m --enhancement $other
  run "${m}__focal"      --model $m --loss focal
  run "${m}__sqrtsamp"   --model $m --sampler sqrt --loss ce
done
echo "[$(date +%T)] QUEUE FINISHED" >> "$LOG"
