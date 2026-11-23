#!/usr/bin/env bash
# Download the genuine Kaggle-hosted datasets with a stall watchdog.
# Needs ~/.kaggle/access_token and accepted competition rules.
# Messidor-2 images come from ADCIS (manual); only its grades are fetched here.
#
# Kaggle's CLI has no network timeout: a silently dropped connection hangs forever.
# Each download runs under a watchdog that kills it when the file has not grown for
# STALL_SECS and restarts it; the CLI resumes from its .kaggle-partial marker.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
STALL_SECS=${STALL_SECS:-120}
MAX_TRIES=${MAX_TRIES:-60}
kg() { uvx --python 3.12 --from 'kaggle>=1.8' --with six kaggle "$@"; }

# dl <expected output file> <kaggle args...>
dl() {
  local out="$1"; shift
  for try in $(seq 1 "$MAX_TRIES"); do
    if [ -f "$out" ] && [ ! -f "$out.kaggle-partial" ]; then
      echo "  [$(date +%T)] $(basename "$out") complete"; return 0
    fi
    echo "  [$(date +%T)] $(basename "$out"): attempt $try (have $(stat -c %s "$out" 2>/dev/null || echo 0) bytes)"
    kg "$@" >/dev/null 2>&1 &
    local pid=$! last=-1 still=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 15
      local now; now=$(stat -c %s "$out" 2>/dev/null || echo 0)
      if [ "$now" = "$last" ]; then still=$((still + 15)); else still=0; last=$now; fi
      if [ "$still" -ge "$STALL_SECS" ]; then
        echo "  [$(date +%T)] stalled ${STALL_SECS}s at $now bytes -> restarting"
        pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; sleep 2
        pkill -f "kaggle competitions download.*$(basename "$out")" 2>/dev/null
        pkill -f "kaggle datasets download" 2>/dev/null
        break
      fi
    done
    wait "$pid" 2>/dev/null
    sleep 3
  done
  echo "  FAILED: $out after $MAX_TRIES attempts"; return 1
}

echo "[$(date +%T)] Messidor-2 adjudicated grades (Krause et al. 2018)"
mkdir -p "$RAW/messidor2"
[ -f "$RAW/messidor2/messidor_data.csv" ] || kg datasets download google-brain/messidor2-dr-grades -p "$RAW/messidor2" --unzip

echo "[$(date +%T)] EyePACS labels"
mkdir -p "$RAW/eyepacs"
if [ ! -f "$RAW/eyepacs/trainLabels.csv" ]; then
  kg competitions download diabetic-retinopathy-detection -f trainLabels.csv.zip -p "$RAW/eyepacs"
  (cd "$RAW/eyepacs" && unzip -o -q trainLabels.csv.zip && rm trainLabels.csv.zip)
fi

echo "[$(date +%T)] APTOS 2019 (full competition archive; only train is labelled)"
mkdir -p "$RAW/aptos2019"
if [ ! -f "$RAW/aptos2019/train.csv" ]; then
  Z="$RAW/aptos2019/aptos2019-blindness-detection.zip"
  dl "$Z" competitions download aptos2019-blindness-detection -p "$RAW/aptos2019" || exit 1
  echo "  [$(date +%T)] verifying + extracting train images"
  (cd "$RAW/aptos2019" && unzip -tq "$(basename "$Z")" && unzip -o -q "$(basename "$Z")" 'train.csv' 'train_images/*' \
     && rm "$(basename "$Z")") || { echo "  APTOS zip failed integrity check"; exit 1; }
fi

echo "[$(date +%T)] EyePACS train parts (5 x ~8.4 GB, byte-split zip; not concatenated)"
# Kaggle serves each part wrapped in a (stored) zip: train.zip.00N.zip -> train.zip.00N.
# The watchdog must watch the wrapper file; unwrap one part at a time to bound disk use.
if [ -f "$ROOT/data/splits/eyepacs_5class.csv" ]; then
  echo "  EyePACS already cached to data/processed/eyepacs (raw parts deleted after caching); skipping"
else
for i in 001 002 003 004 005; do
  f="train.zip.$i"
  if [ -f "$RAW/eyepacs/$f" ] && [ ! -f "$RAW/eyepacs/$f.zip" ]; then
    echo "  [$(date +%T)] $f already present"; continue
  fi
  dl "$RAW/eyepacs/$f.zip" competitions download diabetic-retinopathy-detection -f "$f" -p "$RAW/eyepacs" || exit 1
  echo "  [$(date +%T)] unwrapping $f.zip"
  (cd "$RAW/eyepacs" && unzip -o -q "$f.zip" && rm "$f.zip") || { echo "  unwrap failed for $f"; exit 1; }
done
fi
echo "[$(date +%T)] ALL DOWNLOADS DONE"
