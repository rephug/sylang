#!/usr/bin/env bash
# smoke_cpu.sh - end-to-end CPU smoke test of the pilot package with a tiny randomly initialised
# Qwen3-architecture model (no weight download; interfaces.md, "Stage scripts").
#
#   data.py --smoke --version smoke   (64 train cases per seed for 2 seeds, 16 dev, 32 test)
#   data.py --check --version smoke
#   train_lora.py  configs/smoke_sylang2.json and configs/smoke_english.json (20 steps each; the
#                  tiny model is built by the trainer on first use, common.tiny_model_dir)
#   evaluate.py    --adapter none and --adapter runs/<name>/best, --split test --limit 32 --latency-sample 2
#   analyze.py     --results results/smoke_*/ --out analysis/smoke/
#
# Required: PYTHON (the venv's python with torch, transformers and peft), PILOT_TOKENIZER (the Qwen 3
# tokenizer directory) and SYLANG_PACKET (the study packet, for the english rendering).
# Optional: SMOKE_VERBOSE=1 shows every step's output; OMP_NUM_THREADS (default 2).
# Exits non-zero on the first failure and prints the tail of that step's log.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRIPT=$(basename "${BASH_SOURCE[0]}")

missing=0
for v in PYTHON PILOT_TOKENIZER SYLANG_PACKET; do
  if [ -z "${!v:-}" ]; then echo "$SCRIPT: $v is not set" >&2; missing=1; fi
done
if [ "$missing" -ne 0 ]; then
  cat >&2 <<'EOF'
smoke_cpu.sh needs:  PYTHON=<venv>/bin/python  PILOT_TOKENIZER=<dir with tokenizer.json>  SYLANG_PACKET=<packet path>
EOF
  exit 2
fi
[ -f "$PILOT_TOKENIZER/tokenizer.json" ] || { echo "$SCRIPT: PILOT_TOKENIZER=$PILOT_TOKENIZER holds no tokenizer.json" >&2; exit 2; }
[ -e "$SYLANG_PACKET" ] || { echo "$SCRIPT: SYLANG_PACKET=$SYLANG_PACKET does not exist" >&2; exit 2; }
"$PYTHON" -c 'import torch, transformers, peft, common' 2> /dev/null || {
  echo "$SCRIPT: $PYTHON cannot import torch/transformers/peft/common.py" >&2; exit 2; }
for f in data.py train_lora.py evaluate.py analyze.py configs/smoke_sylang2.json configs/smoke_english.json; do
  [ -f "$f" ] || { echo "$SCRIPT: $f is missing" >&2; exit 2; }
done

export PILOT_MODEL=tiny-random            # never real weights in the smoke test
unset PILOT_ALLOW_DOWNLOAD                # and never a download
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS TOKENIZERS_PARALLELISM=false
VERSION=smoke
ARMS=(sylang2 english)
LOG_DIR=runs/_smoke_logs
PRICE=0.34                                # a nominal price so the cost fields are exercised
START=$SECONDS

# A clean slate: stale smoke outputs must not mask a failure.
rm -rf "$LOG_DIR" analysis/smoke
for arm in "${ARMS[@]}"; do rm -rf "runs/smoke_$arm" "results/smoke_$arm" "results/smoke_${arm}_zeroshot"; done
mkdir -p "$LOG_DIR"

step() {  # step <label> <command...>: short line per step; the log on failure
  local label=$1 log t0=$SECONDS rc
  shift
  log="$LOG_DIR/$(printf '%s' "$label" | tr -c 'A-Za-z0-9_.-' '_').log"
  printf '%-44s' "$label"
  if [ "${SMOKE_VERBOSE:-0}" = 1 ]; then
    echo
    set +e; "$@" 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}; set -e
  else
    set +e; "$@" > "$log" 2>&1; rc=$?; set -e
  fi
  if [ "$rc" -ne 0 ]; then
    echo "FAILED (exit $rc, $((SECONDS - t0)) s) - $log"
    echo "----- last 40 lines of $log -----"; tail -n 40 "$log"
    echo "$SCRIPT: FAILED at step '$label'"
    exit 1
  fi
  echo "ok ($((SECONDS - t0)) s)"
}
expect_file() {  # expect_file <path> [<path>...]
  local p
  for p in "$@"; do [ -s "$p" ] || { echo "$SCRIPT: expected non-empty file $p after the last step"; exit 1; }; done
}
json_get() {  # json_get <file> <dotted.key>: a value out of a JSON file ("missing" when absent)
  "$PYTHON" - "$1" "$2" <<'EOF'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    obj = json.load(f)
for key in sys.argv[2].split('.'):
    obj = obj.get(key, 'missing') if isinstance(obj, dict) else 'missing'
print(obj)
EOF
}

echo "smoke test: $HERE (python $("$PYTHON" -c 'import sys; print(sys.version.split()[0])'), $OMP_NUM_THREADS threads, tiny-random model)"
step "data.py --smoke --version $VERSION" "$PYTHON" data.py --smoke --version "$VERSION"
expect_file "data/$VERSION/manifest.json" "data/$VERSION/legends.json" "data/$VERSION/test_sylang2.jsonl" \
  "data/$VERSION/train_s0_english.jsonl" "data/$VERSION/dev_english.jsonl"
step "data.py --check --version $VERSION" "$PYTHON" data.py --check --version "$VERSION"

for arm in "${ARMS[@]}"; do
  name="smoke_$arm"; cfg="configs/$name.json"
  tmo=$("$PYTHON" -c 'import sys, common; print(int(float(common.load_config(sys.argv[1])["timeout_minutes"]) * 60))' "$cfg")
  step "train_lora.py $name (20 steps)" timeout --kill-after=30 "$tmo" "$PYTHON" train_lora.py --config "$cfg" --out "runs/$name"
  expect_file "runs/$name/run.json" "runs/$name/train_log.jsonl" "runs/$name/best/adapter_config.json"
  status=$(json_get "runs/$name/run.json" status)
  [ "$status" = completed ] || { echo "$SCRIPT: runs/$name/run.json status is '$status', not completed"; exit 1; }
  step "evaluate.py $name zero-shot (32 items)" timeout --kill-after=30 "$tmo" "$PYTHON" evaluate.py --config "$cfg" \
    --adapter none --split test --out "results/${name}_zeroshot" --limit 32 --latency-sample 2 --price-per-hour "$PRICE"
  expect_file "results/${name}_zeroshot/eval.json" "results/${name}_zeroshot/items.jsonl"
  step "evaluate.py $name trained (32 items)" timeout --kill-after=30 "$tmo" "$PYTHON" evaluate.py --config "$cfg" \
    --adapter "runs/$name/best" --split test --out "results/$name" --limit 32 --latency-sample 2 --price-per-hour "$PRICE"
  expect_file "results/$name/eval.json" "results/$name/items.jsonl"
  n_items=$(json_get "results/$name/eval.json" n_items)
  [ "$n_items" = 32 ] || { echo "$SCRIPT: results/$name/eval.json n_items is '$n_items', expected 32"; exit 1; }
done

step "analyze.py -> analysis/smoke/" "$PYTHON" analyze.py --results results/smoke_sylang2_zeroshot/ results/smoke_sylang2/ \
  results/smoke_english_zeroshot/ results/smoke_english/ --out analysis/smoke/
expect_file analysis/smoke/summary.md analysis/smoke/summary.json

echo
echo "accuracy on the 32 smoke test items (a random tiny model is expected to be near zero; the point is that every path runs):"
for arm in "${ARMS[@]}"; do
  for suffix in _zeroshot ''; do
    printf '  %-26s acc_final=%s invalid_final=%s\n' "smoke_$arm$suffix" \
      "$(json_get "results/smoke_$arm$suffix/eval.json" panels.all.all.acc_final)" \
      "$(json_get "results/smoke_$arm$suffix/eval.json" panels.all.all.invalid_final)"
  done
done
echo "smoke test passed in $((SECONDS - START)) s; logs in $HERE/$LOG_DIR/, analysis in $HERE/analysis/smoke/"
