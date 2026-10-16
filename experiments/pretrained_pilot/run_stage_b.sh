#!/usr/bin/env bash
# run_stage_b.sh - stage B of the pretrained pilot (interfaces.md, "Stage scripts").
#
# Cells: the four arms x seed 0 x 2,000 training cases (configs/stage_b_<arm>_s0_n2000.json), the
# zero-shot baseline of every arm (evaluate.py --adapter none), and, when STAGE_B_CURVES=1, the
# english and sylang2 learning-curve points at 500 and 8,000 cases. analyze.py runs at the end over
# every result directory (analysis/stage_b/).
#
# Every cell is idempotent: a cell whose runs/<name>/run.json says "completed" and whose
# results/<name>/eval.json exists is skipped; a run with checkpoints but no completion is resumed.
# train_lora.py and evaluate.py each run under `timeout` with the config's timeout_minutes; a
# timeout (exit 124/137) or a failure is written to the ledger as a note and the next cell runs.
#
# This script never calls a provider API. The owner starts the machine, runs
#     bash run_stage_b.sh 2>&1 | tee stage_b.log
# and when it ends stops/terminates the machine by hand and completes verify_teardown.md.
#
# Required: PILOT_TOKENIZER PILOT_MODEL PILOT_PRICE_PER_HOUR PILOT_PROVIDER PILOT_INSTANCE PILOT_GPU.
# Optional: PYTHON (default python), STAGE_B_CURVES=1, PILOT_CEILING (USD, default 100),
# PILOT_STAGE_BUDGET (USD: no new cell is started once elapsed x price exceeds it),
# PILOT_UPLOAD_CMD (a shell command that receives the results tarball path appended as its last
# argument, the convention of profile.py --upload-cmd; e.g. a small wrapper script).
set -euo pipefail

STAGE=B
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PYTHON=${PYTHON:-python}
SCRIPT=$(basename "${BASH_SOURCE[0]}")
DONE=()
FAILED=()
SKIPPED=()

# ---- preconditions ---------------------------------------------------------------------------
missing=0
for v in PILOT_TOKENIZER PILOT_MODEL PILOT_PRICE_PER_HOUR PILOT_PROVIDER PILOT_INSTANCE PILOT_GPU; do
  if [ -z "${!v:-}" ]; then echo "$SCRIPT: $v is not set" >&2; missing=1; fi
done
[ "$missing" -eq 0 ] || { echo "$SCRIPT: refusing to start (see the header of this script)" >&2; exit 2; }
[ -f "$PILOT_TOKENIZER/tokenizer.json" ] || { echo "$SCRIPT: PILOT_TOKENIZER=$PILOT_TOKENIZER holds no tokenizer.json" >&2; exit 2; }
if [ "$PILOT_MODEL" != tiny-random ] && [ ! -f "$PILOT_MODEL/config.json" ]; then
  echo "$SCRIPT: PILOT_MODEL=$PILOT_MODEL is not a model directory (no config.json)" >&2; exit 2
fi
case "$PILOT_PRICE_PER_HOUR" in
  ''|*[!0-9.]*|.|*.*.*) echo "$SCRIPT: PILOT_PRICE_PER_HOUR=$PILOT_PRICE_PER_HOUR is not a number" >&2; exit 2 ;;
esac
"$PYTHON" -c 'import common' || { echo "$SCRIPT: $PYTHON cannot import common.py (set PYTHON to the venv's python)" >&2; exit 2; }
for f in train_lora.py evaluate.py analyze.py ledger.py; do
  [ -f "$f" ] || { echo "$SCRIPT: $f is missing" >&2; exit 2; }
done

# ---- helpers ---------------------------------------------------------------------------------
cfg_get() {  # cfg_get <config.json> <key>: the resolved config value
  "$PYTHON" -c 'import sys, common; print(common.load_config(sys.argv[1])[sys.argv[2]])' "$1" "$2"
}
cfg_timeout_seconds() {  # cfg_timeout_seconds <config.json>
  "$PYTHON" -c 'import sys, common; print(int(float(common.load_config(sys.argv[1])["timeout_minutes"]) * 60))' "$1"
}
run_status() {  # run_status <name>: completed|running|failed|timeout|missing (runs/<name>/run.json)
  "$PYTHON" - "$1" <<'EOF'
import json, os, sys
path = os.path.join('runs', sys.argv[1], 'run.json')
if not os.path.exists(path):
    print('missing')
else:
    try:
        with open(path, encoding='utf-8') as f:
            print(json.load(f).get('status') or 'unknown')
    except ValueError:
        print('unreadable')
EOF
}
ledger_add() { "$PYTHON" ledger.py add "$@"; }
ledger_note() {  # ledger_note <text> [run-name]
  if [ $# -ge 2 ]; then
    ledger_add --event note --stage "$STAGE" --note "$1" --run-names "$2" || echo "$SCRIPT: WARNING: ledger note not written: $1" >&2
  else
    ledger_add --event note --stage "$STAGE" --note "$1" || echo "$SCRIPT: WARNING: ledger note not written: $1" >&2
  fi
}
elapsed_spend() { awk -v s="$SECONDS" -v p="$PILOT_PRICE_PER_HOUR" 'BEGIN { printf "%.4f", s / 3600 * p }'; }
budget_left() {  # false once elapsed x price exceeds PILOT_STAGE_BUDGET (when set)
  [ -z "${PILOT_STAGE_BUDGET:-}" ] && return 0
  awk -v s="$SECONDS" -v p="$PILOT_PRICE_PER_HOUR" -v b="$PILOT_STAGE_BUDGET" 'BEGIN { exit !(s / 3600 * p < b) }'
}
banner() { printf '\n===== [%s +%ss] %s =====\n' "$(date -u +%H:%M:%S)" "$SECONDS" "$*"; }

run_cell() {  # run_cell <name>: train (or resume) then evaluate on test; never aborts the stage
  local name=$1 cfg="configs/$1.json" tmo status rc minutes
  if [ ! -f "$cfg" ]; then ledger_note "cell $name: $cfg is missing; skipped" "$name"; FAILED+=("$name"); return 0; fi
  tmo=$(cfg_timeout_seconds "$cfg"); minutes=$(cfg_get "$cfg" timeout_minutes)
  status=$(run_status "$name")
  if [ "$status" = completed ] && [ -f "results/$name/eval.json" ]; then
    banner "cell $name: already trained and evaluated; skipping"; SKIPPED+=("$name"); return 0
  fi
  if ! budget_left; then
    ledger_note "cell $name: not started, PILOT_STAGE_BUDGET=\$$PILOT_STAGE_BUDGET reached after ${SECONDS}s" "$name"
    FAILED+=("$name"); return 0
  fi
  if [ "$status" != completed ]; then
    local resume=()
    if compgen -G "runs/$name/checkpoints/epoch*" > /dev/null; then resume=(--resume); fi
    banner "cell $name: train_lora.py (run.json status: $status; timeout $minutes min${resume[0]:+; resuming})"
    set +e
    timeout --kill-after=60 "$tmo" "$PYTHON" train_lora.py --config "$cfg" --out "runs/$name" ${resume[0]:+"${resume[@]}"}
    rc=$?
    set -e
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      ledger_note "cell $name: train_lora.py timed out after $minutes min (status timeout, exit $rc); continuing" "$name"
      FAILED+=("$name"); return 0
    elif [ "$rc" -ne 0 ]; then
      ledger_note "cell $name: train_lora.py failed with exit $rc; continuing" "$name"
      FAILED+=("$name"); return 0
    fi
    status=$(run_status "$name")
    if [ "$status" != completed ]; then
      ledger_note "cell $name: train_lora.py exited 0 but run.json status is $status; continuing" "$name"
      FAILED+=("$name"); return 0
    fi
  fi
  if [ ! -d "runs/$name/best" ]; then
    ledger_note "cell $name: completed but runs/$name/best is missing; not evaluated" "$name"; FAILED+=("$name"); return 0
  fi
  banner "cell $name: evaluate.py --split test (timeout $minutes min)"
  set +e
  timeout --kill-after=60 "$tmo" "$PYTHON" evaluate.py --config "$cfg" --adapter "runs/$name/best" --split test \
    --out "results/$name" --price-per-hour "$PILOT_PRICE_PER_HOUR" --latency-sample 50
  rc=$?
  set -e
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    ledger_note "cell $name: evaluate.py timed out after $minutes min (exit $rc); continuing" "$name"; FAILED+=("$name"); return 0
  elif [ "$rc" -ne 0 ] || [ ! -f "results/$name/eval.json" ]; then
    ledger_note "cell $name: evaluate.py failed (exit $rc, eval.json $([ -f "results/$name/eval.json" ] && echo present || echo absent)); continuing" "$name"
    FAILED+=("$name"); return 0
  fi
  DONE+=("$name")
}

run_zero_shot() {  # run_zero_shot <config-name> <arm> <answer-format>: the untrained baseline, once per arm and format
  local cfg="configs/$1.json" name="zeroshot_$2_$3" tmo rc
  if [ -f "results/$name/eval.json" ]; then banner "$name: already evaluated; skipping"; SKIPPED+=("$name"); return 0; fi
  if ! budget_left; then ledger_note "$name: not started, PILOT_STAGE_BUDGET reached" "$name"; FAILED+=("$name"); return 0; fi
  tmo=$(cfg_timeout_seconds "$cfg")
  banner "$name: evaluate.py --adapter none --split test"
  set +e
  timeout --kill-after=60 "$tmo" "$PYTHON" evaluate.py --config "$cfg" --adapter none --split test \
    --out "results/$name" --price-per-hour "$PILOT_PRICE_PER_HOUR" --latency-sample 50
  rc=$?
  set -e
  if [ "$rc" -ne 0 ] || [ ! -f "results/$name/eval.json" ]; then
    ledger_note "$name: zero-shot evaluate.py failed or timed out (exit $rc); continuing" "$name"; FAILED+=("$name"); return 0
  fi
  DONE+=("$name")
}

run_analysis() {  # run_analysis <out-dir>: analyze.py over every non-smoke result directory
  local dirs=() d rc
  shopt -s nullglob
  for d in results/*/; do case "$d" in results/smoke_*) ;; *) [ -f "$d/eval.json" ] && dirs+=("$d") ;; esac; done
  shopt -u nullglob
  if [ "${#dirs[@]}" -eq 0 ]; then ledger_note "analysis skipped: no result directory holds an eval.json"; return 0; fi
  banner "analyze.py over ${#dirs[@]} result directories -> $1"
  set +e
  "$PYTHON" analyze.py --results "${dirs[@]}" --out "$1"
  rc=$?
  set -e
  [ "$rc" -eq 0 ] || ledger_note "analyze.py failed with exit $rc (results are kept; re-run it by hand)"
}

package_results() {  # a tarball of everything worth keeping, uploaded with PILOT_UPLOAD_CMD when set
  local ts tarball paths=() p
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p runs/_outbox
  tarball="runs/_outbox/stage_$(printf '%s' "$STAGE" | tr '[:upper:]' '[:lower:]')_${ts}.tar.gz"
  shopt -s nullglob
  for p in ledger.jsonl profile.json results analysis runs/*/run.json runs/*/train_log.jsonl runs/*/best; do
    [ -e "$p" ] && paths+=("$p")
  done
  shopt -u nullglob
  [ "${#paths[@]}" -gt 0 ] || { echo "$SCRIPT: nothing to package"; return 0; }
  tar czf "$tarball" --ignore-failed-read "${paths[@]}" || { echo "$SCRIPT: WARNING: tar failed" >&2; return 0; }
  echo "$SCRIPT: packaged $(du -h "$tarball" | cut -f1) -> $HERE/$tarball"
  if [ -n "${PILOT_UPLOAD_CMD:-}" ]; then
    local cmd="$PILOT_UPLOAD_CMD '$HERE/$tarball'"   # the path is appended, as profile.py --upload-cmd does
    banner "upload: $cmd"
    bash -c "$cmd" || ledger_note "upload command failed; copy $HERE/$tarball off the machine by hand"
  else
    echo "$SCRIPT: PILOT_UPLOAD_CMD is not set: copy $HERE/$tarball off the machine before terminating it"
  fi
}

finish() {  # EXIT trap: the ledger stop line (an estimate), the package, the teardown reminder
  local rc=$? spend secs=$SECONDS
  trap - EXIT
  set +e
  spend=$(elapsed_spend)
  banner "stage $STAGE script ending (exit $rc after ${secs}s): done=${#DONE[@]} failed=${#FAILED[@]} skipped=${#SKIPPED[@]}"
  package_results
  if ! ledger_add --event stop --provider "$PILOT_PROVIDER" --instance "$PILOT_INSTANCE" --gpu "$PILOT_GPU" \
      --price-per-hour "$PILOT_PRICE_PER_HOUR" --stage "$STAGE" --measured-spend "$spend" \
      --run-names ${DONE[0]:+"${DONE[@]}"} \
      --note "estimate: ${secs}s x \$$PILOT_PRICE_PER_HOUR/h; script exit $rc; failed: ${FAILED[*]:-none}; correct with a balance line from the provider's billing page"; then
    echo "$SCRIPT: !!! the ledger stop line could not be written; add it by hand:" >&2
    echo "  $PYTHON ledger.py add --event stop --provider $PILOT_PROVIDER --instance $PILOT_INSTANCE --gpu \"$PILOT_GPU\" --price-per-hour $PILOT_PRICE_PER_HOUR --stage $STAGE --measured-spend $spend" >&2
  fi
  cat <<EOF

Stage $STAGE finished on this machine. The instance is STILL RUNNING AND BILLING until you stop it.
  1. copy the package (above) off the machine if no upload command ran;
  2. stop/terminate the instance in the provider console AND check with the provider CLI;
  3. complete every line of the checklist:  $HERE/verify_teardown.md
  4. record the provider's figure:  $PYTHON ledger.py add --event balance --provider $PILOT_PROVIDER --stage $STAGE --balance-after <USD left> --measured-spend <USD the provider charged for this session>
  5. $PYTHON ledger.py report --ceiling ${PILOT_CEILING:-100}
EOF
  exit "$rc"
}

# ---- the stage ----------------------------------------------------------------------------------
CELLS=(stage_b_english_s0_n2000 stage_b_json_s0_n2000 stage_b_sylang2_s0_n2000 stage_b_sylang2g_s0_n2000)
PURPOSE="stage B: four arms x seed 0 x 2000 cases, zero-shot baseline per arm"
if [ "${STAGE_B_CURVES:-0}" = 1 ]; then
  CELLS+=(stage_b_english_s0_n500 stage_b_sylang2_s0_n500 stage_b_english_s0_n8000 stage_b_sylang2_s0_n8000)
  PURPOSE="$PURPOSE, learning curves 500/8000 for english and sylang2 (STAGE_B_CURVES=1)"
fi
for name in "${CELLS[@]}"; do  # every config must load before the ledger start line is written
  [ -f "configs/$name.json" ] || { echo "$SCRIPT: configs/$name.json is missing" >&2; exit 2; }
  cfg_get "configs/$name.json" name > /dev/null || { echo "$SCRIPT: configs/$name.json does not load" >&2; exit 2; }
done
DATA_VERSION=$(cfg_get "configs/${CELLS[0]}.json" data_version)
[ -f "data/$DATA_VERSION/manifest.json" ] || {
  echo "$SCRIPT: data/$DATA_VERSION/manifest.json is missing: unpack the dataset tarball from the development server (setup_gpu.sh)" >&2; exit 2; }

banner "stage $STAGE on $PILOT_PROVIDER/$PILOT_INSTANCE ($PILOT_GPU, \$$PILOT_PRICE_PER_HOUR/h); model $PILOT_MODEL; cells: ${CELLS[*]}"
ledger_add --event start --provider "$PILOT_PROVIDER" --instance "$PILOT_INSTANCE" --gpu "$PILOT_GPU" \
  --price-per-hour "$PILOT_PRICE_PER_HOUR" --stage "$STAGE" --run-names "${CELLS[@]}" --purpose "$PURPOSE"
trap finish EXIT

# Zero-shot baselines first: cheap, and they validate the evaluation path before any training.
for arm in english json sylang2 sylang2g; do run_zero_shot "stage_b_${arm}_s0_n2000" "$arm" json; done
for name in "${CELLS[@]}"; do run_cell "$name"; done
run_analysis analysis/stage_b/
banner "stage $STAGE cells done: ${DONE[*]:-none}; failed: ${FAILED[*]:-none}; skipped (already done): ${SKIPPED[*]:-none}"
