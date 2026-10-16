#!/usr/bin/env bash
# run_stage_a.sh - stage A of the pretrained pilot (interfaces.md, "Stage scripts" and "Profiling").
#
# Stage A measures the machine, it does not run experiments: profile_gpu.py with
# configs/stage_a_profile.json (model load, peak memory, training examples/s, generation items/s,
# checkpoint size and upload time) writes profile.json with the projected cost of stages B and C at
# $PILOT_PRICE_PER_HOUR; then `ledger.py report --ceiling $PILOT_CEILING --profile profile.json`
# says whether the projection fits the ceiling (the script's exit code carries that verdict: do not
# start stage B on a non-zero exit). With STAGE_A_ZERO_SHOT=1 the english zero-shot baseline is
# evaluated as well (a cheap end-to-end check of the evaluation path; stage B skips it when present).
#
# This script never calls a provider API. The owner starts the machine, runs
#     bash run_stage_a.sh 2>&1 | tee stage_a.log
# and when it ends stops/terminates the machine by hand and completes verify_teardown.md.
#
# Required: PILOT_TOKENIZER PILOT_MODEL PILOT_PRICE_PER_HOUR PILOT_PROVIDER PILOT_INSTANCE PILOT_GPU.
# Optional: PYTHON (default python), PILOT_CEILING (USD, default 100), STAGE_A_ZERO_SHOT=1,
# PILOT_UPLOAD_CMD (passed to profile_gpu.py --upload-cmd to time the checkpoint upload, and used at the
# end for the results tarball; in both cases the path is appended as the last argument, e.g.
# 'rclone copy --quiet' or 'scp -r -o BatchMode=yes dev-server:pilot_inbox/' style wrappers).
set -euo pipefail

STAGE=A
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PYTHON=${PYTHON:-python}
SCRIPT=$(basename "${BASH_SOURCE[0]}")
PROFILE_PY=profile_gpu.py   # the profiler script (interfaces.md "Profiling")
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
for f in "$PROFILE_PY" evaluate.py ledger.py configs/plan_stage_b.json configs/plan_stage_c.json; do
  [ -f "$f" ] || { echo "$SCRIPT: $f is missing" >&2; exit 2; }
done

# ---- helpers ---------------------------------------------------------------------------------
cfg_get() {  # cfg_get <config.json> <key>: the resolved config value
  "$PYTHON" -c 'import sys, common; print(common.load_config(sys.argv[1])[sys.argv[2]])' "$1" "$2"
}
cfg_timeout_seconds() {  # cfg_timeout_seconds <config.json>
  "$PYTHON" -c 'import sys, common; print(int(float(common.load_config(sys.argv[1])["timeout_minutes"]) * 60))' "$1"
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
banner() { printf '\n===== [%s +%ss] %s =====\n' "$(date -u +%H:%M:%S)" "$SECONDS" "$*"; }

run_zero_shot() {  # run_zero_shot <config-name> <arm> <answer-format>: the untrained baseline
  local cfg="configs/$1.json" name="zeroshot_$2_$3" tmo rc
  if [ -f "results/$name/eval.json" ]; then banner "$name: already evaluated; skipping"; SKIPPED+=("$name"); return 0; fi
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
    local cmd="$PILOT_UPLOAD_CMD '$HERE/$tarball'"   # the path is appended, as profile_gpu.py --upload-cmd does
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
  5. $PYTHON ledger.py report --ceiling ${PILOT_CEILING:-100} --profile profile.json
EOF
  exit "$rc"
}

# ---- the stage ----------------------------------------------------------------------------------
CFG=configs/stage_a_profile.json
[ -f "$CFG" ] || { echo "$SCRIPT: $CFG is missing" >&2; exit 2; }
cfg_get "$CFG" name > /dev/null || { echo "$SCRIPT: $CFG does not load" >&2; exit 2; }
DATA_VERSION=$(cfg_get "$CFG" data_version)
[ -f "data/$DATA_VERSION/manifest.json" ] || {
  echo "$SCRIPT: data/$DATA_VERSION/manifest.json is missing: unpack the dataset tarball from the development server (setup_gpu.sh)" >&2; exit 2; }
TMO=$(cfg_timeout_seconds "$CFG")

banner "stage $STAGE on $PILOT_PROVIDER/$PILOT_INSTANCE ($PILOT_GPU, \$$PILOT_PRICE_PER_HOUR/h); model $PILOT_MODEL"
ledger_add --event start --provider "$PILOT_PROVIDER" --instance "$PILOT_INSTANCE" --gpu "$PILOT_GPU" \
  --price-per-hour "$PILOT_PRICE_PER_HOUR" --stage "$STAGE" --run-names stage_a_profile \
  --purpose "stage A: profile_gpu.py on the actual GPU, projected cost of stages B and C"
trap finish EXIT

PROFILE_ARGS=(--config "$CFG" --out profile.json --price-per-hour "$PILOT_PRICE_PER_HOUR")
if [ -n "${PILOT_UPLOAD_CMD:-}" ]; then PROFILE_ARGS+=(--upload-cmd "$PILOT_UPLOAD_CMD"); fi
banner "$PROFILE_PY ${PROFILE_ARGS[*]} (timeout $(cfg_get "$CFG" timeout_minutes) min)"
set +e
timeout --kill-after=60 "$TMO" "$PYTHON" "$PROFILE_PY" "${PROFILE_ARGS[@]}"
rc=$?
set -e
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  ledger_note "stage_a_profile: profile_gpu.py timed out (exit $rc)" stage_a_profile; FAILED+=(stage_a_profile)
elif [ "$rc" -ne 0 ] || [ ! -f profile.json ]; then
  ledger_note "stage_a_profile: profile_gpu.py failed (exit $rc, profile.json $([ -f profile.json ] && echo present || echo absent))" stage_a_profile
  FAILED+=(stage_a_profile)
else
  DONE+=(stage_a_profile)
fi

if [ "${STAGE_A_ZERO_SHOT:-0}" = 1 ]; then run_zero_shot stage_b_english_s0_n2000 english json; fi

REPORT_RC=0
if [ -f profile.json ]; then
  banner "ledger.py report --ceiling ${PILOT_CEILING:-100} --profile profile.json"
  set +e
  "$PYTHON" ledger.py report --ceiling "${PILOT_CEILING:-100}" --profile profile.json
  REPORT_RC=$?
  set -e
  if [ "$REPORT_RC" -ne 0 ]; then
    ledger_note "stage A: ledger report exit $REPORT_RC: the projection or the spend exceeds the ceiling \$${PILOT_CEILING:-100}; do not start stage B without a decision"
    echo "$SCRIPT: !!! ceiling check failed (exit $REPORT_RC): DO NOT START STAGE B until the owner has decided" >&2
  fi
else
  REPORT_RC=1
  echo "$SCRIPT: no profile.json: no projection to check" >&2
fi
banner "stage $STAGE done: ${DONE[*]:-none}; failed: ${FAILED[*]:-none}"
exit "$REPORT_RC"
