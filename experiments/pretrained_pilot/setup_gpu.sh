#!/usr/bin/env bash
# setup_gpu.sh - prepare a freshly rented GPU machine for the pilot (run once per rental).
#
#   1. a Python venv with requirements-gpu.txt (torch from the CUDA wheel index first);
#   2. the model weights: downloaded with the Hugging Face CLI ONLY when PILOT_ALLOW_DOWNLOAD=1
#      (Qwen/Qwen3-0.6B, Apache-2.0, about 1.2 GB of bf16 safetensors plus the tokenizer), or found
#      already under $PILOT_MODEL_ROOT (uploaded by the owner);
#   3. the frozen dataset data/v1: unpacked from the owner's tarball (PILOT_DATA_TARBALL), never
#      regenerated here, because data.py needs the study packet, which stays on the development
#      server;
#   4. a check that the venv imports, the GPU is visible, the tokenizer's chat template matches
#      common.PROMPT_TEMPLATE, and every dataset file matches the manifest's SHA-256;
#   5. the `export PILOT_TOKENIZER=... PILOT_MODEL=...` lines to paste before run_stage_*.sh.
#
# Variables (all optional):
#   PILOT_VENV            venv directory (default $HOME/pilot-venv)
#   PILOT_TORCH_CUDA      CUDA wheel tag for torch (default cu128; cu126 for drivers below 570)
#   PILOT_USE_IMAGE_TORCH 1: create the venv with --system-site-packages and keep the image's torch
#   PILOT_MODEL_ROOT      where the weights live (default $HOME/models)
#   PILOT_HF_MODEL        Hub id to download (default Qwen/Qwen3-0.6B; Qwen/Qwen3-1.7B for the extension)
#   PILOT_ALLOW_DOWNLOAD  1: allow the download above (the only way any pilot script downloads weights)
#   PILOT_DATA_TARBALL    path of the dataset tarball made on the development server with
#                           tar czf pilot_data_v1.tgz -C experiments/pretrained_pilot data/v1
#   PILOT_DATA_VERSION    dataset version expected under data/ (default v1)
#
# No provider API is called; no token is read or written. The provider token, if the owner needs
# one for the CLI, lives in PILOT_PROVIDER_TOKEN on the owner's shell only (verify_teardown.md).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRIPT=$(basename "${BASH_SOURCE[0]}")
VENV=${PILOT_VENV:-$HOME/pilot-venv}
TORCH_CUDA=${PILOT_TORCH_CUDA:-cu128}
MODEL_ROOT=${PILOT_MODEL_ROOT:-$HOME/models}
HF_MODEL=${PILOT_HF_MODEL:-Qwen/Qwen3-0.6B}
MODEL_DIR="$MODEL_ROOT/${HF_MODEL##*/}"
DATA_VERSION=${PILOT_DATA_VERSION:-v1}
say() { printf '\n== %s\n' "$*"; }

say "machine"
uname -a
if command -v nvidia-smi > /dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "$SCRIPT: WARNING: nvidia-smi not found; is this a GPU machine?" >&2
fi
df -h "$HOME" "$HERE" | sed -n '1p;2p;3p'

# ---- 1. venv ---------------------------------------------------------------------------------
say "venv at $VENV"
SYS_PY=${PILOT_SYSTEM_PYTHON:-python3}
"$SYS_PY" --version
if [ ! -x "$VENV/bin/python" ]; then
  if [ "${PILOT_USE_IMAGE_TORCH:-0}" = 1 ]; then
    "$SYS_PY" -m venv --system-site-packages "$VENV"
  else
    "$SYS_PY" -m venv "$VENV"
  fi
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip
if [ "${PILOT_USE_IMAGE_TORCH:-0}" = 1 ] && "$PY" -c 'import torch; assert torch.cuda.is_available()' 2> /dev/null; then
  echo "keeping the image's torch $("$PY" -c 'import torch; print(torch.__version__)')"
else
  echo "installing torch from https://download.pytorch.org/whl/$TORCH_CUDA (set PILOT_TORCH_CUDA to match the driver)"
  "$PY" -m pip install --quiet 'torch>=2.6,<2.10' --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
fi
"$PY" -m pip install --quiet -r requirements-gpu.txt
"$PY" - <<'EOF'
import torch, transformers, peft, accelerate
print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}'
      + (f' ({torch.cuda.get_device_name(0)}, bf16 supported={torch.cuda.is_bf16_supported()})' if torch.cuda.is_available() else ''))
print(f'transformers {transformers.__version__}, peft {peft.__version__}, accelerate {accelerate.__version__}')
if not torch.cuda.is_available():
    raise SystemExit('setup_gpu.sh: torch does not see a GPU; wrong CUDA wheel (PILOT_TORCH_CUDA) or no GPU')
EOF

# ---- 2. model and tokenizer ------------------------------------------------------------------
say "model $HF_MODEL under $MODEL_DIR"
if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/tokenizer.json" ]; then
  echo "already present: $(du -sh "$MODEL_DIR" | cut -f1)"
elif [ "${PILOT_ALLOW_DOWNLOAD:-0}" = 1 ]; then
  mkdir -p "$MODEL_ROOT"
  if "$VENV/bin/hf" --help > /dev/null 2>&1; then
    "$VENV/bin/hf" download "$HF_MODEL" --local-dir "$MODEL_DIR"
  else
    "$VENV/bin/huggingface-cli" download "$HF_MODEL" --local-dir "$MODEL_DIR"
  fi
  echo "downloaded $(du -sh "$MODEL_DIR" | cut -f1) ($HF_MODEL is Apache-2.0; about 1.2 GB for the 0.6B in bf16 safetensors)"
else
  cat >&2 <<EOF
$SCRIPT: no weights at $MODEL_DIR and PILOT_ALLOW_DOWNLOAD is not 1.
  Either upload the model directory there (config.json, model.safetensors, tokenizer.json,
  tokenizer_config.json, generation_config.json) or re-run with
    PILOT_ALLOW_DOWNLOAD=1 bash setup_gpu.sh
EOF
  exit 2
fi
for f in config.json tokenizer.json tokenizer_config.json; do
  [ -f "$MODEL_DIR/$f" ] || { echo "$SCRIPT: $MODEL_DIR/$f is missing" >&2; exit 2; }
done
compgen -G "$MODEL_DIR/*.safetensors" > /dev/null || { echo "$SCRIPT: no *.safetensors under $MODEL_DIR" >&2; exit 2; }

# ---- 3. dataset ------------------------------------------------------------------------------
say "dataset data/$DATA_VERSION"
if [ -n "${PILOT_DATA_TARBALL:-}" ]; then
  [ -f "$PILOT_DATA_TARBALL" ] || { echo "$SCRIPT: PILOT_DATA_TARBALL=$PILOT_DATA_TARBALL not found" >&2; exit 2; }
  tar tzf "$PILOT_DATA_TARBALL" | grep -q "^data/$DATA_VERSION/manifest.json$" || {
    echo "$SCRIPT: the tarball holds no data/$DATA_VERSION/manifest.json (make it with: tar czf pilot_data_$DATA_VERSION.tgz -C experiments/pretrained_pilot data/$DATA_VERSION)" >&2; exit 2; }
  tar xzf "$PILOT_DATA_TARBALL" -C "$HERE" "data/$DATA_VERSION"
  echo "unpacked $PILOT_DATA_TARBALL"
fi
if [ ! -f "data/$DATA_VERSION/manifest.json" ]; then
  cat >&2 <<EOF
$SCRIPT: data/$DATA_VERSION/manifest.json is missing. The dataset is generated on the development
  server only (data.py needs the study packet). There:
    cd <repo> && tar czf pilot_data_$DATA_VERSION.tgz -C experiments/pretrained_pilot data/$DATA_VERSION
  copy the tarball here and re-run with PILOT_DATA_TARBALL=<path> bash setup_gpu.sh
EOF
  exit 2
fi
"$PY" - "$DATA_VERSION" <<'EOF'
import json, sys
import common
version = sys.argv[1]
root = common.data_dir(version)
with open(root / 'manifest.json', encoding='utf-8') as f:
    manifest = json.load(f)
hashes = manifest.get('sha256') or {}
if not isinstance(hashes, dict) or not hashes:
    raise SystemExit(f'setup_gpu.sh: {root}/manifest.json has no "sha256" map; cannot verify the upload')
bad = []
for name, expected in sorted(hashes.items()):
    path = root / name
    if not path.exists():
        bad.append(f'{name}: missing')
    elif common.sha256_file(path) != expected:
        bad.append(f'{name}: sha256 mismatch')
if bad:
    raise SystemExit('setup_gpu.sh: dataset does not match its manifest:\n  ' + '\n  '.join(bad))
print(f'{len(hashes)} files match manifest.json (version {manifest.get("version", version)}, '
      f'generated {manifest.get("generated_at", "?")})')
EOF

# ---- 4. the chat template of this tokenizer is the one common.py hard-codes --------------------
say "tokenizer check"
PILOT_TOKENIZER="$MODEL_DIR" "$PY" - <<'EOF'
import common
tok = common.load_tokenizer()
common.check_template(tok)
print(f'chat template matches common.PROMPT_TEMPLATE; vocab {len(tok)}; pad={tok.pad_token!r}')
EOF

# ---- 5. what to paste -------------------------------------------------------------------------
say "ready. Paste these before the stage scripts (and set the rental's facts for the ledger):"
cat <<EOF
export PYTHON=$PY
export PILOT_TOKENIZER=$MODEL_DIR
export PILOT_MODEL=$MODEL_DIR
export PILOT_PROVIDER=<runpod|vast|modal|lightning|thunder>  PILOT_INSTANCE=<instance id>  PILOT_GPU="<card>"  PILOT_PRICE_PER_HOUR=<USD/h as quoted>
export PILOT_CEILING=100
# then: bash run_stage_a.sh 2>&1 | tee stage_a.log      (and verify_teardown.md when it ends)
EOF
