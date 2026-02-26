#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy pillow torch torchvision
python -m pip install git+https://github.com/facebookresearch/sam2.git

echo "Local free setup complete."
echo "Next: set SAM2_LOCAL_CONFIG and SAM2_LOCAL_CHECKPOINT, then run scripts/run_local_free.sh"
