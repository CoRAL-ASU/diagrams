#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run scripts/setup_local_free.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

: "${ANNOTATION_DATA_ROOTS:=/Users/tampu/Documents/Reviewed}"
: "${SAM2_LOCAL_DEVICE:=cpu}"
: "${SAM2_LOCAL_MAX_BOXES:=8}"

if [[ -z "${SAM2_LOCAL_CONFIG:-}" ]]; then
  echo "Set SAM2_LOCAL_CONFIG to your SAM2 config file path." >&2
  exit 1
fi

if [[ -z "${SAM2_LOCAL_CHECKPOINT:-}" ]]; then
  echo "Set SAM2_LOCAL_CHECKPOINT to your SAM2 checkpoint (.pt) path." >&2
  exit 1
fi

export SAM2_BACKEND="local"
export ANNOTATION_DATA_ROOTS
export SAM2_LOCAL_CONFIG
export SAM2_LOCAL_CHECKPOINT
export SAM2_LOCAL_DEVICE
export SAM2_LOCAL_MAX_BOXES

python3 server.py
