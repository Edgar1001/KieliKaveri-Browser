#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/uvicorn" ]]; then
  echo "Local Python environment is missing. Run: npm run local:setup"
  exit 1
fi

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

CUBLAS_PATH=$(find "$VENV_DIR/lib" -name 'libcublas.so.12' -print -quit)
CUDNN_PATH=$(find "$VENV_DIR/lib" -name 'libcudnn.so.9' -print -quit)

if [[ -n "$CUBLAS_PATH" && -n "$CUDNN_PATH" ]]; then
  CUDA_LIBRARY_PATH="$(dirname "$CUBLAS_PATH"):$(dirname "$CUDNN_PATH")"
  export LD_LIBRARY_PATH="$CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$ROOT_DIR"
exec "$VENV_DIR/bin/uvicorn" local_server.main:app --host 0.0.0.0 --port "${PORT:-8787}"