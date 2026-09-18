#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

python3 -m venv "$ROOT_DIR/.venv"
"$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$ROOT_DIR/.venv/bin/pip" install -r "$ROOT_DIR/requirements.txt"

VOICE_DIR="$ROOT_DIR/local_server/voices"
VOICE_NAME="fi_FI-harri-medium.onnx"
mkdir -p "$VOICE_DIR"
if [[ ! -f "$VOICE_DIR/$VOICE_NAME" ]]; then
  curl --fail --location \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/fi/fi_FI/harri/medium/$VOICE_NAME" \
    --output "$VOICE_DIR/$VOICE_NAME"
  curl --fail --location \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/fi/fi_FI/harri/medium/$VOICE_NAME.json" \
    --output "$VOICE_DIR/$VOICE_NAME.json"
fi

echo "Local speech dependencies installed. Add OPENAI_API_KEY to .env before starting the API."