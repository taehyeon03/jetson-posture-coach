#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"
MODEL_PATH="$MODEL_DIR/movenet_singlepose_lightning.tflite"
TFLITE_WHEEL="https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl"
MODEL_URL="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This installer is for the Jetson Nano aarch64 board." >&2
  exit 2
fi

# --no-deps is intentional. A current PyPI NumPy wheel uses CPU instructions
# unavailable on the Nano's Cortex-A57; Ubuntu's board-native NumPy works.
python3 -m pip install --user --no-deps --force-reinstall "$TFLITE_WHEEL"
python3 -c "import numpy, tflite_runtime.interpreter; print('numpy=' + numpy.__version__ + ' tflite=ok')"

mkdir -p "$MODEL_DIR"
wget -q --show-progress -O "$MODEL_PATH.tmp" "$MODEL_URL"
mv "$MODEL_PATH.tmp" "$MODEL_PATH"
echo "Installed model: $MODEL_PATH"

