#!/usr/bin/env bash
# Board bring-up: apt packages needed for camera capture on Jetson Nano (L4T R32.7.6 / Ubuntu 18.04).
# Run on the Nano itself, not the dev PC. Requires sudo.
#
# python3-opencv here is the stock Ubuntu 18.04 build (3.2.0, no GStreamer, no CUDA) — it is
# used only for saving images (cv2.imwrite) and later CPU-side pre/post-processing. Frame
# capture goes through GStreamer's appsink via PyGObject (python3-gi, already present on this
# image), not through cv2.VideoCapture, because the CSI sensor outputs raw Bayer (RG10) and
# needs the nvarguscamerasrc ISP pipeline.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-opencv \
    v4l-utils
