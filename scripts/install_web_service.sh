#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SOURCE="$ROOT_DIR/deploy/posture-coach-web.service"
SERVICE_TARGET="/etc/systemd/system/posture-coach-web.service"

sudo install -m 0644 "$SERVICE_SOURCE" "$SERVICE_TARGET"
sudo systemctl daemon-reload
sudo systemctl enable --now posture-coach-web.service
sudo systemctl --no-pager --full status posture-coach-web.service

