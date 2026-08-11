#!/usr/bin/env bash
# Boot ComfyUI against the network volume, then hand over to the RunPod handler.
set -euo pipefail

VOLUME="${RUNPOD_VOLUME:-/runpod-volume}"
COMFY_DIR="${COMFYUI_PATH:-${VOLUME}/ComfyUI}"

# Opt-in one-off seeding, so the same image can populate an empty volume without
# a separate pod. Never runs by default — it would add ~40GB to every cold start.
if [ "${RUN_SETUP:-0}" = "1" ]; then
  echo "[entrypoint] RUN_SETUP=1 — seeding volume"
  /app/setup_volume.sh
fi

if [ ! -f "${COMFY_DIR}/main.py" ]; then
  echo "[fatal] ComfyUI not found at ${COMFY_DIR}."
  echo "        Seed the network volume first: RUN_SETUP=1, or run setup_volume.sh from a pod."
  exit 1
fi

export PYTHONPATH="/app/src:${PYTHONPATH:-}"

# --disable-metadata is not optional here. SaveImage otherwise writes the whole
# API prompt into the PNG's tEXt chunks — which includes the Ask_Gemini_Batch
# node, and therefore the Gemini API key in plaintext, plus the character LoRA
# filename and the full system prompt. Those images get published.
echo "[entrypoint] starting ComfyUI from ${COMFY_DIR}"
cd "${COMFY_DIR}"
python3 main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-metadata &
COMFY_PID=$!

# If ComfyUI dies the worker is useless — take the container down so RunPod
# replaces it, rather than serving jobs that will all fail to connect.
trap 'kill -TERM "$COMFY_PID" 2>/dev/null || true' EXIT
(
  wait "$COMFY_PID"
  echo "[fatal] ComfyUI exited — stopping worker"
  kill -TERM 1 2>/dev/null || true
) &

echo "[entrypoint] starting handler"
cd /app
exec python3 -u src/handler.py
