#!/usr/bin/env bash
# Boot ComfyUI, then hand over to the RunPod handler.
set -euo pipefail

COMFY_DIR="${COMFYUI_PATH:-/comfyui}"

if [ ! -f "${COMFY_DIR}/main.py" ]; then
  echo "[fatal] ComfyUI not found at ${COMFY_DIR}. The image is broken —"
  echo "        rebuild it; there is nothing to configure at runtime."
  exit 1
fi

# Weights first, ComfyUI second. ComfyUI caches its model folder listing, and a
# checkpoint that appears after boot is a validation error on the first job
# rather than a load — not worth the ~60s of overlap this would buy on a run
# that lasts hours. Fetching is skip-if-present, so a warm worker pays nothing.
/app/scripts/fetch_models.sh

# ComfyUI is started first and in the background, so it loads while RunPod is
# still bringing the worker up. wait_until_ready in the handler covers the rest:
# the first job blocks until ComfyUI answers instead of failing to connect.
#
# --disable-metadata is not optional. SaveImage otherwise writes the whole API
# prompt into the PNG's tEXt chunks, which carries the Ask_Gemini_Batch node and
# therefore the Gemini API key in plaintext, plus the character LoRA filename and
# the full system prompt. These images get published.
echo "[entrypoint] starting ComfyUI from ${COMFY_DIR}"
cd "${COMFY_DIR}"
python3 main.py --listen 127.0.0.1 --port 8188 \
  --disable-auto-launch --disable-metadata &
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
