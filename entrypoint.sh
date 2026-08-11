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
# rather than a load. Fetching is skip-if-present, so a warm worker pays nothing.
/app/scripts/fetch_models.sh

# ComfyUI is started WITHOUT /app/src on the path.
#
# Our modules live there and ComfyUI imports by bare name from its own root, so
# anything of ours that shares a name with one of its top-level modules wins.
# That is not hypothetical: src/comfy.py shadowed ComfyUI's `comfy` package and
# main.py died on its first line with "'comfy' is not a package". ComfyUI never
# started, the container stayed up, and every job spent ten minutes waiting for
# a server that was never coming.
#
# --disable-metadata is not optional either. SaveImage otherwise writes the whole
# API prompt into the PNG's tEXt chunks, which carries the Ask_Gemini_Batch node
# and therefore the Gemini API key in plaintext, plus the character LoRA filename
# and the full system prompt. These images get published.
echo "[entrypoint] starting ComfyUI from ${COMFY_DIR}"
cd "${COMFY_DIR}"
env -u PYTHONPATH python3 main.py --listen 127.0.0.1 --port 8188 \
  --disable-auto-launch --disable-metadata &
COMFY_PID=$!

echo "[entrypoint] starting handler"
cd /app
PYTHONPATH=/app/src python3 -u src/handler.py &
HANDLER_PID=$!

# Whichever of the two exits first takes the container down, so RunPod replaces
# it. Both are load-bearing: a handler with no ComfyUI fails every job it is
# given, and ComfyUI with no handler is invisible to RunPod.
#
# `wait -n` in *this* shell, not a subshell — a subshell cannot wait on a
# process it did not start, which is why the previous version printed
# "pid N is not a child of this shell" and then supervised nothing.
wait -n "$COMFY_PID" "$HANDLER_PID"
status=$?

if ! kill -0 "$COMFY_PID" 2>/dev/null; then
  echo "[fatal] ComfyUI exited (status ${status}) — stopping worker"
else
  echo "[fatal] handler exited (status ${status}) — stopping worker"
fi

kill -TERM "$COMFY_PID" "$HANDLER_PID" 2>/dev/null || true
exit "$status"
