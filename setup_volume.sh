#!/usr/bin/env bash
# Seed the RunPod network volume with ComfyUI, custom nodes and models.
#
# Derived from the pod start.sh this workflow was built with, but split off it
# deliberately: on serverless, downloading ~40GB at container start would be
# paid on every cold start. Run this ONCE against the volume (from a pod, or a
# one-off serverless job with RUN_SETUP=1); afterwards the worker boots straight
# into ComfyUI with everything already resident.
#
# Every fetch is skip-if-present, so re-running it to add a model is cheap.
set -euo pipefail

VOLUME="${RUNPOD_VOLUME:-/runpod-volume}"
COMFY_DIR="${COMFYUI_PATH:-${VOLUME}/ComfyUI}"
CUSTOM_NODES="${COMFY_DIR}/custom_nodes"
MODELS_DIR="${COMFY_DIR}/models"
REPO_CACHE="${VOLUME}/_repos"

echo "=== Seeding ${VOLUME} ==="
mkdir -p "$COMFY_DIR" "$CUSTOM_NODES" "$REPO_CACHE" \
  "${MODELS_DIR}"/{checkpoints,clip,clip_vision,diffusion_models,loras,vae,depthanything}

# ── ComfyUI itself ───────────────────────────────────────────────────────────
if [ ! -f "${COMFY_DIR}/main.py" ]; then
  echo "[setup] cloning ComfyUI..."
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "${COMFY_DIR}.tmp"
  cp -a "${COMFY_DIR}.tmp/." "${COMFY_DIR}/"
  rm -rf "${COMFY_DIR}.tmp"
fi
pip install -q -r "${COMFY_DIR}/requirements.txt" || true

# ── Pinned deps ──────────────────────────────────────────────────────────────
# numpy<2 and mediapipe==0.10.14 are hard requirements of the node set; letting
# pip resolve them freely breaks LayerStyle and Impact-Pack at import time.
CONSTRAINTS="${VOLUME}/pip-constraints.txt"
cat > "$CONSTRAINTS" <<'EOF'
numpy<2
protobuf<5
opencv-python<4.12
transformers>=4.39.3
mediapipe==0.10.14
sageattention
EOF
export PIP_CONSTRAINT="$CONSTRAINTS"
pip install -q --prefer-binary -c "$CONSTRAINTS" \
  "numpy<2" "protobuf<5" "opencv-python<4.12" "mediapipe==0.10.14" "sageattention" || true

pip install -q -U "huggingface_hub[hf_xet]" || true
export HF_XET_HIGH_PERFORMANCE=1

# ── Helpers ──────────────────────────────────────────────────────────────────
hf_download() {
  local repo="$1" remote="$2" out="$3"
  if [ -f "$out" ] && [ -s "$out" ]; then echo "[hf] exists: $(basename "$out")"; return 0; fi
  echo "[hf] $repo/$remote"
  local tmp; tmp="$(mktemp -d)"
  if hf download "$repo" "$remote" --local-dir "$tmp"; then
    mkdir -p "$(dirname "$out")"; mv "${tmp}/${remote}" "$out"
  else
    echo "[hf] WARNING: failed $repo/$remote"
  fi
  rm -rf "$tmp"
}

civit_download() {
  local url="$1" out="$2"
  if [ -f "$out" ] && [ -s "$out" ]; then echo "[civitai] exists: $(basename "$out")"; return 0; fi
  echo "[civitai] $(basename "$out")"
  local dl="$url"
  if [ -n "${CIVITAI_TOKEN:-}" ]; then
    dl=$(curl -sL -I -H "Authorization: Bearer ${CIVITAI_TOKEN}" "$url" \
      | grep -i "^location:" | tail -1 | sed 's/^location: //i' | tr -d '\r\n') || dl="$url"
    [ -z "$dl" ] && dl="$url"
  fi
  curl -L --fail --retry 8 --retry-delay 2 -C - \
    -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
    -o "$out" "$dl"
  # A gated/unauthenticated fetch returns an HTML login page, which would sit
  # on the volume looking like a model until inference fails cryptically.
  if command -v file >/dev/null 2>&1 && file "$out" | grep -qi HTML; then
    echo "[civitai] ERROR: got HTML not a model — removing $out"; rm -f "$out"; return 1
  fi
}

tarball_fetch() {
  local owner_repo="$1" ref="$2" name="$3"
  local dest="${REPO_CACHE}/${name}"
  if [ -f "${dest}/.fetched_ok" ]; then echo "[nodes] exists: ${name}"; return 0; fi
  echo "[nodes] ${name}"
  local tmp; tmp="$(mktemp --suffix=.tar.gz)"
  curl -L --fail --retry 5 -o "$tmp" "https://codeload.github.com/${owner_repo}/tar.gz/${ref}"
  rm -rf "$dest"; mkdir -p "$dest"
  tar -xzf "$tmp" -C "$dest" --strip-components=1
  rm -f "$tmp"; touch "${dest}/.fetched_ok"
}

# ── Custom nodes ─────────────────────────────────────────────────────────────
# Same set the workflows were authored against. BatchnodeI9 and savezipi9 are
# first-party and ship nested subpackages, hence the special-cased symlinking.
for repo in \
  "rgthree-comfy:rgthree/rgthree-comfy:HEAD" \
  "ComfyUI-Easy-Use:yolain/ComfyUI-Easy-Use:HEAD" \
  "ComfyUI_LayerStyle:chflame163/ComfyUI_LayerStyle:HEAD" \
  "ComfyUI-DepthAnythingV2:kijai/ComfyUI-DepthAnythingV2:HEAD" \
  "ComfyUI-KJNodes:kijai/ComfyUI-KJNodes:HEAD" \
  "comfyui-krea2-conditioning:huwhitememes/comfyui-krea2-conditioning:HEAD" \
  "ComfyUI-NovaNoiser:Aloukik21/ComfyUI-NovaNoiser:HEAD" \
  "krea-reference:kgilper/krea-reference:HEAD" \
  "ComfyUI-Anima-PiD:sorryhyun/ComfyUI-Anima-PiD:HEAD" \
  "comfyui-krea2edit:lbouaraba/comfyui-krea2edit:HEAD" \
  "ComfyUI-Impact-Pack:ltdrdata/ComfyUI-Impact-Pack:HEAD" \
  "comfyui-realisim-enhancor:amrnidal999-tech/comfyui-realisim-enhancor:HEAD" \
  "ComfyUI-GridSplit:workordie/ComfyUI-GridSplit:b9941964ff879487aa3e9433b174548039748453" \
  "BatchnodeI9:rvspromotion-glitch/BatchnodeI9:HEAD" \
  "savezipi9:rvspromotion-glitch/savezipi9:HEAD" \
  "RES4LYF:ClownsharkBatwing/RES4LYF:HEAD"
do
  name="${repo%%:*}"; rest="${repo#*:}"
  tarball_fetch "${rest%%:*}" "${rest#*:}" "$name" || echo "[nodes] WARNING: $name failed"
done

echo "[nodes] linking..."
for dir in "${REPO_CACHE}"/*; do
  [ -d "$dir" ] || continue
  name="$(basename "$dir")"
  case "$name" in
    savezipi9|BatchnodeI9)
      for sub in "$dir"/*; do
        [ -d "$sub" ] || continue
        ln -sfn "$sub" "${CUSTOM_NODES}/$(basename "$sub")"
      done ;;
    *) ln -sfn "$dir" "${CUSTOM_NODES}/${name}" ;;
  esac
done

echo "[nodes] installing node requirements..."
for dir in "${REPO_CACHE}"/*; do
  [ -d "$dir" ] || continue
  for req in "$dir/requirements.txt" "$dir"/*/requirements.txt; do
    [ -f "$req" ] || continue
    tmp="$(mktemp)"
    grep -viE '^(torch|torchvision|torchaudio|numpy|transformers|tokenizers|protobuf)([<=> ].*)?$' \
      "$req" > "$tmp" || true
    pip install -q --prefer-binary -c "$CONSTRAINTS" -r "$tmp" 2>/dev/null || true
    rm -f "$tmp"
  done
done

# ── Models ───────────────────────────────────────────────────────────────────
# Only what these two graphs actually load. The UNETLoader node was pruned as
# dead, so krea2_raw is not fetched — the AiO checkpoint does the work.
hf_download "Comfy-Org/Krea-2" "text_encoders/qwen3vl_4b_fp8_scaled.safetensors" \
  "${MODELS_DIR}/clip/qwen3vl_4b_fp8_scaled.safetensors"
hf_download "Kijai/WanVideo_comfy" "Wan2_1_VAE_fp32.safetensors" \
  "${MODELS_DIR}/vae/Wan2_1_VAE_fp32.safetensors"
hf_download "conradlocke/krea2-identity-edit" "krea2_identity_edit_v1_2.safetensors" \
  "${MODELS_DIR}/loras/krea2_identity_edit_v1_2.safetensors"

civit_download "https://civitai.red/api/download/models/3107962?fileId=2996137" \
  "${MODELS_DIR}/checkpoints/AiO_krea2_checkpoint_int8_8steps.safetensors"
civit_download "https://civitai.red/api/download/models/3075606?fileId=2954661" \
  "${MODELS_DIR}/loras/Lenovo_ultrareal.safetensors"

# Character LoRAs are per-persona and seeded separately — either dropped on the
# volume by hand, or fetched on first use via the job's lora_url.
if [ -n "${CHAR_LORA_URL:-}" ]; then
  name="$(basename "${CHAR_LORA_URL}" | sed 's/\?.*$//')"
  curl -L --fail -o "${MODELS_DIR}/loras/${name}" "${CHAR_LORA_URL}"
fi

echo "=== Volume ready ==="
find "${MODELS_DIR}" -name '*.safetensors' -printf '  %-70p %10s bytes\n' 2>/dev/null | sort
