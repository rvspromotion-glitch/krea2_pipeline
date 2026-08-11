# syntax=docker/dockerfile:1.7
#
# Krea2 pipeline worker — RunPod serverless, models baked into the image.
#
# No network volume. A volume pins the endpoint to the one datacentre that holds
# it, which is what was starving this of GPUs; the workload is a weekly
# sequential batch, so paying a cold start once a week is the cheaper trade.
#
# Everything below exists to make that cold start as small as possible:
#
#   * Multi-stage. The CUDA *devel* toolkit (~5GB of nvcc, headers and static
#     libs) is needed to build a couple of wheels and for nothing at runtime, so
#     it stays in the builder and never ships.
#   * One layer per model. Layers are pulled and cached individually, so a code
#     change re-pulls a few MB rather than 18GB, and swapping one LoRA does not
#     invalidate the checkpoint.
#   * Application code last. Anything after a changed layer is rebuilt, so the
#     files that change every commit sit at the very end.
#
# Rebuilding after a code-only change touches the final three layers.

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.4.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

# ═══════════════════════════════════════════════════════════════════════════
# builder — the venv, ComfyUI, the custom nodes. Needs a compiler; ships none.
# ═══════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG TORCH_INDEX
ARG COMFYUI_REF=master

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
        python3-pip build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Torch first and alone: it is the single largest dependency and the one least
# likely to change, so it earns its own layer.
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} \
        torch==${TORCH_VERSION} torchvision torchaudio

# These pins are not preferences. numpy 2 and mediapipe >0.10.14 both break the
# node set at import; letting pip resolve them freely produces an image that
# builds cleanly and fails on the first render.
COPY constraints.txt /build/constraints.txt
ENV PIP_CONSTRAINT=/build/constraints.txt
RUN pip install --no-cache-dir --prefer-binary -r /build/constraints.txt

# ComfyUI. Pin COMFYUI_REF to a sha once a build is green — `master` is a moving
# target and a silent upstream change is the hardest kind of Sunday failure.
RUN git clone --depth 1 --branch ${COMFYUI_REF} \
        https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && rm -rf /comfyui/.git \
    && pip install --no-cache-dir --prefer-binary -r /comfyui/requirements.txt

COPY custom_nodes.txt /build/custom_nodes.txt
COPY scripts/install_nodes.sh /build/install_nodes.sh
RUN chmod +x /build/install_nodes.sh \
    && /build/install_nodes.sh /build/custom_nodes.txt /comfyui/custom_nodes

# The handler's own dependencies.
COPY requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir -r /build/requirements.txt

# Bytecode for the whole tree, so the first import on a cold worker is not also
# a compile. Failures here are not fatal — some node packages ship py2 relics.
RUN python -m compileall -q /opt/venv/lib /comfyui || true

# ═══════════════════════════════════════════════════════════════════════════
# models — downloaded once, copied out one file at a time
# ═══════════════════════════════════════════════════════════════════════════
FROM ubuntu:22.04 AS models

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "huggingface_hub" "hf_transfer"

COPY scripts/fetch_model.sh /build/fetch_model.sh
RUN chmod +x /build/fetch_model.sh

# Tokens arrive as BuildKit secrets, which are mounted for the life of the RUN
# and never written to a layer. A build ARG would be visible in this stage's
# history for anyone who pulls it.
#
# Only what these two graphs load. The UNETLoader that pulled krea2_raw was
# pruned as dead, so the AiO checkpoint does that work alone. Largest first —
# it is the least likely to change, and layers below a changed one are rebuilt.
RUN --mount=type=secret,id=civitai_token \
    /build/fetch_model.sh civit \
      "https://civitai.red/api/download/models/3107962?fileId=2996137" \
      /models/checkpoints/AiO_krea2_checkpoint_int8_8steps.safetensors
RUN --mount=type=secret,id=hf_token \
    /build/fetch_model.sh hf "Comfy-Org/Krea-2" \
      "text_encoders/qwen3vl_4b_fp8_scaled.safetensors" \
      /models/clip/qwen3vl_4b_fp8_scaled.safetensors
RUN --mount=type=secret,id=hf_token \
    /build/fetch_model.sh hf "Kijai/WanVideo_comfy" \
      "Wan2_1_VAE_fp32.safetensors" \
      /models/vae/Wan2_1_VAE_fp32.safetensors
RUN --mount=type=secret,id=hf_token \
    /build/fetch_model.sh hf "conradlocke/krea2-identity-edit" \
      "krea2_identity_edit_v1_2.safetensors" \
      /models/loras/krea2_identity_edit_v1_2.safetensors
RUN --mount=type=secret,id=civitai_token \
    /build/fetch_model.sh civit \
      "https://civitai.red/api/download/models/3075606?fileId=2954661" \
      /models/loras/Lenovo_ultrareal.safetensors

# ═══════════════════════════════════════════════════════════════════════════
# runtime — CUDA *base*, not devel. Torch's wheels bring their own CUDA libs.
# ═══════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 AS runtime

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    COMFYUI_PATH=/comfyui \
    LORA_DIR=/comfyui/models/loras \
    PYTHONPATH=/app/src \
    # These graphs run several samplers back to back at a fixed resolution,
    # which fragments the caching allocator enough to OOM a 24GB card partway
    # through a carousel even though no single step is close to the limit.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Every model these graphs need is in the image. A node reaching for the Hub
    # at render time means something is missing, and the whole point of baking
    # the models in is that a cold start does not download gigabytes — so fail
    # loudly here rather than quietly paying for it on every job.
    HF_HUB_OFFLINE=1

# libgl1 and libglib are opencv's; libgomp is torch's. Nothing else is needed —
# the CUDA runtime arrives with the torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} libgl1 libglib2.0-0 libgomp1 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /comfyui  /comfyui

# One COPY per model = one layer per model.
COPY --from=models /models/checkpoints/AiO_krea2_checkpoint_int8_8steps.safetensors /comfyui/models/checkpoints/
COPY --from=models /models/clip/qwen3vl_4b_fp8_scaled.safetensors                   /comfyui/models/clip/
COPY --from=models /models/vae/Wan2_1_VAE_fp32.safetensors                          /comfyui/models/vae/
COPY --from=models /models/loras/krea2_identity_edit_v1_2.safetensors               /comfyui/models/loras/
COPY --from=models /models/loras/Lenovo_ultrareal.safetensors                       /comfyui/models/loras/

WORKDIR /app
COPY workflows/ /app/workflows/
COPY scripts/verify_nodes.py /app/scripts/verify_nodes.py

# The build fails here rather than the first job of the Sunday batch. Also
# prints which package supplies which node — the only reliable way to trim
# custom_nodes.txt. Set PRUNE_UNUSED_NODES=1 to delete the ones that supply
# nothing; off by default because a package can matter by patching something at
# import time, which no static check can see.
ARG PRUNE_UNUSED_NODES=0
RUN python /app/scripts/verify_nodes.py \
      $([ "${PRUNE_UNUSED_NODES}" = "1" ] && echo --prune)

# Last, so a code change rebuilds and re-pulls only these.
COPY src/ /app/src/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
