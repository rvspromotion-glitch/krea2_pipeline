# syntax=docker/dockerfile:1.7
#
# Krea2 pipeline worker — RunPod serverless, models baked into the image.
#
# No network volume. A volume pins the endpoint to the one datacentre that holds
# it, which is what was starving this of GPUs; the workload is a weekly
# sequential batch, so paying a cold start once a week is the cheaper trade.
#
# The weights are not in the image either. ~18GB of models on top of torch does
# not build on a hosted GitHub runner — it has ~14GB free on / and a build needs
# room for the layers twice over. So the image carries ComfyUI, the custom nodes
# and the code (~8GB), and entrypoint.sh fetches the weights on cold start,
# skipping anything already there. See models.txt.
#
# What is left is still shaped for a small pull:
#
#   * Multi-stage. The CUDA *devel* toolkit (~5GB of nvcc, headers and static
#     libs) is needed to build a couple of wheels and for nothing at runtime, so
#     it stays in the builder and never ships.
#   * Application code last. Anything after a changed layer is rebuilt, so the
#     files that change every commit sit at the very end and a code-only change
#     re-pulls megabytes.

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.10
# All three pinned, as a set, and new enough for the ComfyUI pinned below.
#
# Two separate traps, both of which broke a build here:
#
#  * torchvision and torchaudio are compiled against a specific torch ABI. Pin
#    torch alone and pip resolves the other two to whatever is newest, giving
#    an image that installs cleanly and dies on `import torchaudio` with an
#    undefined symbol.
#  * Too *old* a torch fails differently. ComfyUI now imports comfy_kitchen at
#    startup, which registers custom ops annotated with PEP 585 generics
#    (`list[int]`); torch only learned to infer schemas from those after 2.6,
#    so 2.6 raises ValueError before ComfyUI can start at all.
#
# 2.11.0 is the newest torch with a matching torchaudio on this index.
# The base image's CUDA version is unrelated — the torch wheels carry their own
# CUDA runtime as nvidia-* packages, which is why cu126 wheels sit happily on a
# 12.4 base.
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126

# ═══════════════════════════════════════════════════════════════════════════
# builder — the venv, ComfyUI, the custom nodes. Needs a compiler; ships none.
# ═══════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG TORCH_INDEX
ARG COMFYUI_REF=v0.9.2

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
#
# The import check is not ceremony. A mismatched trio installs without
# complaint and only fails when something loads the shared library, which is
# minutes later and several layers down — this puts the failure on the line
# that caused it.
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
 && python -c "import torch, torchvision, torchaudio; \
print('torch', torch.__version__, '| torchvision', torchvision.__version__, \
      '| torchaudio', torchaudio.__version__)"

# These pins are not preferences. numpy 2 and mediapipe >0.10.14 both break the
# node set at import; letting pip resolve them freely produces an image that
# builds cleanly and fails on the first render.
COPY constraints.txt /build/constraints.txt
ENV PIP_CONSTRAINT=/build/constraints.txt
RUN pip install --no-cache-dir --prefer-binary -r /build/constraints.txt

# ComfyUI, pinned to a release rather than master.
#
# master moving under a pinned torch is what made this repo hard to build:
# every rebuild picked up whatever ComfyUI had merged that day, and a new
# core dependency (comfy_kitchen) needed a newer torch than was pinned. Two
# moving parts, and the failure surfaced as an unrelated-looking traceback.
# A tag fixes one of them.
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
# runtime — CUDA *base*, not devel. Torch's wheels bring their own CUDA libs.
# ═══════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 AS runtime

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    COMFYUI_PATH=/comfyui \
    MODELS_DIR=/comfyui/models \
    LORA_DIR=/comfyui/models/loras \
    PYTHONPATH=/app/src \
    # These graphs run several samplers back to back at a fixed resolution,
    # which fragments the caching allocator enough to OOM a 24GB card partway
    # through a carousel even though no single step is close to the limit.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # The rust downloader, for the multi-GB weights fetched on cold start.
    HF_HUB_ENABLE_HF_TRANSFER=1

# libgl1 and libglib are opencv's; libgomp is torch's. Nothing else is needed —
# the CUDA runtime arrives with the torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} libgl1 libglib2.0-0 libgomp1 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /comfyui  /comfyui

WORKDIR /app
COPY workflows/ /app/workflows/
COPY scripts/verify_nodes.py /app/scripts/verify_nodes.py

# The build fails here rather than the first job of the Sunday batch. Also
# prints which package supplies which node — the only reliable way to trim
# custom_nodes.txt. Set PRUNE_UNUSED_NODES=1 to delete the ones that supply
# nothing; off by default because a package can matter by patching something at
# import time, which no static check can see.
#
# VERIFY_NODES=0 is the escape hatch: a node package that insists on CUDA at
# import cannot be checked on a machine with no GPU, and that should cost the
# check rather than the whole build.
ARG PRUNE_UNUSED_NODES=0
ARG VERIFY_NODES=1
RUN if [ "${VERIFY_NODES}" = "1" ]; then \
      python /app/scripts/verify_nodes.py \
        $([ "${PRUNE_UNUSED_NODES}" = "1" ] && echo --prune); \
    else \
      echo "[verify] skipped (VERIFY_NODES=0)"; \
    fi

# Last, so a code change rebuilds and re-pulls only these.
COPY models.txt /app/models.txt
COPY scripts/fetch_model.sh scripts/fetch_models.sh /app/scripts/
COPY src/ /app/src/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/scripts/fetch_model.sh /app/scripts/fetch_models.sh

CMD ["/app/entrypoint.sh"]
