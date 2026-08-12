# Krea2 pipeline worker — RunPod serverless.
#
# Built on the same base as the detailer worker, which runs this node set in
# production. That is deliberate and it is the main lesson of getting this
# wrong three times: torch, torchvision, torchaudio, CUDA, cuDNN and Triton all
# have to agree, and every attempt to assemble that set by hand here produced a
# different mismatch — first an ABI break in torchaudio, then a torch too old
# for ComfyUI's own imports. RunPod publishes a combination that is already
# consistent. Do not pip install torch on top of it.
#
# Differs from the detailer in one way that matters: there is no network volume.
# The detailer keeps models, node repos and a pip cache on /runpod-volume; this
# endpoint has none, because a volume pins it to one datacentre and starves it
# of GPUs. So the nodes are installed at build time (below) and the weights are
# fetched on cold start by entrypoint.sh (see models.txt).
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Pin to a release, not master: a moving ComfyUI under a fixed dependency set
# is two things in motion, and the collision surfaces as a traceback that points
# at neither.
#
# It has to be recent, though. These graphs load a krea2 text encoder, and
# CLIPLoader only learned that type well after v0.9.2 — which is what this was
# pinned to, from a tag list I sorted lexically instead of by version, so
# "v0.9.2" looked newer than "v0.32.0". It failed validation with
# "type: 'krea2' not in [...]".
ARG COMFYUI_REF=v0.32.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COMFYUI_PATH=/comfyui \
    MODELS_DIR=/comfyui/models \
    LORA_DIR=/comfyui/models/loras \
    # These graphs run several samplers back to back at a fixed resolution,
    # which fragments the caching allocator enough to OOM a 24GB card partway
    # through a carousel even though no single step is close to the limit.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # The rust downloader, for the multi-GB weights fetched on cold start.
    HF_HUB_ENABLE_HF_TRANSFER=1

# aria2 is not optional here — it is what makes an 18GB cold start minutes
# rather than tens of minutes. ffmpeg, libgl and libglib are the node set's.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget aria2 ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hard requirements of the node set, applied as a constraint to every install
# below so a node package's own requirements.txt cannot walk them back.
COPY constraints.txt /build/constraints.txt
ENV PIP_CONSTRAINT=/build/constraints.txt
# The opencv variants all install into the same cv2/ directory, so having two of
# them present is not "both work" — it is whichever unpacked last, with the
# other's files half overwritten. The base image ships one; drop every variant
# before the constraints file puts the contrib build in cleanly.
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true \
    && pip install --no-cache-dir --prefer-binary -r /build/constraints.txt

# The rest of what these nodes import. Taken from the detailer's working set —
# google-generativeai in particular is what Ask_Gemini_Batch needs, and
# segment-anything and ultralytics are Impact-Pack's.
RUN pip install --no-cache-dir --prefer-binary \
        ultralytics segment-anything sentencepiece \
        scikit-image PyWavelets piexif dill google-generativeai \
        scipy matplotlib hydra-core omegaconf iopath lpips

RUN git clone --depth 1 --branch ${COMFYUI_REF} \
        https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && rm -rf /comfyui/.git \
    && pip install --no-cache-dir --prefer-binary -r /comfyui/requirements.txt

COPY custom_nodes.txt /build/custom_nodes.txt
COPY scripts/install_nodes.sh /build/install_nodes.sh
RUN chmod +x /build/install_nodes.sh \
    && /build/install_nodes.sh /build/custom_nodes.txt /comfyui/custom_nodes

COPY requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir -r /build/requirements.txt

WORKDIR /app
COPY workflows/ /app/workflows/
COPY scripts/verify_nodes.py /app/scripts/verify_nodes.py

# Node check — OFF by default.
#
# It is a diagnostic, not a gate. It blocked four builds in a row and that cost
# far more than it was worth: the real answer to "does this image render" is a
# test job on a GPU, which the Radar UI can fire on demand. Turn it on with
# --build-arg VERIFY_NODES=1 when you actually want the report.
#
# What it prints when you do run it: which package supplies which node, and
# which packages supplied nothing.
ARG PRUNE_UNUSED_NODES=0
ARG VERIFY_NODES=0
RUN if [ "${VERIFY_NODES}" = "1" ]; then \
      python /app/scripts/verify_nodes.py \
        $([ "${PRUNE_UNUSED_NODES}" = "1" ] && echo --prune); \
    else \
      echo "[verify] skipped (VERIFY_NODES=0)"; \
    fi

# Last, so a code change rebuilds and re-pulls only these layers.
COPY models.txt /app/models.txt
COPY scripts/fetch_model.sh scripts/fetch_models.sh /app/scripts/
# NOT on PYTHONPATH globally. ComfyUI imports its own top-level modules by bare
# name, so ours sharing a name with one of its (src/comfy.py vs its `comfy`
# package) stops it starting. entrypoint.sh sets the path for the handler only.
COPY src/ /app/src/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/scripts/fetch_model.sh /app/scripts/fetch_models.sh

CMD ["/app/entrypoint.sh"]
