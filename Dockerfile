# Krea2 pipeline worker — RunPod serverless.
#
# Deliberately thin: ComfyUI, the custom nodes and ~40GB of models all live on
# the network volume, seeded once by setup_volume.sh. Baking them into the image
# would mean a multi-tens-of-GB pull on every cold start, which is exactly what
# the volume exists to avoid.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COMFYUI_PATH=/runpod-volume/ComfyUI \
    LORA_DIR=/runpod-volume/ComfyUI/models/loras

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl aria2 file libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY workflows/ /app/workflows/
COPY setup_volume.sh entrypoint.sh /app/
RUN chmod +x /app/setup_volume.sh /app/entrypoint.sh

# handler.py imports its siblings by bare name.
ENV PYTHONPATH=/app/src

CMD ["/app/entrypoint.sh"]
