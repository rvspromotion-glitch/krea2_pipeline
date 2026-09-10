"""Shared fixtures for the worker tests."""
from __future__ import annotations

import json
import struct


def safetensors(payload: bytes, metadata: dict = None, declared: int = None) -> bytes:
    """A real safetensors file holding one tensor.

    Eight bytes of little-endian header length, that many bytes of JSON, then
    the tensor data. `declared` overstates the tensor's length without
    lengthening the body, which is what a download cut off part way looks like
    on disk.
    """
    header = {"weight": {"dtype": "F32", "shape": [1, max(1, len(payload) // 4)],
                         "data_offsets": [0, declared or len(payload)]}}
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob)) + blob + payload


# The file that crash-looped the worker: one [1, 12] tensor, so 48 bytes of
# payload and about a kilobyte all in — three orders of magnitude under the
# 1MiB floor that used to decide what counted as a model.
TINY_MODEL = safetensors(b"\x00" * 48, metadata={
    "name": "fedor_bypass",
    "target_weight": "diffusion_model.txtfusion.projector.weight",
})
