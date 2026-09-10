#!/usr/bin/env python3
"""Is this file a complete model?

Safetensors states its own length: eight bytes of little-endian header size,
that many bytes of JSON, then tensor data at the offsets the JSON declares. So
a file can be asked whether it is whole instead of being guessed at from its
size — which is the only way to tell a legitimately tiny model from an error
page, and a finished 13G checkpoint from one that stopped halfway.

Exit codes:
    0   complete safetensors
    2   safetensors, but shorter than its own header says — a cut-off download
    1   not safetensors at all; the caller decides what to do with it

Used by fetch_model.sh (is what I just downloaded a model?) and by
fetch_models.sh (is what is already on disk worth keeping?), so the two cannot
drift apart on what "a real model" means.
"""
import json
import os
import struct
import sys

# An HTML error page read as a little-endian u64 is astronomically large, so an
# implausible header length is how "this is not safetensors" spells itself.
MAX_HEADER = 100_000_000

NOT_SAFETENSORS = 1
TRUNCATED = 2


def state(path: str) -> int:
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) < 8:
            return NOT_SAFETENSORS
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= MAX_HEADER or 8 + length > size:
            return NOT_SAFETENSORS
        header = json.loads(handle.read(length))

    if not isinstance(header, dict):
        return NOT_SAFETENSORS

    # __metadata__ is the one key that is not a tensor.
    end = max((info["data_offsets"][1]
               for key, info in header.items()
               if key != "__metadata__" and isinstance(info, dict)
               and isinstance(info.get("data_offsets"), list)), default=0)
    return 0 if size >= 8 + length + end else TRUNCATED


if __name__ == "__main__":
    try:
        sys.exit(state(sys.argv[1]))
    except Exception:
        # Unreadable, malformed, not JSON — all the same answer: we cannot call
        # this a complete safetensors, so let the caller fall back.
        sys.exit(NOT_SAFETENSORS)
