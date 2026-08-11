"""Thin client for the ComfyUI HTTP API running inside the worker.

Named comfy_client, not comfy, and that matters: ComfyUI's own package is
`comfy`, and anything of ours on sys.path called that shadows it. main.py's
first line is `import comfy.options`, so the collision does not degrade —
ComfyUI simply does not start, with a "'comfy' is not a package" that says
nothing about where the other one came from.

The worker boots ComfyUI on localhost and talks to it over HTTP rather than
importing it: ComfyUI owns its own model cache and execution queue, and keeping
it as a separate process is what lets a warm worker skip model loading entirely
between jobs. Across a sequential weekly batch that is most of the saving: the
models are loaded once for forty-odd jobs, not once per job.

Progress is tracked by polling /history rather than the websocket. The websocket
gives finer-grained progress, but it also drops silently on long jobs and then
you are waiting forever on a socket that will never speak again — a real risk
when a carousel legitimately runs ten minutes.
"""
from __future__ import annotations

import logging
import time
import urllib.parse
from pathlib import Path

import requests

log = logging.getLogger(__name__)

COMFY_HOST = "http://127.0.0.1:8188"

# A carousel is four samplers plus four refiners plus two Gemini round-trips.
# Ten minutes is normal; this is the ceiling before we call it stuck.
DEFAULT_TIMEOUT = 1800

# Poll fast at first so short jobs return promptly, then back off — a ten minute
# job does not need two hundred status checks.
POLL_FAST_S = 2
POLL_SLOW_S = 10
POLL_FAST_WINDOW_S = 60


class ComfyError(RuntimeError):
    """ComfyUI refused the graph, or the run failed."""


class ComfyTimeout(ComfyError):
    """The run exceeded its deadline."""


def wait_until_ready(timeout: int = 600) -> None:
    """Block until ComfyUI answers.

    Generous, because this covers a cold start: ComfyUI imports every custom
    node package before it serves anything, and RunPod can hand the handler a
    job before that finishes.
    """
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_HOST}/system_stats", timeout=5)
            if r.ok:
                log.info("ComfyUI ready")
                return
            last = f"HTTP {r.status_code}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise ComfyError(f"ComfyUI did not become ready within {timeout}s ({last})")


def upload_image(data: bytes, filename: str) -> str:
    """Push the reference photo into ComfyUI's input dir. Returns its name.

    Uploading beats writing to disk directly: ComfyUI dedupes and sanitises the
    name itself, and returns the name LoadImage will actually resolve.
    """
    files = {"image": (filename, data, "image/png")}
    r = requests.post(f"{COMFY_HOST}/upload/image",
                      files=files, data={"overwrite": "true"}, timeout=120)
    if not r.ok:
        raise ComfyError(f"upload failed HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    name = body.get("name") or filename
    subfolder = body.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name


def submit(graph: dict, client_id: str) -> str:
    r = requests.post(f"{COMFY_HOST}/prompt",
                      json={"prompt": graph, "client_id": client_id}, timeout=60)
    if r.status_code == 400:
        # ComfyUI returns a structured validation error here — surface it whole,
        # because "node 1031 has no output 0" is the entire diagnosis.
        raise ComfyError(f"graph rejected: {r.text[:2000]}")
    if not r.ok:
        raise ComfyError(f"submit failed HTTP {r.status_code}: {r.text[:300]}")
    prompt_id = (r.json() or {}).get("prompt_id")
    if not prompt_id:
        raise ComfyError(f"no prompt_id in response: {r.text[:300]}")
    return prompt_id


def _history(prompt_id: str) -> dict | None:
    try:
        r = requests.get(f"{COMFY_HOST}/history/{prompt_id}", timeout=30)
        if not r.ok:
            return None
        return (r.json() or {}).get(prompt_id)
    except Exception as exc:
        log.warning("history poll failed (retrying): %s", exc)
        return None


def wait(prompt_id: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Block until the run finishes. Returns its history entry."""
    started = time.time()
    deadline = started + timeout

    while time.time() < deadline:
        elapsed = time.time() - started
        time.sleep(POLL_FAST_S if elapsed < POLL_FAST_WINDOW_S else POLL_SLOW_S)

        entry = _history(prompt_id)
        if not entry:
            continue

        status = entry.get("status") or {}
        if status.get("status_str") == "error" or status.get("completed") is False:
            messages = status.get("messages") or []
            raise ComfyError(f"run failed: {str(messages)[:1500]}")
        if status.get("completed"):
            log.info("run %s finished in %.0fs", prompt_id, time.time() - started)
            return entry

    raise ComfyTimeout(f"run {prompt_id} exceeded {timeout}s")


def collect_images(entry: dict, node_id: str) -> list[bytes]:
    """Fetch every image the output node produced, in graph order."""
    outputs = (entry.get("outputs") or {}).get(node_id) or {}
    images = outputs.get("images") or []
    if not images:
        raise ComfyError(f"node {node_id} produced no images")

    out: list[bytes] = []
    for meta in images:
        params = urllib.parse.urlencode({
            "filename": meta["filename"],
            "subfolder": meta.get("subfolder", ""),
            "type": meta.get("type", "output"),
        })
        r = requests.get(f"{COMFY_HOST}/view?{params}", timeout=120)
        if not r.ok:
            raise ComfyError(f"could not fetch {meta['filename']}: HTTP {r.status_code}")
        out.append(r.content)
    return out


def free_memory() -> None:
    """Ask ComfyUI to drop cached models — used only when a job fails hard.

    Not called between successful jobs: keeping models resident is exactly what
    makes the second job in a batch fast.
    """
    try:
        requests.post(f"{COMFY_HOST}/free",
                      json={"unload_models": True, "free_memory": True}, timeout=30)
    except Exception as exc:
        log.warning("free_memory failed: %s", exc)
