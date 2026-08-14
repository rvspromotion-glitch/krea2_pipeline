"""RunPod serverless handler for the Krea2 pipeline.

One endpoint serves both workflows. They share every model — same checkpoint,
CLIP, VAE and LoRAs — so a second endpoint would mean a second copy of the image
and a second cold start for no benefit. Routing both job types here also keeps
the worker warm across a mixed batch, which is where the real time saving is.

Job input
---------
    mode            "single" | "carousel"
    workflow_version "v1" (default) | "v2" | "v3"
    persona_reference_url  v3 only: the persona's own photo, Flux's identity reference
    flux_lora_name / flux_lora_url  v3 only: the persona's Klein LoRA
    image_url       reference photo, fetched over HTTP
    image_b64       ...or inline base64 (image_url wins if both are given)
    lora_name       character LoRA filename
    lora_url        where to fetch it from if it is not already present
    trigger_word    e.g. "3lm1ra"
    description     e.g. "young woman with long platinum blonde hair"
    gemini_api_key  from Radar's settings; never baked into the graph
    seed            optional, for reproducing a specific run

Job output
----------
    images          list of base64 PNGs — 1 for single, 4 for carousel
    count, mode, seed, duration_s

`images` is always a list. A carousel's four entries are slides of one post, not
four alternatives, and the caller is expected to keep them ordered.
"""
from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from pathlib import Path

import requests

import comfy_client as comfy
import graph as graph_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("handler")

LORA_DIR = Path(os.getenv("LORA_DIR", "/comfyui/models/loras"))
FETCH_TIMEOUT = 180

# A carousel legitimately runs ten minutes; a single, two or three. One ceiling
# generous enough for both, since exceeding it means stuck rather than slow.
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "1800"))


class JobError(RuntimeError):
    """The job input is unusable."""


def _require(payload: dict, key: str) -> str:
    value = (payload.get(key) or "").strip()
    if not value:
        raise JobError(f"{key} is required")
    return value


def _fetch_reference(payload: dict) -> bytes:
    """The reference photo, from a URL or inline base64."""
    url = (payload.get("image_url") or "").strip()
    if url:
        r = requests.get(url, timeout=FETCH_TIMEOUT)
        if not r.ok:
            raise JobError(f"could not fetch image_url: HTTP {r.status_code}")
        if not r.content:
            raise JobError("image_url returned an empty body")
        return r.content

    b64 = (payload.get("image_b64") or "").strip()
    if b64:
        try:
            return base64.b64decode(b64)
        except Exception as exc:
            raise JobError(f"image_b64 is not valid base64: {exc}")

    raise JobError("one of image_url or image_b64 is required")


def _fetch_persona_reference(payload: dict) -> bytes | None:
    """The persona's own photo, which v3 hands Flux as its identity reference.

    None when the job did not supply one — v1 and v2 have no slot for it, and
    graph.patch is what decides whether its absence is an error.
    """
    url = (payload.get("persona_reference_url") or "").strip()
    if not url:
        return None
    r = requests.get(url, timeout=FETCH_TIMEOUT)
    if not r.ok:
        raise JobError(f"could not fetch persona_reference_url: HTTP {r.status_code}")
    return r.content


def _ensure_lora(payload: dict, name_key: str = "lora_name",
                 url_key: str = "lora_url", required: bool = True) -> str | None:
    """Make sure a character LoRA is present; return its filename.

    v3 renders with two of these — a Krea2 LoRA for the detail passes and a
    Klein LoRA for the Flux edit, since a Krea2 LoRA cannot load into Flux.
    Same caching and same atomic rename for both, so they share this.

    The shared models are baked into the image, but character LoRAs are not:
    they are per-persona, they change when a persona is retrained, and baking
    them would mean rebuilding and re-pushing an 18GB image to add one. Radar
    hosts them instead and sends a URL.

    So this normally fetches, once per cold start, into the container's own
    filesystem — a few hundred MB against a weekly batch, and it means adding a
    persona is an upload rather than a rebuild. A LoRA that *was* baked in (or
    fetched earlier in this batch) is used as-is.
    """
    if required:
        name = _require(payload, name_key)
    else:
        name = (payload.get(name_key) or "").strip()
        if not name:
            return None
    if "/" in name or "\\" in name:
        raise JobError(f"{name_key} must be a bare filename")

    target = LORA_DIR / name
    if target.exists() and target.stat().st_size > 0:
        return name

    url = (payload.get(url_key) or "").strip()
    if not url:
        raise JobError(
            f"LoRA {name!r} is not in the image and no {url_key} was supplied"
        )

    log.info("fetching character LoRA %s", name)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=FETCH_TIMEOUT) as r:
        if not r.ok:
            raise JobError(f"could not fetch {url_key}: HTTP {r.status_code}")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    # Rename only once complete, so a killed fetch cannot leave a truncated
    # file that looks resident to the next job.
    tmp.rename(target)
    return name


def run_job(payload: dict) -> dict:
    started = time.time()

    mode = (payload.get("mode") or "single").strip().lower()
    if mode not in graph_mod.MODES:
        raise JobError(f"mode must be one of {graph_mod.MODES}, got {mode!r}")

    # Falls back rather than raising: an unrecognised version costs the
    # experiment, not the day's render, and the log says which one ran.
    version = graph_mod.normalise_version(payload.get("workflow_version"))

    trigger_word = _require(payload, "trigger_word")
    description = _require(payload, "description")
    gemini_key = _require(payload, "gemini_api_key")
    lora_name = _ensure_lora(payload)
    flux_lora_name = _ensure_lora(payload, "flux_lora_name", "flux_lora_url",
                                  required=False)
    reference = _fetch_reference(payload)
    persona_reference = _fetch_persona_reference(payload)
    seed = payload.get("seed")

    comfy.wait_until_ready()

    uploaded = comfy.upload_image(reference, f"ref_{uuid.uuid4().hex}.png")
    log.info("reference uploaded as %s", uploaded)

    uploaded_persona = None
    if persona_reference is not None:
        uploaded_persona = comfy.upload_image(
            persona_reference, f"persona_{uuid.uuid4().hex}.png")
        log.info("persona reference uploaded as %s", uploaded_persona)

    job_graph = graph_mod.patch(
        mode,
        image_filename=uploaded,
        lora_name=lora_name,
        trigger_word=trigger_word,
        description=description,
        gemini_api_key=gemini_key,
        seed=seed,
        version=version,
        persona_reference=uploaded_persona,
        flux_lora_name=flux_lora_name,
    )
    log.info("patched %s/%s graph: %s", version, mode, graph_mod.describe(job_graph))

    node = graph_mod.output_node(job_graph)
    prompt_id = comfy.submit(job_graph, client_id=f"krea2-{uuid.uuid4().hex}")
    log.info("submitted %s as %s", mode, prompt_id)

    entry = comfy.wait(prompt_id, timeout=JOB_TIMEOUT)
    images = comfy.collect_images(entry, node)

    expected = 4 if mode == "carousel" else 1
    if len(images) != expected:
        # Not fatal — the caller can still use what came back — but it means the
        # graph changed shape, which is worth seeing in the logs immediately.
        log.warning("%s produced %d image(s), expected %d", mode, len(images), expected)

    return {
        "mode": mode,
        "workflow_version": version,
        "count": len(images),
        "seed": seed,
        "duration_s": round(time.time() - started, 1),
        "images": [base64.b64encode(i).decode() for i in images],
    }


def handler(event: dict) -> dict:
    """RunPod entry point. Errors come back as {"error": ...}, never a crash."""
    payload = (event or {}).get("input") or {}
    try:
        return run_job(payload)
    except JobError as exc:
        log.error("bad job input: %s", exc)
        return {"error": str(exc), "kind": "input"}
    except comfy.ComfyTimeout as exc:
        log.error("timeout: %s", exc)
        comfy.free_memory()
        return {"error": str(exc), "kind": "timeout"}
    except Exception as exc:
        log.exception("job failed")
        comfy.free_memory()
        return {"error": f"{type(exc).__name__}: {exc}", "kind": "render"}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
