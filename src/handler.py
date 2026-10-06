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
    gemini_api_key  from Radar's settings; never baked into the graph. Only
                    needed by graphs that call Gemini directly
    openrouter_api_key  graphs whose Gemini nodes go through OpenRouter (v10)
                    (else OPENROUTER_API_KEY on the worker)
    hero_prompt     v10: the hero prompt Radar wrote; replaces the graph's own
                    Gemini call. Without it the graph writes one itself
    atlascloud_api_key  v10 carousel: Seedream on AtlasCloud draws the slides
                    (else ATLASCLOUD_API_KEY on the worker)
    carousel_slides v10 carousel: slides after the hero, 1-9 (graph default 3)
    output_format   "jpeg" (default) or "png"
    upload_url      where to PUT images that will not fit the result (Radar
                    opens one per render); each goes to <upload_url>/<index>
    seed            optional, for reproducing a specific run

Job output
----------
    images          list of base64 images — 1 for single, 4 for a carousel
                    (v10: the hero plus carousel_slides)
    uploaded        instead of images, when they were sent to upload_url:
                    [{index, bytes, sha256}] in slide order
    format          what they were sent as: "jpeg" or "png". A set that fits
                    neither the result nor an upload comes back as JPEG,
                    with format_note saying why
    quality         the JPEG quality the set was encoded at (None for PNG)
    count, mode, seed, duration_s

The whole result has to fit RunPod's 10 MB limit for /run results. A result
over it is refused when the worker hands it back ("Failed to return job
results | 400"), after the render has been paid for — so the images are sent
as JPEG, stepped down in quality until the set fits, and a set that cannot
fit comes back as an "output" error instead.

`images` is always a list. A carousel's four entries are slides of one post, not
four alternatives, and the caller is expected to keep them ordered.
"""
from __future__ import annotations

import base64
import io
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


class OutputTooLarge(RuntimeError):
    """The images cannot fit RunPod's result limit at any quality we accept."""


# RunPod refuses a /run result over 10 MB. The budget leaves room for the JSON
# around the images.
RESULT_BUDGET = int(os.getenv("RESULT_BUDGET_BYTES", "9000000"))

# Tried in order until the set fits. The graph's own last pass is a JPEG
# simulation at q92, so 95 with full-resolution colour loses nothing visible;
# the later steps only come into play for a long carousel.
JPEG_STEPS = ((95, 0), (95, 2), (92, 2), (88, 2), (84, 2), (80, 2))


def _jpeg(data: bytes, quality: int, subsampling: int) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        out = io.BytesIO()
        img.convert("RGB").save(out, "JPEG", quality=quality, subsampling=subsampling,
                                optimize=True)
        return out.getvalue()


OUTPUT_FORMATS = ("jpeg", "png")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def output_format(raw) -> str:
    value = str(raw or "jpeg").strip().lower()
    value = {"jpg": "jpeg"}.get(value, value)
    if value not in OUTPUT_FORMATS:
        raise JobError(f"output_format must be one of {OUTPUT_FORMATS}, got {raw!r}")
    return value


def _png(data: bytes) -> bytes:
    """ComfyUI saves PNG already; anything else is converted."""
    if data[:8] == PNG_MAGIC:
        return data
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        out = io.BytesIO()
        img.save(out, "PNG")
        return out.getvalue()


UPLOAD_TRIES = 3


class UploadFailed(RuntimeError):
    pass


def upload_images(upload_url: str, blobs: list[bytes]) -> list[dict]:
    """PUT each image to the caller's slot, in order. Returns what was sent.

    A 4xx is final (unknown or expired slot, refused bytes); a network error
    or 5xx is tried again.
    """
    import hashlib

    sent = []
    for index, blob in enumerate(blobs):
        mime = "image/png" if blob[:8] == PNG_MAGIC else "image/jpeg"
        last = None
        for attempt in range(UPLOAD_TRIES):
            try:
                r = requests.put(f"{upload_url.rstrip('/')}/{index}", data=blob,
                                 headers={"Content-Type": mime}, timeout=180)
            except requests.RequestException as exc:
                last = str(exc)
            else:
                if r.status_code < 300:
                    break
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code < 500:
                    raise UploadFailed(f"image {index}: {last}")
            if attempt + 1 < UPLOAD_TRIES:
                time.sleep(2 ** attempt)
        else:
            raise UploadFailed(f"image {index}: {last}")
        sent.append({"index": index, "bytes": len(blob),
                     "sha256": hashlib.sha256(blob).hexdigest()})
    return sent


def encode_output(images: list[bytes], fmt: str, upload_url: str = "") -> dict:
    """The images in the asked-for format, inline when they fit the result.

    Returns {images, format, quality, uploaded?, note}. A set too large for
    RunPod's 10 MB (a PNG carousel, a long JPEG one) goes to upload_url as it
    is. Without one, or if the upload fails, it is stepped down to a JPEG that
    fits rather than lost after the render was paid for — the note says why.
    """
    note = ""
    if fmt == "png":
        try:
            blobs = [_png(raw) for raw in images]
        except Exception as exc:
            note = f"PNG encoding failed ({exc}), sent as JPEG"
        else:
            b64 = [base64.b64encode(b).decode() for b in blobs]
            total = sum(len(s) for s in b64)
            if total <= RESULT_BUDGET:
                return {"images": b64, "format": "png", "quality": None, "note": ""}
            note = (f"{len(images)} PNG(s) come to {total / 1e6:.1f} MB, over RunPod's "
                    f"{RESULT_BUDGET / 1e6:.0f} MB limit")
            if upload_url:
                try:
                    sent = upload_images(upload_url, blobs)
                    log.info("%s; uploaded instead", note)
                    return {"images": [], "uploaded": sent, "format": "png",
                            "quality": None, "note": ""}
                except UploadFailed as exc:
                    note += f"; upload failed ({exc})"
            note += "; sent as JPEG"
        log.warning(note)
    try:
        b64, quality = encode_images(images)
        return {"images": b64, "format": "jpeg", "quality": quality, "note": note}
    except OutputTooLarge:
        if not upload_url:
            raise
    # A JPEG set too large even stepped down: upload it at full quality.
    quality, subsampling = JPEG_STEPS[0]
    blobs = []
    for raw in images:
        try:
            blobs.append(_jpeg(raw, quality, subsampling))
        except Exception:
            blobs.append(raw)
    try:
        sent = upload_images(upload_url, blobs)
    except UploadFailed as exc:
        raise OutputTooLarge(f"{len(images)} image(s) do not fit RunPod's result and the "
                             f"upload failed ({exc}). Ask for fewer carousel slides.")
    return {"images": [], "uploaded": sent, "format": "jpeg", "quality": quality, "note": note}


def encode_images(images: list[bytes]) -> tuple[list[str], int]:
    """The images as base64 JPEGs that fit the result budget, and the quality
    used. Bytes that are not an image are passed through as they are."""
    total = 0
    for quality, subsampling in JPEG_STEPS:
        encoded = []
        for raw in images:
            try:
                encoded.append(_jpeg(raw, quality, subsampling))
            except Exception:
                encoded.append(raw)
        b64 = [base64.b64encode(e).decode() for e in encoded]
        total = sum(len(s) for s in b64)
        if total <= RESULT_BUDGET:
            return b64, quality
    raise OutputTooLarge(
        f"{len(images)} image(s) come to {total / 1e6:.1f} MB even as JPEG quality "
        f"{JPEG_STEPS[-1][0]} — RunPod takes {RESULT_BUDGET / 1e6:.0f} MB per result. "
        f"Ask for fewer carousel slides.")


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
    # Which keys a graph needs depends on its Gemini nodes, so graph.patch()
    # is what refuses a job that is missing one.
    gemini_key = (payload.get("gemini_api_key") or "").strip()
    openrouter_key = (payload.get("openrouter_api_key") or "").strip()
    hero_prompt = (payload.get("hero_prompt") or "").strip() or None
    lora_name = _ensure_lora(payload)
    flux_lora_name = _ensure_lora(payload, "flux_lora_name", "flux_lora_url",
                                  required=False)
    reference = _fetch_reference(payload)
    persona_reference = _fetch_persona_reference(payload)
    # v7 only: the per-persona Flux edit instruction (e.g. the hair recolour).
    # Optional here — the graph is what decides whether it is required, and
    # graph.patch() raises if a v7 graph is handed nothing.
    flux_edit_prompt = (payload.get("flux_edit_prompt") or "").strip() or None

    # Per-persona LoRA strength. Absent or blank means "leave the graph alone",
    # which is what every job sent before this existed — retuning a drifting
    # LoRA is a Radar field now rather than an image rebuild.
    def _strength(field):
        raw = payload.get(field)
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a number, got {raw!r}")

    lora_strength = _strength("lora_strength")
    # v9 applies the persona twice, at strengths tuned apart.
    lora_strength_detail = _strength("lora_strength_detail")
    eye_colour = (payload.get("eye_colour") or "").strip()
    atlas_key = (payload.get("atlascloud_api_key") or "").strip()
    carousel_slides = payload.get("carousel_slides")
    fmt = output_format(payload.get("output_format"))
    upload_url = (payload.get("upload_url") or "").strip()
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
        flux_edit_prompt=flux_edit_prompt,
        lora_strength=lora_strength,
        lora_strength_detail=lora_strength_detail,
        eye_colour=eye_colour,
        atlascloud_api_key=atlas_key,
        carousel_slides=carousel_slides,
        openrouter_api_key=openrouter_key,
        hero_prompt=hero_prompt,
    )
    log.info("patched %s/%s graph: %s", version, mode, graph_mod.describe(job_graph))

    node = graph_mod.output_node(job_graph)
    prompt_id = comfy.submit(job_graph, client_id=f"krea2-{uuid.uuid4().hex}")
    log.info("submitted %s as %s", mode, prompt_id)

    entry = comfy.wait(prompt_id, timeout=JOB_TIMEOUT)
    images = comfy.collect_images(entry, node)

    expected = graph_mod.expected_images(job_graph, mode)
    if len(images) != expected:
        # Not fatal — the caller can still use what came back — but it means the
        # graph changed shape, which is worth seeing in the logs immediately.
        log.warning("%s produced %d image(s), expected %d", mode, len(images), expected)

    out = encode_output(images, fmt, upload_url)
    sent_as, quality, note = out["format"], out["quality"], out["note"]
    log.info("returning %d image(s) as %s%s, %s", len(images), sent_as.upper(),
             f" q{quality}" if quality else "",
             "uploaded" if out.get("uploaded")
             else f"{sum(len(s) for s in out['images']) / 1e6:.1f} MB inline")
    result = {
        "mode": mode,
        "workflow_version": version,
        "count": len(images),
        "seed": seed,
        "duration_s": round(time.time() - started, 1),
        "format": sent_as,
        "quality": quality,
        "images": out["images"],
    }
    if out.get("uploaded"):
        result["uploaded"] = out["uploaded"]
    if note:
        result["format_note"] = note
    return result


def handler(event: dict) -> dict:
    """RunPod entry point. Errors come back as {"error": ...}, never a crash."""
    payload = (event or {}).get("input") or {}
    try:
        return run_job(payload)
    except (JobError, graph_mod.MissingKey) as exc:
        log.error("bad job input: %s", exc)
        return {"error": str(exc), "kind": "input"}
    except OutputTooLarge as exc:
        log.error("result too large: %s", exc)
        return {"error": str(exc), "kind": "output"}
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
