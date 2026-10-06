"""Handler contract tests — ComfyUI and the network are stubbed.

What matters here is the job envelope: what a caller must send, what comes back,
and that every failure returns a structured error instead of taking the worker
down. Radar is written against this shape, so it is a real interop contract.
"""
import base64
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import comfy_client as comfy  # noqa: E402
import handler as handler_mod  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"fake"


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """A worker where ComfyUI always succeeds, returning `state['images']`."""
    state = {"images": [PNG], "submitted": None, "uploaded": None}

    monkeypatch.setattr(comfy, "wait_until_ready", lambda *a, **k: None)
    monkeypatch.setattr(comfy, "upload_image",
                        lambda data, name: state.__setitem__("uploaded", data) or "up.png")
    monkeypatch.setattr(comfy, "submit",
                        lambda g, client_id: state.__setitem__("submitted", g) or "pid-1")
    monkeypatch.setattr(comfy, "wait", lambda pid, timeout=None: {"status": {"completed": True}})
    monkeypatch.setattr(comfy, "collect_images", lambda entry, node: state["images"])

    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "Eva.safetensors").write_bytes(b"x")
    monkeypatch.setattr(handler_mod, "LORA_DIR", lora_dir)
    return state


def job(**over):
    payload = {
        "mode": "single",
        "image_b64": base64.b64encode(PNG).decode(),
        "lora_name": "Eva.safetensors",
        "trigger_word": "3lm1ra",
        "description": "young woman with long platinum blonde hair",
        "gemini_api_key": "key-123",
        "openrouter_api_key": "router-123",
    }
    payload.update(over)
    return {"input": payload}


# ── Happy path ───────────────────────────────────────────────────────────────

def test_single_returns_one_base64_image(stub):
    out = handler_mod.handler(job())

    assert "error" not in out
    assert out["mode"] == "single"
    assert out["count"] == 1
    assert base64.b64decode(out["images"][0]) == PNG
    assert "duration_s" in out


def test_carousel_returns_four_images_in_order(stub):
    slides = [PNG + bytes([i]) for i in range(4)]
    stub["images"] = slides

    out = handler_mod.handler(job(mode="carousel"))

    assert out["count"] == 4
    assert [base64.b64decode(i) for i in out["images"]] == slides


def test_the_reference_photo_reaches_comfy(stub):
    handler_mod.handler(job())
    assert stub["uploaded"] == PNG


def test_the_submitted_graph_carries_the_persona(stub):
    handler_mod.handler(job(trigger_word="ch10e", description="soft pastel streetwear"))
    prompts = [n["inputs"]["value"] for n in stub["submitted"].values()
               if n["class_type"] == "PrimitiveStringMultiline"]

    assert prompts
    assert all("ch10e, soft pastel streetwear" in p for p in prompts)
    assert all("{subject}" not in p for p in prompts)


def test_the_gemini_key_comes_from_the_job_not_the_file(stub):
    handler_mod.handler(job(gemini_api_key="from-radar"))
    keys = [n["inputs"]["api_key"] for n in stub["submitted"].values()
            if n["class_type"] == "Ask_Gemini_Batch"]

    assert keys and all(k == "from-radar" for k in keys)


def test_an_unexpected_image_count_still_returns_them(stub):
    """A shape change should be visible, not fatal — the images are still real."""
    stub["images"] = [PNG, PNG]
    out = handler_mod.handler(job(mode="carousel"))
    assert out["count"] == 2
    assert "error" not in out


# ── Failure modes ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("missing", ["trigger_word", "description", "gemini_api_key"])
def test_missing_required_fields_are_input_errors(stub, missing):
    out = handler_mod.handler(job(**{missing: ""}))
    assert out["kind"] == "input"
    assert missing in out["error"]


def test_no_image_at_all_is_an_input_error(stub):
    payload = job()
    payload["input"].pop("image_b64")
    out = handler_mod.handler(payload)
    assert out["kind"] == "input"
    assert "image_url or image_b64" in out["error"]


def test_bad_mode_is_rejected(stub):
    assert handler_mod.handler(job(mode="video"))["kind"] == "input"


def test_a_missing_lora_without_a_url_is_an_input_error(stub):
    out = handler_mod.handler(job(lora_name="NotThere.safetensors"))
    assert out["kind"] == "input"
    assert "not in the image" in out["error"]


def test_lora_name_cannot_escape_the_lora_dir(stub):
    out = handler_mod.handler(job(lora_name="../../etc/passwd"))
    assert out["kind"] == "input"
    assert "bare filename" in out["error"]


def test_a_render_failure_is_reported_not_raised(stub, monkeypatch):
    def boom(*a, **k):
        raise comfy.ComfyError("run failed: node 1031 has no output 0")

    monkeypatch.setattr(comfy, "wait", boom)
    out = handler_mod.handler(job())

    assert out["kind"] == "render"
    assert "1031" in out["error"]


def test_a_timeout_is_reported_as_a_timeout(stub, monkeypatch):
    def boom(*a, **k):
        raise comfy.ComfyTimeout("run pid-1 exceeded 1800s")

    monkeypatch.setattr(comfy, "wait", boom)
    out = handler_mod.handler(job())

    assert out["kind"] == "timeout"
    assert "1800" in out["error"]


# ── v10 ──────────────────────────────────────────────────────────────────────

def test_a_v10_carousel_takes_the_atlas_key_and_slide_count_from_the_job(stub, monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    stub["images"] = [PNG + bytes([i]) for i in range(6)]
    out = handler_mod.handler(job(mode="carousel", workflow_version="v10", eye_colour="grey eyes",
                                  atlascloud_api_key="atlas-1", carousel_slides=5))
    assert "error" not in out and out["count"] == 6 and out["workflow_version"] == "v10"
    g = stub["submitted"]
    assert [n["inputs"]["api_key"] for n in g.values()
            if n["class_type"] == "SeedreamEditSequentialAtlas"] == ["atlas-1"]


def test_a_v10_carousel_without_an_atlas_key_is_refused_before_rendering(stub, monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    out = handler_mod.handler(job(mode="carousel", workflow_version="v10", eye_colour="grey eyes"))
    assert "AtlasCloud" in out["error"] and stub["submitted"] is None


def test_the_hero_prompt_from_the_job_is_what_the_v10_graph_renders(stub):
    out = handler_mod.handler(job(workflow_version="v10", eye_colour="grey eyes",
                                  gemini_api_key="", hero_prompt="  she laughs on a pier  "))
    assert "error" not in out
    g = stub["submitted"]
    assert [n["inputs"]["value"] for n in g.values()
            if n["_meta"].get("title") == "Hero prompt"] == ["she laughs on a pier"]


def test_a_v10_job_without_an_openrouter_key_is_an_input_error(stub, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = handler_mod.handler(job(mode="carousel", workflow_version="v10", eye_colour="grey eyes",
                                  atlascloud_api_key="atlas-1", openrouter_api_key="",
                                  hero_prompt="x"))
    assert out["kind"] == "input" and "openrouter_api_key" in out["error"]
    assert stub["submitted"] is None


# ── The result has to fit RunPod's 10 MB ─────────────────────────────────────

def _grainy_png(seed: int, size=(1612, 2016)) -> bytes:
    """A v10-sized frame with film grain — the kind of image PNG barely compresses."""
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    w, h = size
    base = np.linspace(40, 210, w, dtype=np.float32)[None, :, None].repeat(h, 0).repeat(3, 2)
    noisy = np.clip(base + rng.normal(0, 14, (h, w, 3)), 0, 255).astype("uint8")
    out = io.BytesIO()
    Image.fromarray(noisy).save(out, "PNG")
    return out.getvalue()


def test_a_v10_carousel_comes_back_as_jpegs_that_fit_runpods_limit(stub, monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    slides = [_grainy_png(i) for i in range(4)]
    assert sum(len(base64.b64encode(s)) for s in slides) > 10_000_000      # as PNG: refused
    stub["images"] = slides
    out = handler_mod.handler(job(mode="carousel", workflow_version="v10", eye_colour="grey eyes",
                                  atlascloud_api_key="atlas-1"))
    assert "error" not in out and out["format"] == "jpeg" and out["count"] == 4
    assert sum(len(i) for i in out["images"]) <= handler_mod.RESULT_BUDGET
    assert all(base64.b64decode(i)[:2] == b"\xff\xd8" for i in out["images"])


def test_a_set_that_cannot_fit_is_an_output_error_not_a_lost_render(stub, monkeypatch):
    monkeypatch.setattr(handler_mod, "RESULT_BUDGET", 50_000)
    stub["images"] = [_grainy_png(0, (400, 500))]
    out = handler_mod.handler(job())
    assert out["kind"] == "output" and "fewer carousel slides" in out["error"]


# ── Output format ────────────────────────────────────────────────────────────

def test_png_is_sent_as_the_png_comfyui_saved(stub):
    stub["images"] = [_grainy_png(0, (400, 500))]
    out = handler_mod.handler(job(output_format="png"))
    assert "error" not in out and out["format"] == "png" and out["quality"] is None
    assert base64.b64decode(out["images"][0]) == stub["images"][0]
    assert "format_note" not in out


@pytest.mark.parametrize("asked", [None, "", "jpeg", "jpg", "JPG"])
def test_jpeg_is_the_default_and_jpg_means_jpeg(stub, asked):
    stub["images"] = [_grainy_png(0, (400, 500))]
    out = handler_mod.handler(job(output_format=asked))
    assert out["format"] == "jpeg"
    assert base64.b64decode(out["images"][0])[:2] == b"\xff\xd8"


def test_a_png_set_over_the_limit_comes_back_as_jpeg_rather_than_lost(stub, monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    stub["images"] = [_grainy_png(i) for i in range(4)]
    out = handler_mod.handler(job(mode="carousel", workflow_version="v10", eye_colour="grey eyes",
                                  atlascloud_api_key="atlas-1", output_format="png"))
    assert "error" not in out and out["format"] == "jpeg"
    assert "PNG" in out["format_note"] and "JPEG" in out["format_note"]
    assert sum(len(i) for i in out["images"]) <= handler_mod.RESULT_BUDGET


def test_an_unknown_output_format_is_an_input_error(stub):
    out = handler_mod.handler(job(output_format="webp"))
    assert out["kind"] == "input" and "output_format" in out["error"]
    assert stub["submitted"] is None
