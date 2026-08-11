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
