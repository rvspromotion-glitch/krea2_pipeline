"""check_model.py — the one place that decides what counts as a model.

Both fetch scripts call it: fetch_model.sh to judge what it just downloaded,
fetch_models.sh to judge what is already on disk. They used to answer that
question separately with a >1MiB size floor, which was wrong in both
directions — it rejected a 1KB LoRA as an error page and would have accepted a
13G checkpoint that stopped halfway.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from conftest import TINY_MODEL, safetensors

CHECK = Path(__file__).resolve().parent.parent / "scripts" / "check_model.py"

COMPLETE, NOT_SAFETENSORS, TRUNCATED = 0, 1, 2


def _state(path: Path) -> int:
    return subprocess.run(["python3", str(CHECK), str(path)]).returncode


def test_a_complete_model_is_complete(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(safetensors(b"\x00" * 4096))

    assert _state(path) == COMPLETE


def test_a_kilobyte_model_is_complete(tmp_path):
    """fedor_bypass: one [1, 12] tensor. Size says nothing about validity."""
    path = tmp_path / "fedor_bypass.safetensors"
    path.write_bytes(TINY_MODEL)

    assert len(TINY_MODEL) < 1048576
    assert _state(path) == COMPLETE


def test_a_cut_off_download_is_truncated(tmp_path):
    """The header promises 8MB of tensor; only 2MB arrived."""
    path = tmp_path / "m.safetensors"
    path.write_bytes(safetensors(b"\x00" * (2 * 1024 * 1024),
                                 declared=8 * 1024 * 1024))

    assert _state(path) == TRUNCATED


def test_an_html_page_is_not_safetensors(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"<!doctype html><html>rate limited</html>")

    assert _state(path) == NOT_SAFETENSORS


def test_zeroes_are_not_safetensors(tmp_path):
    """A header length of 0 is not a model, however many bytes follow it."""
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"\x00" * (2 * 1024 * 1024))

    assert _state(path) == NOT_SAFETENSORS


def test_an_empty_file_is_not_safetensors(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"")

    assert _state(path) == NOT_SAFETENSORS


def test_a_missing_file_does_not_crash(tmp_path):
    assert _state(tmp_path / "nope.safetensors") == NOT_SAFETENSORS


def test_trailing_padding_is_allowed(tmp_path):
    """Some writers pad. A file longer than its header requires is still whole."""
    path = tmp_path / "m.safetensors"
    path.write_bytes(safetensors(b"\x00" * 64) + b"\x00" * 512)

    assert _state(path) == COMPLETE
