"""fetch_models.sh — the cold-start weight fetch.

This runs on every cold start and the worker refuses to serve without it, so
the failure behaviour matters as much as the success path: a partial fetch must
stop the container rather than let ComfyUI start and fail every job with a
model-not-found that says nothing about why.
"""
from __future__ import annotations

import http.server
import re
import socket
import subprocess
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "fetch_models.sh"
MODEL_BYTES = b"\x00" * (2 * 1024 * 1024)
HTML = b"<!doctype html><html></html>"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/slow"):
            # Dribbled out, so the ticker has something to report on.
            import time
            self.send_response(200)
            self.send_header("Content-Length", str(len(MODEL_BYTES)))
            self.end_headers()
            for offset in range(0, len(MODEL_BYTES), 256 * 1024):
                try:
                    self.wfile.write(MODEL_BYTES[offset:offset + 256 * 1024])
                except BrokenPipeError:
                    return
                time.sleep(0.15)
            return
        if self.path.startswith("/model"):
            body = MODEL_BYTES
        elif self.path.startswith("/page"):
            body = HTML
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    httpd = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _run(list_file: Path, models_dir: Path, **env) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin",
             "MODELS_LIST": str(list_file), "MODELS_DIR": str(models_dir),
             "CIVITAI_TOKEN": "test-token", **env})


def _list(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "models.txt"
    path.write_text(rows)
    return path


def test_all_models_are_fetched(server, tmp_path):
    listing = _list(tmp_path, f"""
# a comment, and a blank line follow

civit  {server}/model  checkpoints/a.safetensors
civit  {server}/model  vae/b.safetensors
""")
    models = tmp_path / "models"

    result = _run(listing, models)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (models / "checkpoints/a.safetensors").read_bytes() == MODEL_BYTES
    assert (models / "vae/b.safetensors").read_bytes() == MODEL_BYTES


def test_a_warm_restart_downloads_nothing(server, tmp_path):
    """The saving that makes a per-cold-start fetch acceptable at all."""
    listing = _list(tmp_path, f"civit  {server}/model  checkpoints/a.safetensors\n")
    models = tmp_path / "models"
    _run(listing, models)

    result = _run(listing, models)

    assert result.returncode == 0
    assert "have a.safetensors" in result.stdout
    assert "0 fetched" in result.stdout


def test_one_bad_model_stops_the_worker(server, tmp_path):
    """Starting ComfyUI without a checkpoint would fail every job of the batch
    with an error that never mentions the download."""
    listing = _list(tmp_path, f"""
civit  {server}/model  checkpoints/good.safetensors
civit  {server}/page   vae/bad.safetensors
""")
    models = tmp_path / "models"

    result = _run(listing, models)

    assert result.returncode == 1
    assert "FAILED: bad.safetensors" in result.stderr
    assert "1 of 2 download(s) failed" in result.stderr
    assert not (models / "vae/bad.safetensors").exists()
    # The good one is kept: the next start resumes rather than refetching it.
    assert (models / "checkpoints/good.safetensors").exists()


def test_the_failure_names_the_model_and_shows_the_response(server, tmp_path):
    listing = _list(tmp_path, f"civit  {server}/nothing  loras/x.safetensors\n")

    result = _run(listing, tmp_path / "models")

    assert result.returncode == 1
    assert "x.safetensors" in result.stderr
    assert "HTTP status : 404" in result.stderr


def test_a_missing_token_is_reported_as_a_runpod_setting(server, tmp_path):
    """At runtime the token is an endpoint env var, not a repo secret."""
    listing = _list(tmp_path, f"civit  {server}/model  checkpoints/a.safetensors\n")

    result = _run(listing, tmp_path / "models", CIVITAI_TOKEN="")

    assert result.returncode == 1
    assert "RunPod" in result.stderr
    assert "Environment Variables" in result.stderr


def test_an_unreadable_list_is_fatal(tmp_path):
    result = _run(tmp_path / "nope.txt", tmp_path / "models")
    assert result.returncode == 1
    assert "no model list" in result.stderr


def test_an_unknown_kind_is_rejected(server, tmp_path):
    """A typo in models.txt must not silently skip a model."""
    listing = _list(tmp_path, "torrent  magnet:?xt=whatever  checkpoints/a.safetensors\n")

    result = _run(listing, tmp_path / "models")

    assert result.returncode == 1
    assert "unknown kind" in result.stderr


def test_progress_is_reported_while_downloads_are_in_flight(server, tmp_path):
    """Without this the log goes silent for minutes and the last line on screen
    is whichever download *started* last — which reads as "stuck on a small
    LoRA" when it is really the 12GB checkpoint still going."""
    listing = _list(tmp_path, f"""
civit  {server}/slow  checkpoints/a.safetensors
""")

    result = _run(listing, tmp_path / "models", MODEL_FETCH_PROGRESS_EVERY="1")

    assert result.returncode == 0, result.stdout + result.stderr
    ticks = [l for l in result.stdout.splitlines() if re.match(r"\[models\] \d+s ", l)]
    assert ticks, f"no progress lines:\n{result.stdout}"
    assert "a " in ticks[0], "a tick must name the file it is reporting on"


def test_the_ticker_does_not_outlive_the_fetch(server, tmp_path):
    """It sleeps in a subshell holding this script's stdout. An orphan there
    stalls whatever is reading our output for the rest of the interval."""
    import time

    listing = _list(tmp_path, f"civit  {server}/model  checkpoints/a.safetensors\n")

    started = time.monotonic()
    _run(listing, tmp_path / "models", MODEL_FETCH_PROGRESS_EVERY="30")
    elapsed = time.monotonic() - started

    assert elapsed < 10, (
        f"took {elapsed:.1f}s — the ticker is holding stdout open past the fetch")
