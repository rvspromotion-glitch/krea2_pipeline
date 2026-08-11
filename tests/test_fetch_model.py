"""fetch_model.sh, against a local server standing in for Civitai.

This exists because of a specific bug. The script used to pre-resolve Civitai's
redirect with `curl -I` so the token would not travel to the CDN — but the
signed CDN URL is issued for a GET, so the URL recovered from a HEAD chain
served an HTML page. curl downloaded it happily, and the build produced a 9KB
"checkpoint". The size and content-type checks caught it, but only because they
were there; the download itself reported success.

So: a redirect must be followed correctly, and anything that is not a model must
fail loudly and leave no file behind.
"""
from __future__ import annotations

import http.server
import socket
import subprocess
import threading
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_model.sh"
MODEL_BYTES = b"\x00" * (3 * 1024 * 1024)
HTML = b'<!doctype html>\n<html lang="en"><head><meta charset="utf-8" /></head></html>\n'


class _Handler(http.server.BaseHTTPRequestHandler):
    auth_seen: list = []

    def do_GET(self):
        if self.path.startswith("/redirect"):
            # Civitai's shape: authenticate, then 302 to a signed URL elsewhere.
            _Handler.auth_seen.append(self.headers.get("Authorization", ""))
            self.send_response(302)
            self.send_header("Location", f"http://{self.server.server_address[0]}:"
                                         f"{self.server.server_address[1]}/model?sig=abc")
            self.end_headers()
        elif self.path.startswith("/model"):
            self.send_response(200)
            self.send_header("Content-Length", str(len(MODEL_BYTES)))
            self.end_headers()
            self.wfile.write(MODEL_BYTES)
        elif self.path.startswith("/page"):
            self.send_response(200)
            self.send_header("Content-Length", str(len(HTML)))
            self.end_headers()
            self.wfile.write(HTML)
        elif self.path.startswith("/denied"):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"Unauthorized"}')
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.auth_seen = []
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    httpd = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


@pytest.fixture
def secrets(tmp_path):
    directory = tmp_path / "secrets"
    directory.mkdir()
    (directory / "civitai_token").write_text("test-token\n")
    return directory


def _fetch(url, out: Path, secrets_dir) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), "civit", url, str(out)],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "SECRETS_DIR": str(secrets_dir)})


def test_a_model_is_downloaded(server, secrets, tmp_path):
    out = tmp_path / "model.safetensors"

    result = _fetch(f"{server}/model", out, secrets)

    assert result.returncode == 0, result.stdout + result.stderr
    assert out.read_bytes() == MODEL_BYTES


def test_a_redirect_is_followed(server, secrets, tmp_path):
    """The regression. curl must follow the 302 itself and land on the file."""
    out = tmp_path / "model.safetensors"

    result = _fetch(f"{server}/redirect", out, secrets)

    assert result.returncode == 0, result.stdout + result.stderr
    assert out.read_bytes() == MODEL_BYTES
    assert _Handler.auth_seen and _Handler.auth_seen[0].startswith("Bearer "), \
        "the token must reach the first hop, or gated files 401"


def test_an_html_page_is_not_written_out_as_a_model(server, secrets, tmp_path):
    """HTTP 200 with a web page in the body — exactly what broke the build."""
    out = tmp_path / "model.safetensors"

    result = _fetch(f"{server}/page", out, secrets)

    assert result.returncode == 1
    assert not out.exists(), "a bad download must not be left in the image"
    assert "not a model" in result.stdout
    assert "web page in the body" in result.stdout


def test_the_error_shows_what_the_server_actually_sent(server, secrets, tmp_path):
    """'exit code 1' on its own costs an hour of guessing."""
    out = tmp_path / "model.safetensors"

    result = _fetch(f"{server}/denied", out, secrets)

    assert result.returncode == 1
    assert "HTTP status : 401" in result.stdout
    assert "token was rejected" in result.stdout
    assert "Unauthorized" in result.stdout          # the real response body


def test_a_404_says_so(server, secrets, tmp_path):
    out = tmp_path / "model.safetensors"

    result = _fetch(f"{server}/nothing", out, secrets)

    assert result.returncode == 1
    assert "HTTP status : 404" in result.stdout


def test_a_missing_token_fails_before_downloading_anything(server, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "model.safetensors"

    result = _fetch(f"{server}/model", out, empty)

    assert result.returncode == 1
    assert "CIVITAI_TOKEN" in result.stdout
    assert not out.exists()


def test_the_token_never_appears_in_a_url(server, secrets, tmp_path):
    """curl prints the URL on some errors, and build logs are not a safe place
    for a credential."""
    result = _fetch(f"{server}/denied", tmp_path / "m.safetensors", secrets)
    assert "test-token" not in result.stdout + result.stderr

    script = SCRIPT.read_text()
    assert "token=${token}" not in script and "?token=" not in script
