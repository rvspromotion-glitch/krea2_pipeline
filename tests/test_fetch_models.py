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

from conftest import TINY_MODEL, safetensors

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
        if self.path.startswith("/tiny"):
            body = TINY_MODEL
        elif self.path.startswith("/model"):
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


# ── What counts as already present ──────────────────────────────────────────
#
# With no network volume every cold start refetches all 53G, so "present" only
# ever applies within one container's life — but it still decides whether a
# retry after a partial boot re-hits Civitai, and each avoidable Civitai
# round-trip is another chance to hang.

def test_a_kilobyte_model_counts_as_present(server, tmp_path):
    """The leftover from the size floor: fedor_bypass is 1040 bytes, so a >1MiB
    have-it test called it missing and refetched it from Civitai every boot."""
    listing = _list(tmp_path, f"civit  {server}/tiny  loras/fedor_bypass.safetensors\n")
    models = tmp_path / "models"
    first = _run(listing, models)
    assert first.returncode == 0, first.stdout + first.stderr
    assert (models / "loras/fedor_bypass.safetensors").read_bytes() == TINY_MODEL

    result = _run(listing, models)

    assert result.returncode == 0
    assert "have fedor_bypass.safetensors" in result.stdout
    assert "0 fetched" in result.stdout


def test_a_truncated_file_on_disk_is_refetched(server, tmp_path):
    """A cut-off download must not be trusted on the next start just because it
    is large. Re-fetching is what lets a bad boot heal itself."""
    listing = _list(tmp_path, f"civit  {server}/model  checkpoints/a.safetensors\n")
    models = tmp_path / "models"
    (models / "checkpoints").mkdir(parents=True)
    (models / "checkpoints/a.safetensors").write_bytes(
        safetensors(b"\x00" * (2 * 1024 * 1024), declared=8 * 1024 * 1024))

    result = _run(listing, models)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "have a.safetensors" not in result.stdout
    assert "1 fetched" in result.stdout
    assert (models / "checkpoints/a.safetensors").read_bytes() == MODEL_BYTES


# ── Nothing may wait forever ────────────────────────────────────────────────

def test_every_network_call_can_give_up():
    """The stall that ate the container: Civitai accepted the connection and
    never answered, and the redirect-resolving curl had no timeout of any kind,
    so the boot sat there until RunPod reaped it ~8 minutes later.

    A transfer is bounded by a stall floor rather than a wall clock on purpose —
    --max-time on a 13G download would kill legitimate fetches on a slow night.
    """
    script = (REPO / "scripts" / "fetch_model.sh").read_text()
    # Comments talk about curl too; only real invocations are the subject here.
    code = "\n".join(line for line in script.splitlines()
                     if not line.lstrip().startswith("#"))

    def invocations(command):
        # A command position: start of line, or opening a $(...) substitution.
        # Continuation lines are pulled in so the whole call is inspected.
        # (line ending in a backslash)* then the final line, so a call split
        # over several lines is inspected whole.
        return re.findall(
            rf"(?:^|\$\()\s*(?:if\s+)?{command} (?:[^\n]*\\\n)*[^\n]*",
            code, re.MULTILINE)

    calls = invocations("curl")
    assert len(calls) == 3, f"expected 3 curl calls, found {len(calls)}"
    for call in calls:
        assert "--connect-timeout" in call, f"curl with no connect timeout:\n{call}"
        assert ("--max-time" in call
                or "--speed-time" in call), f"curl that can hang forever:\n{call}"

    aria = invocations("aria2c")
    assert len(aria) == 2, f"expected 2 aria2 calls, found {len(aria)}"
    for call in aria:
        assert "ARIA_STALL" in call or "--lowest-speed-limit" in call, \
            f"aria2 with no stall guard:\n{call}"


# ── Fetching only what a version loads ──────────────────────────────────────
#
# The worker has no network volume, so every cold start refetches whatever it
# is told to — and it was told to fetch the union of all fourteen graphs. v9
# loads eight of the seventeen; the nine it skips are Flux weights and
# superseded checkpoints, some ~47GB, downloaded to sit unread until the
# container is reclaimed.

import sys

sys.path.insert(0, str(REPO / "scripts"))


def test_a_version_asks_only_for_what_its_own_graphs_load():
    from models_for import models_for

    v9 = models_for(["v9"])

    assert "krast_bf16.safetensors" in v9
    assert "Lenovo_ultrareal.safetensors" in v9
    # v9 renders no Flux at all.
    assert not any("Klein" in m or "flux" in m.lower() for m in v9), sorted(v9)
    assert "AiO_krea2_checkpoint_int8_8steps.safetensors" not in v9, \
        "v9 loads the krast checkpoint instead"


def test_a_borrowed_carousel_is_not_counted():
    """v7 onward are single-photo flows whose carousel entry points at v6's, so
    a stray carousel job fails on a render rather than a KeyError. Counting
    that graph as theirs dragged the whole Flux stack into every v9 cold start
    to serve a job that cannot arrive while carousels are retired."""
    from models_for import models_for

    assert ("v9", "carousel") in __import__("graph").BORROWED
    assert len(models_for(["v9"])) < len(models_for(["v6"]))


def test_asking_for_several_versions_unions_them():
    from models_for import models_for

    assert models_for(["v8", "v9"]) == models_for(["v8"]) | models_for(["v9"])


def test_every_version_is_derivable_and_fetchable():
    """A version whose graph names a model the manifest cannot supply would
    boot clean and die on the first render."""
    from models_for import models_for

    import graph as graph_mod

    fetched = {line.split()[-1].split("/")[-1]
               for line in (REPO / "models.txt").read_text().splitlines()
               if line.split() and line.split()[0] in ("hf", "civit", "url")}
    for version in graph_mod.VERSIONS:
        missing = {m for m in models_for([version]) if m not in fetched}
        # Per-persona LoRAs come from Radar, not models.txt.
        missing = {m for m in missing
                   if "Eva" not in m and "3lm1ra" not in m and "Chloe" not in m}
        assert not missing, f"{version} loads {sorted(missing)}, not in models.txt"


def test_an_unknown_version_fails_the_boot_rather_than_fetching_nothing(
        server, tmp_path):
    """A typo in the endpoint's setting must not quietly resolve to an empty
    set — that boots a worker with no weights and fails every job."""
    listing = _list(tmp_path, f"civit  {server}/model  checkpoints/a.safetensors\n")

    result = _run(listing, tmp_path / "models", WORKFLOW_VERSIONS="v99")

    assert result.returncode == 1
    assert "unknown version" in result.stdout + result.stderr


def test_no_filter_still_fetches_everything(server, tmp_path):
    """Every endpoint behaved this way before the setting existed, and an
    endpoint that has not been told is not an endpoint that wants less."""
    listing = _list(tmp_path, f"""
civit  {server}/model  checkpoints/a.safetensors
civit  {server}/model  vae/b.safetensors
""")
    models = tmp_path / "models"

    result = _run(listing, models)

    assert result.returncode == 0
    assert "2 fetched" in result.stdout


def test_the_filter_actually_skips_what_the_version_does_not_load(server, tmp_path):
    """Named with real filenames on purpose. Every other test here uses made-up
    ones, so with a version set they would all be skipped and the suite would
    pass whether the filter worked or not — which it did, until this existed."""
    listing = _list(tmp_path, f"""
civit  {server}/model  checkpoints/krast_bf16.safetensors
civit  {server}/model  checkpoints/AiO_krea2_checkpoint_int8_8steps.safetensors
""")
    models = tmp_path / "models"

    result = _run(listing, models, WORKFLOW_VERSIONS="v9")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (models / "checkpoints/krast_bf16.safetensors").exists()
    assert not (models / "checkpoints/AiO_krea2_checkpoint_int8_8steps.safetensors").exists(), \
        "v9 loads the krast checkpoint; the AiO one is 12G it never reads"
    assert "1 not needed by v9" in result.stdout


def test_several_versions_on_one_endpoint_fetch_the_union(server, tmp_path):
    """An endpoint can serve more than one version while a new one is tried."""
    listing = _list(tmp_path, f"""
civit  {server}/model  checkpoints/krast_bf16.safetensors
civit  {server}/model  checkpoints/AiO_krea2_checkpoint_int8_8steps.safetensors
""")
    models = tmp_path / "models"

    result = _run(listing, models, WORKFLOW_VERSIONS="v1,v9")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (models / "checkpoints/krast_bf16.safetensors").exists()
    assert (models / "checkpoints/AiO_krea2_checkpoint_int8_8steps.safetensors").exists()
