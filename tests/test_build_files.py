"""Static checks on the build, because a build failure costs an hour to find.

None of this needs Docker. It catches the mistakes that only show up forty
minutes into CI: a COPY of a file the build context excludes, a stage that
forgot to re-declare an ARG, a model path in the Dockerfile that does not match
where the fetch script puts it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / "workflows"
DOCKERFILE = (REPO / "Dockerfile").read_text()
DOCKERIGNORE = [l.strip() for l in (REPO / ".dockerignore").read_text().splitlines()
                if l.strip() and not l.startswith("#")]


def _copy_sources() -> list[str]:
    """Local paths each COPY reads from, ignoring --from=<stage> copies."""
    sources = []
    for line in DOCKERFILE.splitlines():
        line = line.strip()
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        parts = line.split()[1:]
        sources.extend(parts[:-1])       # last token is the destination
    return sources


def test_every_copied_path_exists():
    for source in _copy_sources():
        assert (REPO / source).exists(), f"Dockerfile COPYs missing path: {source}"


def test_no_copied_path_is_excluded_by_dockerignore():
    """The classic one: the file is in the repo but never in the build context."""
    for source in _copy_sources():
        top = source.rstrip("/").split("/")[0]
        for pattern in DOCKERIGNORE:
            assert pattern.rstrip("/") != top, \
                f".dockerignore excludes {pattern!r}, but the Dockerfile COPYs {source!r}"


def test_args_used_in_a_stage_are_declared_in_that_stage():
    """ARGs declared before the first FROM are not visible inside a stage."""
    global_args = set(re.findall(r"^ARG\s+(\w+)", DOCKERFILE.split("FROM")[0], flags=re.M))
    for chunk in re.split(r"^FROM ", DOCKERFILE, flags=re.M)[1:]:
        from_line, _, body = chunk.partition("\n")
        # The FROM line itself is the one place a pre-FROM ARG is still visible,
        # so it is excluded from the body along with the stage's own ARG lines.
        name = from_line.split()[-1]
        declared = set(re.findall(r"^ARG\s+(\w+)", body, flags=re.M))
        body = re.sub(r"^ARG\s+.*$", "", body, flags=re.M)
        for var in set(re.findall(r"\$\{(\w+)\}", body)) & global_args:
            assert var in declared, \
                f"stage {name!r} uses ${{{var}}} but does not re-declare ARG {var}"


def _model_list() -> list[tuple[str, str]]:
    """(kind, destination) for each row of models.txt."""
    rows = []
    for number, line in enumerate((REPO / "models.txt").read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        expected = {"hf": 4, "civit": 3}.get(parts[0])
        assert expected, f"models.txt:{number} unknown kind {parts[0]!r}"
        assert len(parts) == expected, f"models.txt:{number} wrong field count: {line}"
        rows.append((parts[0], parts[-1]))
    return rows


def test_the_model_list_is_well_formed():
    rows = _model_list()
    assert rows, "models.txt is empty — the worker would start with no weights"
    destinations = [dest for _, dest in rows]
    assert len(destinations) == len(set(destinations)), "duplicate destination"
    for _, dest in rows:
        assert not dest.startswith("/"), f"{dest} must be relative to MODELS_DIR"
        assert dest.endswith(".safetensors"), dest


def test_every_model_the_graphs_load_is_in_the_model_list():
    """A rename on either side is a render failure, not a download failure.

    ComfyUI resolves these by filename, so a model fetched to a different name
    than the graph asks for downloads perfectly and then fails validation on the
    first job of the batch.
    """
    import json

    referenced = set()
    for path in WORKFLOWS.glob("*.json"):
        for node in json.loads(path.read_text()).values():
            for key, value in (node.get("inputs") or {}).items():
                if isinstance(value, str) and value.endswith(".safetensors"):
                    referenced.add(value)

    fetched = {Path(dest).name for _, dest in _model_list()}

    # The v2 graphs load four weights nobody has a source URL for yet. They are
    # listed in models.txt as PENDING rather than guessed at, and v2 cannot
    # render until they are real. Named here so this test still fails the moment
    # a *fifth* model goes missing — the gap is recorded, not ignored.
    PENDING_V2 = {
        "krea2_turbo_lora_rank_64_bf16.safetensors",
        "Realism_engine.safetensors",
        "NiceGirls_Ultrarealistic.safetensors",
        "krea2_raw_fp8_scaled.safetensors",
    }
    manifest = (REPO / "models.txt").read_text()
    for name in PENDING_V2:
        assert name in manifest, f"{name} is not even recorded as pending"

    # The character LoRA is per-persona: patched into the graph per job and
    # fetched from Radar via lora_url, so it is deliberately not in models.txt.
    character_lora = referenced - fetched - PENDING_V2
    assert len(character_lora) == 1, (
        f"expected exactly one per-job LoRA slot, got {sorted(character_lora)} — "
        f"models.txt covers {sorted(fetched)}")

    from_graph = REPO / "src" / "graph.py"
    assert "TITLE_CHARACTER_LORA" in from_graph.read_text(), \
        "the per-job LoRA is patched by title; that lookup must still exist"


def test_the_image_does_not_bake_the_weights_in():
    """~39GB of models on top of torch does not build on a hosted runner."""
    assert "--from=models" not in DOCKERFILE
    assert "fetch_model.sh civit" not in DOCKERFILE, \
        "models are fetched at container start, not at build time"


def test_the_worker_fetches_models_before_starting_comfyui():
    """ComfyUI caches its model folder listing at boot."""
    entrypoint = (REPO / "entrypoint.sh").read_text()
    assert "fetch_models.sh" in entrypoint
    assert entrypoint.index("fetch_models.sh") < entrypoint.index("python3 main.py")


def test_torch_is_not_reinstalled_over_the_base_image():
    """The base image owns the torch stack. Do not touch it.

    This is the lesson from three consecutive failed builds. torch,
    torchvision, torchaudio, CUDA, cuDNN and Triton all have to agree, and
    assembling that set by hand here produced a different mismatch every time —
    an ABI break in torchaudio, then a torch too old for ComfyUI's own imports.
    The runpod/pytorch base ships a combination that already works, which the
    detailer worker runs in production.
    """
    for line in DOCKERFILE.splitlines():
        if not line.strip().startswith(("RUN pip install", "RUN pip3 install")):
            continue
        for package in ("torch", "torchvision", "torchaudio"):
            assert not re.search(rf"(?<![\w-]){package}(==|\s|$)", line), \
                f"installs {package} over the base image's: {line.strip()}"


def test_the_base_image_is_pinned():
    """A floating base tag would put the whole stack back in motion."""
    base = re.search(r"^FROM\s+(\S+)", DOCKERFILE, flags=re.M)
    assert base, "no FROM line"
    assert ":" in base.group(1), f"base image has no tag: {base.group(1)}"
    assert not base.group(1).endswith((":latest", ":main")), \
        f"base image tag is not fixed: {base.group(1)}"


def test_application_code_is_copied_last():
    """Anything after a changed layer is rebuilt, so the files that change on
    every commit have to sit below ComfyUI and the node install."""
    src_at = DOCKERFILE.index("COPY src/")
    for earlier in ("git clone", "install_nodes.sh"):
        assert DOCKERFILE.index(earlier) < src_at, \
            f"{earlier!r} must come before COPY src/"


def test_entrypoint_expects_no_network_volume():
    text = (REPO / "entrypoint.sh").read_text()
    assert "runpod-volume" not in text
    assert "RUN_SETUP" not in text


@pytest.mark.parametrize("path", ["scripts/install_nodes.sh", "scripts/fetch_model.sh",
                                  "entrypoint.sh"])
def test_shell_scripts_fail_fast(path):
    """A build step that fails silently produces a broken image that boots."""
    assert (REPO / path).read_text().startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in (REPO / path).read_text()


def test_custom_nodes_list_is_well_formed():
    rows = []
    for number, line in enumerate((REPO / "custom_nodes.txt").read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        assert len(parts) == 3, f"custom_nodes.txt:{number} is not 'name repo ref': {line}"
        assert "/" in parts[1], f"custom_nodes.txt:{number} repo needs owner/name: {line}"
        rows.append(parts[0])
    assert len(rows) == len(set(rows)), "duplicate package name in custom_nodes.txt"


# ── Name collisions with ComfyUI ─────────────────────────────────────────────

# ComfyUI imports these by bare name from its own root. Anything of ours on the
# same sys.path with one of these names wins, and ComfyUI does not start.
# src/comfy.py did exactly that: main.py died on its first line with
# "'comfy' is not a package", the container stayed up, and every job spent ten
# minutes waiting for a server that was never coming.
COMFYUI_TOP_LEVEL = {
    "comfy", "comfy_extras", "comfy_execution", "comfy_api", "comfy_config",
    "comfy_api_nodes", "app", "utils", "nodes", "execution", "folder_paths",
    "server", "main", "latent_preview", "node_helpers", "cuda_malloc",
    "hook_breaker_ac10a0", "protocol",
}


def test_no_module_of_ours_shadows_a_comfyui_one():
    for path in (REPO / "src").glob("*.py"):
        assert path.stem not in COMFYUI_TOP_LEVEL, (
            f"src/{path.name} shadows ComfyUI's {path.stem!r} — rename it, or "
            f"ComfyUI will not start")


def test_the_image_does_not_put_our_code_on_the_global_path():
    """PYTHONPATH in ENV applies to ComfyUI too, which is how the shadowing bit."""
    assert "PYTHONPATH=/app/src" not in DOCKERFILE.replace("# ", ""), \
        "set PYTHONPATH for the handler in entrypoint.sh, not image-wide"


def test_comfyui_is_started_without_our_path():
    entrypoint = (REPO / "entrypoint.sh").read_text()
    launch = next(l for l in entrypoint.splitlines() if "main.py" in l and "python3" in l)
    assert "-u PYTHONPATH" in launch, f"ComfyUI must not inherit PYTHONPATH: {launch}"


def test_the_entrypoint_supervises_both_processes():
    """A subshell cannot wait on a process it did not start — the previous
    version printed 'not a child of this shell' and supervised nothing, so a
    dead ComfyUI left the worker up failing every job."""
    entrypoint = (REPO / "entrypoint.sh").read_text()
    assert "wait -n" in entrypoint
    wait_line = next(l for l in entrypoint.splitlines() if l.strip().startswith("wait -n"))
    assert not wait_line.startswith(" "), "wait -n must run in the top-level shell"


# ── opencv ───────────────────────────────────────────────────────────────────

CONSTRAINTS = (REPO / "constraints.txt").read_text()
INSTALL_NODES = (REPO / "scripts" / "install_nodes.sh").read_text()


def test_the_contrib_opencv_build_is_the_one_requested():
    """LayerStyle imports guidedFilter from cv2.ximgproc, which is contrib-only.

    Plain opencv-python-headless gives a boot-time "Cannot import name
    'guidedFilter'" and a subset of its nodes that silently do nothing.
    """
    assert re.search(r"^opencv-contrib-python-headless\b", CONSTRAINTS, re.M)
    assert not re.search(r"^opencv-python(-headless)?\b", CONSTRAINTS, re.M)


def test_conflicting_opencv_builds_are_removed_before_install():
    """They share the cv2/ directory, so two installed at once is not two
    working packages — it is one, partly overwritten by the other."""
    assert re.search(r"pip uninstall.*opencv-python\b.*opencv-contrib-python\b", DOCKERFILE)


def test_a_node_package_cannot_swap_the_contrib_build_out():
    for name in ("opencv-python", "opencv-python-headless", "opencv-contrib-python"):
        assert name in INSTALL_NODES, f"{name} is not filtered from node requirements"
