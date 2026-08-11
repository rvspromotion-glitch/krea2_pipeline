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
    # The character LoRA is per-persona: patched into the graph per job and
    # fetched from Radar via lora_url, so it is deliberately not in models.txt.
    character_lora = referenced - fetched
    assert len(character_lora) == 1, (
        f"expected exactly one per-job LoRA slot, got {sorted(character_lora)} — "
        f"models.txt covers {sorted(fetched)}")

    from_graph = REPO / "src" / "graph.py"
    assert "TITLE_CHARACTER_LORA" in from_graph.read_text(), \
        "the per-job LoRA is patched by title; that lookup must still exist"


def test_the_image_does_not_bake_the_weights_in():
    """~18GB of models on top of torch does not build on a hosted runner."""
    assert "--from=models" not in DOCKERFILE
    assert "fetch_model.sh civit" not in DOCKERFILE, \
        "models are fetched at container start, not at build time"


def test_the_worker_fetches_models_before_starting_comfyui():
    """ComfyUI caches its model folder listing at boot."""
    entrypoint = (REPO / "entrypoint.sh").read_text()
    assert "fetch_models.sh" in entrypoint
    assert entrypoint.index("fetch_models.sh") < entrypoint.index("python3 main.py")


def test_the_runtime_stage_does_not_use_the_cuda_devel_image():
    """devel is ~5GB of toolchain that only the builder needs."""
    runtime = DOCKERFILE.split("AS runtime")[0].splitlines()[-1]
    assert "devel" not in runtime, f"runtime stage is on a devel base: {runtime}"


def test_application_code_is_copied_last():
    """Anything after a changed layer is rebuilt, so the files that change on
    every commit have to sit below the venv and ComfyUI."""
    src_at = DOCKERFILE.index("COPY src/")
    for earlier in ("COPY --from=builder /opt/venv", "COPY --from=builder /comfyui"):
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
