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


def test_models_land_where_the_dockerfile_copies_them_from():
    """The fetch RUNs and the COPY --from=models lines must agree on paths."""
    fetched = set(re.findall(r"(/models/\S+\.safetensors)", DOCKERFILE))
    copied = set(re.findall(r"COPY --from=models\s+(/models/\S+\.safetensors)", DOCKERFILE))
    assert fetched, "no models are fetched — did the stage get renamed?"
    assert fetched == copied, (
        f"fetched but never copied: {sorted(fetched - copied)}; "
        f"copied but never fetched: {sorted(copied - fetched)}")


def test_each_model_gets_its_own_layer():
    """One COPY per model is the whole cold-start argument; a single COPY of a
    directory would make one changed LoRA re-pull every checkpoint."""
    copies = re.findall(r"^COPY --from=models .*$", DOCKERFILE, flags=re.M)
    for line in copies:
        assert line.count(".safetensors") == 1, f"more than one model in: {line}"


def test_the_runtime_stage_does_not_use_the_cuda_devel_image():
    """devel is ~5GB of toolchain that only the builder needs."""
    runtime = DOCKERFILE.split("AS runtime")[0].splitlines()[-1]
    assert "devel" not in runtime, f"runtime stage is on a devel base: {runtime}"


def test_application_code_is_copied_last():
    """Anything after a changed layer is rebuilt, so the files that change on
    every commit have to sit below the models and the venv."""
    src_at = DOCKERFILE.index("COPY src/")
    for earlier in ("COPY --from=builder /opt/venv", "COPY --from=models"):
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
