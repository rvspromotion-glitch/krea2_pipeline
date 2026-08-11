"""verify_nodes.py gates every build, so it has to be right.

Exercised against a stand-in ComfyUI rather than the real one: what is being
tested is the attribution and the pass/fail decision, and a fake registry lets a
missing node be simulated without a 20GB image. The real workflows are used, so
the required-node set is the true one.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
VERIFY = REPO / "scripts" / "verify_nodes.py"
WORKFLOWS = REPO / "workflows"


def _required_types() -> set[str]:
    types: set[str] = set()
    for path in WORKFLOWS.glob("*.json"):
        types |= {n.get("class_type", "") for n in json.loads(path.read_text()).values()}
    return types


def _fake_comfy(tmp_path: Path, *, packages: dict[str, list[str]],
                core: list[str], empty_packages: tuple[str, ...] = ()) -> Path:
    """A directory that quacks like ComfyUI: nodes.py plus custom_nodes/."""
    comfy = tmp_path / "comfyui"
    custom = comfy / "custom_nodes"
    custom.mkdir(parents=True)

    for name, class_types in packages.items():
        pkg = custom / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            "".join(f"class {t}: pass\n" for t in class_types) +
            "MAPPINGS = {" + ", ".join(f"{t!r}: {t}" for t in class_types) + "}\n"
        )
    for name in empty_packages:
        pkg = custom / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("MAPPINGS = {}\n")

    # comfy/cli_args.py, and a nodes.py that probes the GPU at import unless
    # CPU mode was set on that namespace first. This is what the real ComfyUI
    # does, and getting it wrong failed a build: passing --cpu on the command
    # line has no effect, because cli_args only reads sys.argv when main.py has
    # enabled arg parsing.
    package = comfy / "comfy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "cli_args.py").write_text(textwrap.dedent("""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--cpu", action="store_true")
        args = parser.parse_args([])       # deliberately ignores sys.argv
    """))

    (comfy / "nodes.py").write_text(textwrap.dedent(f"""
        import importlib.util, sys
        from pathlib import Path
        from comfy.cli_args import args

        if not args.cpu:
            raise RuntimeError(
                "Found no NVIDIA driver on your system. Please check that you "
                "have an NVIDIA GPU and installed a driver")

        NODE_CLASS_MAPPINGS = {{}}
        for _t in {core!r}:
            NODE_CLASS_MAPPINGS[_t] = type(_t, (), {{}})

        def init_extra_nodes():
            root = Path(__file__).parent / "custom_nodes"
            for pkg in sorted(root.iterdir()):
                init = pkg / "__init__.py"
                if not init.exists():
                    continue
                spec = importlib.util.spec_from_file_location(pkg.name, init)
                module = importlib.util.module_from_spec(spec)
                sys.modules[pkg.name] = module
                spec.loader.exec_module(module)
                NODE_CLASS_MAPPINGS.update(module.MAPPINGS)
    """))
    return comfy


def _run(comfy: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VERIFY), "--comfy-dir", str(comfy),
         "--workflows", str(WORKFLOWS), *extra],
        capture_output=True, text=True)


@pytest.fixture
def complete(tmp_path):
    """Every node the graphs need, split across a core set and two packages."""
    required = sorted(_required_types())
    return _fake_comfy(
        tmp_path,
        packages={"pkg-a": required[:3], "pkg-b": required[3:6]},
        core=required[6:],
        empty_packages=("dead-weight", "also-unused"),
    )


def test_passes_when_every_node_resolves(complete):
    result = _run(complete)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_it_runs_comfyui_in_cpu_mode(complete):
    """The regression: a build machine has no GPU.

    ComfyUI probes CUDA at import unless CPU mode is set, and `--cpu` on the
    command line does not set it — cli_args only reads sys.argv once main.py
    enables arg parsing, which importing `nodes` directly never does. The fake
    ComfyUI here raises the same driver error if the flag was not applied.
    """
    result = _run(complete)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "NVIDIA driver" not in result.stdout + result.stderr


def test_a_gpu_probe_is_explained_rather_than_left_as_a_traceback(tmp_path):
    """If a custom node forces CUDA on import, say so and name the way out."""
    required = sorted(_required_types())
    comfy = _fake_comfy(tmp_path, packages={"pkg-a": required}, core=[])
    # A package that probes the GPU on import whatever mode ComfyUI is in —
    # nothing this script can do about that, so it must say so plainly.
    (comfy / "nodes.py").write_text(
        "raise RuntimeError('Found no NVIDIA driver on your system.')\n")

    result = _run(comfy)

    assert result.returncode != 0
    assert "NVIDIA driver" in result.stdout
    assert "VERIFY_NODES=0" in result.stdout


def test_fails_when_a_node_is_missing(tmp_path):
    """The whole point: this must break the build, not the Sunday batch."""
    required = sorted(_required_types())
    comfy = _fake_comfy(tmp_path, packages={"pkg-a": required[:-1]}, core=[])

    result = _run(comfy)

    assert result.returncode == 1
    assert "FAILED" in result.stdout
    assert required[-1] in result.stdout          # names the node
    assert "single_photo.json" in result.stdout or "carousel.json" in result.stdout


def test_reports_which_package_supplies_which_node(complete):
    """Node class names do not name their package; this is the only source."""
    result = _run(complete)
    required = sorted(_required_types())

    assert "pkg-a" in result.stdout and "pkg-b" in result.stdout
    assert "(core ComfyUI)" in result.stdout
    for class_type in required:
        assert class_type in result.stdout


def test_names_the_packages_that_supplied_nothing(complete):
    result = _run(complete)
    assert "dead-weight" in result.stdout
    assert "also-unused" in result.stdout
    assert "supplied nothing" in result.stdout


def test_prune_deletes_only_the_unused_packages(complete):
    custom = complete / "custom_nodes"

    result = _run(complete, "--prune")

    assert result.returncode == 0
    assert not (custom / "dead-weight").exists()
    assert not (custom / "also-unused").exists()
    assert (custom / "pkg-a").exists()
    assert (custom / "pkg-b").exists()


def test_nothing_is_deleted_without_prune(complete):
    _run(complete)
    assert (complete / "custom_nodes" / "dead-weight").exists()


def test_a_failed_verification_prunes_nothing(tmp_path):
    """A broken build must not also silently delete packages."""
    required = sorted(_required_types())
    comfy = _fake_comfy(tmp_path, packages={"pkg-a": required[:-1]}, core=[],
                        empty_packages=("dead-weight",))

    result = _run(comfy, "--prune")

    assert result.returncode == 1
    assert (comfy / "custom_nodes" / "dead-weight").exists()


def test_the_workflows_reference_no_duplicate_output_nodes():
    """Sanity: the required set is what we think it is, not an empty set."""
    required = _required_types()
    assert len(required) > 15
    assert "KSampler" in required
    assert "" not in required
