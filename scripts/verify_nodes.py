"""Prove, at build time, that both graphs can actually load.

A missing custom node does not fail quietly at a convenient moment. ComfyUI
accepts the container, RunPod reports the worker healthy, and the first job of
the Sunday batch comes back with a validation error at 02:00. This runs inside
the image instead, so that failure happens on the build.

It also answers the question you cannot answer by reading the graphs: *which
package supplies which node*. Node class names do not name their package, so the
only honest way to trim custom_nodes.txt is to have something import the lot and
report the attribution. That is printed on every build; `--prune` acts on it.

No GPU required — registering nodes is an import, not a render.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import shutil
import sys
from pathlib import Path


def _force_cpu_mode() -> None:
    """Make ComfyUI pick the CPU device. There is no GPU on a build machine.

    Passing --cpu on the command line does not work here. comfy/cli_args.py only
    reads sys.argv when comfy.options.args_parsing has been turned on, and the
    only thing that turns it on is main.py — so importing `nodes` directly gets
    `parse_args([])`, every flag at its default, and model_management calls
    torch.cuda.current_device() at import. That is a hard failure on a runner:

        RuntimeError: Found no NVIDIA driver on your system

    So set the flag on the parsed namespace instead, before anything that reads
    it is imported. cli_args is imported by every ComfyUI module that cares, and
    `cpu` has been the attribute name throughout.
    """
    try:
        from comfy.cli_args import args as comfy_args
    except Exception as exc:                      # pragma: no cover
        print(f"[verify] note: could not preset ComfyUI args ({exc}); "
              f"continuing and hoping the import is CPU-safe")
        return
    comfy_args.cpu = True


def _load_comfy_registry(comfy_dir: Path):
    """Import ComfyUI far enough to populate NODE_CLASS_MAPPINGS."""
    sys.path.insert(0, str(comfy_dir))
    os.chdir(comfy_dir)
    _force_cpu_mode()

    import nodes  # noqa: E402  (only importable after the path insert above)

    # init_extra_nodes is the current name and is async on recent ComfyUI;
    # init_custom_nodes is the older sync one. Support both rather than pinning
    # this script to one ComfyUI generation.
    init = getattr(nodes, "init_extra_nodes", None) or getattr(nodes, "init_custom_nodes", None)
    if init is None:
        raise SystemExit("[verify] ComfyUI exposes no custom-node initialiser — "
                         "its API changed; this script needs updating")
    result = init()
    if inspect.isawaitable(result):
        asyncio.run(result)
    return nodes


def _required_class_types(workflow_dir: Path) -> dict[str, set[str]]:
    """Every class_type each graph references."""
    required: dict[str, set[str]] = {}
    for path in sorted(workflow_dir.glob("*.json")):
        graph = json.loads(path.read_text())
        required[path.name] = {n.get("class_type", "") for n in graph.values()}
    if not required:
        raise SystemExit(f"[verify] no workflows found in {workflow_dir}")
    return required


def _owning_package(cls, custom_nodes: Path) -> str | None:
    """Which custom_nodes/<package> defines this class, or None if it is core."""
    module = sys.modules.get(cls.__module__)
    file = getattr(module, "__file__", None)
    if not file:
        return None
    try:
        relative = Path(file).resolve().relative_to(custom_nodes.resolve())
    except ValueError:
        return None            # core ComfyUI, or a site-packages node
    return relative.parts[0] if relative.parts else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-dir", default=os.getenv("COMFYUI_PATH", "/comfyui"))
    parser.add_argument("--workflows", default="/app/workflows")
    parser.add_argument("--prune", action="store_true",
                        help="delete packages that supplied nothing to these graphs")
    args = parser.parse_args()

    comfy_dir = Path(args.comfy_dir)
    workflow_dir = Path(args.workflows)
    custom_nodes = comfy_dir / "custom_nodes"

    required = _required_class_types(workflow_dir)
    every = sorted(set().union(*required.values()))

    try:
        nodes = _load_comfy_registry(comfy_dir)
    except Exception as exc:
        # Distinguish "the image is wrong" from "this machine has no GPU", which
        # is normal here and never a reason to fail a build.
        text = f"{type(exc).__name__}: {exc}"
        print(f"\n[verify] could not load the node registry — {text}")
        if "NVIDIA driver" in text or "CUDA" in text or "cuda" in text:
            print("\n  This is a GPU probe at import time, not a missing node.\n"
                  "  ComfyUI should be in CPU mode here; if a custom node forces\n"
                  "  CUDA on import it cannot be checked on a build machine.\n"
                  "  Rebuild with --build-arg VERIFY_NODES=0 to skip this step,\n"
                  "  and check the graphs by running one job instead.")
        raise

    registry = nodes.NODE_CLASS_MAPPINGS

    missing = {name: sorted(t for t in types if t not in registry)
               for name, types in required.items()}
    missing = {k: v for k, v in missing.items() if v}

    installed = {p.name for p in custom_nodes.iterdir() if p.is_dir()} \
        if custom_nodes.exists() else set()
    used: dict[str, list[str]] = {}
    for class_type in every:
        cls = registry.get(class_type)
        if cls is None:
            continue
        owner = _owning_package(cls, custom_nodes)
        used.setdefault(owner or "(core ComfyUI)", []).append(class_type)

    print(f"\n[verify] {len(every)} distinct node types across "
          f"{len(required)} workflow(s)\n")
    for owner in sorted(used, key=lambda o: (o == "(core ComfyUI)", o.lower())):
        print(f"  {owner}")
        for class_type in sorted(used[owner]):
            print(f"      {class_type}")

    unused = sorted(installed - set(used) - {"__pycache__"})
    if unused:
        print(f"\n[verify] {len(unused)} package(s) supplied nothing to these graphs:")
        for name in unused:
            print(f"      {name}")
        print("  Removing them from custom_nodes.txt shrinks the image and cuts "
              "ComfyUI's boot time.\n  Check first that none of them patches "
              "something globally on import — that would not show up here.")

    if missing:
        print("\n[verify] FAILED — these node types are not registered:")
        for workflow, types in missing.items():
            for class_type in types:
                print(f"      {workflow}: {class_type}")
        print("\n  Add the package that provides them to custom_nodes.txt, or "
              "check whether\n  a package failed to import (look further up "
              "this build log for a traceback).")
        return 1

    if args.prune and unused:
        for name in unused:
            shutil.rmtree(custom_nodes / name, ignore_errors=True)
            print(f"[verify] pruned {name}")

    print("\n[verify] OK — every node both graphs need is registered\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
