#!/usr/bin/env python3
"""Which models a given workflow version actually loads.

The worker has no network volume, so every cold start refetches whatever it is
told to — and it was told to fetch the union of all fourteen graphs. v9 needs
eight of those; the other nine are ~47GB of Flux weights and superseded
checkpoints downloaded to sit on disk unread until the container is reclaimed.

The graphs already declare their own models, so this derives the set rather
than asking anyone to keep a second list in step. That matters more than the
saved typing: a hand-kept list drifts, and the way it fails is a cold start
that succeeds and a render that dies on a missing file an hour later.

Deliberately every node in the file, not only the ones reachable from the
output. A node that cannot affect the image is still a node ComfyUI may
validate, and the cost of being wrong in that direction is a boot that refuses
every job. Dead nodes are worth deleting from the graph — v9 carried a
UNETLoader nothing read — but that is a decision to make in the workflow, not
one to infer here.

    models_for.py v9            -> filenames v9 loads
    models_for.py v8 v9         -> the union of both
    models_for.py --all         -> every version (the old behaviour)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import graph as graph_mod  # noqa: E402


def models_in(path: Path) -> set:
    out = set()
    for node in json.loads(path.read_text()).values():
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, str) and value.endswith(".safetensors"):
                out.add(value)
            # rgthree's Power Lora Loader keeps its rows as nested dicts. A row
            # switched off still names a file, and it is included: whether a
            # disabled row is validated is the node pack's business, not a bet
            # worth taking on a boot path.
            elif isinstance(value, dict) and \
                    str(value.get("lora", "")).endswith(".safetensors"):
                out.add(value["lora"])
    return out


def models_for(versions) -> set:
    wanted = set()
    for version in versions:
        if version not in graph_mod.VERSIONS:
            raise SystemExit(
                f"[models] unknown version {version!r}; "
                f"expected some of {', '.join(graph_mod.VERSIONS)}")
        # Its own graphs only. A version that borrows another's carousel does
        # not need that carousel's weights downloaded — see graph.BORROWED.
        for name in graph_mod.own_graphs(version):
            path = graph_mod.WORKFLOW_DIR / name
            if path.exists():
                wanted |= models_in(path)

    # The per-persona LoRA is patched in per job and fetched from Radar, so it
    # is a placeholder here rather than a filename anyone could download.
    return {m for m in wanted if not m.startswith("{")}


if __name__ == "__main__":
    args = sys.argv[1:]
    versions = graph_mod.VERSIONS if (not args or args == ["--all"]) else args
    for name in sorted(models_for(versions)):
        print(name)
