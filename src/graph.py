"""Load a ComfyUI API graph and patch in the per-job variables.

Only four things change between jobs — the reference photo, the character LoRA,
the subject line (trigger word + description), and the seeds. Everything else in
these graphs is set-and-forget, so this module deliberately refuses to touch
anything it was not asked to.

**Nodes are located by role, never by hardcoded id.** These graphs are exported
from ComfyUI by hand, and a re-export can renumber nodes. Injecting a LoRA into
the wrong node would not raise — it would quietly render the wrong persona's
face for a week. So every lookup asserts it found exactly what it expected, and
a mismatch is a hard failure at job start rather than a silent one at hour three.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"

MODES = ("single", "carousel")
_FILES = {"single": "single_photo.json", "carousel": "carousel.json"}

# Titles carry the role. They are stable across re-export (ComfyUI keeps node
# titles); ids are not.
TITLE_INPUT_IMAGE = "Input image"
TITLE_CHARACTER_LORA = "Character lora"

SUBJECT_PLACEHOLDER = "{subject}"

# ComfyUI rejects seeds outside this range.
SEED_MAX = 2**53 - 1


class GraphError(RuntimeError):
    """The graph is not shaped the way the patcher expects."""


def load(mode: str) -> dict:
    if mode not in MODES:
        raise GraphError(f"unknown mode {mode!r}, expected one of {MODES}")
    path = WORKFLOW_DIR / _FILES[mode]
    if not path.exists():
        raise GraphError(f"workflow missing: {path}")
    return json.loads(path.read_text())


# ── Node lookup ──────────────────────────────────────────────────────────────

def _by_title(graph: dict, title: str, expected: int = 1) -> list[str]:
    hits = [nid for nid, n in graph.items() if (n.get("_meta") or {}).get("title") == title]
    if len(hits) != expected:
        raise GraphError(
            f"expected {expected} node(s) titled {title!r}, found {len(hits)}: {hits}. "
            "The workflow was probably re-exported — re-check the node roles."
        )
    return hits


def _by_class(graph: dict, class_type: str, minimum: int = 1) -> list[str]:
    hits = [nid for nid, n in graph.items() if n.get("class_type") == class_type]
    if len(hits) < minimum:
        raise GraphError(
            f"expected at least {minimum} {class_type} node(s), found {len(hits)}"
        )
    return hits


# ── Patching ─────────────────────────────────────────────────────────────────

def _set_subject(graph: dict, subject: str) -> int:
    """Substitute the persona into every prompt template that carries the slot.

    The templates are identical boilerplate apart from this one span, which is
    why the trigger word and description are stored once per persona rather than
    as two full prompts that can drift apart.
    """
    patched = 0
    for node in graph.values():
        if node.get("class_type") != "PrimitiveStringMultiline":
            continue
        value = node["inputs"].get("value", "")
        if SUBJECT_PLACEHOLDER in value:
            # replace, not format: the prompt text is free-form and a stray
            # brace would make str.format raise.
            node["inputs"]["value"] = value.replace(SUBJECT_PLACEHOLDER, subject)
            patched += 1
    if patched == 0:
        raise GraphError("no prompt template contained the {subject} placeholder")
    return patched


def _randomise_seeds(graph: dict, rng: random.Random) -> int:
    """Every seed field, including the Gemini ones.

    The Gemini seeds matter as much as the sampler seeds: same reference image
    plus same seed yields the same description, hence the same picture. Missing
    them would make repeat runs of one photo identical.
    """
    count = 0
    for node in graph.values():
        for field in ("seed", "noise_seed"):
            if field in (node.get("inputs") or {}):
                node["inputs"][field] = rng.randint(0, SEED_MAX)
                count += 1
    return count


def patch(
    mode: str,
    *,
    image_filename: str,
    lora_name: str,
    trigger_word: str,
    description: str,
    gemini_api_key: str,
    seed: int | None = None,
) -> dict:
    """Return a job-ready copy of the graph. The template on disk is untouched."""
    graph = copy.deepcopy(load(mode))

    subject = ", ".join(p for p in (trigger_word.strip(), description.strip()) if p)
    if not subject:
        raise GraphError("trigger_word and description are both empty")

    image_node = _by_title(graph, TITLE_INPUT_IMAGE)[0]
    graph[image_node]["inputs"]["image"] = image_filename

    lora_node = _by_title(graph, TITLE_CHARACTER_LORA)[0]
    graph[lora_node]["inputs"]["lora_name"] = lora_name

    _set_subject(graph, subject)

    # Every Gemini node gets the key from config; none is baked into the file.
    for nid in _by_class(graph, "Ask_Gemini_Batch"):
        graph[nid]["inputs"]["api_key"] = gemini_api_key

    rng = random.Random(seed)
    _randomise_seeds(graph, rng)

    return graph


def output_node(graph: dict) -> str:
    """The single SaveImage whose results are the job's output."""
    hits = _by_class(graph, "SaveImage")
    if len(hits) != 1:
        raise GraphError(f"expected exactly one SaveImage, found {hits}")
    return hits[0]


def describe(graph: dict) -> dict[str, Any]:
    """Small summary for logs — what got injected, without dumping the graph."""
    return {
        "nodes": len(graph),
        "image": graph[_by_title(graph, TITLE_INPUT_IMAGE)[0]]["inputs"]["image"],
        "lora": graph[_by_title(graph, TITLE_CHARACTER_LORA)[0]]["inputs"]["lora_name"],
        "gemini_nodes": len(_by_class(graph, "Ask_Gemini_Batch")),
    }
