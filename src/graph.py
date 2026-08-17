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
import os
import random
import sys
from pathlib import Path
from typing import Any

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"

MODES = ("single", "carousel")

# Two generations of the same two graphs, switchable per job.
#
# v2 replaces the all-in-one checkpoint with a raw UNET plus a turbo LoRA, adds
# a second realism LoRA, and runs fewer sampler steps on a different
# sampler/scheduler pair. Both still take the same four job variables and still
# end in one SaveImage, which is what lets the worker treat them as
# interchangeable — everything below this line is version-agnostic on purpose.
#
# v1 stays the default. It is what has been rendering, and a version that has
# to be asked for cannot become the default by accident.
VERSIONS = ("v1", "v2", "v3", "v4", "v5")
DEFAULT_VERSION = "v1"

_FILES = {
    ("v1", "single"):   "single_photo.json",
    ("v1", "carousel"): "carousel.json",
    ("v2", "single"):   "single_photo_v2.json",
    ("v2", "carousel"): "carousel_v2.json",
    ("v3", "single"):   "single_photo_v3.json",
    ("v3", "carousel"): "carousel_v3.json",
    ("v4", "single"):   "single_photo_v4.json",
    ("v4", "carousel"): "carousel_v4.json",
    # v5 replaces the hero's Krea2 detail pass with a KSamplerAdvanced that
    # starts partway through its schedule, and lifts the identity-edit LoRA.
    # Its carousel is v4's with that one change at the front — the slide chain
    # after the hero is untouched.
    ("v5", "single"):   "single_photo_v5.json",
    ("v5", "carousel"): "carousel_v5.json",
}


def normalise_version(raw: str | None) -> str:
    """Unknown or empty falls back to the default rather than failing a render.

    A typo in a persona's setting should cost the v2 experiment, not the day's
    output — and the log line says which version actually ran.
    """
    value = (raw or "").strip().lower()
    if value not in VERSIONS:
        if value:
            print(f"[graph] unknown workflow version {raw!r}, using {DEFAULT_VERSION}",
                  file=sys.stderr, flush=True)
        return DEFAULT_VERSION
    return value

# Titles carry the role. They are stable across re-export (ComfyUI keeps node
# titles); ids are not.
TITLE_INPUT_IMAGE = "Input image"
TITLE_CHARACTER_LORA = "Character lora"

# v3 only. Flux edits the scraped frame into the persona, so it needs the
# persona's own face as a second reference and its own LoRA — a Krea2 LoRA
# cannot load into Flux. Patched when the graph has the slot and skipped when it
# does not, which is what lets one patch() serve all three versions.
TITLE_PERSONA_REFERENCE = "Persona reference"
TITLE_FLUX_CHARACTER_LORA = "Flux character lora"

# v4 stacks a fixed style LoRA under the per-persona one on the Flux side. It is
# the same file for every persona, so it is fetched with the shared weights and
# is deliberately NOT a patch target — it only needs its own title so the
# per-persona lookup stays unambiguous.
TITLE_FLUX_STYLE_LORA = "Flux style LoRA"

SUBJECT_PLACEHOLDER = "{subject}"

# The narrowest seed input in these graphs, not the widest. Ask_Gemini_Batch
# caps at 2**31 and rejects the whole prompt above it — "Value 3122519035678089
# bigger than max of 2147483648" fails validation before anything renders. Every
# other node accepts this range too, so one ceiling covers all of them.
SEED_MAX = 2**31 - 1


# ── Nodes the worker refuses to run ──────────────────────────────────────────
#
# PathchSageAttentionKJ replaces ComfyUI's attention with SageAttention's Triton
# kernels. Those kernels do not compile for the GPUs this endpoint actually gets
# — Blackwell, sm_120 — and the failure is not a fallback, it is the job:
#
#   AccelerateMatmul.cpp:40 ... Assertion `false && "computeCapability not
#   supported"' failed
#   RuntimeError: PassManager::run failed
#
# raised inside the first KSampler step, after the twelve-gigabyte checkpoint has
# already been staged. Every render on 12 Aug died there. ComfyUI's own default
# (pytorch/SDPA attention, which it logs at boot) works on the same card, so the
# node buys nothing here and costs everything.
#
# Stripped at load rather than deleted from the JSON on purpose: these workflows
# are re-exported from ComfyUI by hand, the node is still in the desktop graph,
# and a re-export would otherwise walk this straight back in.
#
# Maps class_type -> {output index: the input that output is a pass-through of}.
# Consumers are rewired to that input and the node is dropped.
BYPASS_CLASSES: dict[str, dict[int, str]] = {
    "PathchSageAttentionKJ": {0: "model"},
}

SAGE_ENV = "KREA2_SAGE_ATTENTION"


def _sage_requested() -> bool:
    """Escape hatch for an endpoint on a card SageAttention does support."""
    return os.environ.get(SAGE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


class GraphError(RuntimeError):
    """The graph is not shaped the way the patcher expects."""


def bypass(graph: dict, class_type: str, passthrough: dict[int, str]) -> list[str]:
    """Remove every node of `class_type`, reconnecting its consumers upstream.

    Returns the ids removed. A node whose pass-through input is a literal rather
    than a link is a hard error: there is nothing to reconnect to, and silently
    leaving it in place would reintroduce exactly the crash this exists to stop.
    """
    removed = []
    for nid in [n for n, node in graph.items() if node.get("class_type") == class_type]:
        inputs = graph[nid].get("inputs") or {}
        for out_idx, in_name in passthrough.items():
            upstream = inputs.get(in_name)
            if not (isinstance(upstream, list) and len(upstream) == 2):
                raise GraphError(
                    f"cannot bypass {class_type} node {nid}: its {in_name!r} input is "
                    f"{upstream!r}, not a link to another node"
                )
            for other in graph.values():
                for key, value in (other.get("inputs") or {}).items():
                    if (isinstance(value, list) and len(value) == 2
                            and value[0] == nid and value[1] == out_idx):
                        other["inputs"][key] = list(upstream)
        del graph[nid]
        removed.append(nid)
    return removed


def sanitise(graph: dict) -> dict:
    """Drop the nodes this hardware cannot run. Mutates and returns `graph`."""
    if _sage_requested():
        return graph
    for class_type, passthrough in BYPASS_CLASSES.items():
        removed = bypass(graph, class_type, passthrough)
        if removed:
            print(
                f"[graph] bypassed {len(removed)} {class_type} node(s) {removed} — "
                f"SageAttention's Triton kernels do not build for this GPU; set "
                f"{SAGE_ENV}=1 to keep them.",
                file=sys.stderr, flush=True,
            )
    return graph


def load(mode: str, version: str = DEFAULT_VERSION) -> dict:
    if mode not in MODES:
        raise GraphError(f"unknown mode {mode!r}, expected one of {MODES}")
    if version not in VERSIONS:
        raise GraphError(f"unknown version {version!r}, expected one of {VERSIONS}")
    path = WORKFLOW_DIR / _FILES[(version, mode)]
    if not path.exists():
        raise GraphError(f"workflow missing: {path}")
    return sanitise(json.loads(path.read_text()))


# ── Node lookup ──────────────────────────────────────────────────────────────

def _optional_by_title(graph: dict, title: str) -> str | None:
    """The node with this title, or None. More than one is still an error.

    Used for the roles only some versions have. Absent is fine; ambiguous is
    not — two nodes claiming the same role means a re-export went wrong, and
    guessing between them is how the wrong persona ships.
    """
    hits = [nid for nid, n in graph.items()
            if (n.get("_meta") or {}).get("title") == title]
    if len(hits) > 1:
        raise GraphError(f"expected at most one node titled {title!r}, found {hits}")
    return hits[0] if hits else None


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
    version: str = DEFAULT_VERSION,
    persona_reference: str | None = None,
    flux_lora_name: str | None = None,
) -> dict:
    """Return a job-ready copy of the graph. The template on disk is untouched.

    Every version is patched through this one path. A v2 graph that had lost the
    {subject} placeholder, or its "Character lora" title, would fail here at job
    start rather than quietly rendering a stranger.
    """
    graph = copy.deepcopy(load(mode, version))

    subject = ", ".join(p for p in (trigger_word.strip(), description.strip()) if p)
    if not subject:
        raise GraphError("trigger_word and description are both empty")

    image_node = _by_title(graph, TITLE_INPUT_IMAGE)[0]
    graph[image_node]["inputs"]["image"] = image_filename

    lora_node = _by_title(graph, TITLE_CHARACTER_LORA)[0]
    graph[lora_node]["inputs"]["lora_name"] = lora_name

    # v3's two extra per-persona slots. A graph that has the slot and was given
    # nothing to put in it is a hard error: it would otherwise render whatever
    # persona happened to be saved in the exported file.
    persona_node = _optional_by_title(graph, TITLE_PERSONA_REFERENCE)
    if persona_node is not None:
        if not persona_reference:
            raise GraphError(
                f"{version}/{mode} has a {TITLE_PERSONA_REFERENCE!r} slot but no "
                f"persona_reference was given — it would render the face baked "
                f"into the exported graph")
        graph[persona_node]["inputs"]["image"] = persona_reference

    flux_lora_node = _optional_by_title(graph, TITLE_FLUX_CHARACTER_LORA)
    if flux_lora_node is not None:
        if not flux_lora_name:
            raise GraphError(
                f"{version}/{mode} has a {TITLE_FLUX_CHARACTER_LORA!r} slot but no "
                f"flux_lora_name was given")
        graph[flux_lora_node]["inputs"]["lora_name"] = flux_lora_name

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
