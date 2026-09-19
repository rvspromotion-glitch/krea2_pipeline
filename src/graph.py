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
VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8")
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
    # v6 drops the Krea2 edit patch, supersamples the hero at 2x and brings it
    # back down before the compression pass, and gives Flux its own short
    # prompt assembled from a second Gemini call. Its carousel is v5's slide
    # chain behind that hero.
    ("v6", "single"):   "single_photo_v6.json",
    ("v6", "carousel"): "carousel_v6.json",
    # v7 is a single-photo flow only: a Flux hair/look edit on the anchor, then
    # a krea2 i2i pass with a fixed FameGrid+Fedor style stack under the
    # per-persona identity LoRA. It carries no carousel of its own — carousels
    # are being retired — so a v7 carousel falls back to v6's, which keeps a
    # stray carousel job from a KeyError rather than pretending v7 has one.
    ("v7", "single"):   "single_photo_v7.json",
    ("v7", "carousel"): "carousel_v6.json",
    # v8 drops Flux entirely: one krea2 chain off the anchor, a three-LoRA
    # stack in an rgthree Power Lora Loader, and a depth-masked realism pass
    # from ACG_Realism_Nodes at the end. Single only, like v7, so a stray
    # carousel job falls back to v6's rather than dying on a KeyError.
    ("v8", "single"):   "single_photo_v8.json",
    ("v8", "carousel"): "carousel_v6.json",
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

# v7 only. A Flux pass recolours/edits the scraped anchor before the krea2
# identity pass — for Eva, "change the hair to platinum blonde" — so the LoRA,
# which knows the persona's look, has less to fight. The instruction is
# per-persona (a brunette persona wants a different edit), so the graph carries
# a placeholder node and the worker writes the persona's own instruction into
# it. Patched when the graph has the slot, skipped when it does not.
TITLE_FLUX_EDIT = "Flux edit"

SUBJECT_PLACEHOLDER = "{subject}"

# v7 asks Gemini to refer to the person by the trigger word the character LoRA
# was trained on, so the scene description it writes anchors on the same token
# {subject} does. Older versions describe the anchor generically and never
# carry this, so its absence is not an error.
TRIGGER_PLACEHOLDER = "{trigger}"

# The persona's look, for prompts that state it separately from the trigger.
# v8's shot-director and system prompts both had Eva's hair and eye colour
# written into the prose, in three places between them.
DESCRIPTION_PLACEHOLDER = "{description}"

# rgthree's Power Lora Loader (v8) keeps its rows as nested dicts, so there is
# no lora_name field to write. The row carrying the persona is marked with this
# instead of being found by position: a stack gets reordered while someone is
# tuning strengths, and binding to the wrong row would quietly render another
# persona's face for a week — the failure this whole module is arranged to
# prevent.
CHARACTER_PLACEHOLDER = "{character}"

# Where a persona may be written. The Krea2 shot-director template has always
# been a PrimitiveStringMultiline; v6 added a second slot in the string the
# Flux prompt is concatenated from. See _set_subject.
SUBJECT_NODE_TYPES = ("PrimitiveStringMultiline", "StringConcatenate")

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

def _character_lora_slot(graph: dict, node_id: str):
    """Where the persona's LoRA lives in this node, as (container, key).

    Two shapes. Every version through v7 loads it with a LoraLoaderModelOnly,
    which has a flat ``lora_name``. v8 stacks three LoRAs in one rgthree Power
    Lora Loader, whose rows are nested dicts with no ``lora_name`` anywhere —
    writing that field there would succeed, change nothing, and leave the
    persona baked into the exported graph.

    The row is named by ``_meta.character_slot`` rather than found by position
    or by its placeholder value. Position would rebind to the wrong row the
    first time someone reorders the stack while tuning strengths, and the
    placeholder only exists until the first patch replaces it — this has to
    keep working on a graph that has already been patched, which is exactly
    what describe() reads.
    """
    node = graph[node_id]
    inputs = node["inputs"]
    if "lora_name" in inputs:
        return inputs, "lora_name"

    meta = node.get("_meta") or {}
    slot = meta.get("character_slot")
    row = inputs.get(slot) if slot else None
    if not isinstance(row, dict) or "lora" not in row:
        raise GraphError(
            f"{meta.get('title', node_id)!r} has no lora_name and no usable "
            f"_meta.character_slot (got {slot!r}) — without it the persona's "
            f"LoRA would not be the one that renders")
    return row, "lora"


def character_lora_of(graph: dict) -> str:
    """The persona LoRA currently in the graph, whichever shape holds it."""
    node_id = _by_title(graph, TITLE_CHARACTER_LORA)[0]
    container, key = _character_lora_slot(graph, node_id)
    return container[key]


def _set_subject(graph: dict, subject: str) -> int:
    """Substitute the persona into every prompt slot carrying the placeholder.

    The templates are identical boilerplate apart from this one span, which is
    why the trigger word and description are stored once per persona rather than
    as two full prompts that can drift apart.

    Two node types, because v6 writes the persona twice: the Krea2
    shot-director template is a ``PrimitiveStringMultiline``, and the Flux
    prompt is assembled by ``StringConcatenate`` from an opening line and a
    Gemini clothing description. Missing the second is not a crash — Flux would
    be handed the literal text "{subject}", the character LoRA would have no
    trigger word to anchor on, and the render would come back plausible but
    with a weaker identity swap. Exactly the kind of failure nobody finds.

    Deliberately a named list rather than a scan of every string in the graph:
    these are the only two places a persona is ever written, and a third would
    be a decision worth making on purpose.

    Returns the number of *fields* patched, which is two for v6 and one before.
    """
    patched = 0
    for node in graph.values():
        if node.get("class_type") not in SUBJECT_NODE_TYPES:
            continue
        for field, value in node["inputs"].items():
            # A linked input is a [node_id, slot] pair, not text.
            if not isinstance(value, str) or SUBJECT_PLACEHOLDER not in value:
                continue
            # replace, not format: the prompt text is free-form and a stray
            # brace would make str.format raise.
            node["inputs"][field] = value.replace(SUBJECT_PLACEHOLDER, subject)
            patched += 1
    if patched == 0:
        raise GraphError("no prompt template contained the {subject} placeholder")
    return patched


# Where a bare trigger or description may be written. Ask_Gemini_Batch holds
# v7's inline shot-director prompt; v8 moved most of that prose into a
# PrimitiveStringMultiline wired in as Gemini's system_instruction, which is a
# subject node type — so a scan limited to Gemini nodes would have left "Refer
# to her as `3lm1ra`" sitting in every persona's system prompt.
PROMPT_NODE_TYPES = ("Ask_Gemini_Batch",) + SUBJECT_NODE_TYPES


def _set_trigger(graph: dict, trigger_word: str, description: str = "") -> int:
    """Write the persona's trigger and look into the prompts that ask for them.

    Separate from {subject} because these prompts want the two facts apart: the
    trigger anchors Gemini's prose on the token the LoRA knows ("refer to her
    as X"), while the description states the look to keep ("always include
    ..."). v8 needs both, in three places across two nodes.

    A version carrying neither placeholder is left untouched — returns 0, not
    an error, because v1 through v6 have nothing of the kind.
    """
    replacements = [(TRIGGER_PLACEHOLDER, trigger_word)]
    if description:
        replacements.append((DESCRIPTION_PLACEHOLDER, description))

    patched = 0
    for node in graph.values():
        if node.get("class_type") not in PROMPT_NODE_TYPES:
            continue
        for field, value in node["inputs"].items():
            if not isinstance(value, str):
                continue
            updated = value
            for placeholder, replacement in replacements:
                updated = updated.replace(placeholder, replacement)
            if updated != value:
                node["inputs"][field] = updated
                patched += 1
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
    flux_edit_prompt: str | None = None,
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
    container, key = _character_lora_slot(graph, lora_node)
    container[key] = lora_name

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

    # v7's per-persona Flux edit. Same rule as the persona reference: a graph
    # that has the slot and was handed nothing would render the look baked into
    # the exported file (Eva's hair on every persona), so that is a hard error.
    flux_edit_node = _optional_by_title(graph, TITLE_FLUX_EDIT)
    if flux_edit_node is not None:
        if not flux_edit_prompt:
            raise GraphError(
                f"{version}/{mode} has a {TITLE_FLUX_EDIT!r} slot but no "
                f"flux_edit_prompt was given — it would render the edit baked "
                f"into the exported graph")
        graph[flux_edit_node]["inputs"]["text"] = flux_edit_prompt

    _set_subject(graph, subject)
    _set_trigger(graph, trigger_word, description.strip())

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
        "lora": character_lora_of(graph),
        "gemini_nodes": len(_by_class(graph, "Ask_Gemini_Batch")),
    }
