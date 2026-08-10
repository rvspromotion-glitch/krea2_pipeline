"""Graph contract tests.

These run without a GPU, ComfyUI, or a network. They exist because the failure
mode they guard against is silent: a re-exported workflow with renumbered nodes
would still render, just with the wrong persona's LoRA or last week's trigger
word — and you would not find out until a human looked at the output days later.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import graph as graph_mod  # noqa: E402

BASE = dict(
    image_filename="ref.png",
    lora_name="Eva_step-002750_identity.safetensors",
    trigger_word="3lm1ra",
    description="young woman with long platinum blonde hair",
    gemini_api_key="test-key",
)


# ── Shipped workflows ────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["single", "carousel"])
def test_workflow_loads_and_has_one_output(mode):
    g = graph_mod.load(mode)
    assert len(g) > 20
    assert graph_mod.output_node(g)          # raises unless exactly one SaveImage


@pytest.mark.parametrize("mode", ["single", "carousel"])
def test_no_dead_nodes_remain(mode):
    """Every node feeds the output, or it is waste in a 40GB image."""
    g = graph_mod.load(mode)
    used = set()
    for n in g.values():
        for v in (n.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                used.add(v[0])
    orphans = [k for k, n in g.items() if k not in used and n["class_type"] != "SaveImage"]
    assert orphans == []


@pytest.mark.parametrize("mode", ["single", "carousel"])
def test_every_link_points_at_a_node_that_exists(mode):
    g = graph_mod.load(mode)
    for nid, n in g.items():
        for key, v in (n.get("inputs") or {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, f"{nid}.{key} references missing node {v[0]}"


def test_carousel_anchor_no_longer_flows_through_saveimage():
    """The original export fed node 1031 from a SaveImage, which has no outputs.

    In the UI that goes unnoticed; headless it fails validation and takes the
    whole carousel stage with it.
    """
    g = graph_mod.load("carousel")
    for nid, n in g.items():
        for key, v in (n.get("inputs") or {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert g[v[0]]["class_type"] != "SaveImage", (
                    f"{nid}.{key} reads from SaveImage {v[0]} — SaveImage has no outputs"
                )


@pytest.mark.parametrize("mode", ["single", "carousel"])
def test_no_api_key_is_committed(mode):
    """The exports shipped a live Gemini key inline. It must never come back."""
    raw = (ROOT / "workflows" / graph_mod._FILES[mode]).read_text()
    assert "AIza" not in raw
    g = json.loads(raw)
    for n in g.values():
        if n["class_type"] == "Ask_Gemini_Batch":
            assert n["inputs"]["api_key"] == ""


# ── Patching ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["single", "carousel"])
def test_patch_injects_every_variable(mode):
    g = graph_mod.patch(mode, **BASE)
    summary = graph_mod.describe(g)

    assert summary["image"] == "ref.png"
    assert summary["lora"] == "Eva_step-002750_identity.safetensors"
    assert summary["gemini_nodes"] >= 1

    for n in g.values():
        if n["class_type"] == "Ask_Gemini_Batch":
            assert n["inputs"]["api_key"] == "test-key"


@pytest.mark.parametrize("mode", ["single", "carousel"])
def test_subject_replaces_the_placeholder_everywhere(mode):
    g = graph_mod.patch(mode, **BASE)
    subject = "3lm1ra, young woman with long platinum blonde hair"

    prompts = [n["inputs"]["value"] for n in g.values()
               if n["class_type"] == "PrimitiveStringMultiline"]
    assert prompts
    for p in prompts:
        assert "{subject}" not in p
        assert subject in p


def test_carousel_templates_both_prompts():
    """Anchor and carousel prompts drifted apart in the original export."""
    g = graph_mod.patch("carousel", **BASE)
    prompts = [n["inputs"]["value"] for n in g.values()
               if n["class_type"] == "PrimitiveStringMultiline"]
    assert len(prompts) == 2
    assert all("3lm1ra, young woman with long platinum blonde hair" in p for p in prompts)


@pytest.mark.parametrize("mode", ["single", "carousel"])
def test_seeds_are_randomised_including_the_gemini_ones(mode):
    """Same photo + same Gemini seed = same description = same picture."""
    a = graph_mod.patch(mode, seed=1, **BASE)
    b = graph_mod.patch(mode, seed=2, **BASE)

    def seeds(g):
        return {nid: n["inputs"].get("seed") for nid, n in g.items()
                if "seed" in (n.get("inputs") or {})}

    sa, sb = seeds(a), seeds(b)
    assert sa and sa.keys() == sb.keys()
    assert all(sa[k] != sb[k] for k in sa), "seeds did not change with the seed"

    gemini = [nid for nid, n in a.items() if n["class_type"] == "Ask_Gemini_Batch"]
    assert all(nid in sa for nid in gemini), "Gemini seeds were left fixed"


@pytest.mark.parametrize("mode", ["single", "carousel"])
def test_same_seed_reproduces_the_same_graph(mode):
    assert graph_mod.patch(mode, seed=7, **BASE) == graph_mod.patch(mode, seed=7, **BASE)


def test_patching_does_not_mutate_the_template_on_disk():
    graph_mod.patch("single", **{**BASE, "lora_name": "Other.safetensors"})
    fresh = graph_mod.load("single")
    lora = fresh[graph_mod._by_title(fresh, "Character lora")[0]]["inputs"]["lora_name"]
    assert lora != "Other.safetensors"


# ── Failure modes ────────────────────────────────────────────────────────────

def test_unknown_mode_is_rejected():
    with pytest.raises(graph_mod.GraphError):
        graph_mod.load("video")


def test_empty_subject_is_rejected():
    with pytest.raises(graph_mod.GraphError):
        graph_mod.patch("single", **{**BASE, "trigger_word": "", "description": ""})


def test_a_renamed_role_node_fails_loudly():
    """The whole point of title-based lookup: a bad re-export must not go quiet."""
    g = graph_mod.load("single")
    nid = graph_mod._by_title(g, "Character lora")[0]
    g[nid]["_meta"]["title"] = "renamed by accident"

    with pytest.raises(graph_mod.GraphError, match="Character lora"):
        graph_mod._by_title(g, "Character lora")
