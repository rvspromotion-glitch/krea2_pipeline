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


# ── Links into nodes that produce nothing ────────────────────────────────────

# Terminal display/output nodes: RETURN_TYPES is empty, so ComfyUI's validation
# raises "tuple index out of range" on anything reading from them and rejects
# the whole prompt. Invisible in the UI, where the wire looks perfectly normal.
NO_OUTPUT_CLASSES = {
    "SaveImage", "PreviewImage", "PreviewAny", "SaveAnimatedWEBP", "SaveAnimatedPNG",
}


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_nothing_reads_from_an_output_less_node(mode):
    """This has now bitten twice: node 1031 fed from a SaveImage, and 2344's
    prompt fed from a PreviewAny. Both rejected the prompt at validation."""
    graph = graph_mod.load(mode)
    offenders = [
        f"{nid} ({node['class_type']}).{field} <- {value[0]} ({graph[value[0]]['class_type']})"
        for nid, node in graph.items()
        for field, value in (node.get("inputs") or {}).items()
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
        and graph.get(value[0], {}).get("class_type") in NO_OUTPUT_CLASSES
    ]
    assert not offenders, "links from a node with no outputs:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_an_encoder_reads_the_gemini_description_directly(mode):
    """Not through the preview node that happened to be showing it.

    Only *an* encoder, not every one: the carousel also has encoders fed by a
    fixed instruction string, which is what they are supposed to take.
    """
    graph = graph_mod.load(mode)
    sources = [graph[v[0]]["class_type"]
               for n in graph.values() if n["class_type"] == "Krea2EditGroundedEncode"
               for k, v in n["inputs"].items() if k == "prompt"
               and isinstance(v, list) and isinstance(v[0], str)]
    assert "Ask_Gemini_Batch" in sources, (
        f"no encoder reads the image description; prompts come from {sorted(set(sources))}")


# ── Seed range ───────────────────────────────────────────────────────────────

def test_seeds_fit_the_narrowest_node():
    """Ask_Gemini_Batch caps at 2**31 and rejects the whole prompt above it.

    One ceiling for every seed field, set by the tightest one — a 2**53 seed
    validates fine on KSampler and fails the job on the Gemini node.
    """
    assert graph_mod.SEED_MAX <= 2**31 - 1


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_every_generated_seed_is_in_range(mode):
    for seed in (None, 1, 2**53, 2**63 - 1):
        graph = graph_mod.patch(mode, seed=seed, **BASE)
        values = [v for node in graph.values()
                  for k, v in (node.get("inputs") or {}).items()
                  if k in ("seed", "noise_seed") and isinstance(v, int)]
        assert values
        assert all(0 <= v <= 2**31 - 1 for v in values), \
            f"out of range: {[v for v in values if v > 2**31 - 1]}"


# ── SageAttention ────────────────────────────────────────────────────────────
#
# On 12 Aug every render died in the first sampler step, after the 12GB
# checkpoint had already been staged:
#
#   [INFO] Using sage attention mode: auto
#   AccelerateMatmul.cpp:40 ... Assertion `false && "computeCapability not
#   supported"' failed
#   RuntimeError: PassManager::run failed
#
# SageAttention's Triton kernels do not compile for the endpoint's Blackwell
# cards. ComfyUI's own default attention runs the same graph on the same GPU.

def test_the_shipped_workflows_still_carry_the_sage_node():
    """If this ever fails the graphs changed and the rest of this section is
    testing nothing — which is the quiet way for the crash to come back."""
    for mode in graph_mod.MODES:
        raw = json.loads((ROOT / "workflows" / graph_mod._FILES[mode]).read_text())
        assert any(n["class_type"] == "PathchSageAttentionKJ" for n in raw.values())


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_no_loaded_graph_switches_attention_backends(mode):
    graph = graph_mod.load(mode)
    assert not [nid for nid, n in graph.items()
                if n["class_type"] in graph_mod.BYPASS_CLASSES]


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_bypassing_reconnects_consumers_rather_than_orphaning_them(mode):
    """A removed pass-through must hand its consumers to its own upstream, or
    the LoRA chain loses the checkpoint and the job fails validation instead."""
    raw = json.loads((ROOT / "workflows" / graph_mod._FILES[mode]).read_text())
    sage = [nid for nid, n in raw.items() if n["class_type"] == "PathchSageAttentionKJ"]
    upstream = {nid: raw[nid]["inputs"]["model"] for nid in sage}
    consumers = {
        (nid, field): raw[nid]["inputs"][field][0]
        for nid, n in raw.items()
        for field, v in (n.get("inputs") or {}).items()
        if isinstance(v, list) and len(v) == 2 and v[0] in sage
    }
    assert consumers, "nothing consumed the sage node — check the fixture"

    graph = graph_mod.load(mode)

    for (nid, field), was in consumers.items():
        assert graph[nid]["inputs"][field] == upstream[was], (
            f"{nid}.{field} was not reconnected to {upstream[was]}")


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_the_bypass_leaves_no_dangling_links(mode):
    graph = graph_mod.load(mode)
    for nid, n in graph.items():
        for field, v in (n.get("inputs") or {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in graph, f"{nid}.{field} -> missing {v[0]}"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_sage_can_be_kept_for_an_endpoint_whose_cards_support_it(mode, monkeypatch):
    monkeypatch.setenv(graph_mod.SAGE_ENV, "1")
    graph = graph_mod.load(mode)
    assert [nid for nid, n in graph.items()
            if n["class_type"] == "PathchSageAttentionKJ"]


def test_a_passthrough_fed_by_a_literal_is_a_hard_error():
    """Nothing to reconnect to. Dropping it silently would leave the consumer
    reading a node that no longer exists."""
    graph = {"1": {"class_type": "PathchSageAttentionKJ",
                   "inputs": {"model": "not-a-link"}}}
    with pytest.raises(graph_mod.GraphError, match="not a link"):
        graph_mod.bypass(graph, "PathchSageAttentionKJ", {0: "model"})
