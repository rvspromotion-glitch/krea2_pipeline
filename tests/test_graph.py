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


@pytest.mark.parametrize("version,mode", list(graph_mod._FILES))
def test_no_api_key_is_committed(version, mode):
    """The exports shipped a live Gemini key inline. It must never come back."""
    raw = (ROOT / "workflows" / graph_mod._FILES[(version, mode)]).read_text()
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
    for key, name in graph_mod._FILES.items():
        raw = json.loads((ROOT / "workflows" / name).read_text())
        assert any(n["class_type"] == "PathchSageAttentionKJ" for n in raw.values()), key


@pytest.mark.parametrize("version,mode", list(graph_mod._FILES))
def test_no_loaded_graph_switches_attention_backends(version, mode):
    graph = graph_mod.load(mode, version)
    assert not [nid for nid, n in graph.items()
                if n["class_type"] in graph_mod.BYPASS_CLASSES]


@pytest.mark.parametrize("version,mode", list(graph_mod._FILES))
def test_bypassing_reconnects_consumers_rather_than_orphaning_them(version, mode):
    """A removed pass-through must hand its consumers to its own upstream, or
    the LoRA chain loses the checkpoint and the job fails validation instead."""
    raw = json.loads((ROOT / "workflows" / graph_mod._FILES[(version, mode)]).read_text())
    sage = [nid for nid, n in raw.items() if n["class_type"] == "PathchSageAttentionKJ"]
    upstream = {nid: raw[nid]["inputs"]["model"] for nid in sage}
    consumers = {
        (nid, field): raw[nid]["inputs"][field][0]
        for nid, n in raw.items()
        for field, v in (n.get("inputs") or {}).items()
        if isinstance(v, list) and len(v) == 2 and v[0] in sage
    }
    assert consumers, "nothing consumed the sage node — check the fixture"

    graph = graph_mod.load(mode, version)

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


# ── The Flux2 cleanup pass ───────────────────────────────────────────────────
#
# krea2 is already asked to remove tattoos in the Gemini system prompt, but
# asking a generator not to draw something is far less reliable than editing it
# out afterwards. So the anchor goes krea2 → Flux2 Klein → krea2 refine, and the
# refine at denoise 0.2 puts the krea2 look back over Flux's output.
#
# The placement is the load-bearing part in the carousel: every panel is
# generated *from* the anchor image — as the edit model's source latent and as
# the grounding image — so cleaning the anchor cleans all four published slides
# with one pass rather than four.

FLUX_MODELS = {
    "Flux2-Klein-9B-True-V3-int8mixedrow.safetensors",
    "qwen_3_8b_fp8mixed.safetensors",
    "flux2-vae.safetensors",
}


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_the_cleanup_pass_sits_between_the_two_krea2_samplers(mode):
    """Not after the refine. The refine is what re-imposes the krea2 look on
    Flux's output; downstream of it, Flux's render would be what publishes."""
    g = graph_mod.load(mode)

    # krea2 pass 1 → decode → encode(flux) → flux sampler → decode → encode(krea2)
    assert g["2439"]["inputs"]["samples"] == ["2314", 0]
    assert g["2440"]["inputs"]["pixels"] == ["2439", 0]
    assert g["2434"]["inputs"]["latent_image"] == ["2454", 0]
    assert g["2442"]["inputs"]["samples"] == ["2434", 0]
    assert g["2444"]["inputs"]["pixels"] == ["2442", 0]
    assert g["2346"]["inputs"]["latent_image"] == ["2444", 0], \
        "the krea2 refine must read the cleaned latent, not the raw one"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_each_vae_encode_uses_the_matching_decoder(mode):
    """Krea2 and Flux2 have different latent spaces. Crossing the VAEs here
    produces noise that still renders, still saves, and still publishes."""
    g = graph_mod.load(mode)
    krea2_vae = [nid for nid, n in g.items()
                 if n["class_type"] == "VAELoader"
                 and "Wan" in n["inputs"]["vae_name"]][0]
    flux_vae = [nid for nid, n in g.items()
                if n["class_type"] == "VAELoader"
                and "flux2" in n["inputs"]["vae_name"]][0]

    assert g["2439"]["inputs"]["vae"] == [krea2_vae, 0]   # krea2 latent out
    assert g["2440"]["inputs"]["vae"] == [flux_vae, 0]    # into flux space
    assert g["2442"]["inputs"]["vae"] == [flux_vae, 0]    # flux latent out
    assert g["2444"]["inputs"]["vae"] == [krea2_vae, 0]   # back to krea2 space


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_the_cleanup_prompt_survives_patching(mode):
    """It is a fixed correction, not a per-persona one — the {subject}
    substitution must not touch it and it must never come back empty."""
    g = graph_mod.patch(mode, **BASE)

    text = g["2438"]["inputs"]["text"]
    assert "tattoo" in text.lower()
    assert "{subject}" not in text
    assert g["2453"]["inputs"]["text"] == "", "the negative is deliberately empty"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_the_flux_pass_renders_at_the_anchor_size(mode):
    """A cleanup that resamples to a different size would land back in the
    krea2 refine at the wrong resolution."""
    g = graph_mod.load(mode)

    assert g["2454"]["class_type"] == "EmptyFlux2LatentImage"
    assert g["2454"]["inputs"]["width"] == g["2317"]["inputs"]["width"]
    assert g["2454"]["inputs"]["height"] == g["2317"]["inputs"]["height"]


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_both_reference_latents_point_at_the_source(mode):
    """Flux2's edit pattern: the empty latent is the canvas and the reference
    carries the image. A reference wired to the canvas would edit nothing."""
    g = graph_mod.load(mode)

    assert g["2451"]["inputs"]["latent"] == ["2440", 0]
    assert g["2452"]["inputs"]["latent"] == ["2440", 0]
    assert g["2434"]["inputs"]["positive"] == ["2451", 0]
    assert g["2434"]["inputs"]["negative"] == ["2452", 0]


def test_the_carousel_panels_are_built_from_the_cleaned_anchor():
    """The whole reason one pass is enough. If a panel ever stopped reading the
    anchor, three of four published slides would keep their tattoos and nothing
    would say so."""
    g = graph_mod.load("carousel")

    # anchor refine → decode → resize; that resize is what the panels consume.
    assert g["2318"]["inputs"]["samples"] == ["2346", 0]
    assert g["1031"]["inputs"]["image"] == ["2318", 0]
    assert g["2408:2406"]["inputs"]["image"] == ["1031", 0]

    consumers = {f"{nid}.{field}"
                 for nid, n in g.items()
                 for field, v in n["inputs"].items()
                 if isinstance(v, list) and len(v) == 2 and v[0] == "2408:2406"}
    # Every panel's edit-model source latent and every panel's grounding image.
    assert "1032.pixels" in consumers, "panels lost their source latent"
    for encoder in ("1036", "1037", "2102", "2103"):
        assert f"{encoder}.image" in consumers, f"{encoder} stopped grounding on the anchor"


def test_the_anchor_slide_published_is_the_cleaned_one():
    """It is batched into the carousel as well as feeding the panels."""
    g = graph_mod.load("carousel")
    assert g["2330"]["inputs"]["image2"] == ["1031", 0]


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_the_flux_models_are_loaded_by_the_expected_node_types(mode):
    g = graph_mod.load(mode)
    loaded = {n["inputs"].get("unet_name") or n["inputs"].get("clip_name")
              or n["inputs"].get("vae_name")
              for n in g.values()
              if n["class_type"] in ("UNETLoader", "CLIPLoader", "VAELoader")}

    assert FLUX_MODELS <= loaded
    assert g["2436"]["inputs"]["type"] == "flux2", \
        "the Flux2 text encoder needs the flux2 CLIP type, not krea2's"


# ── Workflow versions ────────────────────────────────────────────────────────
#
# v2 is a second generation of the same two graphs. What makes it swappable is
# not that it looks like v1 but that it still answers to the same four job
# variables and still ends in one SaveImage. These pin that.

@pytest.mark.parametrize("version,mode", list(graph_mod._FILES))
def test_every_version_takes_the_same_job_variables(version, mode):
    graph = graph_mod.patch(
        mode,
        image_filename="ref.png",
        lora_name="Chloe_v1.safetensors",
        trigger_word="ch10e",
        description="young woman with red hair",
        gemini_api_key="KEY",
        seed=7,
        version=version,
    )

    image = graph_mod._by_title(graph, graph_mod.TITLE_INPUT_IMAGE)[0]
    lora = graph_mod._by_title(graph, graph_mod.TITLE_CHARACTER_LORA)[0]
    assert graph[image]["inputs"]["image"] == "ref.png"
    assert graph[lora]["inputs"]["lora_name"] == "Chloe_v1.safetensors"
    assert graph_mod.output_node(graph)

    prompts = [n["inputs"]["value"] for n in graph.values()
               if n["class_type"] == "PrimitiveStringMultiline"]
    assert prompts, "no prompt template survived patching"
    assert all("ch10e, young woman with red hair" in p for p in prompts)
    assert not any(graph_mod.SUBJECT_PLACEHOLDER in p for p in prompts)


@pytest.mark.parametrize("version,mode", list(graph_mod._FILES))
def test_every_link_resolves_after_patching(version, mode):
    """A dangling link is a ComfyUI validation failure at hour three, not here."""
    graph = graph_mod.patch(
        mode, image_filename="r.png", lora_name="l.safetensors",
        trigger_word="t", description="d", gemini_api_key="K", version=version)

    dangling = [
        f"{nid}.{field} -> {value[0]}"
        for nid, node in graph.items()
        for field, value in node["inputs"].items()
        if isinstance(value, list) and len(value) == 2
        and isinstance(value[0], str) and value[0] not in graph
    ]
    assert not dangling, dangling


def test_v1_is_the_default():
    """A version that has to be asked for cannot become the default by accident."""
    assert graph_mod.DEFAULT_VERSION == "v1"
    assert graph_mod.load("single") == graph_mod.load("single", "v1")


def test_an_unknown_version_falls_back_rather_than_failing_the_render():
    assert graph_mod.normalise_version("v2") == "v2"
    assert graph_mod.normalise_version("V2") == "v2"
    assert graph_mod.normalise_version("v3") == graph_mod.DEFAULT_VERSION
    assert graph_mod.normalise_version(None) == graph_mod.DEFAULT_VERSION


def test_load_rejects_a_version_it_does_not_have():
    """normalise_version is the forgiving door; load itself is not."""
    with pytest.raises(graph_mod.GraphError, match="unknown version"):
        graph_mod.load("single", "v3")


def test_the_two_versions_are_actually_different_graphs():
    """Guards against a copy-paste that ships v1 twice under two names."""
    assert graph_mod.load("single", "v1") != graph_mod.load("single", "v2")
    assert graph_mod.load("carousel", "v1") != graph_mod.load("carousel", "v2")


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v2_keeps_the_flux_cleanup_pass(mode):
    """It was deleted from the graph that was handed over for quick testing, and
    without it v2 would silently drop the tattoo and jewellery removal."""
    graph = graph_mod.load(mode, "v2")

    encoders = [n for n in graph.values() if n["class_type"] == "CLIPTextEncode"]
    assert any("body jewellery" in (n["inputs"].get("text") or "") for n in encoders), \
        "the flux2 cleanup prompt is missing from v2"
    assert any(n["class_type"] == "ReferenceLatent" for n in graph.values())


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_the_negative_prompt_does_not_negate_itself(mode):
    """A negative prompt lists what to steer away from, so "no piercings" there
    inverts the entry it prefixes. The terms have to be stated plainly."""
    graph = graph_mod.load(mode, "v2")

    negatives = {
        nid for node in graph.values()
        if node["class_type"] == "KSampler"
        for nid in [node["inputs"].get("negative", [None])[0]]
        if nid in graph
    }
    for nid in negatives:
        text = str(graph[nid]["inputs"].get("prompt") or graph[nid]["inputs"].get("text") or "")
        assert " no " not in f" {text} ", f"{nid} negates its own terms: {text!r}"
        assert not text.strip().startswith("no "), f"{nid} negates its own terms: {text!r}"


def test_the_negative_prompt_still_names_the_things_being_removed():
    graph = graph_mod.load("single", "v2")
    negative = [n for n in graph.values()
                if (n.get("_meta") or {}).get("title") == "EDIT: instruction (negative)"]
    assert len(negative) == 1
    text = negative[0]["inputs"]["prompt"]
    for term in ("tattoo", "piercing", "nose stud", "navel piercing"):
        assert term in text, term
