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


def _job_kwargs(version):
    """The job variables every version takes, plus the two v3 introduced."""
    kwargs = dict(image_filename="ref.png", lora_name="Chloe_v1.safetensors",
                  trigger_word="ch10e", description="young woman with red hair",
                  gemini_api_key="KEY", seed=7, version=version)
    # Every version from v3 on renders through Flux as well as Krea2, so it
    # needs the persona's photo and its own Klein LoRA. Keyed off the graph
    # rather than a list of versions, so adding a version cannot leave this
    # behind — which is exactly what happened when v5 arrived.
    graph = graph_mod.load("single", version)
    if graph_mod._optional_by_title(graph, graph_mod.TITLE_PERSONA_REFERENCE):
        kwargs.update(persona_reference="chloe_face.png",
                      flux_lora_name="Chloe_klein.safetensors")
    return kwargs


@pytest.mark.parametrize("version,mode", list(graph_mod._FILES))
def test_every_version_takes_the_same_job_variables(version, mode):
    graph = graph_mod.patch(mode, **_job_kwargs(version))

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
    graph = graph_mod.patch(mode, **_job_kwargs(version))

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
    assert graph_mod.normalise_version("v3") == "v3"
    assert graph_mod.normalise_version("v9") == graph_mod.DEFAULT_VERSION
    assert graph_mod.normalise_version(None) == graph_mod.DEFAULT_VERSION


def test_load_rejects_a_version_it_does_not_have():
    """normalise_version is the forgiving door; load itself is not."""
    with pytest.raises(graph_mod.GraphError, match="unknown version"):
        graph_mod.load("single", "v9")


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


# ── v3: Flux generates, Krea2 details ────────────────────────────────────────
#
# v1 and v2 generate with Krea2 and use Flux as a cleanup pass. v3 inverts that:
# Flux edits the scraped frame into the persona, then Krea2 bashes the detail
# in. Everything Krea2 reads therefore has to point at the Flux render — left on
# the scraped frame, the edit patch pulls the picture back toward the woman who
# was just swapped out, which is the failure this section exists to prevent.

def _v3(mode):
    return graph_mod.load(mode, "v3")


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v3_gives_flux_two_references(mode):
    """One for the scene, one for the persona. Stock ReferenceLatent chains, so
    both ride a single conditioning without a custom node."""
    graph = _v3(mode)
    chained = [
        nid for nid, n in graph.items()
        if n["class_type"] == "ReferenceLatent"
        and isinstance(n["inputs"].get("conditioning"), list)
        and graph.get(n["inputs"]["conditioning"][0], {}).get("class_type") == "ReferenceLatent"
    ]
    assert chained, "no ReferenceLatent is chained onto another — only one reference reaches Flux"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v3_krea2_reads_the_flux_render_not_the_scraped_frame(mode):
    """The single most likely way a first v3 attempt comes out wrong."""
    graph = _v3(mode)

    flux_decodes = {
        nid for nid, n in graph.items()
        if n["class_type"] == "VAEDecode"
        and graph[n["inputs"]["vae"][0]]["inputs"].get("vae_name") == "flux2-vae.safetensors"
    }
    assert flux_decodes, "no decode uses the flux VAE — flux is not generating"

    for nid, n in graph.items():
        if n["class_type"] not in ("Krea2EditGroundedEncode", "Krea2EditModelPatch"):
            continue
        upstream = n["inputs"].get("image") or n["inputs"].get("source_latent")
        seen, node = set(), upstream
        while isinstance(node, list) and node[0] in graph and node[0] not in seen:
            seen.add(node[0])
            if node[0] in flux_decodes:
                break
            inputs = graph[node[0]]["inputs"]
            node = inputs.get("pixels") or inputs.get("samples") or inputs.get("image")
        else:
            raise AssertionError(
                f"{nid} ({n['class_type']}) is not grounded on a flux render")


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v3_has_no_krea2_generate_pass(mode):
    """Flux generates in v3. A leftover denoise-1 Krea2 sampler would throw the
    flux render away and paint a new picture over the top of it."""
    graph = _v3(mode)
    krea_vae = "Wan2_1_VAE_fp32.safetensors"
    full = [nid for nid, n in graph.items()
            if n["class_type"] == "KSampler" and float(n["inputs"]["denoise"]) >= 0.99
            and "FLUX" not in (n.get("_meta") or {}).get("title", "")]
    assert not full, f"Krea2 still generates from scratch in {mode}: {full}"


def test_v3_single_details_then_settles():
    """Two Krea2 passes on a single: 0.40 to stamp the look, 0.20 to settle."""
    graph = _v3("single")
    denoises = sorted(float(n["inputs"]["denoise"]) for n in graph.values()
                      if n["class_type"] == "KSampler"
                      and "FLUX" not in (n.get("_meta") or {}).get("title", ""))
    assert denoises == [0.2, 0.4]


def test_v3_carousel_settles_every_slide_the_same_way():
    """Four flux renders, four identical Krea2 settle passes — no slide gets a
    different amount of treatment to the others."""
    graph = _v3("carousel")
    flux_gens = [n for n in graph.values()
                 if (n.get("_meta") or {}).get("title", "").endswith(": generate")]
    settles = [n for n in graph.values()
               if "settle pass" in (n.get("_meta") or {}).get("title", "")]
    assert len(flux_gens) == 4, f"expected 4 flux generations, got {len(flux_gens)}"
    assert len(settles) == 4
    assert {float(n["inputs"]["denoise"]) for n in settles} == {0.2}


def test_v3_refuses_to_render_the_persona_baked_into_the_export():
    for field in ("persona_reference", "flux_lora_name"):
        kwargs = _job_kwargs("v3")
        kwargs.pop(field)
        with pytest.raises(graph_mod.GraphError, match="slot but no"):
            graph_mod.patch("single", **kwargs)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_the_older_versions_still_patch_without_the_v3_inputs(version):
    """v3's extra slots must not become required for graphs that lack them."""
    graph = graph_mod.patch("single", **_job_kwargs(version))
    assert graph_mod.output_node(graph)


def _model_chain(graph, nid, seen=None):
    """Walk a sampler's model input back to its loader, collecting LoRAs."""
    seen = seen or []
    node = graph.get(nid)
    if node is None:
        return seen
    if node["class_type"] == "LoraLoaderModelOnly":
        seen = seen + [(node.get("_meta") or {}).get("title")]
    upstream = node["inputs"].get("model")
    if isinstance(upstream, list) and upstream[0] in graph:
        return _model_chain(graph, upstream[0], seen)
    return seen


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v3_carries_both_character_loras(mode):
    """v3 runs two models, so a persona needs two LoRAs — a Krea2 one for the
    detail passes and a Klein one for the flux edit. Neither substitutes for the
    other, and a sampler that lost its own would render a stranger at full
    quality, which is the failure least likely to be noticed in review.
    """
    graph = graph_mod.load(mode, "v3")

    krea2_samplers, flux_samplers = [], []
    for nid, node in graph.items():
        if node["class_type"] != "KSampler":
            continue
        title = (node.get("_meta") or {}).get("title", "")
        (flux_samplers if title.startswith("FLUX") else krea2_samplers).append(nid)

    assert krea2_samplers and flux_samplers

    for nid in krea2_samplers:
        chain = _model_chain(graph, graph[nid]["inputs"]["model"][0])
        assert graph_mod.TITLE_CHARACTER_LORA in chain, \
            f"{nid} ({graph[nid]['_meta']['title']}) lost the Krea2 character LoRA: {chain}"

    for nid in flux_samplers:
        chain = _model_chain(graph, graph[nid]["inputs"]["model"][0])
        assert graph_mod.TITLE_FLUX_CHARACTER_LORA in chain, \
            f"{nid} ({graph[nid]['_meta']['title']}) lost the Klein character LoRA: {chain}"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_the_two_loras_do_not_cross_models(mode):
    """A Krea2 LoRA cannot load into Flux and vice versa — loading one into the
    wrong base is a job-time failure, not a quality one."""
    graph = graph_mod.load(mode, "v3")

    for nid, node in graph.items():
        if node["class_type"] != "KSampler":
            continue
        title = (node.get("_meta") or {}).get("title", "")
        chain = _model_chain(graph, node["inputs"]["model"][0])
        if title.startswith("FLUX"):
            assert graph_mod.TITLE_CHARACTER_LORA not in chain, \
                f"{nid} loads the Krea2 LoRA into Flux"
        else:
            assert graph_mod.TITLE_FLUX_CHARACTER_LORA not in chain, \
                f"{nid} loads the Klein LoRA into Krea2"


# ── v4 ───────────────────────────────────────────────────────────────────────
#
# v4 keeps v3's shape — Flux edits the scraped frame into the persona, Krea2
# details it — and changes what conditions Krea2: the grounded encoders are
# replaced by the KG reference stack, and a fixed style LoRA sits under the
# per-persona one on the Flux side.

@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v4_keeps_the_two_flux_loras_distinguishable(mode):
    """Both arrived titled 'Flux character lora'. The patcher looks nodes up by
    title, so an ambiguous one is refused rather than guessed — and guessing
    would have written the persona's name over the shared style LoRA."""
    graph = graph_mod.load(mode, "v4")

    per_persona = [nid for nid, n in graph.items()
                   if (n.get("_meta") or {}).get("title") == graph_mod.TITLE_FLUX_CHARACTER_LORA]
    style = [nid for nid, n in graph.items()
             if (n.get("_meta") or {}).get("title") == graph_mod.TITLE_FLUX_STYLE_LORA]

    assert len(per_persona) == 1, f"ambiguous per-persona LoRA slot: {per_persona}"
    assert len(style) == 1


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v4_does_not_overwrite_the_style_lora(mode):
    """It is the same file for every persona and is fetched with the shared
    weights, so a job must leave it alone."""
    graph = graph_mod.patch(mode, **_job_kwargs("v4"))

    style = [n for n in graph.values()
             if (n.get("_meta") or {}).get("title") == graph_mod.TITLE_FLUX_STYLE_LORA][0]
    assert style["inputs"]["lora_name"] == "klein_snofs_v1_4.safetensors"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v4_stacks_the_style_lora_under_the_character_lora(mode):
    """Order matters: the character LoRA has to be the last word on identity."""
    graph = graph_mod.load(mode, "v4")

    flux_samplers = [nid for nid, n in graph.items()
                     if n["class_type"] == "KSampler"
                     and (n.get("_meta") or {}).get("title", "").startswith("FLUX")]
    assert flux_samplers

    for nid in flux_samplers:
        chain = _model_chain(graph, graph[nid]["inputs"]["model"][0])
        assert graph_mod.TITLE_FLUX_CHARACTER_LORA in chain, nid
        assert graph_mod.TITLE_FLUX_STYLE_LORA in chain, nid
        assert chain.index(graph_mod.TITLE_FLUX_CHARACTER_LORA) < \
               chain.index(graph_mod.TITLE_FLUX_STYLE_LORA), \
               f"{nid}: the style LoRA is applied after the character LoRA"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v4_krea2_samplers_still_carry_the_krea2_character_lora(mode):
    graph = graph_mod.load(mode, "v4")

    for nid, node in graph.items():
        if node["class_type"] != "KSampler":
            continue
        if (node.get("_meta") or {}).get("title", "").startswith("FLUX"):
            continue
        chain = _model_chain(graph, node["inputs"]["model"][0])
        assert graph_mod.TITLE_CHARACTER_LORA in chain, f"{nid}: {chain}"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v4_sampler_titles_match_their_denoise(mode):
    """The handed-over graphs said 0.40 while running 0.85 — a title that lies
    about the number is worse than no title when tuning."""
    graph = graph_mod.load(mode, "v4")

    for nid, node in graph.items():
        title = (node.get("_meta") or {}).get("title", "")
        if node["class_type"] != "KSampler" or "denoise" not in title:
            continue
        stated = title.split("denoise")[1].strip(" )")
        assert float(stated) == float(node["inputs"]["denoise"]), f"{nid}: {title}"


# ── v5 ───────────────────────────────────────────────────────────────────────
#
# v5 is v4 with one change at the front: the hero's Krea2 detail pass becomes a
# KSamplerAdvanced that joins its schedule partway through instead of a KSampler
# at denoise 0.85, and the identity-edit LoRA comes up. Its carousel is the same
# change on the same hero — everything after the hero decode is v4's slide
# chain, untouched. That last part is the whole reason this section exists: the
# tempting way to build the carousel is to re-export the whole thing, and then
# the slides drift from v4 for reasons nobody meant.

def _v5(mode):
    return graph_mod.load(mode, "v5")


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v5_replaces_the_hero_detail_pass_with_an_advanced_sampler(mode):
    graph = _v5(mode)

    advanced = [nid for nid, n in graph.items()
                if n["class_type"] == "KSamplerAdvanced"]
    assert len(advanced) == 1, f"expected one KSamplerAdvanced, got {advanced}"

    node = graph[advanced[0]]
    # Starting partway through the schedule is what replaces a denoise figure.
    assert 0 < node["inputs"]["start_at_step"] < node["inputs"]["steps"]

    # And the hero's old fixed-denoise pass is gone. Only the hero's: the
    # carousel's three slides keep their own detail passes at 0.8, because v5
    # changes the start of the carousel and nothing after it.
    hero_detail = [nid for nid, n in graph.items()
                   if n["class_type"] == "KSampler"
                   and (n.get("_meta") or {}).get("title", "")
                   == "KREA2: detail pass (denoise 0.85)"]
    assert hero_detail == [], f"v4's hero detail pass survived: {hero_detail}"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v5_settle_pass_reads_the_advanced_sampler(mode):
    """The join. Left pointing at the deleted node the graph would not load at
    all; left pointing at the source encode it would settle an unrendered
    frame, which looks like a soft render rather than a broken one."""
    graph = _v5(mode)

    settle = [nid for nid, n in graph.items()
              if (n.get("_meta") or {}).get("title", "").startswith("KREA2: settle")]
    assert len(settle) == 1

    upstream = graph[settle[0]]["inputs"]["latent_image"][0]
    assert graph[upstream]["class_type"] == "KSamplerAdvanced"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v5_advanced_sampler_carries_the_character_lora(mode):
    """It runs off the character LoRA rather than the edit patch, which is the
    point of the change — but it must still be the persona's LoRA, or the hero
    comes back as somebody else."""
    graph = _v5(mode)

    advanced = [nid for nid, n in graph.items()
                if n["class_type"] == "KSamplerAdvanced"][0]
    chain = _model_chain(graph, graph[advanced]["inputs"]["model"][0])
    assert graph_mod.TITLE_CHARACTER_LORA in chain, chain


def test_v5_carousel_is_v4s_slide_chain_behind_the_new_hero():
    """The requirement in one assertion: only the hero changed.

    Everything from the hero decode onwards — the Gemini call that writes the
    slide instructions, the three Flux slides, their Krea2 passes, the batch —
    must be exactly what v4 ships. If a re-export ever drifts one of them this
    fails and names it.
    """
    import json

    v4 = graph_mod.load("carousel", "v4")
    v5 = graph_mod.load("carousel", "v5")

    # The hero half, which is allowed to differ.
    hero = {"2314", "3094", "2346", "1033", "3013", "2313"}
    drifted = []
    for nid in set(v4) | set(v5):
        if nid in hero:
            continue
        a = json.dumps(v4.get(nid), sort_keys=True)
        b = json.dumps(v5.get(nid), sort_keys=True)
        if a != b:
            title = ((v5.get(nid) or v4.get(nid) or {}).get("_meta") or {}).get("title")
            drifted.append(f"{nid} ({title})")
    # Seeds are randomised per job, so an exported difference in one is noise.
    drifted = [d for d in drifted if "seed" not in d.lower()]
    assert drifted == [], f"the slide chain drifted from v4: {drifted}"


def test_v5_carousel_hero_matches_v5_single():
    """Same hero, both modes — the carousel starts with the single and carries
    on. A hero that differs between them means tuning one does not move the
    other, which is the thing versioning is supposed to prevent."""
    single = graph_mod.load("single", "v5")
    carousel = graph_mod.load("carousel", "v5")

    for nid in ("3094", "2346", "1033", "3013", "2313", "3016"):
        a, b = single[nid]["inputs"], carousel[nid]["inputs"]
        for field, value in a.items():
            if "seed" in field or isinstance(value, list):
                continue          # seeds are per-job; links are per-graph
            assert b.get(field) == value, f"{nid}.{field}: {b.get(field)} != {value}"


@pytest.mark.parametrize("mode", graph_mod.MODES)
def test_v5_prompt_template_still_takes_a_subject(mode):
    """The export bakes in whichever persona was loaded in ComfyUI. Without the
    placeholder back, build() would refuse the job — better than rendering a
    stranger, but still a broken version."""
    graph = _v5(mode)

    templates = [n["inputs"]["value"] for n in graph.values()
                 if n["class_type"] == "PrimitiveStringMultiline"]
    assert any(graph_mod.SUBJECT_PLACEHOLDER in t for t in templates)
    assert not any("3lm1ra" in t for t in templates), "a persona is baked in"
