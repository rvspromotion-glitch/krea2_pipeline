# krea2_pipeline

RunPod serverless worker for the ACG content pipeline. Takes a reference photo
and a persona, returns generated images.

Two workflows, **one endpoint**:

| mode | in | out |
|---|---|---|
| `single` | 1 reference photo | 1 image |
| `carousel` | 1 reference photo | 4 images — slides of *one* post, not alternatives |

They share every model (same checkpoint, CLIP, VAE, LoRAs), and the carousel
graph literally contains the single-photo graph as its first stage. A second
endpoint would mean a second copy of the image and a second cold start for
nothing, and routing both here keeps the worker warm across a mixed batch.

## Job contract

```jsonc
{
  "input": {
    "mode": "single",                    // or "carousel"
    "image_url": "https://…/ref.jpg",    // or "image_b64"
    "lora_name": "Eva_step-002750_identity.safetensors",
    "lora_url": "https://…",             // Radar serves it; fetched per cold start
    "trigger_word": "3lm1ra",
    "description": "young woman with long platinum blonde hair, grey iris eye colour",
    "gemini_api_key": "…",               // from Radar settings
    "seed": 12345                        // optional; omit for random
  }
}
```

```jsonc
{ "mode": "carousel", "count": 4, "seed": null, "duration_s": 412.0,
  "images": ["<base64 png>", "…"] }
```

Failures return `{"error": "...", "kind": "input" | "render" | "timeout"}` — the
worker never crashes on a bad job.

## No network volume

Models are baked into the image. A network volume pins the endpoint to the one
datacentre that holds it, which starves it of GPUs; this workload is a weekly
sequential batch, so paying a cold start once a week is the cheaper trade.

Everything in the Dockerfile follows from that:

- **Multi-stage.** The CUDA *devel* toolkit is needed to build a couple of
  wheels and for nothing at runtime, so it stays in the builder. The runtime
  stage is on `cuda:base` and gets its CUDA libraries from the torch wheels.
- **One layer per model.** Layers are pulled and cached individually, so a code
  change re-pulls a few MB rather than 18GB, and swapping a LoRA does not
  invalidate the checkpoint.
- **Application code last**, because everything below a changed layer is rebuilt.
- **Node set verified at build time** (below).

Character LoRAs are deliberately *not* baked in: they are per-persona and change
when a persona is retrained, and adding one should not mean rebuilding an 18GB
image. Radar hosts them and sends `lora_url`; the worker fetches once per cold
start. That is a few hundred MB against a weekly batch.

## Deploy

Pushing to `main` builds and pushes the image. Repo secrets
(Settings → Secrets and variables → Actions):

| secret | value |
|---|---|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | an access token, **not** your password |
| `CIVITAI_TOKEN` | Civitai API token — the checkpoint and one LoRA are gated |
| `HF_TOKEN` | only if a Hugging Face repo above becomes gated |

Each build publishes `krea2-worker:latest` and `krea2-worker:<commit-sha>`.

**Point the RunPod endpoint at the sha, not at `latest`.** RunPod caches images
by tag, so an endpoint pinned to `latest` can keep serving a stale build after a
push — and when something breaks you have no way to say which build is running.
With a sha tag, deploying is changing one string and rolling back is changing it
back. The workflow prints the exact tag to paste.

Endpoint settings that matter:

- **Execution timeout above 10 minutes.** Carousels legitimately take 5–10, and
  RunPod otherwise kills the job before the handler ever reports.
- **Container disk ≥ 30GB**, or the image will not unpack.
- Leave **FlashBoot** on. It is the only thing that makes a second cold start in
  the same batch cheap.

### If the build runs out of disk

A hosted GitHub runner has ~14GB free on `/` and a large, empty `/mnt`. The
workflow deletes the preinstalled toolchains and moves Docker's storage to
`/mnt` before building, which is enough today — but the image is ~26GB unpacked
and that margin is not huge. If a build dies on "no space left on device", the
fix is a larger runner rather than more pruning.

## Build-time node verification

A missing custom node does not fail at a convenient moment: ComfyUI starts,
RunPod reports the worker healthy, and the first job of the Sunday batch comes
back with a validation error. So the build imports every node package and
asserts that every `class_type` in both graphs resolves. A missing node fails
the build instead.

It also prints which package supplies which node — the only reliable way to know,
since class names do not name their package:

```
  ComfyUI-KJNodes
      GetImageSizeAndCount
      PathchSageAttentionKJ
  (core ComfyUI)
      KSampler
      ...
[verify] 11 package(s) supplied nothing to these graphs:
      ComfyUI-DepthAnythingV2
      ...
```

Only six of the twenty-three node types come from custom packages. **Trim
`custom_nodes.txt` using that output**, not guesswork — a smaller node set is
the one cold-start saving still on the table, since ComfyUI imports every
package before it serves anything. Building with `PRUNE_UNUSED_NODES=1` does it
automatically; it is off by default because a package can matter by patching
something globally on import, which no static check can see.

## What changed from the ComfyUI exports

The graphs in `workflows/` are the hand exports with four edits:

- **`1031` rewired from `2319` to `2318`.** The carousel fed its anchor through
  a `SaveImage` node, which has no outputs. Invisible in the UI; headless it
  fails validation and takes the whole carousel stage with it.
- **Dead nodes removed** — 3 from single, 5 from carousel, including a
  `UNETLoader` that was loading a model nothing referenced.
- **Gemini API key stripped.** The exports shipped a live key inline. It now
  comes per-job from Radar's settings, and a test fails if one is ever
  committed again.
- **Prompts templated.** The persona span became `{subject}`, filled from
  `trigger_word` + `description`. The two prompts had already drifted apart in
  the exports (one carried eye colour, the other didn't) — one source now.

Seeds are randomised per job, **including the two Gemini seeds**: same photo
plus same seed yields the same description, hence the same picture.

## Tests

```bash
pip install pytest && python -m pytest tests -q
```

No GPU, no ComfyUI, no network, no Docker. They guard the things that fail
expensively: a re-exported workflow with renumbered nodes would still render,
just with the wrong persona's LoRA (nodes are looked up by title, and a mismatch
is a hard error at job start), and a Dockerfile that COPYs a path the build
context excludes only says so forty minutes into CI.

## Layout

```
workflows/         the two API graphs, cleaned and templated
custom_nodes.txt   node packages to install, as data
constraints.txt    pins the node set actually requires
src/graph.py       loads a graph, patches in the per-job variables
src/comfy.py       ComfyUI HTTP client (submit, poll, collect)
src/handler.py     RunPod entry point
scripts/           build steps: fetch models, install nodes, verify nodes
```
