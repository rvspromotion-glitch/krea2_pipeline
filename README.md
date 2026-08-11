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

## Where the models live

**No network volume.** A volume pins the endpoint to the one datacentre that
holds it, which starves it of GPUs. This workload is a weekly sequential batch,
so paying a cold start once a week is the cheaper trade.

**Not in the image either.** ~18GB of weights on top of torch does not build on
a hosted GitHub runner — it has ~14GB free on `/`, and a build needs room for
the layers twice over. So the image is ComfyUI, the custom nodes and the code
(~8GB), and `entrypoint.sh` fetches the weights on cold start from
[`models.txt`](models.txt) before starting ComfyUI.

That costs one download per cold start, which for this workload is one download
per weekly batch: the run is forty-odd jobs back to back on a worker that stays
warm, against three to four hours of rendering. Fetches are skip-if-present and
run concurrently, and a failed one stops the worker rather than letting ComfyUI
start without a checkpoint and fail every job with an error that never mentions
the download.

The rest of the Dockerfile is still shaped for a small pull:

- **Multi-stage.** The CUDA *devel* toolkit is needed to build a couple of
  wheels and for nothing at runtime, so it stays in the builder. The runtime
  stage is on `cuda:base` and gets its CUDA libraries from the torch wheels.
- **Application code last**, because everything below a changed layer is
  rebuilt — a code-only change re-pulls megabytes.
- **Node set verified at build time** (below).

Character LoRAs are not in `models.txt`: they are per-persona and change when a
persona is retrained. Radar hosts them and sends `lora_url` per job.

### Adding or changing a model

Edit `models.txt`. The destination filename is what the graph references, so
renaming one there without renaming it in `workflows/*.json` gives you a clean
download and a validation error on the first job — there is a test for exactly
that.

## Deploy

Pushing to `main` builds and pushes the image. Repo secrets
(Settings → Secrets and variables → Actions):

| secret | value |
|---|---|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | an access token, **not** your password |

The model-host tokens are **not** build secrets — the weights are fetched by the
worker, so they go on the RunPod endpoint instead
(Settings → Environment Variables):

| endpoint variable | value |
|---|---|
| `CIVITAI_TOKEN` | required — the checkpoint and one LoRA are gated |
| `HF_TOKEN` | only if a Hugging Face repo goes gated |

Each build publishes `krea2-worker:latest` and `krea2-worker:<commit-sha>`.

**Point the RunPod endpoint at the sha, not at `latest`.** RunPod caches images
by tag, so an endpoint pinned to `latest` can keep serving a stale build after a
push — and when something breaks you have no way to say which build is running.
With a sha tag, deploying is changing one string and rolling back is changing it
back. The workflow prints the exact tag to paste.

Endpoint settings that matter:

- **Execution timeout above 10 minutes.** Carousels legitimately take 5–10, and
  RunPod otherwise kills the job before the handler ever reports.
- **Container disk ≥ 40GB** — ~8GB of image plus ~18GB of weights, with room to
  work.
- **`CIVITAI_TOKEN`** as an environment variable, or the worker stops on start
  and says so.
- Leave **FlashBoot** on. It is what keeps a second cold start in the same batch
  cheap.

### If a model download fails

The worker stops instead of starting ComfyUI without its weights. The log names
the model, the HTTP status and the first 400 bytes of what the server actually
sent, which is normally enough to tell a rejected token from a dead URL:

```
[models] FAILED: AiO_krea2_checkpoint_int8_8steps.safetensors
[models]   [fetch] ERROR: ... is not a model.
[models]           HTTP status : 401
[models]           | {"error":"Unauthorized","message":"The creator of this asset..."}
[models]           -> the token was rejected. Check CIVITAI_TOKEN ...
```

Anything already downloaded is kept, so a restart resumes rather than starting
over.

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

This runs CPU-only — registering nodes is an import, not a render. ComfyUI has
to be *told* that, and not via `--cpu`: `comfy/cli_args.py` only reads `sys.argv`
once `main.py` has enabled arg parsing, so importing `nodes` directly leaves
every flag at its default and `model_management` calls
`torch.cuda.current_device()` at import. The flag is set on the parsed namespace
instead. If a custom node package probes CUDA on import regardless, nothing here
can check it on a build machine — build with `--build-arg VERIFY_NODES=0` and
check the graphs by running one job instead.

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
models.txt         weights to fetch on cold start, as data
custom_nodes.txt   node packages to install, as data
constraints.txt    pins the node set actually requires
src/graph.py       loads a graph, patches in the per-job variables
src/comfy.py       ComfyUI HTTP client (submit, poll, collect)
src/handler.py     RunPod entry point
scripts/           install_nodes + verify_nodes (build), fetch_model(s) (start)
```
