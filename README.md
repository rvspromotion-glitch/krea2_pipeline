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
endpoint would mean a second ~40GB copy and a second cold start for nothing,
and routing both here keeps the worker warm across a mixed batch.

## Job contract

```jsonc
{
  "input": {
    "mode": "single",                    // or "carousel"
    "image_url": "https://…/ref.jpg",    // or "image_b64"
    "lora_name": "Eva_step-002750_identity.safetensors",
    "lora_url": "https://…",             // optional; fetched if not on the volume
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

## Deploy

Pushing to `main` builds and pushes the image via GitHub Actions. Two repo
secrets are needed (Settings → Secrets and variables → Actions):

| secret | value |
|---|---|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | an access token, **not** your password |

Each build publishes two tags: `krea2-worker:latest` and
`krea2-worker:<commit-sha>`.

**Point the RunPod endpoint at the sha, not at `latest`.** RunPod caches images
by tag, so an endpoint pinned to `latest` can keep serving a stale build after a
push — and when something breaks you have no way to say which build is running.
With a sha tag, deploying is changing one string and rolling back is changing it
back. The workflow prints the exact tag to paste.

## Setup

Models live on a **network volume**, not in the image. Seed it once:

```bash
# from a pod with the volume mounted
CIVITAI_TOKEN=… HF_TOKEN=… ./setup_volume.sh
# or, one-off, with the same image: RUN_SETUP=1
```

Then point a serverless endpoint at the image with the volume mounted at
`/runpod-volume`. **Set the endpoint execution timeout above 10 minutes** —
carousels legitimately take 5–10, and RunPod will otherwise kill the job before
the handler ever reports.

Character LoRAs are per-persona: drop them on the volume, or let the first job
fetch one via `lora_url`.

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

No GPU, no ComfyUI, no network. They guard the thing that fails silently — a
re-exported workflow with renumbered nodes would still render, just with the
wrong persona's LoRA. Nodes are looked up by title and a mismatch is a hard
error at job start.

## Layout

```
workflows/    the two API graphs, cleaned and templated
src/graph.py  loads a graph, patches in the per-job variables
src/comfy.py  ComfyUI HTTP client (submit, poll, collect)
src/handler.py RunPod entry point
setup_volume.sh  one-off volume seeding
```
