# Few-Step Distilled Models

[← Back to performance](../performance.md) | [← Back to README](../../README.md)

## 1. Overview

A standard diffusion sampling loop runs the transformer many times — one full
forward per denoising step, commonly 20 or more. Step count is the single
largest linear factor in latency: halving the steps roughly halves sampling
time. *Step-distilled* checkpoints (FLUX.1-schnell, SD3.5-Turbo,
Qwen-Image-Lightning, Wan distill variants, and similar) are trained to
reproduce a full run in far fewer steps — typically 4 to 8 — so they cut
sampling time by 2.5–5x with little quality loss at their intended step count.

edge-dit.cpp treats the distilled checkpoint as an ordinary model of its family.
No separate pipeline or code path is required: the same FLUX / SD3 / Qwen /
Wan pipeline runs it. The only thing that must change is the default step count,
so the runtime auto-detects a distilled checkpoint and picks a sensible few-step
default when the user does not specify one.

This is a speed feature, not a quality one. A distilled model at its intended
low step count is close to its own many-step output, but distillation is itself
an approximation of the base model — validate output for the exact model and
step count you plan to run.

---

## 2. Automatic step selection

Step count is resolved per request in the following priority order:

1. **Explicit `--steps N` (N > 0)** always wins. The user's value is used as-is.
2. **Otherwise (`--steps` unset, i.e. the CLI passes `-1`)** the pipeline
   consults a distilled-detection signal and, if the model looks distilled,
   uses a few-step default (4 or 8). If not, it uses the family's base default
   (20 for most, 50 for Qwen-Image-Edit).

So existing commands that pass an explicit step count are unaffected; the
few-step default only fills in when the user asks the runtime to decide.

### Detection signals

Distilled checkpoints usually share the exact architecture and config of their
base model — the distillation changes only weight *values*, not structure — so
there is often no metadata field that distinguishes them. edge-dit.cpp uses two
signals, in order of reliability:

- **Weight-structure signal (FLUX only).** FLUX.1-schnell is a distinct
  guidance-distilled architecture: it lacks the `guidance_in` layer that
  FLUX.1-dev has. The FLUX pipeline detects this directly and defaults schnell
  to 4 steps, dev to 20.
- **Path signal (all families).** For same-architecture distills, the model or
  `--diffusion-model` path is scanned (case-insensitive) in order of reliability:
  1. An explicit step marker such as `4steps`, `8-step`, or `2_steps` is taken
     as the step count directly, for any value in a sensible few-step range.
     This is the most reliable signal, because the step count is an independent
     property of the checkpoint that a family keyword alone does not convey.
  2. Otherwise, `schnell` defaults to 4 steps.
  3. Otherwise, a distillation keyword (`turbo`, `lightning`, `lightx2v`,
     `distill`, `hyper`) marks the checkpoint as distilled but carries no step
     count, so the runtime defaults to 8 steps and logs that it is doing so.
  4. No match means the base default is used.

Because a family keyword (for example `lightning`) does not itself state the
step count — the same family ships both 4-step and 8-step releases — a checkpoint
whose intended count is not 8 should carry an explicit `Nsteps` marker in its
path, or be run with an explicit `--steps N`. A missing marker is not an error:
the runtime falls back to 8 steps and logs a note, and `--steps` always overrides.

### CFG and guidance

Distilled models are guidance-distilled: the classifier-free-guidance (CFG)
behavior is baked into the weights, so they run a single forward per step rather
than the two forwards CFG requires. edge-dit.cpp does not force this — the
default `cfg-scale` of `1.0` already yields a single forward, so distilled
models run single-forward out of the box. Passing `--cfg-scale > 1` on a
distilled model re-enables the two-forward path and is generally not what you
want.

---

## 3. Usage

The only thing that changes versus running the base model is the step count, and
the runtime fills it in for you: leave `--steps` unset (or pass `-1`) and it
detects the distilled checkpoint and applies the few-step default (see §2).
Keep `--cfg-scale` at its default `1.0` (single forward). A minimal run:

```bash
# schnell detected via weight signal -> 4 steps automatically
ed-cli --backend cuda --type q8_0 --model /path/to/models/flux.1-schnell \
  --steps -1 --cfg-scale 1.0 \
  --prompt "a glass teapot on a wooden table" -o out.png
```

Two loading shapes come up: a **full checkpoint** (Diffusers directory) loads via
`--model`; a **transformer-only file** (or a distilled DiT shipped separately)
loads via `--diffusion-model` while a base `--model` directory supplies the VAE
and text encoders. Variants distributed as **LoRA adapters** are not runnable
directly — they must be merged into the base weights offline first; the CLI
loads full weights, not LoRA deltas (see [Merging LoRA
weights](merging-lora-weights.md)).

**For a ready-to-run command for each specific variant** — with the exact
`--cfg-scale` / `--guidance` / `--flow-shift`, the correct model paths, and which
ones need a LoRA merge — see [§4 Per-variant commands](#4-per-variant-commands)
below. See also [Command line usage](../cli.md) for the full option reference and
[Supported models](../models.md) for per-family notes.

---

## 4. Per-variant commands

Below is a ready-to-run command for **each** distilled variant. Notes that apply
to all of them:

- **`--steps -1`** lets the runtime auto-pick the few-step count (shown per
  variant). Pass an explicit `--steps N` to override.
- **`--cfg-scale 1.0`** (the default) keeps the single-forward path distilled
  models are trained for. Do **not** raise it (see [CFG and guidance](#cfg-and-guidance)).
- **`--guidance`** (FLUX distilled guidance embedding, default `3.5`) applies to
  the FLUX-family and Wan pipelines (FLUX.1-dev, Kontext-dev, Wan). It has **no
  effect** on SD3 and Qwen-Image, and on FLUX.1-schnell (which has no
  `guidance_in` layer), so it is omitted for those.
- **`--flow-shift`** (flow-scheduler shift) has a correct **per-family default**
  applied automatically when omitted (FLUX dev `1.15` / schnell `1.0`, Kontext
  `1.15`, SD3 `3.0`, Wan `5.0`, Qwen dynamic). It is therefore only spelled out
  below for **Wan** (`5.0`), where the benchmark commands set it explicitly for
  clarity; every other family is left at its default.
- **How a variant is packaged decides which flag you use.** Check what you
  actually downloaded with `ls <dir>`:
  - **Full Diffusers directory** — contains `model_index.json` plus `vae/`,
    `text_encoder*/`, `transformer/`. Load the whole thing with **`--model`**;
    no `--diffusion-model` needed. (Several distilled repos ship this way, e.g.
    SD3.5-medium-turbo, and the Kontext-Lightning repo also has a full layout.)
  - **Transformer-only** — just a `transformer/` (or `dit/`) subfolder, or a
    single top-level `.safetensors`, with **no** `model_index.json` / `vae/` /
    `text_encoder*/`. It carries no text encoders or VAE, so load it via
    **`--diffusion-model`** while a base `--model` directory supplies CLIP / T5 /
    VAE. Loading a transformer-only file directly with `--model` fails with a
    "needs … text encoder … and VAE weights" error.
  - **LoRA adapter** — not a full transformer at all (a small `.safetensors` of
    low-rank deltas); it must be merged into the base weights offline first (the
    two Qwen-Image variants).

  `--diffusion-model` overrides the DiT weights from the base `--model` directory
  (the loader applies `--model` first, then overwrites the `model.diffusion_model.*`
  tensors with `--diffusion-model`), so the pair gives you "base VAE/TE + distilled
  DiT". If your download is already a full directory, prefer plain `--model`.
- Memory flags are omitted for clarity. Add `--auto-fit --max-vram <N>
  --vae-tiling auto` (validated at 24/16/8 GiB) to fit a budget — see
  [budget-driven placement](../cli.md#budget-driven-placement---auto-allocate-and-full-auto---auto-fit).
- Paths below use `/path/to/models/` as a placeholder — substitute your actual
  models directory.

### FLUX.1-schnell (auto 4 steps)

schnell is guidance-distilled and **lacks the `guidance_in` layer**, so
`--guidance` does not apply. The published repo is a full Diffusers directory
(transformer + CLIP-L + T5XXL + VAE), so load it with `--model`:

```bash
ed-cli --backend cuda --type q8_0 --model /path/to/models/flux.1-schnell \
  --steps -1 --cfg-scale 1.0 -W 1024 -H 1024 \
  --prompt "a glass teapot on a wooden table" -o schnell.png
```

If instead you downloaded only the `transformer/` shards (no VAE/TE), load them
via `--diffusion-model` over the base FLUX model, which supplies VAE + text
encoders:

```bash
ed-cli --backend cuda --type q8_0 \
  --model /path/to/models/flux.1-schnell \
  --diffusion-model /path/to/models/flux.1-schnell/flux1-schnell.safetensors \
  --steps -1 --cfg-scale 1.0 -W 1024 -H 1024 \
  --prompt "a glass teapot on a wooden table" -o schnell.png
```

### SD3.5-medium-turbo (auto 8 steps)

SD3 family: uses `--cfg-scale` (keep `1.0`). `--guidance` does not apply. 
Full Diffusers directory:

```bash
ed-cli --backend cuda --type q8_0 --model /path/to/models/sd35-medium-turbo \
  --steps -1 --cfg-scale 1.0 -W 1024 -H 1024 \
  --prompt "a glass teapot on a wooden table" -o turbo.png
```

### FLUX.1-Kontext Lightning (auto 8 steps, image editing)

Kontext is FLUX-family and dev-based, so **`--guidance` applies**.
Requires `--image`. The [`camenduru/FLUX.1_Kontext-Lightning`](https://huggingface.co/camenduru/FLUX.1_Kontext-Lightning)
repo has a **full Diffusers directory** (with `model_index.json`, VAE, text
encoders), so if you download the whole thing just point `--model` at it:

```bash
ed-cli --backend cuda --type q8_0 \
  --model /path/to/models/kontext-lightning \
  --image input.png --steps -1 --guidance 2.5 --cfg-scale 1.0 -W 1024 -H 1024 \
  --prompt "make the object look like brushed metal" -o kontext-lightning.png
```

If instead you downloaded only the `transformer/` shards (no VAE/TE), load them
via `--diffusion-model` over the base Kontext model, which supplies VAE + text
encoders:

```bash
ed-cli --backend cuda --type q8_0 \
  --model /path/to/models/flux.1-kontext-dev \
  --diffusion-model /path/to/models/kontext-lightning/transformer/diffusion_pytorch_model.safetensors.index.json \
  --image input.png --steps -1 --guidance 2.5 --cfg-scale 1.0 -W 1024 -H 1024 \
  --prompt "make the object look like brushed metal" -o kontext-lightning.png
```

### Qwen-Image Lightning (4 steps) — merge LoRA first

Published as a **LoRA adapter**; merge it into the base Qwen-Image transformer
offline first (see [Merging LoRA weights](merging-lora-weights.md)), then point
`--diffusion-model` at the merged transformer. Qwen uses `--cfg-scale` (keep
1.0); `--guidance` does not apply.

The published adapter is the 4-step release, so the merged output directory
should carry a `4steps` marker (the merge script adds one automatically from the
LoRA filename) for the runtime to select 4 steps under `--steps -1`. Without the
marker it would default to 8; passing `--steps 4` also forces it explicitly.

```bash
ed-cli --backend cuda --type q8_0 --model /path/to/models/qwen-image \
  --diffusion-model /path/to/models/distilled/qwen-image-lightning-4steps-merged/transformer/diffusion_pytorch_model.safetensors.index.json \
  --steps -1 --cfg-scale 1.0 -W 1024 -H 1024 \
  --prompt "a red apple on a wooden table" -o qwen-lightning.png
```

### Qwen-Image-Edit Lightning (4 steps, image editing) — merge LoRA first

Same LoRA-merge requirement, plus `--image`. As with Qwen-Image Lightning, the
adapter is the 4-step release, so keep the `4steps` marker on the merged output
directory (or pass `--steps 4`):

```bash
ed-cli --backend cuda --type q8_0 --model /path/to/models/qwen-image-edit \
  --diffusion-model /path/to/models/distilled/qwen-image-edit-lightning-4steps-merged/dit/diffusion_pytorch_model.safetensors.index.json \
  --image input.png --steps -1 --cfg-scale 1.0 -W 1024 -H 1024 \
  --prompt "make it brushed metal" -o qwen-edit-lightning.png
```

### Wan2.1-T2V-1.3B Distill (auto 8 steps, video)

Shipped as a **standalone single full-weight `.safetensors`** file. Load it via
`--diffusion-model` with the base Wan model supplying VAE + text encoders. Wan's
`--flow-shift 3.0` explicitly to match the benchmark's 480P setting. `--cfg-scale`
stays `1.0` (distilled runs single-forward). `--guidance` is accepted by the Wan
pipeline (default `3.5`) but is left at default here.

```bash
ed-cli --backend cuda --type q8_0 --video \
  --model /path/to/models/wan2.1-t2v-1.3b \
  --diffusion-model /path/to/models/wan21-t2v-1.3b-distill/Wan2.1-T2V-1.3B-Distill-iter6000.safetensors \
  --steps -1 --cfg-scale 1.0 --flow-shift 3.0 -W 832 -H 480 --frames 41 --fps 16 \
  --prompt "a glass teapot rotating on a wooden table" -o wan-distill.mp4
```
