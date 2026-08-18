# Python Server Runtime Configuration

This page configures the managed Python Server used by the browser console. It does not require changing any tracked profile JSON file.

## One-time environment

Run these commands after activating the Python virtual environment and building `build-cuda-shared`:

```bash
cd /absolute/path/to/edge-dit.cpp
export EDGE_DIT_REPO_ROOT="$PWD"
export EDGE_DIT_PYTHON_BIN="$PWD/.venv/bin/python"
export EDGE_DIT_LIBRARY="$PWD/build-cuda-shared/bin/libedgedit.so"
export EDGE_DIT_DEPENDENCY_DIRS="$PWD/build-cuda-shared/bin"
```

Set these in your shell profile only after confirming the paths work. Keep model paths machine-local; do not commit them into `runtime/profiles/*.json`.

## Select a model profile

| Profile argument | Required variable | Expected directory |
| --- | --- | --- |
| `flux-dev` | `EDGE_DIT_FLUX_MODEL_PATH` | FLUX.1-dev Diffusers directory |
| `flux-schnell` | `EDGE_DIT_FLUX_SCHNELL_MODEL_PATH` | FLUX.1-schnell Diffusers directory |
| `flux-kontext` | `EDGE_DIT_FLUX_KONTEXT_MODEL_PATH` | FLUX.1-Kontext-dev Diffusers directory |
| `kontext-lightning` | Base + DiT variables below | Base Kontext directory + Lightning transformer |
| `flux2-klein-4b` | `EDGE_DIT_FLUX2_KLEIN_4B_MODEL_PATH` | FLUX.2-klein-4B Diffusers directory |
| `qwen-image` | `EDGE_DIT_QWEN_IMAGE_MODEL_PATH` | Qwen-Image Diffusers directory |
| `qwen-image-lightning` | Base + DiT variables below | Base Qwen-Image directory + merged Lightning transformer |
| `qwen-image-edit` | `EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH` | Qwen-Image-Edit Diffusers directory |
| `qwen-image-edit-lightning` | Base + DiT variables below | Base Qwen-Image-Edit directory + merged Lightning transformer |
| `sd3-medium` | `EDGE_DIT_SD3_MODEL_PATH` | Stable Diffusion 3 Medium Diffusers directory |
| `sd35-medium-turbo` | `EDGE_DIT_SD35_TURBO_MODEL_PATH` | Stable Diffusion 3.5 Medium Turbo Diffusers directory |
| `wan-t2v` | `EDGE_DIT_WAN_VIDEO_MODEL_PATH` | Wan2.1 T2V 1.3B Diffusers directory |
| `wan2-t2v-14b` | `EDGE_DIT_WAN_14B_MODEL_PATH` | Wan2.1 T2V 14B Diffusers directory |
| `wan21-t2v-1.3b-distill` | Base + DiT variables below | Base Wan 1.3B directory + distilled transformer |
| `minimax-h3` | Four component variables listed below | MiniMax-H3 FL2VA DiT, Qwen, video VAE, and audio VAE files |

Every value should be absolute and should satisfy:

```bash
test -f "$EDGE_DIT_FLUX_MODEL_PATH/model_index.json"
```

For example:

```bash
export EDGE_DIT_FLUX_MODEL_PATH=/models/FLUX.1-dev
export EDGE_DIT_FLUX_SCHNELL_MODEL_PATH=/models/FLUX.1-schnell
export EDGE_DIT_FLUX_KONTEXT_MODEL_PATH=/models/FLUX.1-Kontext-dev
export EDGE_DIT_FLUX2_KLEIN_4B_MODEL_PATH=/models/FLUX.2-klein-4B
export EDGE_DIT_QWEN_IMAGE_MODEL_PATH=/models/Qwen-Image
export EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH=/models/Qwen-Image-Edit
export EDGE_DIT_SD3_MODEL_PATH=/models/stable-diffusion-3-medium-diffusers
export EDGE_DIT_SD35_TURBO_MODEL_PATH=/models/stable-diffusion-3.5-medium-turbo
export EDGE_DIT_WAN_VIDEO_MODEL_PATH=/models/Wan2.1-T2V-1.3B-Diffusers
export EDGE_DIT_WAN_14B_MODEL_PATH=/models/Wan2.1-T2V-14B-Diffusers
```

Transformer-only distilled profiles combine a full base directory with a
replacement DiT. The Qwen Lightning inputs must be merged transformer weights,
not the original LoRA adapter:

```bash
export EDGE_DIT_KONTEXT_LIGHTNING_DIT_PATH=/models/distilled/kontext-lightning/transformer
export EDGE_DIT_QWEN_IMAGE_LIGHTNING_DIT_PATH=/models/distilled/qwen-image-lightning-merged/transformer/diffusion_pytorch_model.safetensors.index.json
export EDGE_DIT_QWEN_IMAGE_EDIT_LIGHTNING_DIT_PATH=/models/distilled/qwen-image-edit-lightning-merged/dit/diffusion_pytorch_model.safetensors.index.json
export EDGE_DIT_WAN_DISTILL_DIT_PATH=/models/distilled/wan21-t2v-1.3b-distill/Wan2.1-T2V-1.3B-Distill-iter6000.safetensors
```

MiniMax-H3 uses standalone components instead of a Diffusers directory:

```bash
export EDGE_DIT_MINIMAX_DIT_PATH=/models/minimax-h3/minimax_h3_fl2va-Q8_0.gguf
export EDGE_DIT_MINIMAX_LLM_PATH=/models/minimax-h3/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
export EDGE_DIT_MINIMAX_VIDEO_VAE_PATH=/models/minimax-h3/minimax_h3_video_vae_fp16.safetensors
export EDGE_DIT_MINIMAX_AUDIO_VAE_PATH=/models/minimax-h3/minimax_h3_audio_vae_fp32.safetensors
```

Its frame count must be at least 22 and satisfy `17k+5`.

## Start and switch

Start the default `flux-dev` profile:

```bash
cd bindings/python/frontend/server-console
npm run dev:managed
```

Start another profile immediately:

```bash
npm run dev:managed -- --auto-start-profile wan-t2v
```

For MiniMax-H3, set all four component variables above and use
`--auto-start-profile minimax-h3`.

After the console starts, the Local Runtime panel can start, stop, or switch among profiles whose model variables are set. It shows the child process log tail and health result, which is the first place to inspect a failed load.

**Apply preset** loads production generation defaults shared with
`benchmark/models`:

| Profile | Size | Frames | Steps | Guidance / CFG | Flow shift |
| --- | --- | ---: | ---: | --- | ---: |
| `flux-dev` | 1024x1024 | - | 20 | 3.5 / 1.0 | model default |
| `flux-schnell` | 1024x1024 | - | 4 | 3.5 / 1.0 | model default |
| `flux-kontext` | 1024x1024 | - | 20 | 2.5 / 1.0 | model default |
| `kontext-lightning` | 1024x1024 | - | 8 | 2.5 / 1.0 | model default |
| `flux2-klein-4b` | 1024x1024 | - | 4 | 1.0 / 1.0 | model default |
| `qwen-image` | 1024x1024 | - | 30 | 1.0 / 4.0 | model default |
| `qwen-image-lightning` | 1024x1024 | - | 4 | 1.0 / 1.0 | model default |
| `qwen-image-edit` | 1024x1024 | - | 30 | 1.0 / 4.0 | model default |
| `qwen-image-edit-lightning` | 1024x1024 | - | 4 | 1.0 / 1.0 | model default |
| `sd3-medium` | 1024x1024 | - | 20 | 3.5 / 5.0 | 3.0 |
| `sd35-medium-turbo` | 1024x1024 | - | 8 | 3.5 / 1.5 | 3.0 |
| `wan-t2v` | 832x480 | 41 | 30 | 1.0 / 5.0 | 3.0 |
| `wan2-t2v-14b` | 832x480 | 41 | 50 | 1.0 / 5.0 | 3.0 |
| `wan21-t2v-1.3b-distill` | 832x480 | 41 | 8 | 1.0 / 1.0 | 3.0 |
| `minimax-h3` | 864x480 | 22 | 20 | 1.0 / 1.0 | model default |

Lower resolution or one-step requests are explicit smoke tests, not profile
defaults.

## Network bindings

The default addresses are local only:

| Variable | Default | Meaning |
| --- | --- | --- |
| `EDGE_DIT_FRONTEND_HOST` | `127.0.0.1` | Vite browser console address |
| `EDGE_DIT_RUNTIME_MANAGER_HOST` | `127.0.0.1` | Runtime manager address |
| `EDGE_DIT_MANAGED_BACKEND_HOST` | `127.0.0.1` | Python Server address |
| `EDGE_DIT_RUNTIME_MANAGER_PORT` | `8090` | Runtime manager port |
| `EDGE_DIT_MANAGED_BACKEND_PORT` | `8080` | Python Server port |

Run `npm run dev:managed:network` to bind all three to `0.0.0.0`. Treat that as trusted-LAN development only. The built-in server has no authentication, authorization, TLS, rate limiting, or persistent job storage.

## Runtime behavior

All managed profiles default to `auto_allocate`. It preserves source precision,
measures live free VRAM at model load, and decides independently whether each
component remains resident or streams from CPU. With no profile `max_vram_gb`,
the effective budget is live free VRAM; an explicit cap changes it to
`min(max_vram_gb, live free)`. MiniMax-H3 additionally uses its staged phase
lifecycle and video-VAE tiling.

This is placement automation, not automatic quantization. Use `auto_fit` only
when the runtime is also allowed to select TE/DiT precision. Users who require
fixed quantization or placement can start `python -m edge_dit.server` directly
with `--type`, `--no-auto-allocate`, and the component/full-offload flags shown
in the [Python guide](../../README.md#6-start-only-the-python-server).

The Python Server starts listening only after the model has loaded. Confirm readiness with:

```bash
curl http://127.0.0.1:8090/runtime/v1/status
curl http://127.0.0.1:8080/ed/v2/health
```

The manager restarts an unexpectedly exited backend a limited number of times. Fix the reported model path, native library, CUDA, or memory problem before repeatedly retrying.

For complete setup and generation parameters, return to [the Python guide](../../README.md).
