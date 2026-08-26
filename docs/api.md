# API and Bindings Guide

[← Back to the repository README](../README.md)

This page explains the public ways to use edge-dit.cpp. Start with the choice that matches your application, then use the focused section.

| You want to... | Use | What it is |
| --- | --- | --- |
| Embed generation in C or C++ | C API | A native library API declared in `include/edge-dit.h` |
| Send one request to a native executable | Native Server | The `ed-server` binary and its `/ed/v1` HTTP API |
| Generate from a Python program | Python bindings | `Engine`, `EngineConfig`, `ImageRequest`, and `VideoRequest` |
| Run queued jobs from a web UI or another service | Python Server | `edge_dit.server` and its `/ed/v2` HTTP job API |
| Operate a local model in a browser | Python Server Console | React UI, runtime manager, and Python Server |

## Terms used everywhere

| Term | Meaning |
| --- | --- |
| **Model directory** | A complete local model export, normally a Diffusers directory containing `model_index.json` and all required components. |
| **Engine** | One loaded model plus its native runtime resources. It owns GPU/CPU allocations and executes generation. |
| **Backend** | The device implementation that runs the model: `cuda`, `cpu`, `vulkan`, `metal`, or `auto`. Only backends included in the native build are available. |
| **Prompt** | Text describing the requested image or video. It is required for every generation request. |
| **Seed** | Integer used by random sampling. Reusing the same seed and settings makes a run easier to reproduce. |
| **Steps** | Number of denoising iterations. More steps generally cost more time; `1` is only a wiring check. |
| **Offload** | Keep weights in system RAM and move parts to GPU when needed. It lowers VRAM use but increases system-RAM use and can slow generation. |
| **Quantization** | Store or load weights with fewer bits, for example `q4_k`, to reduce memory use. It can affect output quality and speed. |
| **Job** | An asynchronous Python Server request. It has an ID and progresses through a lifecycle before the result can be retrieved. |
| **API version** | The `v1` or `v2` in an HTTP path. It identifies the wire protocol, not the product name. |

## 1. C API

The C API is for applications that link `libedgedit.so` or `libedgedit.a` directly. The complete, authoritative declarations are in [include/edge-dit.h](../include/edge-dit.h).

### Lifecycle

A `context` is the C API equivalent of an Engine. Create one, use it for one or more requests, then free it.

`@c
#include "edge-dit.h"

ed_context_params_t context_params;
ed_context_params_init(&context_params);
context_params.model_path = "/absolute/path/to/FLUX.1-dev";
context_params.backend = "cuda";

ed_context_t *context = ed_create_context(&context_params);
if (context == NULL) {
    /* Context creation failed. Inspect your application logs/configuration. */
    return 1;
}

/* Build a request, call ed_generate_image or ed_generate_video. */

ed_free_context(context);
`@

Every `*_init` function fills a parameter structure with project defaults. Call it before assigning individual fields. `ed_context_t` owns the loaded model and must be released with `ed_free_context`.

### Main C types

| Type or function | Meaning |
| --- | --- |
| `ed_context_params_t` | Model paths, device/backend selection, memory placement, precision, and parallel settings used to load a context. |
| `ed_sample_params_t` | Sampling controls shared by image and video requests: steps, seed, guidance, sampler, scheduler, and cache settings. |
| `ed_image_generation_params_t` | Prompt plus image-specific size, input-image, mask, control, LoRA, and sampling settings. |
| `ed_video_generation_params_t` | Prompt plus video-specific size, frame count, sampling settings, and model inputs. |
| `ed_image_batch_t` | Generated images returned by `ed_generate_image`. |
| `ed_audio_t`, `ed_ref_video_t` | Interleaved float audio and a reference video (frames, fps, optional audio). |
| `ed_video_t` | Generated video frames plus optional interleaved float audio returned by `ed_generate_video`. |
| `ed_status_t` | Return status. `ED_STATUS_OK` means success; other values describe invalid input, model load, generation, memory, unsupported-feature, or cancellation failures. |
| `ed_get_last_error` | Returns the latest error text associated with a context. |
| `ed_context_request_cancel` | Requests cooperative cancellation. The native code checks at safe step boundaries. |
| `ed_context_progress_current_step` | Current sampling step; it does not cover prompt encoding or VAE output conversion. |

### Generate and free an image

`@c
ed_image_generation_params_t request;
ed_image_generation_params_init(&request);
request.prompt = "a glass teapot on a wooden table";
request.width = 1024;
request.height = 1024;
request.sample.steps = 20;
request.sample.seed = 42;

ed_image_batch_t images;
ed_status_t status = ed_generate_image(context, &request, &images);
if (status == ED_STATUS_OK) {
    /* Read images.images[0], then release the entire batch. */
    ed_free_image_batch(&images);
} else {
    const char *message = ed_get_last_error(context);
    /* Handle message. */
}
`@

The caller owns every successful output. Release images with `ed_free_image` or `ed_free_image_batch`, and release a video with `ed_free_video`. Do not free output buffers with `free()`.

## 2. Native Server

`ed-server` is a native HTTP wrapper around the C API. It is useful when you want the smallest non-Python deployment.

`@bash
./build-cuda/bin/ed-server \
  --backend cuda \
  --model /absolute/path/to/FLUX.1-dev \
  --host 127.0.0.1 \
  --port 8080
`@

Its canonical API prefix is `/ed/v1`:

| Endpoint | Meaning |
| --- | --- |
| `GET /ed/v1/health` | Confirms that the native server is reachable. |
| `GET /ed/v1/models` | Lists the model information exposed by the server. |
| `GET /ed/v1/capabilities` | Reports supported generation kinds and defaults. |
| `POST /ed/v1/images/generations` | Runs an image generation request using the native server contract. |
| `POST /ed/v1/videos/generations` | Runs video generation and returns base64 PNG frames plus optional float32 audio. |

The native server and Python Server are different applications. Do not send Python Server job requests to `/ed/v1`; use `/ed/v2` only when the Python Server is running. See [examples/server/README.md](../examples/server/README.md) for native-server request details.

## 3. Python bindings

Install and configure the binding with [bindings/python/README.md](../bindings/python/README.md). The installed distribution is `edge-dit-python`; Python code imports `edge_dit`.

### Direct Python example

`@python
from edge_dit import Engine, ImageRequest

config = {
    "model_path": "/absolute/path/to/FLUX.1-dev",
    "backend": "cuda",
    "offload_params_to_cpu": True,
    "text_encoder_offload": True,
    "max_vram_gb": 8.0,
}

with Engine(**config) as engine:
    request = ImageRequest(
        prompt="a glass teapot on a wooden table",
        width=256,
        height=256,
        steps=4,
        seed=42,
    )
    image = engine.generate_image(request)[0]
    image.save("output.png")
`@

Use the context-manager form so `engine.close()` is always called. An Engine serializes generation calls made through that object.

### `EngineConfig` fields

Create `EngineConfig` when you prefer an explicit configuration object. Passing the same names directly to `Engine` is also supported.

| Field | Meaning | When to set it |
| --- | --- | --- |
| `model_path` | Complete model directory. | Preferred for Diffusers-style models. |
| `diffusion_model_path` | Main transformer/denoiser file. | Use only with the separate component paths below. |
| `high_noise_diffusion_model_path` | Optional high-noise transformer. | Models that split low/high-noise stages. |
| `vae_path` | Variational autoencoder weights. | Required in separate-component mode. |
| `audio_vae_path` | MiniMax-H3 audio decoder weights. | Set to receive generated audio. |
| `clip_l_path`, `clip_g_path` | CLIP text encoder weights. | Required by model families that use them. |
| `clip_vision_path` | CLIP image encoder weights. | Image-conditioned models that require vision features. |
| `t5xxl_path` | T5 text encoder weights. | Models that use T5; omit only when `skip_t5=True` is supported. |
| `llm_path`, `llm_vision_path` | Language-model and vision-language-model weights. | Qwen-family configurations that require them. |
| `minimax_h3_stage_lifecycle` | Stage Qwen/VAE by phase and release between phases. | Useful for VRAM-constrained MiniMax-H3 runs. |
| `backend` | Device backend name. | Set `cuda` for the normal NVIDIA path. |
| `n_threads` | CPU thread count. | Leave unset/zero for automatic choice unless tuning CPU work. |
| `weight_type` | Requested load precision, such as `f16`, `bf16`, or `q4_k`. | Reduce VRAM only after a baseline run works. |
| `tensor_type_rules` | Per-tensor precision overrides. | Advanced mixed-precision tuning. |
| `offload_params_to_cpu` | Keep general parameters in system RAM. | Enable on VRAM-limited GPUs. |
| `dit_offload` | Keep DiT weights in RAM and stage them for each step. | Stronger VRAM saving with more transfer cost. |
| `text_encoder_offload` | Keep text encoder weights in RAM. | Enable when text encoder does not fit alongside generation. |
| `vae_offload` | Keep VAE weights in RAM. | Enable when VAE decode causes VRAM pressure. |
| `max_vram_gb` | Maximum compute-placement budget in GiB. | Set a hard GPU memory budget, for example `8.0`. |
| `auto_allocate` | Automatically place components within the VRAM budget. | Use with `max_vram_gb` when manual offload choices are not enough. |
| `auto_fit` | Select quantization and placement to fit the budget. | Automatic fallback; it overrides `weight_type`. |
| `vae_tiling`, `vae_tile_size` | Decode VAE output in tiles. | Use `auto`/`True` to lower peak VAE memory. |
| `flash_attention` | Enable compatible fast attention. | Keep enabled unless diagnosing a kernel issue. |
| `cfg_parallel_size`, `tp_parallel_size`, `sp_parallel_size` | Multi-GPU parallel sizes. | Leave at defaults for a one-GPU setup. |

A valid configuration needs either `model_path`, the image component set
(`diffusion_model_path`, `vae_path`, `clip_l_path`, and `t5xxl_path` or
`skip_t5=True`), or the MiniMax-H3 component set (`diffusion_model_path`,
`vae_path`, and `llm_path`).

### Image and video request fields

| Field | Image | Video | Meaning |
| --- | --- | --- | --- |
| `prompt` | required | required | Text describing the output. |
| `negative_prompt` | yes | yes | Text to avoid, for models that support it. |
| `width`, `height` | yes | yes | Output dimensions in pixels. |
| `frames` | no | yes | Number of output frames. MiniMax-H3 requires at least 22 and must satisfy `17k+5`. |
| `steps` | yes | yes | Denoising iterations. |
| `seed` | yes | yes | Random seed. |
| `guidance` / `distilled_guidance` | yes | yes | Distilled guidance; the two names must agree when both are set. |
| `cfg_scale` | yes | yes | Classifier-free guidance scale. |
| `image_cfg_scale` | yes | no | Image-condition guidance scale. |
| `flow_shift` | yes | yes | Flow scheduler shift for compatible models. |
| `sampler`, `scheduler` | yes | yes | Sampling algorithm choices. |
| `batch_count` | yes | no | Number of images to generate. |
| `init_image` | yes | yes | Starting image; for MiniMax-H3 this is the first frame. |
| `end_image` | no | yes | MiniMax-H3 last frame. |
| `mask_image` | yes | no | Area to edit for inpainting-style models. |
| `control_image` | yes | no | Control image for compatible ControlNet-style models. |
| `ref_images` | yes | yes | One or more reference images for compatible models. |
| `ref_videos`, `ref_audios` | no | yes | MiniMax-H3 `RefVideoInput` and `AudioInput` references. |
| `ref_image_size` | no | yes | MiniMax-H3 reference sizing: `max` or `match`. |
| `output_type` | yes | yes | `pil` for Pillow images or `numpy` for NumPy arrays. |
| `cache_mode` and related cache fields | yes | yes | Optional computation-reuse tuning. Keep disabled until the normal path is verified. |

`Engine.generate_video()` returns a list-compatible `VideoOutput`. Frames remain
indexable as before; MiniMax audio is copied into `output.audio` with
`output.audio_sample_rate` and `output.audio_channels` before the native result
is freed.

Python validation rejects an empty prompt, non-positive dimensions/steps/frames, invalid image objects, and unsupported `output_type` values before native generation begins.

## 4. Python Server

Python Server is a job-based HTTP service built on the Python bindings. It loads an Engine once and processes jobs serially. It is the backend used by the Python Server Console.

For an end-to-end setup and image/video walkthrough, including the browser console, direct `Engine` examples, repeat generation, and MP4 saving, read the [Python bindings tutorial](../bindings/python/README.md).

### Start it

`@bash
edge-dit-server \
  --model /absolute/path/to/FLUX.1-dev \
  --backend cuda \
  --host 127.0.0.1 \
  --port 8080 \
  --offload-to-cpu \
  --keep-text-encoder-on-cpu \
  --max-vram 8
`@

Use `python -m edge_dit.server` when the package executable is not on `PATH`.

| CLI option | Meaning |
| --- | --- |
| `--model` | Complete model directory. |
| `--diffusion-model`, `--vae`, `--audio-vae`, `--embeddings-connectors`, `--latent-upscaler`, `--clip_l`, `--clip_g`, `--t5xxl`, `--llm`, `--llm-vision` | Separate-component model paths. The connector and latent upscaler are LTX-2.3 components. |
| `--minimax-h3-stage-lifecycle` | Release MiniMax Qwen/VAE allocations between phases. |
| `--backend` | Backend name, normally `cuda`. |
| `--threads` | CPU thread count. |
| `--max-vram` | GPU memory budget in GiB. |
| `--offload-to-cpu` | Keep model parameters in system RAM. |
| `--keep-text-encoder-on-cpu` | Offload text encoder parameters. |
| `--keep-vae-on-cpu` | Offload VAE parameters. |
| `--skip-t5` | Do not load T5 for a model that supports skipping it. |
| `--job-ttl-seconds` | How long terminal jobs and in-memory results are kept. Negative disables automatic cleanup. |

### HTTP prefix and endpoints

The canonical prefix is `/ed/v2`. It remains `v2` because it is the Python Server protocol version and prevents collision with the native server's `/ed/v1` API.

| Method and path | Meaning |
| --- | --- |
| `GET /ed/v2/health` | Server is reachable and its model finished loading. |
| `GET /ed/v2/capabilities` | Loaded model name, supported image/video modes, default sampler/scheduler, endpoints, and retention policy. |
| `POST /ed/v2/images/generations` | Create an image job. |
| `POST /ed/v2/videos/generations` | Create a video job. |
| `GET /ed/v2/jobs` | List jobs; accepts `status`, `kind`, and `limit` query filters. |
| `POST /ed/v2/jobs/cleanup` | Delete expired terminal jobs immediately. |
| `GET /ed/v2/jobs/{job_id}` | Get one job's state, parameters, and sampling-step progress. |
| `DELETE /ed/v2/jobs/{job_id}` | Delete a terminal job. Active jobs cannot be deleted. |
| `POST /ed/v2/jobs/{job_id}/cancel` | Request cooperative cancellation. |
| `GET /ed/v2/jobs/{job_id}/result` | Get image/video output after a successful job. |
| `GET /ed/v2/jobs/{job_id}/video?fps=24` | Encode a successful video result as an MP4 download; requires `ffmpeg` on the server host. |

The aliases `/edgedit/v2` and `/edge-dit/v2` expose the same routes.

### Create, poll, and retrieve a job

`@bash
curl -s http://127.0.0.1:8080/ed/v2/images/generations \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-001' \
  -d '{
    "prompt": "a glass teapot on a wooden table",
    "width": 256,
    "height": 256,
    "steps": 4,
    "seed": 42
  }'
`@

The create response contains:

| Field | Meaning |
| --- | --- |
| `id` | UUID that identifies the job. |
| `kind` | `image` or `video`. |
| `status` | Current lifecycle state. |
| `status_url` | URL to poll for state and progress. |
| `cancel_url` | URL used to request cancellation. |
| `result_url` | URL that returns the final output after success. |
| `parameters` | Normalized request values accepted by the server. |
| `request_id` | Server-generated or client-provided trace identifier. |

Poll the returned `status_url`. When `status` becomes `succeeded`, request `result_url`. Image results use `data[].b64_png`; each entry is a base64-encoded PNG. Video results use `frames[].b64_png`. To save those frames as a container, request `/jobs/{job_id}/video?fps=<fps>`; MiniMax-H3 uses 24 fps, and its generated audio is muxed when available.

### Job lifecycle

| State | Meaning |
| --- | --- |
| `queued` | Accepted but waiting behind earlier work. |
| `running` | The one worker thread is generating. |
| `cancelling` | Cancellation was requested; the native engine will stop at a safe step boundary. |
| `cancelled` | The request stopped before completion. |
| `succeeded` | Output is ready at `result_url`. |
| `failed` | Generation or input handling failed; read `error`. |

Progress reports only sampling steps. `0/N` can mean prompt encoding is running; `N/N` can remain visible while the VAE converts output. It is not by itself an error.

### Image input fields in HTTP JSON

The image endpoint accepts `init_image_b64`, `mask_image_b64`, `control_image_b64`, and `ref_images_b64`. Each value is raw base64 image bytes or a `data:image/...;base64,...` URL. The server decodes it to a Pillow image before creating the request.

### Errors

Every error response has this shape:

`@json
{
  "error": {
    "message": "human-readable explanation",
    "type": "invalid_request_error",
    "code": "invalid_request",
    "status": 400,
    "request_id": "..."
  },
  "request_id": "..."
}
`@

Common `code` values are `invalid_json`, `invalid_request`, `not_found`, `job_not_found`, `job_not_ready`, `job_active`, `unsupported`, `method_not_allowed`, and `edge_dit_error`.

## 5. Python Server Console

The browser console is in [bindings/python/frontend/server-console](../bindings/python/frontend/server-console). It starts a runtime manager at port `8090`, a Python Server at port `8080`, and Vite at port `5173`.

Follow [the Python setup guide](../bindings/python/README.md) for the exact shared-library build, virtual environment, model variables, and `npm run dev:managed` command. The console's [runtime configuration guide](../bindings/python/frontend/server-console/RUNTIME_CONFIGURATION.md) lists each model profile and its required environment variable.
