# Python Bindings and Python Server

This directory contains two ways to use edge-dit.cpp from Python:

- **Python bindings**: call `Engine` directly from a Python program.
- **Python Server**: keep one loaded model in a process and expose image/video jobs over HTTP. The browser console uses this server.

The shortest path for a new machine is: build the shared native library, install the Python package in a virtual environment, set one model path, and start the managed console.

## 1. Requirements

You need:

- Linux with Python 3.10 or newer.
- A C++17 compiler, CMake 3.20 or newer, Git, and the project submodules.
- CUDA Toolkit and a CUDA-capable GPU for `backend=cuda`.
- Node.js and npm for the browser console.
- `ffmpeg` on `PATH`, `EDGE_DIT_FFMPEG=/absolute/path/to/ffmpeg`, or the optional `imageio-ffmpeg` package when you want the Python Server Console to save a completed video as MP4.
- A Diffusers model directory. The directory should contain `model_index.json` and the model component folders/files. Do not point at a single `.safetensors` file unless you use the separate-component options described below.

Fetch the submodules from an existing checkout:

```bash
cd /absolute/path/to/edge-dit.cpp
git submodule update --init --recursive
```

## 2. Build the library Python needs

The normal CUDA build produces `build-cuda/bin/ed-cli`. Python needs a shared library instead. From the repository root run:

```bash
ED_BUILD_SHARED_LIBS=ON \
BUILD_DIR=build-cuda-shared \
bash scripts/build_cuda.sh
```

After a successful build, these files should exist:

```text
build-cuda-shared/bin/libedgedit.so
build-cuda-shared/bin/libggml.so
build-cuda-shared/bin/libggml-base.so
build-cuda-shared/bin/libggml-cpu.so
build-cuda-shared/bin/libggml-cuda.so
```

The CUDA build script can install compatible cuDNN Python wheels in user space when cuDNN is not already available. CUDA Toolkit and the NVIDIA driver remain system dependencies.

## 3. Install the Python environment

Create a virtual environment in the repository and install the bindings in editable mode:

```bash
cd /absolute/path/to/edge-dit.cpp
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './bindings/python[dev]'
```

The base package installs Pillow. The `dev` extra also installs NumPy and pytest. Use `./bindings/python[numpy]` instead of `./bindings/python[dev]` when you only need NumPy output.

Set the native library variables once in the same shell:

```bash
export EDGE_DIT_REPO_ROOT="$PWD"
export EDGE_DIT_PYTHON_BIN="$PWD/.venv/bin/python"
export EDGE_DIT_LIBRARY="$PWD/build-cuda-shared/bin/libedgedit.so"
export EDGE_DIT_DEPENDENCY_DIRS="$PWD/build-cuda-shared/bin"
export PYTHONPATH="$PWD/bindings/python/src:$PYTHONPATH"
```

`EDGE_DIT_LIBRARY` is the exact shared library to load. `EDGE_DIT_DEPENDENCY_DIRS` is a colon-separated list of directories containing CUDA, cuDNN, and sibling ggml shared libraries. The loader also searches common CUDA and NVIDIA Python-wheel locations.

## 4. Configure a model

The managed console reads model paths from environment variables. Directory
variables must point to the complete Diffusers directory containing
`model_index.json`; component variables must point to the stated weight file,
shard index, or component directory.

All managed profiles default to `auto_allocate=true`. This preserves every
source tensor's stored precision and decides per component what remains resident
on the GPU and what streams from CPU. With no profile `max_vram_gb`, the budget
is the live free VRAM. This is automatic placement, not automatic quantization.

| Profile | Required environment variable(s) | Model type |
| --- | --- | --- |
| `flux-dev` | `EDGE_DIT_FLUX_MODEL_PATH` | FLUX.1-dev image model |
| `flux-schnell` | `EDGE_DIT_FLUX_SCHNELL_MODEL_PATH` | FLUX.1-schnell image model |
| `flux-kontext` | `EDGE_DIT_FLUX_KONTEXT_MODEL_PATH` | FLUX.1-Kontext image-edit model |
| `kontext-lightning` | `EDGE_DIT_FLUX_KONTEXT_MODEL_PATH`, `EDGE_DIT_KONTEXT_LIGHTNING_DIT_PATH` | Kontext base components plus Lightning transformer |
| `flux2-klein-4b` | `EDGE_DIT_FLUX2_KLEIN_4B_MODEL_PATH` | FLUX.2 [klein] 4B image model; not FLUX.2-dev |
| `qwen-image` | `EDGE_DIT_QWEN_IMAGE_MODEL_PATH` | Qwen-Image model |
| `qwen-image-lightning` | `EDGE_DIT_QWEN_IMAGE_MODEL_PATH`, `EDGE_DIT_QWEN_IMAGE_LIGHTNING_DIT_PATH` | Qwen-Image base components plus merged Lightning transformer |
| `qwen-image-edit` | `EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH` | Qwen-Image-Edit model |
| `qwen-image-edit-lightning` | `EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH`, `EDGE_DIT_QWEN_IMAGE_EDIT_LIGHTNING_DIT_PATH` | Qwen-Image-Edit base components plus merged Lightning transformer |
| `sd3-medium` | `EDGE_DIT_SD3_MODEL_PATH` | Stable Diffusion 3 Medium image model |
| `sd35-medium-turbo` | `EDGE_DIT_SD35_TURBO_MODEL_PATH` | Stable Diffusion 3.5 Medium Turbo image model |
| `wan-t2v` | `EDGE_DIT_WAN_VIDEO_MODEL_PATH` | Wan2.1 T2V 1.3B video model |
| `wan2-t2v-14b` | `EDGE_DIT_WAN_14B_MODEL_PATH` | Wan2.1 T2V 14B video model |
| `wan21-t2v-1.3b-distill` | `EDGE_DIT_WAN_VIDEO_MODEL_PATH`, `EDGE_DIT_WAN_DISTILL_DIT_PATH` | Wan 1.3B base components plus distilled transformer |
| `minimax-h3` | `EDGE_DIT_MINIMAX_DIT_PATH`, `EDGE_DIT_MINIMAX_LLM_PATH`, `EDGE_DIT_MINIMAX_VIDEO_VAE_PATH`, `EDGE_DIT_MINIMAX_AUDIO_VAE_PATH` | MiniMax-H3 FL2VA components |

MiniMax-H3 additionally enables staged component lifecycle and fixed video-VAE
tiling. Its output must contain at least 22 frames and satisfy `17k+5`.

Example for the default FLUX profile:

```bash
export EDGE_DIT_FLUX_MODEL_PATH=/absolute/path/to/FLUX.1-dev
test -f "$EDGE_DIT_FLUX_MODEL_PATH/model_index.json"
```

The `test` command must succeed. If it fails, fix the path before starting the server. To use another profile, set its variable and pass the profile name to the start command.

## 5. Start the complete frontend and backend

Install the frontend dependencies once:

```bash
cd "$EDGE_DIT_REPO_ROOT/bindings/python/frontend/server-console"
npm install
```

Start the frontend, runtime manager, and managed Python Server together:

```bash
EDGE_DIT_FLUX_MODEL_PATH=/absolute/path/to/FLUX.1-dev \
npm run dev:managed
```

The command starts:

| Service | URL | Purpose |
| --- | --- | --- |
| Browser console | `http://127.0.0.1:5173` | React/Vite user interface |
| Runtime manager | `http://127.0.0.1:8090/runtime/v1` | Starts, stops, and monitors model profiles |
| Python Server | `http://127.0.0.1:8080/ed/v2` | Job API used by the console |

For a different model, replace the variable and profile:

```bash
EDGE_DIT_QWEN_IMAGE_MODEL_PATH=/absolute/path/to/Qwen-Image \
npm run dev:managed -- --auto-start-profile qwen-image
```

MiniMax-H3 is a component profile rather than a complete Diffusers directory:

```bash
export EDGE_DIT_PYTHON_BIN=python
export EDGE_DIT_MINIMAX_DIT_PATH=/models/minimax-h3/diffusion_models/minimax-h3-Q8_0.gguf
export EDGE_DIT_MINIMAX_LLM_PATH=/models/minimax-h3/text_encoders/qwen3vl-minimax-Q4_K_M.gguf
export EDGE_DIT_MINIMAX_VIDEO_VAE_PATH=/models/minimax-h3/vae/minimax-h3-video-vae-fp16.safetensors
export EDGE_DIT_MINIMAX_AUDIO_VAE_PATH=/models/minimax-h3/vae/minimax-h3-audio-vae-fp32.safetensors

CUDA_VISIBLE_DEVICES=6 npm run dev:managed -- --auto-start-profile minimax-h3
```

`CUDA_VISIBLE_DEVICES=6` exposes physical GPU 6 to the managed backend as its
logical CUDA device 0. Omit it to use the CUDA runtime's default visible device.

For access from another device on the same network, bind all three services:

```bash
EDGE_DIT_FLUX_MODEL_PATH=/absolute/path/to/FLUX.1-dev \
npm run dev:managed:network
```

This exposes the services on `0.0.0.0`; use the host machine's IP address in the browser. Do not expose these development endpoints to the public Internet without adding authentication and TLS.

Check the services from another terminal:

```bash
curl http://127.0.0.1:8090/runtime/v1/status
curl http://127.0.0.1:8080/ed/v2/health
```

The Python Server health response is `status: ok` only after the model has finished loading. Loading a large model can take several minutes and uses system RAM while the model is being prepared.

Stop the complete stack with `Ctrl-C` in the terminal running `npm run dev:managed`.

### Generate an image or video in the browser

1. Open `http://127.0.0.1:5173` (or the server IP when using `dev:managed:network`).
2. In **Verified model**, choose the configured model and click **Start model** or **Switch model**.
3. Wait for **backend running** and **health ok**. Click **Use target** only when the button does not already say **Target synced**.
4. Click **Apply preset**. Profiles use the same model-native generation defaults as `benchmark/models`: FLUX and SD3 use 20 steps, Qwen-Image uses 30, Wan 1.3B uses 30, and MiniMax-H3 uses 20. The MiniMax-H3 preset uses 22 frames, the minimum valid `17k+5` value.
5. Edit the prompt, then click **Create image job** or **Create video job**. The **Video** tab only changes composer mode; it does not submit a job.
6. After a video succeeds, choose the playback/export FPS and click **Save Video as MP4**. The server uses `ffmpeg` (including the binary bundled by `imageio-ffmpeg` when installed) and includes MiniMax-H3 audio when the audio VAE was loaded.

To generate another image or video, do not restart the model. Change the prompt,
seed, or other request fields and click **Create image job** or **Create video
job** again. The Python Server queues jobs serially and keeps earlier completed
results in the task list until their TTL expires or you delete them.

MP4 export is available for every successful video job, including Wan and
MiniMax-H3. Select the intended FPS in the Result Viewer and click **Save Video
as MP4**; Wan profiles default to 16 fps and MiniMax-H3 defaults to 24 fps.

## 6. Start the Python Server directly

Use this mode when command-line control over quantization and offload is more
important than managed profile switching. The direct server can be used by
itself or with the browser console.

```bash
cd "$EDGE_DIT_REPO_ROOT"
. .venv/bin/activate
export EDGE_DIT_LIBRARY="$PWD/build-cuda-shared/bin/libedgedit.so"
export EDGE_DIT_DEPENDENCY_DIRS="$PWD/build-cuda-shared/bin"
export PYTHONPATH="$PWD/bindings/python/src:$PYTHONPATH"

edge-dit-server \
  --model /absolute/path/to/FLUX.1-dev \
  --backend cuda \
  --host 127.0.0.1 \
  --port 8080 \
  --max-vram 20
```

The equivalent module command is `python -m edge_dit.server`. The server loads one model at startup, accepts jobs, and executes them serially on one worker thread. `auto_allocate` is the default Server placement policy: it preserves the source tensor types and decides per component whether weights stay resident or stream from CPU under `min(--max-vram, live free VRAM)`. Omit `--max-vram` to plan against live free VRAM.

Choose a different policy explicitly when needed:

```bash
# Automatic quantization plus placement. TE/DiT may be lowered to Q8_0 or Q4_K.
edge-dit-server --model /models/FLUX.1-dev --backend cuda \
  --auto-fit --max-vram 20

# User-selected quantization; automatic component placement remains enabled.
edge-dit-server --model /models/Qwen-Image --backend cuda \
  --type q4_k --max-vram 20

# Fully manual placement. Only the named components stream from CPU.
edge-dit-server --model /models/Qwen-Image --backend cuda \
  --no-auto-allocate --type q8_0 \
  --text-encoder-offload --vae-offload

# Legacy full offload. Every component streams from CPU.
edge-dit-server --model /models/Qwen-Image --backend cuda \
  --no-auto-allocate --offload-to-cpu
```

`--type preserve` (also spelled `auto`) keeps the stored source type. `--type q8_0` or `--type q4_k` quantizes eligible safetensors while loading; a persistently converted GGUF should normally be loaded with `preserve`. `--tensor-type-rules` can override individual tensor groups. Under `--auto-fit`, the runtime owns TE/DiT precision selection and `--type` continues to control other eligible components such as the VAE.

`--max-vram` is a placement-planning budget for `auto_allocate`. With
single-device CUDA, `--auto-fit --max-vram` also enables a guarded allocation
ceiling. In fully manual mode, placement is determined by the explicit offload
flags; `--max-vram` is not a universal process-level peak-memory limit.

### MiniMax-H3 with an explicit manual policy

MiniMax-H3 must be supplied as separate components. When the DiT and LLM are
already quantized GGUF files, keep `--type preserve` so the FP16/FP32 VAEs are
not quantized again while loading:

```bash
CUDA_VISIBLE_DEVICES=6 python -m edge_dit.server \
  --host 127.0.0.1 \
  --port 8080 \
  --backend cuda \
  --diffusion-model /models/minimax-h3/diffusion_models/minimax-h3-Q8_0.gguf \
  --llm /models/minimax-h3/text_encoders/qwen3vl-minimax-Q4_K_M.gguf \
  --vae /models/minimax-h3/vae/minimax-h3-video-vae-fp16.safetensors \
  --audio-vae /models/minimax-h3/vae/minimax-h3-audio-vae-fp32.safetensors \
  --type preserve \
  --no-auto-allocate \
  --text-encoder-offload \
  --vae-offload \
  --minimax-h3-stage-lifecycle \
  --vae-tiling
```

This policy keeps the Q8 DiT resident when it fits, stages the Q4_K_M text
encoder and VAEs from CPU, and releases MiniMax-H3 phase-specific allocations
between conditioning, sampling, and decode.

### Attach the browser console to the direct server

In a second terminal, start only the UI:

```bash
cd /absolute/path/to/edge-dit.cpp/bindings/python/frontend/server-console
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/ed/v2` to the direct Python Server
at `127.0.0.1:8080`. Because no Runtime Manager is running, managed profile
start/stop/switch controls and `/runtime/v1` status are unavailable; image/video
job submission, progress, results, repeat generation, and MP4 export continue
to use the direct backend normally.

## 7. Use the Python binding directly

```python
from edge_dit import Engine

with Engine(
    model_path="/absolute/path/to/FLUX.1-dev",
    backend="cuda",
    auto_allocate=True,
    max_vram_gb=20.0,
) as engine:
    images = engine.generate_image(
        prompt="a glass teapot on a wooden table",
        width=256,
        height=256,
        steps=4,
        seed=42,
    )
    images[0].save("output.png")
```

The repository also includes runnable image and video programs:

```bash
python bindings/python/examples/basic_txt2img.py \
  --model /absolute/path/to/FLUX.1-dev \
  --prompt "a glass teapot on a wooden table" \
  --output output.png \
  --backend cuda

python bindings/python/examples/basic_txt2vid.py \
  --model /absolute/path/to/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "a small robot walking through rain" \
  --output output.gif \
  --backend cuda \
  --frames 17
```

These examples intentionally use PNG/GIF so their only media dependency is Pillow. Use the browser console's **Save Video as MP4** action when you need an MP4 container, or adapt the returned Pillow frames to your own encoder.

`model_path` is the model directory. `backend` selects `cuda`, `cpu`, `vulkan`, `metal`, or `auto` when that backend is available. `auto_allocate` chooses component placement without changing precision. `auto_fit` also chooses TE/DiT precision. `offload_params_to_cpu` forces full offload, while `dit_offload`, `text_encoder_offload`, and `vae_offload` force individual components to stream from CPU. `max_vram_gb` limits the planning budget; it is not a reservation or a universal process-peak clamp.

## 8. Parameters that matter first

| Parameter | Meaning | Practical first value |
| --- | --- | --- |
| `width`, `height` | Output size in pixels | `256` for a smoke test; use the model's native size for quality |
| `steps` | Denoising iterations | `1` only for wiring checks; `20` is a normal starting point |
| `seed` | Reproducible random seed | `42`; omit it for a random result |
| `guidance` | FLUX distilled guidance | Leave unset unless the model recommends a value |
| `cfg_scale` | Classifier-free guidance for supported pipelines | Model-specific; often `1` or `5` |
| `frames` | Number of video frames | Model-specific; MiniMax-H3 needs at least 22 and must satisfy `17k+5` |
| `weight_type` | On-the-fly weight format such as `q4_k` | `preserve`; use `q4_k` only when explicitly desired |
| `vae_tiling` | Decode in tiles to reduce peak VRAM | `auto` |
| `cache_mode` | Optional computation reuse method | `disabled` until the baseline works |

Do not lower `steps` permanently to solve an out-of-memory error. First use `auto_allocate` with a realistic VRAM budget, then select component/full offload or an explicit quantization policy.

### MiniMax-H3 video and audio

```python
from edge_dit import AudioInput, Engine, RefVideoInput, VideoRequest
from PIL import Image

with Engine(
    diffusion_model_path="/models/minimax-h3/dit.gguf",
    vae_path="/models/minimax-h3/video-vae.safetensors",
    audio_vae_path="/models/minimax-h3/audio-vae.safetensors",
    llm_path="/models/minimax-h3/qwen.gguf",
    backend="cuda",
    auto_allocate=True,
    minimax_h3_stage_lifecycle=True,
) as engine:
    output = engine.generate_video(VideoRequest(
        prompt="A cinematic ocean sunrise.",
        init_image=Image.open("first.png"),
        end_image=Image.open("last.png"),
        width=768, height=1344, frames=90, steps=20,
        sampler="res_multistep", scheduler="simple",
    ))
    output[0].save("first-generated-frame.png")
    # output.audio is copied interleaved float data, or None when no audio VAE is loaded.
    print(output.audio_sample_rate, output.audio_channels)
```

`VideoRequest` also accepts `ref_images`, `ref_image_size`, `ref_videos`
(`RefVideoInput`), and `ref_audios` (`AudioInput`). `VideoOutput` remains a
list-compatible frame collection, so existing frame-only code continues to
work.

### LTX-2.3 video and audio

LTX-2.3 uses separate diffusion, Gemma, connector, and VAE files. The connector
and optional spatial upscaler paths are explicit engine configuration fields:

```python
from edge_dit import Engine, VideoRequest

with Engine(
    diffusion_model_path="/models/LTX-2.3-GGUF/diffusion_models/ltx-2.3-22b-dev-UD-Q4_K_M.gguf",
    vae_path="/models/LTX-2.3-GGUF/vae/ltx-2.3-22b-dev_video_vae.safetensors",
    audio_vae_path="/models/LTX-2.3-GGUF/vae/ltx-2.3-22b-dev_audio_vae.safetensors",
    llm_path="/models/LTX-2.3-GGUF/text_encoders/gemma-3-12b-it-UD-Q4_K_XL.gguf",
    embeddings_connectors_path="/models/LTX-2.3-GGUF/text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors",
    latent_upscaler_path="/models/LTX-2.3-GGUF/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    backend="cuda",
    auto_fit=True,
    max_vram_gb=30.0,
    fit_width=1280,
    fit_height=720,
    fit_frames=33,
    fit_fps=24,
) as engine:
    output = engine.generate_video(VideoRequest(
        prompt="a red fox walks through a sunlit autumn forest",
        width=1280,
        height=720,
        frames=33,
        fps=24,
        steps=20,
        cfg_scale=6.0,
    ))
```

`VideoRequest` additionally accepts `init_image` for I2V, `end_image` for E2V,
both for FLF2V, and `hires=True` with `hires_steps`,
`hires_denoising_strength`, or an explicit `hires_sigmas` sequence for the x2
latent refinement pass. LTX requires Euler with the `ltx2` scheduler; cache,
sequence parallelism, and tensor parallelism are currently rejected explicitly.

## 9. HTTP jobs at a glance

The Python Server uses the `/ed/v2` protocol. `v2` is the HTTP contract version; it is not the product name. A request creates a job immediately, then the client polls that job until it succeeds, fails, or is cancelled.

```bash
curl -s http://127.0.0.1:8080/ed/v2/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a glass teapot","width":256,"height":256,"steps":4,"seed":42}'
```

The response contains `id`, `status_url`, and `result_url`. Poll `status_url`; when `status` is `succeeded`, fetch `result_url` and decode `data[].b64_png`. See [docs/api.md](../../docs/api.md) for every endpoint, field, error, and lifecycle state.

Video jobs use `POST /ed/v2/videos/generations`. After the job succeeds, download an MP4 at `GET /ed/v2/jobs/{job_id}/video?fps=24`. MiniMax-H3 should be exported at 24 fps. This endpoint requires `ffmpeg` on the Python Server host.

## 10. Troubleshooting

- **`ModuleNotFoundError: edge_dit`**: activate the virtual environment and set `PYTHONPATH` as shown above.
- **Cannot load `libedgedit.so`**: rebuild with `ED_BUILD_SHARED_LIBS=ON` and check `EDGE_DIT_LIBRARY`.
- **`model_index.json` not found**: point the profile variable at the model directory, not its parent or one weight file.
- **Backend stays `starting`**: wait for model loading; inspect the runtime manager status and its log tail before restarting.
- **CUDA or cuDNN errors**: verify `nvcc --version`, the NVIDIA driver, and `EDGE_DIT_DEPENDENCY_DIRS`. A Python package install cannot replace the system CUDA compiler or driver.
- **Out of memory**: lower the `auto_allocate` VRAM budget, force selected components to offload, or explicitly select `q8_0`/`q4_k`; do not disguise the problem by changing the model's normal sampling steps.

Run the fast local checks with:

```bash
PYTHONPATH=bindings/python/src python -m unittest discover -s bindings/python/tests -p 'test_server*.py' -v
```
