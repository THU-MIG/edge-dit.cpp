# Python Server Console

The Python Server Console is the local browser interface for `edge_dit.server`. It starts one selected model, sends job requests, shows progress, and displays image or video results.

For the complete installation, shared-library build, virtual environment, and model-path setup, read [the Python guide](../../README.md) first.

## Start the local stack

From this directory:

```bash
npm install
EDGE_DIT_FLUX_MODEL_PATH=/absolute/path/to/FLUX.1-dev \
npm run dev:managed
```

Open `http://127.0.0.1:5173`. The first model load can take several minutes. The console is usable once the runtime panel reports that the backend is running.

For a video profile, click **Apply preset**, edit the prompt, and click **Create
video job**. A successful Wan or MiniMax-H3 result can be exported with **Save
Video as MP4** (`ffmpeg` must be on the Python Server host). Wan profiles default
to 16 fps and MiniMax-H3 defaults to 24 fps. To run again without reloading the
model, change the prompt, seed, or other request fields and click **Create video
job** again. MiniMax-H3 must use at least 22 frames and a frame count satisfying
`17k+5`; its verified preset uses 22.

The command starts three local services:

| Service | Address | What it does |
| --- | --- | --- |
| Console | `http://127.0.0.1:5173` | Browser interface and development proxy |
| Runtime manager | `http://127.0.0.1:8090/runtime/v1` | Starts/stops profiles and retains a log tail |
| Python Server | `http://127.0.0.1:8080/ed/v2` | Runs generation jobs |

Every managed profile defaults to `auto_allocate=true`: stored tensor precision
is preserved, and the runtime decides per component what stays resident on the
GPU and what streams from CPU using the live free-VRAM budget. The profile
manager owns the backend process, so model switching, restart, and runtime log
tailing are available in the console.

Use another built-in model profile by setting its model variable and forwarding the profile name:

```bash
EDGE_DIT_WAN_VIDEO_MODEL_PATH=/absolute/path/to/Wan2.1-T2V-1.3B-Diffusers \
npm run dev:managed -- --auto-start-profile wan-t2v
```

The available profiles and their variables are documented in [RUNTIME_CONFIGURATION.md](RUNTIME_CONFIGURATION.md).
The `minimax-h3` profile uses four standalone component variables; MiniMax-H3
requests must contain at least 22 frames and satisfy `17k+5`.

## Start a custom-policy backend with the console

Use this mode when you need command-line control over quantization or offload.
Start the Python Server yourself instead of using `dev:managed`. For example,
this MiniMax-H3 command preserves an existing Q8 DiT and Q4_K_M LLM, keeps the
DiT resident when it fits, and stages the text encoder and VAEs from CPU:

```bash
conda activate edge
cd /absolute/path/to/edge-dit.cpp

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

Then start only the UI in a second terminal:

```bash
conda activate edge
cd /absolute/path/to/edge-dit.cpp/bindings/python/frontend/server-console
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/ed/v2` to the direct Python Server
on port 8080. Because the Runtime Manager is not running, managed profile
start/stop/switch controls and `/runtime/v1` status are unavailable; generation,
progress, results, repeat generation, and MP4 export still work normally.

Other useful policies are:

```text
--type q8_0 --auto-allocate --max-vram 20
--auto-fit --max-vram 20
--no-auto-allocate --dit-offload --text-encoder-offload --vae-offload
--no-auto-allocate --offload-to-cpu
```

`--type` quantizes eligible safetensors while loading. Use `--type preserve`
for pre-quantized GGUF files unless you explicitly want another conversion.
`--auto-fit` chooses TE/DiT quantization as well as placement. See
[Python bindings and Server](../../README.md#6-start-the-python-server-directly)
for complete semantics and non-MiniMax examples.

## Useful commands

```bash
npm run dev
npm run runtime:manager -- --auto-start-profile flux-dev
npm run dev:managed:network
npm run build
npm test
npm run test:e2e
```

`npm run dev` starts only the UI; a separately running Python Server is required. `npm run runtime:manager` starts only the model manager. `npm run dev:managed:network` binds all services to `0.0.0.0` for a trusted local network.

Stop the managed stack with `Ctrl-C`. The runtime manager also stops the Python Server it created.

## Connection targets

The console uses `/ed/v2` by default. This is the Python Server HTTP protocol version. The native C++ server has a different `/ed/v1` contract, so do not point this console at a native server unless it implements the Python Server job endpoints.

The Console's Connection panel can probe a server running on another machine. For a remote target, use its reachable base URL and the `/ed/v2` prefix. The browser must be allowed to reach that address, and the server must be configured for the desired network binding.

## Troubleshooting

- **The UI opens but the backend is starting**: model weights are still loading. Check `http://127.0.0.1:8090/runtime/v1/status`.
- **The manager exits immediately**: ensure `EDGE_DIT_PYTHON_BIN` points to the virtual environment Python and `EDGE_DIT_LIBRARY` points to `libedgedit.so`.
- **A profile cannot find its model**: set the profile's exact `EDGE_DIT_*_MODEL_PATH` variable to the directory containing `model_index.json`.
- **Port already in use**: stop the previous managed stack or override `EDGE_DIT_RUNTIME_MANAGER_PORT` and `EDGE_DIT_MANAGED_BACKEND_PORT`. The frontend port is passed through Vite with `npm run dev -- --port <port>`.
