# edge-dit Python bindings

This package provides a minimal Python binding for `edge-dit.cpp`.

If you want `numpy` outputs, install the optional extra:

```bash
pip install '.[numpy]'
```

So if you want real `numpy` outputs, make sure the interpreter you use has both:

- a compatible native runtime for `libedgedit.so`
- the `numpy` package installed

Current implemented scope:

- create an engine from the public C API
- generate images synchronously
- generate videos synchronously
- expose native pipeline / version / capability queries
- expose native default sampler / scheduler queries
- expose polling-style progress step queries
- expose cooperative cancel requests for in-flight generation
- return `PIL.Image.Image` by default
- optionally return `numpy.ndarray` with `output_type="numpy"`
- return video frames as `list[PIL.Image.Image]` by default
- optionally return video frames as `list[numpy.ndarray]` with `output_type="numpy"`
- accept `PIL.Image.Image` inputs for `init_image` / `mask_image` / `control_image` / `ref_images`
- release native resources explicitly with `close()` or a context manager
- expose the full current `ed_sample_params_t` cache tuning surface
- provide copyable helper scripts for shared-library build and Python smoke test
- provide an optional real-library integration test entrypoint
- enrich common load/generation exceptions with Python-side context
- surface cooperative cancellation as `GenerationCancelledError`
- provide Python `server_v2` non-blocking image / video job APIs over HTTP

Current deferred scope:

- `loras`
- progress callback
- pause / resume
- preview callback

## Python API usage

The main Python API is:

- `Engine`
- `EngineConfig`
- `ImageRequest`
- `VideoRequest`

The current runtime query / control surface on `Engine` also includes:

- `pipeline_name`
- `version_name`
- `supports_image`
- `supports_video`
- `default_sampler`
- `default_scheduler(...)`
- `progress_steps()`
- `request_cancel()`

Common failure modes now include extra Python-side context, for example:

- load failures include configured model/backend hints
- generation failures include request summary such as size, steps, and output type
- cooperative cancellation raises `GenerationCancelledError`

Current cancel / progress semantics are intentionally narrow:

- `progress_steps()` reports sampling-step progress only
- it does not include prompt encoding, VAE decode, or output encoding
- `request_cancel()` is cooperative and only takes effect at step boundaries
- there is no callback push API yet; current progress is polling-only

Minimal usage:

```python
from edge_dit import Engine

with Engine(
    model_path="/path/to/FLUX.1-dev",
    backend="cuda",
    offload_params_to_cpu=True,
    keep_text_encoder_on_cpu=True,
    max_vram_gb=8.0,
) as engine:
    images = engine.generate_image(
        prompt="a glass teapot on a wooden table",
        width=256,
        height=256,
        steps=1,
        seed=42,
    )
    images[0].save("/tmp/output.png")
```

Structured usage:

```python
from edge_dit import Engine, EngineConfig, ImageRequest

config = EngineConfig(
    model_path="/path/to/FLUX.1-dev",
    backend="cuda",
    offload_params_to_cpu=True,
    keep_text_encoder_on_cpu=True,
    max_vram_gb=8.0,
)

request = ImageRequest(
    prompt="a glass teapot on a wooden table",
    width=256,
    height=256,
    steps=1,
    seed=42,
    cache_mode="disabled",
)

with Engine(config) as engine:
    images = engine.generate_image(request)
    images[0].save("/tmp/output.png")
```

`EngineConfig.weight_type` accepts either an integer enum value or a string alias.
Supported string values are:

- `auto`
- `f32`
- `f16`
- `bf16`
- `q4_0`
- `q4_1`
- `q5_0`
- `q5_1`
- `q8_0`
- `q2_k`
- `q3_k`
- `q4_k`
- `q5_k`
- `q6_k`

Additional aliases accepted by the Python bindings:

- `fp32` -> `f32`
- `fp16` -> `f16`
- separators `-`, `_`, and `.` are normalized, so `q4-k`, `q4_k`, and `q4.k` all resolve to the same value

Runtime query example:

```python
from edge_dit import Engine

with Engine(model_path="/path/to/FLUX.1-dev", backend="cuda") as engine:
    print(engine.pipeline_name)
    print(engine.version_name)
    print(engine.supports_image, engine.supports_video)
    print(engine.default_sampler, engine.default_scheduler())
    print(engine.progress_steps())
```

## Python server_v2

The bindings now also ship a first `server_v2` runtime that uses the Python
`Engine` directly and exposes non-blocking job-based image / video generation
HTTP APIs on top of a serial execution runtime.

Start it with:

```bash
PYTHONPATH=bindings/python/src \
python -m edge_dit.server_v2 \
  --model /path/to/model \
  --backend cuda \
  --host 127.0.0.1 \
  --port 8080 \
  --job-ttl-seconds 3600
```

If the package is installed, the same entrypoint is also available as:

```bash
edge-dit-server-v2 --model /path/to/model --backend cuda
```

Current v2 endpoints:

- `GET /ed/v2/health`
- `GET /ed/v2/capabilities`
- `POST /ed/v2/images/generations`
- `POST /ed/v2/videos/generations`
- `GET /ed/v2/jobs`
- `POST /ed/v2/jobs/cleanup`
- `GET /ed/v2/jobs/{job_id}`
- `DELETE /ed/v2/jobs/{job_id}`
- `POST /ed/v2/jobs/{job_id}/cancel`
- `GET /ed/v2/jobs/{job_id}/result`

Aliases are also registered for `/edgedit/v2/...` and `/edge-dit/v2/...`.

Create an image job:

```bash
curl -s http://127.0.0.1:8080/ed/v2/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a cinematic photo of a glass teapot on a wooden table",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "guidance": 3.5
  }'
```

Create a video job:

```bash
curl -s http://127.0.0.1:8080/ed/v2/videos/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a small robot walking through a rainy neon street",
    "width": 416,
    "height": 240,
    "frames": 9,
    "steps": 20,
    "cfg_scale": 5.0,
    "flow_shift": 5.0
  }'
```

The create response returns a job id plus stable `status_url`, `cancel_url`, and
`result_url`. `GET /jobs/{job_id}` reports the current state and live progress.
Image results return `data[].b64_png`; video results return `frames[].b64_png`
where each frame is PNG-encoded.

Image-conditioned pipelines can also provide embedded input images on the image
job endpoint:

- `init_image_b64`
- `mask_image_b64`
- `control_image_b64`
- `ref_images_b64`

These fields accept either raw base64 image bytes or `data:image/...;base64,...`
URLs. `server_v2` decodes them to `PIL.Image.Image` before calling the Python
engine.

Qwen image edit example:

```json
{
  "prompt": "replace the mug with a glass teapot",
  "width": 1024,
  "height": 1024,
  "steps": 20,
  "init_image_b64": "<base64 png or jpeg bytes>"
}
```

For the managed `server_v2` web console and cross-device runtime setup, see:

```text
bindings/python/frontend/server_v2-console/RUNTIME_CONFIGURATION.md
```

FLUX Kontext example:

```json
{
  "prompt": "turn this product photo into a studio catalog shot",
  "width": 1024,
  "height": 1024,
  "steps": 20,
  "ref_images_b64": ["<base64 png or jpeg bytes>"]
}
```

Example with an explicit request id for upstream trace correlation:

```bash
curl -s http://127.0.0.1:8080/ed/v2/images/generations \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: edge-dit-demo-123' \
  -d '{"prompt":"smoke test teapot","width":256,"height":256,"steps":1}'
```

List and cleanup jobs:

```bash
curl -s 'http://127.0.0.1:8080/ed/v2/jobs?kind=video&status=succeeded&limit=20'
curl -s -X POST http://127.0.0.1:8080/ed/v2/jobs/cleanup
curl -s -X DELETE http://127.0.0.1:8080/ed/v2/jobs/<job_id>
```

`--job-ttl-seconds` controls how long terminal job metadata and in-memory
results are retained. The default is `3600`; use a negative value to disable TTL
cleanup. Expired terminal jobs are removed lazily when the registry is accessed
or explicitly through `POST /jobs/cleanup`.

Recommended starting values:

- image-heavy single-user workstation: `--job-ttl-seconds 900`
- shared development server: `--job-ttl-seconds 300` plus periodic `POST /jobs/cleanup`
- long-running debugging sessions: negative TTL only when you explicitly want to keep artifacts in memory

The HTTP layer returns a stable `request_id` in response bodies and the
`X-Request-ID` response header. Clients may provide `X-Request-ID` to correlate
server logs with upstream request traces.

Structured errors use this shape:

```json
{
  "error": {
    "message": "job is not ready; current status is running",
    "type": "invalid_request_error",
    "code": "job_not_ready",
    "status": 409,
    "request_id": "..."
  },
  "request_id": "..."
}
```

Current stable error codes include `invalid_request`, `invalid_json`,
`not_found`, `job_not_found`, `job_not_ready`, `job_active`, `unsupported`,
`method_not_allowed`, and `edge_dit_error`.

Current server_v2 semantics:

- image and video generation are both represented as job resources
- create-job requests return immediately, then clients poll job state and result URLs
- one loaded engine and one worker thread, so actual generation execution is serial
- `progress.current_step` / `total_steps` report sampling-step progress only
- cancellation is cooperative and only takes effect at step boundaries
- job results are kept in memory until TTL cleanup or explicit delete
- `GET /jobs/{job_id}/result` returns a conflict response until the job completes

Large video jobs can look "stuck" even when they are healthy:

- `0/N` often means prompt encoding or other pre-sampling work is still running
- `N/N` can persist during VAE decode, frame conversion, and HTTP result assembly
- this is expected today because the native progress API intentionally tracks sampling steps only

This keeps `server_v2` intentionally small while still exercising the native
capability queries, progress polling, and cooperative cancel APIs.

`ImageRequest` currently supports:

- prompt / negative prompt
- width / height / steps / seed / batch_count
- `init_image` / `mask_image` / `control_image` / `ref_images`
- guidance / cfg_scale / image_cfg_scale / eta / flow_shift
- sampler / scheduler
- full cache parameter set from `ed_sample_params_t`
- `output_type="pil"` or `output_type="numpy"`

`numpy` output example:

```python
from edge_dit import Engine, ImageRequest

with Engine(model_path="/path/to/FLUX.1-dev", backend="cuda") as engine:
    images = engine.generate_image(
        ImageRequest(
            prompt="a glass teapot on a wooden table",
            width=256,
            height=256,
            steps=1,
            seed=42,
            output_type="numpy",
        )
    )
    print(images[0].shape)
```

Minimal video usage:

```python
from edge_dit import Engine, VideoRequest

with Engine(
    model_path="/path/to/Wan2.1-T2V-1.3B-Diffusers",
    backend="cuda",
    offload_params_to_cpu=True,
    keep_text_encoder_on_cpu=True,
    keep_vae_on_cpu=True,
    max_vram_gb=8.0,
) as engine:
    frames = engine.generate_video(
        VideoRequest(
            prompt="a small robot walking through a rainy neon street",
            width=416,
            height=240,
            frames=9,
            steps=1,
            cfg_scale=5.0,
            flow_shift=5.0,
            seed=42,
        )
    )
    frames[0].save("/tmp/wan_first_frame.png")
```

## Build a shared library

The Python binding needs a shared `edgedit` library. Official CUDA performance
builds use the `performance` profile; keep that profile for benchmarkable
results and only use `minimal` for quick CI-style validation.

```bash
ED_BUILD_PROFILE=performance \
ED_BUILD_SHARED_LIBS=ON \
BUILD_DIR=build-cuda-shared \
CUDNN_ROOT=/path/to/nvidia/cudnn \
bash scripts/build_cuda.sh
```

Or use the checked-in helper:

```bash
CUDNN_ROOT=/path/to/nvidia/cudnn \
bindings/python/scripts/build_shared_cuda.sh
```

This produces:

- `build-cuda-shared/bin/libedgedit.so`
- sibling `libggml*.so` runtime dependencies in the same directory

## Development usage

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_LIBRARY=/absolute/path/to/libedgedit.so \
python bindings/python/examples/basic_txt2img.py \
  --model /path/to/model \
  --backend cuda \
  --prompt "a glass teapot on a wooden table" \
  --offload-to-cpu \
  --keep-text-encoder-on-cpu \
  --max-vram 8 \
  --output output.png
```

If your CUDA or cuDNN libraries live outside standard search paths, you can also provide:

```bash
EDGE_DIT_DEPENDENCY_DIRS=/path/to/dir1:/path/to/dir2
```

The loader will also try to preload:

- sibling `libggml*.so` libraries next to `libedgedit.so`
- common `nvidia/*/lib` Python package directories under `~/.local` or `CONDA_PREFIX`
- `CUDNN_ROOT/lib`, `CUDNN_ROOT/lib64`, and common CUDA library directories

The example script also exposes the full cache tuning surface from `ed_sample_params_t`,
including cache thresholds, warmup limits, TaylorSeer knobs, and SCM policy flags.

## Config-driven example

Besides `examples/basic_txt2img.py`, there is also a JSON-driven example:

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so \
EDGE_DIT_MODEL_PATH=/path/to/FLUX.1-dev \
/usr/bin/python3 bindings/python/examples/configured_txt2img.py \
  --config bindings/python/examples/flux_smoke_config.json \
  --output /tmp/edge_dit_python_configured.png
```

The sample config lives at:

- `bindings/python/examples/flux_smoke_config.json`

That file is useful as a stable base for team-shared smoke tests.

There is also a matching video example:

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so \
EDGE_DIT_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers \
/usr/bin/python3 bindings/python/examples/configured_txt2vid.py \
  --config bindings/python/examples/wan_t2v_smoke_config.json \
  --output /tmp/edge_dit_python_wan_smoke.gif
```

## Scripted smoke test

For the current verified FLUX.1-dev setup:

```bash
export EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so
export EDGE_DIT_MODEL_PATH=/path/to/FLUX.1-dev
bindings/python/scripts/run_python_smoke_test.sh
```

For Wan video smoke tests:

```bash
export EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so
export EDGE_DIT_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
bindings/python/scripts/run_python_video_smoke_test.sh
```

For `server_v2` smoke tests through the HTTP job API:

```bash
export EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so
export EDGE_DIT_MODEL_PATH=/path/to/FLUX.1-dev
bindings/python/scripts/run_python_server_v2_smoke_test.sh
```

For Wan video through `server_v2`:

```bash
export EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so
export EDGE_DIT_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
export EDGE_DIT_SERVER_KIND=video
bindings/python/scripts/run_python_server_v2_smoke_test.sh
```

## Conda note

If `python` comes from Conda, you may still hit a native ABI mismatch such as:

```text
libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

That is not a binding logic bug. It means the current Python process picked up an older C++ runtime than the one used to build `libedgedit.so`.

In that case, use one of these:

- run the smoke test with `/usr/bin/python3`
- switch to a Conda environment with a newer `libstdc++`
- or manually provide a compatible runtime on your library search path

Machine-specific validation notes should live in `dev_docs/`, not in this package README.

## Real smoke test

For `FLUX.1-dev`, a conservative smoke test that worked in this environment used:

- `backend="cuda"`
- `offload_params_to_cpu=True`
- `keep_text_encoder_on_cpu=True`
- `max_vram_gb=8.0`
- `width=256`
- `height=256`
- `steps=1`

Example:

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so \
/usr/bin/python3 bindings/python/examples/basic_txt2img.py \
  --model /path/to/FLUX.1-dev \
  --backend cuda \
  --prompt "smoke test teapot" \
  --width 256 \
  --height 256 \
  --steps 1 \
  --seed 42 \
  --offload-to-cpu \
  --keep-text-encoder-on-cpu \
  --max-vram 8 \
  --output /tmp/edge_dit_python_smoke.png
```

This exact CLI path was validated end to end against `/path/to/FLUX.1-dev` in this
workspace.

Additional validation completed in this workspace on `2026-07-06`:

- `stable-diffusion-3-medium-diffusers`
  - worked with `backend="cuda"`, `offload_params_to_cpu=True`,
    `keep_text_encoder_on_cpu=True`, `skip_t5=True`, `max_vram_gb=8.0`,
    `width=256`, `height=256`, `steps=1`
  - sample config: `bindings/python/examples/sd3_smoke_config.json`
- `Qwen-Image`
  - default bf16 single-GPU loading on a 24 GiB RTX 3090 failed because the runtime diffusion
    weights allocation requested about `38.98 GiB`
  - worked after setting `weight_type="q4_k"` plus `offload_params_to_cpu=True`,
    `keep_text_encoder_on_cpu=True`, `keep_vae_on_cpu=True`, `max_vram_gb=8.0`,
    `width=512`, `height=512`, `steps=1`
  - `weight_type` also accepts `f32`, `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`,
    `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, and `auto`
  - sample config: `bindings/python/examples/qwen_image_smoke_config.json`
- `Wan2.1-T2V-1.3B`
  - worked with `backend="cuda"`, `offload_params_to_cpu=True`,
    `keep_text_encoder_on_cpu=True`, `keep_vae_on_cpu=True`, `max_vram_gb=8.0`,
    `width=416`, `height=240`, `frames=9`, `steps=1`, `cfg_scale=5.0`, `flow_shift=5.0`
  - validated against `/path/to/Wan2.1-T2V-1.3B-Diffusers`
  - note that the raw non-diffusers release directory with top-level `.pth` files still did not
    initialize directly in this workspace; the verified path is the diffusers export
  - sample config: `bindings/python/examples/wan_t2v_smoke_config.json`

Additional `server_v2` validation completed in this workspace on `2026-07-06`:

- `FLUX.1-dev`
  - verified end to end through the HTTP job API with
    `bindings/python/scripts/run_python_server_v2_smoke_test.sh`
  - output file written successfully to `/tmp/edge_dit_python_server_v2_smoke.png`
- `Wan2.1-T2V-1.3B-Diffusers`
  - verified end to end through the HTTP job API with
    `EDGE_DIT_SERVER_KIND=video bindings/python/scripts/run_python_server_v2_smoke_test.sh`
  - output file written successfully to `/tmp/edge_dit_python_server_v2_wan_smoke.gif`
  - observed behavior matched current documented semantics exactly:
    `progress.current_step/total_steps` stayed at `0/1` during prompt encoding,
    moved to `1/1` during sampling/post-sampling work, and did not represent VAE decode or output
    encoding progress

Additional `server_v2` validation completed in this workspace on `2026-07-07`:

- `Qwen-Image-Edit`
  - verified end to end through the real HTTP job API with
    `EDGE_DIT_RUN_SERVER_V2_QWEN_IMAGE_EDIT=1 /usr/bin/python3 -m unittest ... test_generate_qwen_image_edit_through_real_server_v2_when_enabled`
  - validated against `/path/to/Qwen-Image-Edit`
  - the test generated a synthetic input image locally, submitted it as `init_image_b64`, and wrote the returned
    edited PNG to `/tmp/edge_dit_server_v2_qwen_image_edit_integration.png`
  - one observed run completed successfully in about `381.9s` end to end on this workspace
- `FLUX.1-Kontext-dev`
  - verified end to end through the real HTTP job API with
    `EDGE_DIT_RUN_SERVER_V2_FLUX_KONTEXT=1 /usr/bin/python3 -m unittest ... test_generate_flux_kontext_through_real_server_v2_when_enabled`
  - validated against `/path/to/FLUX.1-Kontext-dev`
  - the test generated a synthetic reference image locally, submitted it as `ref_images_b64`, and wrote the returned
    PNG to `/tmp/edge_dit_server_v2_flux_kontext_integration.png`
  - one observed run completed successfully in about `78.4s` end to end on this workspace
- full sequential `server_v2` matrix
  - verified end to end through
    `EDGE_DIT_RUN_SERVER_V2_MATRIX=1 /usr/bin/python3 -m unittest ... test_run_full_server_v2_model_matrix_when_enabled`
  - covered this exact sequence in one process: `FLUX.1-dev` -> `stable-diffusion-3-medium-diffusers` -> `Qwen-Image` -> `Qwen-Image-Edit` -> `FLUX.1-Kontext-dev` -> `Wan2.1-T2V-1.3B-Diffusers`
  - for each model the test started a fresh in-process `server_v2`, created one terminal-success job plus one queued-then-cancelled job, exercised `health`, `capabilities`, `jobs`, `result`, active-job delete rejection, `cleanup`, TTL expiry cleanup, then shut the service down before starting the next model
  - one observed run completed successfully in about `1952.2s` end to end on this workspace
  - wrote returned artifacts under `/tmp/edge_dit_server_v2_matrix/`

## Optional integration test

There is also an optional real-library unit-test entrypoint:

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_RUN_INTEGRATION=1 \
EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so \
EDGE_DIT_MODEL_PATH=/path/to/FLUX.1-dev \
/usr/bin/python3 -m unittest discover -s bindings/python/tests -p 'test_optional_real_smoke.py' -v
```

This test is still discovered by default, but it only executes when `EDGE_DIT_RUN_INTEGRATION=1`.

There is also an optional real `server_v2` integration-test entrypoint:

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_RUN_INTEGRATION=1 \
EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so \
EDGE_DIT_MODEL_PATH=/path/to/FLUX.1-dev \
/usr/bin/python3 -m unittest discover -s bindings/python/tests -p 'test_optional_real_server_v2_smoke.py' -v
```

Optional real video coverage through the same entrypoint can be enabled with:

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_RUN_INTEGRATION=1 \
EDGE_DIT_RUN_SERVER_V2_VIDEO=1 \
EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so \
EDGE_DIT_MODEL_PATH=/path/to/FLUX.1-dev \
EDGE_DIT_VIDEO_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers \
/usr/bin/python3 -m unittest discover -s bindings/python/tests -p 'test_optional_real_server_v2_smoke.py' -v
```

Optional real Qwen image-edit coverage through the same entrypoint can be enabled with:

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_RUN_INTEGRATION=1 \
EDGE_DIT_RUN_SERVER_V2_QWEN_IMAGE_EDIT=1 \
EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so \
EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH=/path/to/Qwen-Image-Edit \
/usr/bin/python3 -m unittest discover -s bindings/python/tests -p 'test_optional_real_server_v2_smoke.py' -v
```

That Qwen image-edit path generates a small synthetic input image inside the test
process, submits it as `init_image_b64` over HTTP, polls the job resource, and
writes the returned edited PNG artifact to `/tmp/edge_dit_server_v2_qwen_image_edit_integration.png`
unless overridden with `EDGE_DIT_SERVER_V2_QWEN_IMAGE_EDIT_OUTPUT`.

Optional real FLUX Kontext coverage through the same entrypoint can be enabled with:

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_RUN_INTEGRATION=1 \
EDGE_DIT_RUN_SERVER_V2_FLUX_KONTEXT=1 \
EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so \
EDGE_DIT_FLUX_KONTEXT_MODEL_PATH=/path/to/FLUX.1-Kontext-dev \
/usr/bin/python3 -m unittest discover -s bindings/python/tests -p 'test_optional_real_server_v2_smoke.py' -v
```

That FLUX Kontext path generates a small synthetic reference image inside the
test process, submits it as `ref_images_b64`, polls the job resource, and writes
the returned PNG artifact to `/tmp/edge_dit_server_v2_flux_kontext_integration.png`
unless overridden with `EDGE_DIT_SERVER_V2_FLUX_KONTEXT_OUTPUT`.

Optional full sequential `server_v2` matrix coverage across the currently
validated local models can be enabled with:

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_RUN_INTEGRATION=1 \
EDGE_DIT_RUN_SERVER_V2_MATRIX=1 \
EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so \
/usr/bin/python3 -m unittest \
  bindings.python.tests.test_optional_real_server_v2_smoke.OptionalRealServerV2SmokeTests.test_run_full_server_v2_model_matrix_when_enabled \
  -v
```

By default that matrix test uses these local model-path env vars when present,
otherwise it falls back to the validated workspace paths:

- `EDGE_DIT_FLUX_MODEL_PATH`
- `EDGE_DIT_SD3_MODEL_PATH`
- `EDGE_DIT_QWEN_IMAGE_MODEL_PATH`
- `EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH`
- `EDGE_DIT_FLUX_KONTEXT_MODEL_PATH`
- `EDGE_DIT_WAN_VIDEO_MODEL_PATH`

Useful matrix-specific knobs:

- `EDGE_DIT_SERVER_V2_MATRIX_MODELS=flux-dev,qwen-image-edit,wan-t2v`
  runs only a comma-separated subset of scenario slugs
- `EDGE_DIT_SERVER_V2_MATRIX_OUTPUT_DIR=/tmp/edge_dit_server_v2_matrix`
  overrides the artifact directory for the per-scenario PNG/GIF outputs
- `EDGE_DIT_SERVER_V2_MATRIX_TIMEOUT_SECONDS=1800`
  overrides the per-scenario polling timeout globally

That matrix path validates:

- real `Engine` construction for every model in the sequence
- per-model in-process `server_v2` startup and shutdown
- live `health` and `capabilities` responses during each scenario
- one successful job plus one queued cancellation per model
- job listing, not-ready result handling, active delete rejection, explicit cleanup, and TTL-driven cleanup over HTTP
- final image/video artifact write-out for every scenario

That server-side integration path validates:

- real `Engine` construction with the shared library
- in-process `server_v2` startup
- HTTP job creation, polling, and result retrieval
- artifact write-out from returned base64 PNG payloads

## Implementation notes

The binding implementation lives in:

- `src/edge_dit/_capi.py`: public C API structs and function signatures
- `src/edge_dit/_lib.py`: dynamic library and dependency loading
- `src/edge_dit/config.py`: `EngineConfig` and `ImageRequest`
- `src/edge_dit/engine.py`: high-level runtime wrapper
- `src/edge_dit/image.py`: native image to `PIL.Image` or `numpy.ndarray` conversion
- `src/edge_dit/server_v2.py`: job-based non-blocking HTTP runtime on top of the Python binding

Testing currently covers:

- struct layout sanity
- enum mapping
- config validation
- engine lifecycle and native request mapping
- image conversion
- dependency preloading logic
- example argument mapping
- config-driven example mapping
- `server_v2` HTTP job lifecycle, cancellation, filtering, cleanup, and structured errors

Run unit tests:

```bash
PYTHONPATH=bindings/python/src python -m unittest discover -s bindings/python/tests -v
```
