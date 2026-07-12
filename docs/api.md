# API and Bindings

[← Back to README](../README.md)

edge-dit.cpp exposes a public C API, native server binaries, Python bindings,
and a Python job-style HTTP server.

## C API

The public C API is declared in:

```text
include/edge-dit.h
```

Main lifecycle:

```c
ed_context_params_t ctx_params;
ed_context_params_init(&ctx_params);
ctx_params.model_path = "/path/to/model-dir";

ed_context_t * ctx = ed_create_context(&ctx_params);
if (ctx == NULL) {
    /* handle model load error */
}

ed_free_context(ctx);
```

The context owns model/runtime state. Generated image and video buffers must be
released with the matching `ed_free_*` function.

## Version API

```c
const char * ed_version_string(void);
int ed_version_major(void);
int ed_version_minor(void);
int ed_version_patch(void);
```

For v0.1.0 these return `0.1.0`, `0`, `1`, and `0`.

## Image Generation

```c
ed_image_generation_params_t params;
ed_image_generation_params_init(&params);
params.prompt = "a glass teapot on a wooden table";
params.width = 1024;
params.height = 1024;
params.seed = 0;
params.sample.steps = 20;

ed_image_batch_t out;
ed_status_t status = ed_generate_image(ctx, &params, &out);
if (status == ED_STATUS_OK) {
    /* use out.images */
    ed_free_image_batch(&out);
}
```

The image request supports prompts, negative prompts, image inputs, masks,
control images, reference images, LoRA entries, and cache/sample settings.

## Video Generation

```c
ed_video_generation_params_t params;
ed_video_generation_params_init(&params);
params.prompt = "a glass teapot rotating on a wooden table";
params.width = 832;
params.height = 480;
params.frames = 40;
params.seed = 0;
params.sample.steps = 20;

ed_video_t out;
ed_status_t status = ed_generate_video(ctx, &params, &out);
if (status == ED_STATUS_OK) {
    /* use out.frames */
    ed_free_video(&out);
}
```

Video support is currently focused on Wan-family pipelines.

## Error Handling

Most calls return `ed_status_t`.

```c
const char * err = ed_get_last_error(ctx);
```

Useful status values include:

- `ED_STATUS_OK`
- `ED_STATUS_INVALID_ARGUMENT`
- `ED_STATUS_MODEL_LOAD_FAILED`
- `ED_STATUS_GENERATION_FAILED`
- `ED_STATUS_OUT_OF_MEMORY`
- `ED_STATUS_UNSUPPORTED`
- `ED_STATUS_CANCELLED`

## Capability Checks

```c
const char * ed_context_pipeline_name(const ed_context_t * ctx);
const char * ed_context_version_name(const ed_context_t * ctx);
bool ed_context_supports_image(const ed_context_t * ctx);
bool ed_context_supports_video(const ed_context_t * ctx);
ed_sampler_t ed_context_default_sampler(const ed_context_t * ctx);
ed_scheduler_t ed_context_default_scheduler(const ed_context_t * ctx, ed_sampler_t sampler);
int ed_context_parallel_rank(const ed_context_t * ctx);
int ed_context_parallel_world_size(const ed_context_t * ctx);
bool ed_context_parallel_is_root(const ed_context_t * ctx);
```

Cancellation and progress polling:

```c
void ed_context_request_cancel(ed_context_t * ctx);
int ed_context_progress_current_step(const ed_context_t * ctx);
int ed_context_progress_total_steps(const ed_context_t * ctx);
```

## Native HTTP Server

`ed-server` is a native HTTP wrapper around the C API.

Start:

```bash
./build-cuda/bin/ed-server \
  --backend cuda \
  --model /path/to/flux-dev \
  --host 127.0.0.1 \
  --port 8080
```

Canonical endpoints:

- `GET /ed/v1/health`
- `GET /ed/v1/models`
- `GET /ed/v1/capabilities`
- `POST /ed/v1/images/generations`

Aliases are also registered for `/edgedit/v1/...` and `/edge-dit/v1/...`.

Example:

```bash
curl -s http://127.0.0.1:8080/ed/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a glass teapot on a wooden table",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "seed": 0
  }'
```

See [examples/server/README.md](../examples/server/README.md) for the native
server details.

## Python Bindings

The Python package lives under:

```text
bindings/python/src/edge_dit/
```

Basic usage:

```python
from edge_dit import Engine, EngineConfig, ImageRequest

config = EngineConfig(model_path="/path/to/FLUX.1-dev", backend="cuda")
request = ImageRequest(
    prompt="a glass teapot on a wooden table",
    width=1024,
    height=1024,
    steps=20,
    seed=0,
)

with Engine(config) as engine:
    images = engine.generate_image(request)
    images[0].save("output.png")
```

For local test runs from the repository root:

```bash
PYTHONPATH=bindings/python/src python3 -m pytest bindings/python/tests
```

See [bindings/python/README.md](../bindings/python/README.md) for more Python
examples.

<a id="python-server-v2"></a>

## Python server_v2

The Python bindings include a job-style HTTP server:

```bash
PYTHONPATH=bindings/python/src \
python -m edge_dit.server_v2 \
  --model /path/to/model \
  --backend cuda \
  --host 127.0.0.1 \
  --port 8080
```

Endpoints include:

- `GET /ed/v2/health`
- `GET /ed/v2/capabilities`
- `POST /ed/v2/images/generations`
- `POST /ed/v2/videos/generations`
- `GET /ed/v2/jobs`
- `POST /ed/v2/jobs/cleanup`
- `GET /ed/v2/jobs/{job_id}`
- `POST /ed/v2/jobs/{job_id}/cancel`
- `GET /ed/v2/jobs/{job_id}/result`
- `DELETE /ed/v2/jobs/{job_id}`

## Frontend Console

A development console is available at:

```text
bindings/python/frontend/server_v2-console/
```

It is intended for local development and demonstration. It is not a stable API
contract. See
[RUNTIME_CONFIGURATION.md](../bindings/python/frontend/server_v2-console/RUNTIME_CONFIGURATION.md).

## API Stability

The API is public but not stable. During the v0.x series:

- C API structs and functions may change.
- ABI compatibility is not guaranteed.
- CLI flags may be renamed or reorganized.
- HTTP server response schemas may evolve.
- Python binding behavior may change as the C API settles.

## Related Documentation

- [Build and installation](build.md)
- [Command line usage](cli.md)
- [Supported models and usage](models.md)
- [Performance and optimization](performance.md)
- [Development and contributing](development.md)
