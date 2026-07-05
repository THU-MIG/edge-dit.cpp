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
- return `PIL.Image.Image` by default
- optionally return `numpy.ndarray` with `output_type="numpy"`
- release native resources explicitly with `close()` or a context manager
- expose the full current `ed_sample_params_t` cache tuning surface
- provide copyable helper scripts for shared-library build and Python smoke test
- provide an optional real-library integration test entrypoint
- enrich common load/generation exceptions with Python-side context

Current deferred scope:

- `init_image`
- `mask_image`
- `control_image`
- `ref_images`
- `loras`
- progress callback
- cancel / abort
- async job APIs

Those deferred items are not all just Python work. In particular, input-image capabilities are
being held for a later phase because the inference engine side is not yet ready to expose them as a
stable phase-1 capability.

## Python API usage

The main Python API is:

- `Engine`
- `EngineConfig`
- `ImageRequest`

Common failure modes now include extra Python-side context, for example:

- load failures include configured model/backend hints
- generation failures include request summary such as size, steps, and output type

Minimal usage:

```python
from edge_dit import Engine

with Engine(
    model_path="/mnt/data/yangminghong/FLUX.1-dev",
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
    model_path="/mnt/data/yangminghong/FLUX.1-dev",
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

`ImageRequest` currently supports:

- prompt / negative prompt
- width / height / steps / seed / batch_count
- guidance / cfg_scale / image_cfg_scale / eta / flow_shift
- sampler / scheduler
- full cache parameter set from `ed_sample_params_t`
- `output_type="pil"` or `output_type="numpy"`

`numpy` output example:

```python
from edge_dit import Engine, ImageRequest

with Engine(model_path="/mnt/data/yangminghong/FLUX.1-dev", backend="cuda") as engine:
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

## Build a shared library

The Python binding needs a shared `edgedit` library. One working CUDA build command is:

```bash
cmake -S . -B build-cuda-shared \
  -DBUILD_SHARED_LIBS=ON \
  -DED_BUILD_SHARED_LIBS=ON \
  -DED_BUILD_EXAMPLES=ON \
  -DED_GGML_CUDA=ON \
  -DED_ENABLE_CUDNN_SDPA=ON \
  -DED_ENABLE_MPI=ON \
  -DED_ENABLE_NCCL=ON \
  -DED_ENABLE_PARALLEL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCUDNN_ROOT=/home/yangminghong/.local/lib/python3.12/site-packages/nvidia/cudnn

cmake --build build-cuda-shared --target edgedit -j 8
```

Or use the checked-in helper:

```bash
CUDNN_ROOT=/home/yangminghong/.local/lib/python3.12/site-packages/nvidia/cudnn \
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
EDGE_DIT_MODEL_PATH=/mnt/data/yangminghong/FLUX.1-dev \
/usr/bin/python3 bindings/python/examples/configured_txt2img.py \
  --config bindings/python/examples/flux_smoke_config.json \
  --output /tmp/edge_dit_python_configured.png
```

The sample config lives at:

- `bindings/python/examples/flux_smoke_config.json`

That file is useful as a stable base for team-shared smoke tests.

## Scripted smoke test

For the current verified FLUX.1-dev setup:

```bash
export EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so
export EDGE_DIT_MODEL_PATH=/mnt/data/yangminghong/FLUX.1-dev
bindings/python/scripts/run_python_smoke_test.sh
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
  --model /mnt/data/yangminghong/FLUX.1-dev \
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

This exact CLI path was validated end to end against `/mnt/data/yangminghong/FLUX.1-dev` in this
workspace.

## Optional integration test

There is also an optional real-library unit-test entrypoint:

```bash
PYTHONPATH=bindings/python/src \
EDGE_DIT_RUN_INTEGRATION=1 \
EDGE_DIT_LIBRARY=$PWD/build-cuda-shared/bin/libedgedit.so \
EDGE_DIT_MODEL_PATH=/mnt/data/yangminghong/FLUX.1-dev \
/usr/bin/python3 -m unittest discover -s bindings/python/tests -p 'test_optional_real_smoke.py' -v
```

This test is still discovered by default, but it only executes when `EDGE_DIT_RUN_INTEGRATION=1`.

## Implementation notes

The binding implementation lives in:

- `src/edge_dit/_capi.py`: public C API structs and function signatures
- `src/edge_dit/_lib.py`: dynamic library and dependency loading
- `src/edge_dit/config.py`: `EngineConfig` and `ImageRequest`
- `src/edge_dit/engine.py`: high-level runtime wrapper
- `src/edge_dit/image.py`: native image to `PIL.Image` or `numpy.ndarray` conversion

Testing currently covers:

- struct layout sanity
- enum mapping
- config validation
- engine lifecycle and native request mapping
- image conversion
- dependency preloading logic
- example argument mapping
- config-driven example mapping

Run unit tests:

```bash
PYTHONPATH=bindings/python/src python -m unittest discover -s bindings/python/tests -v
```
