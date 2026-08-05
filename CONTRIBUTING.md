# Contributing to edge-dit.cpp

Thanks for your interest in improving edge-dit.cpp. The project is currently
an alpha release (v0.1.0), so APIs and performance-sensitive implementation details may
still change.

## Development Setup

Start from a fresh checkout with submodules:

```bash
git submodule update --init --recursive
```

For a portable baseline build:

```bash
bash ./scripts/build_cpu.sh
./build-cpu/bin/ed-cli --help
```

For CUDA development:

```bash
bash ./scripts/build_cuda.sh
./build-cuda/bin/ed-cli --help
```

Optional CUDA features can be enabled explicitly:

```bash
ED_ENABLE_NCCL=ON ED_ENABLE_MPI=ON bash ./scripts/build_cuda.sh
ED_INSTALL_CUDNN=ON bash ./scripts/build_cuda.sh
```

## Validation

CPU build:

```bash
bash ./scripts/build_cpu.sh
./build-cpu/bin/ed-cli --help
```

Python validation:

```bash
cd bindings/python
python -m pip install -e '.[dev]'
pytest
```

Real model integration tests are optional and require explicit environment
variables such as `EDGE_DIT_RUN_INTEGRATION=1`, `EDGE_DIT_LIBRARY`, and a model
path. They should not run by default in ordinary development or CI.

## Pull Requests

Please keep changes focused. For performance work, include:

- model family and command line used;
- hardware, CUDA/cuDNN/NCCL versions when relevant;
- before/after latency or memory numbers;
- correctness or image-quality validation, when applicable.

For public-facing behavior changes, update README or docs in the same PR.

## Code Style

- Prefer existing project patterns over new abstractions.
- Keep model-specific optimizations isolated unless they are clearly reusable.
- Avoid committing generated outputs, local benchmark images, videos, logs, or
  machine-specific paths.
- Do not include model weights or private datasets in the repository.

## Licensing

By contributing, you agree that your contributions are licensed under the
Apache License, Version 2.0, unless explicitly stated otherwise.
