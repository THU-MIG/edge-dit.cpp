# edge-dit Server

`ed-server` is a small HTTP wrapper around the public edge-dit C API. It keeps one
model context loaded in-process and serializes generation calls through that
context.

## Start

```bash
./build-cuda/bin/ed-server \
  --backend cuda \
  --model /path/to/flux-dev \
  --host 127.0.0.1 \
  --port 8080 \
  -W 1024 -H 1024 \
  --steps 20 \
  --guidance 3.5
```

The Server defaults to `--auto-allocate`: source precision is preserved while
component residency is planned against live free VRAM. Add `--max-vram 20` to
set an upper planning budget. Use `--auto-fit` when the Server may also choose
TE/DiT quantization, or disable automatic placement for a fully manual policy:

```bash
# Fixed Q8 loading with automatic placement.
./build-cuda/bin/ed-server --backend cuda --model /models/Qwen-Image \
  --type q8_0 --max-vram 20

# Fixed Q8 loading with manual component offload.
./build-cuda/bin/ed-server --backend cuda --model /models/Qwen-Image \
  --no-auto-allocate --type q8_0 \
  --text-encoder-offload --vae-offload --max-vram 20

# Automatic TE/DiT quantization and placement.
./build-cuda/bin/ed-server --backend cuda --model /models/FLUX.1-dev \
  --auto-fit --max-vram 20
```

Canonical endpoints use the `ed` prefix, matching `ed-cli` and the `ed_*` C API:

- `GET /ed/v1/health`
- `GET /ed/v1/models`
- `GET /ed/v1/capabilities`
- `POST /ed/v1/images/generations`
- `POST /ed/v1/videos/generations`

Aliases are also registered for `/edgedit/v1/...` and `/edge-dit/v1/...`.

## Generate An Image

```bash
curl -s http://127.0.0.1:8080/ed/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a cinematic photo of a glass teapot on a wooden table, soft morning light",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "seed": 0,
    "distilled_guidance": 3.5,
    "cfg_scale": 1.0,
    "cache": {
      "mode": "dbcache",
      "Fn_compute_blocks": 8,
      "Bn_compute_blocks": 0,
      "residual_diff_threshold": 0.08,
      "max_warmup_steps": 8
    }
  }' | jq -r '.data[0].b64_png' | base64 -d > output.png
```

The response includes `elapsed_ms`, resolved generation parameters, and one
base64-encoded PNG per generated image.

## MiniMax-H3 Video And Audio

```bash
./build-cuda/bin/ed-server \
  --backend cuda --diffusion-model /models/minimax-h3/dit.gguf \
  --vae /models/minimax-h3/vae.safetensors \
  --audio-vae /models/minimax-h3/audio-vae.safetensors \
  --llm /models/minimax-h3/qwen.gguf \
  --minimax-h3-stage-lifecycle --frames 90 --fps 24 --steps 20

curl -s http://127.0.0.1:8080/ed/v1/videos/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a cinematic ocean sunrise","width":768,"height":1344,"frames":90,"steps":20,"sampler":"res-multistep","scheduler":"simple"}' \
  > result.json
```

The result contains `frames[].b64_png`. When an audio VAE is loaded it also
contains `audio.b64_f32le`, interleaved float32 little-endian samples, plus
`sample_rate`, `channels`, and `sample_count`. Muxing/container encoding remains
the HTTP client's responsibility.

MiniMax-H3 requests must use at least 22 frames and satisfy `17k+5` (for
example 22, 39, 56, 73, or 90).

FL2VA request inputs use `init_image_b64` and `end_image_b64`. Ref2VA accepts
`ref_images_b64`, `ref_image_size` (`max` or `match`), `ref_videos` entries with
`frames_b64`, `fps`, and optional `audio`, plus standalone `ref_audios`.
Reference audio objects use the same `b64_f32le`, `sample_rate`, and `channels`
shape as generated audio.
