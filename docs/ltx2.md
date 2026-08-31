# LTX-2.3 Video And Audio

[Back to supported models](models.md)

edge-dit.cpp supports LTX-2.3 text-to-video (T2V), image-to-video (I2V),
end-frame-to-video (E2V), first/last-frame-to-video (FLF2V), and an optional x2
spatial latent upscale with a second denoising pass. Generated video can include
audio when the LTX audio VAE is loaded.

## Components

Pass the model as separate components:

```text
--diffusion-model ltx-2.3-22b-dev-UD-Q4_K_M.gguf
--vae ltx-2.3-22b-dev_video_vae.safetensors
--audio-vae ltx-2.3-22b-dev_audio_vae.safetensors
--llm gemma-3-12b-it-UD-Q4_K_XL.gguf
--embeddings-connectors ltx-2.3-22b-dev_embeddings_connectors.safetensors
```

The x2 hires mode additionally needs:

```text
--latent-upscaler ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

The stable-diffusion.cpp-compatible directory/name form is also accepted:

```text
--hires-upscalers-dir /models/LTX-2.3-GGUF/latent_upscale_models
--hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1
```

## Modes

Use the common component arguments above with one of these input forms:

```bash
# T2V
ed-cli -M vid_gen ... -p "a cat walking across a sunlit table"

# I2V
ed-cli -M vid_gen ... -p "the cat turns toward the camera" --init-img start.png

# E2V
ed-cli -M vid_gen ... -p "the camera approaches the cat" --end-img end.png

# FLF2V
ed-cli -M vid_gen ... -p "a glass flower blossoms" \
  --init-img start.png --end-img end.png
```

LTX image conditioning center-crops input images to the requested base aspect
ratio, then resizes them with the same uint8 sRGB BOX-filter path as
stable-diffusion.cpp. If the base dimensions need alignment, or during hires
refinement, that preprocessed image is remapped with nearest interpolation to
the actual VAE encoding size.

## Hires Refinement

`--hires` first generates at the 32-aligned `-W` and `-H`, runs the model-backed
x2 spatial latent upscaler, then denoises and decodes at twice the resolved base
width and height:

```bash
ed-cli -M vid_gen ... -W 640 -H 360 --video-frames 33 \
  --init-img start.png --hires \
  --hires-upscalers-dir /models/LTX-2.3-GGUF/latent_upscale_models \
  --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 \
  --hires-steps 4 --hires-denoising-strength 0.7 \
  -o hires-i2v.webm
```

An explicit finite, non-negative, non-increasing sigma schedule overrides the
derived hires schedule:

```text
--hires-sigmas "0.85,0.725,0.421875,0.0"
```

## Constraints

- Width and height are aligned upward to multiples of 32, matching
  stable-diffusion.cpp. For example, `1280x720` generates at `1280x736`.
- Frame count must satisfy `8k+1`, for example 9, 17, or 33.
- LTX-2.3 uses Euler sampling with the `ltx2` scheduler.
- I2V, E2V, and FLF2V require video VAE encoder weights.
- LTX control frames and generic reference image/video/audio inputs are not
  implemented.

## Runtime Optimization Support

The LTX-2.3 pipeline has the following runtime support matrix:

| Feature | Status | Notes |
| --- | --- | --- |
| GGUF quantization | Supported | The GGUF DiT and Gemma 3 files are loaded with their native mixed quantization. |
| `--auto-fit` / `--auto-allocate` | Supported | Placement accounts for the text projection, audio VAE/vocoder, video compute buffer, optional latent upscaler, and requested video fps. |
| Weight offload | Supported | `--offload-to-cpu`, `--dit-offload`, `--text-encoder-offload`, and `--vae-offload` keep compute on the selected GPU while staging weights as needed. |
| Flash attention | Supported | Enabled by default; use `--no-flash-attention` to disable it. |
| CFG parallelism | Supported | Requires an NCCL-enabled build. Use exactly two workers with `--devices 0,1 --cfg-parallel-size 2`; rank 0 evaluates the unconditional branch and rank 1 evaluates the conditional branch. |
| Cache acceleration | Not supported | LTX requests using a cache mode fail explicitly instead of silently ignoring the option. |
| Sequence parallelism | Not supported | `--sp-size` greater than 1 is rejected explicitly. |
| Tensor parallelism | Not supported | `--tp-size` greater than 1 is rejected explicitly. |

CFG parallelism gathers only the two final DiT predictions before applying the
same sampler update on both workers. Video and audio VAE decoding and output
writing are performed by rank 0 only.

The checked-in CFG regression can be run with a local NCCL build and two GPUs:

```bash
LTX_SINGLE_DEVICE=0 LTX_CFG_DEVICES=0,1 \
  scripts/test_ltx2_cfg_parallel.sh
```

It runs the same seeded 2-step T2V request in single-GPU and CFG-2 modes and
compares both the AVI and WAV sidecar byte-for-byte. Override
`ED_CLI`, `LTX_MODEL_ROOT`, or the individual `LTX_*` component variables when
the model files use a different layout.

The native server accepts the same component paths. Video JSON uses
`init_image_b64`, `end_image_b64`, `hires`, `hires_steps`,
`hires_denoising_strength`, and an optional numeric `hires_sigmas` array.
