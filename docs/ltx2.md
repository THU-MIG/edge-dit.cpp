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

The native server accepts the same component paths. Video JSON uses
`init_image_b64`, `end_image_b64`, `hires`, `hires_steps`,
`hires_denoising_strength`, and an optional numeric `hires_sigmas` array.
