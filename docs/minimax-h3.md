# MiniMax-H3

edge-dit.cpp supports MiniMax-H3 video-and-audio generation through standalone
components: one FL2VA or Ref2VA diffusion model, one Qwen3-VL text/vision
encoder, the video VAE, and the optional audio VAE. CUDA is the validated
backend for this model family.

## Checkpoints and inputs

FL2VA and Ref2VA share Qwen3-VL and both VAEs, but require different diffusion
checkpoints.

| Checkpoint | Supported conditioning |
|---|---|
| FL2VA | Text; first frame; last frame; first and last frames |
| Ref2VA | Repeatable reference images, videos, paired video audio, and additional audio |

FL2VA uses `--image`/`--init-img` for the first frame and `--end-img` for the
last frame. Ref2VA uses the following options:

| Input | CLI option | Behavior |
|---|---|---|
| Image | `--ref-image <path>` | Repeatable; presented as `<Picture N>` |
| Image sizing | `--ref-image-size match|max` | `match` only downsizes to the output pixel area; `max` resizes to a 2048px short edge, including upscaling, to match official Diffusers geometry (default) |
| Video | `--ref-video <path>` | Repeatable frame directory or `mp4`/`mov`/`mkv`/`webm`/`avi`; media files require `ffmpeg` |
| Paired audio | `--ref-video-audio <wav>` | The Nth WAV is paired with the Nth video and overrides embedded audio |
| Additional audio | `--ref-audio <wav>` | Repeatable; requires at least one image or video reference |

Reference-image resize policy changes both conditioning cost and every Ref2VA
Transformer step because the encoded reference latents are appended to the DiT
sequence:

| Mode | Reference-image geometry |
|---|---|
| `max` (default) | Scales the short edge to 2048px, including upscaling |
| `match` | Never upscales; caps the image at the output pixel area |

The selected mode and post-resize reference dimensions affect both conditioning
cost and DiT sequence length. Use `match` when smaller reference images should
not be enlarged.

When `--ref-video` points to a media file, the CLI decodes it at 24 fps and
automatically extracts an embedded audio track. Explicit paired WAV files map
positionally to videos. Additional audio is numbered after paired or embedded
video audio. Ref2VA references cannot be combined with FL2VA first/last-frame
options, and audio-only Ref2VA requests are rejected.

## Model files

### Downloadable weights

The complete, unpruned checkpoints are recommended for the best quality.
edge-dit.cpp also supports pruned DiT weights in BF16 safetensors format. Both
full and pruned BF16 DiTs can be converted to Q8_0 GGUF with `ed-convert`, and
the resulting Q8_0 DiTs can be loaded directly. Performance and quality results
from pruned and full DiTs are not directly comparable.

| Precision | Component | File | Repository |
|---|---|---|---|
| Q4_K_M | FL2VA DiT | `minimax_h3_fl2va-Q4_K_M.gguf` | [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| Q4_K_M | Ref2VA DiT | `minimax_h3_ref2va-Q4_K_M.gguf` | [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| Q4_K_M | Qwen3-VL | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| BF16 | FL2VA DiT | `diffusion_models/minimax_h3_fl2va_bf16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| BF16 | Ref2VA DiT | `diffusion_models/minimax_h3_ref2va_bf16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| BF16, pruned | FL2VA DiT | `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | [`Comfy-Org/MiniMax-H3 diffusion models`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| BF16, pruned | Ref2VA DiT | `diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors` | [`Comfy-Org/MiniMax-H3 diffusion models`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| BF16 | Qwen3-VL | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| FP16 | Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| FP32 | Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |

BF16 and FP16 both use 16 bits per weight, so converting the FP16 video VAE to
BF16 would not reduce its weight memory. With `--type preserve`, `--auto-fit`
therefore preserves the supplied VAE precision and focuses automatic
quantization on Qwen3-VL and the DiT. An explicit `--type` still applies to
eligible VAE tensors.

The official [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
Diffusers shard indexes are also accepted for BF16 transformer loading. Merged
Comfy-Org files are usually more convenient for standalone component commands.

Example downloads with the Hugging Face CLI:

```bash
hf download leejet/MiniMax-H3-GGUF \
  minimax_h3_fl2va-Q4_K_M.gguf minimax_h3_ref2va-Q4_K_M.gguf \
  qwen3vl_32b_minimax_h3-Q4_K_M.gguf --local-dir models/minimax-h3-q4

hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir models/minimax-h3
```

### Persistent Q8_0 GGUF

Create persistent Q8_0 GGUF files from the BF16 DiTs and Qwen3-VL. The INT8
ConvRot safetensors use a different representation and are not interchangeable
with GGUF Q8_0. A pruned BF16 DiT can be converted with the same command when
lower storage and memory usage are preferred. Convert once with `ed-convert`
instead of quantizing during every model load. Qwen3-VL uses the `--llm`
component input:

```bash
ed-convert --diffusion-model models/minimax-h3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --type q8_0 --output models/minimax-h3-q8/minimax_h3_fl2va-Q8_0.gguf
ed-convert --diffusion-model models/minimax-h3/diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  --type q8_0 --output models/minimax-h3-q8/minimax_h3_ref2va-Q8_0.gguf
ed-convert --llm models/minimax-h3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --type q8_0 \
  --output models/minimax-h3-q8/qwen3vl_32b_minimax_h3-Q8_0.gguf
```

The same converter accepts an official transformer
`model.safetensors.index.json`; the resulting persistent GGUF is equivalent at
the selected quantization type and avoids repeated online conversion.

Alternatively, merge all four MiniMax components into one GGUF. Each flag names
its component; using several component flags selects merge mode:

```bash
ed-convert \
  --diffusion-model models/minimax-h3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --llm models/minimax-h3-q4/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae models/minimax-h3/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae models/minimax-h3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --type q4_k --output minimax_h3_fl2va-all-Q4_K.gguf

ed-cli --model minimax_h3_fl2va-all-Q4_K.gguf --video \
  --prompt "A cinematic sunset over layered mountain ridges" \
  -W 864 -H 480 --frames 22 --steps 20 --cfg-scale 1 \
  --sampler res_multistep --scheduler simple --output minimax-h3.avi
```

The combined file contains the DiT, Qwen3-VL, video VAE, and audio VAE and is
loaded with `--model`. The component-only output contains just the DiT and is
loaded with `--diffusion-model` alongside separate component flags.

## Duration and frame count

MiniMax-H3 always generates at 24 fps. Its frame count must be at least 22 and
satisfy `17k + 5`, for example `22`, `39`, `56`, `73`, `90`, `107`, or `124`.

Use `--video-duration <seconds>` for the convenient interface. The CLI converts
the requested duration at 24 fps and selects the nearest legal frame count. For
example, `--video-duration 5` resolves to 124 frames, or approximately 5.17
seconds. The resolved value is printed before generation. Use
`--video-frames <count>` when an exact legal frame count is required. The two
options are mutually exclusive.

## Usage

Set component paths for the desired precision. The video and audio VAE files are
shared by every precision and checkpoint.

```bash
# Q4_K_M example. Replace the DiT with the Ref2VA file for reference workflows.
DIT=models/minimax-h3-q4/minimax_h3_fl2va-Q4_K_M.gguf
LLM=models/minimax-h3-q4/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
VIDEO_VAE=models/minimax-h3/vae/minimax_h3_video_vae_fp16.safetensors
AUDIO_VAE=models/minimax-h3/vae/minimax_h3_audio_vae_fp32.safetensors
```

For BF16, point `DIT` and `LLM` to the downloaded BF16 safetensors. For Q8_0,
point them to the converted GGUF files.

### FL2VA

```bash
# Text to video and audio
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --video-duration 5 -W 864 -H 480 --steps 20 --cfg-scale 1 \
  --sampler res_multistep --scheduler simple \
  --prompt "A cinematic sunset over layered mountain ridges with quiet natural ambience." \
  --video-format mp4 --output t2va.mp4

# First frame
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" --image first.png \
  --video-duration 5 -W 864 -H 480 --steps 20 --cfg-scale 1 \
  --sampler res_multistep --scheduler simple \
  --prompt "Starting from <Picture 1>, preserve the scene and add subtle natural motion." \
  --video-format mp4 --output i2va.mp4

# Last frame: use --end-img last.png
# First and last frames: use --image first.png --end-img last.png
```

### Ref2VA

Use the Ref2VA DiT for all commands in this section.

For the recommended MiniMax-H3 sampling path, use `res_multistep` with the
`simple` sigma schedule. The default remains `euler` with `discrete` for
compatibility with existing edge-dit.cpp commands.

```bash
# Image reference
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --ref-image reference.png --video-duration 5 -W 864 -H 480 --steps 20 \
  --cfg-scale 1 --sampler res_multistep --scheduler simple \
  --prompt "Use <Picture 1> as the strict visual reference." \
  --video-format mp4 --output ref-image.mp4

# MP4 reference; embedded audio is paired automatically when present
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --ref-video reference.mp4 --video-duration 5 -W 864 -H 480 --steps 20 \
  --cfg-scale 1 --sampler res_multistep --scheduler simple \
  --prompt "Preserve <Video 1> and its <Audio 1> throughout the result." \
  --video-format mp4 --output ref-video.mp4

# Frame directory with explicit paired audio
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --ref-video reference-frames --ref-video-audio soundtrack.wav \
  --video-duration 5 -W 864 -H 480 --steps 20 --cfg-scale 1 \
  --sampler res_multistep --scheduler simple \
  --prompt "Preserve <Video 1> and synchronize it with <Audio 1>." \
  --video-format mp4 --output ref-video-audio.mp4

# Mixed image, video, and additional audio
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --ref-image reference.png --ref-video reference.mp4 --ref-audio music.wav \
  --video-duration 5 -W 864 -H 480 --steps 20 --cfg-scale 1 \
  --sampler res_multistep --scheduler simple \
  --prompt "Use <Video 1>, <Picture 1>, and every supplied audio reference in a natural transition." \
  --video-format mp4 --output ref-mixed.mp4
```

## Memory placement

The explicit component controls are `--dit-offload`,
`--text-encoder-offload`, and `--vae-offload`. For automatic placement, use
`--auto-fit --max-vram <GB>`. MiniMax-H3 independently places the DiT,
Qwen3-VL, video VAE, and audio VAE while retaining CUDA compute.

```bash
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --auto-fit --max-vram 40 --video-duration 5 -W 864 -H 480 --steps 20 \
  --cfg-scale 1 --sampler res_multistep --scheduler simple \
  --prompt "A cinematic sunset over layered mountain ridges." \
  --video-format mp4 --output auto-fit.mp4
```

For long Ref2VA jobs on GPUs that can hold one major component at a time,
`--auto-fit --max-vram` automatically uses the MiniMax-H3 staged lifecycle.
Resident components are staged serially: Qwen is loaded only for positive and
negative context encoding, then released before the video VAE encodes visual
references; the video VAE is then released before the audio VAE encodes paired
or standalone reference audio. DiT weights are staged for denoising and released
before the video and audio VAEs are loaded again for final decode. Components
that cannot fit as a whole remain layer/graph-segment offloaded instead. The
explicit `--minimax-h3-stage-lifecycle` option enables the same phase behavior
without `--auto-fit`.

On single-device CUDA, the combined `--auto-fit --max-vram 24` mode installs a guarded
allocation ceiling below 24 GiB and reserves 1 GiB for CUDA/cuDNN workspaces.
Automatic quantization, phase lifecycle, component placement, and graph
segmentation are attempted first. If the workload's minimum graph segment still
cannot fit, generation fails before crossing the requested 24 GiB ceiling.
`--max-vram` without `--auto-fit` remains a planning input rather than this hard
allocation guard.

MiniMax-H3 always uses its fixed `16x16` video-VAE tiling path. Generic
`--vae-tiling` and `--vae-tile-size` values do not replace this model-specific
layout.

## RTX 4090 offloaded comparison

Measured on one RTX 4090 (24 GB), `864x480`, 56 frames at 24 fps, 20 steps,
CFG 1, seed `42`, every component offloaded to CPU (`--offload-to-cpu`),
`--max-vram 22`, model-specific `16x16` VAE tiling. On a single 24 GB card
neither the BF16 DiT nor the Qwen3-VL encoder fits resident, so both engines
stream weights per segment. Q8_0 rows quantize the BF16 files on load
(`--type q8_0`). E2E is the end-to-end process time (measured identically for
both engines); TE / Diffusion / VAE are per-stage times. "Load" is not shown as
a column because the engines load differently — edge-dit loads all components up
front, sd.cpp loads each lazily on first use — so a single load figure would not
be comparable. Peak VRAM is sampled over the whole process. Bold marks the
better value in each edge-dit vs stable-diffusion.cpp pair.

| Task | Precision | System | E2E (s) | TE (ms) | Diffusion (ms) | VAE (ms) | Peak VRAM (MiB) |
|---|---|---|---:|---:|---:|---:|---:|
| Text | q4_K_M | edge-dit.cpp | 117 | **1551** | **81466** | **4929** | **18630** |
| | | stable-diffusion.cpp | **113** | 10370 | 82550 | 13450 | 20554 |
| First frame | q4_K_M | edge-dit.cpp | 129 | **3099** | **91589** | **4804** | **18544** |
| | | stable-diffusion.cpp | **123** | 11500 | 91670 | 11170 | 20632 |
| First + last frames | q4_K_M | edge-dit.cpp | 140 | **4065** | **101269** | **4812** | **18258** |
| | | stable-diffusion.cpp | **135** | 11910 | 101350 | 12590 | 20868 |
| Text | q8_0 | edge-dit.cpp | **159** | **2154** | **83656** | **4755** | **18808** |
| | | stable-diffusion.cpp | 185 | 32750 | 132220 | 12810 | 22472 |
| First frame | q8_0 | edge-dit.cpp | **170** | **3366** | **93107** | **4514** | **19094** |
| | | stable-diffusion.cpp | 195 | 33730 | 138440 | 12980 | 22554 |
| Text | bf16 | edge-dit.cpp | 206 | 4060 | 134334 | 4962 | 21120 |
| | | stable-diffusion.cpp | — | — | — | — | — |
| First frame | bf16 | edge-dit.cpp | 218 | 5544 | 145351 | 4850 | 20986 |
| | | stable-diffusion.cpp | — | — | — | — | — |

At q4_K_M the two engines match on diffusion time; edge-dit encodes text
**3–8x faster**, decodes the VAE **~2.5x faster**, and holds peak VRAM roughly
**2 GB lower**. At q8_0 edge-dit leads across the board — **~25–30s lower E2E
time**, diffusion **~40% faster**, peak VRAM **~3.5 GB lower**.

sd.cpp has no BF16 rows on this card: it stages the full-precision DiT weights
to the GPU in one block (~38 GB), which overflows the 24 GB budget and aborts
before sampling. edge-dit streams the BF16 weights per segment, so it runs
within the 24 GB limit.

### RTX 4090 short T2V three-runtime comparison

This FL2VA text-to-video test runs on physical GPU 7 with prompt
`a snowy mountain landscape with pine trees, cinematic wide shot`, `864x480`,
56 frames, 24 fps, 20 steps, CFG 1, seed 42, and
`res_multistep`/`simple`. The 56-frame request is at least 22 frames and
satisfies `17k+5` (`k=3`). Each system uses a fresh process. E2E starts at
process launch and includes model loading, conditioning, sampling, video/audio
VAE decode, and writing the final video.

The runtimes use their practical supported quantized representations rather
than byte-identical weight formats:

| Runtime | FL2VA DiT | Qwen3-VL | Video / audio VAE |
|---|---|---|---|
| edge-dit.cpp | Q8_0 GGUF | Q4_K_M GGUF | FP16 / FP32 |
| stable-diffusion.cpp | Q8_0 GGUF | Q4_K_M GGUF | FP16 / FP32 |
| ComfyUI | INT8 ConvRot safetensors | NVFP4-AWQ safetensors | FP16 / FP32 |

- [Three-runtime T2V comparison](assets/minimax-h3-t2v-4090-edge-sdcpp-comfyui-demo.mp4)
- [Machine-readable benchmark metrics](assets/minimax-h3-t2v-4090-metrics.json)


| Runtime | E2E | Text conditioning | DiT / sampling | Video VAE decode | Audio VAE decode | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| edge-dit.cpp | 290.313s | 1.343s | **91.072s** | **4.427s** | **0.245s** | **17,030 MiB (16.63 GiB)** |
| stable-diffusion.cpp | 313.475s | 88.400s | 196.950s | 19.360s | 1.140s | 22,374 MiB (21.85 GiB) |
| ComfyUI | **287.952s** | 81.945s | 138.958s | 27.576s | 3.059s | 23,840 MiB (23.28 GiB) |

ComfyUI has the lowest E2E, 2.361 seconds (0.8%) below edge-dit.cpp.
edge-dit.cpp uses 6,810 MiB less peak VRAM than ComfyUI and has the shortest
reported conditioning, sampling, and VAE stages. stable-diffusion.cpp has the
highest E2E and sits between the other two in peak VRAM.

Stage boundaries follow each runtime's own timing markers. ComfyUI and
stable-diffusion.cpp defer some component loading into conditioning, sampling,
and decode, whereas edge-dit.cpp reports most loading before generation.

## H200 BF16 comparison

The tables below use one H200, resident components, `864x480`, 124 frames at
24 fps, 20 steps, CFG 1, and seed `424242`. edge-dit.cpp uses the complete
Comfy-Org BF16 DiT/Qwen files, FP16 video VAE, and FP32 audio VAE. Diffusers uses the
official complete BF16 DiT shards, BF16 Qwen3-VL, and its FP32 VAEs. “Generate”
excludes model loading, output muxing, and process cleanup. Peak VRAM is sampled
over the complete process.

### FL2VA

| Task | System | Generate | Peak VRAM |
|---|---|---:|---:|
| Text | edge-dit.cpp | 51.396s | 125,051 MiB |
| | Diffusers | 53.986s | 128,801 MiB |
| First frame | edge-dit.cpp | 54.810s | 125,353 MiB |
| | Diffusers | 57.817s | 129,957 MiB |
| Last frame | edge-dit.cpp | 55.323s | 125,351 MiB |
| | Diffusers | 57.793s | 129,957 MiB |
| First + last frames | edge-dit.cpp | 58.747s | 125,573 MiB |
| | Diffusers | 61.405s | 130,587 MiB |

### Ref2VA

These benchmark rows use Edge-DiT.cpp `--ref-image-size max`, matching the
official Diffusers image geometry: image short edge 2048; video short edge 768
with a pre-rounding `768x1344` area cap; preserved aspect ratio; dimensions
rounded to multiples of 32; Lanczos resize. Edge-DiT.cpp is faster than
Diffusers in all four measured Ref2VA generation paths:

| Current task | System | Generate | Peak VRAM |
|---|---|---:|---:|
| Image | edge-dit.cpp | 126.915s | 130,445 MiB |
| | Diffusers | 136.656s | 139,253 MiB |
| MP4 video / video frames † | edge-dit.cpp | 182.649s | 132,661 MiB |
| | Diffusers | 183.921s | 137,161 MiB |
| Video frames + paired audio † | edge-dit.cpp | 182.667s | 132,663 MiB |
| | Diffusers | 185.035s | 137,183 MiB |
| Mixed references † | edge-dit.cpp | 301.435s | 138,113 MiB |
| | Diffusers | 319.435s | 141,915 MiB |

The image row is a strict same-image, same-prompt comparison. The
mixed-reference path has a clear 1.06x lead. The two video rows retain smaller
1.01x measured leads; they are marked † because the MP4 run lets edge-dit.cpp
extract embedded audio while the Diffusers run uses decoded video frames, and the
paired-audio prompts differ slightly. A locked-command rerun is required before
treating the approximately 1% margins as statistically significant.

### H200 15-second multi-reference comparison

The following longer Ref2VA runs are a separate benchmark from the BF16 tables
above. They compare two reference-image-heavy prompts across edge-dit.cpp,
ComfyUI, and Diffusers on one H200. The four-image action task uses `1280x736`;
the two-character portrait task uses `736x1280`. Both generate 362 frames at
24 fps (approximately 15.08 seconds), with 20 steps, CFG 1, seed
`157368968253448`, and `res_multistep`/`simple`. Each task uses the same prompt,
reference images in the same order, output canvas, frame count, and sampling
parameters across all three frameworks.

These dimensions are not `768x1344` and are not a MiniMax-H3 requirement. The
experiment preserved the submitted UI workflow configuration: a 0.9-megapixel
`ResolutionSelector`, 16:9 or 9:16 aspect ratio, and dimensions aligned to a
multiple of 32. That configuration resolves to `1280x736` or `736x1280`. The
parameters were retained so all three frameworks could replay the actual task;
therefore these results must not be compared directly with an older
`768x1344` run.

Outputs:

- [Four-image action three-framework comparison](assets/minimax-h3-ref2va-four-image-edge-comfyui-diffusers-demo.mp4)
- [Two-character portrait three-framework comparison](assets/minimax-h3-ref2va-two-character-edge-comfyui-diffusers-demo.mp4)
- [Machine-readable benchmark metrics](assets/minimax-h3-ref2va-h200-15s-metrics.json)

The linked comparison videos are compressed documentation assets tracked in the
repository. Full-resolution per-framework outputs remain benchmark artifacts
rather than source-tree dependencies.

The three frameworks use their practical quantized representations rather than
byte-identical weights:

| Framework | Ref2VA DiT | Qwen3-VL | Video / audio VAE |
|---|---|---|---|
| edge-dit.cpp | Pruned Q8_0 GGUF | Q4_K_M GGUF | FP16 / FP32 |
| ComfyUI | Pruned INT8 ConvRot | NVFP4 AWQ | FP16 / FP32 |
| Diffusers | Official full weights with TorchAO weight-only INT8 | Official weights with TorchAO weight-only INT8 | FP16 / FP32 |


| Task | Framework | End-to-end | Load / initialization | Text, image, and reference-VAE conditioning | DiT / sampling | Video VAE decode | Audio VAE decode | Save / mux | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Four-image action | edge-dit.cpp | **919.765s** | 17.773s | 2.641s | **839.890s** | 41.842s | 0.369s | 5.519s | **52,667 MiB (51.43 GiB)** |
| | ComfyUI | 1055.354s | Not isolated | Not isolated | Not isolated | Not isolated | Not isolated | Not isolated | **49,413 MiB (48.25 GiB)** |
| | Diffusers | 2343.937s | 32.260s | 14.924s | 2263.720s | **20.679s** | **0.145s** | **4.699s** | 107.38 GiB allocated / 108.27 GiB reserved |
| Two-character portrait | edge-dit.cpp | **888.636s** | 17.784s | **1.168s** | **811.867s** | 41.871s | 0.355s | 3.737s | 52,809 MiB (51.57 GiB) |
| | ComfyUI | 1025.864s | Not isolated | Not isolated | Not isolated | Not isolated | Not isolated | Not isolated | **49,159 MiB (48.01 GiB)** |
| | Diffusers | 1625.178s | 32.838s | 7.176s | 1553.626s | **20.567s** | **0.144s** | **3.481s** | 96.86 GiB allocated / 106.10 GiB reserved |

E2E time includes loading, conditioning, generation, decode, and
saving. edge-dit.cpp and ComfyUI peaks are device-level samples; Diffusers only
recorded PyTorch allocator peaks, so its allocated/reserved values are shown
separately. ComfyUI runs did not retain node-level profiles, so
unavailable stage values are intentionally not reconstructed.


Reference preprocessing is an important limitation of this comparison.
edge-dit.cpp and the captured ComfyUI workflow use `ref_image_size=match`, which
does not upscale the small source images. Diffusers uses the official
`MiniMaxH3ImageReference` path and enlarges every reference image to a 2048px
short edge. For the four-image task, edge-dit.cpp/ComfyUI use `512x320`,
`576x320`, `512x288`, and `512x288`, while Diffusers uses `3232x2048`,
`3648x2048`, `3648x2048`, and `3648x2048`. For the two-character task, the
respective sizes are `416x224` and `416x224` versus `3488x2048` and
`3616x2048`. Diffusers therefore processes substantially longer reference
sequences, increasing conditioning time, every DiT step, and peak memory. The
table is useful as an actual-workflow result and quality comparison, but it is
not a same-compute kernel benchmark.

## Limitations

- Reference adherence is prompt dependent. Name `<Picture N>`, `<Video N>`, and
  `<Audio N>` explicitly when their roles matter.
- Matching seeds across frameworks does not guarantee identical videos because
  numerical kernels and weight formats differ.
- Media-file reference decoding and embedded-audio extraction require `ffmpeg`.
- Without `--audio-vae`, video generation works but generated audio is not
  decoded or muxed.

## Other frontends

All shipped frontends accept MiniMax-H3's separate component layout.

- `ed-convert` accepts MiniMax-H3's main DiT transformer as `--diffusion-model`. Qwen can
  be converted independently with `--llm`
  and the result loaded through `--llm`. Packed Comfy NVFP4/AWQ safetensors are
  not a generic converter input; use BF16/F16 or a supported GGUF source. The
  Qwen3-VL vision tower can be extracted from a combined supported Qwen GGUF
  with `--llm-vision` and loaded through
  `--llm-vision`; language tensors are excluded from that output. The
  video/audio VAEs can be converted alone or included in a merge.
  `--llm`, `--vae`, and `--audio-vae` can be used together with the DiT to
  produce one combined GGUF that reloads through `ed-cli --model`.
- `ed-sample` accepts the same first/last frame and Ref2VA flags as `ed-cli`.
  A `--ref-video` is a numbered frame directory in this benchmark-oriented
  frontend, and generated audio is saved beside the AVI as `*.avi.wav`.
- `ed-server` accepts `--diffusion-model`, `--vae`, `--audio-vae`, and `--llm`,
  exposes `POST /ed/v1/videos/generations`, returns PNG frames, and returns
  interleaved float32 little-endian audio in `audio.b64_f32le`.
- Python supports `AudioInput`, `RefVideoInput`, MiniMax fields on
  `VideoRequest`, and a list-compatible `VideoOutput` carrying `audio`,
  `audio_sample_rate`, and `audio_channels`. The Python `/ed/v2` server accepts
  base64 first/last/reference frames and returns the same float32 audio fields.

See [API and Bindings Guide](api.md) for request schemas.
