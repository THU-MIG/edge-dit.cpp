# Performance and Benchmarks (RTX 4090)

[Back to README](../README.md) · [H200 snapshot](performance-H200.md)

This is the primary benchmark snapshot for edge-dit.cpp, measured on
**RTX 4090 (24 GB)** with the CUDA `performance` build. It reports speed, VRAM,
and image-quality metrics across three tasks (text-to-image, image editing,
text-to-video) and three runtimes (edge-dit.cpp, Diffusers,
stable-diffusion.cpp). The older, load-inclusive
[H200 snapshot](performance-H200.md) is kept for historical comparison.

## How to read these numbers

- **Compare speed with `DiT sampling ms`** (component-level denoise time) — the
  only reliable cross-system metric.
- **End-to-end excludes model load** (load-once boundary): weights are loaded
  once outside the timed region, generation is timed tightly, output encoding is
  excluded. This differs from the H200 tables, whose end-to-end is
  **load-inclusive** — the two are not directly comparable.
- **Same-precision rule (iron law):** cross-system numbers are only comparable at
  equal weight precision. The 8-bit group is edge `q8_0` / sd.cpp `q8_0` /
  Diffusers `w8` (Optimum-Quanto qint8). The 16-bit group is edge `f16`
  (qwen/wan `bf16`) / sd.cpp `f16` (qwen `bf16`) / Diffusers `bf16`. `q4_k` only
  compares edge vs sd.cpp (Diffusers has no q4). Never compare edge `q4_k`
  against Diffusers `bf16`.
- **sd.cpp quantized tiers fold on-the-fly conversion** (q4_K/q8, tens of
  seconds) into the denoise stage, so their `DiT sampling ms` **and** end-to-end
  are inflated; read them as a same-tier trend only, never as a cross-system
  speed claim. Marked `*` below.
- **Quantization quality loss** (PSNR/SSIM/LPIPS) is only meaningful vs the
  *same system's own* 16-bit baseline (baseline rows show —); not comparable
  across systems.
- **24 GB constraint:** models that don't fit resident (Qwen-Image /
  Qwen-Image-Edit full precision, and their `q8` resident) fall back to an
  offload or `auto-fit` tier; those rows are marked with their budget and are
  **not** comparable to resident tiers. OOM rows show —.

## Test matrix at a glance

| Task | Models | 3-system compare | edge-only | Runs (ok/fail) |
|---|---|---|---|---|
| t2i | flux-dev, sd3-medium, flux-schnell, qwen-image (+ sd35-medium-turbo, qwen-image-lightning) | flux-dev / sd3-medium / flux-schnell / qwen-image | sd35-turbo (edge+diffusers), qwen-image-lightning | 165 (123/42) |
| edit | flux-kontext, qwen-image-edit (+ kontext-lightning, qwen-image-edit-lightning) | flux-kontext / qwen-image-edit | kontext-lightning, qwen-image-edit-lightning | 96 (66/30) |
| video | wan2-t2v-1.3b (+ wan21-t2v-1.3b-distill) | wan2-t2v-1.3b | wan21-t2v-1.3b-distill | 33 (33/0) |

All failures are expected 24 GB OOM on `no-offload` full-precision baselines or
large-model `q8` resident tiers (see [Data completeness](#data-completeness)).

---

## Text-to-image (1024x1024)

Base models: FLUX.1-dev (20 steps), SD3-Medium (20), Qwen-Image (30). Distilled:
SD3.5-Medium-Turbo (8), FLUX-schnell (4), Qwen-Image-Lightning (4).

### Speed (mean, ms)

Row order per model: 16-bit → 8-bit → q4_k → offload/auto tiers, so same-precision
rows sit adjacent.

**flux-dev** (16-bit resident does not fit 24 GB — offloaded only)
| System | Precision | Budget | DiT | E2E | TE | VAE |
|---|---|---|--:|--:|--:|--:|
| diffusers | bf16 | full offload | 21901 | 45241 | 2904 | 20224 |
| diffusers | bf16 | sequential offload | 85008 | 88939 | 3256 | 649 |
| edge-dit.cpp | f16 | full offload (20g) | 39079 | 40513 | 994 | 436 |
| stable-diffusion.cpp | f16 | full offload (20g) | 44723 | 54595 | 8583 | 1267 |
| diffusers | w8 | no-offload | 13190 | 14139 | 242 | 684 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **10569** | **11196** | 196 | 426 |
| stable-diffusion.cpp | q8_0* | no-offload | 17797 | 22194 | 3237 | 1143 |
| edge-dit.cpp | q4_k | no-offload | 10675 | 11334 | 224 | 430 |
| stable-diffusion.cpp | q4_k* | no-offload | 53440 | 78524 | 23927 | 1140 |

**sd3-medium** (all tiers resident — cleanest 3-system comparison, no OOM)
| System | Precision | Budget | DiT | E2E | TE | VAE |
|---|---|---|--:|--:|--:|--:|
| diffusers | bf16 | no-offload | 3040 | 3577 | 243 | 269 |
| edge-dit.cpp | f16 | no-offload | 3258 | 3962 | 317 | 381 |
| stable-diffusion.cpp | f16 | no-offload | 5193 | 7811 | 1493 | 1107 |
| diffusers | w8 | no-offload | 3411 | 3923 | 243 | 250 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **3434** | 4131 | 309 | 381 |
| stable-diffusion.cpp | q8_0* | no-offload | 5087 | 9975 | 3760 | 1110 |
| edge-dit.cpp | q4_k | no-offload | 3436 | 4133 | 309 | 382 |
| stable-diffusion.cpp | q4_k* | no-offload | 12360 | 37075 | 23587 | 1113 |

**flux-schnell** (distilled; 16-bit resident OOM)
| System | Precision | Budget | DiT | E2E | TE | VAE |
|---|---|---|--:|--:|--:|--:|
| edge-dit.cpp | f16 | full offload (20g) | 7960 | 9579 | 1011 | 602 |
| stable-diffusion.cpp | f16 | full offload (20g) | 19950 | 26978 | 5740 | 1270 |
| diffusers | w8 | no-offload | 2643 | 3412 | 275 | 466 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **2220** | 2876 | 223 | 428 |
| stable-diffusion.cpp | q8_0* | no-offload | 7383 | 11744 | 3200 | 1143 |
| edge-dit.cpp | q4_k | no-offload | 2242 | 2891 | 215 | 430 |
| stable-diffusion.cpp | q4_k* | no-offload | 46117 | 71760 | 24487 | 1140 |

**sd35-medium-turbo** (edge + diffusers)
| System | Precision | Budget | DiT | E2E | TE | VAE |
|---|---|---|--:|--:|--:|--:|
| diffusers | bf16 | no-offload | 1717 | 2230 | 251 | 243 |
| edge-dit.cpp | f16 | no-offload | 1698 | 2408 | 319 | 385 |
| diffusers | w8 | no-offload | 1889 | 2368 | 236 | 227 |
| edge-dit.cpp | q8_0 | no-offload | 1767 | 2452 | 295 | 384 |
| edge-dit.cpp | q4_k | no-offload | 1765 | 2469 | 314 | 381 |

**qwen-image** (24 GB: full-precision and q8 resident all OOM — see notes)
| System | Precision | Budget | DiT | E2E | TE | VAE |
|---|---|---|--:|--:|--:|--:|
| diffusers | bf16 | sequential offload | 379937 | 387596 | 6209 | 1361 |
| edge-dit.cpp | bf16 | full offload (20g) | 210592 | 213537 | 2186 | 754 |
| stable-diffusion.cpp | bf16 | full offload (20g) | 182913 | 192832 | 6883 | 3020 |
| diffusers | w8 | full offload | 54696 | 72270 | 8256 | 8920 |
| diffusers | w8 | no-offload | — OOM | — | — | — |
| edge-dit.cpp | q8_0 | DiT+TE offload (auto-allocate) | 129887 | 131534 | 1148 | 494 |
| stable-diffusion.cpp | q8_0* | full offload (20g) | 89720 | 102539 | 9787 | 3013 |
| **edge-dit.cpp** | **q4_k** | **no-offload (resident)** | **35593** | 36269 | 200 | 471 |
| edge-dit.cpp | f16→q4_k | te+vae offload (auto-fit) | 35404 | 37114 | 1000 | 705 |
| stable-diffusion.cpp | q4_k* | no-offload | 157523 | 203449 | 43637 | 2273 |

**qwen-image-lightning** (edge-only, distilled)
| System | Precision | Budget | DiT | E2E | TE | VAE |
|---|---|---|--:|--:|--:|--:|
| edge-dit.cpp | bf16 | full offload (20g) | 14002 | 16439 | 1504 | 926 |
| edge-dit.cpp | q4_k | no-offload (resident) | 2438 | 3119 | 178 | 496 |
| edge-dit.cpp | f16→q4_k | te+vae offload (auto-fit) | 2424 | 3659 | 523 | 707 |

### VRAM (mean peak, MiB)

| Model | edge f16/bf16 | edge q8_0 | edge q4_k | diffusers 16-bit | diffusers w8 | sd.cpp f16/bf16 | sd.cpp q8_0 | sd.cpp q4_k |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| flux-dev | 19610¹ | 19112 | 11222 | 23960¹ | 23866 | 17528¹ | 18559 | 11037 |
| sd3-medium | 15757 | 9147 | 5641 | 20080 | 18172 | 15822 | 9106 | 5968 |
| flux-schnell | 19588¹ | 19092 | 11202 | — OOM | 23869 | 17467¹ | 18539 | 11017 |
| sd35-turbo | 16839 | 10037 | 6431 | 20938 | 18842 | — | — | — |
| qwen-image | 19606¹ | 17019² | 21263 | 4910³ | 21264¹ | 16917¹ | 18799¹ | 17725 |
| qwen-image-lightning | 19499¹ | — OOM | 21210 | — | — | — | — | — |

¹ offloaded (full offload, max-vram 20g). ² auto-allocate (DiT+TE offload).
³ sequential offload. Non-annotated cells are resident (no-offload).

### Quality (mean; PSNR↑/SSIM↑/LPIPS↓ vs same-system 16-bit; baseline = —)

| Model | System | Prec | CLIP | aes | IR | PSNR | SSIM | LPIPS |
|---|---|---|--:|--:|--:|--:|--:|--:|
| flux-dev | diffusers | bf16 | 0.308 | 6.14 | 1.737 | — | — | — |
| | diffusers | w8 | 0.307 | 6.12 | 1.735 | 30.81 | 0.970 | 0.024 |
| | edge | q8_0 | 0.296 | 6.07 | 1.630 | 28.65 | 0.892 | 0.130 |
| | edge | q4_k | 0.304 | 6.03 | 1.611 | 21.45 | 0.808 | 0.243 |
| | sd.cpp | q8_0 | 0.310 | 6.10 | 1.525 | 29.00 | 0.962 | 0.043 |
| sd3-medium | diffusers | bf16 | 0.336 | 5.52 | 1.685 | — | — | — |
| | edge | q8_0 | 0.328 | 5.70 | 1.556 | 22.19 | 0.873 | 0.144 |
| | edge | q4_k | 0.345 | 5.30 | 1.388 | 17.87 | 0.704 | 0.406 |
| | sd.cpp | q8_0 | 0.341 | 5.58 | 1.356 | 23.30 | 0.853 | 0.180 |
| sd35-turbo | diffusers | bf16 | 0.322 | 5.37 | -0.017 | — | — | — |
| | edge | q8_0 | 0.339 | 5.47 | 1.196 | 28.03 | 0.908 | 0.115 |
| qwen-image | edge | q8_0 (auto-alloc) | 0.325 | 5.96 | 1.855 | 28.79 | 0.912 | 0.073 |
| | edge | q4_k | 0.327 | 5.96 | 1.850 | 24.12 | 0.864 | 0.120 |
| | sd.cpp | q8_0 | 0.325 | 6.02 | 1.833 | 28.33 | 0.946 | 0.029 |

(Full per-tier quality including flux-schnell, qwen-lightning, and all q4_k rows
is in `benchmark/reports/t2i/summary-quality.md`.)

### Same-precision cross-system readout

**8-bit weight-only (headline tier), DiT ms / peak MiB / IR:**

| Model | edge q8_0 | diffusers w8 | sd.cpp q8_0* |
|---|---|---|---|
| flux-dev | **10569** / 19112 / 1.63 | 13190 / 23866 / 1.74 | 17797 / 18559 / 1.53 |
| sd3-medium | **3434** / 9147 / 1.56 | 3411 / 18172 / 1.66 | 5087 / 9106 / 1.36 |
| flux-schnell | **2220** / 19092 / 1.41 | 2643 / 23869 / 1.60 | 7383 / 18539 / 1.73 |

edge q8_0 is fastest (DiT) on flux-dev/schnell and ~matches Diffusers on sd3,
while using the least or comparable VRAM. sd.cpp q8 DiT is inflated by on-the-fly
conversion. **16-bit** for flux-dev/schnell is offloaded-only on 24 GB (not pure
resident inference); sd3 16-bit resident: edge f16 3258 / diffusers bf16 3040 /
sd.cpp f16 5193.

### Notes

- **8-bit is the headline usable-quality tier.** q4_k is an extreme
  VRAM-saving reference with visible loss (e.g. SD3 q4_k IR 1.388, LPIPS 0.406).
- **sd.cpp quantized speed is inflated**: its TE_ms reaches 20000–65000 ms
  (on-the-fly conversion), contaminating DiT and E2E (flux-dev q4_k DiT 53440,
  qwen-image q4_k 157523 are not pure inference).
- **flux-dev / flux-schnell 16-bit resident does not fit 24 GB** — all three
  runtimes OOM at no-offload f16/bf16; only offload tiers run, so any 16-bit
  cross-system comparison is offloaded, not resident.
- **Qwen-Image on 24 GB:** full precision *and* q8/w8 resident all OOM. edge's
  only fast resident tier is **q4_k (21263 MiB, DiT 35593 ms)**; edge q8 must
  offload (auto-allocate) → 129887 ms. At the 8-bit offloaded working point,
  **Diffusers w8 full-offload is fastest (DiT 54696 ms)**, ahead of sd.cpp q8
  full-offload (89720 ms) and edge q8 auto-allocate (129887 ms) — but the three
  use different offload budgets, so read them as per-runtime working points, not
  a like-for-like speed ratio. (Diffusers w8 needs `offload: full`, **not**
  `sequential` — the latter crashes Quanto's quantized tensors.)

---

## Image editing (1024x1024)

Base: FLUX.1-Kontext (20 steps, guidance 2.5), Qwen-Image-Edit (30). Distilled
(edge-only): Kontext-Lightning (8), Qwen-Image-Edit-Lightning (4). Input image:
`benchmark/assets/edit_input.png`. Quality metrics: dir-CLIP↑ (edit adherence),
keep-SSIM↑ / keep-LPIPS↓ (source preservation), aesthetic, IR.

### Speed (mean, ms)

**flux-kontext**
| System | Precision | Budget | DiT | E2E | TE | VAE |
|---|---|---|--:|--:|--:|--:|
| diffusers | bf16 | sequential offload | 109253 | 112697 | 3015 | 407 |
| edge-dit.cpp | f16 | full offload (20g) | 55107 | 56985 | 1409 | 465 |
| stable-diffusion.cpp | f16 | full offload (20g) | 67350 | 75191 | 5713 | 1217 |
| diffusers | w8 | no-offload | 27945 | 28704 | 495 | 239 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **24534** | 25510 | 577 | 394 |
| stable-diffusion.cpp | q8_0* | no-offload | 39133 | 44415 | 3280 | 1123 |
| edge-dit.cpp | q4_k | no-offload | 24816 | 25794 | 578 | 395 |
| stable-diffusion.cpp | q4_k* | no-offload | 79403 | 105640 | 24217 | 1123 |

**kontext-lightning** (edge-only): f16 full-offload 21134 / 22894; **q8_0 9910** /
10887; q4_k 10008 / 10983.

**qwen-image-edit** (24 GB: full/q8 resident OOM)
| System | Precision | Budget | DiT | E2E | TE | VAE |
|---|---|---|--:|--:|--:|--:|
| diffusers | bf16 | sequential offload | 400774 | 410003 | 8721 | 482 |
| edge-dit.cpp | bf16 | full offload (20g) | 290035 | 302168 | 11152 | 937 |
| stable-diffusion.cpp | bf16 | full offload (20g) | 148907 | 160446 | 6907 | 3037 |
| edge-dit.cpp | q8_0 | DiT+TE offload (auto-allocate) | 199119 | 204793 | 5205 | 438 |
| stable-diffusion.cpp | q8_0* | full offload (20g) | 73317 | 88141 | 10223 | 3003 |
| **edge-dit.cpp** | **q4_k** | **no-offload (resident)** | **103773** | 107249 | 3022 | 423 |
| edge-dit.cpp | f16→q4_k | te+vae offload (auto-fit) | 103735 | 110068 | 5461 | 845 |
| stable-diffusion.cpp | q4_k* | no-offload | 145637 | 192569 | 43300 | 2247 |

**qwen-image-edit-lightning** (edge-only): bf16 full-offload 18674 / 25445;
**q4_k resident 6864** / 9349; f16→q4_k auto-fit 6912 / 11196.

### VRAM (mean peak, MiB)

| Model | edge f16/bf16 | edge q8_0 | edge q4_k | diffusers | sd.cpp f16/bf16 | sd.cpp q8_0 | sd.cpp q4_k |
|---|--:|--:|--:|--:|--:|--:|--:|
| flux-kontext | 19090¹ | 20111 | 12221 | 23868 (w8) / 2051³ | 17746¹ | 19418 | 11896 |
| kontext-lightning | 19090¹ | 20111 | 12221 | — | — | — | — |
| qwen-image-edit | 19928¹ | 16107² | 23273 | 4977³ | 16966¹ | 18801¹ | 18268 |
| qwen-image-edit-lightning | 19689¹ | — OOM | 23233 | — | — | — | — |

¹ full offload (20g). ² auto-allocate. ³ sequential offload.

### Quality (mean; dir-CLIP↑ / keep-SSIM↑ / keep-LPIPS↓ / aes / IR)

| Model | System | Prec | dir-CLIP | keep-SSIM | keep-LPIPS | aes | IR |
|---|---|---|--:|--:|--:|--:|--:|
| flux-kontext | diffusers | w8 | 0.100 | 0.564 | 0.620 | 5.83 | -0.415 |
| | edge | q8_0 | 0.110 | 0.572 | 0.616 | 5.84 | -0.448 |
| | sd.cpp | q8_0 | 0.104 | 0.545 | 0.635 | 5.68 | -0.359 |
| qwen-image-edit | edge | q8_0 (auto-alloc) | 0.144 | 0.627 | 0.578 | 5.83 | -0.524 |
| | edge | q4_k | 0.133 | 0.619 | 0.580 | 5.79 | -0.540 |
| | edge | f16→q4_k (auto-fit) | **0.031** ⚠ | 0.623 | 0.567 | 5.74 | -0.530 |
| | sd.cpp | q8_0 | 0.132 | 0.675 | 0.544 | 5.80 | -0.671 |
| qwen-image-edit-lightning | edge | bf16 | 0.144 | 0.628 | 0.562 | 5.89 | -0.510 |
| | edge | q4_k | **0.063** ⚠ | 0.624 | 0.547 | 5.79 | -0.737 |
| | edge | f16→q4_k (auto-fit) | **0.019** ⚠ | 0.617 | 0.547 | 5.66 | -0.607 |

### Notes

- **Editing quality = dir-CLIP (adherence) + keep-SSIM/LPIPS (preservation).**
  flux-kontext q8 dir-CLIP is close across systems (0.104–0.110); edge q8
  keep-SSIM (0.572) slightly beats sd.cpp (0.545).
- **⚠ auto-fit (f16→q4_k) breaks editing semantics:** qwen-image-edit auto-fit
  dir-CLIP is only **0.031**, qwen-image-edit-lightning **0.019** (vs ~0.13–0.14
  for bf16/q4_k resident) — the edit barely follows the prompt. This tier is a
  memory/speed reference only, **not usable quality**; lightning q4_k (0.063)
  also degrades badly.
- **Qwen-Image-Edit is heavier than t2i:** even q4_k resident needs 23273 MiB
  (barely fits 24 GB); q8 resident OOMs, so it must offload → ~199 s.
- diffusers flux-kontext `full offload` fails; the `sequential` offload tier is
  used as its baseline.

---

## Text-to-video (832x480, 41 frames)

Wan2.1-T2V-1.3B (30 steps, CFG 5.0, flow-shift 3.0), 3-system. Wan2.1-T2V-1.3B-
Distill (8 steps, edge-only). Quality: frame-CLIP / frame-aesthetic /
temporal-LPIPS↓ / temporal-SSIM↑ / flicker-std↓. **All tiers resident — the only
task with no OOM, so speed is the cleanest resident comparison.**

### Speed (mean, ms)

**wan2-t2v-1.3b**
| System | Precision | DiT | E2E | TE | VAE |
|---|---|--:|--:|--:|--:|
| diffusers | bf16 | 53708 | 56729 | 231 | 2650 |
| edge-dit.cpp | f16 | **49445** | 55424 | 322 | 4965 |
| stable-diffusion.cpp | f16 | 83423 | 105119 | 2263 | 18927 |
| diffusers | w8 | 56580 | 59598 | 227 | 2654 |
| edge-dit.cpp | q8_0 | 53964 | 59927 | 290 | 4971 |
| stable-diffusion.cpp | q8_0* | 80383 | 111750 | 11867 | 18997 |
| edge-dit.cpp | q4_k | 54043 | 60005 | 299 | 4962 |
| stable-diffusion.cpp | q4_k* | 86783 | 125168 | 18867 | 19017 |

**wan21-t2v-1.3b-distill** (edge-only): f16 **6732** / 12594; q8_0 7384 / 13133;
q4_k 7394 / 13198.

### VRAM (mean peak, MiB)

| Model | edge f16 | edge q8_0 | edge q4_k | diffusers bf16 | diffusers w8 | sd.cpp f16 | sd.cpp q8_0 | sd.cpp q4_k |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| wan2-t2v-1.3b | 17708 | 12176 | 9298 | 20498 | 19568 | 17818 | 11308 | 11372 |
| wan21-distill | 17692 | 12160 | 9282 | — | — | — | — | — |

### Quality (mean; fCLIP / f-aes / tLPIPS↓ / tSSIM↑ / flicker↓)

| Model | System | Prec | fCLIP | f-aes | tLPIPS | tSSIM | flicker |
|---|---|---|--:|--:|--:|--:|--:|
| wan2-t2v-1.3b | diffusers | bf16 | 0.317 | 5.36 | 0.072 | 0.873 | 0.010 |
| | edge | f16 | 0.288 | 5.51 | 0.016 | 0.938 | 0.008 |
| | edge | q8_0 | 0.320 | 5.38 | 0.012 | 0.941 | 0.004 |
| | sd.cpp | f16 | 0.315 | 5.59 | 0.032 | 0.893 | 0.014 |
| | sd.cpp | q8_0 | 0.330 | 5.61 | 0.024 | 0.901 | 0.006 |

### Same-precision readout + notes

- **8-bit:** edge q8_0 DiT **53964** / peak 12176 vs diffusers w8 56580 / 19568
  vs sd.cpp q8_0* 80383 / 11308. edge q8 ≈ diffusers w8 in speed but uses ~40%
  less VRAM (12176 vs 19568).
- **16-bit:** edge f16 **49445** (fastest, resident) vs diffusers bf16 53708 vs
  sd.cpp f16 83423.
- edge **VAE_ms is high (~4900–5000 vs diffusers ~2650)** because Wan's 3D-conv
  VAE takes an im2col path — the E2E gap vs diffusers is mostly VAE, not DiT.
  sd.cpp quantized TE_ms is inflated (q8 11867, q4_k 18867).

---

## Reproducibility

**Contract:** RTX 4090 (24 GB) node (the machine has 8× RTX 4090; the benchmark
runs **serially on a single card**, `device: 0`, to avoid PCIe / memory-bandwidth
contention). CUDA `performance` build (`build-cuda/bin/ed-sample`). 1024×1024
(video 832×480, 41 frames); **model-default steps** (table below); batch 1;
seed 0; one untimed warm-up per config on its first prompt; **mean over 3
prompts**. Measurement boundary = **load-once**: end-to-end **excludes model load
and output encoding** (not directly comparable to the load-inclusive H200 page).

```bash
python3 benchmark/run.py \
  --job  benchmark/jobs/<t2i|edit|video>.yaml \
  --site benchmark/sites/<site4090>.yaml \
  --device 0
```

`--dry-run` prints the expansion only. Reports land in
`benchmark/reports/<job>/` (committed); raw artifacts in
`benchmark/results/<job>/` (git-ignored). To re-score / re-table without
regenerating, run `scripts/eval_all.py` + `scripts/make_matrix_tables.py` +
`scripts/summarize.py`.

**Workload parameters** (from `benchmark/models/*.yaml`):

| Model | Steps | cfg_scale | Notes |
|---|--:|--:|---|
| flux-dev | 20 | 1.0 | guidance 3.5 |
| flux-schnell | 4 | 1.0 | distilled |
| sd3-medium | 20 | 5.0 | flow-shift 3.0 |
| sd35-medium-turbo | 8 | 1.5 | flow-shift 3.0, distilled |
| qwen-image | 30 | 4.0 | true CFG (two forward passes) |
| qwen-image-lightning | 4 | 1.0 | distilled, single pass |
| flux-kontext | 20 | 1.0 | guidance 2.5 |
| kontext-lightning | 8 | 1.0 | guidance 2.5, distilled |
| qwen-image-edit | 30 | 4.0 | true CFG |
| qwen-image-edit-lightning | 4 | 1.0 | distilled |
| wan2-t2v-1.3b | 30 | 5.0 | 41 frames, flow-shift 3.0 |
| wan21-t2v-1.3b-distill | 8 | 1.0 | 41 frames, distilled |

**Same-precision mapping:** 8-bit = edge `q8_0` / sd.cpp `q8_0` / diffusers `w8`;
16-bit = edge `f16` (qwen/wan `bf16`) / sd.cpp `f16` (qwen `bf16`) / diffusers
`bf16` (note f16 ≠ bf16 but both 16-bit); `q4_k` = edge / sd.cpp only.

## Data completeness

- **Run counts:** t2i 165 (123 ok / 42 failed); edit 96 (66 / 30); video 33 (33 / 0).
- **All failures are expected 24 GB OOM** on `no-offload` full-precision baselines
  or large-model `q8` resident tiers: flux-dev/schnell 16-bit no-offload (all
  three systems); qwen-image / qwen-image-edit bf16 & q8 no-offload; the two
  qwen lightning variants' bf16 & q8 no-offload. Video has no failures (small
  model, all resident).
- Per-prompt detail (not just means) is in
  `benchmark/reports/{t2i,edit,video}/tables.md`.

## Related Documentation

- [H200 snapshot](performance-H200.md) — earlier load-inclusive H200 tables plus feature results (parallelism, computation reuse, VAE tiling, operator optimization).
- [Supported models](models.md), [CLI](cli.md), [Build and installation](build.md).
