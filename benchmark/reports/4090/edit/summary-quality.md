# Quality summary (mean)

> Quality columns are per task (t2i: CLIP/aesthetic/IR; editing: dir-CLIP/keep-SSIM/keep-LPIPS/aesthetic/IR; video: per-frame CLIP/aesthetic + temporal). PSNR↑/SSIM↑/LPIPS↓ are quantization vs the same system's own FP16 baseline (not comparable across systems). Baseline tiers show —.


## flux-kontext-image-editing  (image-editing)

| system | precision | budget | cache | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 0.104 | 0.564 | 0.620 | 5.83 | -0.410 | — | — | — |
| diffusers | w8 | no-offload | none | 0.100 | 0.564 | 0.620 | 5.83 | -0.415 | 38.44 | 0.972 | 0.011 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.107 | 0.572 | 0.615 | 5.85 | -0.457 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.103 | 0.579 | 0.609 | 5.87 | -0.449 | 28.87 | 0.917 | 0.051 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.110 | 0.572 | 0.616 | 5.84 | -0.448 | 36.71 | 0.985 | 0.006 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 0.105 | 0.547 | 0.633 | 5.67 | -0.342 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.095 | 0.554 | 0.624 | 5.78 | -0.391 | 25.29 | 0.884 | 0.078 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 0.104 | 0.545 | 0.635 | 5.68 | -0.359 | 38.38 | 0.987 | 0.007 |

## kontext-lightning-image-editing  (image-editing)

| system | precision | budget | cache | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.095 | 0.571 | 0.605 | 5.80 | -0.431 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.096 | 0.573 | 0.601 | 5.87 | -0.436 | 30.75 | 0.923 | 0.047 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.095 | 0.572 | 0.605 | 5.80 | -0.430 | 43.36 | 0.993 | 0.003 |

## qwen-image-edit-image-editing  (image-editing)

| system | precision | budget | cache | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 0.122 | 0.621 | 0.587 | 5.84 | -0.514 | — | — | — |
| diffusers | w8 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 0.139 | 0.626 | 0.582 | 5.85 | -0.537 | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 0.031 | 0.623 | 0.567 | 5.74 | -0.530 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.133 | 0.619 | 0.580 | 5.79 | -0.540 | 27.71 | 0.924 | 0.152 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 0.144 | 0.627 | 0.578 | 5.83 | -0.524 | 27.44 | 0.917 | 0.158 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | 0.130 | 0.675 | 0.543 | 5.78 | -0.640 | — | — | — |
| stable-diffusion.cpp | bf16 | no-offload | none | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.075 | 0.660 | 0.533 | 5.78 | -0.644 | 20.41 | 0.808 | 0.336 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 0.132 | 0.675 | 0.544 | 5.80 | -0.671 | 31.38 | 0.971 | 0.032 |
| stable-diffusion.cpp | q8_0 | no-offload | none | — | — | — | — | — | — | — | — |

## qwen-image-edit-lightning-image-editing  (image-editing)

| system | precision | budget | cache | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 0.144 | 0.628 | 0.562 | 5.89 | -0.510 | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 0.019 | 0.617 | 0.547 | 5.66 | -0.607 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.063 | 0.624 | 0.547 | 5.79 | -0.737 | 29.19 | 0.909 | 0.107 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — | — | — |
