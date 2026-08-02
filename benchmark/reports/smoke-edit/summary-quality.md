# Quality summary (mean)

> Quality columns are per task (t2i: CLIP/aesthetic/IR; editing: dir-CLIP/keep-SSIM/keep-LPIPS/aesthetic/IR; video: per-frame CLIP/aesthetic + temporal). PSNR↑/SSIM↑/LPIPS↓ are quantization vs the same system's own FP16 baseline (not comparable across systems). Baseline tiers show —.


## flux-kontext-image-editing  (image-editing)

| system | precision | budget | cache | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | -0.120 | 0.957 | 0.032 | 5.86 | -0.126 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | -0.104 | 0.958 | 0.029 | 5.89 | -0.127 | 45.28 | 0.996 | 0.001 |
| edge-dit.cpp | q8_0 | no-offload | none | -0.131 | 0.957 | 0.032 | 5.86 | -0.126 | 54.61 | 0.998 | 0.000 |

## kontext-lightning-image-editing  (image-editing)

| system | precision | budget | cache | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | -0.133 | 0.954 | 0.034 | 5.81 | -0.125 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | -0.110 | 0.955 | 0.030 | 5.83 | -0.145 | 45.41 | 0.996 | 0.002 |
| edge-dit.cpp | q8_0 | no-offload | none | -0.138 | 0.954 | 0.034 | 5.79 | -0.142 | 55.26 | 0.999 | 0.000 |

## qwen-image-edit-image-editing  (image-editing)

| system | precision | budget | cache | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | -0.103 | 0.528 | 0.671 | 5.65 | -0.740 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | -0.160 | 0.763 | 0.311 | 5.53 | -0.560 | 10.30 | 0.630 | 0.543 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | -0.065 | 0.693 | 0.370 | 5.99 | -0.252 | 10.10 | 0.569 | 0.652 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — | — | — |

## qwen-image-edit-lightning-image-editing  (image-editing)

| system | precision | budget | cache | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | -0.145 | 0.614 | 0.520 | 5.76 | -0.137 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | -0.107 | 0.714 | 0.363 | 5.57 | -0.366 | 15.13 | 0.757 | 0.322 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — | — | — |
