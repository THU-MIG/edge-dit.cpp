# Quality summary (mean)

> Quality columns are per task (t2i: CLIP/aesthetic/IR; editing: dir-CLIP/keep-SSIM/keep-LPIPS/aesthetic/IR; video: per-frame CLIP/aesthetic + temporal). PSNR↑/SSIM↑/LPIPS↓ are quantization vs the same system's own FP16 baseline (not comparable across systems). Baseline tiers show —.


## flux-dev-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.303 | 6.11 | 1.768 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.298 | 5.99 | 1.847 | 20.21 | 0.819 | 0.219 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.282 | 5.95 | 1.771 | 16.42 | 0.713 | 0.364 |

## flux-schnell-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.316 | 5.47 | 1.880 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.310 | 5.74 | 1.822 | 14.70 | 0.692 | 0.298 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.312 | 5.58 | 1.886 | 28.46 | 0.944 | 0.036 |

## qwen-image-lightning-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 0.327 | 5.48 | 1.847 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.330 | 5.55 | 1.857 | 2.80 | 0.347 | 0.934 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — |

## qwen-image-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 0.330 | 5.27 | 1.866 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.329 | 5.26 | 1.858 | 2.62 | 0.355 | 0.931 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 0.312 | 5.57 | 1.882 | 2.35 | 0.328 | 0.930 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — |

## sd3-medium-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 0.329 | 5.54 | 1.831 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.339 | 5.15 | 1.855 | 19.80 | 0.776 | 0.291 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.322 | 5.55 | 1.795 | 25.34 | 0.922 | 0.072 |

## sd35-medium-turbo-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 0.338 | 5.34 | 1.827 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.316 | 5.38 | 0.398 | 19.30 | 0.690 | 0.433 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.333 | 5.27 | 1.791 | 32.38 | 0.967 | 0.047 |
