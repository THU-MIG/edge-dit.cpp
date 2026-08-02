# Summary table (mean, core columns)

> One table at a glance, split by task (quality columns differ per task). For speed look at DiT sampling ms; VRAM unit MiB; PSNR/SSIM/LPIPS are quantization vs same-system FP16.


## flux-dev-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 37192.0 | 38752.0 | 19652 | 0.303 | 6.11 | 1.768 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 23674 | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 23466 | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 10704.6 | 11455.0 | 11316 | 0.298 | 5.99 | 1.847 | 20.21 | 0.819 | 0.219 |
| edge-dit.cpp | q8_0 | no-offload | none | 10565.3 | 11302.0 | 19206 | 0.282 | 5.95 | 1.771 | 16.42 | 0.713 | 0.364 |

## flux-schnell-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 7631.1 | 8857.0 | 19630 | 0.316 | 5.47 | 1.880 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 23654 | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 23694 | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 2305.2 | 3026.0 | 11296 | 0.310 | 5.74 | 1.822 | 14.70 | 0.692 | 0.298 |
| edge-dit.cpp | q8_0 | no-offload | none | 2271.0 | 3010.0 | 19186 | 0.312 | 5.58 | 1.886 | 28.46 | 0.944 | 0.036 |

## qwen-image-lightning-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 12288.2 | 14086.0 | 19530 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 556 | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 556 | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 2471.6 | 3797.0 | 18746 | 0.327 | 5.48 | 1.847 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 2478.2 | 3229.0 | 18050 | 0.330 | 5.55 | 1.857 | 2.80 | 0.347 | 0.934 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 21428 | — | — | — | — | — | — |

## qwen-image-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 91080.2 | 93094.0 | 19530 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 556 | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 556 | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 17511.4 | 18842.0 | 18802 | 0.330 | 5.27 | 1.866 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 17510.4 | 18240.0 | 18050 | 0.329 | 5.26 | 1.858 | 2.62 | 0.355 | 0.931 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 56646.8 | 57815.0 | 16948 | 0.312 | 5.57 | 1.882 | 2.35 | 0.328 | 0.930 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 21430 | — | — | — | — | — | — |

## sd3-medium-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 3263.3 | 4054.0 | 15874 | 0.329 | 5.54 | 1.831 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 3475.9 | 4247.0 | 5758 | 0.339 | 5.15 | 1.855 | 19.80 | 0.776 | 0.291 |
| edge-dit.cpp | q8_0 | no-offload | none | 3500.8 | 4294.0 | 9264 | 0.322 | 5.55 | 1.795 | 25.34 | 0.922 | 0.072 |

## sd35-medium-turbo-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 856.5 | 1595.0 | 16956 | 0.338 | 5.34 | 1.827 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 931.3 | 1652.0 | 6548 | 0.316 | 5.38 | 0.398 | 19.30 | 0.690 | 0.433 |
| edge-dit.cpp | q8_0 | no-offload | none | 941.7 | 1648.0 | 10154 | 0.333 | 5.27 | 1.791 | 32.38 | 0.967 | 0.047 |
