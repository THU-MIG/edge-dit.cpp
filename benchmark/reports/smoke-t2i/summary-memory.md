# VRAM summary (mean, unit MiB)

> budget names the offloaded components (e.g. `te offload`, `full offload`) + `(max-vram Ng)` when --max-vram was set; auto tiers show the engine's real placement + `(auto-fit)`/`(auto-allocate)`. cache is its own column.


## flux-dev-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19652 | 10046 | 19652 | 1452 |
| edge-dit.cpp | f16 | no-offload | none | 23674 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 23466 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 11316 | 10456 | 11316 | 10820 |
| edge-dit.cpp | q8_0 | no-offload | none | 19206 | 18270 | 19206 | 18710 |

## flux-schnell-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19630 | 10040 | 19630 | 3500 |
| edge-dit.cpp | f16 | no-offload | none | 23654 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 23694 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 11296 | 10366 | 11296 | 10800 |
| edge-dit.cpp | q8_0 | no-offload | none | 19186 | 18256 | 19186 | 18690 |

## qwen-image-lightning-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19530 | 14170 | 19530 | 1464 |
| edge-dit.cpp | f16 | no-offload | none | 556 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 556 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 18746 | 18746 | 12342 | 12208 |
| edge-dit.cpp | q4_k | no-offload | none | 18050 | — | 18050 | 17776 |
| edge-dit.cpp | q8_0 | no-offload | none | 21428 | — | — | — |

## qwen-image-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19530 | 14172 | 19530 | 1464 |
| edge-dit.cpp | f16 | no-offload | none | 556 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 556 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 18802 | 18802 | 12342 | 12208 |
| edge-dit.cpp | q4_k | no-offload | none | 18050 | 17334 | 18050 | 17776 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 16948 | 7932 | 16948 | 1352 |
| edge-dit.cpp | q8_0 | no-offload | none | 21430 | — | — | — |

## sd3-medium-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 15874 | 15446 | 15808 | 15874 |
| edge-dit.cpp | q4_k | no-offload | none | 5758 | 5404 | 5692 | 5758 |
| edge-dit.cpp | q8_0 | no-offload | none | 9264 | 8904 | 9198 | 9264 |

## sd35-medium-turbo-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 16956 | 16498 | 16888 | 16956 |
| edge-dit.cpp | q4_k | no-offload | none | 6548 | 6168 | 6478 | 6548 |
| edge-dit.cpp | q8_0 | no-offload | none | 10154 | 9758 | 10086 | 10154 |
