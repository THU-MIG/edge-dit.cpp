# VRAM summary (mean, unit MiB)

> budget names the offloaded components (e.g. `te offload`, `full offload`) + `(max-vram Ng)` when --max-vram was set; auto tiers show the engine's real placement + `(auto-fit)`/`(auto-allocate)`. cache is its own column.


## flux-kontext-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 18930 | 10042 | 18930 | 1858 |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 23762 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 12252 | 10818 | 12252 | 11004 |
| edge-dit.cpp | q8_0 | no-offload | none | 20142 | 18708 | 20142 | 18894 |

## kontext-lightning-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 18930 | 10042 | 18930 | 1858 |
| edge-dit.cpp | f16 | no-offload | none | 23738 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 23532 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 12252 | 10818 | 12252 | 11004 |
| edge-dit.cpp | q8_0 | no-offload | none | 20142 | 18708 | 20142 | 18894 |

## qwen-image-edit-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19232 | 15528 | 19232 | 1838 |
| edge-dit.cpp | f16 | no-offload | none | 556 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 556 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 18810 | 18810 | 13318 | 12420 |
| edge-dit.cpp | q4_k | no-offload | none | 19440 | 18772 | 19440 | 18542 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 16184 | 8976 | 16184 | 1660 |
| edge-dit.cpp | q8_0 | no-offload | none | 21534 | — | — | — |

## qwen-image-edit-lightning-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19232 | 15522 | 19232 | 1838 |
| edge-dit.cpp | f16 | no-offload | none | 556 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 556 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 18876 | 18876 | 13318 | 12420 |
| edge-dit.cpp | q4_k | no-offload | none | 19440 | 18772 | 19440 | 18542 |
| edge-dit.cpp | q8_0 | no-offload | none | 21532 | — | — | — |
