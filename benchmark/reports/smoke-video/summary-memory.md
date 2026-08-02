# VRAM summary (mean, unit MiB)

> budget names the offloaded components (e.g. `te offload`, `full offload`) + `(max-vram Ng)` when --max-vram was set; auto tiers show the engine's real placement + `(auto-fit)`/`(auto-allocate)`. cache is its own column.


## wan2-t2v-1.3b-text-to-video

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 17776 | 16802 | 17776 | 17750 |
| edge-dit.cpp | q4_k | no-offload | none | 9366 | 8270 | 9366 | 9340 |
| edge-dit.cpp | q8_0 | no-offload | none | 12244 | 11148 | 12244 | 12218 |

## wan2-t2v-14b-text-to-video

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 17584 | 5352 | 17584 | 3688 |
| edge-dit.cpp | f16 | no-offload | none | 556 | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 19958 | 19958 | 18562 | 17320 |
| edge-dit.cpp | q4_k | no-offload | none | 18396 | 16146 | 18396 | 17014 |
| edge-dit.cpp | q8_0 | no-offload | none | 23962 | — | — | — |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | 20028 | 20028 | 18562 | 17320 |

## wan21-t2v-1.3b-distill-text-to-video

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 17760 | 16780 | 17760 | 17734 |
| edge-dit.cpp | q4_k | no-offload | none | 9350 | — | 9350 | 9324 |
| edge-dit.cpp | q8_0 | no-offload | none | 12228 | — | 12228 | 12202 |
