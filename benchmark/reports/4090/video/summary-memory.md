# VRAM summary (mean, unit MiB)

> budget names the offloaded components (e.g. `te offload`, `full offload`) + `(max-vram Ng)` when --max-vram was set; auto tiers show the engine's real placement + `(auto-fit)`/`(auto-allocate)`. cache is its own column.


## wan2-t2v-1.3b-text-to-video

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 20498 | 14585 | 17405 | 20498 |
| diffusers | w8 | no-offload | none | 19568 | 15330 | 16181 | 19568 |
| edge-dit.cpp | f16 | no-offload | none | 17708 | 16753 | 17708 | 17612 |
| edge-dit.cpp | q4_k | no-offload | none | 9298 | 8460 | 9298 | 9202 |
| edge-dit.cpp | q8_0 | no-offload | none | 12176 | 11236 | 12176 | 12080 |
| stable-diffusion.cpp | f16 | no-offload | none | 17818 | 12661 | 15237 | 17818 |
| stable-diffusion.cpp | q4_k | no-offload | none | 11372 | 7441 | 8791 | 11372 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 11308 | 6535 | 8727 | 11308 |

## wan21-t2v-1.3b-distill-text-to-video

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 17692 | 16645 | 17692 | 17596 |
| edge-dit.cpp | q4_k | no-offload | none | 9282 | 8448 | 9282 | 9195 |
| edge-dit.cpp | q8_0 | no-offload | none | 12160 | 11286 | 12160 | 12064 |
