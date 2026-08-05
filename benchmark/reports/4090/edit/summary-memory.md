# VRAM summary (mean, unit MiB)

> budget names the offloaded components (e.g. `te offload`, `full offload`) + `(max-vram Ng)` when --max-vram was set; auto tiers show the engine's real placement + `(auto-fit)`/`(auto-allocate)`. cache is its own column.


## flux-kontext-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | 23978 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 2051 | 1219 | 1756 | 1472 |
| diffusers | w8 | no-offload | none | 23868 | 22891 | 23754 | 23513 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19090 | 1631 | 19090 | 1750 |
| edge-dit.cpp | q4_k | no-offload | none | 12221 | 10802 | 12221 | 10874 |
| edge-dit.cpp | q8_0 | no-offload | none | 20111 | 18692 | 20111 | 18764 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 17746 | 9837 | 17746 | 1516 |
| stable-diffusion.cpp | q4_k | no-offload | none | 11896 | 3788 | 11896 | 10872 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 19418 | 5668 | 19418 | 18394 |

## kontext-lightning-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19090 | 1419 | 19090 | 1750 |
| edge-dit.cpp | f16 | no-offload | none | 23600 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 12221 | 10802 | 12221 | 11309 |
| edge-dit.cpp | q8_0 | no-offload | none | 20111 | 18692 | 20111 | 18764 |

## qwen-image-edit-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 24050 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 4977 | 3621 | 2013 | 3682 |
| diffusers | w8 | no-offload | none | 24014 | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 19928 | 19262 | 19928 | 2774 |
| edge-dit.cpp | bf16 | no-offload | none | 418 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 18193 | 18193 | 13243 | 12456 |
| edge-dit.cpp | q4_k | no-offload | none | 23273 | 23273 | 22809 | 21780 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 16107 | 11836 | 16107 | 1786 |
| edge-dit.cpp | q8_0 | no-offload | none | 21396 | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | 16966 | 15108 | 16966 | 1310 |
| stable-diffusion.cpp | bf16 | no-offload | none | 15123 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 18268 | 6168 | 17826 | 17782 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 18801 | 7719 | 18784 | 1198 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 7785 | — | — | — |

## qwen-image-edit-lightning-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 19689 | 19386 | 19689 | 2476 |
| edge-dit.cpp | bf16 | no-offload | none | 418 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 18389 | 18389 | 13105 | 12316 |
| edge-dit.cpp | q4_k | no-offload | none | 23233 | 23233 | 22692 | 21660 |
| edge-dit.cpp | q8_0 | no-offload | none | 21396 | — | — | — |
