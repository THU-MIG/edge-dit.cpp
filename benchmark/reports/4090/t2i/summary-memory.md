# VRAM summary (mean, unit MiB)

> budget names the offloaded components (e.g. `te offload`, `full offload`) + `(max-vram Ng)` when --max-vram was set; auto tiers show the engine's real placement + `(auto-fit)`/`(auto-allocate)`. cache is its own column.


## flux-dev-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | 23960 | 9552 | 23927 | 23883 |
| diffusers | bf16 | no-offload | none | 24042 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 1817 | 737 | 1293 | 1769 |
| diffusers | w8 | no-offload | none | 23866 | 21932 | 23035 | 23405 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19610 | 1393 | 19610 | 1320 |
| edge-dit.cpp | f16 | no-offload | none | 23536 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 11222 | 10239 | 11222 | 10682 |
| edge-dit.cpp | q8_0 | no-offload | none | 19112 | 18126 | 19112 | 18572 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 17528 | 9665 | 17528 | 1258 |
| stable-diffusion.cpp | f16 | no-offload | none | 9859 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 11037 | 3662 | 11037 | 10738 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 18559 | 5542 | 18559 | 18260 |

## flux-schnell-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 24024 | — | — | — |
| diffusers | w8 | no-offload | none | 23869 | 21832 | 22572 | 23491 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19588 | 1371 | 19588 | 1318 |
| edge-dit.cpp | f16 | no-offload | none | 23516 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 11202 | 10198 | 11202 | 10662 |
| edge-dit.cpp | q8_0 | no-offload | none | 19092 | 18243 | 19092 | 18552 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 17467 | 9665 | 17460 | 1258 |
| stable-diffusion.cpp | f16 | no-offload | none | 9834 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 11017 | 3697 | 11017 | 10718 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 18539 | 5542 | 18539 | 18240 |

## qwen-image-lightning-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 19499 | 1992 | 19499 | 1275 |
| edge-dit.cpp | bf16 | no-offload | none | 418 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 18750 | 18750 | 12262 | 12070 |
| edge-dit.cpp | q4_k | no-offload | none | 21210 | 20443 | 21210 | 20878 |
| edge-dit.cpp | q8_0 | no-offload | none | 21289 | — | — | — |

## qwen-image-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 24050 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 4910 | 2916 | 1018 | 2908 |
| diffusers | w8 | full offload | none | 21264 | 16714 | 21264 | 21264 |
| diffusers | w8 | no-offload | none | 24014 | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 19606 | 2421 | 19606 | 1436 |
| edge-dit.cpp | bf16 | no-offload | none | 418 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 18796 | 18796 | 12315 | 12126 |
| edge-dit.cpp | q4_k | no-offload | none | 21263 | 20413 | 21263 | 20934 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 17019 | 8028 | 17019 | 1320 |
| edge-dit.cpp | q8_0 | no-offload | none | 21292 | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | 16917 | 15079 | 16917 | 1310 |
| stable-diffusion.cpp | bf16 | no-offload | none | 14978 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 17725 | 6033 | 17723 | 17680 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 18799 | 7689 | 18799 | 1198 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 7649 | — | — | — |

## sd3-medium-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 20080 | 17080 | 17413 | 17975 |
| diffusers | w8 | no-offload | none | 18172 | 15167 | 15505 | 17401 |
| edge-dit.cpp | f16 | no-offload | none | 15757 | 15415 | 15713 | 15736 |
| edge-dit.cpp | q4_k | no-offload | none | 5641 | 5348 | 5597 | 5620 |
| edge-dit.cpp | q8_0 | no-offload | none | 9147 | 8829 | 9103 | 9126 |
| stable-diffusion.cpp | f16 | no-offload | none | 15822 | 11094 | 15542 | 15822 |
| stable-diffusion.cpp | q4_k | no-offload | none | 5968 | 4062 | 5685 | 5968 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 9106 | 6244 | 8823 | 9106 |

## sd35-medium-turbo-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 20938 | 17874 | 18271 | 18938 |
| diffusers | w8 | no-offload | none | 18842 | 14232 | 16175 | 16530 |
| edge-dit.cpp | f16 | no-offload | none | 16839 | 16368 | 16793 | 16818 |
| edge-dit.cpp | q4_k | no-offload | none | 6431 | 6027 | 6384 | 6410 |
| edge-dit.cpp | q8_0 | no-offload | none | 10037 | 9701 | 9991 | 10016 |
