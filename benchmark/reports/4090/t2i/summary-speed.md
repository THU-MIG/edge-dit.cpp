# Speed summary (mean, unit ms)

> For inference speed look at **DiT sampling ms** (reliable). End-to-end excludes model load (load-once boundary), but sd.cpp quantized tiers fold on-the-fly conversion into DiT, so those are inflated and not comparable across systems.


## flux-dev-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | 21901.0 | 45241.0 | 2903.8 | 20223.9 |
| diffusers | bf16 | no-offload | none | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 85007.5 | 88939.0 | 3256.1 | 648.5 |
| diffusers | w8 | no-offload | none | 13189.9 | 14138.9 | 242.1 | 684.4 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 39078.9 | 40513.0 | 993.5 | 436.1 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 10674.8 | 11334.0 | 224.4 | 429.8 |
| edge-dit.cpp | q8_0 | no-offload | none | 10569.3 | 11196.0 | 195.9 | 425.5 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 44723.3 | 54594.9 | 8583.3 | 1266.7 |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 53440.0 | 78524.4 | 23926.7 | 1140.0 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 17796.7 | 22193.6 | 3236.7 | 1143.3 |

## flux-schnell-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — |
| diffusers | w8 | no-offload | none | 2643.2 | 3411.9 | 274.8 | 465.9 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 7959.9 | 9579.0 | 1010.8 | 601.8 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 2241.5 | 2891.0 | 214.7 | 429.5 |
| edge-dit.cpp | q8_0 | no-offload | none | 2219.7 | 2876.0 | 223.2 | 427.8 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 19950.0 | 26977.9 | 5740.0 | 1270.0 |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 46116.7 | 71760.1 | 24486.7 | 1140.0 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 7383.3 | 11743.5 | 3200.0 | 1143.3 |

## qwen-image-lightning-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 14002.4 | 16438.7 | 1504.2 | 925.8 |
| edge-dit.cpp | bf16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 2423.8 | 3659.3 | 523.2 | 707.0 |
| edge-dit.cpp | q4_k | no-offload | none | 2438.1 | 3119.0 | 177.6 | 495.7 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |

## qwen-image-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 379936.8 | 387595.6 | 6209.3 | 1360.7 |
| diffusers | w8 | full offload | none | 54695.6 | 72269.9 | 8256.1 | 8919.5 |
| diffusers | w8 | no-offload | none | — | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 210591.5 | 213537.3 | 2185.7 | 754.2 |
| edge-dit.cpp | bf16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 35404.2 | 37114.3 | 999.5 | 705.3 |
| edge-dit.cpp | q4_k | no-offload | none | 35592.6 | 36268.7 | 199.5 | 471.3 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 129887.0 | 131534.0 | 1147.7 | 494.1 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | 182913.3 | 192831.5 | 6883.3 | 3020.0 |
| stable-diffusion.cpp | bf16 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 157523.3 | 203449.4 | 43636.7 | 2273.3 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 89720.0 | 102538.6 | 9786.7 | 3013.3 |
| stable-diffusion.cpp | q8_0 | no-offload | none | — | — | — | — |

## sd3-medium-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 3040.4 | 3577.0 | 243.4 | 269.1 |
| diffusers | w8 | no-offload | none | 3411.3 | 3923.3 | 243.4 | 250.0 |
| edge-dit.cpp | f16 | no-offload | none | 3258.0 | 3962.3 | 316.8 | 381.3 |
| edge-dit.cpp | q4_k | no-offload | none | 3435.6 | 4132.7 | 309.1 | 382.0 |
| edge-dit.cpp | q8_0 | no-offload | none | 3434.3 | 4131.3 | 309.4 | 381.4 |
| stable-diffusion.cpp | f16 | no-offload | none | 5193.3 | 7810.8 | 1493.3 | 1106.7 |
| stable-diffusion.cpp | q4_k | no-offload | none | 12360.0 | 37074.8 | 23586.7 | 1113.3 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 5086.7 | 9974.7 | 3760.0 | 1110.0 |

## sd35-medium-turbo-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 1716.5 | 2229.7 | 251.2 | 242.6 |
| diffusers | w8 | no-offload | none | 1889.4 | 2367.5 | 236.0 | 226.6 |
| edge-dit.cpp | f16 | no-offload | none | 1697.8 | 2408.3 | 319.4 | 385.1 |
| edge-dit.cpp | q4_k | no-offload | none | 1765.3 | 2468.7 | 314.4 | 381.4 |
| edge-dit.cpp | q8_0 | no-offload | none | 1767.0 | 2452.0 | 295.1 | 383.9 |
