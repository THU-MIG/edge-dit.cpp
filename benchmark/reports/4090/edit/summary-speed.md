# Speed summary (mean, unit ms)

> For inference speed look at **DiT sampling ms** (reliable). End-to-end excludes model load (load-once boundary), but sd.cpp quantized tiers fold on-the-fly conversion into DiT, so those are inflated and not comparable across systems.


## flux-kontext-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 109252.8 | 112697.3 | 3014.8 | 406.6 |
| diffusers | w8 | no-offload | none | 27945.3 | 28703.5 | 495.3 | 238.8 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 55107.3 | 56985.0 | 1408.7 | 464.7 |
| edge-dit.cpp | q4_k | no-offload | none | 24816.3 | 25794.3 | 578.4 | 394.6 |
| edge-dit.cpp | q8_0 | no-offload | none | 24533.7 | 25509.7 | 576.5 | 394.0 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 67350.0 | 75191.2 | 5713.3 | 1216.7 |
| stable-diffusion.cpp | q4_k | no-offload | none | 79403.3 | 105639.8 | 24216.7 | 1123.3 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 39133.3 | 44414.5 | 3280.0 | 1123.3 |

## kontext-lightning-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 21133.6 | 22894.3 | 1294.4 | 461.8 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 10008.1 | 10983.3 | 577.9 | 391.6 |
| edge-dit.cpp | q8_0 | no-offload | none | 9910.0 | 10887.0 | 576.4 | 394.7 |

## qwen-image-edit-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 400773.8 | 410002.5 | 8721.1 | 482.4 |
| diffusers | w8 | no-offload | none | — | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 290034.8 | 302167.7 | 11152.2 | 936.9 |
| edge-dit.cpp | bf16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 103734.5 | 110068.0 | 5460.8 | 844.8 |
| edge-dit.cpp | q4_k | no-offload | none | 103773.1 | 107248.7 | 3021.6 | 423.3 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 199118.6 | 204793.0 | 5205.4 | 437.8 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | 148906.7 | 160445.8 | 6906.7 | 3036.7 |
| stable-diffusion.cpp | bf16 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 145636.7 | 192568.5 | 43300.0 | 2246.7 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 73316.7 | 88140.6 | 10223.3 | 3003.3 |
| stable-diffusion.cpp | q8_0 | no-offload | none | — | — | — | — |

## qwen-image-edit-lightning-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 18674.0 | 25445.3 | 5809.4 | 926.3 |
| edge-dit.cpp | bf16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 6912.2 | 11196.0 | 3407.0 | 845.9 |
| edge-dit.cpp | q4_k | no-offload | none | 6863.5 | 9349.3 | 2025.2 | 431.0 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |
