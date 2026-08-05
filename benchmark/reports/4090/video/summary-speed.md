# Speed summary (mean, unit ms)

> For inference speed look at **DiT sampling ms** (reliable). End-to-end excludes model load (load-once boundary), but sd.cpp quantized tiers fold on-the-fly conversion into DiT, so those are inflated and not comparable across systems.


## wan2-t2v-1.3b-text-to-video

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 53707.8 | 56728.5 | 231.1 | 2649.6 |
| diffusers | w8 | no-offload | none | 56580.2 | 59598.0 | 227.1 | 2653.6 |
| edge-dit.cpp | f16 | no-offload | none | 49444.8 | 55423.7 | 322.0 | 4965.2 |
| edge-dit.cpp | q4_k | no-offload | none | 54042.6 | 60005.0 | 299.1 | 4962.0 |
| edge-dit.cpp | q8_0 | no-offload | none | 53963.7 | 59927.0 | 289.6 | 4970.8 |
| stable-diffusion.cpp | f16 | no-offload | none | 83423.3 | 105119.1 | 2263.3 | 18926.7 |
| stable-diffusion.cpp | q4_k | no-offload | none | 86783.3 | 125168.3 | 18866.7 | 19016.7 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 80383.3 | 111749.9 | 11866.7 | 18996.7 |

## wan21-t2v-1.3b-distill-text-to-video

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 6732.4 | 12594.0 | 194.8 | 4976.5 |
| edge-dit.cpp | q4_k | no-offload | none | 7394.2 | 13198.0 | 169.2 | 4958.5 |
| edge-dit.cpp | q8_0 | no-offload | none | 7384.4 | 13132.7 | 165.0 | 4913.4 |
