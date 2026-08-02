# Speed summary (mean, unit ms)

> For inference speed look at **DiT sampling ms** (reliable); end-to-end includes loading / on-the-fly quantization conversion and is not comparable across systems. sd.cpp quantized tiers include on-the-fly convert in DiT, so it is inflated.


## wan2-t2v-1.3b-text-to-video

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 32922.0 | 94438.9 | 354.1 | 5003.6 |
| edge-dit.cpp | q4_k | no-offload | none | 36169.0 | 107779.6 | 317.6 | 5101.6 |
| edge-dit.cpp | q8_0 | no-offload | none | 36119.5 | 52362.2 | 309.1 | 5000.3 |

## wan2-t2v-14b-text-to-video

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 238552.1 | 478163.6 | 1942.4 | 5342.9 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 199778.5 | 230415.0 | 1393.1 | 5254.0 |
| edge-dit.cpp | q4_k | no-offload | none | 198495.1 | 501570.8 | 319.1 | 5023.2 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | 199598.3 | 246207.8 | 1398.0 | 5270.7 |

## wan21-t2v-1.3b-distill-text-to-video

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 6857.2 | 31963.6 | 241.1 | 6541.6 |
| edge-dit.cpp | q4_k | no-offload | none | 7489.3 | 77779.6 | 187.6 | 5818.8 |
| edge-dit.cpp | q8_0 | no-offload | none | 7483.6 | 23839.7 | 184.1 | 4979.7 |
