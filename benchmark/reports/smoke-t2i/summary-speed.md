# Speed summary (mean, unit ms)

> For inference speed look at **DiT sampling ms** (reliable); end-to-end includes loading / on-the-fly quantization conversion and is not comparable across systems. sd.cpp quantized tiers include on-the-fly convert in DiT, so it is inflated.


## flux-dev-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 37192.0 | 38752.0 | 934.3 | 617.8 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 10704.6 | 11455.0 | 272.0 | 473.1 |
| edge-dit.cpp | q8_0 | no-offload | none | 10565.3 | 11302.0 | 260.1 | 470.6 |

## flux-schnell-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 7631.1 | 8857.0 | 757.8 | 462.9 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 2305.2 | 3026.0 | 229.6 | 485.1 |
| edge-dit.cpp | q8_0 | no-offload | none | 2271.0 | 3010.0 | 258.7 | 474.9 |

## qwen-image-lightning-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 12288.2 | 14086.0 | 1031.4 | 760.4 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 2471.6 | 3797.0 | 569.1 | 750.8 |
| edge-dit.cpp | q4_k | no-offload | none | 2478.2 | 3229.0 | 225.0 | 520.1 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |

## qwen-image-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 91080.2 | 93094.0 | 1167.5 | 837.5 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 17511.4 | 18842.0 | 571.2 | 753.9 |
| edge-dit.cpp | q4_k | no-offload | none | 17510.4 | 18240.0 | 196.8 | 526.9 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 56646.8 | 57815.0 | 640.2 | 522.8 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |

## sd3-medium-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 3263.3 | 4054.0 | 366.0 | 418.2 |
| edge-dit.cpp | q4_k | no-offload | none | 3475.9 | 4247.0 | 352.3 | 412.6 |
| edge-dit.cpp | q8_0 | no-offload | none | 3500.8 | 4294.0 | 358.4 | 427.4 |

## sd35-medium-turbo-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 856.5 | 1595.0 | 316.5 | 414.3 |
| edge-dit.cpp | q4_k | no-offload | none | 931.3 | 1652.0 | 298.9 | 416.2 |
| edge-dit.cpp | q8_0 | no-offload | none | 941.7 | 1648.0 | 290.1 | 410.4 |
