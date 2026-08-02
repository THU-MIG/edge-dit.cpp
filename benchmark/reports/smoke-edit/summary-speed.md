# Speed summary (mean, unit ms)

> For inference speed look at **DiT sampling ms** (reliable); end-to-end includes loading / on-the-fly quantization conversion and is not comparable across systems. sd.cpp quantized tiers include on-the-fly convert in DiT, so it is inflated.


## flux-kontext-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 50095.8 | 134757.4 | 1588.5 | 490.3 |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 24411.8 | 116583.7 | 675.5 | 428.0 |
| edge-dit.cpp | q8_0 | no-offload | none | 24097.9 | 38894.9 | 673.2 | 427.5 |

## kontext-lightning-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 20436.7 | 100884.9 | 1416.4 | 491.0 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 9957.9 | 112746.6 | 689.2 | 424.5 |
| edge-dit.cpp | q8_0 | no-offload | none | 9855.7 | 24256.3 | 675.2 | 426.8 |

## qwen-image-edit-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 112709.5 | 247394.5 | 2657.2 | 917.6 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 44283.9 | 229346.7 | 1456.7 | 472.3 |
| edge-dit.cpp | q4_k | no-offload | none | 44057.6 | 288073.6 | 1155.4 | 473.4 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 83460.7 | 134864.7 | 1578.7 | 621.9 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |

## qwen-image-edit-lightning-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 15713.1 | 154957.9 | 2497.0 | 1087.7 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 6107.5 | 197969.5 | 1446.9 | 469.0 |
| edge-dit.cpp | q4_k | no-offload | none | 5971.2 | 247480.1 | 1139.2 | 471.5 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |
