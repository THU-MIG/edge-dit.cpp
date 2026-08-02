# Summary table (mean, core columns)

> One table at a glance, split by task (quality columns differ per task). For speed look at DiT sampling ms; VRAM unit MiB; PSNR/SSIM/LPIPS are quantization vs same-system FP16.


## flux-kontext-image-editing  (image-editing)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 50095.8 | 134757.4 | 18930 | -0.120 | 0.957 | 0.032 | 5.86 | -0.126 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 23762 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 24411.8 | 116583.7 | 12252 | -0.104 | 0.958 | 0.029 | 5.89 | -0.127 | 45.28 | 0.996 | 0.001 |
| edge-dit.cpp | q8_0 | no-offload | none | 24097.9 | 38894.9 | 20142 | -0.131 | 0.957 | 0.032 | 5.86 | -0.126 | 54.61 | 0.998 | 0.000 |

## kontext-lightning-image-editing  (image-editing)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 20436.7 | 100884.9 | 18930 | -0.133 | 0.954 | 0.034 | 5.81 | -0.125 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 23738 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 23532 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 9957.9 | 112746.6 | 12252 | -0.110 | 0.955 | 0.030 | 5.83 | -0.145 | 45.41 | 0.996 | 0.002 |
| edge-dit.cpp | q8_0 | no-offload | none | 9855.7 | 24256.3 | 20142 | -0.138 | 0.954 | 0.034 | 5.79 | -0.142 | 55.26 | 0.999 | 0.000 |

## qwen-image-edit-image-editing  (image-editing)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 112709.5 | 247394.5 | 19232 | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 556 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 556 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 44283.9 | 229346.7 | 18810 | -0.103 | 0.528 | 0.671 | 5.65 | -0.740 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 44057.6 | 288073.6 | 19440 | -0.160 | 0.763 | 0.311 | 5.53 | -0.560 | 10.30 | 0.630 | 0.543 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 83460.7 | 134864.7 | 16184 | -0.065 | 0.693 | 0.370 | 5.99 | -0.252 | 10.10 | 0.569 | 0.652 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 21534 | — | — | — | — | — | — | — | — |

## qwen-image-edit-lightning-image-editing  (image-editing)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 15713.1 | 154957.9 | 19232 | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 556 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 556 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 6107.5 | 197969.5 | 18876 | -0.145 | 0.614 | 0.520 | 5.76 | -0.137 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 5971.2 | 247480.1 | 19440 | -0.107 | 0.714 | 0.363 | 5.57 | -0.366 | 15.13 | 0.757 | 0.322 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 21532 | — | — | — | — | — | — | — | — |
