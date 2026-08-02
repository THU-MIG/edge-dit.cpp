# Cross-system comparison matrix (all metrics, one-shot aggregate)

22 runs total | success 13 | failed 9

> **Speed boundary reminder**: to compare inference speed use "DiT sampling ms" (component-level denoise time, reliable). "end-to-end ms" includes one-time on-the-fly quantization conversion / model loading (see the "boundary" column: net-inference = excludes load/encoding, incl-load+encode = single CLI run), and must not be used for cross-system speed claims. Quantization quality loss (PSNR/SSIM/LPIPS vs FP16) is only meaningful within the same system vs its own FP16 baseline; not comparable across systems.

> **Special note for sd.cpp**: stable-diffusion.cpp loads layer-by-layer while sampling, and on-the-fly quantization conversion (q4_K/q8, tens to hundreds of seconds) folds into the denoise-stage timing, so its "DiT sampling ms" is likewise inflated under quantized tiers and does not represent pure inference. sd.cpp speed should be re-measured with pre-quantized weights, or only used as a same-tier trend reference; it cannot be compared directly with edge/diffusers DiT sampling.

> **The headline tier is q8** (usable image quality); q4 is only an extreme VRAM-saving reference point with obvious quality loss, and is not suitable for speed/quality advantage claims.


## flux-kontext-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | edit_01 | success | 50095.8 | 134757.4 | incl-load+encode | 1588.5 | 490.3 | 18930 | 10042 | 18930 | 1858 | -0.120 | 0.957 | 0.032 | 5.86 | -0.126 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(1)** | 50095.8 | 134757.4 |  | 1588.5 | 490.3 | 18930 | 10042 | 18930 | 1858 | -0.120 | 0.957 | 0.032 | 5.86 | -0.126 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | edit_01 | failed | — | — | process-level co | — | — | 23762 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | edit_01 | success | 24411.8 | 116583.7 | incl-load+encode | 675.5 | 428.0 | 12252 | 10818 | 12252 | 11004 | -0.104 | 0.958 | 0.029 | 5.89 | -0.127 | 45.28 | 0.996 | 0.001 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 24411.8 | 116583.7 |  | 675.5 | 428.0 | 12252 | 10818 | 12252 | 11004 | -0.104 | 0.958 | 0.029 | 5.89 | -0.127 | 45.28 | 0.996 | 0.001 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_01 | success | 24097.9 | 38894.9 | incl-load+encode | 673.2 | 427.5 | 20142 | 18708 | 20142 | 18894 | -0.131 | 0.957 | 0.032 | 5.86 | -0.126 | 54.61 | 0.998 | 0.000 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(1)** | 24097.9 | 38894.9 |  | 673.2 | 427.5 | 20142 | 18708 | 20142 | 18894 | -0.131 | 0.957 | 0.032 | 5.86 | -0.126 | 54.61 | 0.998 | 0.000 |

## kontext-lightning-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | edit_01 | success | 20436.7 | 100884.9 | incl-load+encode | 1416.4 | 491.0 | 18930 | 10042 | 18930 | 1858 | -0.133 | 0.954 | 0.034 | 5.81 | -0.125 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(1)** | 20436.7 | 100884.9 |  | 1416.4 | 491.0 | 18930 | 10042 | 18930 | 1858 | -0.133 | 0.954 | 0.034 | 5.81 | -0.125 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 23738 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | edit_01 | failed | — | — | process-level co | — | — | 23532 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | edit_01 | success | 9957.9 | 112746.6 | incl-load+encode | 689.2 | 424.5 | 12252 | 10818 | 12252 | 11004 | -0.110 | 0.955 | 0.030 | 5.83 | -0.145 | 45.41 | 0.996 | 0.002 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 9957.9 | 112746.6 |  | 689.2 | 424.5 | 12252 | 10818 | 12252 | 11004 | -0.110 | 0.955 | 0.030 | 5.83 | -0.145 | 45.41 | 0.996 | 0.002 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_01 | success | 9855.7 | 24256.3 | incl-load+encode | 675.2 | 426.8 | 20142 | 18708 | 20142 | 18894 | -0.138 | 0.954 | 0.034 | 5.79 | -0.142 | 55.26 | 0.999 | 0.000 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(1)** | 9855.7 | 24256.3 |  | 675.2 | 426.8 | 20142 | 18708 | 20142 | 18894 | -0.138 | 0.954 | 0.034 | 5.79 | -0.142 | 55.26 | 0.999 | 0.000 |

## qwen-image-edit-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | edit_01 | success | 112709.5 | 247394.5 | incl-load+encode | 2657.2 | 917.6 | 19232 | 15528 | 19232 | 1838 | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(1)** | 112709.5 | 247394.5 |  | 2657.2 | 917.6 | 19232 | 15528 | 19232 | 1838 | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 556 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | edit_01 | failed | — | — | process-level co | — | — | 556 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | edit_01 | success | 44283.9 | 229346.7 | incl-load+encode | 1456.7 | 472.3 | 18810 | 18810 | 13318 | 12420 | -0.103 | 0.528 | 0.671 | 5.65 | -0.740 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(1)** | 44283.9 | 229346.7 |  | 1456.7 | 472.3 | 18810 | 18810 | 13318 | 12420 | -0.103 | 0.528 | 0.671 | 5.65 | -0.740 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | edit_01 | success | 44057.6 | 288073.6 | incl-load+encode | 1155.4 | 473.4 | 19440 | 18772 | 19440 | 18542 | -0.160 | 0.763 | 0.311 | 5.53 | -0.560 | 10.30 | 0.630 | 0.543 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 44057.6 | 288073.6 |  | 1155.4 | 473.4 | 19440 | 18772 | 19440 | 18542 | -0.160 | 0.763 | 0.311 | 5.53 | -0.560 | 10.30 | 0.630 | 0.543 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | edit_01 | success | 83460.7 | 134864.7 | incl-load+encode | 1578.7 | 621.9 | 16184 | 8976 | 16184 | 1660 | -0.065 | 0.693 | 0.370 | 5.99 | -0.252 | 10.10 | 0.569 | 0.652 |
| **edge-dit.cpp** | **q8_0** | **DiT offload + te offload (max-vram 20g) (auto-allocate)** | **none** | **mean** | **(1)** | 83460.7 | 134864.7 |  | 1578.7 | 621.9 | 16184 | 8976 | 16184 | 1660 | -0.065 | 0.693 | 0.370 | 5.99 | -0.252 | 10.10 | 0.569 | 0.652 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 21534 | — | — | — | — | — | — | — | — | — | — | — |

## qwen-image-edit-lightning-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | edit_01 | success | 15713.1 | 154957.9 | incl-load+encode | 2497.0 | 1087.7 | 19232 | 15522 | 19232 | 1838 | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(1)** | 15713.1 | 154957.9 |  | 2497.0 | 1087.7 | 19232 | 15522 | 19232 | 1838 | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 556 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | edit_01 | failed | — | — | process-level co | — | — | 556 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | edit_01 | success | 6107.5 | 197969.5 | incl-load+encode | 1446.9 | 469.0 | 18876 | 18876 | 13318 | 12420 | -0.145 | 0.614 | 0.520 | 5.76 | -0.137 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(1)** | 6107.5 | 197969.5 |  | 1446.9 | 469.0 | 18876 | 18876 | 13318 | 12420 | -0.145 | 0.614 | 0.520 | 5.76 | -0.137 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | edit_01 | success | 5971.2 | 247480.1 | incl-load+encode | 1139.2 | 471.5 | 19440 | 18772 | 19440 | 18542 | -0.107 | 0.714 | 0.363 | 5.57 | -0.366 | 15.13 | 0.757 | 0.322 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 5971.2 | 247480.1 |  | 1139.2 | 471.5 | 19440 | 18772 | 19440 | 18542 | -0.107 | 0.714 | 0.363 | 5.57 | -0.366 | 15.13 | 0.757 | 0.322 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 21532 | — | — | — | — | — | — | — | — | — | — | — |