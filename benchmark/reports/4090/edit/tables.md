# Cross-system comparison matrix (all metrics, one-shot aggregate)

96 runs total | success 66 | failed 30

> **Speed boundary reminder**: to compare inference speed use "DiT sampling ms" (component-level denoise time, reliable). "end-to-end ms" excludes model load (load-once boundary; see the "boundary" column: net-inference = excludes load/encoding), but sd.cpp quantized tiers fold one-time on-the-fly conversion into the denoise stage, so their end-to-end (and DiT) is inflated and must not be used for cross-system speed claims. Quantization quality loss (PSNR/SSIM/LPIPS vs FP16) is only meaningful within the same system vs its own FP16 baseline; not comparable across systems.

> **Special note for sd.cpp**: stable-diffusion.cpp loads layer-by-layer while sampling, and on-the-fly quantization conversion (q4_K/q8, tens to hundreds of seconds) folds into the denoise-stage timing, so its "DiT sampling ms" is likewise inflated under quantized tiers and does not represent pure inference. sd.cpp speed should be re-measured with pre-quantized weights, or only used as a same-tier trend reference; it cannot be compared directly with edge/diffusers DiT sampling.

> **The headline tier is q8** (usable image quality); q4 is only an extreme VRAM-saving reference point with obvious quality loss, and is not suitable for speed/quality advantage claims.


## flux-kontext-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | edit_01 | failed | — | — | process-level co | — | — | 23978 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | full offload | none | edit_02 | failed | — | — | process-level co | — | — | 23978 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | full offload | none | edit_03 | failed | — | — | process-level co | — | — | 23978 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | edit_01 | success | 114648.4 | 118189.6 | net-inference | 3128.3 | 386.0 | 2326 | 752 | 1756 | 1172 | 0.193 | 0.640 | 0.386 | 5.78 | 0.561 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | edit_02 | success | 107436.6 | 110809.9 | net-inference | 2932.7 | 418.2 | 1756 | 844 | 1756 | 1174 | 0.044 | 0.572 | 0.722 | 5.27 | -1.045 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | edit_03 | success | 105673.5 | 109092.4 | net-inference | 2983.5 | 415.4 | 2070 | 2060 | 1756 | 2070 | 0.074 | 0.480 | 0.752 | 6.43 | -0.746 | — | — | — |
| **diffusers** | **bf16** | **sequential (full offload)** | **none** | **mean** | **(3)** | 109252.8 | 112697.3 |  | 3014.8 | 406.6 | 2051 | 1219 | 1756 | 1472 | 0.104 | 0.564 | 0.620 | 5.83 | -0.410 | — | — | — |
| diffusers | w8 | no-offload | none | edit_01 | success | 28031.6 | 28478.1 | net-inference | 233.7 | 192.1 | 24040 | 23782 | 23782 | 23782 | 0.192 | 0.641 | 0.386 | 5.83 | 0.576 | 39.02 | 0.985 | 0.005 |
| diffusers | w8 | no-offload | none | edit_02 | success | 27920.2 | 28824.2 | net-inference | 613.1 | 261.4 | 23782 | 22702 | 23740 | 23018 | 0.043 | 0.572 | 0.722 | 5.29 | -1.079 | 47.92 | 0.997 | 0.002 |
| diffusers | w8 | no-offload | none | edit_03 | success | 27884.1 | 28808.2 | net-inference | 639.1 | 263.1 | 23782 | 22190 | 23740 | 23740 | 0.065 | 0.480 | 0.752 | 6.36 | -0.742 | 28.39 | 0.936 | 0.025 |
| **diffusers** | **w8** | **no-offload** | **none** | **mean** | **(3)** | 27945.3 | 28703.5 |  | 495.3 | 238.8 | 23868 | 22891 | 23754 | 23513 | 0.100 | 0.564 | 0.620 | 5.83 | -0.415 | 38.44 | 0.972 | 0.011 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | edit_01 | success | 56183.5 | 57753.0 | net-inference | 1166.6 | 399.5 | 19178 | 2284 | 19178 | 1750 | 0.166 | 0.662 | 0.369 | 5.88 | 0.439 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | edit_02 | success | 56446.7 | 58522.0 | net-inference | 1573.3 | 497.2 | 19046 | 1120 | 19046 | 1750 | 0.073 | 0.574 | 0.727 | 5.27 | -1.031 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | edit_03 | success | 52691.6 | 54680.0 | net-inference | 1486.2 | 497.3 | 19046 | 1488 | 19046 | 1750 | 0.081 | 0.481 | 0.749 | 6.38 | -0.779 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 55107.3 | 56985.0 |  | 1408.7 | 464.7 | 19090 | 1631 | 19090 | 1750 | 0.107 | 0.572 | 0.615 | 5.85 | -0.457 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | edit_01 | success | 24666.1 | 25460.0 | net-inference | 459.4 | 330.2 | 12308 | 11046 | 12308 | 10874 | 0.169 | 0.674 | 0.355 | 5.92 | 0.508 | 31.79 | 0.946 | 0.027 |
| edge-dit.cpp | q4_k | no-offload | none | edit_02 | success | 24907.2 | 25976.0 | net-inference | 634.5 | 428.7 | 12178 | 10680 | 12178 | 10874 | 0.063 | 0.577 | 0.723 | 5.29 | -1.028 | 27.93 | 0.942 | 0.054 |
| edge-dit.cpp | q4_k | no-offload | none | edit_03 | success | 24875.5 | 25947.0 | net-inference | 641.2 | 424.8 | 12178 | 10680 | 12178 | 10874 | 0.077 | 0.486 | 0.750 | 6.39 | -0.826 | 26.88 | 0.864 | 0.073 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 24816.3 | 25794.3 |  | 578.4 | 394.6 | 12221 | 10802 | 12221 | 10874 | 0.103 | 0.579 | 0.609 | 5.87 | -0.449 | 28.87 | 0.917 | 0.051 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_01 | success | 24369.9 | 25168.0 | net-inference | 462.9 | 330.6 | 20198 | 18936 | 20198 | 18764 | 0.173 | 0.661 | 0.371 | 5.86 | 0.438 | 31.66 | 0.980 | 0.008 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_02 | success | 24628.1 | 25696.0 | net-inference | 635.2 | 426.4 | 20068 | 18570 | 20068 | 18764 | 0.077 | 0.574 | 0.727 | 5.26 | -1.019 | 35.14 | 0.984 | 0.008 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_03 | success | 24603.1 | 25665.0 | net-inference | 631.3 | 425.0 | 20068 | 18570 | 20068 | 18764 | 0.082 | 0.481 | 0.750 | 6.40 | -0.763 | 43.31 | 0.992 | 0.003 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 24533.7 | 25509.7 |  | 576.5 | 394.0 | 20111 | 18692 | 20111 | 18764 | 0.110 | 0.572 | 0.616 | 5.84 | -0.448 | 36.71 | 0.985 | 0.006 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | edit_01 | success | 53290.0 | 55637.6 | net-inference | 470.0 | 1140.0 | 17810 | 10182 | 17810 | 1516 | 0.153 | 0.630 | 0.399 | 5.77 | 0.519 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | edit_02 | success | 75420.0 | 86004.8 | net-inference | 8330.0 | 1260.0 | 17714 | 9664 | 17714 | 1516 | 0.067 | 0.567 | 0.729 | 5.24 | -0.909 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | edit_03 | success | 73340.0 | 83931.2 | net-inference | 8340.0 | 1250.0 | 17714 | 9664 | 17714 | 1516 | 0.095 | 0.443 | 0.771 | 5.99 | -0.635 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 67350.0 | 75191.2 |  | 5713.3 | 1216.7 | 17746 | 9837 | 17746 | 1516 | 0.105 | 0.547 | 0.633 | 5.67 | -0.342 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | edit_01 | success | 34740.0 | 36580.5 | net-inference | 30.0 | 1090.0 | 11960 | — | 11960 | 10872 | 0.165 | 0.648 | 0.379 | 5.81 | 0.550 | 25.64 | 0.863 | 0.065 |
| stable-diffusion.cpp | q4_k | no-offload | none | edit_02 | success | 96680.0 | 134116.1 | net-inference | 35310.0 | 1140.0 | 11864 | 3788 | 11864 | 10872 | 0.050 | 0.571 | 0.725 | 5.30 | -0.940 | 24.95 | 0.921 | 0.074 |
| stable-diffusion.cpp | q4_k | no-offload | none | edit_03 | success | 106790.0 | 146222.7 | net-inference | 37310.0 | 1140.0 | 11864 | 3788 | 11864 | 10872 | 0.070 | 0.441 | 0.767 | 6.24 | -0.784 | 25.28 | 0.868 | 0.096 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 79403.3 | 105639.8 |  | 24216.7 | 1123.3 | 11896 | 3788 | 11896 | 10872 | 0.095 | 0.554 | 0.624 | 5.78 | -0.391 | 25.29 | 0.884 | 0.078 |
| stable-diffusion.cpp | q8_0 | no-offload | none | edit_01 | success | 34250.0 | 36084.2 | net-inference | 30.0 | 1090.0 | 19482 | — | 19482 | 18394 | 0.159 | 0.627 | 0.400 | 5.80 | 0.514 | 37.60 | 0.980 | 0.008 |
| stable-diffusion.cpp | q8_0 | no-offload | none | edit_02 | success | 41980.0 | 49054.3 | net-inference | 4960.0 | 1140.0 | 19386 | 5668 | 19386 | 18394 | 0.060 | 0.567 | 0.729 | 5.22 | -0.947 | 37.88 | 0.989 | 0.005 |
| stable-diffusion.cpp | q8_0 | no-offload | none | edit_03 | success | 41170.0 | 48104.9 | net-inference | 4850.0 | 1140.0 | 19386 | 5668 | 19386 | 18394 | 0.093 | 0.442 | 0.774 | 6.01 | -0.644 | 39.66 | 0.991 | 0.006 |
| **stable-diffusion.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 39133.3 | 44414.5 |  | 3280.0 | 1123.3 | 19418 | 5668 | 19418 | 18394 | 0.104 | 0.545 | 0.635 | 5.68 | -0.359 | 38.38 | 0.987 | 0.007 |

## kontext-lightning-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | edit_01 | success | 20828.2 | 22311.0 | net-inference | 1081.6 | 397.7 | 19178 | 2016 | 19178 | 1750 | 0.127 | 0.642 | 0.370 | 5.89 | 0.544 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | edit_02 | success | 21381.5 | 23357.0 | net-inference | 1476.9 | 493.6 | 19046 | 1120 | 19046 | 1750 | 0.076 | 0.585 | 0.718 | 5.17 | -1.078 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | edit_03 | success | 21191.2 | 23015.0 | net-inference | 1324.7 | 494.1 | 19046 | 1120 | 19046 | 1750 | 0.082 | 0.487 | 0.729 | 6.36 | -0.759 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 21133.6 | 22894.3 |  | 1294.4 | 461.8 | 19090 | 1419 | 19090 | 1750 | 0.095 | 0.571 | 0.605 | 5.80 | -0.431 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 23600 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | edit_02 | failed | — | — | process-level co | — | — | 23600 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | edit_03 | failed | — | — | process-level co | — | — | 23600 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | edit_01 | success | 9829.8 | 10617.0 | net-inference | 454.7 | 327.5 | 12308 | 11046 | 12308 | 10874 | 0.136 | 0.657 | 0.359 | 5.94 | 0.495 | 30.69 | 0.947 | 0.028 |
| edge-dit.cpp | q4_k | no-offload | none | edit_02 | success | 10105.8 | 11170.0 | net-inference | 636.5 | 422.1 | 12178 | 10680 | 12178 | 10874 | 0.068 | 0.587 | 0.715 | 5.22 | -1.108 | 36.01 | 0.985 | 0.015 |
| edge-dit.cpp | q4_k | no-offload | none | edit_03 | success | 10088.8 | 11163.0 | net-inference | 642.6 | 425.2 | 12178 | 10680 | 12178 | 12178 | 0.084 | 0.474 | 0.729 | 6.46 | -0.696 | 25.56 | 0.838 | 0.097 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 10008.1 | 10983.3 |  | 577.9 | 391.6 | 12221 | 10802 | 12221 | 11309 | 0.096 | 0.573 | 0.601 | 5.87 | -0.436 | 30.75 | 0.923 | 0.047 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_01 | success | 9720.3 | 10519.0 | net-inference | 462.8 | 330.7 | 20198 | 18936 | 20198 | 18764 | 0.125 | 0.644 | 0.369 | 5.90 | 0.554 | 44.79 | 0.994 | 0.003 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_02 | success | 10007.7 | 11075.0 | net-inference | 635.8 | 425.0 | 20068 | 18570 | 20068 | 18764 | 0.074 | 0.585 | 0.718 | 5.16 | -1.108 | 43.94 | 0.995 | 0.003 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_03 | success | 10002.1 | 11067.0 | net-inference | 630.6 | 428.5 | 20068 | 18570 | 20068 | 18764 | 0.086 | 0.486 | 0.728 | 6.34 | -0.736 | 41.35 | 0.990 | 0.005 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 9910.0 | 10887.0 |  | 576.4 | 394.7 | 20111 | 18692 | 20111 | 18764 | 0.095 | 0.572 | 0.605 | 5.80 | -0.430 | 43.36 | 0.993 | 0.003 |

## qwen-image-edit-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 24050 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | edit_02 | failed | — | — | process-level co | — | — | 24050 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | edit_03 | failed | — | — | process-level co | — | — | 24050 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | edit_01 | success | 426666.0 | 438284.3 | net-inference | 11083.7 | 505.2 | 5612 | 5612 | 2508 | 2326 | 0.138 | 0.684 | 0.402 | 5.87 | 0.540 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | edit_02 | success | 409308.5 | 417299.1 | net-inference | 7490.2 | 476.5 | 4572 | 2626 | 1766 | 3972 | 0.088 | 0.625 | 0.639 | 5.34 | -1.239 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | edit_03 | success | 366346.9 | 374424.2 | net-inference | 7589.4 | 465.5 | 4748 | 2626 | 1766 | 4748 | 0.139 | 0.555 | 0.722 | 6.30 | -0.843 | — | — | — |
| **diffusers** | **bf16** | **sequential (full offload)** | **none** | **mean** | **(3)** | 400773.8 | 410002.5 |  | 8721.1 | 482.4 | 4977 | 3621 | 2013 | 3682 | 0.122 | 0.621 | 0.587 | 5.84 | -0.514 | — | — | — |
| diffusers | w8 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 24014 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | w8 | no-offload | none | edit_02 | failed | — | — | process-level co | — | — | 24014 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | w8 | no-offload | none | edit_03 | failed | — | — | process-level co | — | — | 24014 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | edit_01 | success | 275961.3 | 286998.0 | net-inference | 10081.6 | 911.7 | 20020 | 19354 | 20020 | 2866 | 0.175 | 0.695 | 0.388 | 6.02 | 0.588 | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | edit_02 | success | 297002.5 | 309833.0 | net-inference | 11854.3 | 933.4 | 19882 | 19216 | 19882 | 2728 | 0.102 | 0.624 | 0.637 | 5.29 | -1.172 | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | edit_03 | success | 297140.5 | 309672.0 | net-inference | 11520.8 | 965.7 | 19882 | 19216 | 19882 | 2728 | 0.139 | 0.559 | 0.720 | 6.26 | -1.028 | — | — | — |
| **edge-dit.cpp** | **bf16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 290034.8 | 302167.7 |  | 11152.2 | 936.9 | 19928 | 19262 | 19928 | 2774 | 0.139 | 0.626 | 0.582 | 5.85 | -0.537 | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | edit_02 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | edit_03 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | edit_01 | success | 103746.6 | 109812.0 | net-inference | 5253.3 | 789.6 | 18786 | 18786 | 13360 | 12456 | -0.022 | 0.736 | 0.363 | 5.58 | 0.554 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | edit_02 | success | 103706.9 | 110098.0 | net-inference | 5487.8 | 873.6 | 18260 | 18260 | 13184 | 12456 | 0.024 | 0.631 | 0.594 | 5.30 | -1.220 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | edit_03 | success | 103750.1 | 110294.0 | net-inference | 5641.4 | 871.2 | 17534 | 17534 | 13184 | 12456 | 0.091 | 0.503 | 0.745 | 6.33 | -0.924 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload + vae offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(3)** | 103734.5 | 110068.0 |  | 5460.8 | 844.8 | 18193 | 18193 | 13243 | 12456 | 0.031 | 0.623 | 0.567 | 5.74 | -0.530 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | edit_01 | success | 103839.3 | 107053.0 | net-inference | 2809.3 | 372.0 | 23588 | 23588 | 22928 | 21780 | 0.165 | 0.696 | 0.387 | 5.75 | 0.577 | 26.55 | 0.914 | 0.176 |
| edge-dit.cpp | q4_k | no-offload | none | edit_02 | success | 103690.5 | 107294.0 | net-inference | 3125.4 | 449.0 | 23116 | 23116 | 22750 | 21780 | 0.076 | 0.621 | 0.629 | 5.20 | -1.176 | 28.91 | 0.958 | 0.152 |
| edge-dit.cpp | q4_k | no-offload | none | edit_03 | success | 103789.4 | 107399.0 | net-inference | 3130.2 | 449.0 | 23116 | 23116 | 22750 | 21780 | 0.159 | 0.538 | 0.724 | 6.43 | -1.022 | 27.68 | 0.901 | 0.129 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 103773.1 | 107248.7 |  | 3021.6 | 423.3 | 23273 | 23273 | 22809 | 21780 | 0.133 | 0.619 | 0.580 | 5.79 | -0.540 | 27.71 | 0.924 | 0.152 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | edit_01 | success | 206383.7 | 211764.0 | net-inference | 4966.6 | 382.9 | 16226 | 12376 | 16226 | 1786 | 0.176 | 0.693 | 0.394 | 5.98 | 0.563 | 25.32 | 0.887 | 0.204 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | edit_02 | success | 195825.5 | 201601.0 | net-inference | 5279.0 | 465.5 | 16048 | 11566 | 16048 | 1786 | 0.101 | 0.623 | 0.638 | 5.28 | -1.172 | 29.13 | 0.944 | 0.170 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | edit_03 | success | 195146.7 | 201014.0 | net-inference | 5370.7 | 465.0 | 16048 | 11566 | 16048 | 1786 | 0.155 | 0.566 | 0.703 | 6.22 | -0.961 | 27.88 | 0.918 | 0.098 |
| **edge-dit.cpp** | **q8_0** | **DiT offload + te offload (max-vram 20g) (auto-allocate)** | **none** | **mean** | **(3)** | 199118.6 | 204793.0 |  | 5205.4 | 437.8 | 16107 | 11836 | 16107 | 1786 | 0.144 | 0.627 | 0.578 | 5.83 | -0.524 | 27.44 | 0.917 | 0.158 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 21396 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | edit_02 | failed | — | — | process-level co | — | — | 21396 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | edit_03 | failed | — | — | process-level co | — | — | 21396 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | edit_01 | success | 126600.0 | 132587.3 | net-inference | 1650.0 | 2950.0 | 16966 | 15230 | 16966 | 1310 | 0.210 | 0.683 | 0.450 | 5.91 | 0.166 | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | edit_02 | success | 160190.0 | 174722.1 | net-inference | 9780.0 | 3060.0 | 16966 | 15042 | 16966 | 1310 | 0.035 | 0.659 | 0.584 | 5.38 | -1.151 | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | edit_03 | success | 159930.0 | 174028.0 | net-inference | 9290.0 | 3100.0 | 16966 | 15052 | 16966 | 1310 | 0.145 | 0.684 | 0.595 | 6.04 | -0.935 | — | — | — |
| **stable-diffusion.cpp** | **bf16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 148906.7 | 160445.8 |  | 6906.7 | 3036.7 | 16966 | 15108 | 16966 | 1310 | 0.130 | 0.675 | 0.543 | 5.78 | -0.640 | — | — | — |
| stable-diffusion.cpp | bf16 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 15120 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | bf16 | no-offload | none | edit_02 | failed | — | — | process-level co | — | — | 15122 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | bf16 | no-offload | none | edit_03 | failed | — | — | process-level co | — | — | 15126 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | edit_01 | success | 32140.0 | 35726.1 | net-inference | 30.0 | 2210.0 | 19240 | — | 17918 | 17782 | 0.098 | 0.664 | 0.436 | 5.91 | 0.388 | 21.27 | 0.859 | 0.202 |
| stable-diffusion.cpp | q4_k | no-offload | none | edit_02 | success | 201050.0 | 269779.2 | net-inference | 65080.0 | 2270.0 | 17782 | 6168 | 17780 | 17782 | 0.018 | 0.651 | 0.568 | 5.47 | -1.120 | 18.15 | 0.728 | 0.459 |
| stable-diffusion.cpp | q4_k | no-offload | none | edit_03 | success | 203720.0 | 272200.2 | net-inference | 64790.0 | 2260.0 | 17782 | 6168 | 17780 | 17782 | 0.108 | 0.666 | 0.594 | 5.97 | -1.199 | 21.80 | 0.837 | 0.347 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 145636.7 | 192568.5 |  | 43300.0 | 2246.7 | 18268 | 6168 | 17826 | 17782 | 0.075 | 0.660 | 0.533 | 5.78 | -0.644 | 20.41 | 0.808 | 0.336 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | edit_01 | success | 57500.0 | 62695.2 | net-inference | 830.0 | 2960.0 | 18834 | 7756 | 18784 | 1198 | 0.218 | 0.683 | 0.451 | 5.83 | 0.132 | 27.20 | 0.960 | 0.036 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | edit_02 | success | 81170.0 | 100931.9 | net-inference | 15040.0 | 3030.0 | 18784 | 7700 | 18784 | 1198 | 0.032 | 0.657 | 0.587 | 5.50 | -1.153 | 32.98 | 0.979 | 0.022 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | edit_03 | success | 81280.0 | 100794.6 | net-inference | 14800.0 | 3020.0 | 18784 | 7700 | 18784 | 1198 | 0.147 | 0.685 | 0.595 | 6.05 | -0.992 | 33.96 | 0.975 | 0.037 |
| **stable-diffusion.cpp** | **q8_0** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 73316.7 | 88140.6 |  | 10223.3 | 3003.3 | 18801 | 7719 | 18784 | 1198 | 0.132 | 0.675 | 0.544 | 5.80 | -0.671 | 31.38 | 0.971 | 0.032 |
| stable-diffusion.cpp | q8_0 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 7802 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q8_0 | no-offload | none | edit_02 | failed | — | — | process-level co | — | — | 7752 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q8_0 | no-offload | none | edit_03 | failed | — | — | process-level co | — | — | 7800 | — | — | — | — | — | — | — | — | — | — | — |

## qwen-image-edit-lightning-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | edit_01 | success | 17438.1 | 23041.0 | net-inference | 4742.7 | 827.8 | 19808 | 19766 | 19808 | 2476 | 0.179 | 0.704 | 0.344 | 6.03 | 0.551 | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | edit_02 | success | 19084.9 | 25812.0 | net-inference | 5721.5 | 975.5 | 19630 | 19196 | 19630 | 2476 | 0.107 | 0.619 | 0.640 | 5.37 | -1.178 | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | edit_03 | success | 19498.9 | 27483.0 | net-inference | 6963.8 | 975.8 | 19630 | 19196 | 19630 | 2476 | 0.146 | 0.562 | 0.703 | 6.29 | -0.903 | — | — | — |
| **edge-dit.cpp** | **bf16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 18674.0 | 25445.3 |  | 5809.4 | 926.3 | 19689 | 19386 | 19689 | 2476 | 0.144 | 0.628 | 0.562 | 5.89 | -0.510 | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | edit_02 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | edit_03 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | edit_01 | success | 6759.0 | 10801.0 | net-inference | 3224.4 | 786.9 | 18646 | 18646 | 13220 | 12316 | -0.013 | 0.720 | 0.327 | 5.51 | 0.365 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | edit_02 | success | 6988.9 | 11390.0 | net-inference | 3493.9 | 876.6 | 18260 | 18260 | 13048 | 12316 | -0.005 | 0.615 | 0.611 | 5.22 | -1.300 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | edit_03 | success | 6988.7 | 11397.0 | net-inference | 3502.7 | 874.3 | 18260 | 18260 | 13048 | 12316 | 0.076 | 0.516 | 0.703 | 6.24 | -0.887 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload + vae offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(3)** | 6912.2 | 11196.0 |  | 3407.0 | 845.9 | 18389 | 18389 | 13105 | 12316 | 0.019 | 0.617 | 0.547 | 5.66 | -0.607 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | edit_01 | success | 6777.7 | 8947.0 | net-inference | 1772.8 | 369.9 | 23468 | 23468 | 22808 | 21660 | 0.055 | 0.708 | 0.316 | 5.72 | -0.266 | 24.12 | 0.906 | 0.137 |
| edge-dit.cpp | q4_k | no-offload | none | edit_02 | success | 6924.4 | 9601.0 | net-inference | 2178.3 | 468.0 | 23116 | 23116 | 22634 | 21660 | 0.063 | 0.618 | 0.630 | 5.35 | -1.139 | 33.61 | 0.963 | 0.078 |
| edge-dit.cpp | q4_k | no-offload | none | edit_03 | success | 6888.4 | 9500.0 | net-inference | 2124.6 | 455.1 | 23116 | 23116 | 22634 | 21660 | 0.070 | 0.548 | 0.694 | 6.29 | -0.806 | 29.83 | 0.859 | 0.104 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 6863.5 | 9349.3 |  | 2025.2 | 431.0 | 23233 | 23233 | 22692 | 21660 | 0.063 | 0.624 | 0.547 | 5.79 | -0.737 | 29.19 | 0.909 | 0.107 |
| edge-dit.cpp | q8_0 | no-offload | none | edit_01 | failed | — | — | process-level co | — | — | 21396 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | edit_02 | failed | — | — | process-level co | — | — | 21396 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | edit_03 | failed | — | — | process-level co | — | — | 21396 | — | — | — | — | — | — | — | — | — | — | — |