# Cross-system comparison matrix (all metrics, one-shot aggregate)

165 runs total | success 123 | failed 42

> **Speed boundary reminder**: to compare inference speed use "DiT sampling ms" (component-level denoise time, reliable). "end-to-end ms" excludes model load (load-once boundary; see the "boundary" column: net-inference = excludes load/encoding), but sd.cpp quantized tiers fold one-time on-the-fly conversion into the denoise stage, so their end-to-end (and DiT) is inflated and must not be used for cross-system speed claims. Quantization quality loss (PSNR/SSIM/LPIPS vs FP16) is only meaningful within the same system vs its own FP16 baseline; not comparable across systems.

> **Special note for sd.cpp**: stable-diffusion.cpp loads layer-by-layer while sampling, and on-the-fly quantization conversion (q4_K/q8, tens to hundreds of seconds) folds into the denoise-stage timing, so its "DiT sampling ms" is likewise inflated under quantized tiers and does not represent pure inference. sd.cpp speed should be re-measured with pre-quantized weights, or only used as a same-tier trend reference; it cannot be compared directly with edge/diffusers DiT sampling.

> **The headline tier is q8** (usable image quality); q4 is only an extreme VRAM-saving reference point with obvious quality loss, and is not suitable for speed/quality advantage claims.


## flux-dev-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | t2i_01 | success | 22229.0 | 45805.3 | net-inference | 3211.1 | 20196.9 | 24028 | 9452 | 23930 | 23890 | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | bf16 | full offload | none | t2i_02 | success | 21650.7 | 44860.8 | net-inference | 2724.0 | 20250.6 | 23932 | 9602 | 23932 | 23880 | 0.296 | 6.09 | 1.813 | — | — | — |
| diffusers | bf16 | full offload | none | t2i_03 | success | 21823.3 | 45056.9 | net-inference | 2776.3 | 20224.1 | 23920 | 9602 | 23920 | 23880 | 0.321 | 6.27 | 1.536 | — | — | — |
| **diffusers** | **bf16** | **full offload** | **none** | **mean** | **(3)** | 21901.0 | 45241.0 |  | 2903.8 | 20223.9 | 23960 | 9552 | 23927 | 23883 | 0.308 | 6.14 | 1.737 | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 24042 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 24042 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 24042 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | t2i_01 | success | 86628.2 | 92458.7 | net-inference | 5403.3 | 405.9 | 2580 | 750 | 1298 | 2580 | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | t2i_02 | success | 83770.4 | 86748.5 | net-inference | 2197.1 | 752.2 | 1318 | 730 | 1290 | 1172 | 0.296 | 6.09 | 1.813 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | t2i_03 | success | 84624.0 | 87609.7 | net-inference | 2167.9 | 787.2 | 1554 | 730 | 1290 | 1554 | 0.321 | 6.27 | 1.536 | — | — | — |
| **diffusers** | **bf16** | **sequential (full offload)** | **none** | **mean** | **(3)** | 85007.5 | 88939.0 |  | 3256.1 | 648.5 | 1817 | 737 | 1293 | 1769 | 0.308 | 6.14 | 1.737 | — | — | — |
| diffusers | w8 | no-offload | none | t2i_01 | success | 13210.0 | 13505.0 | net-inference | 67.7 | 203.3 | 24038 | — | 23784 | 23784 | 0.305 | 6.02 | 1.851 | 30.91 | 0.964 | 0.031 |
| diffusers | w8 | no-offload | none | t2i_02 | success | 13197.4 | 14050.6 | net-inference | 331.2 | 497.8 | 23780 | 21932 | 22660 | 23404 | 0.295 | 6.06 | 1.827 | 29.90 | 0.972 | 0.024 |
| diffusers | w8 | no-offload | none | t2i_03 | success | 13162.1 | 14861.0 | net-inference | 327.3 | 1351.9 | 23780 | 21932 | 22660 | 23026 | 0.319 | 6.28 | 1.526 | 31.63 | 0.975 | 0.016 |
| **diffusers** | **w8** | **no-offload** | **none** | **mean** | **(3)** | 13189.9 | 14138.9 |  | 242.1 | 684.4 | 23866 | 21932 | 23035 | 23405 | 0.307 | 6.12 | 1.735 | 30.81 | 0.970 | 0.024 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | t2i_01 | success | 38594.7 | 39892.0 | net-inference | 922.5 | 371.0 | 19698 | 1918 | 19698 | 1320 | 0.300 | 6.07 | 1.809 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | t2i_02 | success | 38862.4 | 40454.0 | net-inference | 1118.6 | 467.7 | 19566 | 946 | 19566 | 1320 | 0.293 | 5.94 | 1.838 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | t2i_03 | success | 39779.6 | 41193.0 | net-inference | 939.3 | 469.4 | 19566 | 1314 | 19566 | 1320 | 0.312 | 6.32 | 1.100 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 39078.9 | 40513.0 |  | 993.5 | 436.1 | 19610 | 1393 | 19610 | 1320 | 0.302 | 6.11 | 1.582 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 23536 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 23536 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 23536 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 10545.7 | 11009.0 | net-inference | 127.3 | 331.0 | 11310 | — | 11310 | 10682 | 0.299 | 6.07 | 1.809 | 20.28 | 0.820 | 0.215 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_02 | success | 10749.6 | 11508.0 | net-inference | 272.8 | 480.6 | 11178 | 10248 | 11178 | 10682 | 0.295 | 5.81 | 1.876 | 15.63 | 0.672 | 0.460 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_03 | success | 10729.2 | 11485.0 | net-inference | 273.1 | 477.7 | 11178 | 10230 | 11178 | 10682 | 0.317 | 6.21 | 1.149 | 28.46 | 0.932 | 0.053 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 10674.8 | 11334.0 |  | 224.4 | 429.8 | 11222 | 10239 | 11222 | 10682 | 0.304 | 6.03 | 1.611 | 21.45 | 0.808 | 0.243 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | success | 10449.2 | 10909.0 | net-inference | 128.4 | 326.1 | 19200 | — | 19200 | 18572 | 0.283 | 5.92 | 1.795 | 16.50 | 0.714 | 0.361 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_02 | success | 10631.1 | 11339.0 | net-inference | 225.9 | 476.8 | 19068 | 18132 | 19068 | 18572 | 0.294 | 5.99 | 1.823 | 37.76 | 0.990 | 0.007 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_03 | success | 10627.6 | 11340.0 | net-inference | 233.5 | 473.5 | 19068 | 18120 | 19068 | 18572 | 0.312 | 6.29 | 1.273 | 31.69 | 0.971 | 0.023 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 10569.3 | 11196.0 |  | 195.9 | 425.5 | 19112 | 18126 | 19112 | 18572 | 0.296 | 6.07 | 1.630 | 28.65 | 0.892 | 0.130 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | t2i_01 | success | 30700.0 | 32334.4 | net-inference | 470.0 | 1140.0 | 17592 | 9924 | 17592 | 1258 | 0.304 | 5.83 | 1.649 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | t2i_02 | success | 55030.0 | 73382.6 | net-inference | 16930.0 | 1400.0 | 17496 | 9536 | 17496 | 1258 | 0.295 | 5.98 | 1.652 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | t2i_03 | success | 48440.0 | 58067.6 | net-inference | 8350.0 | 1260.0 | 17496 | 9536 | 17496 | 1258 | 0.328 | 6.38 | 1.126 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 44723.3 | 54594.9 |  | 8583.3 | 1266.7 | 17528 | 9665 | 17528 | 1258 | 0.309 | 6.06 | 1.476 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 9908 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 9832 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 9838 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_01 | success | 13360.0 | 14503.5 | net-inference | 30.0 | 1100.0 | 11102 | — | 11102 | 10738 | 0.303 | 5.72 | 1.611 | 25.94 | 0.942 | 0.066 |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_02 | success | 74810.0 | 111846.2 | net-inference | 35860.0 | 1160.0 | 11004 | 3662 | 11004 | 10738 | 0.301 | 5.90 | 1.772 | 16.76 | 0.745 | 0.310 |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_03 | success | 72150.0 | 109223.6 | net-inference | 35890.0 | 1160.0 | 11004 | 3662 | 11004 | 10738 | 0.327 | 6.53 | 1.192 | 27.85 | 0.948 | 0.053 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 53440.0 | 78524.4 |  | 23926.7 | 1140.0 | 11037 | 3662 | 11037 | 10738 | 0.311 | 6.05 | 1.525 | 23.52 | 0.878 | 0.143 |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_01 | success | 13150.0 | 14294.0 | net-inference | 30.0 | 1100.0 | 18624 | — | 18624 | 18260 | 0.301 | 5.90 | 1.677 | 26.46 | 0.954 | 0.052 |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_02 | success | 20180.0 | 26326.7 | net-inference | 4960.0 | 1170.0 | 18526 | 5542 | 18526 | 18260 | 0.298 | 6.00 | 1.751 | 25.34 | 0.942 | 0.065 |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_03 | success | 20060.0 | 25960.2 | net-inference | 4720.0 | 1160.0 | 18526 | 5542 | 18526 | 18260 | 0.330 | 6.39 | 1.147 | 35.20 | 0.989 | 0.011 |
| **stable-diffusion.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 17796.7 | 22193.6 |  | 3236.7 | 1143.3 | 18559 | 5542 | 18559 | 18260 | 0.310 | 6.10 | 1.525 | 29.00 | 0.962 | 0.043 |

## flux-schnell-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 24024 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 24024 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 24024 | — | — | — | — | — | — | — | — | — |
| diffusers | w8 | no-offload | none | t2i_01 | success | 2644.7 | 3418.8 | net-inference | 270.5 | 469.2 | 23896 | 21836 | 22664 | 23138 | 0.303 | 5.64 | 1.758 | — | — | — |
| diffusers | w8 | no-offload | none | t2i_02 | success | 2638.4 | 3428.4 | net-inference | 283.0 | 479.1 | 23758 | 21910 | 22526 | 23382 | 0.300 | 5.86 | 1.877 | — | — | — |
| diffusers | w8 | no-offload | none | t2i_03 | success | 2646.4 | 3388.3 | net-inference | 270.9 | 449.3 | 23954 | 21750 | 22526 | 23954 | 0.317 | 6.09 | 1.156 | — | — | — |
| **diffusers** | **w8** | **no-offload** | **none** | **mean** | **(3)** | 2643.2 | 3411.9 |  | 274.8 | 465.9 | 23869 | 21832 | 22572 | 23491 | 0.306 | 5.86 | 1.597 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | t2i_01 | success | 7706.0 | 8907.0 | net-inference | 819.8 | 376.4 | 19676 | 1852 | 19676 | 1318 | 0.318 | 5.43 | 1.874 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | t2i_02 | success | 8180.3 | 10147.0 | net-inference | 1118.9 | 840.6 | 19544 | 946 | 19544 | 1318 | 0.338 | 6.16 | 1.091 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | t2i_03 | success | 7993.4 | 9683.0 | net-inference | 1093.6 | 588.5 | 19544 | 1314 | 19544 | 1318 | 0.338 | 6.14 | 1.691 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 7959.9 | 9579.0 |  | 1010.8 | 601.8 | 19588 | 1371 | 19588 | 1318 | 0.331 | 5.91 | 1.552 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 23516 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 23516 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 23516 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 2118.3 | 2582.0 | net-inference | 127.7 | 331.2 | 11290 | — | 11290 | 10662 | 0.312 | 5.82 | 1.796 | 14.62 | 0.688 | 0.304 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_02 | success | 2306.9 | 3076.0 | net-inference | 279.4 | 483.8 | 11158 | 10168 | 11158 | 10662 | 0.331 | 6.07 | -0.128 | 15.52 | 0.540 | 0.380 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_03 | success | 2299.3 | 3015.0 | net-inference | 236.9 | 473.5 | 11158 | 10228 | 11158 | 10662 | 0.336 | 6.16 | 1.671 | 22.78 | 0.863 | 0.102 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 2241.5 | 2891.0 |  | 214.7 | 429.5 | 11202 | 10198 | 11202 | 10662 | 0.327 | 6.02 | 1.113 | 17.64 | 0.697 | 0.262 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | success | 2087.7 | 2557.0 | net-inference | 133.3 | 331.0 | 19180 | 18374 | 19180 | 18552 | 0.314 | 5.49 | 1.888 | 29.02 | 0.950 | 0.030 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_02 | success | 2285.3 | 3037.0 | net-inference | 272.6 | 474.0 | 19048 | 18112 | 19048 | 18552 | 0.335 | 6.13 | 0.746 | 20.64 | 0.763 | 0.161 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_03 | success | 2286.2 | 3034.0 | net-inference | 263.9 | 478.5 | 19048 | — | 19048 | 18552 | 0.335 | 6.08 | 1.607 | 24.85 | 0.925 | 0.051 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 2219.7 | 2876.0 |  | 223.2 | 427.8 | 19092 | 18243 | 19092 | 18552 | 0.328 | 5.90 | 1.414 | 24.84 | 0.879 | 0.081 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | t2i_01 | success | 6340.0 | 7963.1 | net-inference | 460.0 | 1150.0 | 17476 | 9924 | 17456 | 1258 | 0.307 | 5.49 | 1.818 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | t2i_02 | success | 25120.0 | 34764.5 | net-inference | 8360.0 | 1260.0 | 17462 | 9536 | 17462 | 1258 | 0.343 | 6.03 | 1.849 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | t2i_03 | success | 28390.0 | 38206.1 | net-inference | 8400.0 | 1400.0 | 17462 | 9536 | 17462 | 1258 | 0.310 | 6.12 | 1.538 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 19950.0 | 26977.9 |  | 5740.0 | 1270.0 | 17467 | 9665 | 17460 | 1258 | 0.320 | 5.88 | 1.735 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 9832 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 9832 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 9838 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_01 | success | 2750.0 | 3883.1 | net-inference | 30.0 | 1090.0 | 11082 | — | 11082 | 10718 | 0.317 | 5.53 | 1.675 | 12.09 | 0.581 | 0.518 |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_02 | success | 64190.0 | 101754.8 | net-inference | 36380.0 | 1160.0 | 10984 | 3662 | 10984 | 10718 | 0.319 | 6.00 | 1.897 | 14.49 | 0.575 | 0.411 |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_03 | success | 71410.0 | 109642.4 | net-inference | 37050.0 | 1170.0 | 10984 | 3732 | 10984 | 10718 | 0.317 | 6.15 | 1.552 | 23.43 | 0.848 | 0.113 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 46116.7 | 71760.1 |  | 24486.7 | 1140.0 | 11017 | 3697 | 11017 | 10718 | 0.318 | 5.89 | 1.708 | 16.67 | 0.668 | 0.347 |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_01 | success | 2710.0 | 3847.7 | net-inference | 30.0 | 1090.0 | 18604 | — | 18604 | 18240 | 0.313 | 5.47 | 1.818 | 21.95 | 0.891 | 0.075 |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_02 | success | 9790.0 | 15797.7 | net-inference | 4820.0 | 1170.0 | 18506 | 5542 | 18506 | 18240 | 0.348 | 6.00 | 1.790 | 17.96 | 0.758 | 0.209 |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_03 | success | 9650.0 | 15585.0 | net-inference | 4750.0 | 1170.0 | 18506 | 5542 | 18506 | 18240 | 0.312 | 6.12 | 1.587 | 27.44 | 0.951 | 0.030 |
| **stable-diffusion.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 7383.3 | 11743.5 |  | 3200.0 | 1143.3 | 18539 | 5542 | 18539 | 18240 | 0.324 | 5.86 | 1.732 | 22.45 | 0.867 | 0.105 |

## qwen-image-lightning-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | t2i_01 | success | 12748.1 | 14525.0 | net-inference | 1071.6 | 700.3 | 19618 | 1930 | 19618 | 1328 | 0.321 | 5.78 | 1.832 | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | t2i_02 | success | 14936.4 | 18084.0 | net-inference | 1947.1 | 1192.2 | 19440 | 1444 | 19440 | 1170 | 0.342 | 6.05 | 1.971 | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | t2i_03 | success | 14322.5 | 16707.0 | net-inference | 1493.9 | 884.9 | 19440 | 2602 | 19440 | 1328 | 0.335 | 6.31 | 1.710 | — | — | — |
| **edge-dit.cpp** | **bf16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 14002.4 | 16438.7 |  | 1504.2 | 925.8 | 19499 | 1992 | 19499 | 1275 | 0.333 | 6.05 | 1.837 | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | t2i_01 | success | 2307.2 | 3366.0 | net-inference | 438.1 | 615.6 | 18948 | 18948 | 12378 | 12070 | 0.330 | 5.50 | 1.819 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | t2i_02 | success | 2487.6 | 3800.0 | net-inference | 556.9 | 749.8 | 18628 | 18628 | 12204 | 12070 | 0.339 | 5.85 | 1.959 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | t2i_03 | success | 2476.5 | 3812.0 | net-inference | 574.6 | 755.8 | 18674 | 18674 | 12204 | 12070 | 0.340 | 6.22 | 1.719 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload + vae offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(3)** | 2423.8 | 3659.3 |  | 523.2 | 707.0 | 18750 | 18750 | 12262 | 12070 | 0.336 | 5.86 | 1.832 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 2314.6 | 2870.0 | net-inference | 103.4 | 439.9 | 21326 | — | 21326 | 20878 | 0.322 | 5.68 | 1.869 | 23.75 | 0.873 | 0.083 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_02 | success | 2503.6 | 3233.0 | net-inference | 199.4 | 524.9 | 21152 | 20448 | 21152 | 20878 | 0.355 | 5.99 | 1.965 | 15.40 | 0.677 | 0.254 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_03 | success | 2496.1 | 3254.0 | net-inference | 230.0 | 522.3 | 21152 | 20438 | 21152 | 20878 | 0.339 | 6.19 | 1.698 | 22.13 | 0.817 | 0.188 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 2438.1 | 3119.0 |  | 177.6 | 495.7 | 21210 | 20443 | 21210 | 20878 | 0.339 | 5.96 | 1.844 | 20.43 | 0.789 | 0.175 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 21288 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 21290 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 21290 | — | — | — | — | — | — | — | — | — |

## qwen-image-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 24050 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 24050 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 24050 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | t2i_01 | success | 407871.5 | 416243.9 | net-inference | 7840.9 | 506.1 | 5604 | 5604 | 1026 | 1950 | 0.315 | 5.67 | 1.827 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | t2i_02 | success | 356059.3 | 362384.5 | net-inference | 5399.7 | 890.6 | 4562 | 1572 | 1014 | 2810 | 0.332 | 5.97 | 1.963 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | t2i_03 | success | 375879.5 | 384158.4 | net-inference | 5387.3 | 2685.5 | 4564 | 1572 | 1014 | 3964 | 0.330 | 6.30 | 1.703 | — | — | — |
| **diffusers** | **bf16** | **sequential (full offload)** | **none** | **mean** | **(3)** | 379936.8 | 387595.6 |  | 6209.3 | 1360.7 | 4910 | 2916 | 1018 | 2908 | 0.326 | 5.98 | 1.831 | — | — | — |
| diffusers | w8 | full offload | none | t2i_01 | success | 53807.7 | 67759.5 | net-inference | 3463.4 | 10095.9 | 21284 | 16736 | 21284 | 21284 | 0.308 | 5.70 | 1.833 | 21.29 | 0.845 | 0.156 |
| diffusers | w8 | full offload | none | t2i_02 | success | 55512.1 | 81907.8 | net-inference | 17256.0 | 8693.7 | 21254 | 16704 | 21254 | 21254 | 0.328 | 6.05 | 1.964 | 22.64 | 0.912 | 0.057 |
| diffusers | w8 | full offload | none | t2i_03 | success | 54767.2 | 67142.5 | net-inference | 4048.9 | 7968.8 | 21254 | 16702 | 21254 | 21254 | 0.331 | 6.32 | 1.704 | 33.91 | 0.961 | 0.022 |
| **diffusers** | **w8** | **full offload** | **none** | **mean** | **(3)** | 54695.6 | 72269.9 |  | 8256.1 | 8919.5 | 21264 | 16714 | 21264 | 21264 | 0.322 | 6.02 | 1.834 | 25.95 | 0.906 | 0.078 |
| diffusers | w8 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 24014 | — | — | — | — | — | — | — | — | — |
| diffusers | w8 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 24014 | — | — | — | — | — | — | — | — | — |
| diffusers | w8 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 24014 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | t2i_01 | success | 208371.5 | 211145.0 | net-inference | 2071.7 | 694.4 | 19726 | 2038 | 19726 | 1436 | 0.312 | 5.53 | 1.883 | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | t2i_02 | success | 204160.5 | 207192.0 | net-inference | 2250.4 | 776.3 | 19546 | 2602 | 19546 | 1436 | 0.334 | 5.96 | 1.972 | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | t2i_03 | success | 219242.5 | 222275.0 | net-inference | 2234.9 | 792.0 | 19546 | 2624 | 19546 | 1436 | 0.334 | 6.29 | 1.691 | — | — | — |
| **edge-dit.cpp** | **bf16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 210591.5 | 213537.3 |  | 2185.7 | 754.2 | 19606 | 2421 | 19606 | 1436 | 0.327 | 5.92 | 1.848 | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 418 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | t2i_01 | success | 35344.3 | 36880.0 | net-inference | 920.1 | 610.5 | 19022 | 19022 | 12434 | 12126 | 0.322 | 5.35 | 1.891 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | t2i_02 | success | 35401.6 | 37188.0 | net-inference | 1030.4 | 750.3 | 18686 | 18686 | 12256 | 12126 | 0.338 | 5.91 | 1.976 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | t2i_03 | success | 35466.7 | 37275.0 | net-inference | 1048.0 | 755.1 | 18680 | 18680 | 12256 | 12126 | 0.343 | 6.25 | 1.724 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload + vae offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(3)** | 35404.2 | 37114.3 |  | 999.5 | 705.3 | 18796 | 18796 | 12315 | 12126 | 0.334 | 5.84 | 1.864 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 35559.2 | 36057.0 | net-inference | 115.5 | 377.3 | 21382 | — | 21382 | 20934 | 0.311 | 5.53 | 1.884 | 28.23 | 0.899 | 0.072 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_02 | success | 35601.0 | 36389.0 | net-inference | 260.1 | 522.4 | 21204 | 20380 | 21204 | 20934 | 0.338 | 5.95 | 1.969 | 19.31 | 0.825 | 0.151 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_03 | success | 35617.6 | 36360.0 | net-inference | 223.0 | 514.3 | 21204 | 20446 | 21204 | 20934 | 0.334 | 6.38 | 1.697 | 24.82 | 0.868 | 0.137 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 35592.6 | 36268.7 |  | 199.5 | 471.3 | 21263 | 20413 | 21263 | 20934 | 0.327 | 5.96 | 1.850 | 24.12 | 0.864 | 0.120 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | t2i_01 | success | 128788.8 | 130193.0 | net-inference | 955.7 | 442.1 | 17138 | 8356 | 17138 | 1320 | 0.305 | 5.60 | 1.884 | 22.45 | 0.817 | 0.160 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | t2i_02 | success | 125747.7 | 127562.0 | net-inference | 1293.3 | 516.3 | 16960 | 7864 | 16960 | 1320 | 0.333 | 5.93 | 1.977 | 29.57 | 0.953 | 0.032 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | t2i_03 | success | 135124.4 | 136847.0 | net-inference | 1194.1 | 523.9 | 16960 | 7864 | 16960 | 1320 | 0.335 | 6.36 | 1.704 | 34.34 | 0.965 | 0.028 |
| **edge-dit.cpp** | **q8_0** | **DiT offload + te offload (max-vram 20g) (auto-allocate)** | **none** | **mean** | **(3)** | 129887.0 | 131534.0 |  | 1147.7 | 494.1 | 17019 | 8028 | 17019 | 1320 | 0.325 | 5.96 | 1.855 | 28.79 | 0.912 | 0.073 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 21292 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 21292 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 21292 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | t2i_01 | success | 163520.0 | 168267.1 | net-inference | 1720.0 | 3010.0 | 16918 | 15230 | 16918 | 1310 | 0.318 | 5.74 | 1.825 | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | t2i_02 | success | 193870.0 | 206357.3 | net-inference | 9450.0 | 3030.0 | 16916 | 15004 | 16916 | 1310 | 0.333 | 5.96 | 1.970 | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | t2i_03 | success | 191350.0 | 203870.2 | net-inference | 9480.0 | 3020.0 | 16916 | 15002 | 16916 | 1310 | 0.332 | 6.51 | 1.726 | — | — | — |
| **stable-diffusion.cpp** | **bf16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 182913.3 | 192831.5 |  | 6883.3 | 3020.0 | 16917 | 15079 | 16917 | 1310 | 0.328 | 6.07 | 1.840 | — | — | — |
| stable-diffusion.cpp | bf16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 14982 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | bf16 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 14972 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | bf16 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 14980 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_01 | success | 42010.0 | 44275.5 | net-inference | 30.0 | 2220.0 | 17816 | — | 17816 | 17680 | 0.307 | 5.50 | 1.871 | 19.03 | 0.717 | 0.242 |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_02 | success | 214960.0 | 282747.7 | net-inference | 65490.0 | 2290.0 | 17680 | 6062 | 17676 | 17680 | 0.320 | 5.90 | 1.963 | 17.62 | 0.770 | 0.197 |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_03 | success | 215600.0 | 283324.9 | net-inference | 65390.0 | 2310.0 | 17680 | 6004 | 17676 | 17680 | 0.333 | 6.39 | 1.707 | 22.98 | 0.823 | 0.162 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 157523.3 | 203449.4 |  | 43636.7 | 2273.3 | 17725 | 6033 | 17723 | 17680 | 0.320 | 5.93 | 1.847 | 19.87 | 0.770 | 0.200 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | t2i_01 | success | 74700.0 | 78495.1 | net-inference | 810.0 | 2960.0 | 18784 | 7756 | 18784 | 1198 | 0.320 | 5.73 | 1.810 | 29.90 | 0.968 | 0.015 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | t2i_02 | success | 97050.0 | 114413.4 | net-inference | 14310.0 | 3040.0 | 18782 | 7656 | 18782 | 1198 | 0.325 | 5.94 | 1.961 | 26.34 | 0.937 | 0.031 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | t2i_03 | success | 97410.0 | 114707.4 | net-inference | 14240.0 | 3040.0 | 18830 | 7656 | 18830 | 1198 | 0.331 | 6.38 | 1.728 | 28.74 | 0.933 | 0.042 |
| **stable-diffusion.cpp** | **q8_0** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 89720.0 | 102538.6 |  | 9786.7 | 3013.3 | 18799 | 7689 | 18799 | 1198 | 0.325 | 6.02 | 1.833 | 28.33 | 0.946 | 0.029 |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 7644 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_02 | failed | — | — | process-level co | — | — | 7656 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_03 | failed | — | — | process-level co | — | — | 7646 | — | — | — | — | — | — | — | — | — |

## sd3-medium-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | t2i_01 | success | 2969.5 | 3229.3 | net-inference | 94.7 | 149.5 | 20080 | 20080 | 20080 | 20080 | 0.333 | 5.20 | 1.822 | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_02 | success | 3075.9 | 3747.1 | net-inference | 326.4 | 318.1 | 20080 | 15580 | 16080 | 17766 | 0.336 | 5.77 | 1.772 | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_03 | success | 3075.8 | 3754.6 | net-inference | 309.0 | 339.7 | 20080 | 15580 | 16080 | 16080 | 0.339 | 5.61 | 1.463 | — | — | — |
| **diffusers** | **bf16** | **no-offload** | **none** | **mean** | **(3)** | 3040.4 | 3577.0 |  | 243.4 | 269.1 | 20080 | 17080 | 17413 | 17975 | 0.336 | 5.52 | 1.685 | — | — | — |
| diffusers | w8 | no-offload | none | t2i_01 | success | 3345.0 | 3615.0 | net-inference | 103.3 | 149.5 | 18172 | 18172 | 18172 | 18172 | 0.339 | 5.33 | 1.834 | 23.02 | 0.873 | 0.134 |
| diffusers | w8 | no-offload | none | t2i_02 | success | 3444.7 | 4067.1 | net-inference | 312.5 | 293.1 | 18172 | 13642 | 14172 | 15858 | 0.347 | 5.94 | 1.795 | 20.89 | 0.855 | 0.143 |
| diffusers | w8 | no-offload | none | t2i_03 | success | 3444.3 | 4087.8 | net-inference | 314.5 | 307.6 | 18172 | 13686 | 14172 | 18172 | 0.321 | 5.77 | 1.351 | 22.35 | 0.879 | 0.144 |
| **diffusers** | **w8** | **no-offload** | **none** | **mean** | **(3)** | 3411.3 | 3923.3 |  | 243.4 | 250.0 | 18172 | 15167 | 15505 | 17401 | 0.336 | 5.68 | 1.660 | 22.08 | 0.869 | 0.140 |
| edge-dit.cpp | f16 | no-offload | none | t2i_01 | success | 3244.7 | 3785.0 | net-inference | 212.3 | 322.3 | 15800 | 15558 | 15800 | 15736 | 0.327 | 5.54 | 1.834 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_02 | success | 3265.4 | 4054.0 | net-inference | 371.2 | 411.2 | 15736 | 15378 | 15670 | 15736 | 0.344 | 5.78 | 1.681 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_03 | success | 3263.8 | 4048.0 | net-inference | 366.8 | 410.5 | 15736 | 15308 | 15670 | 15736 | 0.329 | 5.82 | 1.439 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 3258.0 | 3962.3 |  | 316.8 | 381.3 | 15757 | 15415 | 15713 | 15736 | 0.333 | 5.72 | 1.652 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 3390.9 | 3922.0 | net-inference | 204.9 | 321.1 | 5684 | 5442 | 5684 | 5620 | 0.339 | 5.16 | 1.811 | 18.85 | 0.729 | 0.378 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_02 | success | 3483.1 | 4263.0 | net-inference | 360.5 | 412.8 | 5620 | 5336 | 5554 | 5620 | 0.364 | 5.04 | 0.843 | 15.58 | 0.575 | 0.538 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_03 | success | 3432.7 | 4213.0 | net-inference | 361.9 | 412.2 | 5620 | 5266 | 5554 | 5620 | 0.332 | 5.71 | 1.510 | 19.18 | 0.807 | 0.300 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 3435.6 | 4132.7 |  | 309.1 | 382.0 | 5641 | 5348 | 5597 | 5620 | 0.345 | 5.30 | 1.388 | 17.87 | 0.704 | 0.406 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | success | 3398.5 | 3939.0 | net-inference | 212.8 | 322.0 | 9190 | 8948 | 9190 | 9126 | 0.331 | 5.51 | 1.780 | 24.28 | 0.914 | 0.093 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_02 | success | 3442.0 | 4216.0 | net-inference | 357.4 | 410.0 | 9126 | 8772 | 9060 | 9126 | 0.321 | 5.66 | 1.513 | 17.78 | 0.761 | 0.265 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_03 | success | 3462.4 | 4239.0 | net-inference | 357.9 | 412.0 | 9126 | 8766 | 9060 | 9126 | 0.331 | 5.92 | 1.375 | 24.51 | 0.945 | 0.074 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 3434.3 | 4131.3 |  | 309.4 | 381.4 | 9147 | 8829 | 9103 | 9126 | 0.328 | 5.70 | 1.556 | 22.19 | 0.873 | 0.144 |
| stable-diffusion.cpp | f16 | no-offload | none | t2i_01 | success | 4420.0 | 5616.1 | net-inference | 90.0 | 1090.0 | 15822 | — | 15606 | 15822 | 0.325 | 5.44 | 1.791 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | t2i_02 | success | 5330.0 | 8649.4 | net-inference | 2200.0 | 1100.0 | 15822 | 11094 | 15510 | 15822 | 0.341 | 5.75 | 1.838 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | t2i_03 | success | 5830.0 | 9166.8 | net-inference | 2190.0 | 1130.0 | 15822 | 11094 | 15510 | 15822 | 0.321 | 5.80 | 1.332 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 5193.3 | 7810.8 |  | 1493.3 | 1106.7 | 15822 | 11094 | 15542 | 15822 | 0.329 | 5.66 | 1.654 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_01 | success | 3950.0 | 5132.7 | net-inference | 80.0 | 1090.0 | 5968 | — | 5752 | 5968 | 0.327 | 5.32 | 1.863 | 13.47 | 0.623 | 0.527 |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_02 | success | 16720.0 | 53485.5 | net-inference | 35620.0 | 1130.0 | 5968 | 4062 | 5652 | 5968 | 0.349 | 5.47 | 1.131 | 13.29 | 0.532 | 0.584 |
| stable-diffusion.cpp | q4_k | no-offload | none | t2i_03 | success | 16410.0 | 52606.0 | net-inference | 35060.0 | 1120.0 | 5968 | 4062 | 5652 | 5968 | 0.332 | 5.93 | 1.350 | 20.07 | 0.856 | 0.242 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 12360.0 | 37074.8 |  | 23586.7 | 1113.3 | 5968 | 4062 | 5685 | 5968 | 0.336 | 5.57 | 1.448 | 15.61 | 0.670 | 0.451 |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_01 | success | 3970.0 | 5167.8 | net-inference | 90.0 | 1090.0 | 9106 | — | 8890 | 9106 | 0.331 | 5.38 | 1.784 | 23.42 | 0.896 | 0.126 |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_02 | success | 5460.0 | 11672.3 | net-inference | 5070.0 | 1120.0 | 9106 | 6244 | 8790 | 9106 | 0.366 | 5.61 | 0.941 | 16.16 | 0.689 | 0.380 |
| stable-diffusion.cpp | q8_0 | no-offload | none | t2i_03 | success | 5830.0 | 13084.0 | net-inference | 6120.0 | 1120.0 | 9106 | 6244 | 8790 | 9106 | 0.326 | 5.76 | 1.343 | 30.33 | 0.974 | 0.034 |
| **stable-diffusion.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 5086.7 | 9974.7 |  | 3760.0 | 1110.0 | 9106 | 6244 | 8823 | 9106 | 0.341 | 5.58 | 1.356 | 23.30 | 0.853 | 0.180 |

## sd35-medium-turbo-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | t2i_01 | success | 1640.5 | 1895.6 | net-inference | 91.8 | 149.3 | 20938 | 20938 | 20938 | — | 0.306 | 5.31 | -1.482 | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_02 | success | 1736.3 | 2375.4 | net-inference | 324.8 | 288.0 | 20938 | 16338 | 16938 | 20938 | 0.328 | 5.62 | 0.105 | — | — | — |
| diffusers | bf16 | no-offload | none | t2i_03 | success | 1772.6 | 2418.0 | net-inference | 337.2 | 290.6 | 20938 | 16346 | 16938 | 16938 | 0.330 | 5.17 | 1.327 | — | — | — |
| **diffusers** | **bf16** | **no-offload** | **none** | **mean** | **(3)** | 1716.5 | 2229.7 |  | 251.2 | 242.6 | 20938 | 17874 | 18271 | 18938 | 0.322 | 5.37 | -0.017 | — | — | — |
| diffusers | w8 | no-offload | none | t2i_01 | success | 1815.9 | 2073.7 | net-inference | 94.4 | 149.4 | 18842 | — | 18842 | — | 0.308 | 5.11 | 0.240 | 36.37 | 0.975 | 0.039 |
| diffusers | w8 | no-offload | none | t2i_02 | success | 1927.7 | 2515.6 | net-inference | 306.0 | 264.8 | 18842 | 14230 | 14842 | 16528 | 0.326 | 5.41 | -0.377 | 30.96 | 0.953 | 0.061 |
| diffusers | w8 | no-offload | none | t2i_03 | success | 1924.6 | 2513.1 | net-inference | 307.8 | 265.7 | 18842 | 14234 | 14842 | 16532 | 0.332 | 5.25 | 1.412 | 38.28 | 0.988 | 0.012 |
| **diffusers** | **w8** | **no-offload** | **none** | **mean** | **(3)** | 1889.4 | 2367.5 |  | 236.0 | 226.6 | 18842 | 14232 | 16175 | 16530 | 0.322 | 5.26 | 0.425 | 35.20 | 0.972 | 0.037 |
| edge-dit.cpp | f16 | no-offload | none | t2i_01 | success | 1688.8 | 2233.0 | net-inference | 215.0 | 323.9 | 16880 | — | 16880 | 16818 | 0.335 | 5.60 | 1.877 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_02 | success | 1700.1 | 2490.0 | net-inference | 371.3 | 411.9 | 16818 | 16366 | 16750 | 16818 | 0.342 | 5.68 | 1.490 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_03 | success | 1704.5 | 2502.0 | net-inference | 372.0 | 419.5 | 16818 | 16370 | 16750 | 16818 | 0.335 | 5.08 | 1.471 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 1697.8 | 2408.3 |  | 319.4 | 385.1 | 16839 | 16368 | 16793 | 16818 | 0.337 | 5.45 | 1.612 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 1716.2 | 2253.0 | net-inference | 210.2 | 321.0 | 6472 | — | 6472 | 6410 | 0.319 | 5.38 | 1.673 | 19.13 | 0.747 | 0.284 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_02 | success | 1785.0 | 2561.0 | net-inference | 359.4 | 410.0 | 6410 | 6030 | 6340 | 6410 | 0.327 | 5.78 | 0.336 | 19.40 | 0.734 | 0.342 |
| edge-dit.cpp | q4_k | no-offload | none | t2i_03 | success | 1794.8 | 2592.0 | net-inference | 373.7 | 413.3 | 6410 | 6024 | 6340 | 6410 | 0.333 | 5.36 | 1.136 | 19.61 | 0.832 | 0.250 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 1765.3 | 2468.7 |  | 314.4 | 381.4 | 6431 | 6027 | 6384 | 6410 | 0.326 | 5.51 | 1.049 | 19.38 | 0.771 | 0.292 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | success | 1708.3 | 2243.0 | net-inference | 207.0 | 322.2 | 10078 | 9838 | 10078 | 10016 | 0.336 | 5.60 | 1.879 | 30.06 | 0.967 | 0.042 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_02 | success | 1794.6 | 2538.0 | net-inference | 322.0 | 415.2 | 10016 | 9632 | 9948 | 10016 | 0.341 | 5.65 | 0.278 | 20.87 | 0.773 | 0.280 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_03 | success | 1798.1 | 2575.0 | net-inference | 356.3 | 414.4 | 10016 | 9632 | 9948 | 10016 | 0.340 | 5.17 | 1.430 | 33.16 | 0.983 | 0.023 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 1767.0 | 2452.0 |  | 295.1 | 383.9 | 10037 | 9701 | 9991 | 10016 | 0.339 | 5.47 | 1.196 | 28.03 | 0.908 | 0.115 |