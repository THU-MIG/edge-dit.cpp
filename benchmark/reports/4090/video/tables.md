# Cross-system comparison matrix (all metrics, one-shot aggregate)

33 runs total | success 33 | failed 0

> **Speed boundary reminder**: to compare inference speed use "DiT sampling ms" (component-level denoise time, reliable). "end-to-end ms" excludes model load (load-once boundary; see the "boundary" column: net-inference = excludes load/encoding), but sd.cpp quantized tiers fold one-time on-the-fly conversion into the denoise stage, so their end-to-end (and DiT) is inflated and must not be used for cross-system speed claims. Quantization quality loss (PSNR/SSIM/LPIPS vs FP16) is only meaningful within the same system vs its own FP16 baseline; not comparable across systems.

> **Special note for sd.cpp**: stable-diffusion.cpp loads layer-by-layer while sampling, and on-the-fly quantization conversion (q4_K/q8, tens to hundreds of seconds) folds into the denoise-stage timing, so its "DiT sampling ms" is likewise inflated under quantized tiers and does not represent pure inference. sd.cpp speed should be re-measured with pre-quantized weights, or only used as a same-tier trend reference; it cannot be compared directly with edge/diffusers DiT sampling.

> **The headline tier is q8** (usable image quality); q4 is only an extreme VRAM-saving reference point with obvious quality loss, and is not suitable for speed/quality advantage claims.


## wan2-t2v-1.3b-text-to-video  (text-to-video)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | video_01 | success | 53614.7 | 56390.3 | net-inference | 113.0 | 2518.9 | 20498 | — | 20498 | 20498 | 0.314 | 5.97 | 0.111 | 0.808 | 0.012 | — | — | — |
| diffusers | bf16 | no-offload | none | video_02 | success | 53745.9 | 56883.5 | net-inference | 284.5 | 2714.8 | 20498 | 14568 | 15858 | 20498 | 0.305 | 4.81 | 0.007 | 0.983 | 0.002 | — | — | — |
| diffusers | bf16 | no-offload | none | video_03 | success | 53762.7 | 56911.6 | net-inference | 295.7 | 2715.2 | 20498 | 14602 | 15858 | 20498 | 0.333 | 5.31 | 0.099 | 0.828 | 0.017 | — | — | — |
| **diffusers** | **bf16** | **no-offload** | **none** | **mean** | **(3)** | 53707.8 | 56728.5 |  | 231.1 | 2649.6 | 20498 | 14585 | 17405 | 20498 | 0.317 | 5.36 | 0.072 | 0.873 | 0.010 | — | — | — |
| diffusers | w8 | no-offload | none | video_01 | success | 56522.7 | 59296.2 | net-inference | 113.7 | 2519.2 | 19568 | 19568 | 19568 | 19568 | 0.318 | 6.02 | 0.108 | 0.807 | 0.017 | 18.32 | 0.735 | 0.225 |
| diffusers | w8 | no-offload | none | video_02 | success | 56601.3 | 59740.5 | net-inference | 283.7 | 2724.0 | 19568 | 13324 | 14488 | 19568 | 0.326 | 4.97 | 0.008 | 0.981 | 0.003 | 18.46 | 0.726 | 0.365 |
| diffusers | w8 | no-offload | none | video_03 | success | 56616.4 | 59757.3 | net-inference | 284.0 | 2717.6 | 19568 | 13098 | 14488 | 19568 | 0.317 | 5.37 | 0.082 | 0.827 | 0.014 | 22.50 | 0.836 | 0.201 |
| **diffusers** | **w8** | **no-offload** | **none** | **mean** | **(3)** | 56580.2 | 59598.0 |  | 227.1 | 2653.6 | 19568 | 15330 | 16181 | 19568 | 0.320 | 5.45 | 0.066 | 0.872 | 0.011 | 19.76 | 0.766 | 0.264 |
| edge-dit.cpp | f16 | no-offload | none | video_01 | success | 49375.9 | 55042.0 | net-inference | 250.3 | 4870.1 | 17848 | 16930 | 17848 | 17612 | 0.319 | 5.97 | 0.025 | 0.893 | 0.006 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | video_02 | success | 49487.4 | 55582.0 | net-inference | 359.1 | 4991.9 | 17638 | 16664 | 17638 | 17612 | 0.232 | 5.52 | 0.019 | 0.965 | 0.015 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | video_03 | success | 49471.1 | 55647.0 | net-inference | 356.7 | 5033.6 | 17638 | 16664 | 17638 | 17612 | 0.313 | 5.05 | 0.005 | 0.955 | 0.002 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 49444.8 | 55423.7 |  | 322.0 | 4965.2 | 17708 | 16753 | 17708 | 17612 | 0.288 | 5.51 | 0.016 | 0.938 | 0.008 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | video_01 | success | 53968.3 | 59640.0 | net-inference | 248.4 | 4878.7 | 9438 | 8520 | 9438 | 9202 | 0.318 | 6.12 | 0.024 | 0.916 | 0.008 | 18.94 | 0.715 | 0.198 |
| edge-dit.cpp | q4_k | no-offload | none | video_02 | success | 54103.8 | 60170.0 | net-inference | 325.3 | 4988.3 | 9228 | 8430 | 9228 | 9202 | 0.241 | 5.39 | 0.022 | 0.958 | 0.020 | 20.00 | 0.706 | 0.239 |
| edge-dit.cpp | q4_k | no-offload | none | video_03 | success | 54055.6 | 60205.0 | net-inference | 323.5 | 5019.2 | 9228 | 8430 | 9228 | 9202 | 0.316 | 5.49 | 0.022 | 0.858 | 0.009 | 16.27 | 0.601 | 0.423 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 54042.6 | 60005.0 |  | 299.1 | 4962.0 | 9298 | 8460 | 9298 | 9202 | 0.292 | 5.67 | 0.023 | 0.911 | 0.012 | 18.41 | 0.674 | 0.286 |
| edge-dit.cpp | q8_0 | no-offload | none | video_01 | success | 53875.7 | 59578.0 | net-inference | 242.9 | 4905.2 | 12316 | 11398 | 12316 | 12080 | 0.320 | 5.98 | 0.025 | 0.894 | 0.006 | 29.29 | 0.944 | 0.020 |
| edge-dit.cpp | q8_0 | no-offload | none | video_02 | success | 54042.8 | 60058.0 | net-inference | 313.9 | 4964.3 | 12106 | 11010 | 12106 | 12080 | 0.327 | 5.04 | 0.005 | 0.972 | 0.004 | 19.04 | 0.693 | 0.270 |
| edge-dit.cpp | q8_0 | no-offload | none | video_03 | success | 53972.6 | 60145.0 | net-inference | 311.9 | 5043.0 | 12106 | 11300 | 12106 | 12080 | 0.314 | 5.11 | 0.005 | 0.956 | 0.002 | 30.95 | 0.921 | 0.021 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 53963.7 | 59927.0 |  | 289.6 | 4970.8 | 12176 | 11236 | 12176 | 12080 | 0.320 | 5.38 | 0.012 | 0.941 | 0.004 | 26.43 | 0.853 | 0.104 |
| stable-diffusion.cpp | f16 | no-offload | none | video_01 | success | 83070.0 | 102574.9 | net-inference | 190.0 | 18830.0 | 17818 | 14810 | 15330 | 17818 | 0.342 | 5.96 | 0.051 | 0.828 | 0.008 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | video_02 | success | 83620.0 | 106429.3 | net-inference | 3290.0 | 19010.0 | 17818 | 11586 | 15190 | 17818 | 0.276 | 5.54 | 0.033 | 0.942 | 0.031 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | video_03 | success | 83580.0 | 106353.3 | net-inference | 3310.0 | 18940.0 | 17818 | 11586 | 15190 | 17818 | 0.325 | 5.25 | 0.011 | 0.909 | 0.003 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 83423.3 | 105119.1 |  | 2263.3 | 18926.7 | 17818 | 12661 | 15237 | 17818 | 0.315 | 5.59 | 0.032 | 0.893 | 0.014 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | video_01 | success | 80120.0 | 99568.5 | net-inference | 180.0 | 18780.0 | 11372 | 8066 | 8884 | 11372 | 0.335 | 6.03 | 0.038 | 0.882 | 0.010 | 16.90 | 0.611 | 0.286 |
| stable-diffusion.cpp | q4_k | no-offload | none | video_02 | success | 90180.0 | 137970.4 | net-inference | 28300.0 | 19000.0 | 11372 | 7278 | 8744 | 11372 | 0.263 | 5.42 | 0.033 | 0.940 | 0.031 | 20.40 | 0.719 | 0.230 |
| stable-diffusion.cpp | q4_k | no-offload | none | video_03 | success | 90050.0 | 137966.1 | net-inference | 28120.0 | 19270.0 | 11372 | 6980 | 8744 | 11372 | 0.327 | 5.46 | 0.027 | 0.802 | 0.010 | 19.43 | 0.620 | 0.284 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 86783.3 | 125168.3 |  | 18866.7 | 19016.7 | 11372 | 7441 | 8791 | 11372 | 0.308 | 5.64 | 0.032 | 0.875 | 0.017 | 18.91 | 0.650 | 0.267 |
| stable-diffusion.cpp | q8_0 | no-offload | none | video_01 | success | 79860.0 | 99264.5 | net-inference | 190.0 | 18730.0 | 11308 | — | 8820 | 11308 | 0.341 | 5.98 | 0.052 | 0.832 | 0.008 | 25.91 | 0.897 | 0.052 |
| stable-diffusion.cpp | q8_0 | no-offload | none | video_02 | success | 80660.0 | 115815.1 | net-inference | 15480.0 | 19180.0 | 11308 | 6526 | 8680 | 11308 | 0.327 | 5.54 | 0.010 | 0.958 | 0.006 | 17.47 | 0.673 | 0.279 |
| stable-diffusion.cpp | q8_0 | no-offload | none | video_03 | success | 80630.0 | 120170.3 | net-inference | 19930.0 | 19080.0 | 11308 | 6544 | 8680 | 11308 | 0.323 | 5.30 | 0.011 | 0.914 | 0.003 | 32.64 | 0.893 | 0.031 |
| **stable-diffusion.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 80383.3 | 111749.9 |  | 11866.7 | 18996.7 | 11308 | 6535 | 8727 | 11308 | 0.330 | 5.61 | 0.024 | 0.901 | 0.006 | 25.34 | 0.821 | 0.120 |

## wan21-t2v-1.3b-distill-text-to-video  (text-to-video)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | video_01 | success | 6604.8 | 12238.0 | net-inference | 123.8 | 4957.8 | 17832 | — | 17832 | 17596 | 0.323 | 5.89 | 0.076 | 0.842 | 0.009 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | video_02 | success | 6793.4 | 12774.0 | net-inference | 229.4 | 4998.7 | 17622 | 16642 | 17622 | 17596 | 0.301 | 5.29 | 0.099 | 0.789 | 0.046 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | video_03 | success | 6799.1 | 12770.0 | net-inference | 231.2 | 4973.2 | 17622 | 16648 | 17622 | 17596 | 0.307 | 5.56 | 0.017 | 0.892 | 0.005 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 6732.4 | 12594.0 |  | 194.8 | 4976.5 | 17692 | 16645 | 17692 | 17596 | 0.310 | 5.58 | 0.064 | 0.841 | 0.020 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | video_01 | success | 7209.8 | 12677.0 | net-inference | 122.0 | 4812.2 | 9422 | 8504 | 9422 | 9186 | 0.330 | 6.19 | 0.050 | 0.872 | 0.007 | 11.38 | 0.358 | 0.448 |
| edge-dit.cpp | q4_k | no-offload | none | video_02 | success | 7487.1 | 13474.0 | net-inference | 193.3 | 5033.6 | 9212 | 8392 | 9212 | 9212 | 0.310 | 5.18 | 0.101 | 0.806 | 0.050 | 11.30 | 0.387 | 0.482 |
| edge-dit.cpp | q4_k | no-offload | none | video_03 | success | 7485.6 | 13443.0 | net-inference | 192.2 | 5029.6 | 9212 | — | 9212 | 9186 | 0.308 | 5.67 | 0.016 | 0.923 | 0.012 | 15.63 | 0.539 | 0.285 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 7394.2 | 13198.0 |  | 169.2 | 4958.5 | 9282 | 8448 | 9282 | 9195 | 0.316 | 5.68 | 0.056 | 0.867 | 0.023 | 12.77 | 0.428 | 0.405 |
| edge-dit.cpp | q8_0 | no-offload | none | video_01 | success | 7224.9 | 12769.0 | net-inference | 121.0 | 4878.2 | 12300 | — | 12300 | 12064 | 0.324 | 5.94 | 0.080 | 0.831 | 0.009 | 17.85 | 0.707 | 0.163 |
| edge-dit.cpp | q8_0 | no-offload | none | video_02 | success | 7466.4 | 13328.0 | net-inference | 187.3 | 4939.8 | 12090 | 11286 | 12090 | 12064 | 0.300 | 5.12 | 0.080 | 0.855 | 0.050 | 11.38 | 0.379 | 0.482 |
| edge-dit.cpp | q8_0 | no-offload | none | video_03 | success | 7462.0 | 13301.0 | net-inference | 186.7 | 4922.1 | 12090 | — | 12090 | 12064 | 0.308 | 5.63 | 0.017 | 0.895 | 0.005 | 22.38 | 0.818 | 0.076 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 7384.4 | 13132.7 |  | 165.0 | 4913.4 | 12160 | 11286 | 12160 | 12064 | 0.311 | 5.56 | 0.059 | 0.860 | 0.021 | 17.21 | 0.635 | 0.240 |