# Cross-system comparison matrix (all metrics, one-shot aggregate)

29 runs total | success 19 | failed 10

> **Speed boundary reminder**: to compare inference speed use "DiT sampling ms" (component-level denoise time, reliable). "end-to-end ms" includes one-time on-the-fly quantization conversion / model loading (see the "boundary" column: net-inference = excludes load/encoding, incl-load+encode = single CLI run), and must not be used for cross-system speed claims. Quantization quality loss (PSNR/SSIM/LPIPS vs FP16) is only meaningful within the same system vs its own FP16 baseline; not comparable across systems.

> **Special note for sd.cpp**: stable-diffusion.cpp loads layer-by-layer while sampling, and on-the-fly quantization conversion (q4_K/q8, tens to hundreds of seconds) folds into the denoise-stage timing, so its "DiT sampling ms" is likewise inflated under quantized tiers and does not represent pure inference. sd.cpp speed should be re-measured with pre-quantized weights, or only used as a same-tier trend reference; it cannot be compared directly with edge/diffusers DiT sampling.

> **The headline tier is q8** (usable image quality); q4 is only an extreme VRAM-saving reference point with obvious quality loss, and is not suitable for speed/quality advantage claims.


## flux-dev-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | t2i_01 | success | 37192.0 | 38752.0 | net-inference | 934.3 | 617.8 | 19652 | 10046 | 19652 | 1452 | 0.303 | 6.11 | 1.768 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(1)** | 37192.0 | 38752.0 |  | 934.3 | 617.8 | 19652 | 10046 | 19652 | 1452 | 0.303 | 6.11 | 1.768 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 23674 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | t2i_01 | failed | — | — | process-level co | — | — | 23466 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 10704.6 | 11455.0 | net-inference | 272.0 | 473.1 | 11316 | 10456 | 11316 | 10820 | 0.298 | 5.99 | 1.847 | 20.21 | 0.819 | 0.219 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 10704.6 | 11455.0 |  | 272.0 | 473.1 | 11316 | 10456 | 11316 | 10820 | 0.298 | 5.99 | 1.847 | 20.21 | 0.819 | 0.219 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | success | 10565.3 | 11302.0 | net-inference | 260.1 | 470.6 | 19206 | 18270 | 19206 | 18710 | 0.282 | 5.95 | 1.771 | 16.42 | 0.713 | 0.364 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(1)** | 10565.3 | 11302.0 |  | 260.1 | 470.6 | 19206 | 18270 | 19206 | 18710 | 0.282 | 5.95 | 1.771 | 16.42 | 0.713 | 0.364 |

## flux-schnell-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | t2i_01 | success | 7631.1 | 8857.0 | net-inference | 757.8 | 462.9 | 19630 | 10040 | 19630 | 3500 | 0.316 | 5.47 | 1.880 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(1)** | 7631.1 | 8857.0 |  | 757.8 | 462.9 | 19630 | 10040 | 19630 | 3500 | 0.316 | 5.47 | 1.880 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 23654 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | t2i_01 | failed | — | — | process-level co | — | — | 23694 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 2305.2 | 3026.0 | net-inference | 229.6 | 485.1 | 11296 | 10366 | 11296 | 10800 | 0.310 | 5.74 | 1.822 | 14.70 | 0.692 | 0.298 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 2305.2 | 3026.0 |  | 229.6 | 485.1 | 11296 | 10366 | 11296 | 10800 | 0.310 | 5.74 | 1.822 | 14.70 | 0.692 | 0.298 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | success | 2271.0 | 3010.0 | net-inference | 258.7 | 474.9 | 19186 | 18256 | 19186 | 18690 | 0.312 | 5.58 | 1.886 | 28.46 | 0.944 | 0.036 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(1)** | 2271.0 | 3010.0 |  | 258.7 | 474.9 | 19186 | 18256 | 19186 | 18690 | 0.312 | 5.58 | 1.886 | 28.46 | 0.944 | 0.036 |

## qwen-image-lightning-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | t2i_01 | success | 12288.2 | 14086.0 | net-inference | 1031.4 | 760.4 | 19530 | 14170 | 19530 | 1464 | 0.174 | 4.30 | -1.144 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(1)** | 12288.2 | 14086.0 |  | 1031.4 | 760.4 | 19530 | 14170 | 19530 | 1464 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 556 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | t2i_01 | failed | — | — | process-level co | — | — | 556 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | t2i_01 | success | 2471.6 | 3797.0 | net-inference | 569.1 | 750.8 | 18746 | 18746 | 12342 | 12208 | 0.327 | 5.48 | 1.847 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload + vae offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(1)** | 2471.6 | 3797.0 |  | 569.1 | 750.8 | 18746 | 18746 | 12342 | 12208 | 0.327 | 5.48 | 1.847 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 2478.2 | 3229.0 | net-inference | 225.0 | 520.1 | 18050 | — | 18050 | 17776 | 0.330 | 5.55 | 1.857 | 2.80 | 0.347 | 0.934 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 2478.2 | 3229.0 |  | 225.0 | 520.1 | 18050 | — | 18050 | 17776 | 0.330 | 5.55 | 1.857 | 2.80 | 0.347 | 0.934 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 21428 | — | — | — | — | — | — | — | — | — |

## qwen-image-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | t2i_01 | success | 91080.2 | 93094.0 | net-inference | 1167.5 | 837.5 | 19530 | 14172 | 19530 | 1464 | 0.174 | 4.30 | -1.144 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(1)** | 91080.2 | 93094.0 |  | 1167.5 | 837.5 | 19530 | 14172 | 19530 | 1464 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 556 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | t2i_01 | failed | — | — | process-level co | — | — | 556 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | t2i_01 | success | 17511.4 | 18842.0 | net-inference | 571.2 | 753.9 | 18802 | 18802 | 12342 | 12208 | 0.330 | 5.27 | 1.866 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload + vae offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(1)** | 17511.4 | 18842.0 |  | 571.2 | 753.9 | 18802 | 18802 | 12342 | 12208 | 0.330 | 5.27 | 1.866 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 17510.4 | 18240.0 | net-inference | 196.8 | 526.9 | 18050 | 17334 | 18050 | 17776 | 0.329 | 5.26 | 1.858 | 2.62 | 0.355 | 0.931 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 17510.4 | 18240.0 |  | 196.8 | 526.9 | 18050 | 17334 | 18050 | 17776 | 0.329 | 5.26 | 1.858 | 2.62 | 0.355 | 0.931 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | t2i_01 | success | 56646.8 | 57815.0 | net-inference | 640.2 | 522.8 | 16948 | 7932 | 16948 | 1352 | 0.312 | 5.57 | 1.882 | 2.35 | 0.328 | 0.930 |
| **edge-dit.cpp** | **q8_0** | **DiT offload + te offload (max-vram 20g) (auto-allocate)** | **none** | **mean** | **(1)** | 56646.8 | 57815.0 |  | 640.2 | 522.8 | 16948 | 7932 | 16948 | 1352 | 0.312 | 5.57 | 1.882 | 2.35 | 0.328 | 0.930 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | failed | — | — | process-level co | — | — | 21430 | — | — | — | — | — | — | — | — | — |

## sd3-medium-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | t2i_01 | success | 3263.3 | 4054.0 | net-inference | 366.0 | 418.2 | 15874 | 15446 | 15808 | 15874 | 0.329 | 5.54 | 1.831 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(1)** | 3263.3 | 4054.0 |  | 366.0 | 418.2 | 15874 | 15446 | 15808 | 15874 | 0.329 | 5.54 | 1.831 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 3475.9 | 4247.0 | net-inference | 352.3 | 412.6 | 5758 | 5404 | 5692 | 5758 | 0.339 | 5.15 | 1.855 | 19.80 | 0.776 | 0.291 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 3475.9 | 4247.0 |  | 352.3 | 412.6 | 5758 | 5404 | 5692 | 5758 | 0.339 | 5.15 | 1.855 | 19.80 | 0.776 | 0.291 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | success | 3500.8 | 4294.0 | net-inference | 358.4 | 427.4 | 9264 | 8904 | 9198 | 9264 | 0.322 | 5.55 | 1.795 | 25.34 | 0.922 | 0.072 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(1)** | 3500.8 | 4294.0 |  | 358.4 | 427.4 | 9264 | 8904 | 9198 | 9264 | 0.322 | 5.55 | 1.795 | 25.34 | 0.922 | 0.072 |

## sd35-medium-turbo-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | t2i_01 | success | 856.5 | 1595.0 | net-inference | 316.5 | 414.3 | 16956 | 16498 | 16888 | 16956 | 0.338 | 5.34 | 1.827 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(1)** | 856.5 | 1595.0 |  | 316.5 | 414.3 | 16956 | 16498 | 16888 | 16956 | 0.338 | 5.34 | 1.827 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | t2i_01 | success | 931.3 | 1652.0 | net-inference | 298.9 | 416.2 | 6548 | 6168 | 6478 | 6548 | 0.316 | 5.38 | 0.398 | 19.30 | 0.690 | 0.433 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 931.3 | 1652.0 |  | 298.9 | 416.2 | 6548 | 6168 | 6478 | 6548 | 0.316 | 5.38 | 0.398 | 19.30 | 0.690 | 0.433 |
| edge-dit.cpp | q8_0 | no-offload | none | t2i_01 | success | 941.7 | 1648.0 | net-inference | 290.1 | 410.4 | 10154 | 9758 | 10086 | 10154 | 0.333 | 5.27 | 1.791 | 32.38 | 0.967 | 0.047 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(1)** | 941.7 | 1648.0 |  | 290.1 | 410.4 | 10154 | 9758 | 10086 | 10154 | 0.333 | 5.27 | 1.791 | 32.38 | 0.967 | 0.047 |