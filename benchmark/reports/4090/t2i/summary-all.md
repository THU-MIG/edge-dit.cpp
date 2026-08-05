# Summary table (mean, core columns)

> One table at a glance, split by task (quality columns differ per task). For speed look at DiT sampling ms; VRAM unit MiB; PSNR/SSIM/LPIPS are quantization vs same-system FP16.


## flux-dev-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | 21901.0 | 45241.0 | 23960 | 0.308 | 6.14 | 1.737 | — | — | — |
| diffusers | bf16 | no-offload | none | — | — | 24042 | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 85007.5 | 88939.0 | 1817 | 0.308 | 6.14 | 1.737 | — | — | — |
| diffusers | w8 | no-offload | none | 13189.9 | 14138.9 | 23866 | 0.307 | 6.12 | 1.735 | 30.81 | 0.970 | 0.024 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 39078.9 | 40513.0 | 19610 | 0.302 | 6.11 | 1.582 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 23536 | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 10674.8 | 11334.0 | 11222 | 0.304 | 6.03 | 1.611 | 21.45 | 0.808 | 0.243 |
| edge-dit.cpp | q8_0 | no-offload | none | 10569.3 | 11196.0 | 19112 | 0.296 | 6.07 | 1.630 | 28.65 | 0.892 | 0.130 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 44723.3 | 54594.9 | 17528 | 0.309 | 6.06 | 1.476 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | 9859 | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 53440.0 | 78524.4 | 11037 | 0.311 | 6.05 | 1.525 | 23.52 | 0.878 | 0.143 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 17796.7 | 22193.6 | 18559 | 0.310 | 6.10 | 1.525 | 29.00 | 0.962 | 0.043 |

## flux-schnell-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | 24024 | — | — | — | — | — | — |
| diffusers | w8 | no-offload | none | 2643.2 | 3411.9 | 23869 | 0.306 | 5.86 | 1.597 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 7959.9 | 9579.0 | 19588 | 0.331 | 5.91 | 1.552 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 23516 | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 2241.5 | 2891.0 | 11202 | 0.327 | 6.02 | 1.113 | 17.64 | 0.697 | 0.262 |
| edge-dit.cpp | q8_0 | no-offload | none | 2219.7 | 2876.0 | 19092 | 0.328 | 5.90 | 1.414 | 24.84 | 0.879 | 0.081 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 19950.0 | 26977.9 | 17467 | 0.320 | 5.88 | 1.735 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | 9834 | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 46116.7 | 71760.1 | 11017 | 0.318 | 5.89 | 1.708 | 16.67 | 0.668 | 0.347 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 7383.3 | 11743.5 | 18539 | 0.324 | 5.86 | 1.732 | 22.45 | 0.867 | 0.105 |

## qwen-image-lightning-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 14002.4 | 16438.7 | 19499 | 0.333 | 6.05 | 1.837 | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | — | — | 418 | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 2423.8 | 3659.3 | 18750 | 0.336 | 5.86 | 1.832 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 2438.1 | 3119.0 | 21210 | 0.339 | 5.96 | 1.844 | 20.43 | 0.789 | 0.175 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 21289 | — | — | — | — | — | — |

## qwen-image-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | 24050 | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 379936.8 | 387595.6 | 4910 | 0.326 | 5.98 | 1.831 | — | — | — |
| diffusers | w8 | full offload | none | 54695.6 | 72269.9 | 21264 | 0.322 | 6.02 | 1.834 | 25.95 | 0.906 | 0.078 |
| diffusers | w8 | no-offload | none | — | — | 24014 | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 210591.5 | 213537.3 | 19606 | 0.327 | 5.92 | 1.848 | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | — | — | 418 | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 35404.2 | 37114.3 | 18796 | 0.334 | 5.84 | 1.864 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 35592.6 | 36268.7 | 21263 | 0.327 | 5.96 | 1.850 | 24.12 | 0.864 | 0.120 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 129887.0 | 131534.0 | 17019 | 0.325 | 5.96 | 1.855 | 28.79 | 0.912 | 0.073 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 21292 | — | — | — | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | 182913.3 | 192831.5 | 16917 | 0.328 | 6.07 | 1.840 | — | — | — |
| stable-diffusion.cpp | bf16 | no-offload | none | — | — | 14978 | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 157523.3 | 203449.4 | 17725 | 0.320 | 5.93 | 1.847 | 19.87 | 0.770 | 0.200 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 89720.0 | 102538.6 | 18799 | 0.325 | 6.02 | 1.833 | 28.33 | 0.946 | 0.029 |
| stable-diffusion.cpp | q8_0 | no-offload | none | — | — | 7649 | — | — | — | — | — | — |

## sd3-medium-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 3040.4 | 3577.0 | 20080 | 0.336 | 5.52 | 1.685 | — | — | — |
| diffusers | w8 | no-offload | none | 3411.3 | 3923.3 | 18172 | 0.336 | 5.68 | 1.660 | 22.08 | 0.869 | 0.140 |
| edge-dit.cpp | f16 | no-offload | none | 3258.0 | 3962.3 | 15757 | 0.333 | 5.72 | 1.652 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 3435.6 | 4132.7 | 5641 | 0.345 | 5.30 | 1.388 | 17.87 | 0.704 | 0.406 |
| edge-dit.cpp | q8_0 | no-offload | none | 3434.3 | 4131.3 | 9147 | 0.328 | 5.70 | 1.556 | 22.19 | 0.873 | 0.144 |
| stable-diffusion.cpp | f16 | no-offload | none | 5193.3 | 7810.8 | 15822 | 0.329 | 5.66 | 1.654 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 12360.0 | 37074.8 | 5968 | 0.336 | 5.57 | 1.448 | 15.61 | 0.670 | 0.451 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 5086.7 | 9974.7 | 9106 | 0.341 | 5.58 | 1.356 | 23.30 | 0.853 | 0.180 |

## sd35-medium-turbo-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 1716.5 | 2229.7 | 20938 | 0.322 | 5.37 | -0.017 | — | — | — |
| diffusers | w8 | no-offload | none | 1889.4 | 2367.5 | 18842 | 0.322 | 5.26 | 0.425 | 35.20 | 0.972 | 0.037 |
| edge-dit.cpp | f16 | no-offload | none | 1697.8 | 2408.3 | 16839 | 0.337 | 5.45 | 1.612 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 1765.3 | 2468.7 | 6431 | 0.326 | 5.51 | 1.049 | 19.38 | 0.771 | 0.292 |
| edge-dit.cpp | q8_0 | no-offload | none | 1767.0 | 2452.0 | 10037 | 0.339 | 5.47 | 1.196 | 28.03 | 0.908 | 0.115 |
