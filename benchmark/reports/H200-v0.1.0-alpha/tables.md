## Single-GPU Inference

| Workload | Scenario | System | Load | Median | P90 | Peak VRAM | Boundary | Status |
|---|---|---|---:|---:|---:|---:|---|---|
| flux1-dev-t2i-1024-s50 | default | diffusers | 14530.5 ms | 10039.8 ms | 10047.6 ms | 37711 MiB | load_once_e2e_generation_no_output_encoding | success |
| qwen-image-t2i-1024-s50 | default | diffusers | 25219.9 ms | 9557.7 ms | 9565.3 ms | 60935 MiB | load_once_e2e_generation_no_output_encoding | success |
| sd3-medium-t2i-1024-s50 | default | diffusers | 11243.8 ms | 3376.3 ms | 3381.2 ms | 20283 MiB | load_once_e2e_generation_no_output_encoding | success |
| flux1-dev-t2i-1024-s50 | default | edge-dit.cpp | 6644.5 ms | 10784.0 ms | 10861.0 ms | 38341 MiB | load_once_e2e_generation_no_output_encoding | success |
| qwen-image-t2i-1024-s50 | default | edge-dit.cpp | 11621.4 ms | 10696.5 ms | 10736.0 ms | 59725 MiB | load_once_e2e_generation_no_output_encoding | success |
| sd3-medium-t2i-1024-s50 | default | edge-dit.cpp | 5840.3 ms | 4003.0 ms | 4049.0 ms | 20833 MiB | load_once_e2e_generation_no_output_encoding | success |
| flux1-dev-t2i-1024-s50 | default | stable-diffusion.cpp | 1332.7 ms | 30370.5 ms | 30379.3 ms | 40331 MiB | load_once_e2e_generation_no_output_encoding | success |
| qwen-image-t2i-1024-s50 | default | stable-diffusion.cpp | 1781.9 ms | 62671.3 ms | 62728.2 ms | 61879 MiB | load_once_e2e_generation_no_output_encoding | success |
| sd3-medium-t2i-1024-s50 | default | stable-diffusion.cpp | 1457.3 ms | 10739.7 ms | 10797.0 ms | 22997 MiB | load_once_e2e_generation_no_output_encoding | success |
