# sites/ — machine path config

One `site-*.yaml` per machine; this is **the only place in the whole benchmark allowed to hold machine absolute paths**.
Other configs (models/ methods/ jobs/) all reference here indirectly via **ref names**, so switching machines only changes this one file.

When running a benchmark, specify with `--site`:
```
python benchmark/run.py --job jobs/xxx.yaml --site benchmark/sites/site-<your-machine>.yaml
```

## 3-step start
1. Copy the template: `cp site-example.yaml site-<your-machine>.yaml`
2. Change the `/path/to/...` placeholders to the **actual paths** on your machine
3. Fill in only what you want to test (see below); delete entire lines you don't need

## Purpose of each paths entry

Fill in only "**where the external things not in git live on your machine**". Four categories:

| category | key | purpose | when required |
|---|---|---|---|
| **core** | `edge_dit_repo` | edge-dit.cpp repo root | always |
| | `edge_dit_cli` / `edge_dit_sample` | the two binaries you built | always |
| **models** | `flux_dev_dir` / `sd3_medium_dir` / … | each model's weight directory | fill in the few you want to test |
| | `*_weights` (distilled) | single weight file for a distilled model | when testing that distilled model |
| **image editing** | `edit_input_image` | input image for the image-editing task | only when testing image-editing |
| **cross-system** | `diffusers_python` | the python used to compare diffusers | only when diffusers is in systems |
| | `stable_diffusion_cpp_repo` / `_cli` | to compare stable-diffusion.cpp | only when comparing sd.cpp |

**Testing edge-dit alone**: core + the models you want to test is enough; delete the rest.

## Key rules

1. **Model key name = `local_path_ref` in the model yaml**
   `models/sd3-medium.yaml` sets `local_path_ref: sd3_medium_dir`, so the site must have `sd3_medium_dir: <path>`.
   When adding a new model, add one entry to the site matching the ref name in the model yaml.

   **Distilled models come in two classes** (check the model yaml's `format` field, determined by how the model is published on HF, not your choice):
   - **Class A — full standalone directory**: only needs the single `local_path_ref` entry (pointing to a full diffusers directory), no base model borrowed.
     i.e. `flux-schnell` (`flux_schnell_dir`), `sd35-medium-turbo` (`sd35_medium_turbo_dir`).
   - **Class B — transformer only**: besides `local_path_ref`, also needs the base `*_dir` pointed to by `base_model_ref` (borrowing its text_encoder/vae/scheduler).
     Whether its `local_path_ref` is a directory or a single file also follows the published form:
     - `kontext-lightning`: `kontext_lightning_dir` (transformer directory only) + base `flux_kontext_dir`
     - `qwen-image-lightning`: `qwen_image_lightning_weights` (single .safetensors) + base `qwen_image_dir`
     - `wan21-t2v-1.3b-distill`: `wan21_t2v_distill_weights` (single .safetensors) + base `wan2_t2v_1.3b_dir`

See the `site-example.yaml` template.
