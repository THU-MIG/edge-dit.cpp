# Merging LoRA weights into a base model

Some few-step distilled variants are published as **LoRA adapters** rather than
full checkpoints. edge-dit.cpp's few-step loading path takes a full-weight
transformer via `--diffusion-model`, so a LoRA must first be **merged into the
base model's weights offline**, producing a standalone full-weight transformer.

## Which models need this

Of the six distilled variants used in the consumer-GPU work, **only the two
Qwen-Image ones require merging**; the rest ship as ready-to-use full weights.

| Distilled variant | Published as | Needs merge? |
|---|---|---|
| FLUX.1-schnell | Full diffusers directory (also has a standalone `transformer/`) | No — download directly |
| FLUX.1-Kontext Lightning | Full diffusers directory (also has a standalone `transformer/`) | No — download directly |
| SD3.5-medium-turbo | Full diffusers directory | No — download directly |
| Wan2.1-T2V-1.3B Distill | Single full-weight file | No — drop-in |
| **Qwen-Image Lightning** | **LoRA adapter** | **Yes** |
| **Qwen-Image-Edit Lightning** | **LoRA adapter** | **Yes** |

Only Qwen's few-step release is LoRA-only, so those two are the ones you merge
yourself.

## What "merge" means

A LoRA is not a runnable model. It stores low-rank update matrices (a "down"
projection A and an "up" projection B) plus a scale. Merging folds that update
back into every matching base weight matrix:

```
W' = W + (alpha / rank) * (B @ A)
```

The output is a new full-weight transformer the same size as the base (~39 GiB
for Qwen-Image), not the small LoRA (~850 MiB). Analogy: the base is the book,
the LoRA is a page of edits, and merging reprints the book with the edits
applied.

## Ingredients

1. **Base model** (full weights, diffusers layout with a
   `diffusion_pytorch_model.safetensors.index.json` + shards):
   - Qwen-Image — HuggingFace `Qwen/Qwen-Image`
   - Qwen-Image-Edit — HuggingFace `Qwen/Qwen-Image-Edit`
   (Confirm the exact repo names on HuggingFace.)
2. **The LoRA adapter** from `lightx2v/Qwen-Image-Lightning`:
   - `Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors` (for Qwen-Image)
   - `Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors` (for Qwen-Image-Edit)
3. **The merge script**: `scripts/merge_qwen_lora.py`.

## Steps

```bash
export HF_ENDPOINT=https://hf-mirror.com   # mirror; optional

# 1. Download the LoRA adapter
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="lightx2v/Qwen-Image-Lightning",
                filename="Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors",
                local_dir="/path/to/models/qwen-image-lightning")
PY

# 2. Merge LoRA into the base transformer
#    args: <base transformer dir> <lora .safetensors> <output dir>
python scripts/merge_qwen_lora.py \
  /path/to/models/Qwen-Image/transformer \
  /path/to/models/qwen-image-lightning/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors \
  /path/to/models/qwen-image-lightning-merged/transformer

# 3. Run with the merged full-weight transformer
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/models/Qwen-Image \
  --diffusion-model /path/to/models/qwen-image-lightning-merged/transformer/diffusion_pytorch_model.safetensors.index.json \
  --steps 8 --cfg-scale 1.0 -W 1024 -H 1024 \
  --prompt "a photorealistic red apple on a wooden table" -o qwen_lightning.png
```

For **Qwen-Image-Edit**, use the base `Qwen/Qwen-Image-Edit/transformer`, the
`Qwen-Image-Edit-Lightning-4steps` LoRA file, and a `.../dit/` output dir; run
with `--model .../qwen-image-edit --qwen-image-zero-cond-t -i <input image>`.

## How the script works

`merge_qwen_lora.py` (args `base_dir lora_path out_dir`):

1. Reads the base `diffusion_pytorch_model.safetensors.index.json` and groups
   weight keys by shard.
2. Parses the LoRA into modules by the suffixes `.lora_down.weight`,
   `.lora_up.weight`, `.alpha`.
3. For each base `*.weight` with a matching LoRA module, computes
   `delta = (alpha/rank) * (up @ down)` and writes `W' = (W.float() + delta)`
   back in the original dtype (shape-mismatched pairs are skipped with a log).
4. Copies `config.json` + the index unchanged and re-saves each shard, so the
   output is a drop-in diffusers transformer directory.
5. Asserts every LoRA module was merged (`merged_count == len(mods)`), so a
   naming mismatch fails loudly instead of silently producing a bad model.
