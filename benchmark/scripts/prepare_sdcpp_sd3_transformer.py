#!/usr/bin/env python3
"""Prepare an SD3 Diffusers transformer safetensors file for stable-diffusion.cpp.

stable-diffusion.cpp commit b5d8120 can read safetensors index files, but its
SD3 version detection expects original SD3 transformer tensor names before it
knows to run the Diffusers-to-original name conversion. This helper rewrites
only tensor keys for the diffusion transformer and leaves tensor values
unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import torch
from safetensors.torch import load_file, save_file


BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")


BASE_MAP = {
    "time_text_embed.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
    "time_text_embed.timestep_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
    "time_text_embed.timestep_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
    "time_text_embed.text_embedder.linear_1.weight": "y_embedder.mlp.0.weight",
    "time_text_embed.text_embedder.linear_1.bias": "y_embedder.mlp.0.bias",
    "time_text_embed.text_embedder.linear_2.weight": "y_embedder.mlp.2.weight",
    "time_text_embed.text_embedder.linear_2.bias": "y_embedder.mlp.2.bias",
    "pos_embed.pos_embed": "pos_embed",
    "pos_embed.proj.weight": "x_embedder.proj.weight",
    "pos_embed.proj.bias": "x_embedder.proj.bias",
    "proj_out.weight": "final_layer.linear.weight",
    "proj_out.bias": "final_layer.linear.bias",
    "norm_out.linear.weight": "final_layer.adaLN_modulation.1.weight",
    "norm_out.linear.bias": "final_layer.adaLN_modulation.1.bias",
}


BLOCK_SUFFIX_MAP = {
    "norm1.linear.weight": "x_block.adaLN_modulation.1.weight",
    "norm1.linear.bias": "x_block.adaLN_modulation.1.bias",
    "norm1_context.linear.weight": "context_block.adaLN_modulation.1.weight",
    "norm1_context.linear.bias": "context_block.adaLN_modulation.1.bias",
    "attn.to_q.weight": "x_block.attn.qkv.weight",
    "attn.to_q.bias": "x_block.attn.qkv.bias",
    "attn.to_k.weight": "x_block.attn.qkv.weight.1",
    "attn.to_k.bias": "x_block.attn.qkv.bias.1",
    "attn.to_v.weight": "x_block.attn.qkv.weight.2",
    "attn.to_v.bias": "x_block.attn.qkv.bias.2",
    "attn.add_q_proj.weight": "context_block.attn.qkv.weight",
    "attn.add_q_proj.bias": "context_block.attn.qkv.bias",
    "attn.add_k_proj.weight": "context_block.attn.qkv.weight.1",
    "attn.add_k_proj.bias": "context_block.attn.qkv.bias.1",
    "attn.add_v_proj.weight": "context_block.attn.qkv.weight.2",
    "attn.add_v_proj.bias": "context_block.attn.qkv.bias.2",
    "attn2.to_q.weight": "x_block.attn2.qkv.weight",
    "attn2.to_q.bias": "x_block.attn2.qkv.bias",
    "attn2.to_k.weight": "x_block.attn2.qkv.weight.1",
    "attn2.to_k.bias": "x_block.attn2.qkv.bias.1",
    "attn2.to_v.weight": "x_block.attn2.qkv.weight.2",
    "attn2.to_v.bias": "x_block.attn2.qkv.bias.2",
    "attn2.add_q_proj.weight": "context_block.attn2.qkv.weight",
    "attn2.add_q_proj.bias": "context_block.attn2.qkv.bias",
    "attn2.add_k_proj.weight": "context_block.attn2.qkv.weight.1",
    "attn2.add_k_proj.bias": "context_block.attn2.qkv.bias.1",
    "attn2.add_v_proj.weight": "context_block.attn2.qkv.weight.2",
    "attn2.add_v_proj.bias": "context_block.attn2.qkv.bias.2",
    "attn.norm_q.weight": "x_block.attn.ln_q.weight",
    "attn.norm_k.weight": "x_block.attn.ln_k.weight",
    "attn.norm_added_q.weight": "context_block.attn.ln_q.weight",
    "attn.norm_added_k.weight": "context_block.attn.ln_k.weight",
    "attn2.norm_q.weight": "x_block.attn2.ln_q.weight",
    "attn2.norm_k.weight": "x_block.attn2.ln_k.weight",
    "ff.net.0.proj.weight": "x_block.mlp.fc1.weight",
    "ff.net.0.proj.bias": "x_block.mlp.fc1.bias",
    "ff.net.2.weight": "x_block.mlp.fc2.weight",
    "ff.net.2.bias": "x_block.mlp.fc2.bias",
    "ff_context.net.0.proj.weight": "context_block.mlp.fc1.weight",
    "ff_context.net.0.proj.bias": "context_block.mlp.fc1.bias",
    "ff_context.net.2.weight": "context_block.mlp.fc2.weight",
    "ff_context.net.2.bias": "context_block.mlp.fc2.bias",
    "attn.to_out.0.weight": "x_block.attn.proj.weight",
    "attn.to_out.0.bias": "x_block.attn.proj.bias",
    "attn.to_add_out.weight": "context_block.attn.proj.weight",
    "attn.to_add_out.bias": "context_block.attn.proj.bias",
    "attn2.to_out.0.weight": "x_block.attn2.proj.weight",
    "attn2.to_out.0.bias": "x_block.attn2.proj.bias",
    "attn2.to_add_out.weight": "context_block.attn2.proj.weight",
    "attn2.to_add_out.bias": "context_block.attn2.proj.bias",
}

QKV_MERGE_SPECS = [
    (
        "attn",
        "x_block.attn.qkv",
        {
            "q": "attn.to_q",
            "k": "attn.to_k",
            "v": "attn.to_v",
        },
    ),
    (
        "attn_context",
        "context_block.attn.qkv",
        {
            "q": "attn.add_q_proj",
            "k": "attn.add_k_proj",
            "v": "attn.add_v_proj",
        },
    ),
    (
        "attn2",
        "x_block.attn2.qkv",
        {
            "q": "attn2.to_q",
            "k": "attn2.to_k",
            "v": "attn2.to_v",
        },
    ),
    (
        "attn2_context",
        "context_block.attn2.qkv",
        {
            "q": "attn2.add_q_proj",
            "k": "attn2.add_k_proj",
            "v": "attn2.add_v_proj",
        },
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"{args.output} already exists")
        return 0

    tensors = load_file(str(args.input), device="cpu")
    renamed = merge_qkv_tensors(tensors)
    for name, tensor in tensors.items():
        if is_merged_qkv_source(name):
            continue
        new_name = convert_name(name)
        if new_name in renamed:
            raise RuntimeError(f"duplicate converted tensor name: {name} -> {new_name}")
        renamed[new_name] = tensor

    prefixed = {f"model.diffusion_model.{name}": tensor for name, tensor in renamed.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(prefixed, str(args.output))
    print(f"wrote {len(prefixed)} tensors to {args.output}")
    return 0


def convert_name(name: str) -> str:
    mapped = BASE_MAP.get(name)
    if mapped is not None:
        return mapped
    match = BLOCK_RE.match(name)
    if match is None:
        return name
    block_index, suffix = match.groups()
    mapped_suffix = BLOCK_SUFFIX_MAP.get(suffix)
    if mapped_suffix is None:
        return name
    return f"joint_blocks.{block_index}.{mapped_suffix}"


def merge_qkv_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    block_ids = sorted(
        {
            int(match.group(1))
            for name in tensors
            if (match := BLOCK_RE.match(name)) is not None
        }
    )
    for block_id in block_ids:
        for _, target_suffix, source_prefixes in QKV_MERGE_SPECS:
            for suffix in ("weight", "bias"):
                source_names = [
                    f"transformer_blocks.{block_id}.{source_prefixes[item]}.{suffix}"
                    for item in ("q", "k", "v")
                ]
                if not all(name in tensors for name in source_names):
                    continue
                target_name = f"joint_blocks.{block_id}.{target_suffix}.{suffix}"
                merged[target_name] = torch.cat([tensors[name] for name in source_names], dim=0)
    return merged


def is_merged_qkv_source(name: str) -> bool:
    match = BLOCK_RE.match(name)
    if match is None:
        return False
    suffix = match.group(2)
    for _, _, source_prefixes in QKV_MERGE_SPECS:
        for source_prefix in source_prefixes.values():
            if suffix == f"{source_prefix}.weight" or suffix == f"{source_prefix}.bias":
                return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
