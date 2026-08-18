#ifndef __ED_MODEL_IO_CONVERT_H__
#define __ED_MODEL_IO_CONVERT_H__

#include "edge-dit.h"  // ed_dtype_t

// Offline model converter / quantizer.
//
// There are two mutually exclusive input forms:
//
//   * `model_path`: a complete model directory. It is converted to one GGUF.
//   * one or more explicitly named component paths: diffusion model, VAE, audio
//     VAE, CLIP-L, CLIP-G, T5-XXL, LLM, and LLM vision. One component produces
//     a typed component GGUF; multiple components are merged into one complete
//     model GGUF.
//
// A component path may be a .safetensors file, a safetensors shard index, a
// previously converted GGUF, or a component directory containing one of those
// standard files. Its command-line argument supplies the semantic identity, so
// no separate component/model-version hint is needed. Pass nullptr / "" for any
// unused input. Applies the target weight type `output_type` plus optional per-tensor
// `tensor_type_rules` (same "name-regex=ggml-type,..." syntax as the CLI
// --tensor-type-rules), and writes a single file to `output_path`. The output
// file is always GGUF; a `.safetensors` output path is rejected.
//
// `convert_name` controls tensor naming and DEFAULTS TO TRUE: the tensor names
// are canonicalized (ModelLoader::convert_tensors_name) before writing. This is
// required for a reusable pre-quantized GGUF, because a GGUF carries no
// config.json / directory layout, so on reload the engine must recover the
// model version purely from tensor names -- and ModelLoader::get_ld_version()
// keys on canonical names (e.g. "joint_blocks." for SD3). The engine re-runs
// convert_tensors_name() on load; convert_tensor_name is idempotent for
// already-canonical names (its prefix maps only match diffusers-style keys), so
// the double pass is a no-op. This mirrors how stable-diffusion.cpp produces and
// consumes pre-quantized GGUF files. Pass false only when the source is already
// canonical and you want to skip conversion.
//
// Returns true on success. Errors are reported through LOG_ERROR.
//
// `imatrix_path` (optional): path to an activation-calibrated imatrix GGUF (see
// tools/imatrix/calibrate.py) holding a per-input-channel importance vector
// per weight, keyed by the ORIGINAL diffusers name. When provided, convert maps
// each imatrix key to its canonical edge tensor name and hands the matching
// vector to the quantizer, so q4_k/q8_0 rounding preserves the most important
// (high-activation) channels. Tensors without a matching, length-compatible
// entry fall back to the historical all-ones weighting. Pass nullptr to keep
// plain quantization (identical to before this feature).
//
// Component GGUFs are written with canonical tensor prefixes and
// `edgedit.component_kind` metadata. The loader uses that metadata to validate
// the matching inference flag and to rebase CLIP prefixes for UNet versus DiT
// pipelines. Component and merged output is GGUF-only and cannot use raw names.
bool convert(const char* model_path,
             const char* diffusion_model_path,
             const char* vae_path,
             const char* audio_vae_path,
             const char* clip_l_path,
             const char* clip_g_path,
             const char* t5xxl_path,
             const char* llm_path,
             const char* llm_vision_path,
             const char* output_path,
             ed_dtype_t output_type,
             const char* tensor_type_rules = nullptr,
             bool convert_name = true,
             const char* imatrix_path = nullptr);

#endif  // __ED_MODEL_IO_CONVERT_H__
