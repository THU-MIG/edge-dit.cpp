#ifndef __ED_MODEL_IO_CONVERT_H__
#define __ED_MODEL_IO_CONVERT_H__

#include "edge-dit.h"  // ed_dtype_t

// Offline model converter / quantizer.
//
// Loads a model from `input_path` (diffusers directory, .safetensors, shard
// index, or .gguf), optionally merges external components -- VAE (`vae_path`),
// CLIP-L (`clip_l_path`), CLIP-G (`clip_g_path`), and T5-XXL (`t5xxl_path`) --
// so a bare transformer file can be combined with its text encoders / VAE into
// one self-contained GGUF that reloads standalone. Pass nullptr / "" for any
// component to skip it; skipping `t5xxl_path` is the offline equivalent of
// `--no-t5`. Applies the target weight type `output_type` plus optional per-tensor
// `tensor_type_rules` (same "name-regex=ggml-type,..." syntax as the CLI
// --tensor-type-rules), and writes a single file to `output_path`. The output
// format is chosen by extension: ".safetensors" -> safetensors, anything else
// (e.g. ".gguf") -> GGUF.
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
bool convert(const char* input_path,
             const char* vae_path,
             const char* clip_l_path,
             const char* clip_g_path,
             const char* t5xxl_path,
             const char* output_path,
             ed_dtype_t output_type,
             const char* tensor_type_rules = nullptr,
             bool convert_name = true,
             const char* imatrix_path = nullptr);

#endif  // __ED_MODEL_IO_CONVERT_H__
