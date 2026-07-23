#ifndef __ED_MODEL_IO_CONVERT_H__
#define __ED_MODEL_IO_CONVERT_H__

#include "edge-dit.h"  // ed_dtype_t

// Offline model converter / quantizer.
//
// Loads a model from `input_path` (diffusers directory, .safetensors, shard
// index, or .gguf), optionally merges an external VAE from `vae_path`, applies
// the target weight type `output_type` plus optional per-tensor
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
bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             ed_dtype_t output_type,
             const char* tensor_type_rules = nullptr,
             bool convert_name = true);

#endif  // __ED_MODEL_IO_CONVERT_H__
