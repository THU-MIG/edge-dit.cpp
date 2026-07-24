#ifndef __ED_MODEL_IO_GGUF_IO_H__
#define __ED_MODEL_IO_GGUF_IO_H__

#include <map>
#include <string>
#include <vector>

#include "tensor_storage.h"

bool is_gguf_file(const std::string& file_path);

// Reads an activation-calibrated imatrix GGUF (as produced by tools/imatrix/calibrate.py):
// each tensor holds a per-input-channel importance vector (E[x^2]) stored as F32
// under the original diffusers weight name (e.g. "transformer_blocks.0.attn.to_q.weight").
// On success, fills `imatrix` with name -> flattened float vector (length = in_features).
// Only F32 tensors are accepted; anything else is skipped with a warning.
bool read_imatrix_gguf(const std::string& file_path,
                       std::map<std::string, std::vector<float>>& imatrix,
                       std::string* error = nullptr);
bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error = nullptr,
                    std::string* model_version = nullptr);
bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     const std::string& model_version = "",
                     std::string* error = nullptr);

#endif  // __ED_MODEL_IO_GGUF_IO_H__
