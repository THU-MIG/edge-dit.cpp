#ifndef __ED_MODEL_IO_GGUF_IO_H__
#define __ED_MODEL_IO_GGUF_IO_H__

#include <string>
#include <vector>

#include "tensor_storage.h"

bool is_gguf_file(const std::string& file_path);
bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error = nullptr,
                    std::string* model_version = nullptr);
bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     const std::string& model_version = "",
                     std::string* error = nullptr);

#endif  // __ED_MODEL_IO_GGUF_IO_H__
