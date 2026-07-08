#ifndef __DIT_NORMALIZATION_HPP__
#define __DIT_NORMALIZATION_HPP__

#include <memory>
#include <string>

#include "backend/ggml/ggml_extend.hpp"

namespace dit {

class RMSNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;
    std::string weight_name;

    void init_params(
        ggml_context* ctx,
        const String2TensorStorage& tensor_storage_map = {},
        const std::string prefix = "") override;

public:
    RMSNorm(
        int64_t hidden_size,
        float eps = 1e-06f,
        std::string weight_name = "scale");

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override;
    ggml_tensor* forward_f16(GGMLRunnerContext* ctx, ggml_tensor* x);
};

struct QKNorm : public GGMLBlock {
public:
    QKNorm(
        int64_t dim,
        float eps = 1e-06f,
        std::string weight_name = "scale");

    ggml_tensor* query_norm(GGMLRunnerContext* ctx, ggml_tensor* x);
    ggml_tensor* key_norm(GGMLRunnerContext* ctx, ggml_tensor* x);
    ggml_tensor* query_norm_f16(GGMLRunnerContext* ctx, ggml_tensor* x);
    ggml_tensor* key_norm_f16(GGMLRunnerContext* ctx, ggml_tensor* x);
};

}  // namespace dit

#endif  // __DIT_NORMALIZATION_HPP__
