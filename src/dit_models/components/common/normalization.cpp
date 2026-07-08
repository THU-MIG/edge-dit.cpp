#include "dit_models/components/common/normalization.hpp"

#include "backend/ggml/ed_ggml_norm_ext.hpp"

#include <utility>

namespace dit {

RMSNorm::RMSNorm(
    int64_t hidden_size,
    float eps,
    std::string weight_name)
    : hidden_size(hidden_size),
      eps(eps),
      weight_name(std::move(weight_name)) {
}

void RMSNorm::init_params(
    ggml_context* ctx,
    const String2TensorStorage&,
    const std::string) {
    ggml_type wtype = GGML_TYPE_F32;
    params[weight_name] = ggml_new_tensor_1d(ctx, wtype, hidden_size);
}

ggml_tensor* RMSNorm::forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
    ggml_tensor* w = params[weight_name];
    x = ggml_rms_norm(ctx->ggml_ctx, x, eps);
    x = ggml_mul(ctx->ggml_ctx, x, w);
    return x;
}

ggml_tensor* RMSNorm::forward_f16(GGMLRunnerContext* ctx, ggml_tensor* x) {
    ggml_tensor* w = params[weight_name];
    if (auto fused = edgedit::ggml_ext::rms_norm_mul_f16_custom(ctx->ggml_ctx, x, w, eps)) {
        return fused;
    }
    return ggml_cast(ctx->ggml_ctx, forward(ctx, x), GGML_TYPE_F16);
}

QKNorm::QKNorm(
    int64_t dim,
    float eps,
    std::string weight_name) {
    blocks["query_norm"] = std::make_shared<RMSNorm>(dim, eps, weight_name);
    blocks["key_norm"]   = std::make_shared<RMSNorm>(dim, eps, weight_name);
}

ggml_tensor* QKNorm::query_norm(GGMLRunnerContext* ctx, ggml_tensor* x) {
    auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["query_norm"]);
    return norm->forward(ctx, x);
}

ggml_tensor* QKNorm::key_norm(GGMLRunnerContext* ctx, ggml_tensor* x) {
    auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["key_norm"]);
    return norm->forward(ctx, x);
}

ggml_tensor* QKNorm::query_norm_f16(GGMLRunnerContext* ctx, ggml_tensor* x) {
    auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["query_norm"]);
    return norm->forward_f16(ctx, x);
}

ggml_tensor* QKNorm::key_norm_f16(GGMLRunnerContext* ctx, ggml_tensor* x) {
    auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["key_norm"]);
    return norm->forward_f16(ctx, x);
}

}  // namespace dit
