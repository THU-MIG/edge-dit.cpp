#pragma once

#include <string>

#include "core/optimization/cache/ir/indicator.hpp"
#include "core/optimization/cache/model/tap_registry.hpp"
#include "ggml.h"

namespace edgedit {
namespace cache {

// Weave one Indicator into the compute graph as a scalar reduction over the
// tapped anchor tensors, returning the scalar node (named "cache_ind:<name>").
// The runner pins + reads it back post-compute; only the scalar leaves the
// device (the on-device red line — large anchor tensors never round-trip).
//
// Semantics match the legacy build_probe_metrics rel():
//   RelL1(cur, ref) = sum|cur - ref| / sum|ref|
//   L2Norm(a)       = sqrt(sum(a^2))
//   L2Delta(a, b)   = sqrt(sum((a-b)^2))
//   Dot(a, b)       = sum(a * b)
// Returns nullptr if a required anchor was not tapped this build.
inline ggml_tensor* lower_indicator(ggml_context* ctx, const Indicator& ind,
                                    const TapRegistry& taps) {
    auto A = [&](size_t i) -> ggml_tensor* {
        return i < ind.anchors.size() ? taps.get(ind.anchors[i]) : nullptr;
    };
    ggml_tensor* out = nullptr;
    switch (ind.kind) {
        case Indicator::RelL1: {
            ggml_tensor* cur = A(0);
            ggml_tensor* ref = A(1);
            if (cur == nullptr || ref == nullptr) {
                return nullptr;
            }
            ggml_tensor* num = ggml_sum(ctx, ggml_abs(ctx, ggml_sub(ctx, cur, ref)));
            ggml_tensor* den = ggml_sum(ctx, ggml_abs(ctx, ref));
            out = ggml_div(ctx, num, den);
            break;
        }
        case Indicator::L2Norm: {
            ggml_tensor* a = A(0);
            if (a == nullptr) {
                return nullptr;
            }
            out = ggml_sqrt(ctx, ggml_sum(ctx, ggml_sqr(ctx, a)));
            break;
        }
        case Indicator::L2Delta: {
            ggml_tensor* a = A(0);
            ggml_tensor* b = A(1);
            if (a == nullptr || b == nullptr) {
                return nullptr;
            }
            ggml_tensor* d = ggml_sub(ctx, a, b);
            out = ggml_sqrt(ctx, ggml_sum(ctx, ggml_sqr(ctx, d)));
            break;
        }
        case Indicator::Dot: {
            ggml_tensor* a = A(0);
            ggml_tensor* b = A(1);
            if (a == nullptr || b == nullptr) {
                return nullptr;
            }
            out = ggml_sum(ctx, ggml_mul(ctx, a, b));
            break;
        }
    }
    if (out != nullptr) {
        const std::string name = "cache_ind:" + ind.name;
        ggml_set_name(out, name.c_str());
    }
    return out;
}

}  // namespace cache
}  // namespace edgedit
