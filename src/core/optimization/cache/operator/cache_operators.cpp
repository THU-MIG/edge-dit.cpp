#include "core/optimization/cache/operator/cache_operator_registry.hpp"

#include <algorithm>

#include "ggml.h"

namespace edgedit {
namespace cache {

void CacheOperatorRegistry::register_operator(std::unique_ptr<ICacheOperator> op) {
    if (op == nullptr) {
        return;
    }
    const std::string id = op->schema().id;
    ops_[id] = std::move(op);
}

const ICacheOperator* CacheOperatorRegistry::find(const std::string& id) const {
    auto it = ops_.find(id);
    return it == ops_.end() ? nullptr : it->second.get();
}

std::vector<std::string> CacheOperatorRegistry::ids() const {
    std::vector<std::string> out;
    out.reserve(ops_.size());
    for (const auto& kv : ops_) {
        out.push_back(kv.first);
    }
    std::sort(out.begin(), out.end());
    return out;
}

namespace {

bool same_numel(const sd::Tensor<float>& a, const sd::Tensor<float>& b) {
    return a.numel() == b.numel() && a.numel() > 0;
}

// cache.copy / cache.load / cache.store all reduce to "pass input[0] through".
// The distinction (which endpoint is a slot) is handled by the lowering, which
// wires slot handles as the input or output tensor. The operator is identity.
class IdentityOperator final : public ICacheOperator {
public:
    explicit IdentityOperator(std::string id) : id_(std::move(id)) {}
    CacheOperatorSchema schema() const override {
        return {id_, 1, 1, 1, false};
    }
    bool apply_host(const std::vector<const sd::Tensor<float>*>& inputs,
                    const CacheOperatorParams&,
                    std::vector<sd::Tensor<float>>* outputs) const override {
        if (inputs.size() != 1 || inputs[0] == nullptr || inputs[0]->empty() ||
            outputs == nullptr) {
            return false;
        }
        outputs->assign(1, *inputs[0]);
        return true;
    }

    // Passthrough: the ggml node IS the slot/temp tensor. LOAD/STORE only move a
    // handle; the lowering wires the slot view as input or output directly.
    bool lower(GraphLoweringContext&,
               const std::vector<ggml_tensor*>& inputs,
               const CacheOperatorParams&,
               std::vector<ggml_tensor*>* outputs) const override {
        if (inputs.size() != 1 || inputs[0] == nullptr || outputs == nullptr) {
            return false;
        }
        outputs->assign(1, inputs[0]);
        return true;
    }

private:
    std::string id_;
};

// out = b - a  (residual capture). inputs: [a, b].
class DifferenceOperator final : public ICacheOperator {
public:
    CacheOperatorSchema schema() const override {
        return {"cache.difference", 2, 2, 1, true};
    }
    bool apply_host(const std::vector<const sd::Tensor<float>*>& inputs,
                    const CacheOperatorParams&,
                    std::vector<sd::Tensor<float>>* outputs) const override {
        if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr ||
            outputs == nullptr || !same_numel(*inputs[0], *inputs[1])) {
            return false;
        }
        sd::Tensor<float> out(inputs[1]->shape());
        const float* a = inputs[0]->data();
        const float* b = inputs[1]->data();
        float* o = out.data();
        const int64_t n = out.numel();
        for (int64_t i = 0; i < n; ++i) {
            o[i] = b[i] - a[i];
        }
        outputs->assign(1, std::move(out));
        return true;
    }

    // out = b - a  (b = inputs[1], a = inputs[0]).
    bool lower(GraphLoweringContext& ctx,
               const std::vector<ggml_tensor*>& inputs,
               const CacheOperatorParams&,
               std::vector<ggml_tensor*>* outputs) const override {
        if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr ||
            ctx.ctx == nullptr || outputs == nullptr) {
            return false;
        }
        outputs->assign(1, ggml_sub(ctx.ctx, inputs[1], inputs[0]));
        return true;
    }
};

// out = base + sum_k coeff[k] * history[k]. inputs: [base, h0, h1, ...].
// params.floats = coefficients aligned to the history inputs (base excluded).
// Used for both linear (Δ) and polynomial (higher-order) prediction; the policy
// supplies the coefficients, so this one operator covers TaylorSeer's needs.
class LinearPredictOperator final : public ICacheOperator {
public:
    CacheOperatorSchema schema() const override {
        return {"cache.linear_predict", 1, 16, 1, true};
    }
    bool apply_host(const std::vector<const sd::Tensor<float>*>& inputs,
                    const CacheOperatorParams& params,
                    std::vector<sd::Tensor<float>>* outputs) const override {
        if (inputs.empty() || inputs[0] == nullptr || inputs[0]->empty() ||
            outputs == nullptr) {
            return false;
        }
        const int64_t n = inputs[0]->numel();
        sd::Tensor<float> out = *inputs[0];
        float* o = out.data();
        for (size_t k = 1; k < inputs.size(); ++k) {
            const size_t ci = k - 1;
            const float c = ci < params.floats.size() ? params.floats[ci] : 1.0f;
            if (inputs[k] == nullptr || inputs[k]->numel() != n) {
                return false;
            }
            const float* h = inputs[k]->data();
            for (int64_t i = 0; i < n; ++i) {
                o[i] += c * h[i];
            }
        }
        outputs->assign(1, std::move(out));
        return true;
    }
};

// out = sum_k w[k] * inputs[k]. params.floats = weights (defaults to 1.0).
class WeightedBlendOperator final : public ICacheOperator {
public:
    CacheOperatorSchema schema() const override {
        return {"cache.weighted_blend", 1, 16, 1, true};
    }
    bool apply_host(const std::vector<const sd::Tensor<float>*>& inputs,
                    const CacheOperatorParams& params,
                    std::vector<sd::Tensor<float>>* outputs) const override {
        if (inputs.empty() || inputs[0] == nullptr || inputs[0]->empty() ||
            outputs == nullptr) {
            return false;
        }
        const int64_t n = inputs[0]->numel();
        sd::Tensor<float> out(inputs[0]->shape());
        float* o = out.data();
        for (int64_t i = 0; i < n; ++i) {
            o[i] = 0.0f;
        }
        for (size_t k = 0; k < inputs.size(); ++k) {
            const float w = k < params.floats.size() ? params.floats[k] : 1.0f;
            if (inputs[k] == nullptr || inputs[k]->numel() != n) {
                return false;
            }
            const float* v = inputs[k]->data();
            for (int64_t i = 0; i < n; ++i) {
                o[i] += w * v[i];
            }
        }
        outputs->assign(1, std::move(out));
        return true;
    }

    // out = sum_k w[k] * inputs[k]. Each weight is either a compile-time constant
    // (params.floats[k], emitted via ggml_scale) or, when ctx.runtime_scalars[k]
    // is a non-null [1] device tensor, an on-device scalar (broadcast ggml_mul) —
    // this is how DiCache's on-device gamma enters the blend without a host copy.
    bool lower(GraphLoweringContext& ctx,
               const std::vector<ggml_tensor*>& inputs,
               const CacheOperatorParams& params,
               std::vector<ggml_tensor*>* outputs) const override {
        if (inputs.empty() || inputs[0] == nullptr || ctx.ctx == nullptr ||
            outputs == nullptr) {
            return false;
        }
        ggml_tensor* acc = nullptr;
        for (size_t k = 0; k < inputs.size(); ++k) {
            if (inputs[k] == nullptr) {
                return false;
            }
            ggml_tensor* term = nullptr;
            if (k < ctx.runtime_scalars.size() && ctx.runtime_scalars[k] != nullptr) {
                term = ggml_mul(ctx.ctx, inputs[k], ctx.runtime_scalars[k]);
            } else {
                const float w = k < params.floats.size() ? params.floats[k] : 1.0f;
                term = ggml_scale(ctx.ctx, inputs[k], w);
            }
            acc = acc == nullptr ? term : ggml_add(ctx.ctx, acc, term);
        }
        outputs->assign(1, acc);
        return true;
    }
};

// out = x_before + resid2 + gamma*(resid1 - resid2). inputs: [x_before, resid1,
// resid2]; gamma is a host-known compile-time constant in params.floats[0] (the
// policy already clamped it), emitted via ggml_scale — DiCache reuse never reads
// gamma back to host, and since the value is known at graph-build time it need not
// be an on-device tensor. Matches the legacy build_tap_inject DeviceBlend weave
// (sub -> scale -> add -> add), modulo scale-vs-mul kernel choice.
class GammaBlendOperator final : public ICacheOperator {
public:
    CacheOperatorSchema schema() const override { return {"cache.gamma_blend", 3, 3, 1, true}; }

    bool apply_host(const std::vector<const sd::Tensor<float>*>& inputs,
                    const CacheOperatorParams& params,
                    std::vector<sd::Tensor<float>>* outputs) const override {
        if (inputs.size() != 3 || inputs[0] == nullptr || inputs[1] == nullptr ||
            inputs[2] == nullptr || params.floats.empty() || outputs == nullptr) {
            return false;
        }
        const int64_t n = inputs[0]->numel();
        if (inputs[1]->numel() != n || inputs[2]->numel() != n) {
            return false;
        }
        const float g = params.floats[0];
        sd::Tensor<float> out(inputs[0]->shape());
        const float* xb = inputs[0]->data();
        const float* r1 = inputs[1]->data();
        const float* r2 = inputs[2]->data();
        float* o = out.data();
        for (int64_t i = 0; i < n; ++i) {
            o[i] = xb[i] + r2[i] + g * (r1[i] - r2[i]);
        }
        outputs->assign(1, std::move(out));
        return true;
    }

    // sub(resid1,resid2) -> scale(diff,gamma) -> add(resid2,·) -> add(x_before,·).
    bool lower(GraphLoweringContext& ctx,
               const std::vector<ggml_tensor*>& inputs,
               const CacheOperatorParams& params,
               std::vector<ggml_tensor*>* outputs) const override {
        if (inputs.size() != 3 || inputs[0] == nullptr || inputs[1] == nullptr ||
            inputs[2] == nullptr || ctx.ctx == nullptr || params.floats.empty() ||
            outputs == nullptr) {
            return false;
        }
        ggml_tensor* diff = ggml_sub(ctx.ctx, inputs[1], inputs[2]);        // resid1 - resid2
        ggml_tensor* scaled = ggml_scale(ctx.ctx, diff, params.floats[0]);  // gamma*(...)
        ggml_tensor* aligned = ggml_add(ctx.ctx, inputs[2], scaled);        // resid2 + ...
        outputs->assign(1, ggml_add(ctx.ctx, inputs[0], aligned));         // x_before + ...
        return true;
    }
};

}  // namespace

void register_builtin_cache_operators(CacheOperatorRegistry* registry) {
    if (registry == nullptr) {
        return;
    }
    registry->register_operator(std::make_unique<IdentityOperator>("cache.load"));
    registry->register_operator(std::make_unique<IdentityOperator>("cache.store"));
    registry->register_operator(std::make_unique<IdentityOperator>("cache.copy"));
    registry->register_operator(std::make_unique<DifferenceOperator>());
    registry->register_operator(std::make_unique<LinearPredictOperator>());
    registry->register_operator(std::make_unique<WeightedBlendOperator>());
    registry->register_operator(std::make_unique<GammaBlendOperator>());
    // cache.history_rotate is a state-manager action, not a tensor op; it has no
    // host lowering and is handled directly by CacheStateManager::rotate_history.
}

}  // namespace cache
}  // namespace edgedit
