#include "core/optimization/cache/operator/cache_operator_registry.hpp"

#include <algorithm>

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
    // cache.history_rotate is a state-manager action, not a tensor op; it has no
    // host lowering and is handled directly by CacheStateManager::rotate_history.
}

}  // namespace cache
}  // namespace edgedit
