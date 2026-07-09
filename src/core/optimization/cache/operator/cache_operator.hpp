#pragma once

#include <string>
#include <vector>

#include "core/optimization/cache/ir/cache_action.hpp"
#include "utils/tensor.hpp"

namespace edgedit {
namespace cache {

// Static description of an operator's arity/shape contract, used for validation
// and diagnostics.
struct CacheOperatorSchema {
    std::string id;
    int min_inputs = 0;
    int max_inputs = 0;
    int num_outputs = 1;
    bool requires_same_shape = true;
};

// Model-agnostic cache math. Option A operators work on host tensors
// (sd::Tensor<float>); the deferred backend path adds a ggml-lowering overload
// without changing the policy-facing IR. A policy only references operators by
// id in its CacheAction list — it never calls these directly.
class ICacheOperator {
public:
    virtual ~ICacheOperator() = default;

    virtual CacheOperatorSchema schema() const = 0;

    // Host lowering: compute outputs from inputs under params. Returns false on
    // a shape/precondition failure so the caller can fall back to full compute.
    virtual bool apply_host(const std::vector<const sd::Tensor<float>*>& inputs,
                            const CacheOperatorParams& params,
                            std::vector<sd::Tensor<float>>* outputs) const = 0;
};

}  // namespace cache
}  // namespace edgedit
