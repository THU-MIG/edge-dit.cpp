#pragma once

#include <string>
#include <vector>

#include "core/optimization/cache/ir/cache_action.hpp"
#include "utils/tensor.hpp"

struct ggml_context;
struct ggml_tensor;

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

// Context for the ggml-lowering path: the compute graph's context the operator
// emits nodes into. `runtime_scalars` carries per-step coefficients the operator
// may need as broadcastable [1] device tensors (e.g. DiCache's on-device gamma),
// aligned to params.floats when the policy computes weights at runtime; empty
// when the operator should use params.floats as compile-time constants.
struct GraphLoweringContext {
    ggml_context* ctx = nullptr;
    std::vector<ggml_tensor*> runtime_scalars;
};

// Model-agnostic cache math. Option A operators work on host tensors
// (sd::Tensor<float>); the backend path lowers the SAME operator to ggml nodes
// via lower(), so the on-device reuse math is driven by the program's actions
// rather than hardcoded in the runner seam. A policy only references operators by
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

    // ggml lowering: emit nodes into ctx.ctx computing the same math as
    // apply_host, with graph-tensor inputs (e.g. device cache-slot views). The
    // default returns false so an operator without a ggml lowering falls back to
    // the host path. Prefer composing ordinary ggml ops (redesign §10.4).
    virtual bool lower(GraphLoweringContext& ctx,
                       const std::vector<ggml_tensor*>& inputs,
                       const CacheOperatorParams& params,
                       std::vector<ggml_tensor*>* outputs) const {
        (void)ctx;
        (void)inputs;
        (void)params;
        (void)outputs;
        return false;
    }
};

}  // namespace cache
}  // namespace edgedit
