#pragma once

#include "core/optimization/cache/cache_types.hpp"  // CacheBranch
#include "core/optimization/cache/ir/cache_program.hpp"
#include "utils/tensor.hpp"

namespace edgedit {
namespace cache {

// Per-step runtime context handed to ICachePolicy::decide(). Pure scalars — no
// tensors — so decisions that don't need model output stay one-phase.
struct StepContext {
    int step_index = -1;
    int num_steps = 0;
    float sigma = 0.0f;
    float sigma_next = 0.0f;
    bool is_first_step = false;
    bool is_last_step = false;
};

// Metrics + per-branch identity a decide() call needs: which CFG branch is being
// run (so the policy selects the right isolated state), plus any measured
// movement from a probe pass or the prior committed step.
struct CacheRuntimeMetrics {
    CacheBranch branch = CacheBranch::Main;
    const void* condition_key = nullptr;
    const sd::Tensor<float>* input = nullptr;  // block-stack input latent

    bool has_probe = false;
    float probe_rel_l1 = 0.0f;   // relative-L1 movement of the probe state
    float input_change = 0.0f;   // movement of the block-stack input latent
};

// What the policy decided for this step: which variant to execute, in which
// phase, and whether the resulting state should be committed.
struct RuntimeDecision {
    GraphVariantId variant = -1;
    CachePhase phase = CachePhase::FORWARD;
    bool commit_cache_state = true;
};

}  // namespace cache
}  // namespace edgedit
