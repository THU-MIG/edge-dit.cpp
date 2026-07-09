#pragma once

#include <string>
#include <vector>

#include "core/optimization/cache/ir/cache_requirements.hpp"
#include "core/optimization/cache/model/model_cache_contract.hpp"

namespace edgedit {
namespace cache {

enum class CacheErrorCode {
    Ok,
    UnsupportedModelCapability,
    SiteNotReplaceable,
    SiteNotProbeable,
};

struct ValidationResult {
    bool ok = true;
    CacheErrorCode code = CacheErrorCode::Ok;
    std::string message;  // doc §8.4-style: required vs. exposed
};

// Checks a policy's requirements against what the model contract exposes this
// run. On mismatch, returns an explicit diagnostic (never a silent no-op). The
// engine decides whether to hard-fail or, if allow_fallback is set, drop to
// no-cache and log the final plan.
ValidationResult validate_requirements(const CacheRequirements& requirements,
                                       const IModelCacheContract& contract);

}  // namespace cache
}  // namespace edgedit
