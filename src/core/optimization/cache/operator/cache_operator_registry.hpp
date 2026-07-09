#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/optimization/cache/operator/cache_operator.hpp"

namespace edgedit {
namespace cache {

// Owns the set of built-in cache operators, looked up by id. Populated once via
// register_builtin_cache_operators(); policies reference operators by id in
// their programs and the lowering resolves them here.
class CacheOperatorRegistry {
public:
    void register_operator(std::unique_ptr<ICacheOperator> op);
    const ICacheOperator* find(const std::string& id) const;

    std::vector<std::string> ids() const;

private:
    std::unordered_map<std::string, std::unique_ptr<ICacheOperator>> ops_;
};

// Registers cache.load / cache.store / cache.difference / cache.linear_predict /
// cache.weighted_blend / cache.history_rotate. Idempotent per registry.
void register_builtin_cache_operators(CacheOperatorRegistry* registry);

}  // namespace cache
}  // namespace edgedit
