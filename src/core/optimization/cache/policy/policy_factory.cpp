#include "core/optimization/cache/policy/cache_policy.hpp"

namespace edgedit {
namespace cache {

// Per-method creators, defined in the policies/*.cpp translation units.
std::unique_ptr<ICachePolicy> create_null_policy();
std::unique_ptr<ICachePolicy> create_easycache_policy();
std::unique_ptr<ICachePolicy> create_ucache_policy();
std::unique_ptr<ICachePolicy> create_condition_policy();  // DBCache + CacheDiT
std::unique_ptr<ICachePolicy> create_taylorseer_policy();
std::unique_ptr<ICachePolicy> create_magcache_policy();
std::unique_ptr<ICachePolicy> create_dicache_policy();
std::unique_ptr<ICachePolicy> create_sencache_policy();

std::unique_ptr<ICachePolicy> create_cache_policy(CacheMode mode) {
    switch (mode) {
        case CacheMode::EasyCache: return create_easycache_policy();
        case CacheMode::UCache: return create_ucache_policy();
        case CacheMode::DBCache:
        case CacheMode::CacheDiT: return create_condition_policy();
        case CacheMode::TaylorSeer: return create_taylorseer_policy();
        case CacheMode::MagCache: return create_magcache_policy();
        case CacheMode::DiCache: return create_dicache_policy();
        case CacheMode::SenCache: return create_sencache_policy();
        case CacheMode::Disabled:
        default: return create_null_policy();
    }
}

}  // namespace cache
}  // namespace edgedit
