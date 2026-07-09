#pragma once

#include <optional>
#include <vector>

#include "core/optimization/cache/model/cache_site.hpp"
#include "core/optimization/cache/model/model_schema.hpp"
#include "core/optimization/cache/model/model_topology.hpp"

namespace edgedit {
namespace cache {

// The model's side of the contract. Exposes capabilities (schema, topology,
// cache sites) — never cache algorithms. A policy negotiates its requirements
// against this; the CacheEngine builds one per run from the SDVersion plus
// whether the block-stack seam is actually usable this run (SP-parallel and
// unsupported models disable it).
class IModelCacheContract {
public:
    virtual ~IModelCacheContract() = default;

    virtual const ModelSchema& schema() const = 0;
    virtual const ModelTopology& topology() const = 0;
    virtual const std::vector<CacheSiteDesc>& cache_sites() const = 0;

    bool supports(CacheSiteRole role) const {
        for (const auto& s : cache_sites()) {
            if (s.role == role) {
                return true;
            }
        }
        return false;
    }

    std::optional<CacheSiteDesc> find_site(CacheSiteRole role) const {
        for (const auto& s : cache_sites()) {
            if (s.role == role) {
                return s;
            }
        }
        return std::nullopt;
    }
};

}  // namespace cache
}  // namespace edgedit
