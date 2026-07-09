#pragma once

#include <vector>

#include "core/optimization/cache/model/model_cache_contract.hpp"
#include "runtime/model_loader.h"  // SDVersion

namespace edgedit {
namespace cache {

// The one IModelCacheContract implementation for all DiT models. Built per run
// from the SDVersion plus whether the block-stack seam is genuinely usable this
// run: when seam_available is false (SP-parallel, or a model whose forward()
// can't cut its stack), the block-stack sites are exposed as non-replaceable so
// capability negotiation reports an explicit unsupported result instead of the
// old silent full-compute fallback.
class DiTModelCacheContract final : public IModelCacheContract {
public:
    DiTModelCacheContract(SDVersion version, bool seam_available);

    const ModelSchema& schema() const override { return schema_; }
    const ModelTopology& topology() const override { return topology_; }
    const std::vector<CacheSiteDesc>& cache_sites() const override { return sites_; }

private:
    ModelSchema schema_;
    ModelTopology topology_;
    std::vector<CacheSiteDesc> sites_;
};

}  // namespace cache
}  // namespace edgedit
