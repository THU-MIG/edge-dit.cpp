#pragma once

#include <vector>

#include "core/optimization/cache/cache_types.hpp"      // CacheGranularity
#include "core/optimization/cache/model/cache_site.hpp"

namespace edgedit {
namespace cache {

struct RequiredSite {
    CacheSiteRole role = CacheSiteRole::DENOISER_OUTPUT;
    bool replaceable = false;         // method needs to inject over this site
    bool supports_partial = false;    // method needs sub-block cuts
};

// What a policy needs from the model + backend. Checked at init by
// validate_requirements(); a mismatch is a hard error (or an explicit, logged
// fallback), never a silent no-op.
struct CacheRequirements {
    std::vector<RequiredSite> required_sites;

    int history_length = 1;   // ring-buffer depth (K-order predict needs K+1)
    bool requires_probe = false;
    bool requires_partial_compute = false;
    bool requires_future_anchor = false;  // trajectory-correction (Phase 4)

    CacheGranularity granularity = CacheGranularity::Output;
};

}  // namespace cache
}  // namespace edgedit
