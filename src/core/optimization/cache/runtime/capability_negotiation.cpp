#include "core/optimization/cache/runtime/capability_negotiation.hpp"

#include <sstream>

namespace edgedit {
namespace cache {

namespace {

const char* role_name(CacheSiteRole role) {
    switch (role) {
        case CacheSiteRole::DENOISER_OUTPUT: return "DENOISER_OUTPUT";
        case CacheSiteRole::BLOCK_STACK_OUTPUT: return "BLOCK_STACK_OUTPUT";
        case CacheSiteRole::BLOCK_STACK_PROBE: return "BLOCK_STACK_PROBE";
    }
    return "?";
}

std::string exposed_sites(const IModelCacheContract& contract) {
    std::ostringstream os;
    for (const auto& s : contract.cache_sites()) {
        os << "\n  - " << role_name(s.role) << " (readable=" << s.capability.readable
           << " replaceable=" << s.capability.replaceable << ")";
    }
    return os.str();
}

}  // namespace

ValidationResult validate_requirements(const CacheRequirements& requirements,
                                       const IModelCacheContract& contract) {
    for (const RequiredSite& req : requirements.required_sites) {
        auto site = contract.find_site(req.role);
        if (!site.has_value()) {
            std::ostringstream os;
            os << "cache method requires site " << role_name(req.role)
               << " which the model does not expose. Exposed sites:" << exposed_sites(contract);
            return {false, CacheErrorCode::UnsupportedModelCapability, os.str()};
        }
        if (req.replaceable && !site->capability.replaceable) {
            std::ostringstream os;
            os << "cache method requires site " << role_name(req.role)
               << " to be replaceable, but it is not usable this run (SP-parallel, or "
                  "the model can't cut its block stack).";
            return {false, CacheErrorCode::SiteNotReplaceable, os.str()};
        }
    }
    if (requirements.requires_probe) {
        auto probe = contract.find_site(CacheSiteRole::BLOCK_STACK_PROBE);
        if (!probe.has_value() || !probe->capability.readable) {
            std::ostringstream os;
            os << "cache method requires a probeable block stack, but the model does not "
                  "expose a usable BLOCK_STACK_PROBE site this run.";
            return {false, CacheErrorCode::SiteNotProbeable, os.str()};
        }
    }
    return {true, CacheErrorCode::Ok, {}};
}

}  // namespace cache
}  // namespace edgedit
