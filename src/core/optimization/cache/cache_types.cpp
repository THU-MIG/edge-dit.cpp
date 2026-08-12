#include "core/optimization/cache/cache_types.hpp"

namespace edgedit {
namespace cache {

const char* cache_mode_name(CacheMode mode) {
    switch (mode) {
        case CacheMode::EasyCache: return "EasyCache";
        case CacheMode::UCache: return "UCache";
        case CacheMode::DBCache: return "DBCache";
        case CacheMode::TaylorSeer: return "TaylorSeer";
        case CacheMode::CacheDiT: return "CacheDiT";
        case CacheMode::MagCache: return "MagCache";
        case CacheMode::DiCache: return "DiCache";
        case CacheMode::SenCache: return "SenCache";
        case CacheMode::Disabled:
        default: return "disabled";
    }
}

// Whether a cache method consumes a precalibrated data table and can therefore
// be profiled via --cache-calibrate. Keep in sync with
// CachePolicy::supports_calibration(); this mode-level view lets callers (CLI
// validation) check without instantiating a policy.
bool cache_mode_supports_calibration(CacheMode mode) {
    switch (mode) {
        case CacheMode::MagCache: return true;
        case CacheMode::SenCache: return true;
        default: return false;
    }
}

CacheModelSpec cache_model_spec_for_version(SDVersion version) {
    CacheModelSpec spec;
    spec.version = version;

    if (version == VERSION_FLUX2_KLEIN) {
        spec.model_name = "flux2-klein";
        spec.block_count = 5 + 20;  // double_blocks + single_blocks
        return spec;
    }
    if (ed_version_is_flux(version) || ed_version_is_flux2(version)) {
        spec.model_name = "flux";
        spec.block_count = 19 + 38;  // double_blocks + single_blocks
        return spec;
    }
    if (ed_version_is_qwen_image(version) || ed_version_is_qwen_image_edit(version)) {
        spec.model_name = "qwen_image";
        spec.block_count = 60;
        return spec;
    }
    if (ed_version_is_sd3(version)) {
        spec.model_name = "sd3";
        spec.separate_cfg = true;
        spec.block_count = 24;
        return spec;
    }
    if (ed_version_is_wan(version)) {
        spec.model_name = "wan";
        spec.separate_cfg = true;
        spec.block_count = 40;
        return spec;
    }

    spec.model_name = "unknown";
    return spec;
}

}  // namespace cache
}  // namespace edgedit
