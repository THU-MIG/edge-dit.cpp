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
        case CacheMode::Disabled:
        default: return "disabled";
    }
}

CacheModelSpec cache_model_spec_for_version(SDVersion version) {
    CacheModelSpec spec;
    spec.version = version;

    if (ed_version_is_flux(version) || ed_version_is_flux2(version)) {
        spec.model_name = "flux";
        spec.regions.push_back({"double", "flux.double_blocks", 19,
                                CacheRegionPattern::ImageText,
                                {"img", "txt"},
                                {"img", "txt"},
                                2,
                                0});
        spec.regions.push_back({"single", "flux.single_blocks", 38,
                                CacheRegionPattern::PackedImageText,
                                {"txt_img"},
                                {"txt_img"},
                                2,
                                0});
        return spec;
    }

    if (ed_version_is_qwen_image(version)) {
        spec.model_name = "qwen_image";
        spec.regions.push_back({"transformer", "qwen_image.transformer_blocks", 60,
                                CacheRegionPattern::ImageText,
                                {"img", "txt"},
                                {"img", "txt"},
                                2,
                                0});
        return spec;
    }

    if (ed_version_is_sd3(version)) {
        spec.model_name = "sd3";
        spec.separate_cfg = true;
        spec.regions.push_back({"joint", "mmdit.joint_blocks", 24,
                                CacheRegionPattern::HiddenContext,
                                {"x", "context"},
                                {"x", "context"},
                                2,
                                0});
        return spec;
    }

    if (ed_version_is_wan(version)) {
        spec.model_name = "wan";
        spec.separate_cfg = true;
        spec.regions.push_back({"blocks", "wan.blocks", 40,
                                CacheRegionPattern::HiddenContext,
                                {"x", "context"},
                                {"x", "context"},
                                2,
                                0});
        return spec;
    }

    spec.model_name = "unknown";
    return spec;
}

}  // namespace cache
}  // namespace edgedit
