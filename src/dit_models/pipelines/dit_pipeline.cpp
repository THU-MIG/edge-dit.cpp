#include "dit_models/pipelines/dit_pipeline.hpp"

#include "dit_models/pipelines/flux_pipeline.hpp"
#include "dit_models/pipelines/qwen_image_edit_pipeline.hpp"
#include "dit_models/pipelines/qwen_image_pipeline.hpp"
#include "dit_models/pipelines/sd3_pipeline.hpp"
#include "dit_models/pipelines/wan_pipeline.hpp"
#include "utils/util.h"

namespace edgedit {

std::unique_ptr<DiTPipeline> create_dit_pipeline(SDVersion version,
                                                 std::string* error) {
    if (ed_version_is_flux(version) || ed_version_is_flux2(version)) {
        return std::make_unique<FluxPipeline>(version);
    }
    if (ed_version_is_sd3(version)) {
        return std::make_unique<SD3Pipeline>(version);
    }
    if (ed_version_is_qwen_image(version)) {
        return std::make_unique<QwenImagePipeline>(version);
    }
    if (ed_version_is_qwen_image_edit(version)) {
        return std::make_unique<QwenImageEditPipeline>(version);
    }
    if (ed_version_is_wan(version)) {
        return std::make_unique<WanPipeline>(version);
    }

    const std::string msg = "unsupported DiT pipeline version: " +
                            std::string(ed_version_name(version));
    if (error != nullptr) {
        *error = msg;
    }
    LOG_ERROR("%s", msg.c_str());
    return nullptr;
}

}  // namespace edgedit
