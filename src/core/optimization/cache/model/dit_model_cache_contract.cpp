#include "core/optimization/cache/model/dit_model_cache_contract.hpp"

namespace edgedit {
namespace cache {

namespace {

// Derive family + block count from the version. Block counts mirror the region
// data in cache_model_spec_for_version() (cache_types.cpp): the whole-stack
// residual spans every block, so flux sums its double + single depths.
struct FamilyInfo {
    ModelFamily family = ModelFamily::Unknown;
    int num_blocks = 0;
    bool dual_stream = false;
    bool spatial_attention = false;
    bool temporal_attention = false;
};

FamilyInfo family_info_for(SDVersion version) {
    FamilyInfo info;
    if (ed_version_is_flux(version) || ed_version_is_flux2(version)) {
        info.family = ModelFamily::Flux;
        info.num_blocks = 19 + 38;  // double_blocks + single_blocks
        info.dual_stream = true;
        return info;
    }
    if (ed_version_is_qwen_image(version) || ed_version_is_qwen_image_edit(version)) {
        info.family = ModelFamily::QwenImage;
        info.num_blocks = 60;
        info.dual_stream = true;
        return info;
    }
    if (ed_version_is_sd3(version)) {
        info.family = ModelFamily::MMDiT;
        info.num_blocks = 24;
        info.dual_stream = true;
        return info;
    }
    if (ed_version_is_wan(version)) {
        info.family = ModelFamily::WanVideo;
        info.num_blocks = 40;
        info.spatial_attention = true;
        info.temporal_attention = true;
        return info;
    }
    return info;
}

}  // namespace

DiTModelCacheContract::DiTModelCacheContract(SDVersion version, bool seam_available) {
    const FamilyInfo info = family_info_for(version);

    schema_.version = version;
    schema_.family = info.family;
    schema_.num_blocks = info.num_blocks;
    schema_.has_dual_stream = info.dual_stream;
    schema_.has_spatial_attention = info.spatial_attention;
    schema_.has_temporal_attention = info.temporal_attention;
    schema_.compute_dtype = CacheDataType::F32;

    // Coarse Option-A topology: input projection, one block-stack segment,
    // output projection. The block-stack segment carries the seam capabilities.
    SegmentDesc input_seg;
    input_seg.id = 0;
    input_seg.kind = SegmentKind::INPUT_PROJECTION;

    SegmentDesc block_seg;
    block_seg.id = 1;
    block_seg.kind = SegmentKind::BLOCK_STACK;
    block_seg.block_start = 0;
    block_seg.block_end = info.num_blocks;
    block_seg.capabilities.can_capture_output = seam_available;
    block_seg.capabilities.can_inject_output = seam_available;
    block_seg.capabilities.can_probe = seam_available;

    SegmentDesc output_seg;
    output_seg.id = 2;
    output_seg.kind = SegmentKind::OUTPUT_PROJECTION;

    topology_.segments = {input_seg, block_seg, output_seg};

    // DENOISER_OUTPUT is always available (Output methods are black-box and work
    // even under SP). BLOCK_STACK_OUTPUT / _PROBE are replaceable/probeable only
    // when the seam is usable this run.
    CacheSiteDesc denoiser;
    denoiser.id = 0;
    denoiser.role = CacheSiteRole::DENOISER_OUTPUT;
    denoiser.segment_id = output_seg.id;
    denoiser.tensor_spec.dtype = CacheDataType::F32;
    denoiser.capability.readable = true;
    denoiser.capability.replaceable = true;   // reuse == serve cached output
    denoiser.capability.storable = true;
    denoiser.capability.supports_history = true;
    sites_.push_back(denoiser);

    CacheSiteDesc block_out;
    block_out.id = 1;
    block_out.role = CacheSiteRole::BLOCK_STACK_OUTPUT;
    block_out.segment_id = block_seg.id;
    block_out.tensor_spec.dtype = CacheDataType::F32;
    block_out.capability.readable = seam_available;
    block_out.capability.replaceable = seam_available;
    block_out.capability.storable = seam_available;
    block_out.capability.supports_history = true;
    sites_.push_back(block_out);

    CacheSiteDesc block_probe;
    block_probe.id = 2;
    block_probe.role = CacheSiteRole::BLOCK_STACK_PROBE;
    block_probe.segment_id = block_seg.id;
    block_probe.tensor_spec.dtype = CacheDataType::F32;
    block_probe.capability.readable = seam_available;
    block_probe.capability.replaceable = false;  // probe state is read-only
    block_probe.capability.storable = seam_available;
    block_probe.capability.supports_history = true;
    sites_.push_back(block_probe);
}

}  // namespace cache
}  // namespace edgedit
