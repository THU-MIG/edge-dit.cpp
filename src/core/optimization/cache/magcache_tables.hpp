#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "runtime/model_loader.h"

namespace edgedit {
namespace cache {

// Precalibrated per-step magnitude-ratio tables for MagCache. Each entry is the
// ratio ||residual_t|| / ||residual_{t-1}|| measured offline on the reference
// implementation. Values verified against github.com/Zehong-Ma/MagCache
// (MagCache4FLUX/magcache_flux.py) — see memory reference_cache_algorithms.md.
//
// Tables are stored at their native calibration length and resampled to the
// actual step count via nearest-neighbour (matching the reference). Unknown
// models fall back to a neutral all-ones table, so the method leans entirely on
// the accumulated-error threshold.

// FLUX.1-dev, 28-step calibration.
inline const std::vector<float>& magcache_flux_table() {
    static const std::vector<float> table = {
        1.0f, 1.21094f, 1.11719f, 1.07812f, 1.0625f, 1.03906f, 1.03125f,
        1.03906f, 1.02344f, 1.03125f, 1.02344f, 0.98047f, 1.01562f, 1.00781f,
        1.0f, 1.00781f, 1.0f, 1.00781f, 1.0f, 1.0f, 0.99609f,
        0.99609f, 0.98047f, 0.98828f, 0.96484f, 0.95703f, 0.93359f, 0.89062f};
    return table;
}

// Qwen-Image, 50-step calibration (single-branch; cond/uncond share).
inline const std::vector<float>& magcache_qwen_table() {
    static const std::vector<float> table = {
        1.0f, 1.06934f, 1.04943f, 1.03855f, 1.02832f, 1.02637f, 1.02051f,
        1.01953f, 1.01758f, 1.01465f, 1.01367f, 1.01172f, 1.01074f, 1.00977f,
        1.00879f, 1.00781f, 1.00684f, 1.00684f, 1.00586f, 1.00488f, 1.00488f,
        1.00391f, 1.00391f, 1.00293f, 1.00293f, 1.00195f, 1.00195f, 1.00195f,
        1.00098f, 1.00098f, 1.00098f, 1.0f, 1.0f, 1.0f, 0.99951f,
        0.99902f, 0.99902f, 0.99854f, 0.99805f, 0.99756f, 0.99658f, 0.99609f,
        0.99512f, 0.99414f, 0.99268f, 0.99072f, 0.98828f, 0.98486f, 0.97949f,
        0.96924f};
    return table;
}

// Resample a native calibration table to `num_steps` via nearest-neighbour.
inline std::vector<float> magcache_resample(const std::vector<float>& table, int num_steps) {
    std::vector<float> out;
    if (num_steps <= 0) {
        return out;
    }
    out.resize(static_cast<size_t>(num_steps), 1.0f);
    if (table.empty()) {
        return out;
    }
    if (static_cast<int>(table.size()) == num_steps) {
        return table;
    }
    const int src = static_cast<int>(table.size());
    for (int i = 0; i < num_steps; ++i) {
        int idx = num_steps > 1
                      ? static_cast<int>(std::lround(static_cast<double>(i) * (src - 1) / (num_steps - 1)))
                      : 0;
        idx = std::max(0, std::min(src - 1, idx));
        out[static_cast<size_t>(i)] = table[static_cast<size_t>(idx)];
    }
    return out;
}

// Pick the calibrated table for a model version, resampled to num_steps.
// Falls back to neutral 1.0 for models without a table.
inline std::vector<float> magcache_table_for(SDVersion version, int num_steps) {
    if (ed_version_is_flux(version) || ed_version_is_flux2(version)) {
        return magcache_resample(magcache_flux_table(), num_steps);
    }
    if (ed_version_is_qwen_image(version)) {
        return magcache_resample(magcache_qwen_table(), num_steps);
    }
    return std::vector<float>(static_cast<size_t>(std::max(0, num_steps)), 1.0f);
}

}  // namespace cache
}  // namespace edgedit
