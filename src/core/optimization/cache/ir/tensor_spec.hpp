#pragma once

#include <cstdint>
#include <vector>

namespace edgedit {
namespace cache {

// Host-side dtype for cache state. Cache slots hold host tensors today
// (std::vector<float>); the enum leaves room for the deferred backend-buffer
// implementation to place compressed state without an interface change.
enum class CacheDataType {
    F32,
    F16,
};

// Shape + dtype descriptor for a cache value. The shape is often unknown at
// compile() time (it depends on resolution / sequence length), so an empty
// shape means "dynamic — resolved on first write". `rank` records the expected
// dimensionality even when the concrete extents are dynamic.
struct TensorSpec {
    std::vector<int64_t> shape;  // empty => dynamic
    int rank = 0;
    CacheDataType dtype = CacheDataType::F32;

    bool is_dynamic() const { return shape.empty(); }

    int64_t numel() const {
        if (shape.empty()) {
            return 0;
        }
        int64_t n = 1;
        for (int64_t d : shape) {
            n *= d;
        }
        return n;
    }
};

}  // namespace cache
}  // namespace edgedit
