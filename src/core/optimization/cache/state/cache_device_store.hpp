#pragma once

#include <cstdint>
#include <vector>

namespace edgedit {
namespace cache {

// Abstract device backing for cache slots. The CacheStateManager stays free of
// any ggml dependency (redesign §27.4 keeps the cache core off the backend API);
// the runner — which already lives in ggml-land — implements this so a slot's
// residual can live in persistent device memory and be reused on-device without a
// host round-trip. A null store on the manager means host-only slots (the CPU
// backend, mmdit/wan, and any run where the seam can't cut its stack).
//
// The opaque `void*` handles are ggml_tensor* under the hood; only the runner's
// implementation and the seam that consumes them ever cast them back.
class ICacheDeviceStore {
public:
    virtual ~ICacheDeviceStore() = default;

    // Allocate (or reuse, if shape matches) a persistent device tensor for one
    // ring entry, keyed by the manager's folded (condition_key<<16 ^ slot) key
    // plus the ring index. Persists across steps in a context OUTSIDE the per-step
    // compute buffer, so it survives the substep pass's reset_graph_cut_run_cache().
    // Returns an opaque device-tensor handle owned by the store, or nullptr on
    // failure. `shape` is the block-stack residual extents [ne0, ne1, ne2...].
    virtual void* ensure_entry(uint64_t ring_key, int ring_index,
                               const std::vector<int64_t>& shape) = 0;

    // Free every device buffer this store allocated. Called from
    // CacheStateManager::reset() (per generation), mirroring the old
    // reset_dicache_gpu_states() lifetime so device state never leaks across
    // generations (see the historical ~1GB/img seam leak).
    virtual void release_all() = 0;
};

}  // namespace cache
}  // namespace edgedit
