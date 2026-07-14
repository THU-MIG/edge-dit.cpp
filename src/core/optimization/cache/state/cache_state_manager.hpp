#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/optimization/cache/ir/cache_slot.hpp"
#include "core/optimization/cache/state/cache_device_store.hpp"
#include "utils/tensor.hpp"

namespace edgedit {
namespace cache {

// Handle to a slot's current backing store. A host-backed slot fills `host`; a
// device-backed slot (residual kept on-device for on-GPU reuse) fills `buffer`
// with an opaque ggml_tensor* the seam uses as a graph operand, and leaves `host`
// null. Callers check whichever they need; the state manager decides per slot.
struct CacheSlotHandle {
    sd::Tensor<float>* host = nullptr;  // host store; null once empty or on a device slot
    void* buffer = nullptr;             // device store: opaque ggml_tensor*; null on a host slot
    size_t offset = 0;
    uint64_t generation = 0;
    bool valid = false;
};

// Owns cache state across steps, isolated per (condition_key, slot). Slots with
// history_depth > 1 are ring buffers. begin_step / rotate_history are load-bearing
// (the ring advances only after a step's write). commit_step is a reserved
// transaction hook that is currently a NO-OP: no method stages writes, so there is
// nothing to publish at step end (see its definition). It exists for a future
// staged-write method, not as active safety today.
//
// The interface is storage-agnostic on purpose. Today's implementation holds
// host vectors plus optional device-backed slots (via ICacheDeviceStore); the
// backend choice is an implementation detail with no signature change.
class CacheStateManager {
public:
    // Frees device buffers via the store on destruction. CacheEngine (and thus
    // its state_) is a per-generation stack object; the runner-owned store
    // outlives it, so releasing here is the per-generation device-state free that
    // prevents cross-generation leaks (see the historical ~1GB/img seam leak).
    ~CacheStateManager() { reset(); }

    // condition_key isolates CFG cond/uncond branches. A key of nullptr is the
    // single-branch (Main) case.
    void initialize(const std::vector<CacheSlotDesc>& slots);

    // Wire an optional device store (owned by the caller — the runner). When set,
    // slots whose desc.device_backed is true allocate their ring entries on-device
    // via the store; the returned handles carry a `buffer` (opaque ggml_tensor*)
    // instead of `host`. Null store => every slot is host-backed (today's path).
    void set_device_store(ICacheDeviceStore* store) { device_store_ = store; }
    bool has_device_store() const { return device_store_ != nullptr; }

    bool has(const void* condition_key, CacheSlotId slot) const;

    // Current (newest) ring entry for read/write. read() returns an invalid
    // handle when the slot was never written for this branch.
    CacheSlotHandle read(const void* condition_key, CacheSlotId slot) const;
    // Acquire the newest ring slot for writing. For a device-backed slot, pass the
    // residual `shape` so the store can allocate the device entry on first use;
    // the returned handle then carries `buffer` (ggml_tensor*) instead of `host`.
    // A host slot ignores `shape` and returns a `host` pointer as before.
    CacheSlotHandle write(const void* condition_key, CacheSlotId slot,
                          const std::vector<int64_t>& shape = {});

    // Device-slot allocation, called by the runner seam once the residual shape is
    // known (the packed block-stack seq shape, unknown to the lowering before
    // capture). Allocates/reuses the newest ring entry's device tensor at `shape`
    // via the store, records it so read()/read_history() return it, marks the ring
    // filled, and returns the opaque ggml_tensor* (nullptr if no device store or on
    // failure). This is the device analogue of write() for the post-capture path.
    void* alloc_device_entry(const void* condition_key, CacheSlotId slot,
                             const std::vector<int64_t>& shape);

    // Read a specific history depth back (0 == newest). Invalid handle if the
    // ring hasn't filled to that depth yet.
    CacheSlotHandle read_history(const void* condition_key, CacheSlotId slot, int depth) const;

    // Advance the ring buffer for a slot on one branch (newest slot becomes the
    // next write target; older entries shift back). Call only after commit.
    void rotate_history(const void* condition_key, CacheSlotId slot);

    void begin_step(int step_index);
    void commit_step(int step_index);
    void reset();

    // Rough persistent-state footprint for the init log / memory planner.
    size_t state_bytes() const;

private:
    struct SlotRing {
        CacheSlotDesc desc;
        std::vector<sd::Tensor<float>> ring;  // size == history_depth (host slot)
        // Device slot: opaque ggml_tensor* per ring entry, allocated lazily via the
        // device store on first write once the residual shape is known. Same size
        // as `ring`; head/filled indexing is shared with the host path.
        std::vector<void*> device_ring;
        std::vector<int64_t> device_shape;  // extents the device entries were built for
        int head = 0;      // index of the newest entry
        int filled = 0;    // how many entries written so far
        // Resolved at ensure_ring() time: device_backed desc AND a store was wired.
        // Kept on the ring (not read from desc) so a device_backed slot with no
        // store cleanly falls back to the host ring instead of indexing an empty
        // device_ring.
        bool device = false;

        bool is_device() const { return device; }
    };

    // Per (condition_key, slot_id) ring. condition_key is stored as uintptr_t.
    std::unordered_map<uint64_t, SlotRing> rings_;
    std::vector<CacheSlotDesc> slot_descs_;
    int current_step_ = -1;
    ICacheDeviceStore* device_store_ = nullptr;  // not owned; null => host-only

    static uint64_t key_of(const void* condition_key, CacheSlotId slot);
    const SlotRing* find_ring(const void* condition_key, CacheSlotId slot) const;
    SlotRing& ensure_ring(const void* condition_key, CacheSlotId slot);
    // True when this slot should use the device store (opt-in desc flag + a store
    // is wired). Device backing is silently skipped when no store is present.
    bool slot_is_device(const CacheSlotDesc& desc) const {
        return desc.device_backed && device_store_ != nullptr;
    }
};

}  // namespace cache
}  // namespace edgedit
