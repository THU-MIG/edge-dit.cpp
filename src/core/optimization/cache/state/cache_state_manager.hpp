#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/optimization/cache/ir/cache_slot.hpp"
#include "utils/tensor.hpp"

namespace edgedit {
namespace cache {

// Handle to a slot's current backing store. Option A backs each slot with a
// host sd::Tensor<float>; the deferred backend implementation fills buffer/
// offset instead and returns the same handle type, so callers don't change.
struct CacheSlotHandle {
    sd::Tensor<float>* host = nullptr;  // Option A store; null once empty
    // Reserved for the deferred backend-buffer implementation:
    void* buffer = nullptr;
    size_t offset = 0;
    uint64_t generation = 0;
    bool valid = false;
};

// Owns cache state across steps, isolated per (condition_key, slot). Slots with
// history_depth > 1 are ring buffers. Transactional: begin_step / commit_step /
// rollback_step give the doc's per-step state consistency; rotate_history
// advances a ring only after a committed step.
//
// The interface is storage-agnostic on purpose. Today's implementation holds
// host vectors; swapping in backend buffers (using the runner's view-patching
// primitive) is an implementation change with no signature change.
class CacheStateManager {
public:
    // condition_key isolates CFG cond/uncond branches. A key of nullptr is the
    // single-branch (Main) case.
    void initialize(const std::vector<CacheSlotDesc>& slots);

    bool has(const void* condition_key, CacheSlotId slot) const;

    // Current (newest) ring entry for read/write. read() returns an invalid
    // handle when the slot was never written for this branch.
    CacheSlotHandle read(const void* condition_key, CacheSlotId slot) const;
    CacheSlotHandle write(const void* condition_key, CacheSlotId slot);

    // Read a specific history depth back (0 == newest). Invalid handle if the
    // ring hasn't filled to that depth yet.
    CacheSlotHandle read_history(const void* condition_key, CacheSlotId slot, int depth) const;

    // Advance the ring buffer for a slot on one branch (newest slot becomes the
    // next write target; older entries shift back). Call only after commit.
    void rotate_history(const void* condition_key, CacheSlotId slot);

    void begin_step(int step_index);
    void commit_step(int step_index);
    void rollback_step(int step_index);
    void reset();

    // Rough persistent-state footprint for the init log / memory planner.
    size_t state_bytes() const;

private:
    struct SlotRing {
        CacheSlotDesc desc;
        std::vector<sd::Tensor<float>> ring;  // size == history_depth
        int head = 0;      // index of the newest entry
        int filled = 0;    // how many entries written so far
    };

    // Per (condition_key, slot_id) ring. condition_key is stored as uintptr_t.
    std::unordered_map<uint64_t, SlotRing> rings_;
    std::vector<CacheSlotDesc> slot_descs_;
    int current_step_ = -1;

    static uint64_t key_of(const void* condition_key, CacheSlotId slot);
    const SlotRing* find_ring(const void* condition_key, CacheSlotId slot) const;
    SlotRing& ensure_ring(const void* condition_key, CacheSlotId slot);
};

}  // namespace cache
}  // namespace edgedit
