#include "core/optimization/cache/state/cache_state_manager.hpp"

#include <cstdint>

namespace edgedit {
namespace cache {

uint64_t CacheStateManager::key_of(const void* condition_key, CacheSlotId slot) {
    // Fold the condition pointer and slot id into one map key. Slot ids are
    // small, so the low 16 bits are reserved for them.
    const uint64_t cond = reinterpret_cast<uintptr_t>(condition_key);
    return (cond << 16) ^ static_cast<uint64_t>(slot & 0xffff);
}

void CacheStateManager::initialize(const std::vector<CacheSlotDesc>& slots) {
    slot_descs_ = slots;
    rings_.clear();
    current_step_ = -1;
}

const CacheStateManager::SlotRing* CacheStateManager::find_ring(const void* condition_key,
                                                                CacheSlotId slot) const {
    auto it = rings_.find(key_of(condition_key, slot));
    return it == rings_.end() ? nullptr : &it->second;
}

CacheStateManager::SlotRing& CacheStateManager::ensure_ring(const void* condition_key,
                                                            CacheSlotId slot) {
    const uint64_t key = key_of(condition_key, slot);
    auto it = rings_.find(key);
    if (it != rings_.end()) {
        return it->second;
    }

    SlotRing ring;
    for (const auto& d : slot_descs_) {
        if (d.id == slot) {
            ring.desc = d;
            break;
        }
    }
    const int depth = ring.desc.history_depth > 0 ? ring.desc.history_depth : 1;
    ring.ring.resize(static_cast<size_t>(depth));
    return rings_.emplace(key, std::move(ring)).first->second;
}

bool CacheStateManager::has(const void* condition_key, CacheSlotId slot) const {
    const SlotRing* ring = find_ring(condition_key, slot);
    return ring != nullptr && ring->filled > 0 && !ring->ring[ring->head].empty();
}

CacheSlotHandle CacheStateManager::read(const void* condition_key, CacheSlotId slot) const {
    return read_history(condition_key, slot, 0);
}

CacheSlotHandle CacheStateManager::read_history(const void* condition_key,
                                                CacheSlotId slot,
                                                int depth) const {
    CacheSlotHandle handle;
    const SlotRing* ring = find_ring(condition_key, slot);
    if (ring == nullptr || depth < 0 || depth >= ring->filled) {
        return handle;
    }
    const int size = static_cast<int>(ring->ring.size());
    const int idx = ((ring->head - depth) % size + size) % size;
    handle.host = const_cast<sd::Tensor<float>*>(&ring->ring[static_cast<size_t>(idx)]);
    handle.valid = handle.host != nullptr && !handle.host->empty();
    return handle;
}

CacheSlotHandle CacheStateManager::write(const void* condition_key, CacheSlotId slot) {
    SlotRing& ring = ensure_ring(condition_key, slot);
    CacheSlotHandle handle;
    handle.host = &ring.ring[static_cast<size_t>(ring.head)];
    handle.valid = true;
    handle.generation = static_cast<uint64_t>(current_step_ < 0 ? 0 : current_step_);
    if (ring.filled == 0) {
        ring.filled = 1;  // the newest entry now exists once written
    }
    return handle;
}

void CacheStateManager::rotate_history(const void* condition_key, CacheSlotId slot) {
    SlotRing* it = nullptr;
    auto found = rings_.find(key_of(condition_key, slot));
    if (found != rings_.end()) {
        it = &found->second;
    }
    if (it == nullptr || it->ring.size() <= 1) {
        return;
    }
    const int size = static_cast<int>(it->ring.size());
    it->head = (it->head + 1) % size;
    if (it->filled < size) {
        it->filled += 1;
    }
    // The new head becomes the next write target; leave its old contents to be
    // overwritten by the next write().
}

void CacheStateManager::begin_step(int step_index) {
    current_step_ = step_index;
}

void CacheStateManager::commit_step(int step_index) {
    (void)step_index;
    // Host-vector storage commits in place; nothing to flush. The transactional
    // hooks exist so the deferred backend-buffer implementation can double-buffer.
}

void CacheStateManager::rollback_step(int step_index) {
    (void)step_index;
    // Option A writes are only consumed after a successful step, so an aborted
    // step leaves stale-but-valid history; nothing to undo. The deferred backend
    // implementation restores the pre-step buffer here.
}

void CacheStateManager::reset() {
    rings_.clear();
    current_step_ = -1;
}

size_t CacheStateManager::state_bytes() const {
    size_t bytes = 0;
    for (const auto& kv : rings_) {
        for (const auto& t : kv.second.ring) {
            bytes += static_cast<size_t>(t.numel()) * sizeof(float);
        }
    }
    return bytes;
}

}  // namespace cache
}  // namespace edgedit
