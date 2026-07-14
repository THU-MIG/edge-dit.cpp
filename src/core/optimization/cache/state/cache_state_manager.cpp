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
    ring.device = slot_is_device(ring.desc);
    if (ring.device) {
        ring.device_ring.assign(static_cast<size_t>(depth), nullptr);
    }
    return rings_.emplace(key, std::move(ring)).first->second;
}

bool CacheStateManager::has(const void* condition_key, CacheSlotId slot) const {
    const SlotRing* ring = find_ring(condition_key, slot);
    if (ring == nullptr || ring->filled <= 0) {
        return false;
    }
    if (ring->is_device()) {
        return ring->device_ring[static_cast<size_t>(ring->head)] != nullptr;
    }
    return !ring->ring[ring->head].empty();
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
    if (ring->is_device()) {
        handle.buffer = ring->device_ring[static_cast<size_t>(idx)];
        handle.valid = handle.buffer != nullptr;
        return handle;
    }
    handle.host = const_cast<sd::Tensor<float>*>(&ring->ring[static_cast<size_t>(idx)]);
    handle.valid = handle.host != nullptr && !handle.host->empty();
    return handle;
}

CacheSlotHandle CacheStateManager::write(const void* condition_key, CacheSlotId slot,
                                         const std::vector<int64_t>& shape) {
    SlotRing& ring = ensure_ring(condition_key, slot);
    CacheSlotHandle handle;
    handle.generation = static_cast<uint64_t>(current_step_ < 0 ? 0 : current_step_);
    if (ring.is_device()) {
        // Device slots allocate lazily via alloc_device_entry() from the runner
        // seam (the residual shape is only known post-capture). write() just hands
        // back the current head's device tensor if one exists; `shape` is ignored
        // here (allocation is alloc_device_entry's job).
        (void)shape;
        handle.buffer = ring.device_ring[static_cast<size_t>(ring.head)];
        handle.valid = handle.buffer != nullptr;
    } else {
        handle.host = &ring.ring[static_cast<size_t>(ring.head)];
        handle.valid = true;
    }
    if (ring.filled == 0 && !ring.is_device()) {
        ring.filled = 1;  // host: the newest entry now exists once written
    }
    return handle;
}

void* CacheStateManager::alloc_device_entry(const void* condition_key, CacheSlotId slot,
                                            const std::vector<int64_t>& shape) {
    if (device_store_ == nullptr || shape.empty()) {
        return nullptr;
    }
    SlotRing& ring = ensure_ring(condition_key, slot);
    if (!ring.is_device()) {
        return nullptr;
    }
    const uint64_t base = key_of(condition_key, slot);
    // (Re)allocate the whole ring on a shape change so history entries stay
    // uniform; a depth-1 slot (MagCache) is a single entry.
    if (ring.device_shape != shape) {
        for (int i = 0; i < static_cast<int>(ring.device_ring.size()); ++i) {
            ring.device_ring[static_cast<size_t>(i)] =
                device_store_->ensure_entry(base, i, shape);
        }
        ring.device_shape = shape;
    }
    void* t = ring.device_ring[static_cast<size_t>(ring.head)];
    if (t != nullptr && ring.filled == 0) {
        ring.filled = 1;  // the newest entry now exists once allocated+written
    }
    return t;
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
    // Intentional no-op. Slot writes (host vector or device tensor) are applied
    // in place by write()/alloc_device_entry(), so there is nothing to flush at
    // step end. Kept as an explicit hook: a future method that stages writes
    // (e.g. double-buffered token-cache) would publish them here.
}

void CacheStateManager::rollback_step(int step_index) {
    (void)step_index;
    // Intentional no-op with NO current caller. Every cache method today either
    // commits a step or falls back to a synchronous full compute within the same
    // step; none aborts mid-step, so there is never a partial write to undo (an
    // aborted step just leaves stale-but-valid history). Kept as a reserved hook
    // for a future staged-write method; wire a real restore here only once such a
    // method exists.
}

void CacheStateManager::reset() {
    // Release device buffers first (the store owns them); mirrors the old
    // reset_dicache_gpu_states() per-generation free so device state never leaks
    // across generations.
    if (device_store_ != nullptr) {
        device_store_->release_all();
    }
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
