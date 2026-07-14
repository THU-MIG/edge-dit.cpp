#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/optimization/cache/model/anchor.hpp"
#include "core/optimization/cache/ir/indicator.hpp"

struct ggml_tensor;

namespace edgedit {
namespace cache {

// Dynamic anchor->tensor tap table. Replaces CacheGraphScope's fixed *_node
// fields + the kCache*Name string constants: the set of tensors the cache system
// reads is driven per-substep by the plan, not enumerated as struct fields.
//
// Flow: the middle layer calls set_requested() with the anchors this substep
// needs (from SubstepPlan.taps plus the anchors its indicators reference); the
// model's forward() conditionally calls the free tap() at each structural
// landmark; the executor reads back by anchor after compute. The registry stores
// only the ggml_tensor* — the runner's existing named-tensor pipeline
// (ggml_set_output + cache(name,t) -> cache_tensor_index_) still does the actual
// buffer pinning and readback, keyed by AnchorRef::tap_name().
class TapRegistry {
public:
    void set_requested(const std::vector<AnchorRef>& anchors) {
        requested_.clear();
        taps_.clear();
        recorded_.clear();
        indicator_nodes_.clear();
        for (const auto& a : anchors) {
            requested_.insert(a.tap_name());
        }
    }

    bool wants(const AnchorRef& a) const { return requested_.count(a.tap_name()) != 0; }

    void put(const AnchorRef& a, ggml_tensor* t) {
        taps_[a.tap_name()] = t;
        recorded_[node_name(a)] = t;
    }

    ggml_tensor* get(const AnchorRef& a) const {
        auto it = taps_.find(a.tap_name());
        return it != taps_.end() ? it->second : nullptr;
    }

    // The tap name the runner binds this tensor under (for named readback).
    static std::string node_name(const AnchorRef& a) { return "ed_tap:" + a.tap_name(); }

    // Recorded taps keyed by node_name -> tensor; the runner promotes these into
    // its named index and pins them as graph outputs.
    const std::unordered_map<std::string, ggml_tensor*>& recorded() const { return recorded_; }

    // Indicator scalar nodes the executor wove this build (each already named
    // "cache_ind:<name>"); the runner expands + pins them for scalar readback.
    void add_indicator_node(ggml_tensor* node) { indicator_nodes_.push_back(node); }
    const std::vector<ggml_tensor*>& indicator_nodes() const { return indicator_nodes_; }

    void clear() {
        requested_.clear();
        taps_.clear();
        recorded_.clear();
        indicator_nodes_.clear();
        indicators_.clear();
        stop_after_block_ = -1;
        capture_residual_ = false;
        inject_tensor_ = nullptr;
    }
    bool empty() const { return requested_.empty(); }

    // Substep control: probe substeps run only a prefix of the block stack. The
    // executor sets the 0-based block index to stop after (return the block-stack
    // state); -1 = run the whole stack. Drives the model forward's early return.
    void set_stop_after(int block_index) { stop_after_block_ = block_index; }
    bool stop_after(int i) const { return stop_after_block_ >= 0 && i == stop_after_block_; }

    // Indicators to weave this substep (in-graph reductions over tapped anchors).
    // The model's build_graph lowers them after forward() populates the taps.
    void set_indicators(const std::vector<Indicator>& inds) { indicators_ = inds; }
    const std::vector<Indicator>& indicators() const { return indicators_; }

    // Residual capture: when set, build_graph weaves (ModelOut - ModelIn) as a named
    // node "ed_cache_feature" so the pass can d2d-copy it into a device slot (the
    // block-stack residual for Feature/Probe reuse). Requires ModelIn + ModelOut
    // taps to be requested.
    void set_capture_residual(bool on) { capture_residual_ = on; }
    bool capture_residual() const { return capture_residual_; }

    // Injected residual to add back at ModelIn on a reuse substep: build_graph forms
    // (ModelIn + inject) when set, so a zero-block reuse still produces the full
    // output on-device. Opaque ggml_tensor* (the device slot buffer).
    void set_inject_tensor(ggml_tensor* t) { inject_tensor_ = t; }
    ggml_tensor* inject_tensor() const { return inject_tensor_; }

private:
    std::unordered_set<std::string> requested_;
    std::unordered_map<std::string, ggml_tensor*> taps_;
    std::unordered_map<std::string, ggml_tensor*> recorded_;
    std::vector<ggml_tensor*> indicator_nodes_;
    std::vector<Indicator> indicators_;
    int stop_after_block_ = -1;
    bool capture_residual_ = false;
    ggml_tensor* inject_tensor_ = nullptr;
};

}  // namespace cache
}  // namespace edgedit
