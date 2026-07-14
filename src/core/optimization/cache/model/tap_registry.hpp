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
        inject_active_ = false;
        inject_kind_ = InjectKind::None;
        inject_input_ = nullptr;
        inject_resid1_ = nullptr;
        inject_resid2_ = nullptr;
        inject_gamma_ = nullptr;
        inject_start_ = 0;
        inject_resume_ = -1;
        probe_metrics_ = false;
        probe_ops_ = ProbeMetricOperands{};
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

    // ---- Tap-driven inject (reuse). Replaces cache_scope->step_inject_region in the
    // model forward: when active, at block `inject_start` the forward replaces the
    // stream with (x_before + <residual>) and jumps to `inject_resume` (skipping the
    // region's blocks). The residual is reconstructed per InjectKind:
    //   HostFeature   : x_before + inject_input           (Wan host reuse)
    //   DeviceResidual: x_before + resid1                 (MagCache device slot)
    //   DeviceBlend   : x_before + resid2 + gamma*(resid1-resid2)  (DiCache gamma)
    enum class InjectKind { None, HostFeature, DeviceResidual, DeviceBlend };

    void set_inject_host(ggml_tensor* inject_input, int start, int resume) {
        inject_kind_ = InjectKind::HostFeature;
        inject_input_ = inject_input;
        inject_start_ = start;
        inject_resume_ = resume;
        inject_active_ = true;
    }
    // MagCache device reuse: single residual slot tensor.
    void set_inject_device_residual(ggml_tensor* resid1, int start, int resume) {
        inject_kind_ = InjectKind::DeviceResidual;
        inject_resid1_ = resid1;
        inject_start_ = start;
        inject_resume_ = resume;
        inject_active_ = true;
    }
    // DiCache device reuse: gamma-blend of the 2-deep residual ring.
    void set_inject_device_blend(ggml_tensor* resid1, ggml_tensor* resid2,
                                 ggml_tensor* gamma, int start, int resume) {
        inject_kind_ = InjectKind::DeviceBlend;
        inject_resid1_ = resid1;
        inject_resid2_ = resid2;
        inject_gamma_ = gamma;
        inject_start_ = start;
        inject_resume_ = resume;
        inject_active_ = true;
    }
    bool inject_active() const { return inject_active_; }
    bool inject_at(int i) const { return inject_active_ && i == inject_start_; }
    int inject_resume() const { return inject_resume_; }
    InjectKind inject_kind() const { return inject_kind_; }
    ggml_tensor* inject_input() const { return inject_input_; }
    ggml_tensor* inject_resid1() const { return inject_resid1_; }
    ggml_tensor* inject_resid2() const { return inject_resid2_; }
    ggml_tensor* inject_gamma() const { return inject_gamma_; }

    // ---- DiCache probe metrics. The delta_y/delta_x/gamma reductions mix tapped
    // anchors (probe = BlockOut[m], before = ModelIn) with runner-owned persistent
    // cross-step tensors (prev_probe/prev_input/probe_prev1/probe_prev2). Those
    // persistent tensors are not StateManager slots, so they're threaded in here as
    // named opaque operands and the runner weaves the exact metric form. ----
    struct ProbeMetricOperands {
        ggml_tensor* prev_probe = nullptr;   // last computed step's probe state
        ggml_tensor* prev_input = nullptr;   // last computed step's block input
        ggml_tensor* probe_prev1 = nullptr;  // newest probe residual (probe - before)
        ggml_tensor* probe_prev2 = nullptr;  // 2nd-newest probe residual
        bool want_gamma = false;             // probe_prev1/2 valid -> emit gamma
        bool want_delta_x = false;           // delta_minus error choice
    };
    void set_probe_metrics(const AnchorRef& probe_anchor, const AnchorRef& before_anchor,
                           const ProbeMetricOperands& ops) {
        probe_metrics_ = true;
        probe_anchor_ = probe_anchor;
        before_anchor_ = before_anchor;
        probe_ops_ = ops;
    }
    bool probe_metrics() const { return probe_metrics_; }
    const AnchorRef& probe_anchor() const { return probe_anchor_; }
    const AnchorRef& before_anchor() const { return before_anchor_; }
    const ProbeMetricOperands& probe_ops() const { return probe_ops_; }

private:
    std::unordered_set<std::string> requested_;
    std::unordered_map<std::string, ggml_tensor*> taps_;
    std::unordered_map<std::string, ggml_tensor*> recorded_;
    std::vector<ggml_tensor*> indicator_nodes_;
    std::vector<Indicator> indicators_;
    int stop_after_block_ = -1;
    bool capture_residual_ = false;
    bool inject_active_ = false;
    InjectKind inject_kind_ = InjectKind::None;
    ggml_tensor* inject_input_ = nullptr;   // HostFeature
    ggml_tensor* inject_resid1_ = nullptr;  // Device: newest residual / slot
    ggml_tensor* inject_resid2_ = nullptr;  // DeviceBlend: 2nd-newest residual
    ggml_tensor* inject_gamma_ = nullptr;   // DeviceBlend: [1] gamma scalar
    int inject_start_ = 0;
    int inject_resume_ = -1;
    bool probe_metrics_ = false;
    AnchorRef probe_anchor_;
    AnchorRef before_anchor_;
    ProbeMetricOperands probe_ops_;
};

}  // namespace cache
}  // namespace edgedit
