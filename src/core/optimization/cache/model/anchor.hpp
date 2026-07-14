#pragma once

#include <string>

namespace edgedit {
namespace cache {

// A structural landmark inside a DiT model that the cache system may read.
// Anchors are named by STRUCTURE (block index), never by USE (no "probe point"):
// two methods probing the same block share the same anchor. The set of anchors a
// model exposes is a function of its architecture, not of how many cache methods
// exist. BlockOut[k] is a parametrized family (any k), so a method that changes
// its probe depth needs zero model changes.
struct AnchorRef {
    enum Kind { ModelIn, BlockOut, ModelOut, Modulation };
    Kind kind = ModelIn;
    int  index = -1;  // meaningful for BlockOut / Modulation; ignored otherwise

    static AnchorRef model_in() { return {ModelIn, -1}; }
    static AnchorRef model_out() { return {ModelOut, -1}; }
    static AnchorRef block_out(int k) { return {BlockOut, k}; }
    static AnchorRef modulation(int k) { return {Modulation, k}; }

    bool operator==(const AnchorRef& o) const { return kind == o.kind && index == o.index; }

    // Stable string key for the runner's tap registry and the compiled-graph memo
    // key. Same anchor -> same name across steps (compiled-graph reuse depends on
    // the tap set being stable per decision).
    std::string tap_name() const {
        switch (kind) {
            case ModelIn: return "model_in";
            case ModelOut: return "model_out";
            case BlockOut: return "block_out[" + std::to_string(index) + "]";
            case Modulation: return "modulation[" + std::to_string(index) + "]";
        }
        return "anchor?";
    }
};

}  // namespace cache
}  // namespace edgedit
