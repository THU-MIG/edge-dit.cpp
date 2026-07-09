#include "core/optimization/cache/ir/cache_program.hpp"

#include <sstream>

namespace edgedit {
namespace cache {

namespace {

const char* variant_kind_name(GraphVariantKind k) {
    switch (k) {
        case GraphVariantKind::FULL: return "FULL";
        case GraphVariantKind::REUSE: return "REUSE";
        case GraphVariantKind::PREDICT: return "PREDICT";
        case GraphVariantKind::PROBE: return "PROBE";
    }
    return "?";
}

const char* exec_mode_name(SegmentExecutionMode m) {
    switch (m) {
        case SegmentExecutionMode::FULL_COMPUTE: return "full_compute";
        case SegmentExecutionMode::LOAD_CACHED: return "load_cached";
        case SegmentExecutionMode::PREDICT_FROM_HISTORY: return "predict_from_history";
        case SegmentExecutionMode::PROBE: return "probe";
    }
    return "?";
}

const char* lifetime_name(CacheLifetime l) {
    switch (l) {
        case CacheLifetime::ONE_STEP: return "one_step";
        case CacheLifetime::MULTI_STEP: return "multi_step";
        case CacheLifetime::FULL_INFERENCE: return "full_inference";
    }
    return "?";
}

}  // namespace

// Human-readable text dump (not strict JSON, but close and diff-friendly) for
// --dump-cache-program. Structural check per doc §26.3.
std::string dump_cache_program(const CacheProgram& program) {
    std::ostringstream os;
    os << "cache_program {\n";
    os << "  method: " << program.method_name << "\n";
    os << "  slots: [\n";
    for (const auto& s : program.slots) {
        os << "    { id: " << s.id << ", name: \"" << s.name << "\", lifetime: "
           << lifetime_name(s.lifetime) << ", history_depth: " << s.history_depth << " }\n";
    }
    os << "  ]\n";
    os << "  variants: [\n";
    for (const auto& v : program.variants) {
        os << "    { id: " << v.id << ", kind: " << variant_kind_name(v.kind) << ", segments: [";
        for (size_t i = 0; i < v.segments.size(); ++i) {
            const auto& seg = v.segments[i];
            os << (i ? ", " : "") << "seg" << seg.segment_id << ":" << exec_mode_name(seg.execution);
            if (seg.execution == SegmentExecutionMode::PROBE) {
                os << "(depth=" << seg.probe_depth << ")";
            }
        }
        os << "] }\n";
    }
    os << "  ]\n";
    os << "}\n";
    return os.str();
}

}  // namespace cache
}  // namespace edgedit
