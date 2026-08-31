#include "runtime/model_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <thread>
#include <utility>

#include "utils/rng_philox.hpp"
#include "utils/util.h"
#include "runtime/model_loader.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

namespace edgedit {
namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string requested_backend_name() {
    const char* value = std::getenv("ED_BACKEND");
    if (value == nullptr) {
        return "";
    }
    return value;
}

bool is_auto_backend(const std::string& name) {
    return name.empty() || lowercase(name) == "auto" || lowercase(name) == "default";
}

bool is_generic_gpu_request(const std::string& requested) {
    const std::string request = lowercase(requested);
    return request == "gpu" || request == "cuda" || request == "vulkan" || request == "metal";
}

bool device_name_matches(ggml_backend_dev_t dev, const std::string& requested) {
    if (dev == nullptr) {
        return false;
    }

    const std::string request = lowercase(requested);
    
    if (request == "gpu") {
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    }

    const char* name_c = ggml_backend_dev_name(dev);
    if (name_c == nullptr) {
        return false;
    }
    const std::string name = lowercase(name_c);
    if (request == "metal") {
        // ggml's Metal device is named "MTL0", not "metal"
        return contains(name, "metal") || contains(name, "mtl");
    }
    return contains(name, request);
}

bool is_gpu_device(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return false;
    }
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

int runtime_gpu_device_ordinal(int fallback) {
    const char* value = std::getenv("ED_CLI_SINGLE_VISIBLE_DEVICE");
    if (value != nullptr && value[0] == '1' && value[1] == '\0') {
        return 0;
    }
    return fallback;
}

ggml_backend_t init_explicit_backend(const std::string& requested, int gpu_device_ordinal) {
    const std::string request = lowercase(requested);
    if (request == "cpu") {
        return ggml_backend_cpu_init();
    }

    ggml_backend_load_all_once();
    const size_t device_count = ggml_backend_dev_count();
    const bool use_gpu_ordinal = is_generic_gpu_request(requested) && gpu_device_ordinal >= 0;
    int matched_gpu_index = 0;
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!device_name_matches(dev, requested)) {
            continue;
        }
        if (use_gpu_ordinal && is_gpu_device(dev) && matched_gpu_index++ != gpu_device_ordinal) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend != nullptr) {
            return backend;
        }
    }

    return init_named_backend(requested);
}

std::string available_backend_names() {
    ggml_backend_load_all_once();
    std::string result;
    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* name = ggml_backend_dev_name(dev);
        if (name == nullptr) {
            continue;
        }
        if (!result.empty()) {
            result += ", ";
        }
        result += name;
    }
    return result.empty() ? "none" : result;
}

}  // namespace

ModelRuntime::~ModelRuntime() {
    reset();
}

bool ModelRuntime::init(const ed_context_params_t* params, std::string* error) {
    if (params == nullptr) {
        return fail(error, "ModelRuntime::init got null params");
    }
    return init(*params, error);
}

bool ModelRuntime::init(const ed_context_params_t& params, std::string* error) {
    reset();
    ggml_log_set(ggml_log_callback_default, nullptr);

    if (!init_threads(params, error)) {
        return false;
    }
    if (!init_flags(params, error)) {
        return false;
    }
    if (!init_rng(params, error)) {
        return false;
    }
    if (!init_backends(params, error)) {
        return false;
    }

    ready_ = true;
    return true;
}

namespace {
// Auto-allocate: VRAM to reserve beyond a resident component's weights, for its own
// compute buffer + activations + allocator fragmentation. A resident (non-segmented)
// component's compute buffer is NOT covered by graph_cut_segment_vram_bytes (that only
// bounds offloaded/segmented components), so a component may only stay resident if its
// weights PLUS this headroom fit the budget. Measured DiT compute activations at
// 1024²/20steps/double-forward: sd3 ~3.4G, flux ~1.9G; 4 GiB covers the upper end with
// margin. Undersized headroom is what let sd3 8g fully-resident peak at 10.3G > 8G budget.
constexpr size_t kResidentComputeHeadroom = static_cast<size_t>(4) * 1024 * 1024 * 1024;
// A measured graph buffer excludes allocator fragmentation and small input/output
// bindings. Keep a bounded margin for a phase that stages all of its weights once.
constexpr size_t kPhaseComputeMinimumMargin = static_cast<size_t>(512) * 1024 * 1024;
// Smallest plausible standalone component (VAE ~0.15-0.5G); used by the all-offload
// fallback: if the budget can't even fit one small component + compute headroom,
// offload everything (equivalent to legacy --offload-to-cpu, safest).
constexpr size_t kMinResidentComponentBytes = static_cast<size_t>(512) * 1024 * 1024;
// Fragmentation + large-segment compute slack subtracted when sizing the SEGMENT
// budget for offloaded components. 2 GiB: an offloaded component's segment carries
// several GB of transient compute/activation on top of its weights that
// graph_cut_segment_vram_bytes underestimates. Shrinking the segment budget further
// (tried 3 GiB) does NOT lower qwen-edit 8g's residual 8292 peak: that peak is two
// text-encode partial segments co-resident during staging, not one oversized segment,
// so a smaller split just makes more segments at the same summed footprint. 2 GiB is
// the sweet spot that keeps other models' segments large (fewer staging round-trips).
constexpr size_t kSegmentBudgetSlack = static_cast<size_t>(2) * 1024 * 1024 * 1024;
// Physical core count (excludes SMT/hyperthreads). On this dual-socket Xeon,
// running matmul-heavy graphs on all 192 logical cores is ~2x SLOWER than on the
// running matmul-heavy graphs on all 192 logical cores is ~2x SLOWER than on the
// 96 physical cores: hyperthreads contend for shared AVX-512/AMX vector units and
// extra threads inflate per-node barrier sync. Parse /proc/cpuinfo for distinct
// (physical id, core id) pairs; fall back to hardware_concurrency() if unavailable.
static int detect_physical_cores() {
    std::ifstream f("/proc/cpuinfo");
    if (!f.is_open()) {
        return 0;
    }
    std::set<std::pair<int, int>> cores;
    int phys = -1;
    int core = -1;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("physical id", 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) { phys = std::atoi(line.c_str() + pos + 1); }
        } else if (line.rfind("core id", 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) { core = std::atoi(line.c_str() + pos + 1); }
        } else if (line.empty()) {
            if (phys >= 0 && core >= 0) { cores.insert({phys, core}); }
            phys = -1;
            core = -1;
        }
    }
    if (phys >= 0 && core >= 0) { cores.insert({phys, core}); }
    return static_cast<int>(cores.size());
}
}  // namespace

bool ModelRuntime::init_threads(const ed_context_params_t& params, std::string* error) {
    (void)error;
    n_threads_ = params.n_threads;
    if (n_threads_ <= 0) {
        int physical = detect_physical_cores();
        int logical  = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        n_threads_   = physical > 0 ? physical : logical;
        LOG_INFO("auto thread count: %d (physical cores=%d, logical=%d)",
                 n_threads_, physical, logical);
    }
    return true;
}

bool ModelRuntime::init_flags(const ed_context_params_t& params, std::string* error) {
    (void)error;
    use_mmap_ = params.use_mmap;
    offload_params_to_cpu_ = params.offload_params_to_cpu;
    text_encoder_offload_ = params.text_encoder_offload;
    minimax_h3_stage_lifecycle_ = params.minimax_h3_stage_lifecycle;
    dit_offload_ = params.dit_offload;
    vae_offload_ = params.vae_offload;
    auto_fit_ = params.auto_fit;
    // auto-fit is a superset of auto-allocate: it decides quantization AND placement, and
    // the placement path (plan_component_offload) is gated on auto_allocate_. So enabling
    // auto-fit implicitly enables auto-allocate; the user only needs one flag.
    auto_allocate_ = params.auto_allocate || params.auto_fit;
    fit_width_ = params.fit_width;
    fit_height_ = params.fit_height;
    fit_frames_ = params.fit_frames;
    fit_fps_ = params.fit_fps > 0 ? params.fit_fps : 24;
    free_params_immediately_ = false;
    max_vram_ = params.max_vram_gb;
    max_graph_vram_bytes_ = max_vram_ <= 0.0f
                                 ? 0
                                 : static_cast<size_t>(static_cast<double>(max_vram_) * 1024.0 * 1024.0 * 1024.0);
    flash_attention_ = params.flash_attention;
    circular_x_ = false;
    circular_y_ = false;
    vae_tiling_ = params.vae_tiling;
    return true;
}

bool ModelRuntime::init_rng(const ed_context_params_t& params, std::string* error) {
    (void)params;
    (void)error;
    rng_ = std::make_shared<PhiloxRNG>();
    sampler_rng_ = rng_;
    return true;
}

bool ModelRuntime::init_backends(const ed_context_params_t& params, std::string* error) {
    const std::string requested_backend = requested_backend_name();
    const int gpu_device_ordinal = runtime_gpu_device_ordinal(parallel_enabled() ? parallel_context_->local_rank() : 0);
    if (is_auto_backend(requested_backend)) {
        backends_.backend = init_named_backend();
    } else {
        LOG_INFO("requested backend: %s", requested_backend.c_str());
        backends_.backend = init_explicit_backend(requested_backend, gpu_device_ordinal);
    }

    if (backends_.backend == nullptr) {
        std::string msg = is_auto_backend(requested_backend)
                              ? "failed to initialize default ggml backend"
                              : "failed to initialize requested ggml backend '" + requested_backend +
                                    "'; available backends: " + available_backend_names();
        return fail(error, msg);
    }
    LOG_INFO("default backend: %s", ggml_backend_name(backends_.backend));

#if defined(GGML_USE_CUDA)
    if (ggml_backend_is_cuda(backends_.backend)) {
        size_t cuda_allocation_budget = 0;
        if (auto_fit_ && max_graph_vram_bytes_ > 0) {
            constexpr size_t external_workspace_reserve = static_cast<size_t>(1) * 1024 * 1024 * 1024;
            if (max_graph_vram_bytes_ <= external_workspace_reserve) {
                return fail(error, "--auto-fit --max-vram must exceed the 1 GiB CUDA workspace reserve");
            }
            cuda_allocation_budget = max_graph_vram_bytes_ - external_workspace_reserve;
        }
        if (!ggml_backend_cuda_set_memory_budget(backends_.backend, cuda_allocation_budget)) {
            return fail(error, "failed to configure CUDA memory budget");
        }
        if (cuda_allocation_budget > 0) {
            LOG_INFO("auto-fit: CUDA allocation guard = %.2f GiB (%.2f GiB requested, 1.00 GiB reserved for external workspaces)",
                     cuda_allocation_budget / (1024.0 * 1024.0 * 1024.0),
                     max_graph_vram_bytes_ / (1024.0 * 1024.0 * 1024.0));
        }
    }
#endif

    // Text encoder and VAE no longer run on a dedicated CPU backend. When their
    // offload flags are set they keep weights on CPU but stage to the GPU per
    // encode/decode (compute on GPU), handled by the GGMLRunner offload path via
    // the offload bool passed to each component's constructor. So both share the
    // default (GPU) backend here.
    backends_.clip_backend = backends_.backend;
    backends_.vae_backend = backends_.backend;

    backends_.control_net_backend = backends_.backend;
    if (params.keep_control_net_on_cpu && !ggml_backend_is_cpu(backends_.backend)) {
        backends_.control_net_backend = ggml_backend_cpu_init();
        if (backends_.control_net_backend == nullptr) {
            return fail(error, "failed to initialize CPU backend for ControlNet");
        }
        backends_.control_net_owns_backend = true;
        LOG_INFO("ControlNet backend: CPU");
    }

    maybe_enable_vae_tiling_for_low_vram();

    // Auto-derive a VRAM budget when the user enabled weight offload but gave no
    // explicit --max-vram. Without a budget, graph-cut segmentation is disabled and
    // offload_all_params() copies every weight back to the GPU at once, which OOMs
    // for large DiTs (e.g. FLUX ~22.7GB on a 24GB card). Segment the compute graph
    // against most of the device's free VRAM instead of failing.
    if ((offload_params_to_cpu_ || text_encoder_offload_ || dit_offload_ || vae_offload_) &&
        max_graph_vram_bytes_ == 0 &&
        backends_.backend != nullptr && !ggml_backend_is_cpu(backends_.backend)) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backends_.backend);
        if (dev != nullptr) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            if (free_bytes > 0) {
                // Reserve headroom for fragmentation and non-graph allocations.
                max_graph_vram_bytes_ = static_cast<size_t>(static_cast<double>(free_bytes) * 0.85);
                LOG_INFO("offload enabled without --max-vram; auto graph VRAM budget = %.2f GB "
                         "(0.85 x %.2f GB free) to enable segmented compute",
                         max_graph_vram_bytes_ / (1024.0 * 1024.0 * 1024.0),
                         free_bytes / (1024.0 * 1024.0 * 1024.0));
            }
        }
    }

    return true;
}

// Adaptive offload decision (see header). Called by each pipeline's build_components
// once per major component (DiT / text-encoder / VAE), BEFORE the runner is
// constructed, because a runner's params_backend (GPU-resident vs CPU-staged) is
// fixed at construction. Only active under --auto-allocate; otherwise returns the
// legacy global offload flag so existing behavior is untouched.
//
// The budget is a HARD cap: effective = min(user --max-vram, live free VRAM). Each
// component's quantized weight bytes are compared against a running tally seeded with
// that budget; a resident component debits the tally. Components are decided in
// priority order DiT -> TE -> VAE (the caller passes them in that order), so the
// largest / most-reused weights get first claim on resident VRAM. After all three are
// decided the caller invokes finalize_auto_segment_budget() to set the graph VRAM
// budget for whatever ended up offloaded, using the leftover (budget - resident).
// Sum a component's weights using the EFFECTIVE (post-quantization) type: set_wtype_override
// records the target type in expected_type, so nbytes() alone (which uses `type`) would
// overestimate a q8/q4 component. Mirrors the effective-type logic in collect_wtype_stat.
size_t ModelRuntime::resident_headroom_bytes() const {
    // Headroom = space reserved for a resident component's own compute buffer +
    // activations (NOT bounded by the segment budget, which only covers offloaded
    // components). Prefer the measured DiT compute buffer (real activation footprint at
    // the target resolution) over the fixed 4 GiB constant.
    //
    // The DiT measure covers the DiT's activations, but the VAE decode allocates its own
    // (larger, ~0.9 GB) compute buffer from the same budget, plus CUDA pool
    // fragmentation. Add a fixed VAE+fragmentation allowance instead of a blind floor so
    // the estimate tracks the measured value (small models get small headroom) while
    // still covering the largest resident component. ~1.5 GiB covers VAE compute (~0.9)
    // + fragmentation, and keeps flux at ~2.4 GiB (vs the old blanket 4).
    if (measured_dit_headroom_ > 0) {
        const size_t vae_and_frag_allowance = (static_cast<size_t>(3) * 1024 * 1024 * 1024) / 2;  // 1.5 GiB
        return measured_dit_headroom_ + measured_dit_headroom_ / 10 + vae_and_frag_allowance;
    }
    return kResidentComputeHeadroom;
}

static bool starts_with_any_prefix(const std::string& name,
                                   const std::vector<std::string>& weight_prefixes) {
    for (const std::string& prefix : weight_prefixes) {
        if (name.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

static std::string component_prefix_label(const std::vector<std::string>& weight_prefixes) {
    std::string label;
    for (const std::string& prefix : weight_prefixes) {
        if (!label.empty()) {
            label += "+";
        }
        label += prefix.empty() ? "<all>" : prefix;
    }
    return label.empty() ? "<none>" : label;
}

static size_t storage_component_effective_bytes(const ::ModelLoader& loader,
                                                const std::vector<std::string>& weight_prefixes) {
    size_t comp_bytes = 0;
    for (const auto& item : loader.get_tensor_storage_map()) {
        if (!starts_with_any_prefix(item.first, weight_prefixes)) {
            continue;  // not in this component
        }
        TensorStorage ts = item.second;
        if (ts.expected_type != GGML_TYPE_COUNT && ts.expected_type != ts.type) {
            ts.type = ts.expected_type;  // account for quantization
        }
        comp_bytes += static_cast<size_t>(ts.nbytes());
    }
    return comp_bytes;
}

size_t ModelRuntime::component_memory_bytes(const ::ModelLoader& loader,
                                            const std::vector<std::string>& weight_prefixes) const {
    if (component_memory_estimator_) {
        const size_t materialized_bytes = component_memory_estimator_(loader, weight_prefixes);
        if (materialized_bytes > 0) {
            return materialized_bytes;
        }
    }
    return storage_component_effective_bytes(loader, weight_prefixes);
}

bool ModelRuntime::plan_component_offload(const ::ModelLoader& loader,
                                          const std::string& weight_prefix,
                                          size_t& remaining_free_bytes) {
    return plan_component_offload(loader,
                                  std::vector<std::string>{weight_prefix},
                                  remaining_free_bytes);
}

bool ModelRuntime::plan_component_offload(const ::ModelLoader& loader,
                                          const std::vector<std::string>& weight_prefixes,
                                          size_t& remaining_free_bytes) {
    // Not in auto-allocate mode: keep legacy behavior (global offload flag).
    if (!auto_allocate_) {
        return offload_params_to_cpu_;
    }

    // If the runtime backend is CPU there is no GPU to fit into; offload is moot and
    // the resident path is correct (weights already live where compute happens).
    if (backends_.backend == nullptr || ggml_backend_is_cpu(backends_.backend)) {
        return offload_params_to_cpu_;
    }

    const size_t comp_bytes = component_memory_bytes(loader, weight_prefixes);
    if (comp_bytes == 0) {
        // No weights matched this prefix (component absent) -> honor the global flag.
        return offload_params_to_cpu_;
    }

    // All-offload fallback: if the budget can't even fit one small component plus the
    // compute headroom, nothing can stay resident safely -> offload everything (legacy
    // --offload-to-cpu behavior, safest). Prevents a tiny-budget fully-resident from overshooting.
    const size_t headroom = resident_headroom_bytes();
    const std::string component_label = component_prefix_label(weight_prefixes);
    if (remaining_free_bytes < kMinResidentComponentBytes + headroom) {
        LOG_INFO("auto-allocate: '%s' budget %.2f GB too small for resident+compute headroom "
                 "-> OFFLOAD (all-offload fallback)",
                 component_label.c_str(),
                 remaining_free_bytes / (1024.0 * 1024.0 * 1024.0));
        any_component_offloaded_ = true;
        return true;
    }

    // Resident iff weights PLUS compute headroom fit the remaining budget. The headroom
    // covers the resident component's own compute buffer + activations (measured when a
    // target resolution is known, else the fixed constant), which are NOT bounded by the
    // segment budget (that only bounds offloaded components). remaining_free_bytes is the
    // running tally = effective_budget minus components already decided resident.
    bool fits = comp_bytes + headroom <= remaining_free_bytes;

    // Segment-safety guard: if an earlier component was already offloaded (it streams in
    // per-step segments), making THIS component resident shrinks the leftover segment
    // budget = (remaining_free - comp_bytes - headroom). An offloaded component's segment
    // needs room for its activations (~headroom) PLUS a meaningful weight chunk, or it
    // over-commits VRAM at run time (resident set + segment peak > budget). This is what
    // made kontext 8g overshoot: TE flipped to resident once measure shrank the headroom,
    // squeezing the offloaded DiT's segment budget to ~1 GB. So: don't grant residency if
    // it would drop the segment budget below a safe floor -- keep this component offloaded
    // instead (its own offload cost is small; protecting the big DiT's segments matters
    // more). Only active once something is already offloaded (image-only, all-resident
    // budgets are unaffected).
    if (fits && any_component_offloaded_) {
        const size_t kSafeSegmentBudget = 2 * headroom;  // segment activation + a weight chunk
        const size_t leftover_after = remaining_free_bytes - comp_bytes;  // before subtracting headroom
        if (leftover_after < headroom + kSafeSegmentBudget) {
            LOG_INFO("auto-allocate: '%s' %.2f GB fits but would squeeze offloaded segment budget "
                     "(%.2f GB left < %.2f GB safe) -> OFFLOAD instead",
                     component_label.c_str(),
                     comp_bytes / (1024.0 * 1024.0 * 1024.0),
                     (leftover_after > headroom ? (leftover_after - headroom) : 0) / (1024.0 * 1024.0 * 1024.0),
                     kSafeSegmentBudget / (1024.0 * 1024.0 * 1024.0));
            fits = false;
        }
    }

    if (fits) {
        remaining_free_bytes -= comp_bytes;   // this component sits resident on GPU
        resident_bytes_total_ += comp_bytes;  // accumulated for finalize_auto_segment_budget()
        LOG_INFO("auto-allocate: '%s' %.2f GB -> RESIDENT (%.2f GB budget left)",
                 component_label.c_str(),
                 comp_bytes / (1024.0 * 1024.0 * 1024.0),
                 remaining_free_bytes / (1024.0 * 1024.0 * 1024.0));
        return false;
    }

    LOG_INFO("auto-allocate: '%s' %.2f GB > %.2f GB budget left -> OFFLOAD+segment",
             component_label.c_str(),
             comp_bytes / (1024.0 * 1024.0 * 1024.0),
             remaining_free_bytes / (1024.0 * 1024.0 * 1024.0));
    any_component_offloaded_ = true;
    return true;
}

// After all components of a pipeline have been decided via plan_component_offload,
// set the graph VRAM budget for the offloaded ones: whatever budget is left after the
// resident components, minus compute headroom. Because graph_cut_segment_vram_bytes
// already counts each segment's compute + weights + IO, this cap keeps
// (resident + max_segment) within the effective budget. No-op outside auto-allocate.
void ModelRuntime::finalize_auto_segment_budget(size_t effective_budget_bytes,
                                                size_t additional_slack_bytes) {
    if (!auto_allocate_) {
        return;
    }
    size_t leftover = 0;
    const size_t reserved_slack = kSegmentBudgetSlack + additional_slack_bytes;
    if (effective_budget_bytes > resident_bytes_total_ + reserved_slack) {
        leftover = effective_budget_bytes - resident_bytes_total_ - reserved_slack;
    }
    // Floor: if the leftover is tiny (resident nearly filled the budget) an offloaded
    // component still needs *some* budget to segment against; use a 1 GB floor so a
    // single segment can at least stage. Better a large segment than a hard abort.
    const size_t kMinSegmentBudget = static_cast<size_t>(1) * 1024 * 1024 * 1024;
    if (leftover < kMinSegmentBudget) {
        leftover = kMinSegmentBudget;
    }
    max_graph_vram_bytes_ = leftover;
    LOG_INFO("auto-allocate: segment budget = %.2f GB (effective %.2f GB - resident %.2f GB - %.2f GB staging slack)",
             max_graph_vram_bytes_ / (1024.0 * 1024.0 * 1024.0),
             effective_budget_bytes / (1024.0 * 1024.0 * 1024.0),
             resident_bytes_total_ / (1024.0 * 1024.0 * 1024.0),
             reserved_slack / (1024.0 * 1024.0 * 1024.0));
}

void ModelRuntime::replan_dit_quant_for_budget(::ModelLoader& loader) {
    // Only active under auto-fit, and only on a real GPU backend.
    if (!auto_fit_) {
        return;
    }
    if (backends_.backend == nullptr || ggml_backend_is_cpu(backends_.backend)) {
        return;
    }

    const std::string kDiT = "model.diffusion_model";
    std::vector<std::string> te_prefixes = {"text_encoders"};
    if (ed_version_is_ltxav(loader.version())) {
        // LTX's runner owns the Gemma subcomponent under this exact prefix and
        // also accounts for the adjacent projection in the same planning unit.
        te_prefixes = {"text_encoders.llm", "text_embedding_projection"};
    }

    const size_t budget = effective_budget_bytes();
    if (budget == 0) {
        return;
    }

    // auto-fit also OWNS the text-encoder quantization. TE (CLIP+T5) is often bf16 and
    // large (~9 GB for flux's T5), but its compute time is negligible vs the DiT, so
    // quantizing it to q8_0 is near-lossless in quality, costs no speed, and halves its
    // footprint. Lowering TE to q8_0 frees budget so the DiT can stay resident without the
    // DiT+TE bf16 pair overflowing VRAM (the 24G OOM cause). A source already below q8_0
    // stays at its existing precision; tensor_should_be_converted also protects embeddings/projections.
    const size_t te_before = component_memory_bytes(loader, te_prefixes);
    if (te_before > 0) {
        for (const std::string& prefix : te_prefixes) {
            loader.override_component_wtype(prefix, GGML_TYPE_Q8_0);
        }
        const size_t te_after = component_memory_bytes(loader, te_prefixes);
        if (te_after != te_before) {
            LOG_INFO("auto-fit: TE quant set to q8_0 (superseding global --type for TE), %.2f GB -> %.2f GB",
                     te_before / (1024.0 * 1024.0 * 1024.0),
                     te_after / (1024.0 * 1024.0 * 1024.0));
        }
    }

    // auto-fit owns the DiT decision after the global --type policy is applied. It lowers
    // eligible high-precision tensors toward q8_0 -> q4_k, but never increases an existing
    // lower-precision tensor. On a 4090 q8_0 is the best high-quality starting point (bf16
    // is both larger and slower due to the FP32-accumulate penalty), so we do not retain
    // bf16 merely to satisfy the automatic budget.
    const size_t bytes_before = component_memory_bytes(loader, std::vector<std::string>{kDiT});
    loader.override_component_wtype(kDiT, GGML_TYPE_Q8_0);
    const size_t q8_bytes = component_memory_bytes(loader, std::vector<std::string>{kDiT});
    std::vector<std::pair<std::string, ggml_type>> q8_precision_state;
    for (const auto& item : loader.get_tensor_storage_map()) {
        if (item.first.rfind(kDiT, 0) == 0) {
            q8_precision_state.emplace_back(item.first, item.second.expected_type);
        }
    }
    if (q8_bytes != bytes_before) {
        LOG_INFO("auto-fit: DiT quant set to q8_0 (superseding global --type for DiT), %.2f GB",
                 q8_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    // DiT is decided FIRST in plan_component_offload (caller passes DiT -> TE -> VAE), so
    // it gets first claim on the budget: it stays resident iff its weights + compute
    // headroom fit. Mirror exactly that test (do NOT also subtract TE/VAE -- they compete
    // for whatever is left AFTER DiT), matching plan_component_offload's
    // `comp_bytes + headroom <= remaining` with remaining == full budget. Use the same
    // resident_headroom_bytes() (measured when a target resolution is known).
    const size_t headroom = resident_headroom_bytes();
    if (budget <= headroom) {
        return;  // budget too small for even headroom; DiT can't be resident, leave at q8_0
    }
    const size_t dit_budget = budget - headroom;

    if (q8_bytes <= dit_budget) {
        LOG_INFO("auto-fit: DiT source-or-Q8 precision %.2f GB fits %.2f GB budget -> resident",
                 q8_bytes / (1024.0 * 1024.0 * 1024.0),
                 dit_budget / (1024.0 * 1024.0 * 1024.0));
        return;  // q8_0 already fits -> keep it (highest precision in the ladder)
    }

    // The source-or-Q8 candidate does not fit -> try q4_k (the floor). Keep it if it fits;
    // otherwise restore the candidate precision and let plan_component_offload offload it.
    loader.override_component_wtype(kDiT, GGML_TYPE_Q4_K);
    const size_t q4_bytes = component_memory_bytes(loader, std::vector<std::string>{kDiT});
    if (q4_bytes <= dit_budget) {
        LOG_INFO("auto-fit: DiT source-or-Q8 %.2f GB -> q4_K %.2f GB to fit %.2f GB budget (resident)",
                 q8_bytes / (1024.0 * 1024.0 * 1024.0),
                 q4_bytes / (1024.0 * 1024.0 * 1024.0),
                 dit_budget / (1024.0 * 1024.0 * 1024.0));
        return;
    }

    // Even q4_k does not fit: restore the post-Q8 planning state. This keeps BF16/F16
    // sources at Q8 while preserving any user-supplied Q5/Q6/Q4-or-lower tensors instead
    // of accidentally increasing their precision during the fallback.
    if (q8_bytes != q4_bytes) {
        auto& storage_map = loader.get_tensor_storage_map();
        for (const auto& state : q8_precision_state) {
            storage_map.at(state.first).expected_type = state.second;
        }
        LOG_INFO("auto-fit: DiT does not fit %.2f GB budget even at q4_K (q4=%.2f GB) -> restore source-or-Q8 precision, will offload",
                 dit_budget / (1024.0 * 1024.0 * 1024.0),
                 q4_bytes / (1024.0 * 1024.0 * 1024.0));
    } else {
        LOG_INFO("auto-fit: source DiT is already q4_K or lower and does not fit %.2f GB budget -> preserve source precision and offload",
                 dit_budget / (1024.0 * 1024.0 * 1024.0));
    }
}

size_t ModelRuntime::effective_budget_bytes() const {
    if (backends_.backend == nullptr || ggml_backend_is_cpu(backends_.backend)) {
        return 0;
    }
    size_t live_free = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends_.backend);
    if (dev != nullptr) {
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &live_free, &total_bytes);
    }
    // max_graph_vram_bytes_ holds the user's --max-vram (bytes) here, 0 if unset.
    if (max_graph_vram_bytes_ > 0 && max_graph_vram_bytes_ < live_free) {
        return max_graph_vram_bytes_;  // user budget is the tighter (hard) cap
    }
    return live_free;
}

size_t ModelRuntime::phase_staging_budget_bytes() const {
    if (backends_.backend == nullptr || ggml_backend_is_cpu(backends_.backend)) {
        return 0;
    }

    size_t live_free = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends_.backend);
    if (dev != nullptr) {
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &live_free, &total_bytes);
    }

    size_t policy_budget = live_free;
    if (auto_allocate_ && max_vram_ > 0.0f) {
        const size_t user_budget = static_cast<size_t>(max_vram_ * 1024.0 * 1024.0 * 1024.0);
        policy_budget = user_budget > resident_bytes_total_
                            ? user_budget - resident_bytes_total_
                            : 0;
    } else if (max_graph_vram_bytes_ > 0) {
        // In manual offload mode this is either the explicit --max-vram limit
        // or the runtime's safety-scaled live-free budget.
        policy_budget = max_graph_vram_bytes_;
    }
    return std::min(live_free, policy_budget);
}

size_t ModelRuntime::phase_staging_headroom_bytes(size_t phase_compute_bytes) const {
    if (phase_compute_bytes == 0) {
        return resident_headroom_bytes();
    }
    const size_t measured_margin = phase_compute_bytes / 10;
    const size_t margin = std::max(measured_margin, kPhaseComputeMinimumMargin);
    return phase_compute_bytes + margin;
}

// Segment-VRAM budget for the text encoder specifically. Unlike the DiT, the TE
// (T5-XXL ~9.8G / qwen2.5vl ~15G) is SMALLER than the global graph budget
// (0.85 x free, ~20G+, or the user's --max-vram), so apply_max_vram_budget merges
// all its cut segments into one -> the whole TE stages to the GPU at once -> OOM on
// mid/small cards. The DiT is larger than the budget so it segments naturally; the TE
// does not. Fix: when the TE is offloaded, hand it a SMALL fixed-size segment budget so
// it stages in many small chunks, matching the DiT's per-single-block granularity
// (~1 GiB/segment) rather than a few huge ones.
//
// Why a fixed target and not te_params_bytes/2: te_bytes/2 scales with the ENCODER size,
// so a 9.4G T5 gets 4.7G segments (only ~3 segments, ~3G each) — far too coarse for a
// 8/12G card. The DiT does NOT size segments by model size; it segments against a VRAM
// budget so each chunk is small regardless of total model size. Mirror that: target a
// ~1 GiB segment so the TE stages like the DiT (T5 -> ~8-10 segments, ~1G each).
//
// te_params_bytes is the TE's weight-buffer size (conditioner_->get_params_buffer_size()).
// Resident TE (not offloaded) is untouched: we return the global value unchanged.
size_t ModelRuntime::text_encoder_segment_budget(size_t te_params_bytes, bool component_offloaded) const {
    // Resident TE: no staging, keep the existing global budget (no behavior change).
    if (!component_offloaded && !clip_offload_params_to_cpu()) {
        return max_graph_vram_bytes_;
    }
    if (te_params_bytes == 0) {
        return max_graph_vram_bytes_;  // unknown TE size: fall back, don't over-constrain
    }
    // Target per-segment size: match the DiT's per-single-block granularity (~1 GiB). Small
    // and FIXED (not scaled by encoder size) so a big T5 stages in many small chunks. A TE
    // smaller than one target simply stays a single segment (correct: nothing to split).
    const size_t kTargetSegmentBytes = (static_cast<size_t>(5) * 1024 * 1024 * 1024) / 4;  // 1.25 GiB
    size_t cap = kTargetSegmentBytes;
    // Cap by the ACTUAL free VRAM (× safety fraction) at this point. This method runs after
    // the DiT params buffer is allocated (register_tensors: DiT alloc precedes conditioner
    // set). When the DiT is resident (--text-encoder-offload only, DiT stays on GPU ~22.7G),
    // live_free is already tiny (~1.3G), so the TE segments even finer to fit the remainder.
    // The fraction leaves room for the segment's own compute/activation buffer.
    size_t live_free = 0;
    if (backends_.backend != nullptr && !ggml_backend_is_cpu(backends_.backend)) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backends_.backend);
        if (dev != nullptr) {
            size_t total_bytes = 0;
            ggml_backend_dev_memory(dev, &live_free, &total_bytes);
        }
    }
    if (live_free > 0) {
        const size_t free_cap = static_cast<size_t>(static_cast<double>(live_free) * 0.6);
        cap = std::min(cap, free_cap);
    }
    if (max_graph_vram_bytes_ > 0) {
        cap = std::min(cap, max_graph_vram_bytes_);
    }
    // Floor: a segment still needs SOME budget to stage against. 512 MiB lets a single small
    // segment run rather than aborting. On a card so full that even this overshoots, the run
    // was going to OOM regardless; a floor at least attempts it.
    const size_t kMinSegmentBudget = static_cast<size_t>(512) * 1024 * 1024;
    if (cap < kMinSegmentBudget) {
        cap = kMinSegmentBudget;
    }
    LOG_INFO("text-encoder segment budget = %.2f GB (TE weights %.2f GB, live_free %.2f GB, "
             "offloaded -> force segmentation)",
             cap / (1024.0 * 1024.0 * 1024.0),
             te_params_bytes / (1024.0 * 1024.0 * 1024.0),
             live_free / (1024.0 * 1024.0 * 1024.0));
    return cap;
}

// so consumer cards stay under their VRAM wall without a manual flag.
void ModelRuntime::maybe_enable_vae_tiling_for_low_vram() {
    if (vae_tiling_.enabled || vae_tiling_.force_disable) {
        return;  // user explicitly set tiling on or off; respect it, skip low-VRAM auto-enable
    }
    if (backends_.vae_backend == nullptr || ggml_backend_is_cpu(backends_.vae_backend)) {
        return;  // VAE runs on CPU, GPU tiling does not apply
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backends_.vae_backend);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return;
    }
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    const double total_gib = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
    constexpr double kLowVramThresholdGiB = 25.0;
    if (total_bytes == 0 || total_gib > kLowVramThresholdGiB) {
        return;  // large GPU: leave VAE untiled for max throughput
    }
    vae_tiling_.enabled        = true;
    vae_tiling_.rel_size_x     = 5.0f;  // ~32x32 latent tile: matches min VAE peak (empirically measured)
    vae_tiling_.rel_size_y     = 5.0f;
    if (vae_tiling_.target_overlap <= 0.0f) {
        vae_tiling_.target_overlap = 0.25f;
    }
    LOG_INFO("auto-enabled VAE tiling (GPU total VRAM %.1f GiB <= %.0f GiB threshold)",
             total_gib, kLowVramThresholdGiB);
}

void ModelRuntime::reset() {
    ready_ = false;
    rng_.reset();
    sampler_rng_.reset();
    release_backends();

    n_threads_ = 0;
    use_mmap_ = false;
    offload_params_to_cpu_ = false;
    text_encoder_offload_ = false;
    minimax_h3_stage_lifecycle_ = false;
    dit_offload_ = false;
    vae_offload_ = false;
    free_params_immediately_ = false;
    max_vram_ = 0.0f;
    max_graph_vram_bytes_ = 0;
    component_memory_estimator_ = {};
    flash_attention_ = false;
    circular_x_ = false;
    circular_y_ = false;
}

void ModelRuntime::release_backends() {
    if (backends_.control_net_owns_backend && backends_.control_net_backend != nullptr) {
        ggml_backend_free(backends_.control_net_backend);
    }
    if (backends_.vae_owns_backend && backends_.vae_backend != nullptr) {
        ggml_backend_free(backends_.vae_backend);
    }
    if (backends_.clip_owns_backend && backends_.clip_backend != nullptr) {
        ggml_backend_free(backends_.clip_backend);
    }
    if (backends_.backend != nullptr) {
        ggml_backend_free(backends_.backend);
    }
    backends_ = {};
}

bool ModelRuntime::fail(std::string* error, const std::string& msg) {
    if (error != nullptr) {
        *error = msg;
    }
    LOG_ERROR("%s", msg.c_str());
    reset();
    return false;
}

} // namespace edgedit
