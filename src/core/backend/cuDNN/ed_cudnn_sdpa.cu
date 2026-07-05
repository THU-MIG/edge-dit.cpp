#include "ed_cudnn_sdpa.h"

#include "common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace fe = cudnn_frontend;

namespace {

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t SEQ_LEN_Q_UID = 5;
constexpr int64_t SEQ_LEN_KV_UID = 6;
constexpr int64_t MIN_CUDNN_SDPA_FAST_SEQ = 4096;

struct ed_cudnn_sdpa_key {
    int device = 0;
    int64_t b = 0;
    int64_t h = 0;
    int64_t sq = 0;
    int64_t sk = 0;
    int64_t sk_actual = 0;
    int64_t d = 0;
    float attn_scale = 0.0f;
    bool padding_mask = false;
    int64_t q_stride[4] = {};
    int64_t k_stride[4] = {};
    int64_t v_stride[4] = {};
    int64_t o_stride[4] = {};

    bool operator==(const ed_cudnn_sdpa_key & other) const {
        return device == other.device && b == other.b && h == other.h && sq == other.sq && sk == other.sk &&
               sk_actual == other.sk_actual && d == other.d && attn_scale == other.attn_scale && padding_mask == other.padding_mask &&
               std::memcmp(q_stride, other.q_stride, sizeof(q_stride)) == 0 &&
               std::memcmp(k_stride, other.k_stride, sizeof(k_stride)) == 0 &&
               std::memcmp(v_stride, other.v_stride, sizeof(v_stride)) == 0 &&
               std::memcmp(o_stride, other.o_stride, sizeof(o_stride)) == 0;
    }
};

struct ed_cudnn_sdpa_key_hash {
    size_t operator()(const ed_cudnn_sdpa_key & key) const {
        size_t h = std::hash<int>()(key.device);
        auto mix = [&h](int64_t v) {
            h ^= std::hash<int64_t>()(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(key.b);
        mix(key.h);
        mix(key.sq);
        mix(key.sk);
        mix(key.sk_actual);
        mix(key.d);
        uint32_t scale_bits = 0;
        std::memcpy(&scale_bits, &key.attn_scale, sizeof(scale_bits));
        mix(scale_bits);
        mix(key.padding_mask ? 1 : 0);
        for (int i = 0; i < 4; ++i) {
            mix(key.q_stride[i]);
            mix(key.k_stride[i]);
            mix(key.v_stride[i]);
            mix(key.o_stride[i]);
        }
        return h;
    }
};

struct ed_cudnn_sdpa_plan {
    cudnnHandle_t handle = nullptr;
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_size = 0;
    void * workspace = nullptr;
    half * q_f16 = nullptr;
    half * o_f16 = nullptr;
    int32_t * seq_len_q = nullptr;
    int32_t * seq_len_kv = nullptr;
    int64_t elements = 0;

    ~ed_cudnn_sdpa_plan() {
        if (seq_len_kv != nullptr) {
            cudaFree(seq_len_kv);
        }
        if (seq_len_q != nullptr) {
            cudaFree(seq_len_q);
        }
        if (o_f16 != nullptr) {
            cudaFree(o_f16);
        }
        if (q_f16 != nullptr) {
            cudaFree(q_f16);
        }
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        if (handle != nullptr) {
            cudnnDestroy(handle);
        }
    }
};

enum class ed_cudnn_sdpa_plan_state {
    BUILDING,
    READY,
    FAILED,
};

struct ed_cudnn_sdpa_plan_entry {
    std::mutex mutex;
    std::unique_ptr<ed_cudnn_sdpa_plan> plan;
    ed_cudnn_sdpa_plan_state state = ed_cudnn_sdpa_plan_state::BUILDING;
    ed_cudnn_sdpa_result_t result = ED_CUDNN_SDPA_BUILD_PENDING;
};

static std::mutex g_plan_mutex;
static std::unordered_map<ed_cudnn_sdpa_key, std::shared_ptr<ed_cudnn_sdpa_plan_entry>, ed_cudnn_sdpa_key_hash> g_plan_cache;

struct ed_cudnn_sdpa_profile_stat {
    uint64_t calls = 0;
    double get_plan_ms = 0.0;
    double q_cast_ms = 0.0;
    double execute_ms = 0.0;
    double o_cast_ms = 0.0;
};

static std::mutex g_profile_mutex;
static std::unordered_map<ed_cudnn_sdpa_key, ed_cudnn_sdpa_profile_stat, ed_cudnn_sdpa_key_hash> g_profile_stats;

static bool profile_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("ED_PROFILE_CUDNN_SDPA");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

static bool cudnn_sdpa_disabled() {
    static const bool disabled = [] {
        const char * env = getenv("ED_DISABLE_CUDNN_SDPA");
        return env != nullptr && atoi(env) != 0;
    }();
    return disabled;
}

static bool cudnn_sdpa_vec_convert_disabled() {
    static const bool disabled = [] {
        const char * env = getenv("ED_DISABLE_CUDNN_SDPA_VEC_CONVERT");
        return env != nullptr && atoi(env) != 0;
    }();
    return disabled;
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static std::vector<std::string> split_paths(const char * paths) {
    std::vector<std::string> result;
    if (paths == nullptr || paths[0] == '\0') {
        return result;
    }

    std::stringstream ss(paths);
    std::string item;
    while (std::getline(ss, item, ':')) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

static void append_unique_path(std::vector<std::string> & paths, const std::string & path) {
    if (!path.empty() && std::find(paths.begin(), paths.end(), path) == paths.end()) {
        paths.push_back(path);
    }
}

static bool ensure_nvrtc_loaded() {
#if defined(__linux__)
    static const bool loaded = [] {
        void * handle = dlopen("libnvrtc.so.12", RTLD_NOW | RTLD_GLOBAL);
        if (handle != nullptr) {
            return true;
        }

        std::vector<std::string> paths;
        for (const auto & path : split_paths(getenv("LD_LIBRARY_PATH"))) {
            append_unique_path(paths, path);
        }
#ifdef ED_CUDNN_RUNTIME_LIBRARY_DIRS
        for (const auto & path : split_paths(ED_CUDNN_RUNTIME_LIBRARY_DIRS)) {
            append_unique_path(paths, path);
        }
#endif

        const char * initial_error = dlerror();
        std::string last_error = initial_error != nullptr ? initial_error : "";
        for (const std::string & path : paths) {
            const std::string candidate = path + "/libnvrtc.so.12";
            handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (handle != nullptr) {
                if (profile_enabled()) {
                    GGML_LOG_INFO("ED_CUDNN_SDPA loaded %s\n", candidate.c_str());
                }
                return true;
            }
            const char * err = dlerror();
            if (err != nullptr) {
                last_error = err;
            }
        }

        GGML_LOG_WARN("ED_CUDNN_SDPA could not load libnvrtc.so.12 before cuDNN graph build: %s\n", last_error.c_str());
        return false;
    }();
    return loaded;
#else
    return true;
#endif
}

static size_t type_size(ggml_type type) {
    return ggml_type_size(type) / ggml_blck_size(type);
}

static bool is_contiguous_3d(const ggml_tensor * t) {
    const size_t ts = type_size(t->type);
    return t->nb[0] == ts && t->nb[1] == (size_t) t->ne[0] * ts && t->nb[2] == (size_t) t->ne[1] * t->nb[1] &&
           t->nb[3] == (size_t) t->ne[2] * t->nb[2];
}

static bool supports_tensor_shape(const ggml_tensor * t, ggml_type type, int64_t d, int64_t s, int64_t h) {
    return t != nullptr && t->type == type && t->ne[0] == d && t->ne[1] == s && t->ne[2] == h && t->ne[3] == 1 &&
           is_contiguous_3d(t);
}

static void log_unsupported_shape(const char * reason, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * dst, const ggml_tensor * mask) {
    if (!profile_enabled()) {
        return;
    }
    GGML_LOG_INFO("ED_CUDNN_SDPA unsupported: %s q=(type=%d ne=[%lld,%lld,%lld,%lld]) k=(type=%d ne=[%lld,%lld,%lld,%lld]) v=(type=%d ne=[%lld,%lld,%lld,%lld]) dst=(type=%d ne=[%lld,%lld,%lld,%lld]) mask=%s\n",
                  reason,
                  q != nullptr ? (int) q->type : -1,
                  q != nullptr ? (long long) q->ne[0] : -1,
                  q != nullptr ? (long long) q->ne[1] : -1,
                  q != nullptr ? (long long) q->ne[2] : -1,
                  q != nullptr ? (long long) q->ne[3] : -1,
                  k != nullptr ? (int) k->type : -1,
                  k != nullptr ? (long long) k->ne[0] : -1,
                  k != nullptr ? (long long) k->ne[1] : -1,
                  k != nullptr ? (long long) k->ne[2] : -1,
                  k != nullptr ? (long long) k->ne[3] : -1,
                  v != nullptr ? (int) v->type : -1,
                  v != nullptr ? (long long) v->ne[0] : -1,
                  v != nullptr ? (long long) v->ne[1] : -1,
                  v != nullptr ? (long long) v->ne[2] : -1,
                  v != nullptr ? (long long) v->ne[3] : -1,
                  dst != nullptr ? (int) dst->type : -1,
                  dst != nullptr ? (long long) dst->ne[0] : -1,
                  dst != nullptr ? (long long) dst->ne[1] : -1,
                  dst != nullptr ? (long long) dst->ne[2] : -1,
                  dst != nullptr ? (long long) dst->ne[3] : -1,
                  mask != nullptr ? "present" : "null");
}

static bool make_key(const ggml_tensor * dst, ed_cudnn_sdpa_key & key, int device) {
    if (dst == nullptr) {
        log_unsupported_shape("null_dst", nullptr, nullptr, nullptr, dst, nullptr);
        return false;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (q == nullptr || k == nullptr || v == nullptr || dst == nullptr) {
        log_unsupported_shape("null_tensor", q, k, v, dst, mask);
        return false;
    }
    if ((q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) ||
        k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 ||
        dst->type != GGML_TYPE_F32) {
        log_unsupported_shape("type", q, k, v, dst, mask);
        return false;
    }

    const int64_t d = q->ne[0];
    const int64_t sq = q->ne[1];
    const int64_t sk = k->ne[1];
    const int64_t h = q->ne[2];
    float attn_scale = 0.0f;
    std::memcpy(&attn_scale, dst->op_params, sizeof(attn_scale));
    if (!std::isfinite(attn_scale) || attn_scale <= 0.0f) {
        log_unsupported_shape("attn_scale", q, k, v, dst, mask);
        return false;
    }
    int64_t sk_actual = sk;
    bool padding_mask = false;

    if (mask != nullptr) {
        if (mask->type != GGML_TYPE_F16 || mask->ne[0] != sk || mask->ne[1] != sq || mask->ne[2] != 1 || mask->ne[3] != 1) {
            log_unsupported_shape("mask_shape", q, k, v, dst, mask);
            return false;
        }

        sk_actual = sk - (sk - sq);
        if (sk_actual <= 0 || sk_actual > sk) {
            log_unsupported_shape("mask_actual_len", q, k, v, dst, mask);
            return false;
        }
        padding_mask = sk_actual < sk;
    }

    if (d <= 0 || sq <= 0 || sk <= 0 || h <= 0 || q->ne[3] != 1 || k->ne[2] != h || v->ne[2] != h || k->ne[3] != 1 || v->ne[3] != 1) {
        log_unsupported_shape("rank_or_dim", q, k, v, dst, mask);
        return false;
    }
    if (!supports_tensor_shape(q, q->type, d, sq, h) ||
        !supports_tensor_shape(k, GGML_TYPE_F16, d, sk, h) ||
        !supports_tensor_shape(v, GGML_TYPE_F16, d, sk, h)) {
        log_unsupported_shape("input_layout", q, k, v, dst, mask);
        return false;
    }

    if (dst->ne[0] != d || dst->ne[1] != h || dst->ne[2] != sq || dst->ne[3] != 1) {
        log_unsupported_shape("dst_shape", q, k, v, dst, mask);
        return false;
    }

    key = {};
    key.device = device;
    key.b = 1;
    key.h = h;
    key.sq = sq;
    key.sk = sk;
    key.sk_actual = sk_actual;
    key.d = d;
    key.attn_scale = attn_scale;
    key.padding_mask = padding_mask;

    // cuDNN logical dim is [B,H,S,D]. The current ggml attention inputs are contiguous [D,S,H].
    key.q_stride[0] = h * sq * d;
    key.q_stride[1] = sq * d;
    key.q_stride[2] = d;
    key.q_stride[3] = 1;
    key.k_stride[0] = h * sk * d;
    key.k_stride[1] = sk * d;
    key.k_stride[2] = d;
    key.k_stride[3] = 1;
    key.v_stride[0] = h * sk * d;
    key.v_stride[1] = sk * d;
    key.v_stride[2] = d;
    key.v_stride[3] = 1;
    key.o_stride[0] = h * sq * d;
    key.o_stride[1] = sq * d;
    key.o_stride[2] = d;
    key.o_stride[3] = 1;
    return true;
}

static bool should_use_cudnn_sdpa(const ed_cudnn_sdpa_key & key) {
    const bool supported_head_dim = key.d == 64 || key.d == 128;
    const bool supported_sequence = key.sq >= MIN_CUDNN_SDPA_FAST_SEQ &&
                                    ((key.sq == key.sk && !key.padding_mask) ||
                                     (key.padding_mask && key.sk_actual == key.sq && key.sk >= key.sq));
    const bool use_cudnn = supported_head_dim && supported_sequence;

    if (!use_cudnn && profile_enabled()) {
        GGML_LOG_INFO("ED_CUDNN_SDPA unsupported: gate d=%lld sq=%lld sk=%lld sk_actual=%lld scale=%g padding_mask=%d supported_head_dim=%d supported_sequence=%d\n",
                      (long long) key.d,
                      (long long) key.sq,
                      (long long) key.sk,
                      (long long) key.sk_actual,
                      key.attn_scale,
                      key.padding_mask ? 1 : 0,
                      supported_head_dim ? 1 : 0,
                      supported_sequence ? 1 : 0);
    }

    return use_cudnn;
}

static std::shared_ptr<fe::graph::Graph> create_graph(const ed_cudnn_sdpa_key & key) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const std::vector<int64_t> q_dim = {key.b, key.h, key.sq, key.d};
    const std::vector<int64_t> kv_dim = {key.b, key.h, key.sk, key.d};

    auto q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim(q_dim)
                               .set_stride({key.q_stride[0], key.q_stride[1], key.q_stride[2], key.q_stride[3]}));
    auto k = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim(kv_dim)
                               .set_stride({key.k_stride[0], key.k_stride[1], key.k_stride[2], key.k_stride[3]}));
    auto v = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim(kv_dim)
                               .set_stride({key.v_stride[0], key.v_stride[1], key.v_stride[2], key.v_stride[3]}));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("edge_dit_sdpa")
                            .set_generate_stats(false)
                            .set_attn_scale(key.attn_scale);

    if (key.padding_mask) {
        auto seq_q = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_q")
                                       .set_uid(SEQ_LEN_Q_UID)
                                       .set_dim({key.b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
        auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_kv")
                                        .set_uid(SEQ_LEN_KV_UID)
                                        .set_dim({key.b, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));
        sdpa_options.set_padding_mask(true).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    }

    auto [o, stats] = graph->sdpa(q, k, v, sdpa_options);
    (void) stats;
    o->set_output(true)
        .set_uid(O_UID)
        .set_dim(q_dim)
        .set_stride({key.o_stride[0], key.o_stride[1], key.o_stride[2], key.o_stride[3]});

    return graph;
}

static std::unique_ptr<ed_cudnn_sdpa_plan> create_plan(const ed_cudnn_sdpa_key & key, ed_cudnn_sdpa_result_t & result) {
    auto plan = std::make_unique<ed_cudnn_sdpa_plan>();
    if (cudnnCreate(&plan->handle) != CUDNN_STATUS_SUCCESS) {
        result = ED_CUDNN_SDPA_BUILD_FAILED;
        return nullptr;
    }
    if (!ensure_nvrtc_loaded()) {
        result = ED_CUDNN_SDPA_BUILD_FAILED;
        return nullptr;
    }
    plan->graph = create_graph(key);
    auto status = plan->graph->build(plan->handle, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        if (profile_enabled()) {
            GGML_LOG_WARN("ED_CUDNN_SDPA build failed: %s\n", status.get_message().c_str());
        }
        result = ED_CUDNN_SDPA_BUILD_FAILED;
        return nullptr;
    }

    auto workspace_status = plan->graph->get_workspace_size(plan->workspace_size);
    if (!workspace_status.is_good()) {
        result = ED_CUDNN_SDPA_BUILD_FAILED;
        return nullptr;
    }
    if (plan->workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&plan->workspace, (size_t) plan->workspace_size));
    }
    plan->elements = key.b * key.h * key.sq * key.d;
    CUDA_CHECK(cudaMalloc(&plan->q_f16, (size_t) plan->elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&plan->o_f16, (size_t) plan->elements * sizeof(half)));
    if (key.padding_mask) {
        const int32_t seq_q = (int32_t) key.sq;
        const int32_t seq_kv = (int32_t) key.sk_actual;
        CUDA_CHECK(cudaMalloc(&plan->seq_len_q, sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&plan->seq_len_kv, sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(plan->seq_len_q, &seq_q, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(plan->seq_len_kv, &seq_kv, sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    result = ED_CUDNN_SDPA_SUCCESS;
    return plan;
}

static void build_plan_async(ed_cudnn_sdpa_key key, std::shared_ptr<ed_cudnn_sdpa_plan_entry> entry) {
    std::thread([key, entry]() {
        int old_device = 0;
        const bool have_old_device = cudaGetDevice(&old_device) == cudaSuccess;
        cudaSetDevice(key.device);

        ed_cudnn_sdpa_result_t result = ED_CUDNN_SDPA_SUCCESS;
        const double start_ms = profile_enabled() ? now_ms() : 0.0;
        std::unique_ptr<ed_cudnn_sdpa_plan> plan = create_plan(key, result);
        const double build_ms = profile_enabled() ? now_ms() - start_ms : 0.0;

        if (have_old_device) {
            cudaSetDevice(old_device);
        }

        {
            std::lock_guard<std::mutex> lock(entry->mutex);
            if (plan != nullptr) {
                entry->plan = std::move(plan);
                entry->state = ed_cudnn_sdpa_plan_state::READY;
                entry->result = ED_CUDNN_SDPA_SUCCESS;
            } else {
                entry->state = ed_cudnn_sdpa_plan_state::FAILED;
                entry->result = result == ED_CUDNN_SDPA_SUCCESS ? ED_CUDNN_SDPA_BUILD_FAILED : result;
            }
        }

        if (profile_enabled()) {
            GGML_LOG_INFO("ED_CUDNN_SDPA async build b=%lld h=%lld sq=%lld sk=%lld d=%lld scale=%g result=%s build=%.3fms\n",
                          (long long) key.b,
                          (long long) key.h,
                          (long long) key.sq,
                          (long long) key.sk,
                          (long long) key.d,
                          key.attn_scale,
                          ed_cudnn_sdpa_result_name(result),
                          build_ms);
        }
    }).detach();
}

static ed_cudnn_sdpa_plan * get_plan(const ed_cudnn_sdpa_key & key, cudaStream_t stream, ed_cudnn_sdpa_result_t & result) {
    std::shared_ptr<ed_cudnn_sdpa_plan_entry> entry;
    bool start_build = false;
    {
        std::lock_guard<std::mutex> lock(g_plan_mutex);
        auto it = g_plan_cache.find(key);
        if (it == g_plan_cache.end()) {
            entry = std::make_shared<ed_cudnn_sdpa_plan_entry>();
            g_plan_cache.emplace(key, entry);
            start_build = true;
        } else {
            entry = it->second;
        }
    }

    if (start_build) {
        build_plan_async(key, entry);
        result = ED_CUDNN_SDPA_BUILD_PENDING;
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(entry->mutex);
    if (entry->state == ed_cudnn_sdpa_plan_state::READY && entry->plan != nullptr) {
        cudnnSetStream(entry->plan->handle, stream);
        result = ED_CUDNN_SDPA_SUCCESS;
        return entry->plan.get();
    }
    result = entry->state == ed_cudnn_sdpa_plan_state::FAILED ? entry->result : ED_CUDNN_SDPA_BUILD_PENDING;
    return nullptr;
}

__global__ void f32_to_f16_kernel(const float * src, half * dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void f32_to_f16_vec2_kernel(const float * src, half * dst, int64_t n) {
    const int64_t i2 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n2 = n / 2;

    if (i2 < n2) {
        const float2 v = ((const float2 *) src)[i2];
        ((half2 *) dst)[i2] = __float22half2_rn(v);
    }

    if ((n & 1) && i2 == n2) {
        dst[n - 1] = __float2half(src[n - 1]);
    }
}

__global__ void f16_bhsd_to_f32_dst_kernel(const half * src, float * dst, int64_t d, int64_t sq, int64_t h) {
    const int64_t n = d * sq * h;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t dd = i % d;
    const int64_t ss = (i / d) % sq;
    const int64_t hh = i / (d * sq);
    // cuDNN output is contiguous BHSD: [H,S,D]. ggml dst is [D,H,S].
    dst[dd + hh * d + ss * d * h] = __half2float(src[i]);
}

__global__ void f16_bhsd_to_f32_dst_vec2_kernel(const half * src, float * dst, int64_t d, int64_t sq, int64_t h) {
    const int64_t d2 = d / 2;
    const int64_t n2 = d2 * sq * h;

    const int64_t i2 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 >= n2) {
        return;
    }

    const int64_t dd2 = i2 % d2;
    const int64_t row = i2 / d2;
    const int64_t ss  = row % sq;
    const int64_t hh  = row / sq;

    const half2 v = ((const half2 *) (src + (hh * sq + ss) * d))[dd2];
    ((float2 *) (dst + ss * d * h + hh * d))[dd2] = __half22float2(v);
}

static bool is_aligned_to(const void * ptr, uintptr_t alignment) {
    return ((uintptr_t) ptr % alignment) == 0;
}

static bool can_use_vec2_convert(const void * src, const void * dst) {
    return !cudnn_sdpa_vec_convert_disabled() &&
           is_aligned_to(src, alignof(float2)) &&
           is_aligned_to(dst, alignof(half2));
}

static bool can_use_vec2_output_convert(const void * src, const void * dst, int64_t d) {
    return !cudnn_sdpa_vec_convert_disabled() &&
           (d % 2) == 0 &&
           is_aligned_to(src, alignof(half2)) &&
           is_aligned_to(dst, alignof(float2));
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
}

static void profile_record(const ed_cudnn_sdpa_key & key,
                           double get_plan_ms,
                           double q_cast_ms,
                           double execute_ms,
                           double o_cast_ms) {
    if (!profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    ed_cudnn_sdpa_profile_stat & stat = g_profile_stats[key];
    stat.calls++;
    stat.get_plan_ms += get_plan_ms;
    stat.q_cast_ms += q_cast_ms;
    stat.execute_ms += execute_ms;
    stat.o_cast_ms += o_cast_ms;
    if ((stat.calls & (stat.calls - 1)) == 0) {
        GGML_LOG_INFO("ED_CUDNN_SDPA profile b=%lld h=%lld sq=%lld sk=%lld d=%lld scale=%g padding_mask=%d calls=%" PRIu64
                      " get_plan=%.3fms q_cast=%.3fms execute=%.3fms o_cast=%.3fms\n",
                      (long long) key.b,
                      (long long) key.h,
                      (long long) key.sq,
                      (long long) key.sk,
                      (long long) key.d,
                      key.attn_scale,
                      key.padding_mask ? 1 : 0,
                      stat.calls,
                      stat.get_plan_ms,
                      stat.q_cast_ms,
                      stat.execute_ms,
                      stat.o_cast_ms);
    }
}

} // namespace

bool ed_cudnn_sdpa_supported(const ggml_tensor * dst, int device) {
    if (cudnn_sdpa_disabled()) {
        return false;
    }

    ed_cudnn_sdpa_key key;
    return make_key(dst, key, device) && should_use_cudnn_sdpa(key);
}

ed_cudnn_sdpa_result_t ed_cudnn_sdpa_compute(ggml_tensor * dst, ed_cudnn_sdpa_stream_t stream_ptr) {
    if (cudnn_sdpa_disabled()) {
        return ED_CUDNN_SDPA_DISABLED;
    }

    ed_cudnn_sdpa_key key;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (!make_key(dst, key, device)) {
        return ED_CUDNN_SDPA_UNSUPPORTED;
    }
    if (!should_use_cudnn_sdpa(key)) {
        return ED_CUDNN_SDPA_UNSUPPORTED;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    ed_cudnn_sdpa_result_t result = ED_CUDNN_SDPA_SUCCESS;
    const bool do_profile = profile_enabled();
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    double get_plan_ms = 0.0;
    double q_cast_ms = 0.0;
    double execute_ms = 0.0;
    double o_cast_ms = 0.0;
    const double get_plan_start_ms = do_profile ? now_ms() : 0.0;
    ed_cudnn_sdpa_plan * plan = get_plan(key, stream, result);
    if (do_profile) {
        cudaEventCreate(&profile_start);
        cudaEventCreate(&profile_stop);
        get_plan_ms = now_ms() - get_plan_start_ms;
    }
    if (plan == nullptr) {
        if (do_profile) {
            cudaEventDestroy(profile_start);
            cudaEventDestroy(profile_stop);
        }
        return result;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const int64_t n = key.b * key.h * key.sq * key.d;

    const int threads = 256;
    const int blocks = (int) ((n + threads - 1) / threads);
    const void * q_data = q->data;
    if (q->type == GGML_TYPE_F32) {
        if (do_profile) {
            cudaEventRecord(profile_start, stream);
        }
        if (can_use_vec2_convert(q->data, plan->q_f16)) {
            const int blocks_vec = (int) (((n + 1) / 2 + threads - 1) / threads);
            f32_to_f16_vec2_kernel<<<blocks_vec, threads, 0, stream>>>((const float *) q->data, plan->q_f16, n);
        } else {
            f32_to_f16_kernel<<<blocks, threads, 0, stream>>>((const float *) q->data, plan->q_f16, n);
        }
        CUDA_CHECK(cudaGetLastError());
        q_data = plan->q_f16;
        if (do_profile) {
            cudaEventRecord(profile_stop, stream);
            cudaEventSynchronize(profile_stop);
            q_cast_ms = elapsed_ms(profile_start, profile_stop);
        }
    } else {
        GGML_ASSERT(q->type == GGML_TYPE_F16);
    }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack = {
        {Q_UID, const_cast<void *>(q_data)},
        {K_UID, k->data},
        {V_UID, v->data},
        {O_UID, plan->o_f16},
    };
    if (key.padding_mask) {
        variant_pack[SEQ_LEN_Q_UID] = plan->seq_len_q;
        variant_pack[SEQ_LEN_KV_UID] = plan->seq_len_kv;
    }

    if (do_profile) {
        cudaEventRecord(profile_start, stream);
    }
    auto status = plan->graph->execute(plan->handle, variant_pack, plan->workspace);
    if (do_profile) {
        cudaEventRecord(profile_stop, stream);
        cudaEventSynchronize(profile_stop);
        execute_ms = elapsed_ms(profile_start, profile_stop);
    }
    if (!status.is_good()) {
        if (profile_enabled()) {
            GGML_LOG_WARN("ED_CUDNN_SDPA execute failed: %s\n", status.get_message().c_str());
        }
        if (do_profile) {
            cudaEventDestroy(profile_start);
            cudaEventDestroy(profile_stop);
        }
        return ED_CUDNN_SDPA_EXECUTE_FAILED;
    }

    if (do_profile) {
        cudaEventRecord(profile_start, stream);
    }
    if (can_use_vec2_output_convert(plan->o_f16, dst->data, key.d)) {
        const int blocks_vec = (int) (((n / 2) + threads - 1) / threads);
        f16_bhsd_to_f32_dst_vec2_kernel<<<blocks_vec, threads, 0, stream>>>(plan->o_f16, (float *) dst->data, key.d, key.sq, key.h);
    } else {
        f16_bhsd_to_f32_dst_kernel<<<blocks, threads, 0, stream>>>(plan->o_f16, (float *) dst->data, key.d, key.sq, key.h);
    }
    CUDA_CHECK(cudaGetLastError());
    if (do_profile) {
        cudaEventRecord(profile_stop, stream);
        cudaEventSynchronize(profile_stop);
        o_cast_ms = elapsed_ms(profile_start, profile_stop);
    }

    if (profile_enabled()) {
        profile_record(key, get_plan_ms, q_cast_ms, execute_ms, o_cast_ms);
        GGML_LOG_INFO("ED_CUDNN_SDPA success b=%lld h=%lld sq=%lld sk=%lld sk_actual=%lld d=%lld scale=%g padding_mask=%d workspace=%lld\n",
                      (long long) key.b,
                      (long long) key.h,
                      (long long) key.sq,
                      (long long) key.sk,
                      (long long) key.sk_actual,
                      (long long) key.d,
                      key.attn_scale,
                      key.padding_mask ? 1 : 0,
                      (long long) plan->workspace_size);
    }

    if (do_profile) {
        cudaEventDestroy(profile_start);
        cudaEventDestroy(profile_stop);
    }
    return ED_CUDNN_SDPA_SUCCESS;
}

bool ed_cudnn_sdpa_allows_fallback() {
    const char * env = getenv("ED_CUDNN_SDPA_NO_FALLBACK");
    return env == nullptr || atoi(env) == 0;
}

const char * ed_cudnn_sdpa_result_name(ed_cudnn_sdpa_result_t result) {
    switch (result) {
        case ED_CUDNN_SDPA_SUCCESS:
            return "success";
        case ED_CUDNN_SDPA_DISABLED:
            return "disabled";
        case ED_CUDNN_SDPA_UNSUPPORTED:
            return "unsupported";
        case ED_CUDNN_SDPA_BUILD_PENDING:
            return "build_pending";
        case ED_CUDNN_SDPA_BUILD_FAILED:
            return "build_failed";
        case ED_CUDNN_SDPA_EXECUTE_FAILED:
            return "execute_failed";
    }
    return "unknown";
}
