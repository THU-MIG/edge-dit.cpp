#include "ed_cudnn_conv3d.h"

#include "common.cuh"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {

constexpr size_t ED_CUDNN_CONV3D_MAX_WORKSPACE = 1152ull * 1024ull * 1024ull;

static bool uses_f16_io_cast(ggml_type x_type, ggml_type w_type, ggml_type y_type) {
    return x_type == GGML_TYPE_F32 && w_type == GGML_TYPE_F16 && y_type == GGML_TYPE_F32;
}

struct ed_cudnn_conv3d_key {
    int device = 0;
    ggml_type x_type = GGML_TYPE_COUNT;
    ggml_type w_type = GGML_TYPE_COUNT;
    ggml_type y_type = GGML_TYPE_COUNT;
    int64_t n = 0;
    int64_t ic = 0;
    int64_t id = 0;
    int64_t ih = 0;
    int64_t iw = 0;
    int64_t oc = 0;
    int64_t kd = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    int stride_x = 1;
    int stride_y = 1;
    int stride_z = 1;
    int pad_x = 0;
    int pad_y = 0;
    int pad_z = 0;
    int dilation_x = 1;
    int dilation_y = 1;
    int dilation_z = 1;

    bool operator==(const ed_cudnn_conv3d_key & other) const {
        return device == other.device &&
               x_type == other.x_type &&
               w_type == other.w_type &&
               y_type == other.y_type &&
               n == other.n &&
               ic == other.ic &&
               id == other.id &&
               ih == other.ih &&
               iw == other.iw &&
               oc == other.oc &&
               kd == other.kd &&
               kh == other.kh &&
               kw == other.kw &&
               od == other.od &&
               oh == other.oh &&
               ow == other.ow &&
               stride_x == other.stride_x &&
               stride_y == other.stride_y &&
               stride_z == other.stride_z &&
               pad_x == other.pad_x &&
               pad_y == other.pad_y &&
               pad_z == other.pad_z &&
               dilation_x == other.dilation_x &&
               dilation_y == other.dilation_y &&
               dilation_z == other.dilation_z;
    }
};

struct ed_cudnn_conv3d_key_hash {
    size_t operator()(const ed_cudnn_conv3d_key & key) const {
        size_t h = std::hash<int>()(key.device);
        auto mix = [&h](int64_t v) {
            h ^= std::hash<int64_t>()(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix((int64_t) key.x_type);
        mix((int64_t) key.w_type);
        mix((int64_t) key.y_type);
        mix(key.n);
        mix(key.ic);
        mix(key.id);
        mix(key.ih);
        mix(key.iw);
        mix(key.oc);
        mix(key.kd);
        mix(key.kh);
        mix(key.kw);
        mix(key.od);
        mix(key.oh);
        mix(key.ow);
        mix(key.stride_x);
        mix(key.stride_y);
        mix(key.stride_z);
        mix(key.pad_x);
        mix(key.pad_y);
        mix(key.pad_z);
        mix(key.dilation_x);
        mix(key.dilation_y);
        mix(key.dilation_z);
        return h;
    }
};

struct ed_cudnn_conv3d_plan {
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspace_size = 0;

    ~ed_cudnn_conv3d_plan() {
        if (y_desc != nullptr) {
            cudnnDestroyTensorDescriptor(y_desc);
        }
        if (conv_desc != nullptr) {
            cudnnDestroyConvolutionDescriptor(conv_desc);
        }
        if (w_desc != nullptr) {
            cudnnDestroyFilterDescriptor(w_desc);
        }
        if (x_desc != nullptr) {
            cudnnDestroyTensorDescriptor(x_desc);
        }
    }
};

struct ed_cudnn_conv3d_handle {
    int device = 0;
    cudnnHandle_t handle = nullptr;

    ~ed_cudnn_conv3d_handle() {
        if (handle != nullptr) {
            int old_device = 0;
            cudaGetDevice(&old_device);
            cudaSetDevice(device);
            cudnnDestroy(handle);
            cudaSetDevice(old_device);
        }
    }
};

struct ed_cudnn_conv3d_workspace_key {
    int device = 0;
    cudaStream_t stream = nullptr;

    bool operator==(const ed_cudnn_conv3d_workspace_key & other) const {
        return device == other.device && stream == other.stream;
    }
};

struct ed_cudnn_conv3d_workspace_key_hash {
    size_t operator()(const ed_cudnn_conv3d_workspace_key & key) const {
        size_t h = std::hash<int>()(key.device);
        h ^= std::hash<uintptr_t>()((uintptr_t) key.stream) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct ed_cudnn_conv3d_workspace {
    int device = 0;
    void * ptr = nullptr;
    size_t size = 0;

    ~ed_cudnn_conv3d_workspace() {
        if (ptr != nullptr) {
            int old_device = 0;
            cudaGetDevice(&old_device);
            cudaSetDevice(device);
            cudaFree(ptr);
            cudaSetDevice(old_device);
        }
    }
};

struct ed_cudnn_conv3d_profile_stat {
    uint64_t calls = 0;
    double get_plan_ms = 0.0;
    double workspace_ms = 0.0;
    double prepare_ms = 0.0;
    double execute_ms = 0.0;
    double finalize_ms = 0.0;
};

static std::mutex g_plan_mutex;
static std::unordered_map<ed_cudnn_conv3d_key, std::unique_ptr<ed_cudnn_conv3d_plan>, ed_cudnn_conv3d_key_hash> g_plan_cache;

static std::mutex g_handle_mutex;
static std::unordered_map<ed_cudnn_conv3d_workspace_key,
                          std::unique_ptr<ed_cudnn_conv3d_handle>,
                          ed_cudnn_conv3d_workspace_key_hash>
    g_handle_cache;

static std::mutex g_workspace_mutex;
static std::unordered_map<ed_cudnn_conv3d_workspace_key,
                          std::unique_ptr<ed_cudnn_conv3d_workspace>,
                          ed_cudnn_conv3d_workspace_key_hash>
    g_workspace_cache;

static std::mutex g_scratch_mutex;
static std::unordered_map<ed_cudnn_conv3d_workspace_key,
                          std::unique_ptr<ed_cudnn_conv3d_workspace>,
                          ed_cudnn_conv3d_workspace_key_hash>
    g_scratch_cache;

static std::mutex g_profile_mutex;
static std::unordered_map<ed_cudnn_conv3d_key, ed_cudnn_conv3d_profile_stat, ed_cudnn_conv3d_key_hash> g_profile_stats;

static bool profile_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("ED_PROFILE_CUDNN_CONV3D");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

static cudnnDataType_t ggml_type_to_cudnn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return CUDNN_DATA_FLOAT;
        case GGML_TYPE_F16:
            return CUDNN_DATA_HALF;
        case GGML_TYPE_BF16:
            return CUDNN_DATA_BFLOAT16;
        default:
            return CUDNN_DATA_FLOAT;
    }
}

static bool is_supported_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static bool is_supported_dtype_combination(ggml_type x_type, ggml_type w_type, ggml_type y_type) {
    return (x_type == w_type && w_type == y_type) || uses_f16_io_cast(x_type, w_type, y_type);
}

static ggml_type plan_x_type(const ed_cudnn_conv3d_key & key) {
    return uses_f16_io_cast(key.x_type, key.w_type, key.y_type) ? GGML_TYPE_F16 : key.x_type;
}

static ggml_type plan_w_type(const ed_cudnn_conv3d_key & key) {
    return uses_f16_io_cast(key.x_type, key.w_type, key.y_type) ? GGML_TYPE_F16 : key.w_type;
}

static ggml_type plan_y_type(const ed_cudnn_conv3d_key & key) {
    return uses_f16_io_cast(key.x_type, key.w_type, key.y_type) ? GGML_TYPE_F16 : key.y_type;
}

static size_t type_size(ggml_type type) {
    return ggml_type_size(type) / ggml_blck_size(type);
}

static bool is_contiguous_4d(const ggml_tensor * t) {
    const size_t ts = type_size(t->type);
    size_t next_nb = ts;
    if (t->ne[0] != 1 && t->nb[0] != next_nb) {
        return false;
    }
    next_nb *= (size_t) t->ne[0];
    for (int i = 1; i < 4; ++i) {
        if (t->ne[i] != 1 && t->nb[i] != next_nb) {
            return false;
        }
        next_nb *= (size_t) t->ne[i];
    }
    return true;
}

static bool read_key(const ggml_tensor * dst, int device, ed_cudnn_conv3d_key & key) {
    if (device < 0) {
        return false;
    }
    if (dst == nullptr || dst->op != GGML_OP_CONV_3D || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    if (!is_supported_type(x->type) || !is_supported_type(w->type) || !is_supported_type(dst->type)) {
        return false;
    }
    if (!is_supported_dtype_combination(x->type, w->type, dst->type)) {
        return false;
    }
    if (!is_contiguous_4d(x) || !is_contiguous_4d(w) || !is_contiguous_4d(dst)) {
        return false;
    }
    const ggml_tensor * bias = dst->src[2];
    if (bias != nullptr) {
        if (bias->type != GGML_TYPE_F32 || !is_contiguous_4d(bias)) {
            return false;
        }
    }

    const int32_t * p = (const int32_t *) dst->op_params;
    const int64_t ic = p[9];
    const int64_t n = p[10];
    const int64_t oc = p[11];
    if (ic <= 0 || n <= 0 || oc <= 0) {
        return false;
    }
    if (w->ne[3] != ic * oc || x->ne[3] != ic * n || dst->ne[3] != oc * n) {
        return false;
    }
    if (bias != nullptr && ggml_nelements(bias) != oc) {
        return false;
    }
    if (x->ne[0] <= 0 || x->ne[1] <= 0 || x->ne[2] <= 0 ||
        w->ne[0] <= 0 || w->ne[1] <= 0 || w->ne[2] <= 0 ||
        dst->ne[0] <= 0 || dst->ne[1] <= 0 || dst->ne[2] <= 0) {
        return false;
    }
    if (x->ne[0] > INT32_MAX || x->ne[1] > INT32_MAX || x->ne[2] > INT32_MAX || x->ne[3] > INT32_MAX ||
        w->ne[0] > INT32_MAX || w->ne[1] > INT32_MAX || w->ne[2] > INT32_MAX || w->ne[3] > INT32_MAX ||
        dst->ne[0] > INT32_MAX || dst->ne[1] > INT32_MAX || dst->ne[2] > INT32_MAX || dst->ne[3] > INT32_MAX) {
        return false;
    }

    key.x_type = x->type;
    key.w_type = w->type;
    key.y_type = dst->type;
    key.n = n;
    key.ic = ic;
    key.id = x->ne[2];
    key.ih = x->ne[1];
    key.iw = x->ne[0];
    key.oc = oc;
    key.kd = w->ne[2];
    key.kh = w->ne[1];
    key.kw = w->ne[0];
    key.od = dst->ne[2];
    key.oh = dst->ne[1];
    key.ow = dst->ne[0];
    key.stride_x = p[0];
    key.stride_y = p[1];
    key.stride_z = p[2];
    key.pad_x = p[3];
    key.pad_y = p[4];
    key.pad_z = p[5];
    key.dilation_x = p[6];
    key.dilation_y = p[7];
    key.dilation_z = p[8];
    if (key.stride_x <= 0 || key.stride_y <= 0 || key.stride_z <= 0 ||
        key.dilation_x <= 0 || key.dilation_y <= 0 || key.dilation_z <= 0 ||
        key.pad_x < 0 || key.pad_y < 0 || key.pad_z < 0) {
        return false;
    }

    key.device = device;
    return true;
}

static void log_unsupported_tensor(const ggml_tensor * dst, const char * reason) {
    if (!profile_enabled() || dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return;
    }
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int32_t * p = (const int32_t *) dst->op_params;
    GGML_LOG_INFO("ED_CUDNN_CONV3D unsupported reason=%s x=(type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu])"
                  " w=(type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu])"
                  " dst=(type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu])"
                  " params=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n",
                  reason,
                  ggml_type_name(x->type),
                  x->ne[0], x->ne[1], x->ne[2], x->ne[3],
                  x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                  ggml_type_name(w->type),
                  w->ne[0], w->ne[1], w->ne[2], w->ne[3],
                  w->nb[0], w->nb[1], w->nb[2], w->nb[3],
                  ggml_type_name(dst->type),
                  dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                  dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                  p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]);
}

static bool set_tensor_5d(cudnnTensorDescriptor_t desc,
                          cudnnDataType_t type,
                          int n,
                          int c,
                          int d,
                          int h,
                          int w) {
    const int dims[5] = { n, c, d, h, w };
    const int strides[5] = { c * d * h * w, d * h * w, h * w, w, 1 };
    return cudnnSetTensorNdDescriptor(desc, type, 5, dims, strides) == CUDNN_STATUS_SUCCESS;
}

static bool set_filter_5d(cudnnFilterDescriptor_t desc,
                          cudnnDataType_t type,
                          int oc,
                          int ic,
                          int kd,
                          int kh,
                          int kw) {
    const int dims[5] = { oc, ic, kd, kh, kw };
    return cudnnSetFilterNdDescriptor(desc, type, CUDNN_TENSOR_NCHW, 5, dims) == CUDNN_STATUS_SUCCESS;
}

static cudnnHandle_t get_handle(int device, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    ed_cudnn_conv3d_workspace_key key;
    key.device = device;
    key.stream = stream;

    auto it = g_handle_cache.find(key);
    if (it == g_handle_cache.end()) {
        auto handle = std::make_unique<ed_cudnn_conv3d_handle>();
        handle->device = device;

        int old_device = 0;
        cudaGetDevice(&old_device);
        cudaSetDevice(device);
        if (cudnnCreate(&handle->handle) != CUDNN_STATUS_SUCCESS) {
            cudaSetDevice(old_device);
            return nullptr;
        }
        cudaSetDevice(old_device);

        it = g_handle_cache.emplace(key, std::move(handle)).first;
    }

    cudnnHandle_t handle = it->second->handle;
    if (cudnnSetStream(handle, stream) != CUDNN_STATUS_SUCCESS) {
        return nullptr;
    }
    return handle;
}

static bool algo_workspace_ok(cudnnHandle_t handle,
                              ed_cudnn_conv3d_plan * plan,
                              cudnnConvolutionFwdAlgo_t algo,
                              size_t * workspace_size_out) {
    size_t workspace_size = 0;
    const cudnnStatus_t status = cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                                          plan->x_desc,
                                                                          plan->w_desc,
                                                                          plan->conv_desc,
                                                                          plan->y_desc,
                                                                          algo,
                                                                          &workspace_size);
    if (status != CUDNN_STATUS_SUCCESS || workspace_size > ED_CUDNN_CONV3D_MAX_WORKSPACE) {
        return false;
    }
    *workspace_size_out = workspace_size;
    return true;
}

static bool select_forward_algo(cudnnHandle_t handle, ed_cudnn_conv3d_plan * plan) {
    const cudnnConvolutionFwdAlgo_t candidates[] = {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    };
    for (cudnnConvolutionFwdAlgo_t algo : candidates) {
        size_t workspace_size = 0;
        if (algo_workspace_ok(handle, plan, algo, &workspace_size)) {
            plan->algo = algo;
            plan->workspace_size = workspace_size;
            return true;
        }
    }

    return false;
}

static ed_cudnn_conv3d_plan * create_plan(const ed_cudnn_conv3d_key & key,
                                          cudnnHandle_t handle,
                                          ed_cudnn_conv3d_result_t & result) {
    auto plan = std::make_unique<ed_cudnn_conv3d_plan>();
    if (cudnnCreateTensorDescriptor(&plan->x_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateFilterDescriptor(&plan->w_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateConvolutionDescriptor(&plan->conv_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateTensorDescriptor(&plan->y_desc) != CUDNN_STATUS_SUCCESS) {
        result = ED_CUDNN_CONV3D_BUILD_FAILED;
        return nullptr;
    }

    const cudnnDataType_t x_type = ggml_type_to_cudnn(plan_x_type(key));
    const cudnnDataType_t w_type = ggml_type_to_cudnn(plan_w_type(key));
    const cudnnDataType_t y_type = ggml_type_to_cudnn(plan_y_type(key));
    const cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;

    if (!set_tensor_5d(plan->x_desc, x_type, (int) key.n, (int) key.ic, (int) key.id, (int) key.ih, (int) key.iw) ||
        !set_filter_5d(plan->w_desc, w_type, (int) key.oc, (int) key.ic, (int) key.kd, (int) key.kh, (int) key.kw) ||
        !set_tensor_5d(plan->y_desc, y_type, (int) key.n, (int) key.oc, (int) key.od, (int) key.oh, (int) key.ow)) {
        result = ED_CUDNN_CONV3D_BUILD_FAILED;
        return nullptr;
    }

    const int pad[3] = { key.pad_z, key.pad_y, key.pad_x };
    const int stride[3] = { key.stride_z, key.stride_y, key.stride_x };
    const int dilation[3] = { key.dilation_z, key.dilation_y, key.dilation_x };
    if (cudnnSetConvolutionNdDescriptor(plan->conv_desc,
                                        3,
                                        pad,
                                        stride,
                                        dilation,
                                        CUDNN_CROSS_CORRELATION,
                                        compute_type) != CUDNN_STATUS_SUCCESS) {
        result = ED_CUDNN_CONV3D_BUILD_FAILED;
        return nullptr;
    }
    cudnnSetConvolutionMathType(plan->conv_desc, CUDNN_TENSOR_OP_MATH);

    if (!select_forward_algo(handle, plan.get())) {
        result = ED_CUDNN_CONV3D_BUILD_FAILED;
        return nullptr;
    }
    if (profile_enabled()) {
        GGML_LOG_INFO("ED_CUDNN_CONV3D plan type=%s n=%" PRId64 " ic=%" PRId64 " id=%" PRId64 " ih=%" PRId64 " iw=%" PRId64
                      " oc=%" PRId64 " kd=%" PRId64 " kh=%" PRId64 " kw=%" PRId64 " od=%" PRId64 " oh=%" PRId64 " ow=%" PRId64
                      " stride=%d,%d,%d pad=%d,%d,%d dilation=%d,%d,%d algo=%d workspace=%zu mode=%s\n",
                      ggml_type_name(key.x_type),
                      key.n,
                      key.ic,
                      key.id,
                      key.ih,
                      key.iw,
                      key.oc,
                      key.kd,
                      key.kh,
                      key.kw,
                      key.od,
                      key.oh,
                      key.ow,
                      key.stride_x,
                      key.stride_y,
                      key.stride_z,
                      key.pad_x,
                      key.pad_y,
                      key.pad_z,
                      key.dilation_x,
                      key.dilation_y,
                      key.dilation_z,
                      (int) plan->algo,
                      plan->workspace_size,
                      uses_f16_io_cast(key.x_type, key.w_type, key.y_type) ? "f16_io_cast" : "direct");
    }

    ed_cudnn_conv3d_plan * raw = plan.get();
    g_plan_cache.emplace(key, std::move(plan));
    result = ED_CUDNN_CONV3D_SUCCESS;
    return raw;
}

static ed_cudnn_conv3d_plan * get_plan(const ed_cudnn_conv3d_key & key,
                                       cudnnHandle_t handle,
                                       ed_cudnn_conv3d_result_t & result) {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end()) {
        result = ED_CUDNN_CONV3D_SUCCESS;
        return it->second.get();
    }
    return create_plan(key, handle, result);
}

static void * get_workspace(int device, cudaStream_t stream, size_t workspace_size) {
    if (workspace_size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    ed_cudnn_conv3d_workspace_key key;
    key.device = device;
    key.stream = stream;

    auto it = g_workspace_cache.find(key);
    if (it == g_workspace_cache.end()) {
        auto workspace = std::make_unique<ed_cudnn_conv3d_workspace>();
        workspace->device = device;
        it = g_workspace_cache.emplace(key, std::move(workspace)).first;
    }

    ed_cudnn_conv3d_workspace * workspace = it->second.get();
    if (workspace->size < workspace_size) {
        int old_device = 0;
        cudaGetDevice(&old_device);
        cudaSetDevice(device);
        if (workspace->ptr != nullptr) {
            cudaFree(workspace->ptr);
            workspace->ptr = nullptr;
            workspace->size = 0;
        }
        if (cudaMalloc(&workspace->ptr, workspace_size) != cudaSuccess) {
            cudaSetDevice(old_device);
            workspace->ptr = nullptr;
            workspace->size = 0;
            return nullptr;
        }
        workspace->size = workspace_size;
        cudaSetDevice(old_device);
    }

    return workspace->ptr;
}

static void * get_scratch(int device, cudaStream_t stream, size_t scratch_size) {
    if (scratch_size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_scratch_mutex);
    ed_cudnn_conv3d_workspace_key key;
    key.device = device;
    key.stream = stream;

    auto it = g_scratch_cache.find(key);
    if (it == g_scratch_cache.end()) {
        auto scratch = std::make_unique<ed_cudnn_conv3d_workspace>();
        scratch->device = device;
        it = g_scratch_cache.emplace(key, std::move(scratch)).first;
    }

    ed_cudnn_conv3d_workspace * scratch = it->second.get();
    if (scratch->size < scratch_size) {
        int old_device = 0;
        cudaGetDevice(&old_device);
        cudaSetDevice(device);
        if (scratch->ptr != nullptr) {
            cudaFree(scratch->ptr);
            scratch->ptr = nullptr;
            scratch->size = 0;
        }
        if (cudaMalloc(&scratch->ptr, scratch_size) != cudaSuccess) {
            cudaSetDevice(old_device);
            scratch->ptr = nullptr;
            scratch->size = 0;
            return nullptr;
        }
        scratch->size = scratch_size;
        cudaSetDevice(old_device);
    }

    return scratch->ptr;
}

static __global__ void float_to_half_kernel(const float * __restrict__ src, half * __restrict__ dst, int64_t n) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

static __global__ void float_to_half2_kernel(const float * __restrict__ src, half * __restrict__ dst, int64_t n2) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        const float2 v = reinterpret_cast<const float2 *>(src)[idx];
        reinterpret_cast<half2 *>(dst)[idx] = __float22half2_rn(v);
    }
}

static __global__ void half_to_float_kernel(const half * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

static __global__ void half_to_float2_kernel(const half * __restrict__ src, float * __restrict__ dst, int64_t n2) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        const float2 v = __half22float2(reinterpret_cast<const half2 *>(src)[idx]);
        reinterpret_cast<float2 *>(dst)[idx] = v;
    }
}

static __global__ void half_to_float_bias_kernel(const half * __restrict__ src,
                                                 const float * __restrict__ bias,
                                                 float * __restrict__ dst,
                                                 int64_t ow,
                                                 int64_t oh,
                                                 int64_t od,
                                                 int64_t oc,
                                                 int64_t n) {
    const int64_t total = ow * oh * od * oc * n;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t plane = ow * oh * od;
    const int64_t oc_idx = (idx / plane) % oc;
    dst[idx] = __half2float(src[idx]) + bias[oc_idx];
}

static __global__ void half_to_float2_bias_kernel(const half * __restrict__ src,
                                                  const float * __restrict__ bias,
                                                  float * __restrict__ dst,
                                                  int64_t plane,
                                                  int64_t oc,
                                                  int64_t n2) {
    const int64_t idx2 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx2 >= n2) {
        return;
    }
    const int64_t idx = idx2 * 2;
    const int64_t oc_idx = (idx / plane) % oc;
    const float b = bias[oc_idx];
    float2 v = __half22float2(reinterpret_cast<const half2 *>(src)[idx2]);
    v.x += b;
    v.y += b;
    reinterpret_cast<float2 *>(dst)[idx2] = v;
}

static bool aligned_for_float2_to_half2(const void * src, const void * dst, int64_t n) {
    return (n & 1) == 0 &&
           (reinterpret_cast<uintptr_t>(src) % alignof(float2)) == 0 &&
           (reinterpret_cast<uintptr_t>(dst) % alignof(half2)) == 0;
}

static bool aligned_for_half2_to_float2(const void * src, const void * dst, int64_t n) {
    return (n & 1) == 0 &&
           (reinterpret_cast<uintptr_t>(src) % alignof(half2)) == 0 &&
           (reinterpret_cast<uintptr_t>(dst) % alignof(float2)) == 0;
}

static void float_to_half_async(const void * src, void * dst, int64_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    if (aligned_for_float2_to_half2(src, dst, n)) {
        const int64_t n2 = n / 2;
        const int64_t grid_size = (n2 + block_size - 1) / block_size;
        float_to_half2_kernel<<<grid_size, block_size, 0, stream>>>((const float *) src, (half *) dst, n2);
        return;
    }
    const int64_t grid_size = (n + block_size - 1) / block_size;
    float_to_half_kernel<<<grid_size, block_size, 0, stream>>>((const float *) src, (half *) dst, n);
}

static void half_to_float_async(const void * src, void * dst, int64_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    if (aligned_for_half2_to_float2(src, dst, n)) {
        const int64_t n2 = n / 2;
        const int64_t grid_size = (n2 + block_size - 1) / block_size;
        half_to_float2_kernel<<<grid_size, block_size, 0, stream>>>((const half *) src, (float *) dst, n2);
        return;
    }
    const int64_t grid_size = (n + block_size - 1) / block_size;
    half_to_float_kernel<<<grid_size, block_size, 0, stream>>>((const half *) src, (float *) dst, n);
}

static void half_to_float_bias_async(const void * src,
                                     const void * bias,
                                     void * dst,
                                     int64_t ow,
                                     int64_t oh,
                                     int64_t od,
                                     int64_t oc,
                                     int64_t n,
                                     cudaStream_t stream) {
    constexpr int block_size = 256;
    const int64_t total = ow * oh * od * oc * n;
    const int64_t plane = ow * oh * od;
    if ((plane & 1) == 0 && aligned_for_half2_to_float2(src, dst, total)) {
        const int64_t n2 = total / 2;
        const int64_t grid_size = (n2 + block_size - 1) / block_size;
        half_to_float2_bias_kernel<<<grid_size, block_size, 0, stream>>>(
            (const half *) src, (const float *) bias, (float *) dst, plane, oc, n2);
        return;
    }
    const int64_t grid_size = (total + block_size - 1) / block_size;
    half_to_float_bias_kernel<<<grid_size, block_size, 0, stream>>>(
        (const half *) src, (const float *) bias, (float *) dst, ow, oh, od, oc, n);
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
}

static void profile_record(const ed_cudnn_conv3d_key & key,
                           double get_plan_ms,
                           double workspace_ms,
                           double prepare_ms,
                           double finalize_ms,
                           double execute_ms) {
    if (!profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    ed_cudnn_conv3d_profile_stat & stat = g_profile_stats[key];
    stat.calls++;
    stat.get_plan_ms += get_plan_ms;
    stat.workspace_ms += workspace_ms;
    stat.prepare_ms += prepare_ms;
    stat.finalize_ms += finalize_ms;
    stat.execute_ms += execute_ms;
    if ((stat.calls & (stat.calls - 1)) == 0) {
        GGML_LOG_INFO("ED_CUDNN_CONV3D profile type=%s n=%" PRId64 " ic=%" PRId64 " id=%" PRId64 " ih=%" PRId64 " iw=%" PRId64
                      " oc=%" PRId64 " kd=%" PRId64 " kh=%" PRId64 " kw=%" PRId64 " od=%" PRId64 " oh=%" PRId64 " ow=%" PRId64
                      " calls=%" PRIu64 " get_plan=%.3fms workspace=%.3fms prepare=%.3fms execute=%.3fms finalize=%.3fms mode=%s\n",
                      ggml_type_name(key.x_type),
                      key.n,
                      key.ic,
                      key.id,
                      key.ih,
                      key.iw,
                      key.oc,
                      key.kd,
                      key.kh,
                      key.kw,
                      key.od,
                      key.oh,
                      key.ow,
                      stat.calls,
                      stat.get_plan_ms,
                      stat.workspace_ms,
                      stat.prepare_ms,
                      stat.execute_ms,
                      stat.finalize_ms,
                      uses_f16_io_cast(key.x_type, key.w_type, key.y_type) ? "f16_io_cast" : "direct");
    }
}

} // namespace

ed_cudnn_conv3d_result_t ed_cudnn_conv3d_compute(ggml_tensor * dst, ed_cudnn_conv3d_stream_t stream_ptr) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        log_unsupported_tensor(dst, "device");
        return ED_CUDNN_CONV3D_UNSUPPORTED;
    }

    ed_cudnn_conv3d_key key;
    if (!read_key(dst, device, key)) {
        log_unsupported_tensor(dst, "read_key");
        return ED_CUDNN_CONV3D_UNSUPPORTED;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool do_profile = profile_enabled();
    if (do_profile) {
        cudaEventCreate(&profile_start);
        cudaEventCreate(&profile_stop);
    }

    ed_cudnn_conv3d_result_t result = ED_CUDNN_CONV3D_SUCCESS;
    double get_plan_ms = 0.0;
    double workspace_ms = 0.0;
    double prepare_ms = 0.0;
    double execute_ms = 0.0;
    double finalize_ms = 0.0;
    if (do_profile) {
        cudaEventRecord(profile_start, stream);
    }
    cudnnHandle_t handle = get_handle(key.device, stream);
    ed_cudnn_conv3d_plan * plan = handle == nullptr ? nullptr : get_plan(key, handle, result);
    if (do_profile) {
        cudaEventRecord(profile_stop, stream);
        cudaEventSynchronize(profile_stop);
        get_plan_ms = elapsed_ms(profile_start, profile_stop);
    }
    if (handle == nullptr || plan == nullptr) {
        log_unsupported_tensor(dst, ed_cudnn_conv3d_result_name(result));
        if (do_profile) {
            cudaEventDestroy(profile_start);
            cudaEventDestroy(profile_stop);
        }
        return result;
    }

    if (do_profile) {
        cudaEventRecord(profile_start, stream);
    }
    void * workspace = get_workspace(key.device, stream, plan->workspace_size);
    if (do_profile) {
        cudaEventRecord(profile_stop, stream);
        cudaEventSynchronize(profile_stop);
        workspace_ms = elapsed_ms(profile_start, profile_stop);
    }
    if (plan->workspace_size > 0 && workspace == nullptr) {
        log_unsupported_tensor(dst, "workspace");
        if (do_profile) {
            cudaEventDestroy(profile_start);
            cudaEventDestroy(profile_stop);
        }
        return ED_CUDNN_CONV3D_BUILD_FAILED;
    }

    const void * x_data = dst->src[1]->data;
    const void * w_data = dst->src[0]->data;
    void * y_data = dst->data;
    void * x_cast_data = nullptr;
    void * y_cast_data = nullptr;
    const bool cast_io_to_f16 = uses_f16_io_cast(key.x_type, key.w_type, key.y_type);
    if (cast_io_to_f16) {
        if (do_profile) {
            cudaEventRecord(profile_start, stream);
        }
        const int64_t x_elems = key.n * key.ic * key.id * key.ih * key.iw;
        const int64_t y_elems = key.n * key.oc * key.od * key.oh * key.ow;
        const size_t x_bytes = (size_t) x_elems * sizeof(half);
        const size_t y_bytes = (size_t) y_elems * sizeof(half);
        void * scratch = get_scratch(key.device, stream, x_bytes + y_bytes);
        if (scratch == nullptr) {
            log_unsupported_tensor(dst, "scratch");
            if (do_profile) {
                cudaEventDestroy(profile_start);
                cudaEventDestroy(profile_stop);
            }
            return ED_CUDNN_CONV3D_BUILD_FAILED;
        }
        x_cast_data = scratch;
        x_data = x_cast_data;
        y_cast_data = (char *) scratch + x_bytes;
        y_data = y_cast_data;
        float_to_half_async(dst->src[1]->data, x_cast_data, x_elems, stream);
        if (do_profile) {
            cudaEventRecord(profile_stop, stream);
            cudaEventSynchronize(profile_stop);
            prepare_ms = elapsed_ms(profile_start, profile_stop);
        }
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (do_profile) {
        cudaEventRecord(profile_start, stream);
    }
    const cudnnStatus_t status = cudnnConvolutionForward(handle,
                                                         &alpha,
                                                         plan->x_desc,
                                                         x_data,
                                                         plan->w_desc,
                                                         w_data,
                                                         plan->conv_desc,
                                                         plan->algo,
                                                         workspace,
                                                         plan->workspace_size,
                                                         &beta,
                                                         plan->y_desc,
                                                         y_data);
    if (do_profile) {
        cudaEventRecord(profile_stop, stream);
        cudaEventSynchronize(profile_stop);
        execute_ms = elapsed_ms(profile_start, profile_stop);
    }
    if (status != CUDNN_STATUS_SUCCESS) {
        log_unsupported_tensor(dst, "execute");
        if (do_profile) {
            cudaEventDestroy(profile_start);
            cudaEventDestroy(profile_stop);
        }
        return ED_CUDNN_CONV3D_EXECUTE_FAILED;
    }
    if (cast_io_to_f16) {
        if (do_profile) {
            cudaEventRecord(profile_start, stream);
        }
        const int64_t y_elems = key.n * key.oc * key.od * key.oh * key.ow;
        if (dst->src[2] != nullptr) {
            half_to_float_bias_async(y_cast_data,
                                     dst->src[2]->data,
                                     dst->data,
                                     key.ow,
                                     key.oh,
                                     key.od,
                                     key.oc,
                                     key.n,
                                     stream);
        } else {
            half_to_float_async(y_cast_data, dst->data, y_elems, stream);
        }
        if (do_profile) {
            cudaEventRecord(profile_stop, stream);
            cudaEventSynchronize(profile_stop);
            finalize_ms = elapsed_ms(profile_start, profile_stop);
        }
    }
    if (do_profile) {
        profile_record(key, get_plan_ms, workspace_ms, prepare_ms, finalize_ms, execute_ms);
        cudaEventDestroy(profile_start);
        cudaEventDestroy(profile_stop);
    }
    return ED_CUDNN_CONV3D_SUCCESS;
}

bool ed_cudnn_conv3d_supported(const ggml_tensor * dst, int device) {
    ed_cudnn_conv3d_key key;
    if (!read_key(dst, device, key)) {
        return false;
    }
    return true;
}

const char * ed_cudnn_conv3d_result_name(ed_cudnn_conv3d_result_t result) {
    switch (result) {
        case ED_CUDNN_CONV3D_SUCCESS:
            return "success";
        case ED_CUDNN_CONV3D_UNSUPPORTED:
            return "unsupported";
        case ED_CUDNN_CONV3D_BUILD_FAILED:
            return "build_failed";
        case ED_CUDNN_CONV3D_EXECUTE_FAILED:
            return "execute_failed";
    }
    return "unknown";
}
