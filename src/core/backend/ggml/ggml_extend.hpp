#ifndef __GGML_EXTEND_HPP__
#define __GGML_EXTEND_HPP__

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "backend/ggml/ed_ggml_attention_ext.hpp"
#include "backend/ggml/ed_ggml_norm_ext.hpp"
#include "backend/ggml/ed_ggml_sage_attn_ext.hpp"
#include "backend/ggml/ed_ggml_sp_flux_ext.hpp"
#if defined(ED_ENABLE_ASYNC_OFFLOAD)
#include "backend/ggml/cuda/ed_async_offload.h"
#endif
#include "backend/ggml/ggml_extend_backend.hpp"
#include "backend/ggml/ggml_graph_cut.h"
#include "optimization/cache/cache_graph_scope.hpp"
#include "optimization/cache/state/cache_device_store.hpp"
#include "optimization/cache/model/tap_registry.hpp"
#include "optimization/cache/compile/indicator_lowering.hpp"
#include "optimization/cache/operator/cache_operator.hpp"

#include "edge-dit.h"
#include "core/runtime/model_loader.h"
#include "utils/tensor.hpp"

#include "utils/rng.hpp"
#include "backend/ggml/tensor_ggml.hpp"
#include "utils/util.h"

#include <utility>

#include "parallel/process_group.hpp"


#define EPS 1e-05f

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

#ifndef ED_UNUSED
#define ED_UNUSED(x) (void)(x)
#endif

__STATIC_INLINE__ int align_up_offset(int n, int multiple) {
    return (multiple - n % multiple) % multiple;
}

__STATIC_INLINE__ int align_up(int n, int multiple) {
    return n + align_up_offset(n, multiple);
}

__STATIC_INLINE__ void ggml_log_callback_default(ggml_log_level level, const char* text, void*) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG:
            LOG_DEBUG(text);
            break;
        case GGML_LOG_LEVEL_INFO:
            LOG_INFO(text);
            break;
        case GGML_LOG_LEVEL_WARN:
            LOG_WARN(text);
            break;
        case GGML_LOG_LEVEL_ERROR:
            LOG_ERROR(text);
            break;
        default:
            LOG_DEBUG(text);
    }
}

__STATIC_INLINE__ bool backend_name_exists(std::string name) {
    ggml_backend_load_all_once();
    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count; ++i) {
        if (name == ggml_backend_dev_name(ggml_backend_dev_get(i))) {
            return true;
        }
    }
    return false;
}

__STATIC_INLINE__ std::string sanitize_backend_name(std::string name) {
    if (name == "" || backend_name_exists(name)) {
        return name;
    } else {
        LOG_WARN("Backend %s not found, using default backend", name.c_str());
        return "";
    }
}

__STATIC_INLINE__ std::string get_default_backend_name() {
    ggml_backend_load_all_once();
    // should pick the same backend as ggml_backend_init_best
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev                    = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev                    = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        return "";
    }
    return ggml_backend_dev_name(dev);
}

__STATIC_INLINE__ ggml_backend_t init_named_backend(std::string name = "") {
    ggml_backend_load_all_once();
    LOG_DEBUG("Initializing backend: %s", name.c_str());
    if (name.empty()) {
        return ggml_backend_init_best();
    } else {
        return ggml_backend_init_by_name(name.c_str(), nullptr);
    }
}

static_assert(GGML_MAX_NAME >= 128, "GGML_MAX_NAME must be at least 128");

// n-mode tensor-matrix product
// example: 2-mode product
// A: [ne03, k, ne01, ne00]
// B: k rows, m columns => [k, m]
// result is [ne03, m, ne01, ne00]
__STATIC_INLINE__ ggml_tensor* ggml_ext_mul_n_mode(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int mode = 0) {
    // reshape A
    // swap 0th and nth axis
    a           = ggml_cont(ctx, ggml_permute(ctx, a, mode, mode != 1 ? 1 : 0, mode != 2 ? 2 : 0, mode != 3 ? 3 : 0));
    int64_t ne1 = a->ne[1];
    int64_t ne2 = a->ne[2];
    int64_t ne3 = a->ne[3];
    // make 2D
    a = ggml_cont(ctx, ggml_reshape_2d(ctx, a, a->ne[0], (ne3 * ne2 * ne1)));

    ggml_tensor* result = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, a, b)));

    // reshape output (same shape as a after permutation except first dim)
    result = ggml_reshape_4d(ctx, result, result->ne[0], ne1, ne2, ne3);
    // swap back 0th and nth axis
    result = ggml_permute(ctx, result, mode, mode != 1 ? 1 : 0, mode != 2 ? 2 : 0, mode != 3 ? 3 : 0);
    return result;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_merge_lora(ggml_context* ctx,
                                                   ggml_tensor* lora_down,
                                                   ggml_tensor* lora_up,
                                                   ggml_tensor* lora_mid = nullptr) {
    ggml_tensor* updown;
    // flat lora tensors to multiply it
    int64_t lora_up_rows  = lora_up->ne[ggml_n_dims(lora_up) - 1];
    lora_up               = ggml_reshape_2d(ctx, lora_up, ggml_nelements(lora_up) / lora_up_rows, lora_up_rows);
    auto lora_down_n_dims = ggml_n_dims(lora_down);
    // assume n_dims should always be a multiple of 2 (otherwise rank 1 doesn't work)
    lora_down_n_dims       = (lora_down_n_dims + lora_down_n_dims % 2);
    int64_t lora_down_rows = lora_down->ne[lora_down_n_dims - 1];
    lora_down              = ggml_reshape_2d(ctx, lora_down, ggml_nelements(lora_down) / lora_down_rows, lora_down_rows);

    // ggml_mul_mat requires tensor b transposed
    lora_down = ggml_cont(ctx, ggml_transpose(ctx, lora_down));
    if (lora_mid == nullptr) {
        updown = ggml_mul_mat(ctx, lora_up, lora_down);
        updown = ggml_cont(ctx, ggml_transpose(ctx, updown));
    } else {
        // undoing tucker decomposition for conv layers.
        // lora_mid  has shape (3,    3,   Rank, Rank)
        // lora_down has shape (Rank, In,  1,    1)
        // lora_up   has shape (Rank, Out, 1,    1)
        // conv layer shape is (3,    3,   Out,  In)
        updown = ggml_ext_mul_n_mode(ctx, ggml_ext_mul_n_mode(ctx, lora_mid, lora_down, 3), lora_up, 2);
        updown = ggml_cont(ctx, updown);
    }
    return updown;
}

// Kronecker product
// [ne03,ne02,ne01,ne00] x [ne13,ne12,ne11,ne10] => [ne03*ne13,ne02*ne12,ne01*ne11,ne00*ne10]
__STATIC_INLINE__ ggml_tensor* ggml_ext_kronecker(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_mul(ctx,
                    ggml_interpolate(ctx,
                                     a,
                                     a->ne[0] * b->ne[0],
                                     a->ne[1] * b->ne[1],
                                     a->ne[2] * b->ne[2],
                                     a->ne[3] * b->ne[3],
                                     GGML_SCALE_MODE_NEAREST),
                    b);
}

__STATIC_INLINE__ void ggml_ext_im_set_randn_f32(ggml_tensor* tensor, std::shared_ptr<RNG> rng) {
    uint32_t n                        = (uint32_t)ggml_nelements(tensor);
    std::vector<float> random_numbers = rng->randn(n);
    for (uint32_t i = 0; i < n; i++) {
        ggml_set_f32_1d(tensor, i, random_numbers[i]);
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_set_f32(ggml_tensor* tensor, float value, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]) = value;
}

__STATIC_INLINE__ float ggml_ext_tensor_get_f32(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    if (tensor->buffer != nullptr) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0], sizeof(float));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ int ggml_ext_tensor_get_i32(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    if (tensor->buffer != nullptr) {
        int value;
        ggml_backend_tensor_get(tensor, &value, i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0], sizeof(int));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(int));
    return *(int*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ ggml_fp16_t ggml_ext_tensor_get_f16(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
    return *(ggml_fp16_t*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ float sd_image_get_f32(ed_image_t image, int64_t iw, int64_t ih, int64_t ic, bool scale = true) {
    float value = *(image.data + ih * image.width * image.channels + iw * image.channels + ic);
    if (scale) {
        value /= 255.f;
    }
    return value;
}

__STATIC_INLINE__ void print_ggml_tensor(ggml_tensor* tensor, bool shape_only = false, const char* mark = "") {
    printf("%s (%s): shape(%zu, %zu, %zu, %zu)\n", mark, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    fflush(stdout);
    if (shape_only) {
        return;
    }
    int range = 3;
    for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
        if (i3 >= range && i3 + range < tensor->ne[3]) {
            continue;
        }
        for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
            if (i2 >= range && i2 + range < tensor->ne[2]) {
                continue;
            }
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                if (i1 >= range && i1 + range < tensor->ne[1]) {
                    continue;
                }
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    if (i0 >= range && i0 + range < tensor->ne[0]) {
                        continue;
                    }
                    if (tensor->type == GGML_TYPE_F32) {
                        printf("  [%d, %d, %d, %d] = %f\n", i3, i2, i1, i0, ggml_ext_tensor_get_f32(tensor, i0, i1, i2, i3));
                    } else if (tensor->type == GGML_TYPE_F16) {
                        printf("  [%d, %d, %d, %d] = %f\n", i3, i2, i1, i0, ggml_fp16_to_fp32(ggml_ext_tensor_get_f16(tensor, i0, i1, i2, i3)));
                    } else if (tensor->type == GGML_TYPE_I32) {
                        printf("  [%d, %d, %d, %d] = %i3\n", i3, i2, i1, i0, ggml_ext_tensor_get_i32(tensor, i0, i1, i2, i3));
                    }
                    fflush(stdout);
                }
            }
        }
    }
}

template <typename T>
__STATIC_INLINE__ void print_sd_tensor(const sd::Tensor<T>& tensor, bool shape_only = false, const char* mark = "") {
    printf("%s: shape(", mark);
    for (size_t i = 0; i < static_cast<size_t>(tensor.dim()); ++i) {
        printf("%s%lld", i == 0 ? "" : ", ", static_cast<long long>(tensor.shape()[i]));
    }
    printf(")\n");
    fflush(stdout);
    if (shape_only) {
        return;
    }
    int range                  = 3;
    std::vector<int64_t> shape = tensor.shape();
    while (shape.size() < 4) {
        shape.push_back(1);
    }
    for (int64_t i3 = 0; i3 < shape[3]; i3++) {
        if (i3 >= range && i3 + range < shape[3]) {
            continue;
        }
        for (int64_t i2 = 0; i2 < shape[2]; i2++) {
            if (i2 >= range && i2 + range < shape[2]) {
                continue;
            }
            for (int64_t i1 = 0; i1 < shape[1]; i1++) {
                if (i1 >= range && i1 + range < shape[1]) {
                    continue;
                }
                for (int64_t i0 = 0; i0 < shape[0]; i0++) {
                    if (i0 >= range && i0 + range < shape[0]) {
                        continue;
                    }
                    size_t offset = static_cast<size_t>(i0 + shape[0] * (i1 + shape[1] * (i2 + shape[2] * i3)));
                    printf("  [%lld, %lld, %lld, %lld] = ", static_cast<long long>(i3), static_cast<long long>(i2), static_cast<long long>(i1), static_cast<long long>(i0));
                    if constexpr (std::is_same_v<T, float>) {
                        printf("%f\n", tensor[static_cast<int64_t>(offset)]);
                    } else if constexpr (std::is_same_v<T, ggml_fp16_t>) {
                        printf("%f\n", ggml_fp16_to_fp32(tensor[static_cast<int64_t>(offset)]));
                    } else if constexpr (std::is_same_v<T, int32_t>) {
                        printf("%d\n", tensor[static_cast<int64_t>(offset)]);
                    } else if constexpr (std::is_same_v<T, int64_t>) {
                        printf("%lld\n", static_cast<long long>(tensor[static_cast<int64_t>(offset)]));
                    }
                    fflush(stdout);
                }
            }
        }
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_iter(
    ggml_tensor* tensor,
    const std::function<void(ggml_tensor*, int64_t, int64_t, int64_t, int64_t)>& fn) {
    int64_t n0 = tensor->ne[0];
    int64_t n1 = tensor->ne[1];
    int64_t n2 = tensor->ne[2];
    int64_t n3 = tensor->ne[3];

    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                for (int64_t i0 = 0; i0 < n0; i0++) {
                    fn(tensor, i0, i1, i2, i3);
                }
            }
        }
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_iter(
    ggml_tensor* tensor,
    const std::function<void(ggml_tensor*, int64_t)>& fn) {
    int64_t n0 = tensor->ne[0];
    int64_t n1 = tensor->ne[1];
    int64_t n2 = tensor->ne[2];
    int64_t n3 = tensor->ne[3];

    for (int64_t i = 0; i < ggml_nelements(tensor); i++) {
        fn(tensor, i);
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_diff(
    ggml_tensor* a,
    ggml_tensor* b,
    float gap = 0.1f) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    ggml_ext_tensor_iter(a, [&](ggml_tensor* a, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        float a_value = ggml_ext_tensor_get_f32(a, i0, i1, i2, i3);
        float b_value = ggml_ext_tensor_get_f32(b, i0, i1, i2, i3);
        if (abs(a_value - b_value) > gap) {
            LOG_WARN("[%ld, %ld, %ld, %ld] %f %f", i3, i2, i1, i0, a_value, b_value);
        }
    });
}

__STATIC_INLINE__ ggml_tensor* load_tensor_from_file(ggml_context* ctx, const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return nullptr;
    }
    int32_t n_dims;
    int32_t length;
    int32_t ttype;

    file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    file.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

    LOG_DEBUG("load_tensor_from_file %d %d %d", n_dims, length, ttype);

    if (file.eof()) {
        LOG_ERROR("incomplete file '%s'", file_path.c_str());
        return nullptr;
    }

    int32_t nelements = 1;
    int32_t ne[4]     = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
        file.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
    }
    std::string name(length, 0);
    file.read(&name[0], length);
    ggml_tensor* tensor = ggml_new_tensor_4d(ctx, (ggml_type)ttype, ne[0], ne[1], ne[2], ne[3]);
    const size_t bpe    = ggml_type_size(ggml_type(ttype));
    file.read(reinterpret_cast<char*>(tensor->data), ggml_nbytes(tensor));
    return tensor;
}

// __STATIC_INLINE__ void save_tensor_to_file(const std::string& file_name, ggml_tensor* tensor, const std::string & name) {
//     std::string file_name_ = file_name + ".tensor";
//     std::string name_ = name;
//     std::ofstream file("./" + file_name_, std::ios::binary);
//     file.write(reinterpret_cast<char*>(&tensor->n_dims), sizeof(tensor->n_dims));
//     int len = (int)name_.size();
//     file.write(reinterpret_cast<char*>(&len), sizeof(len));
//     int ttype = (int)tensor->type;
//     file.write(reinterpret_cast<char*>(&ttype), sizeof(ttype));
//     for (int i = 0; i < tensor->n_dims; ++i) {
//         int ne_ = (int) tensor->ne[i];
//         file.write(reinterpret_cast<char*>(&ne_), sizeof(ne_));
//     }
//     file.write(&name_[0], len);
//     char* data = nullptr;
//     file.write((char*)tensor->data, ggml_nbytes(tensor));
//     file.close();
// }

__STATIC_INLINE__ void copy_ggml_tensor(ggml_tensor* dst, ggml_tensor* src) {
    if (dst->type == src->type) {
        dst->nb[0] = src->nb[0];
        dst->nb[1] = src->nb[1];
        dst->nb[2] = src->nb[2];
        dst->nb[3] = src->nb[3];

        memcpy(((char*)dst->data), ((char*)src->data), ggml_nbytes(dst));
        return;
    }
    ggml_init_params params;
    params.mem_size   = 10 * 1024 * 1024;  // for padding
    params.mem_buffer = nullptr;
    params.no_alloc   = false;
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERROR("ggml_init() failed");
        return;
    }
    ggml_tensor* final = ggml_cpy(ctx, src, dst);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, final);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    ggml_free(ctx);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_dup_and_cpy_tensor(ggml_context* ctx, ggml_tensor* src) {
    ggml_tensor* dup = ggml_dup_tensor(ctx, src);
    copy_ggml_tensor(dup, src);
    return dup;
}

__STATIC_INLINE__ float sigmoid(float x) {
    return 1 / (1.0f + expf(-x));
}

// SPECIAL OPERATIONS WITH TENSORS

__STATIC_INLINE__ uint8_t* ggml_tensor_to_sd_image(ggml_tensor* input, uint8_t* image_data = nullptr) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];
    GGML_ASSERT(channels == 3 && input->type == GGML_TYPE_F32);
    if (image_data == nullptr) {
        image_data = (uint8_t*)malloc(width * height * channels);
    }
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value                                               = ggml_ext_tensor_get_f32(input, ix, iy, k);
                *(image_data + iy * width * channels + ix * channels + k) = (uint8_t)(value * 255.0f);
            }
        }
    }
    return image_data;
}

__STATIC_INLINE__ uint8_t* ggml_tensor_to_sd_image(ggml_tensor* input, int idx, bool video = false) {
    int64_t width  = input->ne[0];
    int64_t height = input->ne[1];
    int64_t channels;
    if (video) {
        channels = input->ne[3];
    } else {
        channels = input->ne[2];
    }
    GGML_ASSERT(channels == 3 && input->type == GGML_TYPE_F32);
    uint8_t* image_data = (uint8_t*)malloc(width * height * channels);
    for (int ih = 0; ih < height; ih++) {
        for (int iw = 0; iw < width; iw++) {
            for (int ic = 0; ic < channels; ic++) {
                float value;
                if (video) {
                    value = ggml_ext_tensor_get_f32(input, iw, ih, idx, ic);
                } else {
                    value = ggml_ext_tensor_get_f32(input, iw, ih, ic, idx);
                }
                *(image_data + ih * width * channels + iw * channels + ic) = (uint8_t)(value * 255.0f);
            }
        }
    }
    return image_data;
}

__STATIC_INLINE__ void ed_image_to_ggml_tensor(ed_image_t image,
                                               ggml_tensor* tensor,
                                               bool scale = true) {
    GGML_ASSERT(image.width == tensor->ne[0]);
    GGML_ASSERT(image.height == tensor->ne[1]);
    GGML_ASSERT(image.channels == tensor->ne[2]);
    GGML_ASSERT(1 == tensor->ne[3]);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    ggml_ext_tensor_iter(tensor, [&](ggml_tensor* tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        float value = sd_image_get_f32(image, i0, i1, i2, scale);
        ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2, i3);
    });
}

__STATIC_INLINE__ void ggml_ext_tensor_apply_mask(ggml_tensor* image_data,
                                                  ggml_tensor* mask,
                                                  ggml_tensor* output,
                                                  float masked_value = 0.5f) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    float rescale_mx = 1.f * mask->ne[0] / output->ne[0];
    float rescale_my = 1.f * mask->ne[1] / output->ne[1];
    GGML_ASSERT(output->type == GGML_TYPE_F32);
    for (int ix = 0; ix < width; ix++) {
        for (int iy = 0; iy < height; iy++) {
            int mx  = (int)(ix * rescale_mx);
            int my  = (int)(iy * rescale_my);
            float m = ggml_ext_tensor_get_f32(mask, mx, my);
            m       = round(m);  // inpaint models need binary masks
            ggml_ext_tensor_set_f32(mask, m, mx, my);
            for (int k = 0; k < channels; k++) {
                float value = ggml_ext_tensor_get_f32(image_data, ix, iy, k);
                value       = (1 - m) * (value - masked_value) + masked_value;
                ggml_ext_tensor_set_f32(output, value, ix, iy, k);
            }
        }
    }
}

__STATIC_INLINE__ float ggml_ext_tensor_mean(ggml_tensor* src) {
    float mean        = 0.0f;
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        mean += data[i] / nelements * 1.0f;
    }
    return mean;
}

// a = a+b
__STATIC_INLINE__ void ggml_ext_tensor_add_inplace(ggml_tensor* a, ggml_tensor* b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    int64_t nelements = ggml_nelements(a);
    float* vec_a      = (float*)a->data;
    float* vec_b      = (float*)b->data;
    for (int i = 0; i < nelements; i++) {
        vec_a[i] = vec_a[i] + vec_b[i];
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_scale_inplace(ggml_tensor* src, float scale) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        data[i] = data[i] * scale;
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_clamp_inplace(ggml_tensor* src, float min, float max) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val < min ? min : (val > max ? max : val);
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_tensor_concat(ggml_context* ctx,
                                                      ggml_tensor* a,
                                                      ggml_tensor* b,
                                                      int dim) {
    int64_t ne[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            continue;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }
    ggml_tensor* result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);
    int64_t o[4]        = {0, 0, 0, 0};
    o[dim]              = a->ne[dim];

    float v;
    for (int i3 = 0; i3 < result->ne[3]; i3++) {
        for (int i2 = 0; i2 < result->ne[2]; i2++) {
            for (int i1 = 0; i1 < result->ne[1]; i1++) {
                for (int i0 = 0; i0 < result->ne[0]; i0++) {
                    if (i0 < a->ne[0] && i1 < a->ne[1] && i2 < a->ne[2] && i3 < a->ne[3]) {
                        v = ggml_ext_tensor_get_f32(a, i0, i1, i2, i3);
                    } else {
                        v = ggml_ext_tensor_get_f32(b, i0 - o[0], i1 - o[1], i2 - o[2], i3 - o[3]);
                    }

                    ggml_ext_tensor_set_f32(result, v, i0, i1, i2, i3);
                }
            }
        }
    }
    return result;
}

// convert values from [0, 1] to [-1, 1]
__STATIC_INLINE__ void scale_to_minus1_1(ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val * 2.0f - 1.0f;
    }
}

// convert values from [-1, 1] to [0, 1]
__STATIC_INLINE__ void scale_to_0_1(ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = (val + 1.0f) * 0.5f;
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_cont(ggml_context* ctx,
                                             ggml_tensor* x) {
    if (ggml_is_contiguous(x)) {
        return x;
    }
    return ggml_cont(ctx, x);
}

// torch like permute
__STATIC_INLINE__ ggml_tensor* ggml_ext_torch_permute(ggml_context* ctx,
                                                      ggml_tensor* x,
                                                      int axis0,
                                                      int axis1,
                                                      int axis2,
                                                      int axis3) {
    int torch_axes[4] = {axis0, axis1, axis2, axis3};

    int ggml_axes[4] = {0};
    for (int i = 0; i < 4; ++i) {
        int found = 0;
        for (int j = 0; j < 4; ++j) {
            if (torch_axes[j] == i) {
                ggml_axes[i] = j;
                found        = 1;
                break;
            }
        }
        GGML_ASSERT(found && "Invalid permute input: must be a permutation of 0-3");
    }

    return ggml_permute(ctx, x, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_slice(ggml_context* ctx,
                                              ggml_tensor* x,
                                              int dim,
                                              int64_t start,
                                              int64_t end,
                                              bool cont = true) {
    GGML_ASSERT(dim >= 0 && dim < 4);
    if (x->ne[dim] == 1) {
        return x;
    }
    while (start < 0) {
        start = x->ne[dim] + start;
    }
    while (end < 0) {
        end = x->ne[dim] + end;
    }
    GGML_ASSERT(end > start);
    GGML_ASSERT(start >= 0 && start < x->ne[dim]);
    GGML_ASSERT(end > start && end <= x->ne[dim]);

    int64_t slice_size  = end - start;
    int64_t slice_ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    slice_ne[dim]       = slice_size;

    x = ggml_view_4d(ctx, x,
                     slice_ne[0], slice_ne[1], slice_ne[2], slice_ne[3],
                     x->nb[1], x->nb[2], x->nb[3], start * x->nb[dim]);

    if (cont) {
        x = ggml_cont(ctx, x);
    }

    return x;
}

// example: [N, 3*C, H, W] => ([N, C, H, W], [N, C, H, W], [N, C, H, W])
__STATIC_INLINE__ std::vector<ggml_tensor*> ggml_ext_chunk(ggml_context* ctx,
                                                           ggml_tensor* x,
                                                           int num,
                                                           int64_t dim,
                                                           bool cont = true) {
    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(x->ne[dim] % num == 0);

    std::vector<ggml_tensor*> chunks;
    int64_t chunk_size  = x->ne[dim] / num;
    int64_t stride      = chunk_size * x->nb[dim];
    int64_t chunk_ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    chunk_ne[dim]       = chunk_size;
    for (int i = 0; i < num; i++) {
        auto chunk = ggml_view_4d(
            ctx, x,
            chunk_ne[0], chunk_ne[1], chunk_ne[2], chunk_ne[3],
            x->nb[1], x->nb[2], x->nb[3], stride * i);
        if (cont) {
            chunk = ggml_cont(ctx, chunk);
        }
        chunks.push_back(chunk);
    }

    return chunks;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_silu_act(ggml_context* ctx, ggml_tensor* x, bool gate_first = true) {
    // x: [ne3, ne2, ne1, ne0]
    // return: [ne3, ne2, ne1, ne0/2]

    auto x_vec = ggml_ext_chunk(ctx, x, 2, 0, false);
    ggml_tensor* gate;
    if (gate_first) {
        gate = x_vec[0];
        x    = x_vec[1];
    } else {
        x    = x_vec[0];
        gate = x_vec[1];
    }
    gate = ggml_cont(ctx, gate);
    gate = ggml_silu_inplace(ctx, gate);

    x = ggml_mul(ctx, x, gate);  // [ne3, ne2, ne1, ne0/2]

    return x;
}

typedef std::function<bool(ggml_tensor*, ggml_tensor*, bool)> on_tile_process;

__STATIC_INLINE__ void sd_tiling_calc_tiles(int& num_tiles_dim,
                                            float& tile_overlap_factor_dim,
                                            int small_dim,
                                            int tile_size,
                                            const float tile_overlap_factor,
                                            bool circular) {
    int tile_overlap     = static_cast<int>(tile_size * tile_overlap_factor);
    int non_tile_overlap = tile_size - tile_overlap;

    if (circular) {
        // circular means the last and first tile are overlapping (wraping around)
        num_tiles_dim = small_dim / non_tile_overlap;

        if (num_tiles_dim < 1) {
            num_tiles_dim = 1;
        }

        tile_overlap_factor_dim = (tile_size - small_dim / num_tiles_dim) / (float)tile_size;

        // if single tile and tile_overlap_factor is not 0, add one to ensure we have at least two overlapping tiles
        if (num_tiles_dim == 1 && tile_overlap_factor_dim > 0) {
            num_tiles_dim++;
            tile_overlap_factor_dim = 0.5;
        }

        return;
    }
    // else, non-circular means the last and first tile are not overlapping

    num_tiles_dim     = (small_dim - tile_overlap) / non_tile_overlap;
    int overshoot_dim = ((num_tiles_dim + 1) * non_tile_overlap + tile_overlap) % small_dim;

    if ((overshoot_dim != non_tile_overlap) && (overshoot_dim <= num_tiles_dim * (tile_size / 2 - tile_overlap))) {
        // if tiles don't fit perfectly using the desired overlap
        // and there is enough room to squeeze an extra tile without overlap becoming >0.5
        num_tiles_dim++;
    }

    tile_overlap_factor_dim = (float)(tile_size * num_tiles_dim - small_dim) / (float)(tile_size * (num_tiles_dim - 1));
    if (num_tiles_dim <= 2) {
        if (small_dim <= tile_size) {
            num_tiles_dim           = 1;
            tile_overlap_factor_dim = 0;
        } else {
            num_tiles_dim           = 2;
            tile_overlap_factor_dim = (2 * tile_size - small_dim) / (float)tile_size;
        }
    }
}

// Tiling

__STATIC_INLINE__ int64_t sd_tensor_plane_size(const sd::Tensor<float>& tensor) {
    GGML_ASSERT(tensor.dim() >= 2);
    return tensor.shape()[0] * tensor.shape()[1];
}

template <typename Fn>
__STATIC_INLINE__ void sd_parallel_for(int64_t begin, int64_t end, int threads, Fn&& fn) {
    if (threads <= 1 || end - begin <= 1) {
        for (int64_t index = begin; index < end; ++index) {
            fn(index);
        }
        return;
    }
    threads = std::min<int64_t>(threads, end - begin);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int thread_index = 0; thread_index < threads; ++thread_index) {
        const int64_t start = begin + (end - begin) * thread_index / threads;
        const int64_t stop  = begin + (end - begin) * (thread_index + 1) / threads;
        workers.emplace_back([&, start, stop]() {
            for (int64_t index = start; index < stop; ++index) {
                fn(index);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
}

__STATIC_INLINE__ int sd_vae_parallel_tile_copy_threads() {
    static const int threads = []() {
        const char* enabled = std::getenv("ED_VAE_PARALLEL_TILE_COPY");
        if (enabled != nullptr && enabled[0] != '\0' && std::strcmp(enabled, "0") == 0) {
            return 1;
        }
        const char* value = std::getenv("ED_VAE_PARALLEL_TILE_COPY_THREADS");
        if (value != nullptr && value[0] != '\0') {
            return std::max(1, std::atoi(value));
        }
        return 8;
    }();
    return threads;
}

__STATIC_INLINE__ bool sd_vae_plane_parallel_tile_copy_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("ED_VAE_PLANE_PARALLEL_TILE_COPY");
        return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

__STATIC_INLINE__ sd::Tensor<float> sd_tensor_split_2d(const sd::Tensor<float>& input, int width, int height, int x, int y) {
    GGML_ASSERT(input.dim() >= 4);
    std::vector<int64_t> output_shape = input.shape();
    output_shape[0]                   = width;
    output_shape[1]                   = height;
    sd::Tensor<float> output(std::move(output_shape));
    int64_t input_width  = input.shape()[0];
    int64_t input_height = input.shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(output);
    int64_t plane_count  = input.numel() / input_plane;
    const int threads = sd_vae_parallel_tile_copy_threads();
    if (threads > 1 && sd_vae_plane_parallel_tile_copy_enabled()) {
        sd_parallel_for(0, plane_count, threads, [&](int64_t plane) {
            const int64_t src_plane_offset = plane * input_plane;
            const int64_t dst_plane_offset = plane * output_plane;
            for (int iy = 0; iy < height; ++iy) {
                for (int ix = 0; ix < width; ++ix) {
                    const int64_t src_xy = (ix + x) % input_width + input_width * ((iy + y) % input_height);
                    const int64_t dst_xy = ix + width * iy;
                    output[dst_plane_offset + dst_xy] = input[src_plane_offset + src_xy];
                }
            }
        });
        return output;
    }
    sd_parallel_for(0, static_cast<int64_t>(width) * height, threads, [&](int64_t xy) {
        const int iy = static_cast<int>(xy / width);
        const int ix = static_cast<int>(xy - static_cast<int64_t>(iy) * width);
        int64_t src_xy = (ix + x) % input_width + input_width * ((iy + y) % input_height);
        int64_t dst_xy = ix + width * iy;
        for (int64_t plane = 0; plane < plane_count; ++plane) {
            output[plane * output_plane + dst_xy] = input[plane * input_plane + src_xy];
        }
    });
    return output;
}

__STATIC_INLINE__ void sd_tensor_merge_2d_serial(const sd::Tensor<float>& input,
                                                 sd::Tensor<float>* output,
                                                 int x,
                                                 int y,
                                                 int overlap_x,
                                                 int overlap_y,
                                                 bool circular_x,
                                                 bool circular_y,
                                                 int x_skip,
                                                 int y_skip) {
    GGML_ASSERT(output != nullptr);
    int64_t width        = input.shape()[0];
    int64_t height       = input.shape()[1];
    int64_t img_width    = output->shape()[0];
    int64_t img_height   = output->shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(*output);
    int64_t plane_count  = input.numel() / input_plane;

    auto smootherstep_f32 = [](const float v) -> float {
        GGML_ASSERT(v >= 0.f && v <= 1.f);
        return v * v * v * (v * (6.0f * v - 15.0f) + 10.0f);
    };

    for (int iy = y_skip; iy < height; iy++) {
        for (int ix = x_skip; ix < width; ix++) {
            int64_t src_xy = ix + width * iy;
            int64_t ox     = (x + ix) % img_width;
            int64_t oy     = (y + iy) % img_height;
            int64_t dst_xy = ox + img_width * oy;
            for (int64_t plane = 0; plane < plane_count; ++plane) {
                float new_value = input[plane * input_plane + src_xy];
                if (overlap_x > 0 || overlap_y > 0) {
                    float old_value   = (*output)[plane * output_plane + dst_xy];
                    const float x_f_0 = (circular_x || (overlap_x > 0 && x > 0)) ? (ix - x_skip) / float(overlap_x) : 1.f;
                    const float x_f_1 = (circular_x || (overlap_x > 0 && x < (img_width - width))) ? (width - ix) / float(overlap_x) : 1.f;
                    const float y_f_0 = (circular_y || (overlap_y > 0 && y > 0)) ? (iy - y_skip) / float(overlap_y) : 1.f;
                    const float y_f_1 = (circular_y || (overlap_y > 0 && y < (img_height - height))) ? (height - iy) / float(overlap_y) : 1.f;
                    const float x_f   = std::min(std::min(x_f_0, x_f_1), 1.f);
                    const float y_f   = std::min(std::min(y_f_0, y_f_1), 1.f);
                    (*output)[plane * output_plane + dst_xy] =
                        old_value + new_value * smootherstep_f32(y_f) * smootherstep_f32(x_f);
                } else {
                    (*output)[plane * output_plane + dst_xy] = new_value;
                }
            }
        }
    }
}

__STATIC_INLINE__ void sd_tensor_merge_2d(const sd::Tensor<float>& input,
                                          sd::Tensor<float>* output,
                                          int x,
                                          int y,
                                          int overlap_x,
                                          int overlap_y,
                                          bool circular_x,
                                          bool circular_y,
                                          int x_skip = 0,
                                          int y_skip = 0) {
    GGML_ASSERT(output != nullptr);
    int64_t width        = input.shape()[0];
    int64_t height       = input.shape()[1];
    int64_t img_width    = output->shape()[0];
    int64_t img_height   = output->shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(*output);
    int64_t plane_count  = input.numel() / input_plane;
    GGML_ASSERT(output->numel() / output_plane == plane_count);

    const int threads = sd_vae_parallel_tile_copy_threads();
    if (threads <= 1) {
        sd_tensor_merge_2d_serial(input, output, x, y, overlap_x, overlap_y, circular_x, circular_y, x_skip, y_skip);
        return;
    }

    if (sd_vae_plane_parallel_tile_copy_enabled()) {
        auto smootherstep_f32 = [](const float value) -> float {
            GGML_ASSERT(value >= 0.f && value <= 1.f);
            return value * value * value * (value * (6.0f * value - 15.0f) + 10.0f);
        };
        std::vector<int64_t> destination_x(static_cast<size_t>(width));
        std::vector<int64_t> destination_y(static_cast<size_t>(height));
        std::vector<float> weight_x(static_cast<size_t>(width), 1.f);
        std::vector<float> weight_y(static_cast<size_t>(height), 1.f);
        for (int ix = x_skip; ix < width; ++ix) {
            destination_x[static_cast<size_t>(ix)] = (x + ix) % img_width;
            if (overlap_x > 0 || overlap_y > 0) {
                const float x_f_0 = (circular_x || (overlap_x > 0 && x > 0))
                                        ? (ix - x_skip) / float(overlap_x)
                                        : 1.f;
                const float x_f_1 = (circular_x || (overlap_x > 0 && x < (img_width - width)))
                                        ? (width - ix) / float(overlap_x)
                                        : 1.f;
                weight_x[static_cast<size_t>(ix)] = smootherstep_f32(std::min(std::min(x_f_0, x_f_1), 1.f));
            }
        }
        for (int iy = y_skip; iy < height; ++iy) {
            destination_y[static_cast<size_t>(iy)] = (y + iy) % img_height;
            if (overlap_x > 0 || overlap_y > 0) {
                const float y_f_0 = (circular_y || (overlap_y > 0 && y > 0))
                                        ? (iy - y_skip) / float(overlap_y)
                                        : 1.f;
                const float y_f_1 = (circular_y || (overlap_y > 0 && y < (img_height - height)))
                                        ? (height - iy) / float(overlap_y)
                                        : 1.f;
                weight_y[static_cast<size_t>(iy)] = smootherstep_f32(std::min(std::min(y_f_0, y_f_1), 1.f));
            }
        }
        sd_parallel_for(0, plane_count, threads, [&](int64_t plane) {
            const int64_t src_plane_offset = plane * input_plane;
            const int64_t dst_plane_offset = plane * output_plane;
            for (int iy = y_skip; iy < height; ++iy) {
                for (int ix = x_skip; ix < width; ++ix) {
                    const int64_t src_xy = ix + width * iy;
                    const int64_t ox = destination_x[static_cast<size_t>(ix)];
                    const int64_t oy = destination_y[static_cast<size_t>(iy)];
                    const int64_t dst_xy = ox + img_width * oy;
                    const float new_value = input[src_plane_offset + src_xy];
                    if (overlap_x > 0 || overlap_y > 0) {
                        const float old_value = (*output)[dst_plane_offset + dst_xy];
                        (*output)[dst_plane_offset + dst_xy] =
                            old_value + new_value * weight_y[static_cast<size_t>(iy)] * weight_x[static_cast<size_t>(ix)];
                    } else {
                        (*output)[dst_plane_offset + dst_xy] = new_value;
                    }
                }
            }
        });
        return;
    }

    // unclamped -> expects x in the range [0-1]
    auto smootherstep_f32 = [](const float x) -> float {
        GGML_ASSERT(x >= 0.f && x <= 1.f);
        return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
    };

    const int64_t active_width  = width - x_skip;
    const int64_t active_height = height - y_skip;
    sd_parallel_for(0, active_width * active_height, threads, [&](int64_t active_xy) {
        const int iy = y_skip + static_cast<int>(active_xy / active_width);
        const int ix = x_skip + static_cast<int>(active_xy - static_cast<int64_t>(iy - y_skip) * active_width);
        int64_t src_xy = ix + width * iy;
        int64_t ox     = (x + ix) % img_width;
        int64_t oy     = (y + iy) % img_height;
        int64_t dst_xy = ox + img_width * oy;
        for (int64_t plane = 0; plane < plane_count; ++plane) {
            float new_value = input[plane * input_plane + src_xy];
            if (overlap_x > 0 || overlap_y > 0) {
                float old_value   = (*output)[plane * output_plane + dst_xy];
                const float x_f_0 = (circular_x || (overlap_x > 0 && x > 0)) ? (ix - x_skip) / float(overlap_x) : 1.f;
                const float x_f_1 = (circular_x || (overlap_x > 0 && x < (img_width - width))) ? (width - ix) / float(overlap_x) : 1.f;
                const float y_f_0 = (circular_y || (overlap_y > 0 && y > 0)) ? (iy - y_skip) / float(overlap_y) : 1.f;
                const float y_f_1 = (circular_y || (overlap_y > 0 && y < (img_height - height))) ? (height - iy) / float(overlap_y) : 1.f;
                const float x_f   = std::min(std::min(x_f_0, x_f_1), 1.f);
                const float y_f   = std::min(std::min(y_f_0, y_f_1), 1.f);
                (*output)[plane * output_plane + dst_xy] =
                    old_value + new_value * smootherstep_f32(y_f) * smootherstep_f32(x_f);
            } else {
                (*output)[plane * output_plane + dst_xy] = new_value;
            }
        }
    });
}

template <typename Fn>
__STATIC_INLINE__ sd::Tensor<float> process_tiles_2d(const sd::Tensor<float>& input,
                                                     int output_width,
                                                     int output_height,
                                                     int scale,
                                                     int p_tile_size_x,
                                                     int p_tile_size_y,
                                                     float tile_overlap_factor,
                                                     bool circular_x,
                                                     bool circular_y,
                                                     Fn&& on_processing,
                                                     bool silent = false) {
    sd::Tensor<float> output;
    int input_width  = static_cast<int>(input.shape()[0]);
    int input_height = static_cast<int>(input.shape()[1]);

    GGML_ASSERT(((input_width / output_width) == (input_height / output_height)) &&
                ((output_width / input_width) == (output_height / input_height)));
    GGML_ASSERT(((input_width / output_width) == scale) ||
                ((output_width / input_width) == scale));

    int small_width  = output_width;
    int small_height = output_height;
    bool decode      = output_width > input_width;
    if (decode) {
        small_width  = input_width;
        small_height = input_height;
    }

    int num_tiles_x;
    float tile_overlap_factor_x;
    sd_tiling_calc_tiles(num_tiles_x, tile_overlap_factor_x, small_width, p_tile_size_x, tile_overlap_factor, circular_x);

    int num_tiles_y;
    float tile_overlap_factor_y;
    sd_tiling_calc_tiles(num_tiles_y, tile_overlap_factor_y, small_height, p_tile_size_y, tile_overlap_factor, circular_y);

    int tile_overlap_x     = static_cast<int32_t>(p_tile_size_x * tile_overlap_factor_x);
    int non_tile_overlap_x = p_tile_size_x - tile_overlap_x;
    int tile_overlap_y     = static_cast<int32_t>(p_tile_size_y * tile_overlap_factor_y);
    int non_tile_overlap_y = p_tile_size_y - tile_overlap_y;
    int tile_size_x        = p_tile_size_x < small_width ? p_tile_size_x : small_width;
    int tile_size_y        = p_tile_size_y < small_height ? p_tile_size_y : small_height;
    int input_tile_size_x  = tile_size_x;
    int input_tile_size_y  = tile_size_y;
    int output_tile_size_x = tile_size_x;
    int output_tile_size_y = tile_size_y;
    if (decode) {
        output_tile_size_x *= scale;
        output_tile_size_y *= scale;
    } else {
        input_tile_size_x *= scale;
        input_tile_size_y *= scale;
    }

    int num_tiles   = num_tiles_x * num_tiles_y;
    int tile_count  = 1;
    bool last_y     = false;
    bool last_x     = false;
    float last_time = 0.0f;
    const bool profile_tiles = std::getenv("ED_PROFILE_VAE_TILES") != nullptr;
    int64_t split_us = 0;
    int64_t process_us = 0;
    int64_t allocate_us = 0;
    int64_t merge_us = 0;
    if (!silent) {
        LOG_DEBUG("num tiles : %d, %d ", num_tiles_x, num_tiles_y);
        LOG_DEBUG("optimal overlap : %f, %f (targeting %f)", tile_overlap_factor_x, tile_overlap_factor_y, tile_overlap_factor);
        LOG_DEBUG("processing %i tiles", num_tiles);
        pretty_progress(0, num_tiles, 0.0f);
    }
    for (int y = 0; y < small_height && !last_y; y += non_tile_overlap_y) {
        int dy = 0;
        if (!circular_y && y + tile_size_y >= small_height) {
            int original_y = y;
            y              = small_height - tile_size_y;
            dy             = original_y - y;
            if (decode) {
                dy *= scale;
            }
            last_y = true;
        }
        for (int x = 0; x < small_width && !last_x; x += non_tile_overlap_x) {
            int dx = 0;
            if (!circular_x && x + tile_size_x >= small_width) {
                int original_x = x;
                x              = small_width - tile_size_x;
                dx             = original_x - x;
                if (decode) {
                    dx *= scale;
                }
                last_x = true;
            }

            int x_in  = decode ? x : scale * x;
            int y_in  = decode ? y : scale * y;
            int x_out = decode ? x * scale : x;
            int y_out = decode ? y * scale : y;

            int overlap_x_out = decode ? tile_overlap_x * scale : tile_overlap_x;
            int overlap_y_out = decode ? tile_overlap_y * scale : tile_overlap_y;

            int64_t t1       = ggml_time_ms();
            const int64_t split_begin = profile_tiles ? ggml_time_us() : 0;
            auto input_tile  = sd_tensor_split_2d(input, input_tile_size_x, input_tile_size_y, x_in, y_in);
            const int64_t process_begin = profile_tiles ? ggml_time_us() : 0;
            if (profile_tiles) split_us += process_begin - split_begin;
            auto output_tile = on_processing(input_tile);
            const int64_t allocate_begin = profile_tiles ? ggml_time_us() : 0;
            if (profile_tiles) process_us += allocate_begin - process_begin;
            if (output_tile.empty()) {
                return {};
            }
            GGML_ASSERT(output_tile.shape()[0] == output_tile_size_x && output_tile.shape()[1] == output_tile_size_y);
            if (output.empty()) {
                std::vector<int64_t> output_shape = output_tile.shape();
                output_shape[0]                   = output_width;
                output_shape[1]                   = output_height;
                output                            = sd::Tensor<float>::zeros(std::move(output_shape));
            }
            const int64_t merge_begin = profile_tiles ? ggml_time_us() : 0;
            if (profile_tiles) allocate_us += merge_begin - allocate_begin;
            sd_tensor_merge_2d(output_tile, &output, x_out, y_out, overlap_x_out, overlap_y_out, circular_x, circular_y, dx, dy);
            if (profile_tiles) merge_us += ggml_time_us() - merge_begin;

            if (!silent) {
                int64_t t2 = ggml_time_ms();
                last_time  = (t2 - t1) / 1000.0f;
                pretty_progress(tile_count, num_tiles, last_time);
            }
            tile_count++;
        }
        last_x = false;
    }
    if (!silent && tile_count < num_tiles) {
        pretty_progress(num_tiles, num_tiles, last_time);
    }
    if (output.empty()) {
        return {};
    }
    if (profile_tiles) {
        LOG_INFO("ED_VAE_TILE_PROFILE tiles=%d split_ms=%.3f process_ms=%.3f allocate_ms=%.3f merge_ms=%.3f",
                 num_tiles,
                 split_us / 1000.0,
                 process_us / 1000.0,
                 allocate_us / 1000.0,
                 merge_us / 1000.0);
    }
    return output;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_group_norm_32(ggml_context* ctx,
                                                      ggml_tensor* a) {
    const float eps = 1e-6f;  // default eps parameter
    return ggml_group_norm(ctx, a, 32, eps);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_scale(ggml_context* ctx,
                                              ggml_tensor* x,
                                              float factor,
                                              bool inplace = false) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_scale_inplace(ctx, x, factor);
    } else {
        x = ggml_scale(ctx, x, factor);
    }
    return x;
}

// ggml-vulkan's unary shaders (gelu/silu/...) read their input with a flat linear
// index and ignore nb[] strides, so a non-contiguous VIEW source is miscomputed
// (its supports_op even requires ggml_is_contiguous). ggml_is_contiguous_rows() is
// too weak a guard: a row-packed view with a wide row stride passes it but is not
// fully contiguous. On Vulkan, force a full ggml_cont in that case. CPU/CUDA handle
// strided rows natively, so they keep the cheaper contiguous-rows guard (backend
// defaults to nullptr => unchanged behavior for every existing caller).
__STATIC_INLINE__ bool ggml_ext_unary_needs_full_cont(ggml_backend_t backend, ggml_tensor* x) {
    return sd_backend_is(backend, "Vulkan") && !ggml_is_contiguous(x);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_gelu(ggml_context* ctx,
                                             ggml_tensor* x,
                                             bool inplace           = false,
                                             ggml_backend_t backend = nullptr) {
    if (!ggml_is_contiguous_rows(x) || ggml_ext_unary_needs_full_cont(backend, x)) {
        x = ggml_cont(ctx, x);
    } else if (inplace && !ggml_is_contiguous(x)) {
        inplace = false;
    }
    if (inplace) {
        x = ggml_gelu_inplace(ctx, x);
    } else {
        x = ggml_gelu(ctx, x);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_gelu_quick(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   bool inplace           = false,
                                                   ggml_backend_t backend = nullptr) {
    if (!ggml_is_contiguous_rows(x) || ggml_ext_unary_needs_full_cont(backend, x)) {
        x = ggml_cont(ctx, x);
    } else if (inplace && !ggml_is_contiguous(x)) {
        inplace = false;
    }
    if (inplace) {
        x = ggml_gelu_quick_inplace(ctx, x);
    } else {
        x = ggml_gelu_quick(ctx, x);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_linear(ggml_context* ctx,
                                               ggml_tensor* x,
                                               ggml_tensor* w,
                                               ggml_tensor* b,
                                               bool force_prec_f32 = false,
                                               float scale         = 1.f,
                                               const char* matmul_name = nullptr) {
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }
    if (x->ne[2] * x->ne[3] > 1024) {
        // workaround: avoid ggml cuda error
        int64_t ne2 = x->ne[2];
        int64_t ne3 = x->ne[3];
        x           = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
        x           = ggml_mul_mat(ctx, w, x);
        if (matmul_name != nullptr && matmul_name[0] != '\0') {
            ggml_set_name(x, matmul_name);
        }
        if (force_prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
    } else {
        x = ggml_mul_mat(ctx, w, x);
        if (matmul_name != nullptr && matmul_name[0] != '\0') {
            ggml_set_name(x, matmul_name);
        }
        if (force_prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_pad_ext(ggml_context* ctx,
                                                ggml_tensor* x,
                                                int lp0,
                                                int rp0,
                                                int lp1,
                                                int rp1,
                                                int lp2,
                                                int rp2,
                                                int lp3,
                                                int rp3,
                                                bool circular_x = false,
                                                bool circular_y = false) {
    if (circular_x && circular_y) {
        return ggml_pad_ext_circular(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    }

    if (circular_x && (lp0 != 0 || rp0 != 0)) {
        x   = ggml_pad_ext_circular(ctx, x, lp0, rp0, 0, 0, 0, 0, 0, 0);
        lp0 = rp0 = 0;
    }
    if (circular_y && (lp1 != 0 || rp1 != 0)) {
        x   = ggml_pad_ext_circular(ctx, x, 0, 0, lp1, rp1, 0, 0, 0, 0);
        lp1 = rp1 = 0;
    }

    if (lp0 != 0 || rp0 != 0 || lp1 != 0 || rp1 != 0 || lp2 != 0 || rp2 != 0 || lp3 != 0 || rp3 != 0) {
        x = ggml_pad_ext(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_pad(ggml_context* ctx,
                                            ggml_tensor* x,
                                            int p0,
                                            int p1,
                                            int p2          = 0,
                                            int p3          = 0,
                                            bool circular_x = false,
                                            bool circular_y = false) {
    return ggml_ext_pad_ext(ctx, x, 0, p0, 0, p1, 0, p2, 0, p3, circular_x, circular_y);
}

// w: [OC, IC, KH, KW]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
__STATIC_INLINE__ ggml_tensor* ggml_ext_conv_2d(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* w,
                                                ggml_tensor* b,
                                                int s0          = 1,
                                                int s1          = 1,
                                                int p0          = 0,
                                                int p1          = 0,
                                                int d0          = 1,
                                                int d1          = 1,
                                                bool direct     = false,
                                                bool circular_x = false,
                                                bool circular_y = false,
                                                float scale     = 1.f,
                                                ggml_backend_t backend = nullptr) {
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }
    if (w->ne[2] != x->ne[2] && ggml_n_dims(w) == 2) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], w->ne[1]);
    }

    if ((p0 != 0 || p1 != 0) && (circular_x || circular_y)) {
        x  = ggml_ext_pad_ext(ctx, x, p0, p0, p1, p1, 0, 0, 0, 0, circular_x, circular_y);
        p0 = 0;
        p1 = 0;
    }

    if (direct) {
#if !defined(ED_ENABLE_CUDNN_CONV2D)
        // On CPU with oneDNN, a bf16 conv kernel makes the internal im2col GEMM
        // run on Intel AMX (this box has amx_bf16 but not amx_fp16), which is a
        // large VAE-decode win. Cast the f16 kernel to bf16 once here. Default on;
        // set ED_CONV_BF16=0 to keep the f16 path (non-AMX / non-oneDNN builds).
        // Guarded out of cuDNN-conv2d builds: there the cuDNN path only accepts
        // (f32,f16,f32) or all-same dtype, so an f16->bf16 cast yields a
        // (f32,bf16,f32) node that cuDNN rejects and native conv2d aborts on.
        // This conversion is CPU-only. CUDA's native direct conv2d accepts f16
        // and f32 kernels but rejects bf16, while Vulkan has no f16->bf16 CPY
        // pipeline and already accepts an f16 kernel.
        const char* conv_bf16 = std::getenv("ED_CONV_BF16");
        if ((!conv_bf16 || conv_bf16[0] != '0') && w->type == GGML_TYPE_F16 &&
            sd_backend_is(backend, "CPU")) {
            w = ggml_cast(ctx, w, GGML_TYPE_BF16);
        }
#endif
        x = ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0, d1);
    } else {
        x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

// w: [OC, IC, KD, 1 * 1]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N*OC, OD, OH, OW]
__STATIC_INLINE__ ggml_tensor* ggml_ext_conv_3d(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* w,
                                                ggml_tensor* b,
                                                int64_t IC,
                                                int s0 = 1,
                                                int s1 = 1,
                                                int s2 = 1,
                                                int p0 = 0,
                                                int p1 = 0,
                                                int p2 = 0,
                                                int d0 = 1,
                                                int d1 = 1,
                                                int d2 = 1,
                                                bool direct = false,
                                                ggml_backend_t backend = nullptr) {
    int64_t OC = w->ne[3] / IC;
    int64_t N  = x->ne[3] / IC;
    if (direct) {
#if !defined(ED_ENABLE_CUDNN_CONV3D)
        // Same AMX win as conv2d: a bf16 conv kernel makes the internal CONV_3D
        // im2col GEMM route through oneDNN (Intel AMX bf16). Cast the f16 kernel
        // to bf16 once here. Default on; set ED_CONV_BF16=0 to keep the f16 path
        // (non-AMX / non-oneDNN builds).
        // Guarded out of cuDNN-conv3d builds: there the cuDNN path only accepts
        // (f32,f16,f32) or all-same dtype, so an f16->bf16 cast yields a
        // (f32,bf16,f32) CONV_3D node that cuDNN rejects and native conv3d aborts on.
        // This conversion is CPU-only: CUDA's native direct conv3d and Vulkan
        // must retain their supported f16 kernel representation.
        const char* conv_bf16 = std::getenv("ED_CONV_BF16");
        if ((!conv_bf16 || conv_bf16[0] != '0') && w->type == GGML_TYPE_F16 &&
            sd_backend_is(backend, "CPU")) {
            w = ggml_cast(ctx, w, GGML_TYPE_BF16);
        }
#endif
    }
    x          = direct ? ggml_conv_3d_direct(ctx, w, x, s0, s1, s2, p0, p1, p2, d0, d1, d2, (int)IC, (int)N, (int)OC)
                        : ggml_conv_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2);

    if (b != nullptr) {
#if defined(ED_ENABLE_CUDNN_CONV3D)
        if (direct && b->type == GGML_TYPE_F32 && ggml_nelements(b) == OC) {
            x->src[2] = b;
            return x;
        }
#endif
        b = ggml_reshape_4d(ctx, b, 1, 1, 1, b->ne[0]);  // [OC, 1, 1, 1]
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_conv_3d_direct_typed(ggml_context* ctx,
                                                             ggml_tensor* x,
                                                             ggml_tensor* w,
                                                             ggml_tensor* b,
                                                             int64_t IC,
                                                             int64_t N,
                                                             int64_t OC,
                                                             int s0,
                                                             int s1,
                                                             int s2,
                                                             int p0,
                                                             int p1,
                                                             int p2,
                                                             int d0,
                                                             int d1,
                                                             int d2,
                                                             ggml_type dst_type) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(w != nullptr);
    GGML_ASSERT(IC > 0 && N > 0 && OC > 0);
    GGML_ASSERT(w->ne[3] == IC * OC);
    GGML_ASSERT(x->ne[3] == IC * N);

    auto conv_output_size = [](int64_t input, int64_t kernel, int stride, int pad, int dilation) -> int64_t {
        return (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
    };

    int64_t ne[4];
    ne[0] = conv_output_size(x->ne[0], w->ne[0], s0, p0, d0);
    ne[1] = conv_output_size(x->ne[1], w->ne[1], s1, p1, d1);
    ne[2] = conv_output_size(x->ne[2], w->ne[2], s2, p2, d2);
    ne[3] = OC * N;

    ggml_tensor* y = ggml_new_tensor(ctx, dst_type, 4, ne);
    y->op_params[0] = s0;
    y->op_params[1] = s1;
    y->op_params[2] = s2;
    y->op_params[3] = p0;
    y->op_params[4] = p1;
    y->op_params[5] = p2;
    y->op_params[6] = d0;
    y->op_params[7] = d1;
    y->op_params[8] = d2;
    y->op_params[9] = static_cast<int32_t>(IC);
    y->op_params[10] = static_cast<int32_t>(N);
    y->op_params[11] = static_cast<int32_t>(OC);
    y->op     = GGML_OP_CONV_3D;
    y->src[0] = w;
    y->src[1] = x;

    if (b != nullptr) {
#if defined(ED_ENABLE_CUDNN_CONV3D)
        if (b->type == GGML_TYPE_F32 && ggml_nelements(b) == OC) {
            y->src[2] = b;
            return y;
        }
#endif
        b = ggml_reshape_4d(ctx, b, 1, 1, 1, b->ne[0]);
        y = ggml_add_inplace(ctx, y, b);
    }
    return y;
}

// w: [OC, IC, KD, 1 * 1]
// x: [N, IC, ID, IH*IW]
// b: [OC,]
// result: [N, OC, OD, OH*OW]
__STATIC_INLINE__ ggml_tensor* ggml_ext_conv_3d_nx1x1(ggml_context* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* w,
                                                      ggml_tensor* b,
                                                      int s2 = 1,
                                                      int p2 = 1,
                                                      int d2 = 1) {
    x = ggml_conv_2d(ctx, w, x, 1, s2, 0, p2, 1, d2);  // [N, OC, T, OH * OW]
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }
    return x;  // [N, OC, T, OH * OW]
}

// qkv: [N, L, 3*C]
// return: ([N, L, C], [N, L, C], [N, L, C])
__STATIC_INLINE__ std::vector<ggml_tensor*> split_qkv(ggml_context* ctx,
                                                      ggml_tensor* qkv) {
    qkv = ggml_reshape_4d(ctx, qkv, qkv->ne[0] / 3, 3, qkv->ne[1], qkv->ne[2]);  // [N, L, 3, C]
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 3, 1, 2));                    // [3, N, L, C]

    int64_t offset = qkv->nb[2] * qkv->ne[2];
    auto q         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 0);  // [N, L, C]
    auto k         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 1);  // [N, L, C]
    auto v         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 2);  // [N, L, C]
    return {q, k, v};
}

// qkv: [N, 3*C, H, W]
// return: ([N, C, H, W], [N, C, H, W], [N, C, H, W])
__STATIC_INLINE__ std::vector<ggml_tensor*> split_image_qkv(ggml_context* ctx,
                                                            ggml_tensor* qkv) {
    int64_t W   = qkv->ne[0];
    int64_t H   = qkv->ne[1];
    int64_t C   = qkv->ne[2] / 3;
    int64_t N   = qkv->ne[3];
    int64_t nb1 = qkv->nb[1];
    int64_t nb2 = qkv->nb[2];
    qkv         = ggml_reshape_4d(ctx, qkv, W * H, C, 3, N);                     // [N, 3, C, H*W]
    qkv         = ggml_cont(ctx, ggml_ext_torch_permute(ctx, qkv, 0, 1, 3, 2));  // [3, N, C, H*W]

    int64_t offset = qkv->nb[2] * qkv->ne[2];
    auto q         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 0);  // [N, C, H, W]
    auto k         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 1);  // [N, C, H, W]
    auto v         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 2);  // [N, C, H, W]
    return {q, k, v};
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_full(ggml_context* ctx,
                                             float value,
                                             int64_t ne0,
                                             int64_t ne1,
                                             int64_t ne2,
                                             int64_t ne3) {
    auto one = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:one");
    auto t   = ggml_ext_scale(ctx, one, value);             // [1,]
    t        = ggml_repeat_4d(ctx, t, ne0, ne1, ne2, ne3);  // [ne0, ne1, ne2, ne3]
    return t;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_zeros(ggml_context* ctx,
                                              int64_t ne0,
                                              int64_t ne1,
                                              int64_t ne2,
                                              int64_t ne3) {
    return ggml_ext_full(ctx, 0.f, ne0, ne1, ne2, ne3);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_zeros_like(ggml_context* ctx,
                                                   ggml_tensor* x) {
    return ggml_ext_zeros(ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_ones(ggml_context* ctx,
                                             int64_t ne0,
                                             int64_t ne1,
                                             int64_t ne2,
                                             int64_t ne3) {
    return ggml_ext_full(ctx, 1.f, ne0, ne1, ne2, ne3);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_ones_like(ggml_context* ctx,
                                                  ggml_tensor* x) {
    return ggml_ext_ones(ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_cast_f32(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* a) {
    if (sd_backend_is(backend, "Vulkan")) {
        auto zero_index = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:zero_int");
        auto out        = ggml_reshape_1d(ctx, a, ggml_nelements(a));
        out             = ggml_get_rows(ctx, out, zero_index);
        out             = ggml_reshape(ctx, out, a);
        // auto out = ggml_cast(ctx, a, GGML_TYPE_F32);
        return out;
    } else {
        auto out         = ggml_reshape_2d(ctx, a, 1, ggml_nelements(a));
        ggml_tensor* one = ggml_ext_ones(ctx, 1, 1, 1, 1);  // [1,]
        if (ggml_is_transposed(out)) {
            out = ggml_mul_mat(ctx, one, out);
        } else {
            out = ggml_mul_mat(ctx, out, one);
        }
        out = ggml_reshape(ctx, out, a);
        return out;
    }
}

__STATIC_INLINE__ bool ggml_ext_env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

__STATIC_INLINE__ bool ggml_ext_env_flag_enabled_or_default(const char* name, bool default_enabled) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_enabled;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

__STATIC_INLINE__ bool ggml_ext_prefer_cudnn_sdpa_unpadded(ggml_backend_t backend,
                                                          int64_t L_q,
                                                          int64_t L_k,
                                                          int64_t d_head,
                                                          ggml_tensor* mask,
                                                          bool allow_short_f16_self_attn = false) {
#ifdef ED_ENABLE_CUDNN_SDPA
    if (mask != nullptr ||
        !sd_backend_is(backend, "CUDA") ||
        ggml_ext_env_flag_enabled("ED_DISABLE_CUDNN_SDPA") ||
        ggml_ext_env_flag_enabled("ED_DISABLE_CUDNN_SDPA_UNPAD")) {
        return false;
    }

    const bool supported_head_dim = d_head == 64 || d_head == 128;
    const bool supported_long_self_attn = L_q == L_k && L_q >= 4096;
    const bool supported_short_f16_self_attn =
        (allow_short_f16_self_attn ||
         ggml_ext_env_flag_enabled_or_default("ED_CUDNN_SDPA_SHORT_F16_SELF_ATTN", false)) &&
        L_q == L_k &&
        L_q >= 1024;
    return supported_head_dim && (supported_long_self_attn || supported_short_f16_self_attn);
#else
    ED_UNUSED(backend);
    ED_UNUSED(L_q);
    ED_UNUSED(L_k);
    ED_UNUSED(d_head);
    ED_UNUSED(mask);
    ED_UNUSED(allow_short_f16_self_attn);
    return false;
#endif
}

// q: [N, L_q, C(n_head*d_head)] or [N*n_head, L_q, d_head]
// k: [N, L_k, n_kv_head*d_head] or [N*n_kv_head, L_k, d_head]
// v: [N, L_k, n_kv_head*d_head] or [N, L_k, n_kv_head, d_head]
// mask: [N, L_q, L_k]
// return: [N, L_q, C]
__STATIC_INLINE__ ggml_tensor* ggml_ext_attention_ext(ggml_context* ctx,
                                                      ggml_backend_t backend,
                                                      ggml_tensor* q,
                                                      ggml_tensor* k,
                                                      ggml_tensor* v,
                                                      int64_t n_head,
                                                      ggml_tensor* mask = nullptr,
                                                      bool skip_reshape = false,
                                                      bool flash_attn   = false,
                                                      float kv_scale    = 1.0f,
                                                      bool pad_kv_for_flash_attn = true,
                                                      bool v_is_seq_major = false,
                                                      int sage_layer_idx = -1,
                                                      int sage_total_layers = -1,
                                                      bool allow_masked_flash_attn = false,
                                                      bool allow_short_cudnn_self_attn = false) {  // avoid overflow
    int64_t L_q;
    int64_t L_k;
    int64_t C;
    int64_t N;
    int64_t d_head;
    int64_t n_kv_head;
    if (!skip_reshape) {
        L_q       = q->ne[1];
        L_k       = k->ne[1];
        C         = q->ne[0];
        N         = q->ne[2];
        d_head    = C / n_head;
        n_kv_head = k->ne[0] / d_head;

        // q/k/v may arrive as non-contiguous views (e.g. MMDiT-x dual-attention
        // splits the second QKV projection into strided views). ggml_reshape_4d
        // asserts contiguity, so materialize first when needed.
        if (!ggml_is_contiguous(q)) { q = ggml_cont(ctx, q); }
        if (!ggml_is_contiguous(k)) { k = ggml_cont(ctx, k); }
        if (!ggml_is_contiguous(v)) { v = ggml_cont(ctx, v); }

        q = ggml_reshape_4d(ctx, q, d_head, n_head, L_q, N);       // [N, L_q, n_head, d_head]
        q = ggml_ext_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [N, n_head, L_q, d_head]
        q = ggml_reshape_3d(ctx, q, d_head, L_q, n_head * N);      // [N * n_head, L_q, d_head]

        k = ggml_reshape_4d(ctx, k, d_head, n_kv_head, L_k, N);  // [N, L_k, n_kv_head, d_head]
        const bool will_pad_kv_for_flash_attn =
            pad_kv_for_flash_attn &&
            !ggml_ext_prefer_cudnn_sdpa_unpadded(backend,
                                                 L_q,
                                                 L_k,
                                                 d_head,
                                                 mask,
                                                 allow_short_cudnn_self_attn) &&
            L_k % 256 != 0;
        if (flash_attn && mask == nullptr && sd_backend_is(backend, "CUDA") && !will_pad_kv_for_flash_attn) {
            if (auto k_f16 = edgedit::ggml_ext::attention_v_prep_custom_f16(ctx, k, false)) {
                k = k_f16;
            } else {
                k = ggml_ext_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [N, n_kv_head, L_k, d_head]
                k = ggml_reshape_3d(ctx, k, d_head, L_k, n_kv_head * N);   // [N * n_kv_head, L_k, d_head]
            }
        } else {
            k = ggml_ext_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [N, n_kv_head, L_k, d_head]
            k = ggml_reshape_3d(ctx, k, d_head, L_k, n_kv_head * N);   // [N * n_kv_head, L_k, d_head]
        }

        v = ggml_reshape_4d(ctx, v, d_head, n_kv_head, L_k, N);  // [N, L_k, n_kv_head, d_head]
    } else {
        L_q       = q->ne[1];
        L_k       = k->ne[1];
        d_head    = v->ne[0];
        N         = v->ne[3];
        n_kv_head = k->ne[2] / N;
        C         = d_head * n_head;
    }

    float scale = (1.0f / sqrt((float)d_head));

    int kv_pad       = 0;
    ggml_tensor* kqv = nullptr;

    auto build_kqv = [&](ggml_tensor* q_in, ggml_tensor* k_in, ggml_tensor* v_in, ggml_tensor* mask_in) -> ggml_tensor* {
        // ggml CUDA flash-attn kernels currently require Q to be F32.
        if (q_in->type != GGML_TYPE_F32) {
            q_in = ggml_cast(ctx, q_in, GGML_TYPE_F32);
        }

        if (kv_pad != 0) {
            k_in = ggml_pad(ctx, k_in, 0, kv_pad, 0, 0);
        }
        if (kv_scale != 1.0f) {
            k_in = ggml_ext_scale(ctx, k_in, kv_scale);
        }
        if (k_in->type != GGML_TYPE_F16 || !ggml_is_contiguous(k_in)) {
            k_in = ggml_cast(ctx, k_in, GGML_TYPE_F16);
        }

        if (kv_pad == 0 && kv_scale == 1.0f) {
            if (auto v_f16 = edgedit::ggml_ext::attention_v_prep_custom_f16(ctx, v_in, v_is_seq_major)) {
                if (ggml_backend_supports_op(backend, v_f16)) {
                    v_in = v_f16;
                }
            }
        }
        if (v_in->type != GGML_TYPE_F16 || !ggml_is_contiguous(v_in)) {
            if (!v_is_seq_major) {
                v_in = ggml_ext_cont(ctx, ggml_permute(ctx, v_in, 0, 2, 1, 3));
            }
            v_in = ggml_reshape_3d(ctx, v_in, d_head, L_k, n_kv_head * N);
            if (kv_pad != 0) {
                v_in = ggml_pad(ctx, v_in, 0, kv_pad, 0, 0);
            }
            if (kv_scale != 1.0f) {
                v_in = ggml_ext_scale(ctx, v_in, kv_scale);
            }
            if (v_in->type != GGML_TYPE_F16 || !ggml_is_contiguous(v_in)) {
                v_in = ggml_cast(ctx, v_in, GGML_TYPE_F16);
            }
        }

        if (mask_in != nullptr) {
            mask_in = ggml_transpose(ctx, mask_in);
            if (kv_pad > 0) {
                auto pad_tensor = ggml_ext_full(ctx, -INFINITY, kv_pad, L_q, 1, 1);
                mask_in = ggml_concat(ctx, mask_in, pad_tensor, 0);
            }
        } else {
            if (kv_pad > 0) {
                mask_in         = ggml_ext_zeros(ctx, L_k, L_q, 1, 1);
                auto pad_tensor = ggml_ext_full(ctx, -INFINITY, kv_pad, L_q, 1, 1);
                mask_in         = ggml_concat(ctx, mask_in, pad_tensor, 0);
            }
        }

        if (mask_in != nullptr) {
            // the need for padding got removed in ggml 4767bda
            // ensure we can still use the old version for now
#ifdef GGML_KQ_MASK_PAD
            int mask_pad = 0;
            if (mask_in->ne[1] % GGML_KQ_MASK_PAD != 0) {
                mask_pad = GGML_PAD(L_q, GGML_KQ_MASK_PAD) - mask_in->ne[1];
            }
            if (mask_pad > 0) {
                mask_in = ggml_pad(ctx, mask_in, 0, mask_pad, 0, 0);
            }
#endif
            mask_in = ggml_cast(ctx, mask_in, GGML_TYPE_F16);
        }

        auto out = ggml_flash_attn_ext(ctx, q_in, k_in, v_in, mask_in, scale / kv_scale, 0, 0);
        if (allow_short_cudnn_self_attn) {
            ggml_set_name(out, "minimax_h3.vae.short_f16_self_attn");
        }
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (kv_scale != 1.0f) {
            out = ggml_ext_scale(ctx, out, 1.0f / kv_scale);
        }
        return out;
    };

    if (flash_attn) {
        // LOG_DEBUG("attention_ext L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);

        // SageAttention2-style INT8-QK + F16-PV fast path (opt-in via ED_SAGE_ATTN,
        // default OFF). Handles unpadded L_k natively, so it runs before the
        // kv_pad computation used by the stock flash path. Produces a tensor
        // byte-identical in layout to ggml_flash_attn_ext, so the downstream
        // view/cont/reshape below is reused unchanged. Any unsupported case
        // (mask, d_head != 64, non-CUDA, kv_scale != 1) falls through to the
        // stock path.
        if (edgedit::ggml_ext::sage_attn_enabled() &&
            mask == nullptr &&
            L_q == L_k &&
            (d_head == 64 || d_head == 128) &&
            kv_scale == 1.0f &&
            sd_backend_is(backend, "CUDA") &&
            !edgedit::ggml_ext::sage_attn_skip_layer(sage_layer_idx, sage_total_layers)) {
            if (auto sage = edgedit::ggml_ext::sage_attn_custom(ctx, q, k, v, n_head)) {
                if (ggml_backend_supports_op(backend, sage)) {
                    kqv = ggml_view_3d(ctx, sage, d_head, n_head, L_q, sage->nb[1], sage->nb[2], 0);
                }
            }
        }

        bool can_use_flash_attn = true;
        const bool prefer_cudnn_unpadded = ggml_ext_prefer_cudnn_sdpa_unpadded(backend, L_q, L_k, d_head, mask);
        // KV-256 padding is a CUDA-flash constraint. On CPU the oneDNN brgemm flash
        // kernel tiles kvBlk internally and zero-pads its last tile, so it handles
        // arbitrary L_k with no mask — padding here would instead build a -inf mask
        // that disqualifies the oneDNN path (mask must be null) and forces the slow
        // native f32 fallback. This is why Qwen (L_k=1031, not a 256-multiple) ran
        // attention at ~1844 GFLOP/s while Flux (L_k=1536=6*256, no pad) hit AMX.
        const bool cpu_skip_pad = mask == nullptr && sd_backend_is(backend, "CPU") &&
                                  !ggml_ext_env_flag_enabled("ED_CPU_FLASH_NOPAD_OFF");
        if (pad_kv_for_flash_attn && can_use_flash_attn && !prefer_cudnn_unpadded && !cpu_skip_pad && L_k % 256 != 0) {
            kv_pad = GGML_PAD(L_k, 256) - static_cast<int>(L_k);
        }

        if (mask != nullptr && !allow_masked_flash_attn) {
            can_use_flash_attn = false;
        }

        if (kqv == nullptr && can_use_flash_attn) {
            kqv = build_kqv(q, k, v, mask);
            if (!ggml_backend_supports_op(backend, kqv)) {
                kqv = nullptr;
            } else {
                kqv = ggml_view_3d(ctx, kqv, d_head, n_head, L_q, kqv->nb[1], kqv->nb[2], 0);
            }
        }
    }

    if (kqv == nullptr) {
        // if (flash_attn) {
        //     LOG_DEBUG("fallback to default attention, L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        // }
        v = v_is_seq_major ? ggml_ext_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)) :
                             ggml_ext_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [N, n_kv_head, d_head, L_k]
        v = ggml_reshape_3d(ctx, v, L_k, d_head, n_kv_head * N);   // [N * n_kv_head, d_head, L_k]

        auto kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, L_q, L_k]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_scale_inplace(ctx, kq, scale);
        if (mask) {
            kq = ggml_add_inplace(ctx, kq, mask);
        }
        kq = ggml_soft_max_inplace(ctx, kq);

        kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, L_q, d_head]

        kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head, N);  // [N, n_head, L_q, d_head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                 // [N, L_q, n_head, d_head]
    }

    kqv = ggml_ext_cont(ctx, kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);  // [N, L_q, C]

    return kqv;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_layer_norm(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* w,
                                                   ggml_tensor* b,
                                                   float eps = EPS) {
    x = ggml_norm(ctx, x, eps);
    if (w != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        if (b != nullptr) {
            x = ggml_add_inplace(ctx, x, b);
        }
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_group_norm(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* w,
                                                   ggml_tensor* b,
                                                   int num_groups = 32) {
    if (ggml_n_dims(x) >= 3 && w != nullptr && b != nullptr) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    const float eps = 1e-6f;  // default eps parameter
    x               = ggml_group_norm(ctx, x, num_groups, eps);
    if (w != nullptr && b != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        // b = ggml_repeat(ctx, b, x);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ void ggml_ext_backend_tensor_get_and_sync(ggml_backend_t backend, const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    if ((sd_backend_is(backend, "ROCm") || sd_backend_is(backend, "CUDA") || sd_backend_is(backend, "SYCL")) &&
        !ggml_backend_is_cpu(backend)) {
        ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
        ggml_backend_synchronize(backend);
        return;
    }

    ggml_backend_tensor_get(tensor, data, offset, size);
}

__STATIC_INLINE__ float ggml_ext_backend_tensor_get_f32(ggml_tensor* tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_I32 || tensor->type == GGML_TYPE_BF16);
    float value;
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_t bf16_value;
        ggml_backend_tensor_get(tensor, &bf16_value, 0, sizeof(bf16_value));
        value = ggml_bf16_to_fp32(bf16_value);
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t f16_value;
        ggml_backend_tensor_get(tensor, &f16_value, 0, sizeof(f16_value));
        value = ggml_fp16_to_fp32(f16_value);
    } else {  // GGML_TYPE_I32
        int int32_value;
        ggml_backend_tensor_get(tensor, &int32_value, 0, sizeof(int32_value));
        value = (float)int32_value;
    }
    return value;
}

__STATIC_INLINE__ ggml_tensor* vector_to_ggml_tensor(ggml_context* ctx,
                                                     const std::vector<float>& vec) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

__STATIC_INLINE__ ggml_tensor* vector_to_ggml_tensor_i32(ggml_context* ctx,
                                                         const std::vector<int>& vec) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

__STATIC_INLINE__ std::vector<float> arange(float start, float end, float step = 1.f) {
    std::vector<float> result;

    for (float value = start; value < end; value += step) {
        result.push_back(value);
    }

    return result;
}

// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
__STATIC_INLINE__ std::vector<float> timestep_embedding(std::vector<float> timesteps,
                                                        int dim,
                                                        int max_period       = 10000,
                                                        bool flip_sin_to_cos = true,
                                                        float scale          = 1.f) {
    // timesteps: [N,]
    // embedding: [N, dim]
    size_t N = timesteps.size();
    std::vector<float> embedding(N * dim, 0.f);
    int half = dim / 2;
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i) {
        freqs[i] = (float)std::exp(-std::log(max_period) * i / half);
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < half; ++j) {
            float arg = timesteps[i] * freqs[j] * scale;
            if (flip_sin_to_cos) {
                embedding[i * dim + j]        = std::cos(arg);
                embedding[i * dim + j + half] = std::sin(arg);
            } else {
                embedding[i * dim + j]        = std::sin(arg);
                embedding[i * dim + j + half] = std::cos(arg);
            }
        }
    }
    return embedding;
}

__STATIC_INLINE__ void set_timestep_embedding(std::vector<float> timesteps,
                                              ggml_tensor* embedding,
                                              int dim,
                                              int max_period = 10000) {
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    memcpy(((char*)embedding->data), ((char*)embedding_vec.data()), ggml_nbytes(embedding));
}

__STATIC_INLINE__ void set_timestep_embedding(std::vector<float> timesteps,
                                              sd::Tensor<float>* embedding,
                                              int dim,
                                              int max_period = 10000) {
    GGML_ASSERT(embedding != nullptr);
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    if (embedding->numel() != static_cast<int64_t>(embedding_vec.size())) {
        embedding->resize({dim, static_cast<int64_t>(timesteps.size())});
    }
    std::copy(embedding_vec.begin(), embedding_vec.end(), embedding->values().begin());
}

__STATIC_INLINE__ ggml_tensor* new_timestep_embedding(ggml_context* ctx,
                                                      std::vector<float> timesteps,
                                                      int dim,
                                                      int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [N, dim]
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    ggml_tensor* embedding           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps.size());
    if (embedding->data != nullptr) {
        memcpy(((char*)embedding->data), ((char*)embedding_vec.data()), ggml_nbytes(embedding));
    } else {
        ggml_backend_tensor_set(embedding, embedding_vec.data(), 0, ggml_nbytes(embedding));
    }
    return embedding;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_timestep_embedding(
    ggml_context* ctx,
    ggml_tensor* timesteps,
    int dim,
    int max_period    = 10000,
    float time_factor = 1.0f) {
    timesteps = ggml_ext_scale(ctx, timesteps, time_factor);
    return ggml_timestep_embedding(ctx, timesteps, dim, max_period);
}

__STATIC_INLINE__ size_t ggml_tensor_num(ggml_context* ctx) {
    size_t num = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        num++;
    }
    return num;
}

/* SDXL with LoRA requires more space */
#define MAX_PARAMS_TENSOR_NUM 32768
#define MAX_GRAPH_SIZE 327680

struct WeightAdapter {
    struct ForwardParams {
        enum class op_type_t {
            OP_LINEAR,
            OP_CONV2D,
        } op_type;
        struct {
            bool force_prec_f32 = false;
            float scale         = 1.f;
        } linear;
        struct conv2d_params_t {
            int s0          = 1;
            int s1          = 1;
            int p0          = 0;
            int p1          = 0;
            int d0          = 1;
            int d1          = 1;
            bool direct     = false;
            bool circular_x = false;
            bool circular_y = false;
            float scale     = 1.f;
        } conv2d;
    };
    virtual ggml_tensor* patch_weight(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* weight, const std::string& weight_name) = 0;
    virtual ggml_tensor* forward_with_lora(ggml_context* ctx,
                                           ggml_backend_t backend,
                                           ggml_tensor* x,
                                           ggml_tensor* w,
                                           ggml_tensor* b,
                                           const std::string& prefix,
                                           ForwardParams forward_params)                                                              = 0;
    virtual size_t get_extra_graph_size()                                                                                             = 0;
};

struct GGMLRunnerContext {
    ggml_backend_t backend                        = nullptr;
    ggml_context* ggml_ctx                        = nullptr;
    edgedit::parallel::ProcessGroup* process_group = nullptr;
    bool flash_attn_enabled                       = false;
    bool conv2d_direct_enabled                    = false;
    bool conv2d_auto_direct_enabled               = false;
    bool conv3d_auto_direct_enabled               = false;
    bool conv3d_force_direct_enabled              = false;
    bool circular_x_enabled                       = false;
    bool circular_y_enabled                       = false;
    std::shared_ptr<WeightAdapter> weight_adapter = nullptr;
    // Substep tap registry; null on the uncached path (no taps recorded,
    // graph identical). When set, the model's forward() conditionally taps
    // structural anchors it requests (see cache/model/tap_registry.hpp).
    edgedit::cache::TapRegistry* tap_registry     = nullptr;
    // SageAttention layer-skip context: the model's forward() sets these to the
    // current transformer block index (0-based) and total block count before
    // each block runs, so the sage attention fast path can keep the first/last
    // few layers in F16 (they are most sensitive to INT8 quantization). -1
    // means "unknown" -> sage applies to all layers (stage-1 behavior).
    int sage_layer_idx    = -1;
    int sage_total_layers = -1;
    size_t max_graph_vram_bytes = 0;
    std::function<void(ggml_tensor*, const void*)> bind_backend_tensor_data;
};

// Model-facing tap primitive. Called by a model's forward() at a structural
// landmark (model input, block output k, model output). Conditional: a no-op
// unless the middle layer requested this anchor this substep, so unrequested
// anchors are never pinned as graph outputs (ggml_set_output pins the buffer and
// perturbs gallocr — only pin what is actually read back). When requested, names
// the node and records it; the runner promotes recorded taps into its
// named-tensor index (see expand_tap_registry_nodes) for readback.
inline void tap(GGMLRunnerContext* ctx, const edgedit::cache::AnchorRef& a, ggml_tensor* t) {
    if (ctx == nullptr || ctx->tap_registry == nullptr || t == nullptr) {
        return;
    }
    if (!ctx->tap_registry->wants(a)) {
        return;
    }
    ctx->tap_registry->put(a, t);
}

// Reconstruct the reuse output for a tap-driven inject: x_before + <residual>,
// where the residual depends on the registry's inject kind (host feature / device
// single residual / device gamma-blend). Called by a model forward at the inject
// region start.
inline ggml_tensor* build_tap_inject(GGMLRunnerContext* ctx, ggml_tensor* x_before) {
    using edgedit::cache::TapRegistry;
    ggml_context* c = ctx->ggml_ctx;
    TapRegistry* reg = ctx->tap_registry;
    switch (reg->inject_kind()) {
        case TapRegistry::InjectKind::HostFeature:
            // SD3/Wan host reuse: x_before + host-reconstructed feature. The device
            // residual/blend reuse (MagCache/DiCache) is woven through the cache.*
            // operators via build_stream_override, not this switch.
            return ggml_add(c, x_before, reg->inject_input());
        default:
            return x_before;
    }
}

// Cache-layer stream override for substep reuse. The declarative counterpart to
// build_tap_inject: instead of the runner switching on a hardcoded InjectKind, the
// cache lowering hands over a ReplaceStream GraphExtension whose operator emits the
// reconstruction (x_before + residual, blend, etc.) via op->lower(). The runner
// stays blind to the math. x_before (the region-entry activation) is always the
// first operand; the ext's extra_inputs carry the cache operands (device slot,
// uploaded residual). Emits into whatever ggml_ctx the calling model's forward
// owns, so the same code weaves nodes into flux's / qwen's / any model's graph.
// Falls back to x_before (identity) when no ReplaceStream ext is present or the
// operator lowering fails, so an unserved reuse degrades to a no-op, never a crash.
inline ggml_tensor* build_stream_override(GGMLRunnerContext* ctx, ggml_tensor* x_before) {
    if (ctx == nullptr || ctx->tap_registry == nullptr || x_before == nullptr) {
        return x_before;
    }
    for (const auto& ext : ctx->tap_registry->extensions()) {
        if (ext.sink != edgedit::cache::GraphExtension::Sink::ReplaceStream ||
            ext.op == nullptr) {
            continue;
        }
        std::vector<ggml_tensor*> inputs;
        inputs.push_back(x_before);
        for (ggml_tensor* t : ext.extra_inputs) {
            inputs.push_back(t);
        }
        edgedit::cache::GraphLoweringContext gctx;
        gctx.ctx = ctx->ggml_ctx;
        gctx.runtime_scalars = ext.runtime_scalars;
        std::vector<ggml_tensor*> outputs;
        if (ext.op->lower(gctx, inputs, ext.params, &outputs) && !outputs.empty() &&
            outputs[0] != nullptr) {
            return outputs[0];
        }
    }
    return x_before;
}

// Device backing for CacheStateManager slots (implements the cache core's
// abstract ICacheDeviceStore). Owns one persistent ggml_context + backend buffer,
// allocated on the runtime backend OUTSIDE any per-step compute buffer, so slot
// tensors survive the substep pass's compute(free=false) + reset_graph_cut_run_cache().
// Entries are keyed by (ring_key, ring_index); a shape change frees and rebuilds
// (mirrors ensure_dicache_gpu_state). release_all() runs per generation from
// CacheStateManager::reset(), matching the old reset_dicache_gpu_states() lifetime.
class RunnerCacheDeviceStore final : public edgedit::cache::ICacheDeviceStore {
public:
    explicit RunnerCacheDeviceStore(ggml_backend_t backend) : backend_(backend) {}
    ~RunnerCacheDeviceStore() override { release_all(); }

    void* ensure_entry(uint64_t ring_key, int ring_index,
                       const std::vector<int64_t>& shape) override {
        if (backend_ == nullptr || shape.empty() || shape.size() > GGML_MAX_DIMS) {
            return nullptr;
        }
        const uint64_t key = entry_key(ring_key, ring_index);
        auto it = entries_.find(key);
        if (it != entries_.end() && it->second.shape == shape) {
            return it->second.tensor;
        }
        // Shape changed (or first use): rebuild this entry's own context+buffer.
        if (it != entries_.end()) {
            it->second.free();
            entries_.erase(it);
        }
        Entry e;
        // Own tiny no_alloc context (1 tensor); mirrors new_cache_context but keeps
        // this store independent of GGMLRunner's declaration order.
        {
            ggml_init_params p;
            p.mem_size = ggml_tensor_overhead() + 64;
            p.mem_buffer = nullptr;
            p.no_alloc = true;
            e.ctx = ggml_init(p);
        }
        if (e.ctx == nullptr) {
            return nullptr;
        }
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (size_t i = 0; i < shape.size(); ++i) {
            ne[i] = shape[i];
        }
        e.tensor = ggml_new_tensor(e.ctx, GGML_TYPE_F32,
                                   static_cast<int>(shape.size()), ne);
        e.buffer = ggml_backend_alloc_ctx_tensors(e.ctx, backend_);
        if (e.buffer == nullptr || e.tensor == nullptr) {
            e.free();
            return nullptr;
        }
        e.shape = shape;
        ggml_tensor* t = e.tensor;
        entries_.emplace(key, std::move(e));
        return t;
    }

    void release_all() override {
        for (auto& kv : entries_) {
            kv.second.free();
        }
        entries_.clear();
    }

private:
    struct Entry {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor* tensor = nullptr;
        std::vector<int64_t> shape;
        void free() {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            tensor = nullptr;
            shape.clear();
        }
    };
    static uint64_t entry_key(uint64_t ring_key, int ring_index) {
        // ring_index is tiny (< history_depth); reserve the low 8 bits for it.
        return (ring_key << 8) ^ static_cast<uint64_t>(ring_index & 0xff);
    }
    ggml_backend_t backend_ = nullptr;
    std::unordered_map<uint64_t, Entry> entries_;
};

struct GGMLRunner {
protected:
    typedef std::function<ggml_cgraph*()> get_graph_cb_t;
    typedef std::function<bool(ggml_cgraph*)> pre_compute_cb_t;
    typedef std::function<bool(ggml_cgraph*)> post_compute_cb_t;
    using GraphCutSegment = sd::ggml_graph_cut::Segment;
    using GraphCutPlan    = sd::ggml_graph_cut::Plan;

    struct GraphCopyProfile {
        int64_t graph_tensor_set_us  = 0;
        int64_t backend_map_scan_us  = 0;
        int64_t backend_tensor_set_us = 0;

        size_t graph_leafs          = 0;
        size_t graph_nodes          = 0;
        size_t graph_tensor_entries = 0;
        size_t backend_map_entries  = 0;
        size_t copied_tensors       = 0;
        size_t skipped_not_in_graph = 0;
        size_t skipped_no_buffer    = 0;
        size_t copied_bytes         = 0;
        size_t runtime_const_cache_hits = 0;
        size_t runtime_const_cache_uploads = 0;
        size_t runtime_const_cache_hit_bytes = 0;
        size_t runtime_const_cache_upload_bytes = 0;
        int64_t runtime_const_cache_hit_us = 0;
        int64_t runtime_const_cache_upload_us = 0;
    };

    struct GraphExecuteProfile {
        int64_t offload_ms = 0;
        int64_t alloc_ms   = 0;
        int64_t copy_ms    = 0;
        int64_t compute_ms = 0;
        int64_t post_ms    = 0;
        int64_t cache_ms   = 0;
        int64_t total_ms   = 0;

        int64_t pre_compute_callback_us = 0;
        GraphCopyProfile copy_detail;

        size_t cache_live_bytes   = 0;
        size_t cache_buffer_bytes = 0;
        size_t cache_chunks       = 0;
        size_t cache_pool_bytes   = 0;
        size_t cache_pool_chunks  = 0;
    };

    struct GraphCutMaterializeStageProfile {
        size_t ops = 0;
        size_t bytes = 0;
        size_t cont_ops = 0;
        size_t cont_bytes = 0;
        size_t cpy_ops = 0;
        size_t cpy_bytes = 0;
        size_t concat_ops = 0;
        size_t concat_bytes = 0;
        size_t dup_ops = 0;
        size_t dup_bytes = 0;
        size_t boundary_output_bytes = 0;
        size_t cached_output_bytes = 0;
        size_t comm_input_bytes = 0;
        size_t comm_output_bytes = 0;
        size_t src_boundary_input_bytes = 0;
        size_t src_comm_output_bytes = 0;
        size_t materialize_after_materialize_ops = 0;
        size_t materialize_after_materialize_bytes = 0;
        size_t cont_from_cont_ops = 0;
        size_t cont_from_cont_bytes = 0;
        size_t concat_to_cont_ops = 0;
        size_t concat_to_cont_bytes = 0;
        size_t permute_view_to_cont_ops = 0;
        size_t permute_view_to_cont_bytes = 0;
        size_t materialize_view_materialize_ops = 0;
        size_t materialize_view_materialize_bytes = 0;
        size_t cont_permute_cont_ops = 0;
        size_t cont_permute_cont_bytes = 0;
        std::map<std::string, std::pair<size_t, size_t>> chain_groups;
    };

    struct GraphCutSegmentProfile {
        std::string name;
        std::string comm_names;
        std::string op_histogram;
        std::string io_summary;
        std::array<size_t, GGML_OP_COUNT> op_counts = {};

        size_t nodes      = 0;
        size_t comm_ops   = 0;
        size_t comm_bytes = 0;
        size_t output_bytes = 0;
        size_t cached_output_bytes = 0;
        size_t math_ops = 0;
        size_t layout_ops = 0;
        size_t materialize_ops = 0;
        size_t materialize_bytes = 0;
        size_t cont_ops = 0;
        size_t cont_bytes = 0;
        size_t cpy_ops = 0;
        size_t cpy_bytes = 0;
        size_t concat_ops = 0;
        size_t concat_bytes = 0;
        size_t dup_ops = 0;
        size_t dup_bytes = 0;
        size_t materialize_boundary_output_bytes = 0;
        size_t materialize_cached_output_bytes = 0;
        size_t materialize_comm_input_bytes = 0;
        size_t materialize_comm_output_bytes = 0;
        size_t repeated_materialize_source_groups = 0;
        size_t repeated_materialize_source_ops = 0;
        size_t repeated_materialize_source_bytes = 0;
        size_t materialize_after_materialize_ops = 0;
        size_t materialize_after_materialize_bytes = 0;
        size_t cont_from_cont_ops = 0;
        size_t cont_from_cont_bytes = 0;
        size_t concat_to_cont_ops = 0;
        size_t concat_to_cont_bytes = 0;
        size_t permute_view_to_cont_ops = 0;
        size_t permute_view_to_cont_bytes = 0;
        size_t materialize_view_materialize_ops = 0;
        size_t materialize_view_materialize_bytes = 0;
        size_t cont_permute_cont_ops = 0;
        size_t cont_permute_cont_bytes = 0;
        std::map<std::string, GraphCutMaterializeStageProfile> materialize_stages;
        std::string materialize_top_nodes;
        std::string repeated_materialize_sources;
        std::string materialize_chain_top_nodes;
        std::string materialize_stage_summary;
        std::string materialize_stage_details;

        int64_t build_ms        = 0;
        int64_t runtime_param_ms = 0;
        int64_t offload_ms      = 0;
        int64_t alloc_ms        = 0;
        int64_t copy_ms         = 0;
        int64_t compute_ms      = 0;
        int64_t comm_ms         = 0;
        int64_t cache_ms        = 0;
        int64_t total_ms        = 0;

        int64_t collect_future_inputs_us = 0;
        int64_t reset_runtime_tensors_us = 0;
        int64_t bind_cached_inputs_us    = 0;
        int64_t mark_cache_outputs_us    = 0;
        int64_t pre_compute_callback_us  = 0;
        int64_t segment_graph_free_us    = 0;
        GraphCopyProfile copy_detail;

        size_t cache_live_bytes   = 0;
        size_t cache_buffer_bytes = 0;
        size_t cache_chunks       = 0;
        size_t cache_pool_bytes   = 0;
        size_t cache_pool_chunks  = 0;
    };

    struct GraphCacheChunk {
        GraphCacheChunk() = default;
        GraphCacheChunk(const GraphCacheChunk&) = delete;
        GraphCacheChunk& operator=(const GraphCacheChunk&) = delete;

        GraphCacheChunk(GraphCacheChunk&& other) noexcept {
            move_from(std::move(other));
        }

        GraphCacheChunk& operator=(GraphCacheChunk&& other) noexcept {
            if (this != &other) {
                reset();
                move_from(std::move(other));
            }
            return *this;
        }

        ~GraphCacheChunk() {
            reset();
        }

        ggml_context* ctx            = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::string layout_key;
        std::map<std::string, ggml_tensor*> tensors;
        std::vector<ggml_tensor*> cache_tensors;

        void reset() {
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
                buffer = nullptr;
            }
            if (ctx != nullptr) {
                ggml_free(ctx);
                ctx = nullptr;
            }
            layout_key.clear();
            tensors.clear();
            cache_tensors.clear();
        }

    private:
        void move_from(GraphCacheChunk&& other) noexcept {
            ctx           = other.ctx;
            buffer        = other.buffer;
            layout_key    = std::move(other.layout_key);
            tensors       = std::move(other.tensors);
            cache_tensors = std::move(other.cache_tensors);

            other.ctx    = nullptr;
            other.buffer = nullptr;
            other.layout_key.clear();
            other.tensors.clear();
            other.cache_tensors.clear();
        }
    };

    struct RuntimeConstCacheEntry {
        RuntimeConstCacheEntry() = default;
        RuntimeConstCacheEntry(const RuntimeConstCacheEntry&) = delete;
        RuntimeConstCacheEntry& operator=(const RuntimeConstCacheEntry&) = delete;

        RuntimeConstCacheEntry(RuntimeConstCacheEntry&& other) noexcept {
            move_from(std::move(other));
        }

        RuntimeConstCacheEntry& operator=(RuntimeConstCacheEntry&& other) noexcept {
            if (this != &other) {
                reset();
                move_from(std::move(other));
            }
            return *this;
        }

        ~RuntimeConstCacheEntry() {
            reset();
        }

        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor* tensor = nullptr;
        const void* host_data = nullptr;
        std::string key;
        ggml_type type = GGML_TYPE_COUNT;
        int64_t ne[GGML_MAX_DIMS] = {0, 0, 0, 0};
        size_t nbytes = 0;

        void reset() {
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
                buffer = nullptr;
            }
            if (ctx != nullptr) {
                ggml_free(ctx);
                ctx = nullptr;
            }
            tensor = nullptr;
            host_data = nullptr;
            key.clear();
            type = GGML_TYPE_COUNT;
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                ne[i] = 0;
            }
            nbytes = 0;
        }

    private:
        void move_from(RuntimeConstCacheEntry&& other) noexcept {
            ctx       = other.ctx;
            buffer    = other.buffer;
            tensor    = other.tensor;
            host_data = other.host_data;
            key       = std::move(other.key);
            type      = other.type;
            nbytes    = other.nbytes;
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                ne[i] = other.ne[i];
            }

            other.ctx = nullptr;
            other.buffer = nullptr;
            other.tensor = nullptr;
            other.host_data = nullptr;
            other.key.clear();
            other.type = GGML_TYPE_COUNT;
            other.nbytes = 0;
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                other.ne[i] = 0;
            }
        }
    };

    ggml_backend_t params_backend  = nullptr;
    ggml_backend_t runtime_backend = nullptr;

    ggml_context* params_ctx                    = nullptr;
    ggml_backend_buffer_t params_buffer         = nullptr;
    ggml_context* offload_ctx                   = nullptr;
    ggml_backend_buffer_t runtime_params_buffer = nullptr;
    bool params_on_runtime_backend              = false;
    bool phase_params_pinned_                   = false;

    ggml_context* cache_ctx            = nullptr;
    ggml_backend_buffer_t cache_buffer = nullptr;
    std::vector<GraphCacheChunk> cache_chunks_;
    std::vector<GraphCacheChunk> cache_chunk_pool_;
    size_t cache_chunk_pool_bytes_ = 0;

    ggml_context* compute_ctx    = nullptr;
    ggml_gallocr* compute_allocr = nullptr;

    // ---- Experimental: build-once / reuse-across-steps compute graph ----
    // (ED_CACHE_COMPILED_GRAPHS). When capturing, make_input() stages each input
    // leaf's bytes into a stable runner-owned buffer and records the node in call
    // order; on a later step with matching input shapes the graph is NOT rebuilt —
    // only the staging contents are refreshed. persistent_.gf lives inside
    // compute_ctx, so free_compute_ctx() invalidates it (no separate lifetime).
    struct ReuseInput {
        ggml_tensor* node = nullptr;   // leaf in compute_ctx (valid while gf valid)
        // Heap-stable via unique_ptr: the graph leaf is bound to staging->data(),
        // so the byte buffer's address MUST NOT move when persistent_.inputs (the
        // outer vector) reallocates as later make_input() calls push more slots.
        // A plain std::vector<uint8_t> member would move its storage on realloc and
        // leave earlier leaves bound to freed memory (nondeterministic garbage).
        std::unique_ptr<std::vector<uint8_t>> staging;
    };
    struct PersistentGraph {
        bool valid = false;
        bool reuse_disabled = false;  // built once, but ordered_inputs mismatched -> never reuse
        ggml_cgraph* gf = nullptr;
        std::vector<ReuseInput> inputs;
    };
    PersistentGraph persistent_;
    bool reuse_capture_mode_ = false;  // make_input records+stages while true

    ggml_context* partial_offload_ctx                   = nullptr;
    ggml_backend_buffer_t partial_runtime_params_buffer = nullptr;
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> partial_offload_pairs;
    size_t max_graph_vram_bytes = 0;

    // --- P0-A Phase 2-3: async double-buffered weight prefetch ---
    // ED_ASYNC_OFFLOAD gates the ladder: 0=off(single-slot, Phase 0),
    // 1=structural double-slot with conservative sync, 2=copy-stream+event_sync,
    // 3=true overlap (streamWaitEvent). Default 0.
    struct AsyncOffloadSlot {
        ggml_context*         ctx    = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        size_t                buffer_capacity = 0;  // persistent buffer size (reused across segments)
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> pairs;  // {orig_param, gpu_dup}
        void*  copy_done_event = nullptr;  // ed_copy_event_t (opaque cudaEvent*)
        size_t seg_idx = SIZE_MAX;
        bool   applied = false;            // swap into live params done
        bool   active  = false;            // slot currently holds a prefetched segment
    };
    AsyncOffloadSlot async_slots_[2];
    void* async_copy_stream_ = nullptr;    // ed_copy_stream_t (opaque cudaStream*)
    std::unordered_map<ggml_tensor*, void*> async_cpu_src_;  // orig param -> stable CPU data ptr

    static int async_offload_level() {
        static const int lvl = [] {
            const char* e = std::getenv("ED_ASYNC_OFFLOAD");
            return (e != nullptr && e[0] != '\0') ? std::atoi(e) : 0;
        }();
        return lvl;
    }

    std::shared_ptr<WeightAdapter> weight_adapter = nullptr;

    std::vector<float> one_vec = {1.f};
    ggml_tensor* one_tensor    = nullptr;

    std::vector<int> zero_int_vec = {0};
    ggml_tensor* zero_int_tensor  = nullptr;

    std::map<ggml_tensor*, const void*> backend_tensor_data_map;
    std::map<std::string, ggml_tensor*> cache_tensor_map;  // name -> tensor
    std::unordered_map<std::string, ggml_tensor*> cache_tensor_index_;
    std::vector<RuntimeConstCacheEntry> runtime_const_cache_;
    const std::string final_result_name = "ggml_runner_final_result_tensor";

    bool flash_attn_enabled    = false;
    bool conv2d_direct_enabled = false;
    bool circular_x_enabled    = false;
    bool circular_y_enabled    = false;

    // Non-owning substep tap registry; set before a substep-path build, cleared
    // after. Null on uncached builds (graph identical). Threaded onto the
    // GGMLRunnerContext so the model's tap() calls land here.
    edgedit::cache::TapRegistry* tap_registry_ = nullptr;

    // Device backing for CacheStateManager slots (on-GPU residual reuse). Created
    // lazily on the runtime backend; handed to the engine's state manager via
    // cache_device_store(). Freed with the runner.
    std::unique_ptr<RunnerCacheDeviceStore> cache_device_store_;
public:
    // Public so pipelines can hand it to CacheEngine::init(). The field stays
    // protected; only lazy access is exposed.
    RunnerCacheDeviceStore* cache_device_store() {
        if (cache_device_store_ == nullptr && runtime_backend != nullptr) {
            cache_device_store_ = std::make_unique<RunnerCacheDeviceStore>(runtime_backend);
        }
        return cache_device_store_.get();
    }
protected:

    sd::ggml_graph_cut::PlanCache graph_cut_plan_cache_;
    std::unordered_set<const ggml_tensor*> params_tensor_set_;
    std::shared_ptr<edgedit::parallel::ProcessGroup> process_group_ = nullptr;
    size_t graph_cut_profile_index_ = 0;

    template <typename T>
    static sd::Tensor<T> take_or_empty(std::optional<sd::Tensor<T>> tensor) {
        if (!tensor.has_value()) {
            return {};
        }
        return std::move(*tensor);
    }

    template <typename T>
    static sd::Tensor<T> restore_trailing_singleton_dims(std::optional<sd::Tensor<T>> tensor,
                                                         size_t expected_dim) {
        return restore_trailing_singleton_dims(take_or_empty(std::move(tensor)), expected_dim);
    }

    template <typename T>
    static sd::Tensor<T> restore_trailing_singleton_dims(sd::Tensor<T> tensor,
                                                         size_t expected_dim) {
        if (tensor.empty()) {
            return tensor;
        }
        while (static_cast<size_t>(tensor.dim()) < expected_dim) {
            tensor.unsqueeze_(tensor.dim());
        }
        return tensor;
    }

    static bool env_flag_enabled(const char* name) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0 &&
               std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "OFF") != 0;
    }

    static int env_int_or_default(const char* name, int fallback) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return fallback;
        }
        char* end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0') {
            return fallback;
        }
        return static_cast<int>(parsed);
    }

    bool graph_cut_profile_enabled() const {
        return env_flag_enabled("ED_PROFILE_GRAPH_CUTS");
    }

    bool runner_profile_enabled() const {
        return env_flag_enabled("ED_PROFILE_RUNNER");
    }

    bool graph_cut_cache_compact_enabled() const {
        return env_flag_enabled("ED_GRAPH_CUT_CACHE_COMPACT");
    }

    bool graph_cut_cache_sync_after_copy_enabled() const {
        return env_flag_enabled("ED_GRAPH_CUT_CACHE_SYNC_AFTER_COPY");
    }

    bool graph_cut_cache_pool_enabled() const {
        const char* value = std::getenv("ED_GRAPH_CUT_CACHE_POOL");
        if (value == nullptr || value[0] == '\0') {
            return true;
        }
        return env_flag_enabled("ED_GRAPH_CUT_CACHE_POOL");
    }

    size_t graph_cut_cache_pool_max_bytes() const {
        int max_mb = env_int_or_default("ED_GRAPH_CUT_CACHE_POOL_MAX_MB", 1024);
        if (max_mb <= 0) {
            return 0;
        }
        return static_cast<size_t>(max_mb) * 1024 * 1024;
    }

    bool graph_cut_cache_pool_retained_between_runs() const {
        return !graph_cut_cache_compact_enabled() && graph_cut_cache_pool_enabled();
    }

    bool runtime_const_cache_enabled() const {
        const char* value = std::getenv("ED_RUNTIME_CONST_CACHE");
        if (value == nullptr || value[0] == '\0') {
            return true;
        }
        return env_flag_enabled("ED_RUNTIME_CONST_CACHE");
    }

    bool graph_cut_profile_should_log_rank() const {
        if (env_flag_enabled("ED_PROFILE_GRAPH_CUTS_ALL_RANKS")) {
            return true;
        }
        return process_group_ == nullptr || !process_group_->enabled() || process_group_->rank() == 0;
    }

    int graph_cut_profile_top_n() const {
        return std::max(0, env_int_or_default("ED_PROFILE_GRAPH_CUTS_TOP", 8));
    }

    int graph_cut_profile_compute_top_n() const {
        return std::max(0, env_int_or_default("ED_PROFILE_GRAPH_CUTS_COMPUTE_TOP", 0));
    }

    int graph_cut_profile_materialize_top_n() const {
        return std::max(0, env_int_or_default("ED_PROFILE_GRAPH_CUTS_MATERIALIZE_TOP", 0));
    }

    static size_t graph_cut_comm_tensor_bytes(ggml_cgraph* gf,
                                              const GraphCutSegment::CommOp& comm_op) {
        ggml_tensor* output = sd::ggml_graph_cut::comm_output_tensor(gf, comm_op);
        if (output != nullptr) {
            return ggml_nbytes(output);
        }
        ggml_tensor* input = sd::ggml_graph_cut::comm_input_tensor(gf, comm_op);
        return input != nullptr ? ggml_nbytes(input) : 0;
    }

    static std::string graph_cut_comm_names(const GraphCutSegment& segment) {
        if (segment.comm_ops.empty()) {
            return "-";
        }

        std::string names;
        for (size_t i = 0; i < segment.comm_ops.size(); ++i) {
            const auto& op = segment.comm_ops[i];
            if (i > 0) {
                names += ",";
            }
            names += sd::ggml_graph_cut::comm_kind_name(op.kind);
            if (!op.name.empty()) {
                names += ":";
                names += op.name;
            }
            if (names.size() > 240) {
                names.resize(240);
                names += "...";
                break;
            }
        }
        return names;
    }

    static double bytes_to_mib(size_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }

    static ggml_context* new_cache_context(size_t tensor_count_hint = MAX_PARAMS_TENSOR_NUM) {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(std::max<size_t>(1, tensor_count_hint) * ggml_tensor_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);
        return ctx;
    }

    static bool graph_cut_segment_name_starts_with(const std::string& name,
                                                   const char* prefix) {
        return name.rfind(prefix, 0) == 0;
    }

    static bool graph_cut_segment_is_double_block_output(const std::string& name) {
        return graph_cut_segment_name_starts_with(name, "flux.double_blocks.");
    }

    static bool graph_cut_segment_is_single_block_output(const std::string& name) {
        return graph_cut_segment_name_starts_with(name, "flux.single_blocks.");
    }

    static bool graph_cut_op_is_math(enum ggml_op op) {
        switch (op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_FLASH_ATTN_EXT:
            case GGML_OP_RMS_NORM:
            case GGML_OP_NORM:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID:
            case GGML_OP_ADD1:
            case GGML_OP_MUL:
            case GGML_OP_DIV:
            case GGML_OP_SCALE:
            case GGML_OP_SOFT_MAX:
            case GGML_OP_UNARY:
            case GGML_OP_GLU:
            case GGML_OP_ROPE:
                return true;
            default:
                return false;
        }
    }

    static bool graph_cut_op_is_layout(enum ggml_op op) {
        switch (op) {
            case GGML_OP_DUP:
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CONCAT:
            case GGML_OP_GET_ROWS:
            case GGML_OP_SET_ROWS:
            case GGML_OP_REPEAT:
                return true;
            default:
                return false;
        }
    }

    static bool graph_cut_op_is_materialize(enum ggml_op op) {
        switch (op) {
            case GGML_OP_CONT:
            case GGML_OP_CPY:
            case GGML_OP_CONCAT:
            case GGML_OP_DUP:
                return true;
            default:
                return false;
        }
    }

    static std::string graph_cut_tensor_summary(const ggml_tensor* tensor) {
        if (tensor == nullptr) {
            return "<null>";
        }
        std::ostringstream oss;
        std::string name = tensor->name[0] != '\0' ? std::string(tensor->name) : std::string("<unnamed>");
        if (name.size() > 96) {
            name.resize(96);
            name += "...";
        }
        oss << name
            << "["
            << ggml_type_name(tensor->type)
            << ":"
            << tensor->ne[0]
            << "x"
            << tensor->ne[1]
            << "x"
            << tensor->ne[2]
            << "x"
            << tensor->ne[3]
            << "]";
        return oss.str();
    }

    static std::string graph_cut_tensor_layout_summary(const ggml_tensor* tensor) {
        if (tensor == nullptr) {
            return "<null>";
        }
        std::ostringstream oss;
        oss << graph_cut_tensor_summary(tensor)
            << "/bytes=" << bytes_to_mib(ggml_nbytes(tensor)) << "MiB"
            << "/nb=" << tensor->nb[0]
            << ":" << tensor->nb[1]
            << ":" << tensor->nb[2]
            << ":" << tensor->nb[3]
            << "/view=" << (tensor->view_src != nullptr ? "yes" : "no")
            << "/contig=" << (ggml_is_contiguous(tensor) ? "yes" : "no");
        return oss.str();
    }

    static void graph_cut_append_limited(std::ostringstream& oss,
                                         const std::string& value,
                                         size_t& emitted,
                                         size_t limit,
                                         const char* separator = ",") {
        if (emitted >= limit) {
            return;
        }
        if (emitted > 0) {
            oss << separator;
        }
        oss << value;
        ++emitted;
    }

    static std::string graph_cut_segment_io_summary(ggml_cgraph* gf,
                                                    const GraphCutSegment& segment,
                                                    size_t input_limit = 4,
                                                    size_t output_limit = 4) {
        std::ostringstream oss;
        oss << "inputs=";
        size_t emitted_inputs = 0;
        for (const auto& input_ref : segment.input_refs) {
            graph_cut_append_limited(oss,
                                     graph_cut_tensor_summary(sd::ggml_graph_cut::input_tensor(gf, input_ref)),
                                     emitted_inputs,
                                     input_limit);
        }
        if (segment.input_refs.size() > input_limit) {
            oss << ",...";
        }

        oss << " outputs=";
        size_t emitted_outputs = 0;
        for (size_t output_idx = 0; output_idx < segment.output_node_indices.size(); ++output_idx) {
            graph_cut_append_limited(oss,
                                     graph_cut_tensor_summary(sd::ggml_graph_cut::output_tensor(gf, segment, output_idx)),
                                     emitted_outputs,
                                     output_limit);
        }
        if (segment.output_node_indices.size() > output_limit) {
            oss << ",...";
        }
        return oss.str();
    }

    struct GraphCutConsumerProfile {
        size_t total = 0;
        std::array<size_t, GGML_OP_COUNT> op_counts = {};
    };

    static GraphCutConsumerProfile graph_cut_tensor_consumers(ggml_cgraph* gf,
                                                              const GraphCutSegment& segment,
                                                              const ggml_tensor* tensor) {
        GraphCutConsumerProfile profile;
        if (gf == nullptr || tensor == nullptr) {
            return profile;
        }
        for (int node_idx : segment.internal_node_indices) {
            if (node_idx < 0 || node_idx >= ggml_graph_n_nodes(gf)) {
                continue;
            }
            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            if (node == nullptr ||
                node->op < GGML_OP_NONE ||
                node->op >= GGML_OP_COUNT) {
                continue;
            }
            bool consumes = false;
            if (node->view_src == tensor) {
                consumes = true;
            }
            for (int src_idx = 0; src_idx < GGML_MAX_SRC && !consumes; ++src_idx) {
                if (node->src[src_idx] == tensor) {
                    consumes = true;
                }
            }
            if (!consumes) {
                continue;
            }
            ++profile.total;
            ++profile.op_counts[static_cast<size_t>(node->op)];
        }
        return profile;
    }

    static std::string graph_cut_consumer_histogram(const GraphCutConsumerProfile& profile,
                                                    size_t limit = 6) {
        if (profile.total == 0) {
            return "-";
        }
        std::vector<size_t> order;
        order.reserve(GGML_OP_COUNT);
        for (size_t i = 0; i < GGML_OP_COUNT; ++i) {
            if (profile.op_counts[i] > 0) {
                order.push_back(i);
            }
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (profile.op_counts[a] != profile.op_counts[b]) {
                return profile.op_counts[a] > profile.op_counts[b];
            }
            return a < b;
        });

        std::ostringstream oss;
        const size_t hist_limit = std::min(order.size(), limit);
        for (size_t i = 0; i < hist_limit; ++i) {
            if (i > 0) {
                oss << ",";
            }
            const auto op = static_cast<enum ggml_op>(order[i]);
            oss << ggml_op_name(op) << ":" << profile.op_counts[order[i]];
        }
        if (order.size() > hist_limit) {
            oss << ",...";
        }
        return oss.str();
    }

    struct GraphCutMaterializeTrace {
        int node_idx = -1;
        enum ggml_op op = GGML_OP_NONE;
        size_t bytes = 0;
        size_t src_bytes = 0;
        bool dst_is_boundary_output = false;
        bool dst_is_cached_output = false;
        bool dst_is_comm_input = false;
        bool dst_is_comm_output = false;
        bool src_is_boundary_input = false;
        bool src_is_comm_output = false;
        bool src_is_cached_input = false;
        bool src_is_view = false;
        bool src_is_contiguous = false;
        bool is_materialize_after_materialize = false;
        bool is_cont_from_cont = false;
        bool is_concat_to_cont = false;
        bool is_permute_view_to_cont = false;
        bool is_materialize_view_materialize = false;
        bool is_cont_permute_cont = false;
        std::string dst;
        std::string src;
        std::string producer_op;
        std::string consumer_ops;
        std::string concat_inputs;
        std::string chain;
    };

    struct GraphCutMaterializeProfile {
        size_t ops = 0;
        size_t bytes = 0;
        size_t cont_ops = 0;
        size_t cont_bytes = 0;
        size_t cpy_ops = 0;
        size_t cpy_bytes = 0;
        size_t concat_ops = 0;
        size_t concat_bytes = 0;
        size_t dup_ops = 0;
        size_t dup_bytes = 0;
        size_t boundary_output_bytes = 0;
        size_t cached_output_bytes = 0;
        size_t comm_input_bytes = 0;
        size_t comm_output_bytes = 0;
        size_t repeated_source_groups = 0;
        size_t repeated_source_ops = 0;
        size_t repeated_source_bytes = 0;
        size_t materialize_after_materialize_ops = 0;
        size_t materialize_after_materialize_bytes = 0;
        size_t cont_from_cont_ops = 0;
        size_t cont_from_cont_bytes = 0;
        size_t concat_to_cont_ops = 0;
        size_t concat_to_cont_bytes = 0;
        size_t permute_view_to_cont_ops = 0;
        size_t permute_view_to_cont_bytes = 0;
        size_t materialize_view_materialize_ops = 0;
        size_t materialize_view_materialize_bytes = 0;
        size_t cont_permute_cont_ops = 0;
        size_t cont_permute_cont_bytes = 0;
        std::map<std::string, GraphCutMaterializeStageProfile> stages;
        std::string top_nodes;
        std::string repeated_sources;
        std::string top_chains;
    };

    static const ggml_tensor* graph_cut_primary_tensor_src(const ggml_tensor* tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        if (tensor->src[0] != nullptr) {
            return tensor->src[0];
        }
        return tensor->view_src;
    }

    static std::string graph_cut_tensor_name_or_empty(const ggml_tensor* tensor) {
        if (tensor == nullptr || tensor->name[0] == '\0') {
            return std::string();
        }
        return std::string(tensor->name);
    }

    static bool graph_cut_string_contains_any(const std::string& value,
                                              std::initializer_list<const char*> needles) {
        for (const char* needle : needles) {
            if (needle != nullptr && value.find(needle) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    static bool graph_cut_tensor_name_contains_any(const ggml_tensor* tensor,
                                                   std::initializer_list<const char*> needles) {
        return graph_cut_string_contains_any(graph_cut_tensor_name_or_empty(tensor), needles);
    }

    static bool graph_cut_tensor_name_ends_with(const ggml_tensor* tensor,
                                                const char* suffix) {
        if (tensor == nullptr || suffix == nullptr) {
            return false;
        }
        const std::string name = graph_cut_tensor_name_or_empty(tensor);
        const size_t suffix_len = std::strlen(suffix);
        return name.size() >= suffix_len &&
               name.compare(name.size() - suffix_len, suffix_len, suffix) == 0;
    }

    static std::string graph_cut_attention_qkv_layout_stage(const ggml_tensor* tensor,
                                                            const std::string& fallback) {
        if (graph_cut_tensor_name_contains_any(tensor, {"_q_attn"})) {
            return "attention.q_layout";
        }
        if (graph_cut_tensor_name_contains_any(tensor, {"_k_attn"})) {
            return "attention.k_layout";
        }
        if (graph_cut_tensor_name_contains_any(tensor, {"_v_attn"})) {
            return "attention.v_layout";
        }
        return fallback;
    }

    static std::string graph_cut_chain_op_label(const ggml_tensor* tensor) {
        if (tensor == nullptr ||
            tensor->op < GGML_OP_NONE ||
            tensor->op >= GGML_OP_COUNT) {
            return "<invalid>";
        }
        std::string label = ggml_op_name(tensor->op);
        if (tensor->view_src != nullptr) {
            label += "(view)";
        }
        return label;
    }

    static std::string graph_cut_materialize_chain_signature(const ggml_tensor* node) {
        if (node == nullptr) {
            return "-";
        }

        std::vector<std::string> chain;
        std::unordered_set<const ggml_tensor*> seen;
        const ggml_tensor* current = node;
        for (size_t depth = 0; current != nullptr && depth < 8; ++depth) {
            if (seen.find(current) != seen.end()) {
                break;
            }
            seen.insert(current);

            const bool include =
                current == node ||
                graph_cut_op_is_materialize(current->op) ||
                graph_cut_op_is_layout(current->op) ||
                current->view_src != nullptr;
            if (!include) {
                break;
            }

            chain.push_back(graph_cut_chain_op_label(current));

            // CONCAT has multiple semantic inputs; following src[0] would make
            // the chain look more precise than it really is.  Stop at CONCAT
            // and let concat_inputs carry the detailed source shapes.
            if (current != node && current->op == GGML_OP_CONCAT) {
                break;
            }

            const ggml_tensor* src = graph_cut_primary_tensor_src(current);
            if (src == nullptr || src == current) {
                break;
            }
            current = src;
        }

        if (chain.empty()) {
            return "-";
        }
        std::reverse(chain.begin(), chain.end());
        std::ostringstream oss;
        for (size_t i = 0; i < chain.size(); ++i) {
            if (i > 0) {
                oss << "->";
            }
            oss << chain[i];
        }
        return oss.str();
    }

    static void graph_cut_accumulate_materialize_stage(GraphCutMaterializeStageProfile& stage,
                                                       const GraphCutMaterializeTrace& trace) {
        ++stage.ops;
        stage.bytes += trace.bytes;
        switch (trace.op) {
            case GGML_OP_CONT:
                ++stage.cont_ops;
                stage.cont_bytes += trace.bytes;
                break;
            case GGML_OP_CPY:
                ++stage.cpy_ops;
                stage.cpy_bytes += trace.bytes;
                break;
            case GGML_OP_CONCAT:
                ++stage.concat_ops;
                stage.concat_bytes += trace.bytes;
                break;
            case GGML_OP_DUP:
                ++stage.dup_ops;
                stage.dup_bytes += trace.bytes;
                break;
            default:
                break;
        }
        if (trace.dst_is_boundary_output) {
            stage.boundary_output_bytes += trace.bytes;
        }
        if (trace.dst_is_cached_output) {
            stage.cached_output_bytes += trace.bytes;
        }
        if (trace.dst_is_comm_input) {
            stage.comm_input_bytes += trace.bytes;
        }
        if (trace.dst_is_comm_output) {
            stage.comm_output_bytes += trace.bytes;
        }
        if (trace.src_is_boundary_input) {
            stage.src_boundary_input_bytes += trace.bytes;
        }
        if (trace.src_is_comm_output) {
            stage.src_comm_output_bytes += trace.bytes;
        }
        if (trace.is_materialize_after_materialize) {
            ++stage.materialize_after_materialize_ops;
            stage.materialize_after_materialize_bytes += trace.bytes;
        }
        if (trace.is_cont_from_cont) {
            ++stage.cont_from_cont_ops;
            stage.cont_from_cont_bytes += trace.bytes;
        }
        if (trace.is_concat_to_cont) {
            ++stage.concat_to_cont_ops;
            stage.concat_to_cont_bytes += trace.bytes;
        }
        if (trace.is_permute_view_to_cont) {
            ++stage.permute_view_to_cont_ops;
            stage.permute_view_to_cont_bytes += trace.bytes;
        }
        if (trace.is_materialize_view_materialize) {
            ++stage.materialize_view_materialize_ops;
            stage.materialize_view_materialize_bytes += trace.bytes;
        }
        if (trace.is_cont_permute_cont) {
            ++stage.cont_permute_cont_ops;
            stage.cont_permute_cont_bytes += trace.bytes;
        }
        if (trace.chain.find("->") != std::string::npos) {
            auto& chain_group = stage.chain_groups[trace.chain];
            ++chain_group.first;
            chain_group.second += trace.bytes;
        }
    }

    static std::string graph_cut_materialize_stage_from_trace(const GraphCutMaterializeTrace& trace,
                                                              const ggml_tensor* node,
                                                              const ggml_tensor* primary_src) {
        if (trace.dst_is_comm_input) {
            return "comm.send_input";
        }
        if (trace.dst_is_comm_output) {
            return "comm.recv_placeholder";
        }
        if (trace.src_is_comm_output || trace.src_is_boundary_input) {
            if (graph_cut_tensor_name_contains_any(node, {"qkv_seq_to_head_output"})) {
                return "qkv.recv_output_restore";
            }
            if (graph_cut_tensor_name_contains_any(node, {"attn_head_to_seq_output"})) {
                return "head_to_seq.recv_output_restore";
            }
            if (graph_cut_tensor_name_contains_any(node, {"_attn_head_to_seq_output_"})) {
                return "head_to_seq.recv_output_restore";
            }
            if (graph_cut_tensor_name_contains_any(node, {"_q_attn", "_k_attn", "_v_attn"})) {
                return "attention.qkv_from_boundary";
            }
            if (trace.op == GGML_OP_CONCAT &&
                graph_cut_tensor_name_contains_any(node, {"double_txt_img_gather", "txt_img_split"})) {
                return "double_to_single.gather_resplit";
            }
            return "boundary_input_restore";
        }

        if (graph_cut_tensor_name_contains_any(node, {"_qkv_seq_to_head_combined"})) {
            return "qkv.send_combine";
        }
        if (graph_cut_tensor_name_contains_any(node, {"_qkv_seq_to_head_send_flat"})) {
            return "qkv.send_pack";
        }
        if (graph_cut_tensor_name_contains_any(node, {"_qkv_seq_to_head_output_"})) {
            return "qkv.recv_output_restore";
        }
        if (graph_cut_tensor_name_contains_any(node, {"_attn_head_to_seq_send_chunk"})) {
            return "head_to_seq.send_chunk";
        }
        if (graph_cut_tensor_name_contains_any(node, {"_attn_head_to_seq_send_flat"})) {
            return "head_to_seq.send_pack";
        }
        if (graph_cut_tensor_name_contains_any(node, {"_attn_head_to_seq_output"})) {
            return "head_to_seq.recv_output_restore";
        }
        if (graph_cut_tensor_name_contains_any(node, {"_txt_attn_head", "_img_attn_head", "_attn_4d"})) {
            return "attention.output_split_head";
        }
        if (graph_cut_tensor_name_contains_any(node, {"_attn_flat", "_attn_mlp"})) {
            return "single.tail.attn_mlp";
        }
        if (graph_cut_tensor_name_contains_any(node, {"_mlp_in", "_mlp_out"})) {
            return "block.mlp";
        }
        if (graph_cut_tensor_name_contains_any(node, {"_post_attn", "_after_attn"})) {
            return "double.post_attention";
        }
        if (graph_cut_tensor_name_contains_any(node, {"_q_attn", "_k_attn", "_v_attn"})) {
            if (trace.chain.find("CONCAT") != std::string::npos ||
                graph_cut_tensor_name_contains_any(primary_src, {"_txt_", "_img_"})) {
                return "double.attn_qkv_concat";
            }
            if (trace.chain.find("PERMUTE(view)->CONT->RESHAPE(view)->PERMUTE(view)->CONT") != std::string::npos) {
                return graph_cut_attention_qkv_layout_stage(node, "attention.rope_qk_layout");
            }
            if (trace.consumer_ops.find("MUL_MAT") != std::string::npos ||
                trace.consumer_ops.find("RMS_NORM") != std::string::npos) {
                return graph_cut_attention_qkv_layout_stage(node, "attention.qkv_input_layout");
            }
            return graph_cut_attention_qkv_layout_stage(node, "attention.qkv_layout");
        }
        if (trace.chain.find("CONCAT->PERMUTE(view)->CONT") != std::string::npos) {
            return "double.attn_qkv_concat";
        }
        if (trace.chain.find("PERMUTE(view)->CONT->RESHAPE(view)->PERMUTE(view)->CONT") != std::string::npos) {
            return "attention.rope_qk_layout";
        }
        if (trace.chain.find("PERMUTE(view)->CONT") != std::string::npos &&
            graph_cut_tensor_name_contains_any(primary_src, {"_v_attn"})) {
            return "attention.v_layout";
        }
        if (trace.chain.find("DUP->RESHAPE(view)->VIEW(view)->CONT") != std::string::npos) {
            return "qkv.recv_output_restore";
        }
        if (trace.chain.find("CONCAT->RESHAPE(view)->PERMUTE(view)->CONT") != std::string::npos) {
            return "send_pack_layout";
        }
        if (trace.chain.find("ADD(view)->VIEW(view)->CONT") != std::string::npos) {
            return "qkv.input_view_materialize";
        }
        if (trace.chain.find("CONCAT") != std::string::npos) {
            return "concat_layout";
        }
        if (trace.chain.find("PERMUTE(view)->CONT") != std::string::npos) {
            return "permute_to_cont_layout";
        }
        if (trace.dst_is_boundary_output) {
            return "boundary_output_materialize";
        }
        return "other";
    }

    static std::string graph_cut_materialize_stage_summary(const GraphCutMaterializeProfile& profile,
                                                           size_t limit = 12) {
        if (profile.stages.empty()) {
            return "-";
        }

        std::vector<std::pair<std::string, const GraphCutMaterializeStageProfile*>> stages;
        stages.reserve(profile.stages.size());
        for (const auto& entry : profile.stages) {
            stages.emplace_back(entry.first, &entry.second);
        }
        std::sort(stages.begin(), stages.end(), [](const auto& a, const auto& b) {
            if (a.second->bytes != b.second->bytes) {
                return a.second->bytes > b.second->bytes;
            }
            return a.first < b.first;
        });

        std::ostringstream oss;
        const size_t stage_limit = std::min(stages.size(), limit);
        for (size_t i = 0; i < stage_limit; ++i) {
            if (i > 0) {
                oss << " || ";
            }
            const auto& name = stages[i].first;
            const auto& s = *stages[i].second;
            oss << name
                << ":ops=" << s.ops
                << "/bytes=" << bytes_to_mib(s.bytes) << "MiB"
                << "/cont=" << s.cont_ops << "/" << bytes_to_mib(s.cont_bytes) << "MiB"
                << "/concat=" << s.concat_ops << "/" << bytes_to_mib(s.concat_bytes) << "MiB"
                << "/dup=" << s.dup_ops << "/" << bytes_to_mib(s.dup_bytes) << "MiB"
                << "/src_boundary=" << bytes_to_mib(s.src_boundary_input_bytes) << "MiB"
                << "/comm_in=" << bytes_to_mib(s.comm_input_bytes) << "MiB"
                << "/comm_out=" << bytes_to_mib(s.comm_output_bytes) << "MiB"
                << "/permute_view_to_cont=" << s.permute_view_to_cont_ops << "/"
                << bytes_to_mib(s.permute_view_to_cont_bytes) << "MiB"
                << "/cont_permute_cont=" << s.cont_permute_cont_ops << "/"
                << bytes_to_mib(s.cont_permute_cont_bytes) << "MiB";
        }
        if (stages.size() > stage_limit) {
            oss << " || ...";
        }
        return oss.str();
    }

    static std::string graph_cut_materialize_stage_chain_summary(const GraphCutMaterializeStageProfile& stage,
                                                                 size_t limit = 3) {
        if (stage.chain_groups.empty()) {
            return "-";
        }
        std::vector<std::pair<std::string, std::pair<size_t, size_t>>> chains;
        chains.reserve(stage.chain_groups.size());
        for (const auto& entry : stage.chain_groups) {
            chains.push_back(entry);
        }
        std::sort(chains.begin(), chains.end(), [](const auto& a, const auto& b) {
            if (a.second.second != b.second.second) {
                return a.second.second > b.second.second;
            }
            return a.second.first > b.second.first;
        });
        std::ostringstream oss;
        const size_t chain_limit = std::min(chains.size(), limit);
        for (size_t i = 0; i < chain_limit; ++i) {
            if (i > 0) {
                oss << " ; ";
            }
            oss << chains[i].first
                << ":" << chains[i].second.first
                << "/" << bytes_to_mib(chains[i].second.second) << "MiB";
        }
        return oss.str().empty() ? "-" : oss.str();
    }

    static std::string graph_cut_materialize_stage_detail_summary(const GraphCutMaterializeProfile& profile,
                                                                  size_t limit = 8) {
        if (profile.stages.empty()) {
            return "-";
        }
        std::vector<std::pair<std::string, const GraphCutMaterializeStageProfile*>> stages;
        stages.reserve(profile.stages.size());
        for (const auto& entry : profile.stages) {
            stages.emplace_back(entry.first, &entry.second);
        }
        std::sort(stages.begin(), stages.end(), [](const auto& a, const auto& b) {
            if (a.second->bytes != b.second->bytes) {
                return a.second->bytes > b.second->bytes;
            }
            return a.first < b.first;
        });

        std::ostringstream oss;
        const size_t stage_limit = std::min(stages.size(), limit);
        for (size_t i = 0; i < stage_limit; ++i) {
            if (i > 0) {
                oss << " || ";
            }
            const auto& name = stages[i].first;
            const auto& s = *stages[i].second;
            oss << "stage=" << name
                << "/ops=" << s.ops
                << "/bytes=" << bytes_to_mib(s.bytes) << "MiB"
                << "/chains=" << graph_cut_materialize_stage_chain_summary(s);
        }
        if (stages.size() > stage_limit) {
            oss << " || ...";
        }
        return oss.str();
    }

    static GraphCutMaterializeProfile graph_cut_segment_materialize_profile(ggml_cgraph* gf,
                                                                            const GraphCutSegment& segment,
                                                                            size_t top_limit = 8) {
        GraphCutMaterializeProfile profile;
        if (gf == nullptr) {
            return profile;
        }

        std::unordered_set<int> output_indices(segment.output_node_indices.begin(),
                                               segment.output_node_indices.end());
        std::unordered_set<int> comm_input_indices;
        std::unordered_set<int> comm_output_indices;
        for (const auto& comm_op : segment.comm_ops) {
            if (comm_op.input_node_index >= 0) {
                comm_input_indices.insert(comm_op.input_node_index);
            }
            if (comm_op.output_node_index >= 0) {
                comm_output_indices.insert(comm_op.output_node_index);
            }
        }
        std::unordered_set<const ggml_tensor*> previous_cut_inputs;
        for (const auto& input_ref : segment.input_refs) {
            if (input_ref.type == GraphCutSegment::INPUT_PREVIOUS_CUT) {
                ggml_tensor* input = sd::ggml_graph_cut::input_tensor(gf, input_ref);
                if (input != nullptr) {
                    previous_cut_inputs.insert(input);
                }
            }
        }

        std::vector<GraphCutMaterializeTrace> traces;
        std::map<const ggml_tensor*, std::pair<size_t, size_t>> source_repeats;
        std::map<std::string, std::pair<size_t, size_t>> chain_groups;
        for (int node_idx : segment.internal_node_indices) {
            if (node_idx < 0 || node_idx >= ggml_graph_n_nodes(gf)) {
                continue;
            }
            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            if (node == nullptr ||
                node->op < GGML_OP_NONE ||
                node->op >= GGML_OP_COUNT ||
                !graph_cut_op_is_materialize(node->op)) {
                continue;
            }

            GraphCutMaterializeTrace trace;
            trace.node_idx = node_idx;
            trace.op = node->op;
            trace.bytes = ggml_nbytes(node);
            trace.dst_is_boundary_output = output_indices.find(node_idx) != output_indices.end();
            trace.dst_is_cached_output = trace.dst_is_boundary_output &&
                                         node->name[0] != '\0' &&
                                         segment.future_input_names.find(node->name) != segment.future_input_names.end();
            trace.dst_is_comm_input = comm_input_indices.find(node_idx) != comm_input_indices.end();
            trace.dst_is_comm_output = comm_output_indices.find(node_idx) != comm_output_indices.end();
            trace.dst = graph_cut_tensor_layout_summary(node);
            trace.chain = graph_cut_materialize_chain_signature(node);

            ggml_tensor* primary_src = node->src[0] != nullptr ? node->src[0] : node->view_src;
            if (primary_src != nullptr) {
                trace.src = graph_cut_tensor_layout_summary(primary_src);
                trace.src_bytes = ggml_nbytes(primary_src);
                trace.src_is_view = primary_src->view_src != nullptr;
                trace.src_is_contiguous = ggml_is_contiguous(primary_src);
                trace.src_is_boundary_input = previous_cut_inputs.find(primary_src) != previous_cut_inputs.end();
                trace.src_is_cached_input = trace.src_is_boundary_input;
                trace.src_is_comm_output = false;
                for (int comm_output_idx : comm_output_indices) {
                    if (comm_output_idx >= 0 && comm_output_idx < ggml_graph_n_nodes(gf) &&
                        ggml_graph_node(gf, comm_output_idx) == primary_src) {
                        trace.src_is_comm_output = true;
                        break;
                    }
                }
                trace.producer_op = primary_src->op >= GGML_OP_NONE && primary_src->op < GGML_OP_COUNT ?
                                        ggml_op_name(primary_src->op) :
                                        "<invalid>";
                auto& repeat = source_repeats[primary_src];
                ++repeat.first;
                repeat.second += trace.bytes;

                if (graph_cut_op_is_materialize(primary_src->op)) {
                    trace.is_materialize_after_materialize = true;
                    ++profile.materialize_after_materialize_ops;
                    profile.materialize_after_materialize_bytes += trace.bytes;
                }
                if (node->op == GGML_OP_CONT && primary_src->op == GGML_OP_CONT) {
                    trace.is_cont_from_cont = true;
                    ++profile.cont_from_cont_ops;
                    profile.cont_from_cont_bytes += trace.bytes;
                }
                if (node->op == GGML_OP_CONT && primary_src->op == GGML_OP_CONCAT) {
                    trace.is_concat_to_cont = true;
                    ++profile.concat_to_cont_ops;
                    profile.concat_to_cont_bytes += trace.bytes;
                }
                if (node->op == GGML_OP_CONT &&
                    primary_src->op == GGML_OP_PERMUTE &&
                    primary_src->view_src != nullptr) {
                    trace.is_permute_view_to_cont = true;
                    ++profile.permute_view_to_cont_ops;
                    profile.permute_view_to_cont_bytes += trace.bytes;
                }
                if (primary_src->view_src != nullptr &&
                    graph_cut_op_is_materialize(primary_src->view_src->op)) {
                    trace.is_materialize_view_materialize = true;
                    ++profile.materialize_view_materialize_ops;
                    profile.materialize_view_materialize_bytes += trace.bytes;
                    if (node->op == GGML_OP_CONT &&
                        primary_src->op == GGML_OP_PERMUTE &&
                        primary_src->view_src->op == GGML_OP_CONT) {
                        trace.is_cont_permute_cont = true;
                        ++profile.cont_permute_cont_ops;
                        profile.cont_permute_cont_bytes += trace.bytes;
                    }
                }
            } else {
                trace.src = "<null>";
                trace.producer_op = "<none>";
            }

            if (node->op == GGML_OP_CONCAT) {
                std::ostringstream concat;
                size_t emitted = 0;
                for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                    if (node->src[src_idx] == nullptr) {
                        continue;
                    }
                    graph_cut_append_limited(concat,
                                             graph_cut_tensor_layout_summary(node->src[src_idx]),
                                             emitted,
                                             4,
                                             "|");
                }
                trace.concat_inputs = concat.str();
            }

            const auto consumers = graph_cut_tensor_consumers(gf, segment, node);
            trace.consumer_ops = graph_cut_consumer_histogram(consumers);

            ++profile.ops;
            profile.bytes += trace.bytes;
            if (trace.dst_is_boundary_output) {
                profile.boundary_output_bytes += trace.bytes;
            }
            if (trace.dst_is_cached_output) {
                profile.cached_output_bytes += trace.bytes;
            }
            if (trace.dst_is_comm_input) {
                profile.comm_input_bytes += trace.bytes;
            }
            if (trace.dst_is_comm_output) {
                profile.comm_output_bytes += trace.bytes;
            }
            switch (node->op) {
                case GGML_OP_CONT:
                    ++profile.cont_ops;
                    profile.cont_bytes += trace.bytes;
                    break;
                case GGML_OP_CPY:
                    ++profile.cpy_ops;
                    profile.cpy_bytes += trace.bytes;
                    break;
                case GGML_OP_CONCAT:
                    ++profile.concat_ops;
                    profile.concat_bytes += trace.bytes;
                    break;
                case GGML_OP_DUP:
                    ++profile.dup_ops;
                    profile.dup_bytes += trace.bytes;
                    break;
                default:
                    break;
            }
            if (trace.chain.find("->") != std::string::npos) {
                auto& chain_group = chain_groups[trace.chain];
                ++chain_group.first;
                chain_group.second += trace.bytes;
            }
            const std::string stage =
                graph_cut_materialize_stage_from_trace(trace, node, primary_src);
            graph_cut_accumulate_materialize_stage(profile.stages[stage], trace);
            traces.push_back(std::move(trace));
        }

        for (const auto& entry : source_repeats) {
            if (entry.second.first > 1) {
                ++profile.repeated_source_groups;
                profile.repeated_source_ops += entry.second.first;
                profile.repeated_source_bytes += entry.second.second;
            }
        }

        std::sort(traces.begin(), traces.end(), [](const auto& a, const auto& b) {
            if (a.bytes != b.bytes) {
                return a.bytes > b.bytes;
            }
            return a.node_idx < b.node_idx;
        });
        std::ostringstream top;
        const size_t trace_limit = std::min(traces.size(), top_limit);
        for (size_t i = 0; i < trace_limit; ++i) {
            const auto& t = traces[i];
            if (i > 0) {
                top << " || ";
            }
            top << "#idx=" << t.node_idx
                << "/op=" << ggml_op_name(t.op)
                << "/bytes=" << bytes_to_mib(t.bytes) << "MiB"
                << "/dst=" << t.dst
                << "/src_op=" << t.producer_op
                << "/src=" << t.src
                << "/src_view=" << (t.src_is_view ? "yes" : "no")
                << "/src_contig=" << (t.src_is_contiguous ? "yes" : "no")
                << "/consumers=" << t.consumer_ops
                << "/flags="
                << (t.dst_is_boundary_output ? "boundary_out," : "")
                << (t.dst_is_cached_output ? "cached_out," : "")
                << (t.dst_is_comm_input ? "comm_in," : "")
                << (t.dst_is_comm_output ? "comm_out," : "")
                << (t.src_is_boundary_input ? "src_boundary_in," : "")
                << (t.src_is_comm_output ? "src_comm_out," : "");
            if (!t.concat_inputs.empty()) {
                top << "/concat_inputs=" << t.concat_inputs;
            }
            top << "/chain=" << t.chain;
        }
        profile.top_nodes = top.str().empty() ? "-" : top.str();

        std::vector<std::pair<const ggml_tensor*, std::pair<size_t, size_t>>> repeated;
        repeated.reserve(source_repeats.size());
        for (const auto& entry : source_repeats) {
            if (entry.second.first > 1) {
                repeated.push_back(entry);
            }
        }
        std::sort(repeated.begin(), repeated.end(), [](const auto& a, const auto& b) {
            if (a.second.second != b.second.second) {
                return a.second.second > b.second.second;
            }
            return a.second.first > b.second.first;
        });
        std::ostringstream repeated_oss;
        const size_t repeat_limit = std::min<size_t>(repeated.size(), 6);
        for (size_t i = 0; i < repeat_limit; ++i) {
            if (i > 0) {
                repeated_oss << " || ";
            }
            repeated_oss << "src=" << graph_cut_tensor_layout_summary(repeated[i].first)
                         << "/ops=" << repeated[i].second.first
                         << "/bytes=" << bytes_to_mib(repeated[i].second.second) << "MiB";
        }
        profile.repeated_sources = repeated_oss.str().empty() ? "-" : repeated_oss.str();

        std::vector<std::pair<std::string, std::pair<size_t, size_t>>> chains;
        chains.reserve(chain_groups.size());
        for (const auto& entry : chain_groups) {
            chains.push_back(entry);
        }
        std::sort(chains.begin(), chains.end(), [](const auto& a, const auto& b) {
            if (a.second.second != b.second.second) {
                return a.second.second > b.second.second;
            }
            return a.second.first > b.second.first;
        });
        std::ostringstream chains_oss;
        const size_t chain_limit = std::min<size_t>(chains.size(), 8);
        for (size_t i = 0; i < chain_limit; ++i) {
            if (i > 0) {
                chains_oss << " || ";
            }
            chains_oss << "chain=" << chains[i].first
                       << "/ops=" << chains[i].second.first
                       << "/bytes=" << bytes_to_mib(chains[i].second.second) << "MiB";
        }
        profile.top_chains = chains_oss.str().empty() ? "-" : chains_oss.str();
        return profile;
    }

    static std::string graph_cut_segment_op_histogram(ggml_cgraph* gf,
                                                      const GraphCutSegment& segment,
                                                      std::array<size_t, GGML_OP_COUNT>* op_counts,
                                                      size_t* math_ops,
                                                      size_t* layout_ops,
                                                      size_t limit = 12) {
        std::array<size_t, GGML_OP_COUNT> local_counts = {};
        size_t local_math_ops = 0;
        size_t local_layout_ops = 0;
        for (int node_idx : segment.internal_node_indices) {
            if (node_idx < 0 || node_idx >= ggml_graph_n_nodes(gf)) {
                continue;
            }
            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            if (node == nullptr ||
                node->op < GGML_OP_NONE ||
                node->op >= GGML_OP_COUNT) {
                continue;
            }
            const auto op = node->op;
            ++local_counts[static_cast<size_t>(op)];
            if (graph_cut_op_is_math(op)) {
                ++local_math_ops;
            }
            if (graph_cut_op_is_layout(op)) {
                ++local_layout_ops;
            }
        }

        if (op_counts != nullptr) {
            *op_counts = local_counts;
        }
        if (math_ops != nullptr) {
            *math_ops = local_math_ops;
        }
        if (layout_ops != nullptr) {
            *layout_ops = local_layout_ops;
        }

        std::vector<size_t> order;
        order.reserve(GGML_OP_COUNT);
        for (size_t i = 0; i < GGML_OP_COUNT; ++i) {
            if (local_counts[i] > 0) {
                order.push_back(i);
            }
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (local_counts[a] != local_counts[b]) {
                return local_counts[a] > local_counts[b];
            }
            return a < b;
        });

        std::ostringstream oss;
        const size_t hist_limit = std::min(order.size(), limit);
        for (size_t i = 0; i < hist_limit; ++i) {
            if (i > 0) {
                oss << ",";
            }
            const auto op = static_cast<enum ggml_op>(order[i]);
            oss << ggml_op_name(op) << ":" << local_counts[order[i]];
        }
        if (order.size() > hist_limit) {
            oss << ",...";
        }
        if (oss.str().empty()) {
            return "-";
        }
        return oss.str();
    }

    static std::string graph_cut_segment_compute_bucket(const GraphCutSegmentProfile& segment) {
        const std::string& name = segment.name;
        const std::string& comm = segment.comm_names;
        if (name.rfind("sp:flux_double", 0) == 0 ||
            comm.find("flux_double") != std::string::npos) {
            if (comm.find("qkv_seq_to_head") != std::string::npos) {
                return "sp.double.qkv_seq_to_head";
            }
            if (comm.find("head_to_seq") != std::string::npos) {
                return "sp.double.attn_head_to_seq";
            }
            return "sp.double.other";
        }
        if (name.rfind("sp:flux_single", 0) == 0 ||
            comm.find("flux_single") != std::string::npos) {
            if (comm.find("qkv_seq_to_head") != std::string::npos) {
                return "sp.single.qkv_seq_to_head";
            }
            if (comm.find("head_to_seq") != std::string::npos) {
                return "sp.single.attn_head_to_seq";
            }
            return "sp.single.other";
        }
        if (graph_cut_segment_is_double_block_output(name)) {
            return "flux.double_blocks.tail";
        }
        if (graph_cut_segment_is_single_block_output(name)) {
            return "flux.single_blocks.tail";
        }
        if (comm.find("gather") != std::string::npos) {
            return "sp.gather";
        }
        if (name.rfind("flux.prelude", 0) == 0) {
            return "flux.prelude";
        }
        return "other";
    }

    struct GraphCutComputeBucketProfile {
        size_t segments = 0;
        size_t nodes = 0;
        size_t math_ops = 0;
        size_t layout_ops = 0;
        size_t comm_ops = 0;
        size_t comm_bytes = 0;
        size_t materialize_ops = 0;
        size_t materialize_bytes = 0;
        size_t cont_ops = 0;
        size_t cont_bytes = 0;
        size_t cpy_ops = 0;
        size_t cpy_bytes = 0;
        size_t concat_ops = 0;
        size_t concat_bytes = 0;
        size_t dup_ops = 0;
        size_t dup_bytes = 0;
        size_t materialize_boundary_output_bytes = 0;
        size_t materialize_cached_output_bytes = 0;
        size_t materialize_comm_input_bytes = 0;
        size_t materialize_comm_output_bytes = 0;
        size_t repeated_materialize_source_groups = 0;
        size_t repeated_materialize_source_ops = 0;
        size_t repeated_materialize_source_bytes = 0;
        size_t materialize_after_materialize_ops = 0;
        size_t materialize_after_materialize_bytes = 0;
        size_t cont_from_cont_ops = 0;
        size_t cont_from_cont_bytes = 0;
        size_t concat_to_cont_ops = 0;
        size_t concat_to_cont_bytes = 0;
        size_t permute_view_to_cont_ops = 0;
        size_t permute_view_to_cont_bytes = 0;
        size_t materialize_view_materialize_ops = 0;
        size_t materialize_view_materialize_bytes = 0;
        size_t cont_permute_cont_ops = 0;
        size_t cont_permute_cont_bytes = 0;
        std::map<std::string, GraphCutMaterializeStageProfile> materialize_stages;
        int64_t total_ms = 0;
        int64_t compute_ms = 0;
        int64_t comm_ms = 0;
        int64_t cache_ms = 0;
        int64_t copy_ms = 0;
        std::array<size_t, GGML_OP_COUNT> op_counts = {};
    };

    static void graph_cut_merge_materialize_stage_profile(GraphCutMaterializeStageProfile& dst,
                                                          const GraphCutMaterializeStageProfile& src) {
        dst.ops += src.ops;
        dst.bytes += src.bytes;
        dst.cont_ops += src.cont_ops;
        dst.cont_bytes += src.cont_bytes;
        dst.cpy_ops += src.cpy_ops;
        dst.cpy_bytes += src.cpy_bytes;
        dst.concat_ops += src.concat_ops;
        dst.concat_bytes += src.concat_bytes;
        dst.dup_ops += src.dup_ops;
        dst.dup_bytes += src.dup_bytes;
        dst.boundary_output_bytes += src.boundary_output_bytes;
        dst.cached_output_bytes += src.cached_output_bytes;
        dst.comm_input_bytes += src.comm_input_bytes;
        dst.comm_output_bytes += src.comm_output_bytes;
        dst.src_boundary_input_bytes += src.src_boundary_input_bytes;
        dst.src_comm_output_bytes += src.src_comm_output_bytes;
        dst.materialize_after_materialize_ops += src.materialize_after_materialize_ops;
        dst.materialize_after_materialize_bytes += src.materialize_after_materialize_bytes;
        dst.cont_from_cont_ops += src.cont_from_cont_ops;
        dst.cont_from_cont_bytes += src.cont_from_cont_bytes;
        dst.concat_to_cont_ops += src.concat_to_cont_ops;
        dst.concat_to_cont_bytes += src.concat_to_cont_bytes;
        dst.permute_view_to_cont_ops += src.permute_view_to_cont_ops;
        dst.permute_view_to_cont_bytes += src.permute_view_to_cont_bytes;
        dst.materialize_view_materialize_ops += src.materialize_view_materialize_ops;
        dst.materialize_view_materialize_bytes += src.materialize_view_materialize_bytes;
        dst.cont_permute_cont_ops += src.cont_permute_cont_ops;
        dst.cont_permute_cont_bytes += src.cont_permute_cont_bytes;
        for (const auto& chain : src.chain_groups) {
            auto& dst_chain = dst.chain_groups[chain.first];
            dst_chain.first += chain.second.first;
            dst_chain.second += chain.second.second;
        }
    }

    static void graph_cut_merge_materialize_stage_maps(std::map<std::string, GraphCutMaterializeStageProfile>& dst,
                                                       const std::map<std::string, GraphCutMaterializeStageProfile>& src) {
        for (const auto& entry : src) {
            graph_cut_merge_materialize_stage_profile(dst[entry.first], entry.second);
        }
    }

    static std::string graph_cut_bucket_op_histogram(const GraphCutComputeBucketProfile& bucket,
                                                     size_t limit = 10) {
        std::vector<size_t> order;
        order.reserve(GGML_OP_COUNT);
        for (size_t i = 0; i < GGML_OP_COUNT; ++i) {
            if (bucket.op_counts[i] > 0) {
                order.push_back(i);
            }
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (bucket.op_counts[a] != bucket.op_counts[b]) {
                return bucket.op_counts[a] > bucket.op_counts[b];
            }
            return a < b;
        });

        std::ostringstream oss;
        const size_t hist_limit = std::min(order.size(), limit);
        for (size_t i = 0; i < hist_limit; ++i) {
            if (i > 0) {
                oss << ",";
            }
            const auto op = static_cast<enum ggml_op>(order[i]);
            oss << ggml_op_name(op) << ":" << bucket.op_counts[order[i]];
        }
        if (order.size() > hist_limit) {
            oss << ",...";
        }
        if (oss.str().empty()) {
            return "-";
        }
        return oss.str();
    }

    static bool runtime_const_cache_tensor_shape_matches(const ggml_tensor* a,
                                                         const RuntimeConstCacheEntry& entry) {
        if (a == nullptr || a->type != entry.type || ggml_nbytes(a) != entry.nbytes) {
            return false;
        }
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (a->ne[i] != entry.ne[i]) {
                return false;
            }
        }
        return true;
    }

    bool should_cache_runtime_const_tensor(const ggml_tensor* tensor,
                                           const void* data) const {
        if (!runtime_const_cache_enabled() || tensor == nullptr || data == nullptr) {
            return false;
        }
        if (tensor->view_src != nullptr || tensor->name[0] == '\0') {
            return false;
        }
        if (tensor->type != GGML_TYPE_F32) {
            return false;
        }

        const std::string name(tensor->name);
        if (name == "ggml_runner_build_in_tensor:one" ||
            name == "ggml_runner_build_in_tensor:zero_int") {
            return false;
        }

        // Flux RoPE position embeddings are generated once for a denoise step
        // and then read by many SP graph-cut segments.  Caching the uploaded
        // device tensor fixes repeated runtime-input upload; it does not change
        // RoPE math or operator placement.
        const size_t nbytes = ggml_nbytes(tensor);
        if (name == "pe" && nbytes >= 1024 * 1024) {
            return true;
        }

        // Wan text context is the same external tensor for every transformer
        // block in one SP graph-cut forward. Upload it once, then bind the
        // later segment inputs to the cached device tensor.
        if (name == "wan.context" && nbytes >= 1024 * 1024) {
            return process_group_ != nullptr && process_group_->enabled();
        }

        return false;
    }

    std::string runtime_const_cache_key(const ggml_tensor* tensor,
                                        const void* data) const {
        std::ostringstream oss;
        oss << ggml_backend_name(runtime_backend)
            << "|" << tensor->name
            << "|data=" << data
            << "|type=" << static_cast<int>(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            oss << "|ne" << i << "=" << tensor->ne[i];
        }
        return oss.str();
    }

    void bind_tensor_to_runtime_const_cache(ggml_tensor* tensor,
                                            RuntimeConstCacheEntry& entry) {
        GGML_ASSERT(tensor != nullptr);
        GGML_ASSERT(entry.tensor != nullptr);
        tensor->buffer = entry.tensor->buffer;
        tensor->data   = entry.tensor->data;
        tensor->extra  = entry.tensor->extra;
    }

    RuntimeConstCacheEntry* find_runtime_const_cache_entry(const ggml_tensor* tensor,
                                                           const void* data) {
        const std::string key = runtime_const_cache_key(tensor, data);
        for (auto& entry : runtime_const_cache_) {
            if (entry.tensor == nullptr || entry.key.empty()) {
                continue;
            }
            if (key == entry.key &&
                entry.host_data == data &&
                runtime_const_cache_tensor_shape_matches(tensor, entry)) {
                return &entry;
            }
        }
        return nullptr;
    }

    bool runtime_const_cache_entry_matches_tensor_slot(const ggml_tensor* tensor,
                                                       const RuntimeConstCacheEntry& entry) const {
        if (tensor == nullptr || entry.tensor == nullptr) {
            return false;
        }
        if (std::strcmp(tensor->name, entry.tensor->name) != 0) {
            return false;
        }
        return runtime_const_cache_tensor_shape_matches(tensor, entry);
    }

    size_t runtime_const_cache_slot_entry_limit(const ggml_tensor* tensor) const {
        if (tensor == nullptr || tensor->name[0] == '\0') {
            return 1;
        }

        // Wan alternates cond/uncond text contexts during CFG sampling. Both
        // tensors are stable across denoise steps, so keep both device uploads
        // instead of pruning one when the other branch runs.
        if (std::strcmp(tensor->name, "wan.context") == 0) {
            return 2;
        }

        return 1;
    }

    void prune_stale_runtime_const_cache_entries(const ggml_tensor* tensor,
                                                 const void* data) {
        const size_t slot_limit = runtime_const_cache_slot_entry_limit(tensor);
        size_t slot_entries = 0;
        for (const auto& entry : runtime_const_cache_) {
            if (runtime_const_cache_entry_matches_tensor_slot(tensor, entry)) {
                ++slot_entries;
            }
        }
        if (slot_entries < slot_limit) {
            return;
        }

        runtime_const_cache_.erase(std::remove_if(runtime_const_cache_.begin(),
                                                  runtime_const_cache_.end(),
                                                  [&](const RuntimeConstCacheEntry& entry) {
                                                      return entry.host_data != data &&
                                                             runtime_const_cache_entry_matches_tensor_slot(tensor, entry);
                                                  }),
                                   runtime_const_cache_.end());
    }

    RuntimeConstCacheEntry* upload_runtime_const_cache_entry(ggml_tensor* tensor,
                                                             const void* data) {
        prune_stale_runtime_const_cache_entries(tensor, data);

        RuntimeConstCacheEntry entry;
        entry.host_data = data;
        entry.type = tensor->type;
        entry.nbytes = ggml_nbytes(tensor);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            entry.ne[i] = tensor->ne[i];
        }

        entry.ctx = new_cache_context(1);
        entry.tensor = ggml_dup_tensor(entry.ctx, tensor);
        entry.key = runtime_const_cache_key(tensor, data);
        ggml_set_name(entry.tensor, tensor->name);
        entry.buffer = ggml_backend_alloc_ctx_tensors(entry.ctx, runtime_backend);
        if (entry.buffer == nullptr) {
            LOG_WARN("%s runtime const cache alloc failed: tensor=%s bytes=%.2fMiB",
                     get_desc().c_str(),
                     tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                     bytes_to_mib(entry.nbytes));
            return nullptr;
        }
        ggml_backend_tensor_set(entry.tensor, data, 0, entry.nbytes);

        runtime_const_cache_.push_back(std::move(entry));
        return &runtime_const_cache_.back();
    }

    void free_runtime_const_cache() {
        runtime_const_cache_.clear();
    }

    void refresh_graph_view_bindings(ggml_cgraph* gf) {
        GGML_ASSERT(gf != nullptr);
        const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
        for (int i = 0; i < n_leafs; ++i) {
            ggml_tensor* leaf = sd::ggml_graph_cut::leaf_tensor(gf, i);
            if (leaf != nullptr &&
                leaf->view_src != nullptr &&
                leaf->view_src->data != nullptr) {
                leaf->buffer = leaf->view_src->buffer;
                leaf->data   = static_cast<void*>(static_cast<char*>(leaf->view_src->data) + leaf->view_offs);
                leaf->extra  = leaf->view_src->extra;
            }
        }
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (node != nullptr &&
                node->view_src != nullptr &&
                node->view_src->data != nullptr) {
                node->buffer = node->view_src->buffer;
                node->data   = static_cast<void*>(static_cast<char*>(node->view_src->data) + node->view_offs);
                node->extra  = node->view_src->extra;
            }
        }
    }

    void log_graph_cut_profile(const std::vector<GraphCutSegmentProfile>& segments,
                               int64_t plan_ms,
                               int64_t total_ms) {
        if (segments.empty() || !graph_cut_profile_should_log_rank()) {
            return;
        }

        int rank = 0;
        int world_size = 1;
        if (process_group_ != nullptr && process_group_->enabled()) {
            rank = process_group_->rank();
            world_size = process_group_->size();
        }

        size_t comm_segments = 0;
        size_t comm_ops = 0;
        size_t comm_bytes = 0;
        size_t qkv_seq_to_head_ops = 0;
        size_t q_seq_to_head_ops = 0;
        size_t k_seq_to_head_ops = 0;
        size_t v_seq_to_head_ops = 0;
        size_t txt_img_head_to_seq_ops = 0;
        size_t head_to_seq_ops = 0;
        size_t gather_ops = 0;
        size_t sp_attn_materialize_segments = 0;
        size_t output_bytes = 0;
        size_t cached_output_bytes = 0;
        size_t math_ops = 0;
        size_t layout_ops = 0;
        size_t double_block_cached_bytes = 0;
        size_t single_block_cached_bytes = 0;
        size_t other_cached_bytes = 0;
        size_t peak_cache_live_bytes = 0;
        size_t peak_cache_buffer_bytes = 0;
        size_t peak_cache_chunks = 0;
        size_t peak_cache_pool_bytes = 0;
        size_t peak_cache_pool_chunks = 0;
        int64_t build_ms = 0;
        int64_t runtime_param_ms = 0;
        int64_t offload_ms = 0;
        int64_t alloc_ms = 0;
        int64_t copy_ms = 0;
        int64_t compute_ms = 0;
        int64_t comm_ms = 0;
        int64_t cache_ms = 0;
        int64_t double_block_cache_ms = 0;
        int64_t single_block_cache_ms = 0;
        int64_t other_cache_ms = 0;
        int64_t collect_future_inputs_us = 0;
        int64_t reset_runtime_tensors_us = 0;
        int64_t bind_cached_inputs_us = 0;
        int64_t mark_cache_outputs_us = 0;
        int64_t pre_compute_callback_us = 0;
        int64_t segment_graph_free_us = 0;
        int64_t copy_graph_tensor_set_us = 0;
        int64_t copy_backend_map_scan_us = 0;
        int64_t copy_backend_tensor_set_us = 0;
        size_t copy_graph_tensor_entries = 0;
        size_t copy_backend_map_entries = 0;
        size_t copied_tensors = 0;
        size_t skipped_not_in_graph = 0;
        size_t skipped_no_buffer = 0;
        size_t copied_bytes = 0;
        size_t runtime_const_cache_hits = 0;
        size_t runtime_const_cache_uploads = 0;
        size_t runtime_const_cache_hit_bytes = 0;
        size_t runtime_const_cache_upload_bytes = 0;
        int64_t runtime_const_cache_hit_us = 0;
        int64_t runtime_const_cache_upload_us = 0;
        int64_t segment_total_ms = 0;
        const int compute_top_n = graph_cut_profile_compute_top_n();
        const int materialize_top_n = graph_cut_profile_materialize_top_n();
        const bool collect_materialize_breakdown = materialize_top_n > 0;
        const bool collect_compute_breakdown = compute_top_n > 0 || collect_materialize_breakdown;
        std::map<std::string, GraphCutComputeBucketProfile> compute_buckets;

        for (const auto& segment : segments) {
            if (segment.comm_ops > 0) {
                ++comm_segments;
            }
            if (segment.name.find("sp_attn_q") != std::string::npos ||
                segment.name.find("sp_attn_k") != std::string::npos ||
                segment.name.find("sp_attn_v") != std::string::npos) {
                ++sp_attn_materialize_segments;
            }
            comm_ops += segment.comm_ops;
            comm_bytes += segment.comm_bytes;
            if (segment.comm_names.find("qkv_seq_to_head") != std::string::npos) {
                ++qkv_seq_to_head_ops;
            }
            if (segment.comm_names.find("_q_seq_to_head") != std::string::npos &&
                segment.comm_names.find("qkv_seq_to_head") == std::string::npos) {
                ++q_seq_to_head_ops;
            }
            if (segment.comm_names.find("_k_seq_to_head") != std::string::npos &&
                segment.comm_names.find("qkv_seq_to_head") == std::string::npos) {
                ++k_seq_to_head_ops;
            }
            if (segment.comm_names.find("_v_seq_to_head") != std::string::npos &&
                segment.comm_names.find("qkv_seq_to_head") == std::string::npos) {
                ++v_seq_to_head_ops;
            }
            if (segment.comm_names.find("txt_img_attn_head_to_seq") != std::string::npos) {
                ++txt_img_head_to_seq_ops;
            }
            if (segment.comm_names.find("head_to_seq") != std::string::npos) {
                ++head_to_seq_ops;
            }
            if (segment.comm_names.find("gather") != std::string::npos) {
                ++gather_ops;
            }
            output_bytes += segment.output_bytes;
            cached_output_bytes += segment.cached_output_bytes;
            math_ops += segment.math_ops;
            layout_ops += segment.layout_ops;
            if (graph_cut_segment_is_double_block_output(segment.name)) {
                double_block_cached_bytes += segment.cached_output_bytes;
                double_block_cache_ms += segment.cache_ms;
            } else if (graph_cut_segment_is_single_block_output(segment.name)) {
                single_block_cached_bytes += segment.cached_output_bytes;
                single_block_cache_ms += segment.cache_ms;
            } else {
                other_cached_bytes += segment.cached_output_bytes;
                other_cache_ms += segment.cache_ms;
            }
            peak_cache_live_bytes = std::max(peak_cache_live_bytes, segment.cache_live_bytes);
            peak_cache_buffer_bytes = std::max(peak_cache_buffer_bytes, segment.cache_buffer_bytes);
            peak_cache_chunks = std::max(peak_cache_chunks, segment.cache_chunks);
            peak_cache_pool_bytes = std::max(peak_cache_pool_bytes, segment.cache_pool_bytes);
            peak_cache_pool_chunks = std::max(peak_cache_pool_chunks, segment.cache_pool_chunks);
            build_ms += segment.build_ms;
            runtime_param_ms += segment.runtime_param_ms;
            offload_ms += segment.offload_ms;
            alloc_ms += segment.alloc_ms;
            copy_ms += segment.copy_ms;
            collect_future_inputs_us += segment.collect_future_inputs_us;
            reset_runtime_tensors_us += segment.reset_runtime_tensors_us;
            bind_cached_inputs_us += segment.bind_cached_inputs_us;
            mark_cache_outputs_us += segment.mark_cache_outputs_us;
            pre_compute_callback_us += segment.pre_compute_callback_us;
            segment_graph_free_us += segment.segment_graph_free_us;
            copy_graph_tensor_set_us += segment.copy_detail.graph_tensor_set_us;
            copy_backend_map_scan_us += segment.copy_detail.backend_map_scan_us;
            copy_backend_tensor_set_us += segment.copy_detail.backend_tensor_set_us;
            copy_graph_tensor_entries += segment.copy_detail.graph_tensor_entries;
            copy_backend_map_entries += segment.copy_detail.backend_map_entries;
            copied_tensors += segment.copy_detail.copied_tensors;
            skipped_not_in_graph += segment.copy_detail.skipped_not_in_graph;
            skipped_no_buffer += segment.copy_detail.skipped_no_buffer;
            copied_bytes += segment.copy_detail.copied_bytes;
            runtime_const_cache_hits += segment.copy_detail.runtime_const_cache_hits;
            runtime_const_cache_uploads += segment.copy_detail.runtime_const_cache_uploads;
            runtime_const_cache_hit_bytes += segment.copy_detail.runtime_const_cache_hit_bytes;
            runtime_const_cache_upload_bytes += segment.copy_detail.runtime_const_cache_upload_bytes;
            runtime_const_cache_hit_us += segment.copy_detail.runtime_const_cache_hit_us;
            runtime_const_cache_upload_us += segment.copy_detail.runtime_const_cache_upload_us;
            compute_ms += segment.compute_ms;
            comm_ms += segment.comm_ms;
            cache_ms += segment.cache_ms;
            segment_total_ms += segment.total_ms;

            if (collect_compute_breakdown) {
                auto& bucket = compute_buckets[graph_cut_segment_compute_bucket(segment)];
                ++bucket.segments;
                bucket.nodes += segment.nodes;
                bucket.math_ops += segment.math_ops;
                bucket.layout_ops += segment.layout_ops;
                bucket.comm_ops += segment.comm_ops;
                bucket.comm_bytes += segment.comm_bytes;
                bucket.materialize_ops += segment.materialize_ops;
                bucket.materialize_bytes += segment.materialize_bytes;
                bucket.cont_ops += segment.cont_ops;
                bucket.cont_bytes += segment.cont_bytes;
                bucket.cpy_ops += segment.cpy_ops;
                bucket.cpy_bytes += segment.cpy_bytes;
                bucket.concat_ops += segment.concat_ops;
                bucket.concat_bytes += segment.concat_bytes;
                bucket.dup_ops += segment.dup_ops;
                bucket.dup_bytes += segment.dup_bytes;
                bucket.materialize_boundary_output_bytes += segment.materialize_boundary_output_bytes;
                bucket.materialize_cached_output_bytes += segment.materialize_cached_output_bytes;
                bucket.materialize_comm_input_bytes += segment.materialize_comm_input_bytes;
                bucket.materialize_comm_output_bytes += segment.materialize_comm_output_bytes;
                bucket.repeated_materialize_source_groups += segment.repeated_materialize_source_groups;
                bucket.repeated_materialize_source_ops += segment.repeated_materialize_source_ops;
                bucket.repeated_materialize_source_bytes += segment.repeated_materialize_source_bytes;
                bucket.materialize_after_materialize_ops += segment.materialize_after_materialize_ops;
                bucket.materialize_after_materialize_bytes += segment.materialize_after_materialize_bytes;
                bucket.cont_from_cont_ops += segment.cont_from_cont_ops;
                bucket.cont_from_cont_bytes += segment.cont_from_cont_bytes;
                bucket.concat_to_cont_ops += segment.concat_to_cont_ops;
                bucket.concat_to_cont_bytes += segment.concat_to_cont_bytes;
                bucket.permute_view_to_cont_ops += segment.permute_view_to_cont_ops;
                bucket.permute_view_to_cont_bytes += segment.permute_view_to_cont_bytes;
                bucket.materialize_view_materialize_ops += segment.materialize_view_materialize_ops;
                bucket.materialize_view_materialize_bytes += segment.materialize_view_materialize_bytes;
                bucket.cont_permute_cont_ops += segment.cont_permute_cont_ops;
                bucket.cont_permute_cont_bytes += segment.cont_permute_cont_bytes;
                graph_cut_merge_materialize_stage_maps(bucket.materialize_stages,
                                                       segment.materialize_stages);
                bucket.total_ms += segment.total_ms;
                bucket.compute_ms += segment.compute_ms;
                bucket.comm_ms += segment.comm_ms;
                bucket.cache_ms += segment.cache_ms;
                bucket.copy_ms += segment.copy_ms;
                for (size_t op_idx = 0; op_idx < bucket.op_counts.size(); ++op_idx) {
                    bucket.op_counts[op_idx] += segment.op_counts[op_idx];
                }
            }
        }

        const int64_t known_ms = plan_ms + build_ms + runtime_param_ms + offload_ms +
                                 alloc_ms + copy_ms + compute_ms + comm_ms + cache_ms;
        const int64_t other_ms = std::max<int64_t>(0, total_ms - known_ms);

        ++graph_cut_profile_index_;
        LOG_INFO("%s graph cut profile #%zu rank=%d/%d segments=%zu comm_segments=%zu comm_ops=%zu comm_bytes=%.2fMiB total=%lldms plan=%lldms build=%lldms params=%lldms offload=%lldms alloc=%lldms copy=%lldms compute=%lldms comm=%lldms cache=%lldms other=%lldms segment_total=%lldms",
                 get_desc().c_str(),
                 graph_cut_profile_index_,
                 rank,
                 world_size,
                 segments.size(),
                 comm_segments,
                 comm_ops,
                 bytes_to_mib(comm_bytes),
                 static_cast<long long>(total_ms),
                 static_cast<long long>(plan_ms),
                 static_cast<long long>(build_ms),
                 static_cast<long long>(runtime_param_ms),
                 static_cast<long long>(offload_ms),
                 static_cast<long long>(alloc_ms),
                 static_cast<long long>(copy_ms),
                 static_cast<long long>(compute_ms),
                 static_cast<long long>(comm_ms),
                 static_cast<long long>(cache_ms),
                 static_cast<long long>(other_ms),
                 static_cast<long long>(segment_total_ms));

        LOG_INFO("%s graph cut profile #%zu cache-summary output_bytes=%.2fMiB cached_output_bytes=%.2fMiB double_block_cache=%.2fMiB/%lldms single_block_cache=%.2fMiB/%lldms other_cache=%.2fMiB/%lldms",
                 get_desc().c_str(),
                 graph_cut_profile_index_,
                 bytes_to_mib(output_bytes),
                 bytes_to_mib(cached_output_bytes),
                 bytes_to_mib(double_block_cached_bytes),
                 static_cast<long long>(double_block_cache_ms),
                 bytes_to_mib(single_block_cached_bytes),
                 static_cast<long long>(single_block_cache_ms),
                 bytes_to_mib(other_cached_bytes),
                 static_cast<long long>(other_cache_ms));

        LOG_INFO("%s graph cut profile #%zu cache-live peak_live=%.2fMiB peak_buffer=%.2fMiB peak_chunks=%zu peak_pool=%.2fMiB pool_chunks=%zu",
                 get_desc().c_str(),
                 graph_cut_profile_index_,
                 bytes_to_mib(peak_cache_live_bytes),
                 bytes_to_mib(peak_cache_buffer_bytes),
                 peak_cache_chunks,
                 bytes_to_mib(peak_cache_pool_bytes),
                 peak_cache_pool_chunks);

        LOG_INFO("%s graph cut profile #%zu comm-name-summary qkv_seq_to_head=%zu q_seq_to_head=%zu k_seq_to_head=%zu v_seq_to_head=%zu txt_img_attn_head_to_seq=%zu head_to_seq_total=%zu gather=%zu sp_attn_materialize_segments=%zu",
                 get_desc().c_str(),
                 graph_cut_profile_index_,
                 qkv_seq_to_head_ops,
                 q_seq_to_head_ops,
                 k_seq_to_head_ops,
                 v_seq_to_head_ops,
                 txt_img_head_to_seq_ops,
                 head_to_seq_ops,
                 gather_ops,
                 sp_attn_materialize_segments);

        if (collect_compute_breakdown) {
            LOG_INFO("%s graph cut profile #%zu op-category-summary math_ops=%zu layout_ops=%zu layout_per_compute_ms=%.3f math_per_compute_ms=%.3f",
                     get_desc().c_str(),
                     graph_cut_profile_index_,
                     math_ops,
                     layout_ops,
                     compute_ms > 0 ? static_cast<double>(layout_ops) / static_cast<double>(compute_ms) : 0.0,
                     compute_ms > 0 ? static_cast<double>(math_ops) / static_cast<double>(compute_ms) : 0.0);
        }

        LOG_INFO("%s graph cut profile #%zu overhead-detail collect_future=%.3fms reset_runtime=%.3fms bind_cached=%.3fms mark_cache_outputs=%.3fms pre_compute_cb=%.3fms segment_graph_free=%.3fms copy_set_build=%.3fms copy_map_scan=%.3fms copy_tensor_set=%.3fms copied_tensors=%zu copied_bytes=%.2fMiB skipped_not_in_graph=%zu skipped_no_buffer=%zu runtime_const_hits=%zu runtime_const_hit=%.3fms runtime_const_hit_bytes=%.2fMiB runtime_const_uploads=%zu runtime_const_upload=%.3fms runtime_const_upload_bytes=%.2fMiB avg_graph_tensors=%.1f avg_backend_map=%.1f",
                 get_desc().c_str(),
                 graph_cut_profile_index_,
                 collect_future_inputs_us / 1000.0,
                 reset_runtime_tensors_us / 1000.0,
                 bind_cached_inputs_us / 1000.0,
                 mark_cache_outputs_us / 1000.0,
                 pre_compute_callback_us / 1000.0,
                 segment_graph_free_us / 1000.0,
                 copy_graph_tensor_set_us / 1000.0,
                 copy_backend_map_scan_us / 1000.0,
                 copy_backend_tensor_set_us / 1000.0,
                 copied_tensors,
                 bytes_to_mib(copied_bytes),
                 skipped_not_in_graph,
                 skipped_no_buffer,
                 runtime_const_cache_hits,
                 runtime_const_cache_hit_us / 1000.0,
                 bytes_to_mib(runtime_const_cache_hit_bytes),
                 runtime_const_cache_uploads,
                 runtime_const_cache_upload_us / 1000.0,
                 bytes_to_mib(runtime_const_cache_upload_bytes),
                 segments.empty() ? 0.0 : static_cast<double>(copy_graph_tensor_entries) / static_cast<double>(segments.size()),
                 segments.empty() ? 0.0 : static_cast<double>(copy_backend_map_entries) / static_cast<double>(segments.size()));

        if (collect_compute_breakdown) {
            std::vector<std::pair<std::string, const GraphCutComputeBucketProfile*>> bucket_order;
            bucket_order.reserve(compute_buckets.size());
            for (const auto& entry : compute_buckets) {
                bucket_order.emplace_back(entry.first, &entry.second);
            }
            std::sort(bucket_order.begin(), bucket_order.end(), [](const auto& a, const auto& b) {
                if (a.second->compute_ms != b.second->compute_ms) {
                    return a.second->compute_ms > b.second->compute_ms;
                }
                return a.first < b.first;
            });

            for (size_t bucket_idx = 0; bucket_idx < bucket_order.size(); ++bucket_idx) {
                const auto& bucket_name = bucket_order[bucket_idx].first;
                const auto& bucket = *bucket_order[bucket_idx].second;
                if (bucket.compute_ms <= 0 && bucket.comm_ms <= 0 && bucket.total_ms <= 0) {
                    continue;
                }
                LOG_INFO("%s graph cut profile #%zu compute-bucket%zu bucket=%s segments=%zu nodes=%zu math_ops=%zu layout_ops=%zu total=%lldms compute=%lldms comm=%lldms copy=%lldms cache=%lldms comm_ops=%zu comm_bytes=%.2fMiB ops=%s",
                         get_desc().c_str(),
                         graph_cut_profile_index_,
                         bucket_idx + 1,
                         bucket_name.c_str(),
                         bucket.segments,
                         bucket.nodes,
                         bucket.math_ops,
                         bucket.layout_ops,
                         static_cast<long long>(bucket.total_ms),
                         static_cast<long long>(bucket.compute_ms),
                         static_cast<long long>(bucket.comm_ms),
                         static_cast<long long>(bucket.copy_ms),
                         static_cast<long long>(bucket.cache_ms),
                         bucket.comm_ops,
                         bytes_to_mib(bucket.comm_bytes),
                         graph_cut_bucket_op_histogram(bucket).c_str());
                if (collect_materialize_breakdown) {
                    GraphCutMaterializeProfile bucket_materialize_view;
                    bucket_materialize_view.stages = bucket.materialize_stages;
                    LOG_INFO("%s graph cut profile #%zu materialize-bucket%zu bucket=%s materialize_ops=%zu materialize_bytes=%.2fMiB cont=%zu/%.2fMiB cpy=%zu/%.2fMiB concat=%zu/%.2fMiB dup=%zu/%.2fMiB boundary_out=%.2fMiB cached_out=%.2fMiB comm_in=%.2fMiB comm_out=%.2fMiB repeated_sources=%zu groups/%zu ops/%.2fMiB mat_after_mat=%zu/%.2fMiB cont_from_cont=%zu/%.2fMiB concat_to_cont=%zu/%.2fMiB permute_view_to_cont=%zu/%.2fMiB mat_view_mat=%zu/%.2fMiB cont_permute_cont=%zu/%.2fMiB stages=%s",
                             get_desc().c_str(),
                             graph_cut_profile_index_,
                             bucket_idx + 1,
                             bucket_name.c_str(),
                             bucket.materialize_ops,
                             bytes_to_mib(bucket.materialize_bytes),
                             bucket.cont_ops,
                             bytes_to_mib(bucket.cont_bytes),
                             bucket.cpy_ops,
                             bytes_to_mib(bucket.cpy_bytes),
                             bucket.concat_ops,
                             bytes_to_mib(bucket.concat_bytes),
                             bucket.dup_ops,
                             bytes_to_mib(bucket.dup_bytes),
                             bytes_to_mib(bucket.materialize_boundary_output_bytes),
                             bytes_to_mib(bucket.materialize_cached_output_bytes),
                             bytes_to_mib(bucket.materialize_comm_input_bytes),
                             bytes_to_mib(bucket.materialize_comm_output_bytes),
                             bucket.repeated_materialize_source_groups,
                             bucket.repeated_materialize_source_ops,
                             bytes_to_mib(bucket.repeated_materialize_source_bytes),
                             bucket.materialize_after_materialize_ops,
                             bytes_to_mib(bucket.materialize_after_materialize_bytes),
                             bucket.cont_from_cont_ops,
                             bytes_to_mib(bucket.cont_from_cont_bytes),
                             bucket.concat_to_cont_ops,
                             bytes_to_mib(bucket.concat_to_cont_bytes),
                             bucket.permute_view_to_cont_ops,
                             bytes_to_mib(bucket.permute_view_to_cont_bytes),
                             bucket.materialize_view_materialize_ops,
                             bytes_to_mib(bucket.materialize_view_materialize_bytes),
                             bucket.cont_permute_cont_ops,
                             bytes_to_mib(bucket.cont_permute_cont_bytes),
                             graph_cut_materialize_stage_summary(bucket_materialize_view).c_str());
                }
            }
        }

        const int top_n = graph_cut_profile_top_n();
        const int breakdown_top_n = std::max(compute_top_n, materialize_top_n);
        if (top_n <= 0 && breakdown_top_n <= 0) {
            return;
        }

        std::vector<size_t> order;
        order.reserve(segments.size());
        for (size_t i = 0; i < segments.size(); ++i) {
            order.push_back(i);
        }
        if (breakdown_top_n > 0) {
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                if (segments[a].compute_ms != segments[b].compute_ms) {
                    return segments[a].compute_ms > segments[b].compute_ms;
                }
                return segments[a].total_ms > segments[b].total_ms;
            });

            const size_t compute_limit = std::min(order.size(), static_cast<size_t>(breakdown_top_n));
            for (size_t rank_idx = 0; rank_idx < compute_limit; ++rank_idx) {
                const auto& segment = segments[order[rank_idx]];
                if (segment.compute_ms <= 0) {
                    break;
                }
                LOG_INFO("%s graph cut profile #%zu top-compute%zu segment=%s bucket=%s nodes=%zu math_ops=%zu layout_ops=%zu total=%lldms compute=%lldms comm=%lldms copy=%lldms cache=%lldms comm_ops=%zu comm_bytes=%.2fMiB ops=%s io=%s comm_names=%s",
                         get_desc().c_str(),
                         graph_cut_profile_index_,
                         rank_idx + 1,
                         segment.name.empty() ? "<unnamed>" : segment.name.c_str(),
                         graph_cut_segment_compute_bucket(segment).c_str(),
                         segment.nodes,
                         segment.math_ops,
                         segment.layout_ops,
                         static_cast<long long>(segment.total_ms),
                         static_cast<long long>(segment.compute_ms),
                         static_cast<long long>(segment.comm_ms),
                         static_cast<long long>(segment.copy_ms),
                         static_cast<long long>(segment.cache_ms),
                         segment.comm_ops,
                         bytes_to_mib(segment.comm_bytes),
                         segment.op_histogram.c_str(),
                         segment.io_summary.c_str(),
                         segment.comm_names.c_str());
                if (collect_materialize_breakdown && segment.materialize_ops > 0) {
                    LOG_INFO("%s graph cut profile #%zu top-materialize%zu segment=%s bucket=%s materialize_ops=%zu materialize_bytes=%.2fMiB cont=%zu/%.2fMiB cpy=%zu/%.2fMiB concat=%zu/%.2fMiB dup=%zu/%.2fMiB boundary_out=%.2fMiB cached_out=%.2fMiB comm_in=%.2fMiB comm_out=%.2fMiB repeated_sources=%zu groups/%zu ops/%.2fMiB mat_after_mat=%zu/%.2fMiB cont_from_cont=%zu/%.2fMiB concat_to_cont=%zu/%.2fMiB permute_view_to_cont=%zu/%.2fMiB mat_view_mat=%zu/%.2fMiB cont_permute_cont=%zu/%.2fMiB stages=%s stage_details=%s repeated=%s chains=%s nodes=%s",
                             get_desc().c_str(),
                             graph_cut_profile_index_,
                             rank_idx + 1,
                             segment.name.empty() ? "<unnamed>" : segment.name.c_str(),
                             graph_cut_segment_compute_bucket(segment).c_str(),
                             segment.materialize_ops,
                             bytes_to_mib(segment.materialize_bytes),
                             segment.cont_ops,
                             bytes_to_mib(segment.cont_bytes),
                             segment.cpy_ops,
                             bytes_to_mib(segment.cpy_bytes),
                             segment.concat_ops,
                             bytes_to_mib(segment.concat_bytes),
                             segment.dup_ops,
                             bytes_to_mib(segment.dup_bytes),
                             bytes_to_mib(segment.materialize_boundary_output_bytes),
                             bytes_to_mib(segment.materialize_cached_output_bytes),
                             bytes_to_mib(segment.materialize_comm_input_bytes),
                             bytes_to_mib(segment.materialize_comm_output_bytes),
                             segment.repeated_materialize_source_groups,
                             segment.repeated_materialize_source_ops,
                             bytes_to_mib(segment.repeated_materialize_source_bytes),
                             segment.materialize_after_materialize_ops,
                             bytes_to_mib(segment.materialize_after_materialize_bytes),
                             segment.cont_from_cont_ops,
                             bytes_to_mib(segment.cont_from_cont_bytes),
                             segment.concat_to_cont_ops,
                             bytes_to_mib(segment.concat_to_cont_bytes),
                             segment.permute_view_to_cont_ops,
                             bytes_to_mib(segment.permute_view_to_cont_bytes),
                             segment.materialize_view_materialize_ops,
                             bytes_to_mib(segment.materialize_view_materialize_bytes),
                             segment.cont_permute_cont_ops,
                             bytes_to_mib(segment.cont_permute_cont_bytes),
                             segment.materialize_stage_summary.c_str(),
                             segment.materialize_stage_details.c_str(),
                             segment.repeated_materialize_sources.c_str(),
                             segment.materialize_chain_top_nodes.c_str(),
                             segment.materialize_top_nodes.c_str());
                }
            }
        }

        if (top_n <= 0) {
            return;
        }

        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return segments[a].total_ms > segments[b].total_ms;
        });

        const size_t limit = std::min(order.size(), static_cast<size_t>(top_n));
        for (size_t rank_idx = 0; rank_idx < limit; ++rank_idx) {
            const auto& segment = segments[order[rank_idx]];
            LOG_INFO("%s graph cut profile #%zu top%zu segment=%s nodes=%zu comm_ops=%zu comm_bytes=%.2fMiB output_bytes=%.2fMiB cached_output_bytes=%.2fMiB cache_live=%.2fMiB cache_chunks=%zu cache_pool=%.2fMiB pool_chunks=%zu total=%lldms build=%lldms params=%lldms offload=%lldms alloc=%lldms copy=%lldms compute=%lldms comm=%lldms cache=%lldms comm_names=%s",
                     get_desc().c_str(),
                     graph_cut_profile_index_,
                     rank_idx + 1,
                     segment.name.empty() ? "<unnamed>" : segment.name.c_str(),
                     segment.nodes,
                     segment.comm_ops,
                     bytes_to_mib(segment.comm_bytes),
                     bytes_to_mib(segment.output_bytes),
                     bytes_to_mib(segment.cached_output_bytes),
                     bytes_to_mib(segment.cache_live_bytes),
                     segment.cache_chunks,
                     bytes_to_mib(segment.cache_pool_bytes),
                     segment.cache_pool_chunks,
                     static_cast<long long>(segment.total_ms),
                     static_cast<long long>(segment.build_ms),
                     static_cast<long long>(segment.runtime_param_ms),
                     static_cast<long long>(segment.offload_ms),
                     static_cast<long long>(segment.alloc_ms),
                     static_cast<long long>(segment.copy_ms),
                     static_cast<long long>(segment.compute_ms),
                     static_cast<long long>(segment.comm_ms),
                     static_cast<long long>(segment.cache_ms),
                     segment.comm_names.c_str());
        }

        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return segments[a].copy_ms > segments[b].copy_ms;
        });

        for (size_t rank_idx = 0; rank_idx < limit; ++rank_idx) {
            const auto& segment = segments[order[rank_idx]];
            if (segment.copy_ms <= 0 &&
                segment.copy_detail.graph_tensor_set_us <= 0 &&
                segment.copy_detail.backend_map_scan_us <= 0 &&
                segment.copy_detail.backend_tensor_set_us <= 0 &&
                segment.pre_compute_callback_us <= 0) {
                break;
            }
            LOG_INFO("%s graph cut profile #%zu top-copy%zu segment=%s copy=%lldms copy_set_build=%.3fms copy_map_scan=%.3fms copy_tensor_set=%.3fms pre_compute_cb=%.3fms copied_tensors=%zu copied_bytes=%.2fMiB runtime_const_hits=%zu runtime_const_hit=%.3fms runtime_const_hit_bytes=%.2fMiB runtime_const_uploads=%zu runtime_const_upload=%.3fms runtime_const_upload_bytes=%.2fMiB skipped_not_in_graph=%zu skipped_no_buffer=%zu graph_tensors=%zu backend_map=%zu total=%lldms compute=%lldms comm=%lldms comm_names=%s",
                     get_desc().c_str(),
                     graph_cut_profile_index_,
                     rank_idx + 1,
                     segment.name.empty() ? "<unnamed>" : segment.name.c_str(),
                     static_cast<long long>(segment.copy_ms),
                     segment.copy_detail.graph_tensor_set_us / 1000.0,
                     segment.copy_detail.backend_map_scan_us / 1000.0,
                     segment.copy_detail.backend_tensor_set_us / 1000.0,
                     segment.pre_compute_callback_us / 1000.0,
                     segment.copy_detail.copied_tensors,
                     bytes_to_mib(segment.copy_detail.copied_bytes),
                     segment.copy_detail.runtime_const_cache_hits,
                     segment.copy_detail.runtime_const_cache_hit_us / 1000.0,
                     bytes_to_mib(segment.copy_detail.runtime_const_cache_hit_bytes),
                     segment.copy_detail.runtime_const_cache_uploads,
                     segment.copy_detail.runtime_const_cache_upload_us / 1000.0,
                     bytes_to_mib(segment.copy_detail.runtime_const_cache_upload_bytes),
                     segment.copy_detail.skipped_not_in_graph,
                     segment.copy_detail.skipped_no_buffer,
                     segment.copy_detail.graph_tensor_entries,
                     segment.copy_detail.backend_map_entries,
                     static_cast<long long>(segment.total_ms),
                     static_cast<long long>(segment.compute_ms),
                     static_cast<long long>(segment.comm_ms),
                     segment.comm_names.c_str());
        }

        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const int64_t a_non_compute = a < segments.size()
                                              ? segments[a].total_ms - segments[a].compute_ms - segments[a].comm_ms
                                              : 0;
            const int64_t b_non_compute = b < segments.size()
                                              ? segments[b].total_ms - segments[b].compute_ms - segments[b].comm_ms
                                              : 0;
            return a_non_compute > b_non_compute;
        });

        for (size_t rank_idx = 0; rank_idx < limit; ++rank_idx) {
            const auto& segment = segments[order[rank_idx]];
            const int64_t non_compute_ms = segment.total_ms - segment.compute_ms - segment.comm_ms;
            if (non_compute_ms <= 0) {
                break;
            }
            LOG_INFO("%s graph cut profile #%zu top-non-compute%zu segment=%s non_compute=%lldms total=%lldms compute=%lldms comm=%lldms build=%lldms params=%lldms offload=%lldms alloc=%lldms copy=%lldms cache=%lldms collect_future=%.3fms reset_runtime=%.3fms bind_cached=%.3fms mark_cache_outputs=%.3fms graph_free=%.3fms comm_names=%s",
                     get_desc().c_str(),
                     graph_cut_profile_index_,
                     rank_idx + 1,
                     segment.name.empty() ? "<unnamed>" : segment.name.c_str(),
                     static_cast<long long>(non_compute_ms),
                     static_cast<long long>(segment.total_ms),
                     static_cast<long long>(segment.compute_ms),
                     static_cast<long long>(segment.comm_ms),
                     static_cast<long long>(segment.build_ms),
                     static_cast<long long>(segment.runtime_param_ms),
                     static_cast<long long>(segment.offload_ms),
                     static_cast<long long>(segment.alloc_ms),
                     static_cast<long long>(segment.copy_ms),
                     static_cast<long long>(segment.cache_ms),
                     segment.collect_future_inputs_us / 1000.0,
                     segment.reset_runtime_tensors_us / 1000.0,
                     segment.bind_cached_inputs_us / 1000.0,
                     segment.mark_cache_outputs_us / 1000.0,
                     segment.segment_graph_free_us / 1000.0,
                     segment.comm_names.c_str());
        }

        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return segments[a].cache_ms > segments[b].cache_ms;
        });

        for (size_t rank_idx = 0; rank_idx < limit; ++rank_idx) {
            const auto& segment = segments[order[rank_idx]];
            if (segment.cache_ms <= 0 && segment.cached_output_bytes == 0) {
                break;
            }
            LOG_INFO("%s graph cut profile #%zu top-cache%zu segment=%s cache=%lldms cached_output_bytes=%.2fMiB output_bytes=%.2fMiB cache_live=%.2fMiB cache_chunks=%zu cache_pool=%.2fMiB pool_chunks=%zu total=%lldms comm_names=%s",
                     get_desc().c_str(),
                     graph_cut_profile_index_,
                     rank_idx + 1,
                     segment.name.empty() ? "<unnamed>" : segment.name.c_str(),
                     static_cast<long long>(segment.cache_ms),
                     bytes_to_mib(segment.cached_output_bytes),
                     bytes_to_mib(segment.output_bytes),
                     bytes_to_mib(segment.cache_live_bytes),
                     segment.cache_chunks,
                     bytes_to_mib(segment.cache_pool_bytes),
                     segment.cache_pool_chunks,
                     static_cast<long long>(segment.total_ms),
                     segment.comm_names.c_str());
        }
    }

    void alloc_params_ctx() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        params_ctx = ggml_init(params);
        GGML_ASSERT(params_ctx != nullptr);
        params_tensor_set_.clear();
        if (params_backend != runtime_backend) {
            offload_ctx = ggml_init(params);
            GGML_ASSERT(offload_ctx != nullptr);
        }
    }

    void free_params_ctx() {
        if (params_ctx != nullptr) {
            ggml_free(params_ctx);
            params_ctx = nullptr;
        }
        params_tensor_set_.clear();
        if (offload_ctx != nullptr) {
            ggml_free(offload_ctx);
            offload_ctx = nullptr;
        }
        if (partial_offload_ctx != nullptr) {
            ggml_free(partial_offload_ctx);
            partial_offload_ctx = nullptr;
        }
    }

    void alloc_cache_ctx() {
        cache_ctx = new_cache_context();
    }

    void free_cache_ctx() {
        if (cache_ctx != nullptr) {
            ggml_free(cache_ctx);
            cache_ctx = nullptr;
        }
    }

    void free_cache_chunk(GraphCacheChunk& chunk) {
        chunk.reset();
    }

    void free_active_cache_chunks(bool release_to_pool) {
        if (release_to_pool) {
            for (auto& chunk : cache_chunks_) {
                release_cache_chunk(std::move(chunk));
            }
        } else {
            for (auto& chunk : cache_chunks_) {
                free_cache_chunk(chunk);
            }
        }
        cache_chunks_.clear();
    }

    void free_cache_chunk_pool() {
        for (auto& chunk : cache_chunk_pool_) {
            free_cache_chunk(chunk);
        }
        cache_chunk_pool_.clear();
        cache_chunk_pool_bytes_ = 0;
    }

    void free_cache_chunks(bool keep_pool = false) {
        free_active_cache_chunks(keep_pool);
        if (!keep_pool) {
            free_cache_chunk_pool();
        }
    }

    void alloc_compute_ctx() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        compute_ctx = ggml_init(params);
        GGML_ASSERT(compute_ctx != nullptr);
    }

    void free_compute_ctx() {
        if (compute_ctx != nullptr) {
            ggml_free(compute_ctx);
            compute_ctx = nullptr;
        }
        sd::ggml_graph_cut::clear_graph_cut_marks();
        backend_tensor_data_map.clear();
        invalidate_persistent_graph();
    }

    // The reuse graph (persistent_.gf) is valid only while BOTH compute_ctx (its
    // nodes) and compute_allocr (the device buffer its tensors point into) are
    // alive and unchanged. Any teardown of either must invalidate it, else a later
    // reuse step re-executes a graph whose tensors dangle into freed memory. This
    // is the single safety anchor; both free_compute_ctx() and free_compute_buffer()
    // call it, so within a generation (buffer kept alive across steps) reuse holds,
    // and a new generation (buffer freed post-loop) rebuilds once.
    void invalidate_persistent_graph() {
        persistent_.valid = false;
        persistent_.reuse_disabled = false;
        persistent_.gf = nullptr;
        persistent_.inputs.clear();
        reuse_capture_mode_ = false;
    }

    void rebuild_params_tensor_set() {
        params_tensor_set_.clear();
        if (params_ctx == nullptr) {
            return;
        }
        for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
            params_tensor_set_.insert(t);
        }
    }

    void prepare_build_in_tensor_before() {
        one_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(one_tensor, "ggml_runner_build_in_tensor:one");
        set_backend_tensor_data(one_tensor, one_vec.data());

        zero_int_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(zero_int_tensor, "ggml_runner_build_in_tensor:zero_int");
        set_backend_tensor_data(zero_int_tensor, zero_int_vec.data());
    }

    void prepare_build_in_tensor_after(ggml_cgraph* gf) {
        ggml_build_forward_expand(gf, one_tensor);
        ggml_build_forward_expand(gf, zero_int_tensor);
    }

    ggml_cgraph* new_graph_custom(size_t graph_size) {
        if (weight_adapter) {
            graph_size += weight_adapter->get_extra_graph_size();
        }
        return ggml_new_graph_custom(compute_ctx, graph_size, false);
    }

    ggml_cgraph* get_compute_graph(get_graph_cb_t get_graph) {
        prepare_build_in_tensor_before();
        ggml_cgraph* gf = get_graph();
        // The model result is the last node get_graph expanded. execute_graph
        // reads it back BY NAME (final_result_name), not by position, so name it
        // first — then the cache seam can append aux capture branches without
        // stealing the result slot.
        if (ggml_graph_n_nodes(gf) > 0) {
            auto result = ggml_graph_node(gf, -1);
            ggml_set_name(result, final_result_name.c_str());
        }
        if (tap_registry_ != nullptr && ggml_graph_n_nodes(gf) > 0) {
            expand_tap_registry_nodes(gf);
        }
        prepare_build_in_tensor_after(gf);
        return gf;
    }

    // Promote the substep's recorded taps + woven indicator scalars into the
    // named-tensor index so they can be read back post-compute. Each tapped anchor
    // is pinned as a graph output under a stable name; indicator scalar nodes name
    // themselves (cache_ind:<name>) and are expanded here too. The registry only
    // holds anchors the model actually tapped this build.
    void expand_tap_registry_nodes(ggml_cgraph* gf) {
        auto expand_named = [&](ggml_tensor* node, const std::string& name) {
            if (node == nullptr) {
                return;
            }
            ggml_set_name(node, name.c_str());
            ggml_set_output(node);
            cache(name, node);
            ggml_build_forward_expand(gf, node);
        };
        // Weave indicators + residual capture HERE (after the final-result node was
        // named at prepare-time), so the model output stays the graph's result and
        // these aux nodes are appended without stealing the result slot.
        for (const auto& ind : tap_registry_->indicators()) {
            ggml_tensor* s = edgedit::cache::lower_indicator(compute_ctx, ind, *tap_registry_);
            if (s != nullptr) {
                expand_named(s, s->name);
            }
        }
        if (tap_registry_->capture_residual()) {
            ggml_tensor* min = tap_registry_->get(edgedit::cache::AnchorRef::model_in());
            ggml_tensor* mout = tap_registry_->get(edgedit::cache::AnchorRef::model_out());
            if (min != nullptr && mout != nullptr) {
                ggml_tensor* feat = ggml_sub(compute_ctx, mout, min);
                expand_named(feat, "ed_cache_feature");
            }
        }
        // Cache-layer graph extensions: the cache lowering asked us to weave these
        // operator nodes over the tapped tensors and pin the result. The runner
        // executes them blindly — op->lower() emits the ggml nodes (e.g. a
        // DIFFERENCE = ggml_sub), and this code does NOT know the math means
        // "residual". Replaces the hardcoded capture_residual weave above for
        // methods that have migrated (MagCache). The pass handoff then d2d-copies
        // the pinned tensor into a device slot by output_name (CaptureToSlot).
        for (const auto& ext : tap_registry_->extensions()) {
            if (ext.op == nullptr) {
                continue;
            }
            std::vector<ggml_tensor*> inputs;
            bool inputs_ok = true;
            for (const auto& a : ext.input_anchors) {
                ggml_tensor* t = tap_registry_->get(a);
                if (t == nullptr) {
                    inputs_ok = false;
                    break;
                }
                inputs.push_back(t);
            }
            if (!inputs_ok) {
                continue;
            }
            for (ggml_tensor* t : ext.extra_inputs) {
                inputs.push_back(t);
            }
            edgedit::cache::GraphLoweringContext gctx;
            gctx.ctx = compute_ctx;
            std::vector<ggml_tensor*> outputs;
            if (ext.op->lower(gctx, inputs, ext.params, &outputs) && !outputs.empty() &&
                outputs[0] != nullptr) {
                expand_named(outputs[0], ext.output_name);
            }
        }
        // Capture-writeback (DiCache device seed): also weave the probe residual
        // (BlockOut[m-1] - ModelIn) as "ed_cache_probe_resid" so the pass's
        // post-readback can d2d it into the runner's persistent probe-residual ring.
        // The raw probe/before taps are already pinned by the recorded() loop below,
        // so the handoff reads those by their ed_tap: names. Same sub-expr form as the
        // gamma weave above.
        if (tap_registry_->capture_writeback()) {
            const int m = tap_registry_->writeback_probe_depth();
            ggml_tensor* before = tap_registry_->get(edgedit::cache::AnchorRef::model_in());
            ggml_tensor* probe = tap_registry_->get(edgedit::cache::AnchorRef::block_out(m - 1));
            if (before != nullptr && probe != nullptr) {
                expand_named(ggml_sub(compute_ctx, probe, before), "ed_cache_probe_resid");
            }
        }
        // DiCache probe metrics (delta_y/delta_x/gamma) are now woven by cache
        // operators (cache.rel_l1 / cache.gamma_indicator) via the extensions loop
        // above — the model builds Indicator-sink extensions in compute_substep_probe.
        // The hardcoded reduction weave that lived here has been retired.
        // Also pin the raw taps (so a method that reads an anchor tensor directly
        // can). Indicator/feature operands are already expanded above.
        for (const auto& kv : tap_registry_->recorded()) {
            expand_named(kv.second, kv.first);
        }
    }

    bool prepare_compute_graph(get_graph_cb_t get_graph,
                               ggml_cgraph** gf_out) {
        GGML_ASSERT(gf_out != nullptr);

        reset_compute_ctx();
        ggml_cgraph* gf = get_compute_graph(get_graph);
        if (gf == nullptr) {
            sd::ggml_graph_cut::clear_comm_marks();
            free_compute_ctx();
            return false;
        }

        *gf_out = gf;
        return true;
    }

    bool alloc_compute_buffer(ggml_cgraph* gf) {
        if (compute_allocr != nullptr) {
            return true;
        }
        compute_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_backend));

        if (!ggml_gallocr_reserve(compute_allocr, gf)) {
            // failed to allocate the compute buffer
            LOG_ERROR("%s: failed to allocate the compute buffer\n", get_desc().c_str());
            free_compute_buffer();
            return false;
        }

        // compute the required memory
        size_t compute_buffer_size = ggml_gallocr_get_buffer_size(compute_allocr, 0);
        return true;
    }

    // Measure this runner's compute buffer (activation) footprint for a given graph
    // WITHOUT allocating any device memory or touching the real compute buffer. Used by
    // the auto-allocate/auto-fit scheduler to size the resident headroom from the actual
    // model+resolution instead of a fixed constant. Returns 0 on failure (caller falls
    // back to the fixed headroom).
    //
    // Safe to call before alloc_params_buffer: the weight tensors exist as shapes but
    // have data==NULL/buffer==NULL, which ggml_gallocr would otherwise COUNT as tensors
    // needing allocation (inflating the result by the weight bytes). We temporarily set a
    // non-null sentinel on each params_ctx tensor's ->data so the allocator treats them
    // as externally-provided (weights excluded), then restore NULL. If params are already
    // allocated (data/buffer set), the sentinel loop is a no-op and this measures pure
    // activations directly. The sentinel is restored on every exit path via RAII.
    size_t measure_compute_buffer(get_graph_cb_t get_graph) {
        if (runtime_backend == nullptr || params_ctx == nullptr) {
            return 0;
        }

        struct WeightSentinelGuard {
            ggml_context* ctx;
            std::vector<ggml_tensor*> patched;
            explicit WeightSentinelGuard(ggml_context* c) : ctx(c) {
                static char sentinel_byte = 0;
                for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr;
                     t = ggml_get_next_tensor(ctx, t)) {
                    if (t->data == nullptr && t->buffer == nullptr) {
                        t->data = &sentinel_byte;  // any non-null pointer; never dereferenced
                        patched.push_back(t);
                    }
                }
            }
            ~WeightSentinelGuard() {
                for (ggml_tensor* t : patched) {
                    t->data = nullptr;
                }
            }
        } guard(params_ctx);

        reset_compute_ctx();
        ggml_cgraph* gf = get_compute_graph(get_graph);
        if (gf == nullptr) {
            sd::ggml_graph_cut::clear_comm_marks();
            free_compute_ctx();
            return 0;
        }

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_backend));
        size_t sizes[1] = {0};
        ggml_gallocr_reserve_n_size(allocr, gf, nullptr, nullptr, sizes);
        const size_t buffer_size = sizes[0];
        ggml_gallocr_free(allocr);

        free_compute_ctx();  // clears backend_tensor_data_map + invalidates persistent graph
        return buffer_size;
    }

    void free_cache_buffer() {
        if (cache_buffer != nullptr) {
            ggml_backend_buffer_free(cache_buffer);
            cache_buffer = nullptr;
        }
    }

    void rebuild_compact_cache_tensor_index() {
        cache_tensor_index_.clear();
        if (cache_ctx == nullptr) {
            return;
        }
        for (ggml_tensor* tensor = ggml_get_first_tensor(cache_ctx); tensor != nullptr; tensor = ggml_get_next_tensor(cache_ctx, tensor)) {
            if (tensor->name[0] != '\0') {
                cache_tensor_index_[tensor->name] = tensor;
            }
        }
    }

    size_t graph_cache_live_bytes() const {
        size_t bytes = 0;
        std::unordered_set<const ggml_tensor*> seen;
        seen.reserve(cache_tensor_index_.size());
        for (const auto& kv : cache_tensor_index_) {
            if (kv.second != nullptr && seen.insert(kv.second).second) {
                bytes += ggml_nbytes(kv.second);
            }
        }
        return bytes;
    }

    size_t graph_cache_buffer_bytes() const {
        size_t bytes = 0;
        if (cache_buffer != nullptr) {
            bytes += ggml_backend_buffer_get_size(cache_buffer);
        }
        for (const auto& chunk : cache_chunks_) {
            if (chunk.buffer != nullptr) {
                bytes += ggml_backend_buffer_get_size(chunk.buffer);
            }
        }
        bytes += cache_chunk_pool_bytes_;
        return bytes;
    }

    size_t graph_cache_chunk_count() const {
        return cache_chunks_.size() + (cache_buffer != nullptr ? 1 : 0);
    }

    size_t graph_cache_pool_chunk_count() const {
        return cache_chunk_pool_.size();
    }

    static std::string graph_cache_layout_key(const std::vector<ggml_tensor*>& source_tensors) {
        std::ostringstream oss;
        oss << source_tensors.size();
        for (ggml_tensor* tensor : source_tensors) {
            oss << '|';
            if (tensor == nullptr) {
                oss << "null";
                continue;
            }
            oss << static_cast<int>(tensor->type);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                oss << ':' << tensor->ne[d] << '/' << tensor->nb[d];
            }
        }
        return oss.str();
    }

    static bool graph_cache_same_layout(const ggml_tensor* a, const ggml_tensor* b) {
        if (a == nullptr || b == nullptr || a->type != b->type) {
            return false;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (a->ne[d] != b->ne[d] || a->nb[d] != b->nb[d]) {
                return false;
            }
        }
        return true;
    }

    GraphCacheChunk make_cache_chunk_for_sources(const std::vector<ggml_tensor*>& source_tensors,
                                                 const std::string& layout_key) {
        GraphCacheChunk chunk;
        chunk.ctx = new_cache_context(source_tensors.size());
        chunk.layout_key = layout_key;
        chunk.cache_tensors.reserve(source_tensors.size());
        for (ggml_tensor* source_tensor : source_tensors) {
            ggml_tensor* cache_tensor = ggml_dup_tensor(chunk.ctx, source_tensor);
            chunk.cache_tensors.push_back(cache_tensor);
        }
        return chunk;
    }

    bool cache_chunk_layout_matches(const GraphCacheChunk& chunk,
                                    const std::vector<ggml_tensor*>& source_tensors,
                                    const std::string& layout_key) const {
        if (chunk.layout_key != layout_key || chunk.cache_tensors.size() != source_tensors.size()) {
            return false;
        }
        for (size_t i = 0; i < source_tensors.size(); ++i) {
            if (source_tensors[i] == nullptr || chunk.cache_tensors[i] == nullptr) {
                return false;
            }
            if (!graph_cache_same_layout(source_tensors[i], chunk.cache_tensors[i])) {
                return false;
            }
        }
        return true;
    }

    GraphCacheChunk acquire_cache_chunk(const std::vector<ggml_tensor*>& source_tensors,
                                        const std::string& layout_key) {
        if (graph_cut_cache_pool_enabled()) {
            for (auto it = cache_chunk_pool_.begin(); it != cache_chunk_pool_.end(); ++it) {
                if (!cache_chunk_layout_matches(*it, source_tensors, layout_key)) {
                    continue;
                }
                GraphCacheChunk chunk = std::move(*it);
                if (chunk.buffer != nullptr) {
                    size_t buffer_size = ggml_backend_buffer_get_size(chunk.buffer);
                    cache_chunk_pool_bytes_ = cache_chunk_pool_bytes_ >= buffer_size
                                                  ? cache_chunk_pool_bytes_ - buffer_size
                                                  : 0;
                }
                cache_chunk_pool_.erase(it);
                chunk.tensors.clear();
                return chunk;
            }
        }
        return make_cache_chunk_for_sources(source_tensors, layout_key);
    }

    void release_cache_chunk(GraphCacheChunk&& chunk) {
        chunk.tensors.clear();
        if (!graph_cut_cache_pool_enabled() || chunk.buffer == nullptr || chunk.ctx == nullptr) {
            free_cache_chunk(chunk);
            return;
        }

        const size_t buffer_size = ggml_backend_buffer_get_size(chunk.buffer);
        const size_t max_pool_bytes = graph_cut_cache_pool_max_bytes();
        if (max_pool_bytes == 0 || cache_chunk_pool_bytes_ + buffer_size > max_pool_bytes) {
            free_cache_chunk(chunk);
            return;
        }

        cache_chunk_pool_bytes_ += buffer_size;
        cache_chunk_pool_.push_back(std::move(chunk));
    }

    bool copy_cache_tensor_data(ggml_tensor* src, ggml_tensor* dst) {
        ggml_backend_buffer_t src_buf = sd::ggml_graph_cut::tensor_buffer(src);
        ggml_backend_buffer_t dst_buf = sd::ggml_graph_cut::tensor_buffer(dst);
        if (src_buf == nullptr || dst_buf == nullptr) {
            LOG_ERROR("%s cache copy tensor buffer missing: name=%s src_buffer=%p src_view_src=%p src_view_src_buffer=%p dst_buffer=%p",
                      get_desc().c_str(),
                      src && src->name[0] != '\0' ? src->name : "<unnamed>",
                      src ? src->buffer : nullptr,
                      src ? src->view_src : nullptr,
                      (src && src->view_src) ? src->view_src->buffer : nullptr,
                      dst ? dst->buffer : nullptr);
            return false;
        }
        const bool use_staging_copy = src->view_src != nullptr || !ggml_is_contiguous(src) || src->buffer == nullptr;
        if (use_staging_copy) {
            std::vector<uint8_t> host_data(ggml_nbytes(src));
            ggml_backend_tensor_get(src, host_data.data(), 0, host_data.size());
            ggml_backend_tensor_set(dst, host_data.data(), 0, host_data.size());
        } else {
            ggml_backend_tensor_copy(src, dst);
        }
        return true;
    }

    bool compact_cache_tensors_to_cache_buffer(const std::unordered_set<std::string>* cache_keep_names = nullptr) {
        ggml_context* old_cache_ctx            = cache_ctx;
        ggml_backend_buffer_t old_cache_buffer = cache_buffer;
        cache_ctx                              = nullptr;
        cache_buffer                           = nullptr;
        cache_tensor_index_.clear();
        std::map<std::string, ggml_tensor*> merged_cache_sources;
        if (old_cache_ctx != nullptr) {
            for (ggml_tensor* tensor = ggml_get_first_tensor(old_cache_ctx); tensor != nullptr; tensor = ggml_get_next_tensor(old_cache_ctx, tensor)) {
                if (cache_keep_names != nullptr && cache_keep_names->find(tensor->name) == cache_keep_names->end()) {
                    continue;
                }
                merged_cache_sources[tensor->name] = tensor;
            }
        }
        for (const auto& kv : cache_tensor_map) {
            if (cache_keep_names != nullptr && cache_keep_names->find(kv.first) == cache_keep_names->end()) {
                continue;
            }
            merged_cache_sources[kv.first] = kv.second;
        }
        cache_tensor_map.clear();
        if (merged_cache_sources.empty()) {
            if (old_cache_buffer != nullptr) {
                ggml_backend_buffer_free(old_cache_buffer);
            }
            if (old_cache_ctx != nullptr) {
                ggml_free(old_cache_ctx);
            }
            return true;
        }

        alloc_cache_ctx();
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> source_to_cache_tensors;
        source_to_cache_tensors.reserve(merged_cache_sources.size());
        for (const auto& kv : merged_cache_sources) {
            ggml_tensor* source_tensor = sd::ggml_graph_cut::cache_source_tensor(kv.second);
            auto cache_tensor          = ggml_dup_tensor(cache_ctx, source_tensor);
            ggml_set_name(cache_tensor, kv.first.c_str());
            source_to_cache_tensors.push_back({source_tensor, cache_tensor});
        }
        size_t num_tensors = ggml_tensor_num(cache_ctx);
        cache_buffer       = ggml_backend_alloc_ctx_tensors(cache_ctx, runtime_backend);
        GGML_ASSERT(cache_buffer != nullptr);
        for (const auto& kv : source_to_cache_tensors) {
            if (!copy_cache_tensor_data(kv.first, kv.second)) {
                free_cache_buffer();
                free_cache_ctx();
                if (old_cache_buffer != nullptr) {
                    ggml_backend_buffer_free(old_cache_buffer);
                }
                if (old_cache_ctx != nullptr) {
                    ggml_free(old_cache_ctx);
                }
                return false;
            }
        }
        if (graph_cut_cache_sync_after_copy_enabled()) {
            ggml_backend_synchronize(runtime_backend);
        }
        size_t cache_buffer_size = ggml_backend_buffer_get_size(cache_buffer);
        ED_UNUSED(num_tensors);
        ED_UNUSED(cache_buffer_size);
        rebuild_compact_cache_tensor_index();
        if (old_cache_buffer != nullptr) {
            ggml_backend_buffer_free(old_cache_buffer);
        }
        if (old_cache_ctx != nullptr) {
            ggml_free(old_cache_ctx);
        }
        return true;
    }

    void prune_incremental_cache_chunks(const std::unordered_set<std::string>* cache_keep_names) {
        if (cache_keep_names == nullptr) {
            return;
        }

        for (auto it = cache_tensor_index_.begin(); it != cache_tensor_index_.end();) {
            if (cache_keep_names->find(it->first) == cache_keep_names->end()) {
                it = cache_tensor_index_.erase(it);
            } else {
                ++it;
            }
        }

        for (auto chunk_it = cache_chunks_.begin(); chunk_it != cache_chunks_.end();) {
            for (auto tensor_it = chunk_it->tensors.begin(); tensor_it != chunk_it->tensors.end();) {
                if (cache_keep_names->find(tensor_it->first) == cache_keep_names->end()) {
                    tensor_it = chunk_it->tensors.erase(tensor_it);
                } else {
                    ++tensor_it;
                }
            }
            if (chunk_it->tensors.empty()) {
                GraphCacheChunk released = std::move(*chunk_it);
                chunk_it = cache_chunks_.erase(chunk_it);
                release_cache_chunk(std::move(released));
            } else {
                ++chunk_it;
            }
        }
    }

    bool append_cache_tensors_to_cache_chunks(const std::unordered_set<std::string>* cache_keep_names = nullptr) {
        prune_incremental_cache_chunks(cache_keep_names);

        std::map<ggml_tensor*, std::vector<std::string>> source_to_names;
        for (const auto& kv : cache_tensor_map) {
            if (cache_keep_names != nullptr && cache_keep_names->find(kv.first) == cache_keep_names->end()) {
                continue;
            }
            ggml_tensor* source_tensor = sd::ggml_graph_cut::cache_source_tensor(kv.second);
            if (source_tensor == nullptr) {
                continue;
            }
            source_to_names[source_tensor].push_back(kv.first);
        }
        cache_tensor_map.clear();
        if (source_to_names.empty()) {
            return true;
        }

        GraphCacheChunk chunk;
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> source_to_cache_tensors;
        source_to_cache_tensors.reserve(source_to_names.size());
        std::vector<ggml_tensor*> source_tensors;
        source_tensors.reserve(source_to_names.size());

        for (const auto& kv : source_to_names) {
            source_tensors.push_back(kv.first);
        }

        const std::string layout_key = graph_cache_layout_key(source_tensors);
        chunk = acquire_cache_chunk(source_tensors, layout_key);

        if (chunk.buffer == nullptr) {
            chunk.buffer = ggml_backend_alloc_ctx_tensors(chunk.ctx, runtime_backend);
            if (chunk.buffer == nullptr) {
                LOG_ERROR("%s alloc graph cut cache chunk failed, num_tensors = %zu",
                          get_desc().c_str(),
                          source_tensors.size());
                free_cache_chunk(chunk);
                return false;
            }
        }

        size_t tensor_idx = 0;
        for (const auto& kv : source_to_names) {
            ggml_tensor* source_tensor = kv.first;
            const auto& names          = kv.second;
            GGML_ASSERT(tensor_idx < chunk.cache_tensors.size());
            ggml_tensor* cache_tensor = chunk.cache_tensors[tensor_idx++];
            GGML_ASSERT(graph_cache_same_layout(source_tensor, cache_tensor));
            ggml_set_name(cache_tensor, names.front().c_str());
            for (const auto& name : names) {
                chunk.tensors[name] = cache_tensor;
            }
            source_to_cache_tensors.push_back({source_tensor, cache_tensor});
        }

        for (const auto& kv : source_to_cache_tensors) {
            if (!copy_cache_tensor_data(kv.first, kv.second)) {
                free_cache_chunk(chunk);
                return false;
            }
        }
        if (graph_cut_cache_sync_after_copy_enabled()) {
            ggml_backend_synchronize(runtime_backend);
        }

        for (const auto& kv : chunk.tensors) {
            cache_tensor_index_[kv.first] = kv.second;
        }
        cache_chunks_.push_back(std::move(chunk));
        return true;
    }

    bool copy_cache_tensors_to_cache_buffer(const std::unordered_set<std::string>* cache_keep_names = nullptr) {
        if (graph_cut_cache_compact_enabled()) {
            return compact_cache_tensors_to_cache_buffer(cache_keep_names);
        }
        return append_cache_tensors_to_cache_chunks(cache_keep_names);
    }

    void copy_data_to_backend_tensor(ggml_cgraph* gf,
                                     bool clear_after_copy = true,
                                     GraphCopyProfile* profile_out = nullptr) {
        GGML_ASSERT(gf != nullptr);
        if (profile_out != nullptr) {
            *profile_out = GraphCopyProfile{};
        }

        const int64_t t_graph_tensor_set_begin = profile_out != nullptr ? ggml_time_us() : 0;
        std::unordered_set<const ggml_tensor*> graph_tensor_set;
        const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
        const int n_nodes = ggml_graph_n_nodes(gf);
        graph_tensor_set.reserve(static_cast<size_t>(n_leafs + n_nodes));
        for (int i = 0; i < n_leafs; ++i) {
            graph_tensor_set.insert(sd::ggml_graph_cut::leaf_tensor(gf, i));
        }
        for (int i = 0; i < n_nodes; ++i) {
            graph_tensor_set.insert(ggml_graph_node(gf, i));
        }
        if (profile_out != nullptr) {
            const int64_t t_graph_tensor_set_end = ggml_time_us();
            profile_out->graph_tensor_set_us = t_graph_tensor_set_end - t_graph_tensor_set_begin;
            profile_out->graph_leafs = static_cast<size_t>(n_leafs);
            profile_out->graph_nodes = static_cast<size_t>(n_nodes);
            profile_out->graph_tensor_entries = graph_tensor_set.size();
            profile_out->backend_map_entries = backend_tensor_data_map.size();
        }

        const int64_t t_backend_map_scan_begin = profile_out != nullptr ? ggml_time_us() : 0;
        bool refreshed_runtime_const_binding = false;
        for (auto& kv : backend_tensor_data_map) {
            auto tensor = kv.first;
            auto data   = kv.second;

            if (graph_tensor_set.find(tensor) == graph_tensor_set.end()) {
                if (profile_out != nullptr) {
                    ++profile_out->skipped_not_in_graph;
                }
                continue;
            }

            ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
            if (buf == nullptr) {
                if (profile_out != nullptr) {
                    ++profile_out->skipped_no_buffer;
                }
                LOG_WARN("%s graph exec skip tensor copy: name=%s op=%s reason=buffer_not_set data=%p view_src=%p view_src_buffer=%p",
                         get_desc().c_str(),
                         tensor && tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                         tensor ? ggml_op_name(tensor->op) : "<null>",
                         data,
                         tensor ? tensor->view_src : nullptr,
                         (tensor && tensor->view_src) ? tensor->view_src->buffer : nullptr);
                continue;
            }

            const size_t nbytes = ggml_nbytes(tensor);
            if (should_cache_runtime_const_tensor(tensor, data)) {
                const int64_t t_runtime_const_begin = profile_out != nullptr ? ggml_time_us() : 0;
                RuntimeConstCacheEntry* entry = find_runtime_const_cache_entry(tensor, data);
                if (entry != nullptr) {
                    bind_tensor_to_runtime_const_cache(tensor, *entry);
                    refreshed_runtime_const_binding = true;
                    if (profile_out != nullptr) {
                        const int64_t t_runtime_const_end = ggml_time_us();
                        ++profile_out->runtime_const_cache_hits;
                        profile_out->runtime_const_cache_hit_bytes += nbytes;
                        profile_out->runtime_const_cache_hit_us += t_runtime_const_end - t_runtime_const_begin;
                    }
                    continue;
                }

                entry = upload_runtime_const_cache_entry(tensor, data);
                if (entry != nullptr) {
                    bind_tensor_to_runtime_const_cache(tensor, *entry);
                    refreshed_runtime_const_binding = true;
                    if (profile_out != nullptr) {
                        const int64_t t_runtime_const_end = ggml_time_us();
                        ++profile_out->runtime_const_cache_uploads;
                        profile_out->runtime_const_cache_upload_bytes += nbytes;
                        profile_out->runtime_const_cache_upload_us += t_runtime_const_end - t_runtime_const_begin;
                    }
                    continue;
                }
            }

            const int64_t t_tensor_set_begin = profile_out != nullptr ? ggml_time_us() : 0;
            ggml_backend_tensor_set(tensor, data, 0, nbytes);
            if (profile_out != nullptr) {
                const int64_t t_tensor_set_end = ggml_time_us();
                profile_out->backend_tensor_set_us += t_tensor_set_end - t_tensor_set_begin;
                ++profile_out->copied_tensors;
                profile_out->copied_bytes += nbytes;
            }
        }
        if (profile_out != nullptr) {
            const int64_t t_backend_map_scan_end = ggml_time_us();
            profile_out->backend_map_scan_us = t_backend_map_scan_end - t_backend_map_scan_begin -
                                               profile_out->backend_tensor_set_us;
        }

        if (clear_after_copy) {
            backend_tensor_data_map.clear();
        }

        if (refreshed_runtime_const_binding) {
            refresh_graph_view_bindings(gf);
        }
    }

    bool offload_all_params() {
        restore_partial_params();
        if (params_backend == runtime_backend) {
            return true;
        }
        if (params_on_runtime_backend) {
            return true;
        }
        GGML_ASSERT(runtime_params_buffer == nullptr);
        int64_t t0         = ggml_time_ms();
        size_t num_tensors = ggml_tensor_num(offload_ctx);
        if (num_tensors == 0) {
            for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
                GGML_ASSERT(t->view_src == nullptr);
                ggml_dup_tensor(offload_ctx, t);
            }
        }
        num_tensors = ggml_tensor_num(offload_ctx);
        GGML_ASSERT(num_tensors == ggml_tensor_num(params_ctx));

        runtime_params_buffer = ggml_backend_alloc_ctx_tensors(offload_ctx, runtime_backend);

        if (runtime_params_buffer == nullptr) {
            LOG_ERROR("%s alloc runtime params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }
        ggml_backend_buffer_set_usage(runtime_params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_tensor* t         = ggml_get_first_tensor(params_ctx);
        ggml_tensor* offload_t = ggml_get_first_tensor(offload_ctx);

        while (t != nullptr && offload_t != nullptr) {
            ggml_backend_tensor_copy(t, offload_t);
            std::swap(t->buffer, offload_t->buffer);
            std::swap(t->data, offload_t->data);
            std::swap(t->extra, offload_t->extra);

            t         = ggml_get_next_tensor(params_ctx, t);
            offload_t = ggml_get_next_tensor(offload_ctx, offload_t);
        }

        int64_t t1 = ggml_time_ms();

        size_t params_buffer_size = ggml_backend_buffer_get_size(runtime_params_buffer);
        LOG_INFO("%s offload params (%6.2f MB, %i tensors) to runtime backend (%s), taking %.2fs",
                 get_desc().c_str(),
                 params_buffer_size / (1024.f * 1024.f),
                 num_tensors,
                 ggml_backend_name(runtime_backend),
                 (t1 - t0) * 1.0f / 1000);

        params_on_runtime_backend = true;

        return true;
    }

    bool offload_partial_params(const std::vector<ggml_tensor*>& tensors) {
        restore_partial_params();
        if (params_backend == runtime_backend) {
            return true;
        }
        if (tensors.empty()) {
            return true;
        }
        GGML_ASSERT(!params_on_runtime_backend);
        GGML_ASSERT(partial_runtime_params_buffer == nullptr);

        std::vector<ggml_tensor*> unique_tensors;
        std::unordered_set<ggml_tensor*> seen_tensors;
        unique_tensors.reserve(tensors.size());
        seen_tensors.reserve(tensors.size());
        for (ggml_tensor* tensor : tensors) {
            if (tensor == nullptr) {
                continue;
            }
            if (seen_tensors.insert(tensor).second) {
                unique_tensors.push_back(tensor);
            }
        }
        if (unique_tensors.empty()) {
            return true;
        }

        ggml_init_params params;
        params.mem_size   = std::max<size_t>(1, unique_tensors.size()) * ggml_tensor_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        partial_offload_ctx = ggml_init(params);
        GGML_ASSERT(partial_offload_ctx != nullptr);

        partial_offload_pairs.clear();
        partial_offload_pairs.reserve(unique_tensors.size());

        for (ggml_tensor* tensor : unique_tensors) {
            GGML_ASSERT(tensor->view_src == nullptr);
            ggml_tensor* offload_tensor = ggml_dup_tensor(partial_offload_ctx, tensor);
            ggml_set_name(offload_tensor, tensor->name);
            partial_offload_pairs.push_back({tensor, offload_tensor});
        }

        partial_runtime_params_buffer = ggml_backend_alloc_ctx_tensors(partial_offload_ctx, runtime_backend);
        if (partial_runtime_params_buffer == nullptr) {
            LOG_ERROR("%s alloc partial runtime params backend buffer failed, num_tensors = %zu",
                      get_desc().c_str(),
                      partial_offload_pairs.size());
            ggml_free(partial_offload_ctx);
            partial_offload_ctx = nullptr;
            partial_offload_pairs.clear();
            return false;
        }
        ggml_backend_buffer_set_usage(partial_runtime_params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (auto& pair : partial_offload_pairs) {
            ggml_tensor* tensor         = pair.first;
            ggml_tensor* offload_tensor = pair.second;

            // Async H2D from pinned params (Phase 0: still synced below before use).
            ggml_backend_tensor_set_async(runtime_backend, offload_tensor, tensor->data, 0, ggml_nbytes(tensor));
            std::swap(tensor->buffer, offload_tensor->buffer);
            std::swap(tensor->data, offload_tensor->data);
            std::swap(tensor->extra, offload_tensor->extra);
        }
        // Phase 0 conservative barrier: guarantee all async copies land before
        // compute reads the weights. Removed once double-buffering overlaps copy/compute.
        ggml_backend_synchronize(runtime_backend);

        size_t params_buffer_size = ggml_backend_buffer_get_size(partial_runtime_params_buffer);
        LOG_DEBUG("%s offload partial params (%6.2f MB, %zu tensors) to runtime backend (%s)",
                  get_desc().c_str(),
                  params_buffer_size / (1024.f * 1024.f),
                  partial_offload_pairs.size(),
                  ggml_backend_name(runtime_backend));

        return true;
    }

    void restore_all_params() {
        restore_partial_params();
        if (!params_on_runtime_backend) {
            return;
        }
        ggml_tensor* t         = ggml_get_first_tensor(params_ctx);
        ggml_tensor* offload_t = ggml_get_first_tensor(offload_ctx);

        while (t != nullptr && offload_t != nullptr) {
            t->buffer         = offload_t->buffer;
            t->data           = offload_t->data;
            t->extra          = offload_t->extra;
            offload_t->buffer = nullptr;
            offload_t->data   = nullptr;
            offload_t->extra  = nullptr;

            t         = ggml_get_next_tensor(params_ctx, t);
            offload_t = ggml_get_next_tensor(offload_ctx, offload_t);
        }

        if (runtime_params_buffer != nullptr) {
            ggml_backend_buffer_free(runtime_params_buffer);
            runtime_params_buffer = nullptr;
        }
        params_on_runtime_backend = false;
    }

#if defined(ED_ENABLE_ASYNC_OFFLOAD)
    static bool async_offload_debug() {
        static const bool dbg = [] {
            const char* e = std::getenv("ED_ASYNC_OFFLOAD_DEBUG");
            return e != nullptr && e[0] == '1';
        }();
        return dbg;
    }

    // Prefetch one segment's weights into `slot`: dup tensors, alloc GPU buffer,
    // async H2D from the STABLE cpu source (async_cpu_src_), record completion event.
    // No swap here — swap happens in async_apply_slot after the event is waited.
    bool async_prefetch_slot(AsyncOffloadSlot& slot,
                             const std::vector<ggml_tensor*>& tensors,
                             size_t seg_idx) {
        const int lvl = async_offload_level();
        // dedup within this segment
        std::vector<ggml_tensor*> uniq;
        std::unordered_set<ggml_tensor*> seen;
        for (ggml_tensor* t : tensors) {
            if (t != nullptr && seen.insert(t).second) uniq.push_back(t);
        }
        if (uniq.empty()) {
            slot.seg_idx = seg_idx;
            slot.active  = true;
            slot.applied = false;
            return true;
        }

        ggml_init_params ip;
        ip.mem_size   = std::max<size_t>(1, uniq.size()) * ggml_tensor_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        const int64_t t_pf0 = async_offload_debug() ? ggml_time_us() : 0;
        slot.ctx = ggml_init(ip);
        GGML_ASSERT(slot.ctx != nullptr);
        slot.pairs.clear();
        slot.pairs.reserve(uniq.size());
        for (ggml_tensor* t : uniq) {
            GGML_ASSERT(t->view_src == nullptr);
            ggml_tensor* dup = ggml_dup_tensor(slot.ctx, t);
            ggml_set_name(dup, t->name);
            slot.pairs.push_back({t, dup});
        }
        const int64_t t_pf_dup = async_offload_debug() ? ggml_time_us() : 0;
        // Reuse a persistent per-slot GPU buffer. Allocating a fresh buffer every
        // segment triggers a synchronizing cudaMalloc that stalls the compute stream
        // (profiled: compute 277ms -> 8674ms). Compute needed size exactly as
        // ggml_tallocr would (per-tensor alloc_size padded to buffer alignment).
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(runtime_backend);
        const size_t talign = ggml_backend_buft_get_alignment(buft);
        size_t need = 0;
        for (auto& pr : slot.pairs) {
            need += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, pr.second), talign);
        }
        need += talign;  // headroom for the initial aligned_offset in ggml_tallocr
        if (slot.buffer == nullptr || slot.buffer_capacity < need) {
            if (slot.buffer != nullptr) ggml_backend_buffer_free(slot.buffer);
            slot.buffer = ggml_backend_buft_alloc_buffer(buft, need);
            slot.buffer_capacity = (slot.buffer != nullptr) ? need : 0;
        }
        const int64_t t_pf_alloc = async_offload_debug() ? ggml_time_us() : 0;
        if (slot.buffer == nullptr) {
            LOG_ERROR("%s async_prefetch_slot: alloc buffer failed (seg=%zu, %zu tensors, need=%.1fMB)",
                      get_desc().c_str(), seg_idx, uniq.size(), need / (1024.0 * 1024.0));
            ggml_free(slot.ctx);
            slot.ctx = nullptr;
            slot.pairs.clear();
            return false;
        }
        ggml_backend_buffer_set_usage(slot.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // Bind each dup into the persistent buffer (tallocr resets offset to 0 each call).
        {
            struct ggml_tallocr talloc = ggml_tallocr_new(slot.buffer);
            for (auto& pr : slot.pairs) {
                const enum ggml_status st = ggml_tallocr_alloc(&talloc, pr.second);
                GGML_ASSERT(st == GGML_STATUS_SUCCESS);
            }
        }

        size_t bytes = 0;
        for (auto& pr : slot.pairs) {
            ggml_tensor* orig = pr.first;
            ggml_tensor* dup  = pr.second;
            // STABLE cpu source: never read orig->data (may already be swapped to GPU
            // by an adjacent slot); use the precomputed map.
            void* cpu_src = async_cpu_src_.count(orig) ? async_cpu_src_[orig] : orig->data;
            const size_t n = ggml_nbytes(orig);
            if (lvl >= 2 && async_copy_stream_ != nullptr) {
                ed_async_offload_h2d(dup->data, cpu_src, n, async_copy_stream_);  // copy stream
            } else {
                // Ladder 1: use ggml async on the compute stream (no independent stream yet)
                ggml_backend_tensor_set_async(runtime_backend, dup, cpu_src, 0, n);
            }
            bytes += n;
        }
        if (lvl >= 2 && async_copy_stream_ != nullptr) {
            if (slot.copy_done_event == nullptr) slot.copy_done_event = ed_async_offload_event_create();
            ed_async_offload_event_record(slot.copy_done_event, async_copy_stream_);
        }
        slot.seg_idx = seg_idx;
        slot.active  = true;
        slot.applied = false;
        if (async_offload_debug()) {
            const int64_t t_pf_copy = ggml_time_us();
            LOG_INFO("%s [async] prefetch seg=%zu tensors=%zu bytes=%.1fMB lvl=%d | dup=%.0fus alloc=%.0fus copy_issue=%.0fus",
                     get_desc().c_str(), seg_idx, slot.pairs.size(), bytes / (1024.0 * 1024.0), lvl,
                     (double)(t_pf_dup - t_pf0), (double)(t_pf_alloc - t_pf_dup), (double)(t_pf_copy - t_pf_alloc));
        }
        return true;
    }

    // Ensure the slot's H2D is complete, then swap the GPU dups into the live params.
    void async_apply_slot(AsyncOffloadSlot& slot) {
        if (!slot.active || slot.applied) return;
        const int lvl = async_offload_level();
        const int64_t t_ap0 = async_offload_debug() ? ggml_time_us() : 0;
        if (lvl >= 3 && slot.copy_done_event != nullptr) {
            ed_async_offload_compute_wait(runtime_backend, slot.copy_done_event);  // true overlap
        } else if (lvl == 2 && slot.copy_done_event != nullptr) {
            ed_async_offload_event_synchronize(slot.copy_done_event);
        } else {
            // Ladder 1: conservative full barrier (proves swap/dedup/lifecycle bookkeeping)
            ggml_backend_synchronize(runtime_backend);
        }
        const int64_t t_ap_wait = async_offload_debug() ? ggml_time_us() : 0;
        for (auto& pr : slot.pairs) {
            ggml_tensor* orig = pr.first;
            ggml_tensor* dup  = pr.second;
            std::swap(orig->buffer, dup->buffer);
            std::swap(orig->data, dup->data);
            std::swap(orig->extra, dup->extra);
        }
        slot.applied = true;
        if (async_offload_debug()) {
            LOG_INFO("%s [async] apply  seg=%zu wait=%.0fus swap=%.0fus",
                     get_desc().c_str(), slot.seg_idx,
                     (double)(t_ap_wait - t_ap0), (double)(ggml_time_us() - t_ap_wait));
        }
    }

    // Swap CPU pointers back. Keeps the slot's GPU buffer alive for reuse (freeing
    // it per-segment triggers a synchronizing cudaFree that serializes the pipeline);
    // buffers are released once at loop end via async_free_slot_buffers().
    void async_restore_slot(AsyncOffloadSlot& slot) {
        if (!slot.active) return;
        if (slot.applied) {
            for (auto& pr : slot.pairs) {
                ggml_tensor* orig = pr.first;
                ggml_tensor* dup  = pr.second;
                orig->buffer = dup->buffer;
                orig->data   = dup->data;
                orig->extra  = dup->extra;
                dup->buffer = nullptr;
                dup->data   = nullptr;
                dup->extra  = nullptr;
            }
        }
        // Do NOT free slot.buffer here — persistent for reuse; freed at loop end.
        if (slot.ctx != nullptr) {
            ggml_free(slot.ctx);
            slot.ctx = nullptr;
        }
        if (async_offload_debug()) {
            LOG_INFO("%s [async] restore seg=%zu (buffer kept)", get_desc().c_str(), slot.seg_idx);
        }
        slot.pairs.clear();
        slot.applied = false;
        slot.active  = false;
        slot.seg_idx = SIZE_MAX;
    }
#endif  // ED_ENABLE_ASYNC_OFFLOAD

    void restore_partial_params() {
        if (partial_offload_pairs.empty()) {
            if (partial_runtime_params_buffer != nullptr) {
                ggml_backend_buffer_free(partial_runtime_params_buffer);
                partial_runtime_params_buffer = nullptr;
            }
            if (partial_offload_ctx != nullptr) {
                ggml_free(partial_offload_ctx);
                partial_offload_ctx = nullptr;
            }
            return;
        }

        for (auto& pair : partial_offload_pairs) {
            ggml_tensor* tensor         = pair.first;
            ggml_tensor* offload_tensor = pair.second;

            tensor->buffer         = offload_tensor->buffer;
            tensor->data           = offload_tensor->data;
            tensor->extra          = offload_tensor->extra;
            offload_tensor->buffer = nullptr;
            offload_tensor->data   = nullptr;
            offload_tensor->extra  = nullptr;
        }

        if (partial_runtime_params_buffer != nullptr) {
            ggml_backend_buffer_free(partial_runtime_params_buffer);
            partial_runtime_params_buffer = nullptr;
        }
        partial_offload_pairs.clear();

        if (partial_offload_ctx != nullptr) {
            ggml_free(partial_offload_ctx);
            partial_offload_ctx = nullptr;
        }
    }

    bool plan_has_comm_ops(const GraphCutPlan& plan) const {
        for (const auto& segment : plan.segments) {
            if (!segment.comm_ops.empty()) {
                return true;
            }
        }
        return false;
    }

    bool should_use_graph_cut_segmented_compute(const GraphCutPlan& plan) {
        if (!plan.has_cuts || !plan.valid || plan.segments.empty()) {
            return false;
        }

        const bool has_graph_comm = process_group_ != nullptr &&
                                    process_group_->enabled() &&
                                    plan_has_comm_ops(plan);
        if (has_graph_comm) {
            return true;
        }

        const bool offload_active = max_graph_vram_bytes > 0 &&
                                    params_backend != runtime_backend &&
                                    !ggml_backend_is_cpu(runtime_backend);
        if (!offload_active) {
            return false;
        }
        if (plan.segments.size() > 1) {
            return true;
        }
        // Single segment: the whole-graph fallback path (execute_graph with an empty
        // runtime_param list) calls offload_all_params(), which stages EVERY weight in
        // params_ctx to the GPU at once. When that full set exceeds the budget, staging
        // it whole spikes VRAM past the cap (e.g. qwen-image-edit's vision encode stages
        // all 7979MB though the segment only references ~800MB). Route the single segment
        // through the segmented path too so it partial-offloads just the weights it uses.
        // If the full set already fits the budget, whole-graph offload is cheaper (no
        // per-segment overhead), so keep the fallback.
        const size_t full_param_bytes =
            params_buffer != nullptr ? ggml_backend_buffer_get_size(params_buffer) : 0;
        return full_param_bytes > max_graph_vram_bytes;
    }

    bool can_attempt_graph_cut_segmented_compute() const {
        return (max_graph_vram_bytes > 0 &&
                params_backend != runtime_backend &&
                !ggml_backend_is_cpu(runtime_backend)) ||
               (process_group_ != nullptr && process_group_->enabled());
    }

    bool resolve_graph_cut_plan(ggml_cgraph* gf,
                                GraphCutPlan* plan_out) {
        GGML_ASSERT(plan_out != nullptr);
        GGML_ASSERT(gf != nullptr);
        *plan_out = sd::ggml_graph_cut::resolve_plan(runtime_backend,
                                                     gf,
                                                     &graph_cut_plan_cache_,
                                                     max_graph_vram_bytes,
                                                     params_tensor_set_,
                                                     get_desc().c_str());
        return true;
    }

    void reset_segment_runtime_tensors(const GraphCutSegment& segment,
                                       ggml_cgraph* gf) {
        GGML_ASSERT(gf != nullptr);

        auto reset_runtime_binding = [](ggml_tensor* tensor) {
            if (tensor == nullptr) {
                return;
            }
            tensor->buffer = nullptr;
            tensor->data   = nullptr;
            tensor->extra  = nullptr;
        };

        for (const auto& input : segment.input_refs) {
            ggml_tensor* input_tensor = sd::ggml_graph_cut::input_tensor(gf, input);
            if (input_tensor == nullptr) {
                continue;
            }
            switch (input.type) {
                case GraphCutSegment::INPUT_PREVIOUS_CUT:
                case GraphCutSegment::INPUT_EXTERNAL:
                    reset_runtime_binding(input_tensor);
                    break;
                case GraphCutSegment::INPUT_PARAM:
                    break;
            }
        }

        const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
        for (int leaf_idx = 0; leaf_idx < n_leafs; ++leaf_idx) {
            ggml_tensor* leaf = sd::ggml_graph_cut::leaf_tensor(gf, leaf_idx);
            if (leaf == nullptr || leaf->name[0] == '\0') {
                continue;
            }
            static constexpr const char* k_runner_builtin_prefix = "ggml_runner_build_in_tensor:";
            if (std::strncmp(leaf->name,
                             k_runner_builtin_prefix,
                             std::strlen(k_runner_builtin_prefix)) == 0) {
                reset_runtime_binding(leaf);
            }
        }

        for (int node_idx : segment.internal_node_indices) {
            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            reset_runtime_binding(node);
        }
    }

    bool bind_segment_cached_inputs(ggml_cgraph* gf, const GraphCutSegment& segment) {
        GGML_ASSERT(gf != nullptr);
        std::unordered_map<ggml_tensor*, ggml_tensor*> cached_view_sources;
        for (const auto& input : segment.input_refs) {
            ggml_tensor* input_tensor = sd::ggml_graph_cut::input_tensor(gf, input);
            if (input_tensor == nullptr) {
                continue;
            }
            switch (input.type) {
                case GraphCutSegment::INPUT_PREVIOUS_CUT: {
                    ggml_tensor* cache_tensor = get_cache_tensor_by_name(input.display_name);
                    if (cache_tensor == nullptr) {
                        LOG_ERROR("%s missing graph cut cache tensor: %s",
                                  get_desc().c_str(),
                                  input.display_name.c_str());
                        return false;
                    }
                    if (input_tensor->view_src != nullptr) {
                        ggml_tensor* original_view_src = input_tensor->view_src;
                        cached_view_sources[original_view_src] = cache_tensor;
                        input_tensor->view_src = cache_tensor;
                        input_tensor->buffer   = cache_tensor->buffer;
                        input_tensor->data     = cache_tensor->data == nullptr
                                                     ? nullptr
                                                     : static_cast<void*>(static_cast<char*>(cache_tensor->data) + input_tensor->view_offs);
                        input_tensor->extra    = cache_tensor->extra;
                    } else {
                        input_tensor->buffer = cache_tensor->buffer;
                        input_tensor->data   = cache_tensor->data;
                        input_tensor->extra  = cache_tensor->extra;
                    }
                    for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                        input_tensor->src[src_idx] = nullptr;
                    }
                    input_tensor->op = GGML_OP_NONE;
                    break;
                }
                case GraphCutSegment::INPUT_EXTERNAL:
                case GraphCutSegment::INPUT_PARAM:
                    break;
            }
        }
        if (!cached_view_sources.empty()) {
            auto rebind_cached_view = [&cached_view_sources](ggml_tensor* tensor) {
                if (tensor == nullptr || tensor->view_src == nullptr) {
                    return;
                }
                auto it = cached_view_sources.find(tensor->view_src);
                if (it == cached_view_sources.end()) {
                    return;
                }
                tensor->view_src = it->second;
                tensor->buffer   = it->second->buffer;
                tensor->data     = it->second->data == nullptr
                                       ? nullptr
                                       : static_cast<void*>(static_cast<char*>(it->second->data) + tensor->view_offs);
                tensor->extra    = it->second->extra;
            };
            for (const auto& input : segment.input_refs) {
                rebind_cached_view(sd::ggml_graph_cut::input_tensor(gf, input));
            }
            for (int node_idx : segment.internal_node_indices) {
                ggml_tensor* node = ggml_graph_node(gf, node_idx);
                rebind_cached_view(node);
                if (node == nullptr) {
                    continue;
                }
                for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                    rebind_cached_view(node->src[src_idx]);
                }
            }
        }
        for (int node_idx : segment.internal_node_indices) {
            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            if (node == nullptr || node->view_src == nullptr || node->view_src->data == nullptr) {
                continue;
            }
            node->buffer = node->view_src->buffer;
            node->data   = static_cast<void*>(static_cast<char*>(node->view_src->data) + node->view_offs);
            node->extra  = node->view_src->extra;
        }
        return true;
    }

    template <typename T>
    std::optional<sd::Tensor<T>> execute_graph(ggml_cgraph* gf,
                                            int n_threads,
                                            bool free_compute_buffer_immediately,
                                            const std::vector<ggml_tensor*>& runtime_param_tensors,
                                            bool preserve_backend_tensor_data_map,
                                            bool no_return                                          = false,
                                            const std::unordered_set<std::string>* cache_keep_names = nullptr,
                                            post_compute_cb_t post_compute_cb                      = nullptr,
                                            pre_compute_cb_t pre_compute_cb                        = nullptr,
                                            GraphExecuteProfile* profile_out                       = nullptr,
                                            bool params_already_resident                           = false) {
        if (profile_out != nullptr) {
            *profile_out = GraphExecuteProfile{};
        }
        int64_t t_execute_begin              = ggml_time_ms();
        // params_already_resident: the segment loop (double-buffer prefetch) has
        // already swapped this segment's weights onto the GPU, so skip the internal
        // offload/restore entirely. Default false keeps every other caller identical.
        const bool weights_already_resident = params_already_resident || params_on_runtime_backend;
        const bool use_partial_param_offload = !runtime_param_tensors.empty() && !weights_already_resident;
        int64_t t_offload_begin              = ggml_time_ms();
        if (weights_already_resident) {
            // Weights were pre-swapped by the async segment caller or pinned for the
            // current component phase; do not stage the same tensors a second time.
        } else if (use_partial_param_offload) {
            if (!offload_partial_params(runtime_param_tensors)) {
                LOG_ERROR("%s offload partial params to runtime backend failed", get_desc().c_str());
                return std::nullopt;
            }
        } else {
            if (!offload_all_params()) {
                LOG_ERROR("%s offload params to runtime backend failed", get_desc().c_str());
                return std::nullopt;
            }
        }
        int64_t t_offload_end = ggml_time_ms();
        if (profile_out != nullptr) {
            profile_out->offload_ms = t_offload_end - t_offload_begin;
        }

        // After a weight is streamed to the device by offload_*_params(), its
        // buffer/data pointers change, but any reshape/view tensor built over that
        // weight while it still lived on the host params buffer keeps the stale host
        // binding. ggml_gallocr_init_tensor() only re-derives a view when the view's
        // ->buffer is NULL, so a view that already carries the pre-swap host buffer
        // is never fixed up. Re-derive view bindings from view_src so segmented
        // graph-cut runs see the same weight views as whole-graph offload. This only
        // touches views whose view_src is already materialized, so compute
        // intermediates are untouched. Skipped when params were pre-swapped by the
        // async double-buffer caller or when no offload is configured.
        if (!params_already_resident &&
            params_backend != runtime_backend) {
            refresh_graph_view_bindings(gf);
        }

        int64_t t_alloc_begin = ggml_time_ms();
        if (!alloc_compute_buffer(gf)) {
            LOG_ERROR("%s alloc compute buffer failed", get_desc().c_str());
            if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }

        if (!ggml_gallocr_alloc_graph(compute_allocr, gf)) {
            LOG_ERROR("%s alloc compute graph failed", get_desc().c_str());
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        int64_t t_alloc_end = ggml_time_ms();
        if (profile_out != nullptr) {
            profile_out->alloc_ms = t_alloc_end - t_alloc_begin;
        }

        int64_t t_copy_begin = ggml_time_ms();
        GraphCopyProfile copy_detail;
        copy_data_to_backend_tensor(gf,
                                    !preserve_backend_tensor_data_map,
                                    profile_out != nullptr ? &copy_detail : nullptr);
        int64_t t_copy_data_end = ggml_time_ms();
        if (profile_out != nullptr) {
            profile_out->copy_detail = copy_detail;
        }
        if (pre_compute_cb) {
            const int64_t t_pre_compute_begin = profile_out != nullptr ? ggml_time_us() : 0;
            if (!pre_compute_cb(gf)) {
                LOG_ERROR("%s pre compute callback failed", get_desc().c_str());
                if (free_compute_buffer_immediately) {
                    free_compute_buffer();
                } else if (use_partial_param_offload) {
                    restore_partial_params();
                }
                return std::nullopt;
            }
            if (profile_out != nullptr) {
                const int64_t t_pre_compute_end = ggml_time_us();
                profile_out->pre_compute_callback_us = t_pre_compute_end - t_pre_compute_begin;
            }
        }
        int64_t t_copy_end = ggml_time_ms();
        if (profile_out != nullptr) {
            profile_out->copy_ms = t_copy_end - t_copy_begin;
            ED_UNUSED(t_copy_data_end);
        }
        if (ggml_backend_is_cpu(runtime_backend)) {
            ggml_backend_cpu_set_n_threads(runtime_backend, n_threads);
        }

        int64_t t_compute_begin = ggml_time_ms();
        ggml_status status      = ggml_backend_graph_compute(runtime_backend, gf);
        int64_t t_compute_end   = ggml_time_ms();
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERROR("%s compute failed: %s", get_desc().c_str(), ggml_status_to_string(status));
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        if (profile_out != nullptr) {
            profile_out->compute_ms = t_compute_end - t_compute_begin;
        }

        if (post_compute_cb) {
            int64_t t_comm_begin = ggml_time_ms();
            if (!post_compute_cb(gf)) {
                LOG_ERROR("%s post compute callback failed", get_desc().c_str());
                if (free_compute_buffer_immediately) {
                    free_compute_buffer();
                } else if (use_partial_param_offload) {
                    restore_partial_params();
                }
                return std::nullopt;
            }
            int64_t t_comm_end = ggml_time_ms();
            if (profile_out != nullptr) {
                profile_out->post_ms = t_comm_end - t_comm_begin;
            }
        }

        auto result = ggml_get_tensor(compute_ctx, final_result_name.c_str());
        std::optional<sd::Tensor<T>> output;
        if (!no_return) {
            output = sd::make_sd_tensor_from_ggml<T>(result);
        } else {
            output = sd::Tensor<T>();
        }

        int64_t t_cache_begin = ggml_time_ms();
        if (!copy_cache_tensors_to_cache_buffer(cache_keep_names)) {
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        int64_t t_cache_end = ggml_time_ms();
        if (profile_out != nullptr) {
            profile_out->cache_ms = t_cache_end - t_cache_begin;
            profile_out->cache_live_bytes = graph_cache_live_bytes();
            profile_out->cache_buffer_bytes = graph_cache_buffer_bytes();
            profile_out->cache_chunks = graph_cache_chunk_count();
            profile_out->cache_pool_bytes = cache_chunk_pool_bytes_;
            profile_out->cache_pool_chunks = graph_cache_pool_chunk_count();
        }

        if (free_compute_buffer_immediately) {
            free_compute_buffer();
        } else if (use_partial_param_offload) {
            restore_partial_params();
        }
        if (profile_out != nullptr) {
            profile_out->total_ms = ggml_time_ms() - t_execute_begin;
            if (runner_profile_enabled()) {
                LOG_INFO("%s runner profile total=%lldms offload=%lldms alloc=%lldms copy=%lldms compute=%lldms post=%lldms cache=%lldms copied_tensors=%zu copied_bytes=%.2fMiB cache_live=%.2fMiB cache_buffer=%.2fMiB",
                         get_desc().c_str(),
                         (long long) profile_out->total_ms,
                         (long long) profile_out->offload_ms,
                         (long long) profile_out->alloc_ms,
                         (long long) profile_out->copy_ms,
                         (long long) profile_out->compute_ms,
                         (long long) profile_out->post_ms,
                         (long long) profile_out->cache_ms,
                         profile_out->copy_detail.copied_tensors,
                         profile_out->copy_detail.copied_bytes / 1024.0 / 1024.0,
                         profile_out->cache_live_bytes / 1024.0 / 1024.0,
                         profile_out->cache_buffer_bytes / 1024.0 / 1024.0);
            }
        }
        return output;
    }

    template <typename T>
    std::optional<sd::Tensor<T>> compute_with_graph_cuts(ggml_cgraph* gf,
                                                         const GraphCutPlan& plan,
                                                         int n_threads,
                                                         bool free_compute_buffer_immediately,
                                                         bool no_return = false,
                                                         int64_t plan_ms = 0) {
        GGML_ASSERT(gf != nullptr);

        const bool collect_profile = graph_cut_profile_enabled();
        const int materialize_top_n = graph_cut_profile_materialize_top_n();
        const bool collect_materialize_breakdown =
            collect_profile && materialize_top_n > 0;
        const bool collect_compute_breakdown =
            collect_profile && (graph_cut_profile_compute_top_n() > 0 || collect_materialize_breakdown);
        const int64_t t_profile_begin = ggml_time_ms();
        std::vector<GraphCutSegmentProfile> segment_profiles;
        if (collect_profile) {
            segment_profiles.reserve(plan.segments.size());
        }

        free_compute_buffer();
        reset_graph_cut_run_cache();

        std::optional<sd::Tensor<T>> output = sd::Tensor<T>();

#if defined(ED_ENABLE_ASYNC_OFFLOAD)
        // P0-A Phase 2-3: async double-buffered weight prefetch across segments.
        // Only engages when offloading (params on CPU) and ED_ASYNC_OFFLOAD >= 1.
        const int async_lvl = async_offload_level();
        const bool async_prefetch_on = async_lvl >= 1 &&
                                       params_backend != runtime_backend &&
                                       !params_on_runtime_backend &&
                                       plan.segments.size() > 1;
        std::vector<std::vector<ggml_tensor*>> async_seg_params;
        if (async_prefetch_on) {
            // Precompute every segment's param list + a STABLE cpu-source map
            // (a tensor's own ->data is clobbered by swap, so snapshot it now).
            async_seg_params.resize(plan.segments.size());
            async_cpu_src_.clear();
            for (size_t i = 0; i < plan.segments.size(); ++i) {
                async_seg_params[i] =
                    sd::ggml_graph_cut::runtime_param_tensors(gf, plan.segments[i], get_desc().c_str());
                for (ggml_tensor* t : async_seg_params[i]) {
                    if (t != nullptr) async_cpu_src_.emplace(t, t->data);
                }
            }
            if (async_lvl >= 2 && async_copy_stream_ == nullptr) {
                ggml_backend_dev_t dev = ggml_backend_get_device(runtime_backend);
                int devidx = 0;  // single-GPU path; multi-GPU would map dev->index
                (void)dev;
                async_copy_stream_ = ed_async_offload_stream_create(devidx);
            }
            if (async_offload_debug()) {
                LOG_INFO("%s [async] prefetch ENABLED lvl=%d segments=%zu copy_stream=%p",
                         get_desc().c_str(), async_lvl, plan.segments.size(), async_copy_stream_);
            }
            // Prefetch segment 0 before the loop.
            async_prefetch_slot(async_slots_[0], async_seg_params[0], 0);
        }
#endif

        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); ++seg_idx) {
            int64_t t_segment_begin = ggml_time_ms();
            const auto& segment     = plan.segments[seg_idx];
            const int64_t t_collect_future_begin = collect_profile ? ggml_time_us() : 0;
            const auto& future_cut_names = segment.future_input_names;
            const int64_t t_collect_future_end = collect_profile ? ggml_time_us() : 0;
            GraphCutSegmentProfile segment_profile;
            if (collect_profile) {
                segment_profile.name = segment.group_name;
                segment_profile.nodes = segment.internal_node_indices.size();
                segment_profile.comm_ops = segment.comm_ops.size();
                segment_profile.output_bytes = segment.output_bytes;
                segment_profile.comm_names = graph_cut_comm_names(segment);
                if (collect_compute_breakdown) {
                    segment_profile.io_summary = graph_cut_segment_io_summary(gf, segment);
                    segment_profile.op_histogram =
                        graph_cut_segment_op_histogram(gf,
                                                       segment,
                                                       &segment_profile.op_counts,
                                                       &segment_profile.math_ops,
                                                       &segment_profile.layout_ops);
                }
                if (collect_materialize_breakdown) {
                    GraphCutMaterializeProfile materialize =
                        graph_cut_segment_materialize_profile(gf,
                                                              segment,
                                                              static_cast<size_t>(materialize_top_n));
                    segment_profile.materialize_ops = materialize.ops;
                    segment_profile.materialize_bytes = materialize.bytes;
                    segment_profile.cont_ops = materialize.cont_ops;
                    segment_profile.cont_bytes = materialize.cont_bytes;
                    segment_profile.cpy_ops = materialize.cpy_ops;
                    segment_profile.cpy_bytes = materialize.cpy_bytes;
                    segment_profile.concat_ops = materialize.concat_ops;
                    segment_profile.concat_bytes = materialize.concat_bytes;
                    segment_profile.dup_ops = materialize.dup_ops;
                    segment_profile.dup_bytes = materialize.dup_bytes;
                    segment_profile.materialize_boundary_output_bytes = materialize.boundary_output_bytes;
                    segment_profile.materialize_cached_output_bytes = materialize.cached_output_bytes;
                    segment_profile.materialize_comm_input_bytes = materialize.comm_input_bytes;
                    segment_profile.materialize_comm_output_bytes = materialize.comm_output_bytes;
                    segment_profile.repeated_materialize_source_groups = materialize.repeated_source_groups;
                    segment_profile.repeated_materialize_source_ops = materialize.repeated_source_ops;
                    segment_profile.repeated_materialize_source_bytes = materialize.repeated_source_bytes;
                    segment_profile.materialize_after_materialize_ops = materialize.materialize_after_materialize_ops;
                    segment_profile.materialize_after_materialize_bytes = materialize.materialize_after_materialize_bytes;
                    segment_profile.cont_from_cont_ops = materialize.cont_from_cont_ops;
                    segment_profile.cont_from_cont_bytes = materialize.cont_from_cont_bytes;
                    segment_profile.concat_to_cont_ops = materialize.concat_to_cont_ops;
                    segment_profile.concat_to_cont_bytes = materialize.concat_to_cont_bytes;
                    segment_profile.permute_view_to_cont_ops = materialize.permute_view_to_cont_ops;
                    segment_profile.permute_view_to_cont_bytes = materialize.permute_view_to_cont_bytes;
                    segment_profile.materialize_view_materialize_ops = materialize.materialize_view_materialize_ops;
                    segment_profile.materialize_view_materialize_bytes = materialize.materialize_view_materialize_bytes;
                    segment_profile.cont_permute_cont_ops = materialize.cont_permute_cont_ops;
                    segment_profile.cont_permute_cont_bytes = materialize.cont_permute_cont_bytes;
                    segment_profile.materialize_stage_summary =
                        graph_cut_materialize_stage_summary(materialize);
                    segment_profile.materialize_stage_details =
                        graph_cut_materialize_stage_detail_summary(materialize);
                    segment_profile.materialize_stages = std::move(materialize.stages);
                    segment_profile.materialize_top_nodes = std::move(materialize.top_nodes);
                    segment_profile.repeated_materialize_sources = std::move(materialize.repeated_sources);
                    segment_profile.materialize_chain_top_nodes = std::move(materialize.top_chains);
                }
                segment_profile.collect_future_inputs_us = t_collect_future_end - t_collect_future_begin;
                for (const auto& comm_op : segment.comm_ops) {
                    if (comm_op.enabled) {
                        segment_profile.comm_bytes += graph_cut_comm_tensor_bytes(gf, comm_op);
                    }
                }
            }

            const int64_t t_reset_runtime_begin = collect_profile ? ggml_time_us() : 0;
            reset_segment_runtime_tensors(segment, gf);
            const int64_t t_reset_runtime_end = collect_profile ? ggml_time_us() : 0;
            if (collect_profile) {
                segment_profile.reset_runtime_tensors_us = t_reset_runtime_end - t_reset_runtime_begin;
            }

            const int64_t t_bind_cached_begin = collect_profile ? ggml_time_us() : 0;
            if (!bind_segment_cached_inputs(gf, segment)) {
                free_cache_ctx_and_buffer();
                free_compute_buffer();
                free_compute_ctx();
                return std::nullopt;
            }
            const int64_t t_bind_cached_end = collect_profile ? ggml_time_us() : 0;
            if (collect_profile) {
                segment_profile.bind_cached_inputs_us = t_bind_cached_end - t_bind_cached_begin;
            }

            const bool is_last_segment = seg_idx + 1 == plan.segments.size();
            const int64_t t_mark_cache_outputs_begin = collect_profile ? ggml_time_us() : 0;
            if (!is_last_segment) {
                for (size_t output_idx = 0; output_idx < segment.output_node_indices.size(); ++output_idx) {
                    ggml_tensor* output_tensor = sd::ggml_graph_cut::output_tensor(gf, segment, output_idx);
                    if (output_tensor != nullptr &&
                        sd::ggml_graph_cut::is_graph_cut_tensor(output_tensor) &&
                        future_cut_names.find(output_tensor->name) != future_cut_names.end()) {
                        cache(output_tensor->name, output_tensor);
                        if (collect_profile) {
                            segment_profile.cached_output_bytes +=
                                sd::ggml_graph_cut::cache_tensor_bytes(output_tensor);
                        }
                    }
                }
            }
            const int64_t t_mark_cache_outputs_end = collect_profile ? ggml_time_us() : 0;
            if (collect_profile) {
                segment_profile.mark_cache_outputs_us = t_mark_cache_outputs_end - t_mark_cache_outputs_begin;
            }
            
            post_compute_cb_t segment_post_compute_cb = nullptr;
            if (process_group_ != nullptr &&
                process_group_->enabled() &&
                !segment.comm_ops.empty()) {
                segment_post_compute_cb = [this, gf, &segment](ggml_cgraph*) -> bool {
                    std::string comm_error;
                    if (!sd::ggml_graph_cut::execute_segment_comm_ops(*process_group_,
                                                                    gf,
                                                                    segment,
                                                                    &comm_error)) {
                        LOG_ERROR("%s graph cut segment communication failed: segment=%s error=%s",
                                get_desc().c_str(),
                                segment.group_name.c_str(),
                                comm_error.c_str());
                        return false;
                    }
                    return true;
                };
            }
            pre_compute_cb_t segment_pre_compute_cb = nullptr;
            bool has_previous_cut_input = false;
            for (const auto& input_ref : segment.input_refs) {
                if (input_ref.type == GraphCutSegment::INPUT_PREVIOUS_CUT) {
                    has_previous_cut_input = true;
                    break;
                }
            }
            if (has_previous_cut_input) {
                segment_pre_compute_cb = [this, gf, &segment](ggml_cgraph*) -> bool {
                    return bind_segment_cached_inputs(gf, segment);
                };
            }

            ggml_context* segment_graph_ctx = nullptr;
            int64_t t_build_begin = ggml_time_ms();
            ggml_cgraph* segment_graph      = sd::ggml_graph_cut::build_segment_graph(gf, segment, &segment_graph_ctx);
            int64_t t_build_end = ggml_time_ms();
            if (collect_profile) {
                segment_profile.build_ms = t_build_end - t_build_begin;
            }
            int64_t t_runtime_param_begin = ggml_time_ms();
            auto runtime_param_tensors = sd::ggml_graph_cut::runtime_param_tensors(gf, segment, get_desc().c_str());
            int64_t t_runtime_param_end = ggml_time_ms();
            if (collect_profile) {
                segment_profile.runtime_param_ms = t_runtime_param_end - t_runtime_param_begin;
            }
            GraphExecuteProfile execute_profile;
#if defined(ED_ENABLE_ASYNC_OFFLOAD)
            bool async_this_segment = false;
            if (async_prefetch_on) {
                AsyncOffloadSlot& cur = async_slots_[seg_idx & 1];
                // (a) ensure this segment's weights are resident (wait event + swap)
                async_apply_slot(cur);
                // (b) launch next segment's prefetch NOW (overlaps this segment's compute)
                if (seg_idx + 1 < plan.segments.size()) {
                    async_prefetch_slot(async_slots_[(seg_idx + 1) & 1],
                                        async_seg_params[seg_idx + 1], seg_idx + 1);
                }
                async_this_segment = true;
            }
#endif
            auto segment_output             = execute_graph<T>(segment_graph,
                                       n_threads,
                                       true,
                                       runtime_param_tensors,
                                       true,
                                       !is_last_segment || no_return,
                                       &future_cut_names,
                                       segment_post_compute_cb,
                                       segment_pre_compute_cb,
                                       collect_profile ? &execute_profile : nullptr,
#if defined(ED_ENABLE_ASYNC_OFFLOAD)
                                       /*params_already_resident=*/async_this_segment);
#else
                                       /*params_already_resident=*/false);
#endif
#if defined(ED_ENABLE_ASYNC_OFFLOAD)
            if (async_prefetch_on && seg_idx >= 1) {
                // (d) deferred restore of the previous segment's slot (compute of this
                // segment is already enqueued; safe to swap-back + free seg_idx-1)
                async_restore_slot(async_slots_[(seg_idx - 1) & 1]);
            }
#endif
            const int64_t t_segment_graph_free_begin = collect_profile ? ggml_time_us() : 0;
            ggml_free(segment_graph_ctx);
            const int64_t t_segment_graph_free_end = collect_profile ? ggml_time_us() : 0;
            if (!segment_output.has_value()) {
                free_cache_ctx_and_buffer();
                free_compute_buffer();
                free_compute_ctx();
                return std::nullopt;
            }
            output = std::move(segment_output);
            if (collect_profile) {
                segment_profile.offload_ms = execute_profile.offload_ms;
                segment_profile.alloc_ms = execute_profile.alloc_ms;
                segment_profile.copy_ms = execute_profile.copy_ms;
                segment_profile.pre_compute_callback_us = execute_profile.pre_compute_callback_us;
                segment_profile.copy_detail = execute_profile.copy_detail;
                segment_profile.segment_graph_free_us = t_segment_graph_free_end - t_segment_graph_free_begin;
                segment_profile.compute_ms = execute_profile.compute_ms;
                segment_profile.comm_ms = execute_profile.post_ms;
                segment_profile.cache_ms = execute_profile.cache_ms;
                segment_profile.cache_live_bytes = execute_profile.cache_live_bytes;
                segment_profile.cache_buffer_bytes = execute_profile.cache_buffer_bytes;
                segment_profile.cache_chunks = execute_profile.cache_chunks;
                segment_profile.cache_pool_bytes = execute_profile.cache_pool_bytes;
                segment_profile.cache_pool_chunks = execute_profile.cache_pool_chunks;
                segment_profile.total_ms = ggml_time_ms() - t_segment_begin;
                segment_profiles.push_back(std::move(segment_profile));
            }
        }

#if defined(ED_ENABLE_ASYNC_OFFLOAD)
        if (async_prefetch_on) {
            // Restore the last segment's slot (its compute is done) + drain both slots.
            const size_t last = plan.segments.size() - 1;
            async_restore_slot(async_slots_[last & 1]);
            async_restore_slot(async_slots_[(last + 1) & 1]);  // no-op if inactive
            // Free the persistent per-slot buffers + events now that the run is done.
            for (auto& s : async_slots_) {
                if (s.buffer != nullptr) { ggml_backend_buffer_free(s.buffer); s.buffer = nullptr; }
                s.buffer_capacity = 0;
                if (s.copy_done_event != nullptr) { ed_async_offload_event_destroy(s.copy_done_event); s.copy_done_event = nullptr; }
            }
            async_cpu_src_.clear();
            if (async_offload_debug()) {
                LOG_INFO("%s [async] prefetch loop done, slots drained", get_desc().c_str());
            }
        }
#endif

        if (collect_profile) {
            log_graph_cut_profile(segment_profiles, plan_ms, plan_ms + ggml_time_ms() - t_profile_begin);
        }
        backend_tensor_data_map.clear();
        reset_graph_cut_run_cache();
        free_compute_buffer();
        free_compute_ctx();
        return output;
    }

public:
    virtual std::string get_desc() = 0;

    bool stage_params_for_phase() {
        if (params_backend == runtime_backend) {
            return true;
        }
        phase_params_pinned_ = true;
        if (offload_all_params()) {
            return true;
        }
        phase_params_pinned_ = false;
        return false;
    }

    void release_params_after_phase() {
        if (!phase_params_pinned_) {
            return;
        }
        phase_params_pinned_ = false;
        invalidate_persistent_graph();
        restore_all_params();
    }

    // On Apple Silicon the Metal GPU and the CPU share the same physical RAM
    // (unified memory). "Offloading" weights between a CPU backend and the Metal
    // backend therefore copies bytes that already live in the very same physical
    // memory: it wastes time AND inflates peak memory (a second live copy exists
    // while offload_all_params() runs). Detecting this lets us keep params on the
    // runtime backend and turn offload into a no-op.
    //
    // Detection is deliberately backend-neutral (only ggml-backend.h APIs) so this
    // compiles/links in every build flavor (CPU / CUDA / Vulkan builds do NOT link
    // the Metal backend, so we must not reference ggml_backend_is_metal here). We
    // identify the Metal backend by its registry name ("MTL"/"Metal") and confirm
    // Apple Silicon via the device description ("Apple ..."). Intel Macs with a
    // discrete AMD GPU also expose Metal but are NOT unified memory; their device
    // description does not contain "Apple", so they keep the real offload path.
    static bool runtime_backend_is_uma(ggml_backend_t backend) {
        if (backend == nullptr || ggml_backend_is_cpu(backend)) {
            return false;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) {
            return false;
        }
        // Only the Metal backend qualifies. Discrete-VRAM backends (CUDA/Vulkan)
        // must keep real offload, so they must never be treated as UMA.
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char* reg_name   = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
        const bool is_metal    = reg_name != nullptr &&
                              (std::strstr(reg_name, "Metal") != nullptr ||
                               std::strstr(reg_name, "MTL") != nullptr);
        if (!is_metal) {
            return false;
        }
        // Confirm Apple Silicon (unified memory). If the description is unavailable,
        // fall back to treating Metal as UMA since Apple Silicon is the only Metal
        // target this project ships for.
        const char* desc = ggml_backend_dev_description(dev);
        if (desc == nullptr) {
            return true;
        }
        return std::strstr(desc, "Apple") != nullptr;
    }

    GGMLRunner(ggml_backend_t backend, bool offload_params_to_cpu = false)
        : runtime_backend(backend) {
        if (!ggml_backend_is_cpu(runtime_backend) && offload_params_to_cpu) {
            if (runtime_backend_is_uma(runtime_backend) &&
                getenv("ED_NO_UMA_SHORTCIRCUIT") == nullptr) {
                // Unified memory: keep params on the runtime (Metal) backend so
                // params_backend == runtime_backend and offload_all_params() /
                // offload_partial_params() short-circuit to a no-op. No CPU<->GPU
                // copy, no duplicated weight buffer, lower peak memory.
                params_backend = runtime_backend;
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    LOG_WARN("detected unified memory (Apple Silicon / Metal): weight offload "
                             "(--offload-to-cpu) is a no-op on UMA and only raises peak memory, "
                             "so it is ignored. Set ED_NO_UMA_SHORTCIRCUIT=1 to force the legacy "
                             "CPU<->GPU copy path.");
                }
            } else {
                params_backend = ggml_backend_cpu_init();
            }
        } else {
            params_backend = runtime_backend;
        }
        alloc_params_ctx();
    }

    virtual ~GGMLRunner() {
        release_params_after_phase();
        free_runtime_const_cache();
        free_params_buffer();
        free_compute_buffer();
        free_params_ctx();
        free_compute_ctx();
        if (params_backend != runtime_backend) {
            ggml_backend_free(params_backend);
        }
        free_cache_ctx_and_buffer();
    }

    virtual GGMLRunnerContext get_context() {
        GGMLRunnerContext runner_ctx;
        runner_ctx.ggml_ctx              = compute_ctx;
        runner_ctx.backend               = runtime_backend;
        runner_ctx.process_group         = process_group_.get();
        runner_ctx.flash_attn_enabled    = flash_attn_enabled;
        runner_ctx.conv2d_direct_enabled      = conv2d_direct_enabled;
        runner_ctx.conv3d_auto_direct_enabled = false;
        runner_ctx.circular_x_enabled    = circular_x_enabled;
        runner_ctx.circular_y_enabled    = circular_y_enabled;
        runner_ctx.weight_adapter        = weight_adapter;
        runner_ctx.tap_registry          = tap_registry_;
        runner_ctx.max_graph_vram_bytes  = max_graph_vram_bytes;
        runner_ctx.bind_backend_tensor_data = [this](ggml_tensor* tensor, const void* data) {
            set_backend_tensor_data(tensor, data);
        };
        return runner_ctx;
    }

    bool should_use_cuda_auto_conv2d() const {
#if defined(ED_ENABLE_CUDNN_CONV2D)
        if (runtime_backend == nullptr) {
            return false;
        }
        ggml_backend_dev_t device = ggml_backend_get_device(runtime_backend);
        if (device == nullptr) {
            return false;
        }
        const char* name = ggml_backend_dev_name(device);
        return name != nullptr && std::string(name).find("CUDA") != std::string::npos;
#else
        return false;
#endif
    }

    bool should_use_cuda_auto_conv3d() const {
#if defined(ED_ENABLE_CUDNN_CONV3D)
        if (runtime_backend == nullptr) {
            return false;
        }
        ggml_backend_dev_t device = ggml_backend_get_device(runtime_backend);
        if (device == nullptr) {
            return false;
        }
        const char* name = ggml_backend_dev_name(device);
        return name != nullptr && std::string(name).find("CUDA") != std::string::npos;
#else
        return false;
#endif
    }

    void reset_compute_ctx() {
        free_compute_ctx();
        sd::ggml_graph_cut::clear_comm_marks();
        alloc_compute_ctx();
    }

    bool alloc_params_buffer() {
        size_t num_tensors = ggml_tensor_num(params_ctx);
        // When params live on CPU for offload (params_backend != runtime_backend),
        // allocate them from the runtime device's pinned host buffer type so the
        // per-segment H2D copy can go async (cudaMemcpyAsync from pageable memory
        // silently serializes). Falls back to the normal CPU buffer otherwise.
        ggml_backend_buffer_type_t pinned_buft = nullptr;
        if (params_backend != runtime_backend && runtime_backend != nullptr) {
            ggml_backend_dev_t dev = ggml_backend_get_device(runtime_backend);
            if (dev != nullptr) {
                pinned_buft = ggml_backend_dev_host_buffer_type(dev);
            }
        }
        params_buffer = pinned_buft != nullptr
                            ? ggml_backend_alloc_ctx_tensors_from_buft(params_ctx, pinned_buft)
                            : ggml_backend_alloc_ctx_tensors(params_ctx, params_backend);
        if (params_buffer == nullptr && pinned_buft != nullptr) {
            // pinned allocation failed (e.g. too large to lock) -> fall back to pageable
            LOG_WARN("%s pinned params buffer alloc failed, falling back to pageable", get_desc().c_str());
            params_buffer = ggml_backend_alloc_ctx_tensors(params_ctx, params_backend);
        }
        if (params_buffer == nullptr) {
            LOG_ERROR("%s alloc params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }
        rebuild_params_tensor_set();
        ggml_backend_buffer_set_usage(params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        size_t params_buffer_size = ggml_backend_buffer_get_size(params_buffer);
        LOG_DEBUG("%s params backend buffer size = % 6.2f MB(%s) (%i tensors)",
                  get_desc().c_str(),
                  params_buffer_size / (1024.f * 1024.f),
                  ggml_backend_is_cpu(params_backend) ? "RAM" : "VRAM",
                  num_tensors);
        return true;
    }

    void free_params_buffer() {
        if (params_buffer != nullptr) {
            ggml_backend_buffer_free(params_buffer);
            params_buffer = nullptr;
        }
    }

    size_t get_params_buffer_size() {
        if (params_buffer != nullptr) {
            return ggml_backend_buffer_get_size(params_buffer);
        }
        return 0;
    }

    void free_cache_ctx_and_buffer() {
        free_cache_buffer();
        free_cache_ctx();
        free_cache_chunks();
        cache_tensor_map.clear();
        cache_tensor_index_.clear();
    }

    void reset_graph_cut_run_cache() {
        free_cache_buffer();
        free_cache_ctx();
        free_cache_chunks(graph_cut_cache_pool_retained_between_runs());
        cache_tensor_map.clear();
        cache_tensor_index_.clear();
    }

    void free_compute_buffer() {
        if (compute_allocr != nullptr) {
            ggml_gallocr_free(compute_allocr);
            compute_allocr = nullptr;
        }
        // The reuse graph's tensors point into the buffer just freed; invalidate so
        // the next compute_reuse() rebuilds instead of re-executing dangling nodes.
        invalidate_persistent_graph();
        restore_partial_params();
        if (!phase_params_pinned_) {
            restore_all_params();
        }
    }

    // do copy after alloc graph
    void set_backend_tensor_data(ggml_tensor* tensor, const void* data) {
        backend_tensor_data_map[tensor] = data;
    }

    template <typename T>
    ggml_tensor* make_input(const sd::Tensor<T>& tensor) {
        ggml_tensor* input = sd::make_ggml_tensor(compute_ctx, tensor, false);
        if (reuse_capture_mode_) {
            // Stage the bytes into a heap-stable runner-owned buffer and bind the
            // node to it, so the binding survives across steps (only contents
            // change) AND survives persistent_.inputs reallocating as more inputs
            // are captured this build.
            const size_t nbytes = ggml_nbytes(input);
            persistent_.inputs.emplace_back();
            ReuseInput& ri = persistent_.inputs.back();
            ri.node = input;
            ri.staging = std::make_unique<std::vector<uint8_t>>(nbytes);
            std::memcpy(ri.staging->data(), tensor.data(), nbytes);
            set_backend_tensor_data(input, ri.staging->data());
            return input;
        }
        set_backend_tensor_data(input, tensor.data());
        return input;
    }

    template <typename T>
    ggml_tensor* make_optional_input(const sd::Tensor<T>& tensor) {
        if (tensor.empty()) {
            return nullptr;
        }
        return make_input(tensor);
    }

    template <typename T>
    ggml_tensor* make_optional_input(const sd::Tensor<T>* tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        return make_input(*tensor);
    }

    ggml_tensor* to_backend(ggml_tensor* tensor) {
        GGML_ASSERT(compute_ctx != nullptr);
        if (tensor == nullptr) {
            return nullptr;
        }
        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(runtime_backend) && (tensor->buffer == nullptr || ggml_backend_buffer_is_host(tensor->buffer))) {
            // pass input tensors to gpu memory
            auto backend_tensor = ggml_dup_tensor(compute_ctx, tensor);

            set_backend_tensor_data(backend_tensor, tensor->data);
            return backend_tensor;
        } else {
            return tensor;
        }
    }

    void cache(const std::string name, ggml_tensor* tensor) {
        cache_tensor_map[name] = tensor;
    }

    ggml_tensor* get_cache_tensor_by_name(const std::string& name) {
        auto iter = cache_tensor_index_.find(name);
        if (iter == cache_tensor_index_.end()) {
            return nullptr;
        }
        return iter->second;
    }

    // Device-to-device copy of a named cache tensor (resident in the run cache
    // until reset_graph_cut_run_cache) into a caller-owned persistent tensor.
    // Used by the GPU DiCache handoff. Returns false if the source is missing or
    // the byte sizes differ.
    bool copy_named_cache_tensor_to(const std::string& name, ggml_tensor* dst) {
        ggml_tensor* src = get_cache_tensor_by_name(name);
        if (src == nullptr || dst == nullptr) {
            return false;
        }
        if (ggml_nbytes(src) != ggml_nbytes(dst)) {
            return false;
        }
        ggml_backend_tensor_copy(src, dst);
        return true;
    }

    // ---- Substep-path pass. The middle layer configures a TapRegistry (requested anchors,
    // indicators, stop-after-block) before calling; the model forward() taps the
    // anchors and build_graph weaves the indicator scalars. This reads back the
    // requested indicator scalars by name and, when a residual capture is asked for,
    // the ModelOut/ModelIn tapped tensors. Model-agnostic. ----
    struct SubstepPassResult {
        sd::Tensor<float> output;
        std::unordered_map<std::string, float> indicators;  // name -> scalar
        // Host readbacks for the host-path (no device slot): the captured residual
        // (ModelOut - ModelIn) and the probe/before tap tensors. Empty unless the
        // caller asks via read_feature / read_taps.
        sd::Tensor<float> feature;
        sd::Tensor<float> before;
        sd::Tensor<float> probe;
    };
    SubstepPassResult run_substep_pass(get_graph_cb_t get_graph,
                                       int n_threads,
                                       edgedit::cache::TapRegistry* registry,
                                       size_t expected_dim,
                                       const std::vector<std::string>& indicator_names,
                                       const std::function<void()>& post_readback = nullptr,
                                       bool read_feature = false,
                                       bool read_taps = false,
                                       bool no_return = false) {
        SubstepPassResult result;
        set_tap_registry(registry);
        auto out = GGMLRunner::compute<float>(get_graph, n_threads, /*free=*/false, no_return);
        result.output = restore_trailing_singleton_dims(std::move(out), expected_dim);

        for (const auto& name : indicator_names) {
            const std::string node = "cache_ind:" + name;
            ggml_tensor* t = get_cache_tensor_by_name(node);
            float v = std::numeric_limits<float>::quiet_NaN();
            if (t != nullptr) {
                ggml_backend_tensor_get(t, &v, 0, sizeof(float));
            }
            result.indicators[name] = v;
        }

        // Host-path readbacks (no device slot): the captured residual and the
        // probe/before tap tensors, read back to host for the host cache path.
        if (read_feature) {
            ggml_tensor* feat = get_cache_tensor_by_name("ed_cache_feature");
            if (feat != nullptr) {
                result.feature = sd::make_sd_tensor_from_ggml<float>(feat);
            }
        }
        if (read_taps && registry != nullptr) {
            ggml_tensor* before = registry->get(registry->before_anchor());
            ggml_tensor* probe = registry->get(registry->probe_anchor());
            if (before != nullptr) {
                result.before = sd::make_sd_tensor_from_ggml<float>(before);
            }
            if (probe != nullptr) {
                result.probe = sd::make_sd_tensor_from_ggml<float>(probe);
            }
        }

        set_tap_registry(nullptr);
        // Device-to-device handoff (residual capture into a persistent slot, etc.)
        // while the named tap tensors are still resident.
        if (post_readback) {
            post_readback();
        }
        reset_graph_cut_run_cache();
        return result;
    }

    template <typename T>
    std::optional<sd::Tensor<T>> compute(get_graph_cb_t get_graph,
                                         int n_threads,
                                         bool free_compute_buffer_immediately,
                                         bool no_return = false) {
        const bool rprof = runner_profile_enabled();
        int64_t t_build_begin = rprof ? ggml_time_ms() : 0;
        ggml_cgraph* gf = nullptr;
        if (!prepare_compute_graph(get_graph, &gf)) {
            return std::nullopt;
        }
        GGML_ASSERT(gf != nullptr);
        int64_t t_build_end = rprof ? ggml_time_ms() : 0;

        if (can_attempt_graph_cut_segmented_compute()) {
            GraphCutPlan plan;
            int64_t t_plan_begin = ggml_time_ms();
            if (!resolve_graph_cut_plan(gf, &plan)) {
                sd::ggml_graph_cut::clear_comm_marks();
                free_compute_ctx();
                return std::nullopt;
            }
            int64_t t_plan_end = ggml_time_ms();
            sd::ggml_graph_cut::clear_comm_marks();
            if (should_use_graph_cut_segmented_compute(plan)) {
                return compute_with_graph_cuts<T>(gf,
                                                  plan,
                                                  n_threads,
                                                  free_compute_buffer_immediately,
                                                  no_return,
                                                  t_plan_end - t_plan_begin);
            }
        }
        sd::ggml_graph_cut::clear_comm_marks();
        int64_t t_alloc_begin = rprof ? ggml_time_ms() : 0;
        if (!alloc_compute_buffer(gf)) {
            LOG_ERROR("%s alloc compute buffer failed", get_desc().c_str());
            return std::nullopt;
        }
        int64_t t_alloc_end = rprof ? ggml_time_ms() : 0;
        if (rprof) {
            LOG_INFO("%s compute prep build=%lldms alloc=%lldms",
                     get_desc().c_str(),
                     static_cast<long long>(t_build_end - t_build_begin),
                     static_cast<long long>(t_alloc_end - t_alloc_begin));
        }
        GraphExecuteProfile runner_profile;
        const int64_t t_rprof_begin = rprof ? ggml_time_ms() : 0;
        auto rprof_result = execute_graph<T>(gf,
                                n_threads,
                                free_compute_buffer_immediately,
                                {},
                                false,
                                no_return,
                                nullptr,
                                nullptr,
                                nullptr,
                                rprof ? &runner_profile : nullptr);
        if (rprof) {
            const int64_t t_rprof_end = ggml_time_ms();
            const int64_t wall = t_rprof_end - t_rprof_begin;
            const int64_t known = runner_profile.offload_ms + runner_profile.alloc_ms +
                                  runner_profile.copy_ms + runner_profile.compute_ms +
                                  runner_profile.post_ms;
            LOG_INFO("%s runner profile: wall=%lldms offload=%lldms alloc=%lldms copy=%lldms compute=%lldms post=%lldms other=%lldms",
                     get_desc().c_str(),
                     (long long) wall,
                     (long long) runner_profile.offload_ms,
                     (long long) runner_profile.alloc_ms,
                     (long long) runner_profile.copy_ms,
                     (long long) runner_profile.compute_ms,
                     (long long) runner_profile.post_ms,
                     (long long) (wall - known));
        }
        return rprof_result;
    }

    // Experimental build-once / reuse-across-steps path (ED_CACHE_COMPILED_GRAPHS).
    // `ordered_inputs` MUST list the per-step input tensors in the exact order the
    // model's build_graph() calls make_input() — their staged bytes are refreshed
    // in that order on a reuse step. On the first call (or when the input count /
    // any per-slot byte size differs) the graph is rebuilt with capture on; every
    // later matching call skips build entirely and only memcpy's new input bytes.
    // The caller is responsible for only routing here on the plain (non-segmented)
    // path with stable shapes; a mismatch safely falls back to a rebuild.
    template <typename T>
    std::optional<sd::Tensor<T>> compute_reuse(get_graph_cb_t get_graph,
                                               const std::vector<const sd::Tensor<T>*>& ordered_inputs,
                                               int n_threads,
                                               bool no_return = false) {
        // Decide reuse vs rebuild: same input count and same per-slot byte sizes.
        bool can_reuse = persistent_.valid &&
                         !persistent_.reuse_disabled &&
                         compute_ctx != nullptr &&
                         persistent_.inputs.size() == ordered_inputs.size();
        if (can_reuse) {
            for (size_t i = 0; i < ordered_inputs.size(); ++i) {
                const size_t want = ordered_inputs[i] != nullptr
                                        ? ordered_inputs[i]->numel() * sizeof(T)
                                        : 0;
                if (want != persistent_.inputs[i].staging->size()) {
                    can_reuse = false;
                    break;
                }
            }
        }

        const bool rprof = runner_profile_enabled();
        if (!can_reuse) {
            // (Re)build with capture on so make_input records + stages each leaf.
            reset_compute_ctx();
            reuse_capture_mode_ = true;
            const int64_t t_build_begin = rprof ? ggml_time_ms() : 0;
            ggml_cgraph* gf = get_compute_graph(get_graph);
            reuse_capture_mode_ = false;
            sd::ggml_graph_cut::clear_comm_marks();
            if (gf == nullptr) {
                free_compute_ctx();
                return std::nullopt;
            }
            persistent_.valid = true;
            persistent_.gf = gf;
            // Consistency guard: on the build step make_input() captured exactly the
            // leaves the model's build_graph() created, in call order. The caller's
            // ordered_inputs MUST describe that same sequence, so the count and each
            // per-slot byte size have to match what was just captured. A mismatch
            // means the ordered_inputs list is wrong (missing/extra/misordered slot)
            // — fail loudly here rather than silently corrupt a later reuse step.
            bool ordered_ok = persistent_.inputs.size() == ordered_inputs.size();
            for (size_t i = 0; ordered_ok && i < ordered_inputs.size(); ++i) {
                const size_t want = ordered_inputs[i] != nullptr
                                        ? ordered_inputs[i]->numel() * sizeof(T)
                                        : 0;
                if (want != persistent_.inputs[i].staging->size()) {
                    ordered_ok = false;
                }
            }
            if (!ordered_ok) {
                LOG_ERROR("%s compute_reuse: ordered_inputs (%zu) does not match captured "
                          "graph inputs (%zu) — disabling reuse for this run",
                          get_desc().c_str(), ordered_inputs.size(), persistent_.inputs.size());
                // Fall back to a correct one-shot result; leave the graph built but
                // mark it non-reusable so we never memcpy into the wrong slots.
                persistent_.reuse_disabled = true;
            }
            if (rprof) {
                LOG_INFO("%s compiled-graph build (reuse path) build=%lldms inputs=%zu",
                         get_desc().c_str(),
                         static_cast<long long>(ggml_time_ms() - t_build_begin),
                         persistent_.inputs.size());
            }
        } else {
            // Reuse: refresh only the staged input bytes; graph & bindings persist.
            for (size_t i = 0; i < ordered_inputs.size(); ++i) {
                if (ordered_inputs[i] == nullptr) {
                    continue;
                }
                ReuseInput& ri = persistent_.inputs[i];
                std::memcpy(ri.staging->data(), ordered_inputs[i]->data(), ri.staging->size());
                // Re-bind in case a prior step's copy cleared the map entry.
                set_backend_tensor_data(ri.node, ri.staging->data());
            }
        }

        if (persistent_.gf == nullptr) {
            LOG_ERROR("%s reuse path has no graph", get_desc().c_str());
            return std::nullopt;
        }
        GraphExecuteProfile runner_profile;
        // execute_graph() handles alloc_compute_buffer + ggml_gallocr_alloc_graph.
        // preserve_backend_tensor_data_map=true keeps the staged/const bindings
        // alive across steps; free=false keeps the graph + allocr resident.
        return execute_graph<T>(persistent_.gf,
                                n_threads,
                                /*free_compute_buffer_immediately=*/false,
                                {},
                                /*preserve_backend_tensor_data_map=*/true,
                                no_return,
                                nullptr,
                                nullptr,
                                nullptr,
                                runner_profile_enabled() ? &runner_profile : nullptr);
    }
    void set_flash_attention_enabled(bool enabled) {
        flash_attn_enabled = enabled;
    }

    // Attach/detach the substep tap registry consulted by model forward().
    void set_tap_registry(edgedit::cache::TapRegistry* reg) {
        tap_registry_ = reg;
    }
    edgedit::cache::TapRegistry* tap_registry() const {
        return tap_registry_;
    }
    // True when the block-stack feature seam can run: only on the plain compute
    // path (no process-group comm, no VRAM-budgeted segmented execution), since
    // mid-graph capture is not preserved across segments.
    bool feature_cache_available() const {
        return !can_attempt_graph_cut_segmented_compute();
    }

    void set_conv2d_direct_enabled(bool enabled) {
        conv2d_direct_enabled = enabled;
    }

    void set_circular_axes(bool circular_x, bool circular_y) {
        circular_x_enabled = circular_x;
        circular_y_enabled = circular_y;
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {
        weight_adapter = adapter;
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) {
        max_graph_vram_bytes = max_vram_bytes;
    }
    void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> process_group) {
        process_group_ = std::move(process_group);
    }

    edgedit::parallel::ProcessGroup* get_process_group() {
        return process_group_.get();
    }
    ggml_backend_t get_runtime_backend() {
        return runtime_backend;
    }

    ggml_backend_t get_params_backend() {
        return params_backend;
    }
};

class GGMLBlock {
protected:
    typedef std::unordered_map<std::string, ggml_tensor*> ParameterMap;
    typedef std::unordered_map<std::string, std::shared_ptr<GGMLBlock>> GGMLBlockMap;
    GGMLBlockMap blocks;
    ParameterMap params;

    ggml_type get_type(const std::string& name, const String2TensorStorage& tensor_storage_map, ggml_type default_type) {
        ggml_type wtype = default_type;
        auto iter       = tensor_storage_map.find(name);
        if (iter != tensor_storage_map.end()) {
            const TensorStorage& tensor_storage = iter->second;
            if (tensor_storage.expected_type != GGML_TYPE_COUNT) {
                wtype = tensor_storage.expected_type;
            } else {
                wtype = tensor_storage.type;
            }
        }
        return wtype;
    }

    void init_blocks(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->init(ctx, tensor_storage_map, prefix + pair.first);
        }
    }

    virtual void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {}

public:
    void init(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        init_params(ctx, tensor_storage_map, prefix);
        init_blocks(ctx, tensor_storage_map, prefix);
    }

    size_t get_params_num() {
        size_t num_tensors = params.size();
        for (auto& pair : blocks) {
            auto& block = pair.second;

            num_tensors += block->get_params_num();
        }
        return num_tensors;
    };

    size_t get_params_mem_size() {
        size_t mem_size = 0;
        for (auto& pair : blocks) {
            auto& block = pair.second;

            mem_size += block->get_params_mem_size();
        }

        for (auto& pair : params) {
            mem_size += ggml_nbytes(pair.second);
        }

        return mem_size;
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->get_param_tensors(tensors, prefix + pair.first);
        }

        for (auto& pair : params) {
            ggml_tensor* param           = pair.second;
            tensors[prefix + pair.first] = pair.second;
        }
    }

    virtual std::string get_desc() {
        return "GGMLBlock";
    }

    void get_all_blocks(std::vector<GGMLBlock*>& result) {
        result.push_back(this);
        for (auto& block_iter : blocks) {
            if (block_iter.second) {
                block_iter.second->get_all_blocks(result);
            }
        }
    }
};

class UnaryBlock : public GGMLBlock {
public:
    virtual ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) = 0;
};

class Identity : public UnaryBlock {
public:
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        return x;
    }
};

class Linear : public UnaryBlock {
protected:
    int64_t in_features;
    int64_t out_features;
    bool bias;
    bool force_f32;
    bool force_prec_f32;
    float scale;
    bool scale_quantized_only;
    bool use_model_bias_type;
    bool cast_output_to_input_type;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (in_features % ggml_blck_size(wtype) != 0 || force_f32) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (bias) {
            enum ggml_type btype = use_model_bias_type ? get_type(prefix + "bias", tensor_storage_map, GGML_TYPE_F32)
                                                        : GGML_TYPE_F32;
            if (btype != GGML_TYPE_F32 && btype != GGML_TYPE_F16 && btype != GGML_TYPE_BF16) {
                btype = GGML_TYPE_F32;
            }
            params["bias"]       = ggml_new_tensor_1d(ctx, btype, out_features);
        }
    }

public:
    Linear(int64_t in_features,
           int64_t out_features,
           bool bias           = true,
           bool force_f32      = false,
           bool force_prec_f32 = false,
           float scale         = 1.f,
           bool scale_quantized_only = false,
           bool use_model_bias_type = false,
           bool cast_output_to_input_type = false)
        : in_features(in_features),
          out_features(out_features),
          bias(bias),
          force_f32(force_f32),
          force_prec_f32(force_prec_f32),
          scale(scale),
          scale_quantized_only(scale_quantized_only),
          use_model_bias_type(use_model_bias_type),
          cast_output_to_input_type(cast_output_to_input_type) {}

    void set_scale(float scale_) {
        scale = scale_;
    }

    void set_force_prec_f32(bool force_prec_f32_) {
        force_prec_f32 = force_prec_f32_;
    }

    bool weight_is_quantized() const {
        auto it = params.find("weight");
        return it != params.end() && ggml_is_quantized(it->second->type);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        const ggml_type output_type = x->type;
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }
        const float effective_scale = scale_quantized_only && !ggml_is_quantized(w->type) ? 1.f : scale;
        if (ctx->weight_adapter) {
            WeightAdapter::ForwardParams forward_params;
            forward_params.op_type               = WeightAdapter::ForwardParams::op_type_t::OP_LINEAR;
            forward_params.linear.force_prec_f32 = force_prec_f32;
            forward_params.linear.scale          = effective_scale;
            return ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
        }
        const std::string matmul_name = prefix + "weight";
        x = ggml_ext_linear(ctx->ggml_ctx, x, w, b, force_prec_f32, effective_scale, matmul_name.c_str());
        if (cast_output_to_input_type &&
            (output_type == GGML_TYPE_F16 || output_type == GGML_TYPE_BF16) &&
            x->type != output_type) {
            x = ggml_cast(ctx->ggml_ctx, x, output_type);
        }
        return x;
    }

    ggml_tensor* forward_output_slice(GGMLRunnerContext* ctx,
                                      ggml_tensor* x,
                                      int64_t output_offset,
                                      int64_t output_count) {
        if (ctx == nullptr ||
            x == nullptr ||
            ctx->weight_adapter != nullptr ||
            output_offset < 0 ||
            output_count <= 0 ||
            output_offset + output_count > out_features) {
            return nullptr;
        }

        ggml_tensor* w = params["weight"];
        ggml_tensor* w_slice = ggml_view_2d(ctx->ggml_ctx,
                                            w,
                                            w->ne[0],
                                            output_count,
                                            w->nb[1],
                                            static_cast<size_t>(output_offset) * w->nb[1]);

        ggml_tensor* b_slice = nullptr;
        if (bias) {
            ggml_tensor* b = params["bias"];
            b_slice = ggml_view_1d(ctx->ggml_ctx,
                                   b,
                                   output_count,
                                   static_cast<size_t>(output_offset) * b->nb[0]);
        }

        return ggml_ext_linear(ctx->ggml_ctx, x, w_slice, b_slice, force_prec_f32, scale);
    }

    ggml_tensor* forward_input_concat_split(GGMLRunnerContext* ctx,
                                            ggml_tensor* x0,
                                            ggml_tensor* x1) {
        if (ctx == nullptr ||
            x0 == nullptr ||
            x1 == nullptr ||
            ctx->weight_adapter != nullptr ||
            x0->ne[0] <= 0 ||
            x1->ne[0] <= 0 ||
            x0->ne[0] + x1->ne[0] != in_features ||
            x0->ne[1] != x1->ne[1] ||
            x0->ne[2] != x1->ne[2] ||
            x0->ne[3] != x1->ne[3]) {
            return nullptr;
        }

        ggml_tensor* w = params["weight"];
        const int64_t block_size = ggml_blck_size(w->type);
        if (block_size <= 0 ||
            x0->ne[0] % block_size != 0 ||
            x1->ne[0] % block_size != 0) {
            return nullptr;
        }

        ggml_tensor* w0 = ggml_view_2d(ctx->ggml_ctx,
                                       w,
                                       x0->ne[0],
                                       out_features,
                                       w->nb[1],
                                       0);
        ggml_tensor* w1 = ggml_view_2d(ctx->ggml_ctx,
                                       w,
                                       x1->ne[0],
                                       out_features,
                                       w->nb[1],
                                       static_cast<size_t>(x0->ne[0] / block_size) * w->nb[0]);

        ggml_tensor* y0 = ggml_ext_linear(ctx->ggml_ctx, x0, w0, nullptr, force_prec_f32, scale);
        ggml_tensor* y1 = ggml_ext_linear(ctx->ggml_ctx, x1, w1, nullptr, force_prec_f32, scale);
        ggml_tensor* y  = ggml_add(ctx->ggml_ctx, y0, y1);
        if (bias) {
            y = ggml_add_inplace(ctx->ggml_ctx, y, params["bias"]);
        }
        return y;
    }

    ggml_tensor* forward_input_concat_fused(GGMLRunnerContext* ctx,
                                            ggml_tensor* x0,
                                            ggml_tensor* x1) {
        if (ctx == nullptr ||
            x0 == nullptr ||
            x1 == nullptr ||
            ctx->weight_adapter != nullptr ||
            force_prec_f32 ||
            scale != 1.f ||
            x0->ne[0] <= 0 ||
            x1->ne[0] <= 0 ||
            x0->ne[0] + x1->ne[0] != in_features ||
            x0->ne[1] != x1->ne[1] ||
            x0->ne[2] != x1->ne[2] ||
            x0->ne[3] != x1->ne[3]) {
            return nullptr;
        }

        ggml_tensor* w = params["weight"];
        ggml_tensor* b = bias ? params["bias"] : nullptr;
        return edgedit::ggml_ext::flux_sp_concat_linear_custom(ctx->ggml_ctx, x0, x1, w, b);
    }

    ggml_tensor* forward_input_concat_residual_gate_fused(GGMLRunnerContext* ctx,
                                                          ggml_tensor* residual,
                                                          ggml_tensor* x0,
                                                          ggml_tensor* x1,
                                                          ggml_tensor* gate) {
        if (ctx == nullptr ||
            residual == nullptr ||
            x0 == nullptr ||
            x1 == nullptr ||
            gate == nullptr ||
            ctx->weight_adapter != nullptr ||
            force_prec_f32 ||
            scale != 1.f ||
            x0->ne[0] <= 0 ||
            x1->ne[0] <= 0 ||
            x0->ne[0] + x1->ne[0] != in_features ||
            residual->ne[0] != out_features ||
            x0->ne[1] != x1->ne[1] ||
            x0->ne[2] != x1->ne[2] ||
            x0->ne[3] != x1->ne[3] ||
            residual->ne[1] != x0->ne[1] ||
            residual->ne[2] != x0->ne[2] ||
            residual->ne[3] != x0->ne[3]) {
            return nullptr;
        }

        ggml_tensor* w = params["weight"];
        ggml_tensor* b = bias ? params["bias"] : nullptr;
        return edgedit::ggml_ext::flux_sp_concat_linear_residual_gate_custom(ctx->ggml_ctx, residual, x0, x1, w, b, gate);
    }

};

__STATIC_INLINE__ bool support_get_rows(ggml_type wtype) {
    std::set<ggml_type> allow_types = {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0};
    if (allow_types.find(wtype) != allow_types.end()) {
        return true;
    }
    return false;
}

class Embedding : public UnaryBlock {
protected:
    int64_t embedding_dim;
    int64_t num_embeddings;
    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (!support_get_rows(wtype)) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, embedding_dim, num_embeddings);
    }

public:
    Embedding(int64_t num_embeddings, int64_t embedding_dim)
        : embedding_dim(embedding_dim),
          num_embeddings(num_embeddings) {
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* input_ids) {
        // input_ids: [N, n_token]
        auto weight = params["weight"];

        // There are issues with ggml batch inference, so we are expanding it here first.
        // TODO: fix ggml batch inference
        int64_t n = input_ids->ne[1];
        input_ids = ggml_reshape_1d(ctx->ggml_ctx, input_ids, input_ids->ne[0] * input_ids->ne[1]);

        input_ids      = ggml_reshape_3d(ctx->ggml_ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
        auto embedding = ggml_get_rows(ctx->ggml_ctx, weight, input_ids);
        embedding      = ggml_reshape_3d(ctx->ggml_ctx, embedding, embedding->ne[0], embedding->ne[1] / n, n);

        // [N, n_token, embedding_dim]
        return embedding;
    }
};

class Conv2d : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride;
    std::pair<int, int> padding;
    std::pair<int, int> dilation;
    bool bias;
    float scale = 1.f;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_channels);
        }
    }

public:
    Conv2d(int64_t in_channels,
           int64_t out_channels,
           std::pair<int, int> kernel_size,
           std::pair<int, int> stride   = {1, 1},
           std::pair<int, int> padding  = {0, 0},
           std::pair<int, int> dilation = {1, 1},
           bool bias                    = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    void set_scale(float scale_value) {
        scale = scale_value;
    }

    std::string get_desc() {
        return "Conv2d";
    }

    bool should_use_auto_direct(ggml_tensor* x, ggml_tensor* w) const {
#if defined(ED_ENABLE_CUDNN_CONV2D)
        if (x == nullptr || w == nullptr) {
            return false;
        }
        if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F16) {
            return false;
        }
        const bool is_1x1 = kernel_size.first == 1 && kernel_size.second == 1;
        const bool is_3x3 = kernel_size.first == 3 && kernel_size.second == 3;
        if (!is_1x1 && !is_3x3) {
            return false;
        }
        if (dilation.first != 1 || dilation.second != 1) {
            return false;
        }
        if (stride.first != stride.second || (stride.first != 1 && stride.first != 2)) {
            return false;
        }
        if (in_channels < 64 || out_channels < 64 || x->ne[0] < 32 || x->ne[1] < 32) {
            return false;
        }
        if (is_1x1 && (stride.first != 1 || x->ne[0] * x->ne[1] < 4096)) {
            return false;
        }
        return true;
#else
        GGML_UNUSED(x);
        GGML_UNUSED(w);
        return false;
#endif
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }
        const bool use_direct = ctx->conv2d_direct_enabled ||
                                (ctx->conv2d_auto_direct_enabled && should_use_auto_direct(x, w));
        if (ctx->weight_adapter) {
            WeightAdapter::ForwardParams forward_params;
            forward_params.op_type           = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
            forward_params.conv2d.s0         = stride.second;
            forward_params.conv2d.s1         = stride.first;
            forward_params.conv2d.p0         = padding.second;
            forward_params.conv2d.p1         = padding.first;
            forward_params.conv2d.d0         = dilation.second;
            forward_params.conv2d.d1         = dilation.first;
            forward_params.conv2d.direct     = use_direct;
            forward_params.conv2d.circular_x = ctx->circular_x_enabled;
            forward_params.conv2d.circular_y = ctx->circular_y_enabled;
            forward_params.conv2d.scale      = scale;
            return ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
        }
        return ggml_ext_conv_2d(ctx->ggml_ctx,
                                x,
                                w,
                                b,
                                stride.second,
                                stride.first,
                                padding.second,
                                padding.first,
                                dilation.second,
                                dilation.first,
                                use_direct,
                                ctx->circular_x_enabled,
                                ctx->circular_y_enabled,
                                scale,
                                ctx->backend);
    }
};

class Conv3d : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    std::tuple<int, int, int> kernel_size;
    std::tuple<int, int, int> stride;
    std::tuple<int, int, int> padding;
    std::tuple<int, int, int> dilation;
    bool bias;
    bool use_model_weight_type;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = use_model_weight_type ? get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F16)
                                                     : GGML_TYPE_F16;
        if (wtype != GGML_TYPE_F32 && wtype != GGML_TYPE_F16 && wtype != GGML_TYPE_BF16) {
            wtype = GGML_TYPE_F16;
        }
        params["weight"]     = ggml_new_tensor_4d(ctx,
                                                  wtype,
                                                  std::get<2>(kernel_size),
                                                  std::get<1>(kernel_size),
                                                  std::get<0>(kernel_size),
                                                  in_channels * out_channels);
        if (bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

public:
    Conv3d(int64_t in_channels,
           int64_t out_channels,
           std::tuple<int, int, int> kernel_size,
           std::tuple<int, int, int> stride   = {1, 1, 1},
           std::tuple<int, int, int> padding  = {0, 0, 0},
           std::tuple<int, int, int> dilation = {1, 1, 1},
           bool bias                          = true,
           bool use_model_weight_type         = false)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias),
          use_model_weight_type(use_model_weight_type) {}

    ggml_tensor* weight_for_forward(GGMLRunnerContext* ctx, bool preserve_type = false) {
        ggml_tensor* w = params["weight"];
        if (ctx->weight_adapter) {
            w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
            if (!preserve_type && w->type != GGML_TYPE_F16) {
                w = ggml_cast(ctx->ggml_ctx, w, GGML_TYPE_F16);
            }
        }
        return w;
    }

    ggml_tensor* bias_for_forward(GGMLRunnerContext* ctx) {
        if (!bias) {
            return nullptr;
        }
        ggml_tensor* b = params["bias"];
        if (ctx->weight_adapter) {
            b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
        }
        return b;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = weight_for_forward(ctx, false);
        ggml_tensor* b = bias_for_forward(ctx);
        const bool direct_compatible = ctx->conv3d_auto_direct_enabled &&
                                       x != nullptr &&
                                       w != nullptr &&
                                       x->type == GGML_TYPE_F32 &&
                                       w->type == GGML_TYPE_F16 &&
                                       std::get<0>(kernel_size) == 3 &&
                                       std::get<1>(kernel_size) == 3 &&
                                       std::get<2>(kernel_size) == 3 &&
                                       ggml_is_contiguous(x) &&
                                       ggml_is_contiguous(w);
        const bool use_direct = direct_compatible &&
                                (ctx->conv3d_force_direct_enabled ||
                                 (in_channels >= 64 && x->ne[0] >= 128 && x->ne[1] >= 128));
        return ggml_ext_conv_3d(ctx->ggml_ctx, x, w, b, in_channels,
                                std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                std::get<2>(padding), std::get<1>(padding), std::get<0>(padding),
                                std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation),
                                use_direct, ctx->backend);
    }
};

class LayerNorm : public UnaryBlock {
protected:
    int64_t normalized_shape;
    float eps;
    bool elementwise_affine;
    bool bias;
    bool cast_output_to_input_type;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix = prefix;
        if (elementwise_affine) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["weight"]     = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            if (bias) {
                enum ggml_type wtype = GGML_TYPE_F32;
                params["bias"]       = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            }
        }
    }

public:
    LayerNorm(int64_t normalized_shape,
              float eps               = 1e-05f,
              bool elementwise_affine = true,
              bool bias               = true,
              bool cast_output_to_input_type = false)
        : normalized_shape(normalized_shape),
          eps(eps),
          elementwise_affine(elementwise_affine),
          bias(bias),
          cast_output_to_input_type(cast_output_to_input_type) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        const ggml_type output_type = x->type;
        if (cast_output_to_input_type &&
            (output_type == GGML_TYPE_F16 || output_type == GGML_TYPE_BF16)) {
            x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
        }
        ggml_tensor* w = nullptr;
        ggml_tensor* b = nullptr;

        if (elementwise_affine) {
            w = params["weight"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
            }
            if (bias) {
                b = params["bias"];
                if (ctx->weight_adapter) {
                    b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
                }
            }
        }
        x = ggml_ext_layer_norm(ctx->ggml_ctx, x, w, b, eps);
        if (cast_output_to_input_type &&
            (output_type == GGML_TYPE_F16 || output_type == GGML_TYPE_BF16) &&
            x->type != output_type) {
            x = ggml_cast(ctx->ggml_ctx, x, output_type);
        }
        return x;
    }
};

class GroupNorm : public GGMLBlock {
protected:
    int num_groups;
    int64_t num_channels;
    float eps;
    bool affine;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix = prefix;
        if (affine) {
            enum ggml_type wtype      = GGML_TYPE_F32;
            enum ggml_type bias_wtype = GGML_TYPE_F32;
            params["weight"]          = ggml_new_tensor_1d(ctx, wtype, num_channels);
            params["bias"]            = ggml_new_tensor_1d(ctx, bias_wtype, num_channels);
        }
    }

public:
    GroupNorm(int num_groups,
              int64_t num_channels,
              float eps   = 1e-05f,
              bool affine = true)
        : num_groups(num_groups),
          num_channels(num_channels),
          eps(eps),
          affine(affine) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = nullptr;
        ggml_tensor* b = nullptr;
        if (affine) {
            w = params["weight"];
            b = params["bias"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
                b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
            }
        }
        return ggml_ext_group_norm(ctx->ggml_ctx, x, w, b, num_groups);
    }
};

class GroupNorm32 : public GroupNorm {
public:
    GroupNorm32(int64_t num_channels)
        : GroupNorm(32, num_channels, 1e-06f) {}
};

class RMSNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;
    std::string prefix;
    bool use_model_weight_type;
    bool cast_output_to_input_type;
    ggml_type model_weight_type = GGML_TYPE_COUNT;

    static bool qwen_vl_rms_norm_mul_bf16_enabled() {
        const char* env = std::getenv("ED_QWEN_VL_RMS_NORM_BF16");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        return std::strcmp(env, "0") != 0 &&
               std::strcmp(env, "false") != 0 &&
               std::strcmp(env, "FALSE") != 0 &&
               std::strcmp(env, "off") != 0 &&
               std::strcmp(env, "OFF") != 0;
    }

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
        this->prefix         = prefix;
        model_weight_type    = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_COUNT);
        enum ggml_type wtype = use_model_weight_type ? model_weight_type : GGML_TYPE_F32;
        if (wtype != GGML_TYPE_F32 && wtype != GGML_TYPE_F16 && wtype != GGML_TYPE_BF16) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"]     = ggml_new_tensor_1d(ctx, wtype, hidden_size);
    }

public:
    RMSNorm(int64_t hidden_size,
            float eps = 1e-06f,
            bool use_model_weight_type = false,
            bool cast_output_to_input_type = false)
        : hidden_size(hidden_size),
          eps(eps),
          use_model_weight_type(use_model_weight_type),
          cast_output_to_input_type(cast_output_to_input_type) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_type output_type = x->type;
        if (output_type != GGML_TYPE_F16 && output_type != GGML_TYPE_BF16 &&
            (model_weight_type == GGML_TYPE_F16 || model_weight_type == GGML_TYPE_BF16)) {
            output_type = model_weight_type;
        }
        ggml_tensor* w = params["weight"];
        if (ctx->weight_adapter) {
            w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
        }
        if (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) {
            x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
        }
        if (qwen_vl_rms_norm_mul_bf16_enabled() &&
            sd_backend_is(ctx->backend, "CUDA") &&
            cast_output_to_input_type &&
            output_type == GGML_TYPE_BF16 &&
            model_weight_type == GGML_TYPE_BF16) {
            if (auto fused = edgedit::ggml_ext::qwen_vl_rms_norm_mul_bf16_custom(ctx->ggml_ctx, x, w, eps)) {
                return fused;
            }
        }
        x = ggml_rms_norm(ctx->ggml_ctx, x, eps);
        if (cast_output_to_input_type &&
            (output_type == GGML_TYPE_F16 || output_type == GGML_TYPE_BF16)) {
            x = ggml_cast(ctx->ggml_ctx, x, output_type);
        }
        x = ggml_mul_inplace(ctx->ggml_ctx, x, w);
        if (cast_output_to_input_type &&
            (output_type == GGML_TYPE_F16 || output_type == GGML_TYPE_BF16) &&
            x->type != output_type) {
            x = ggml_cast(ctx->ggml_ctx, x, output_type);
        }
        return x;
    }
};

class MultiheadAttention : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t n_head;
    bool proj_in;
    std::string q_proj_name;
    std::string k_proj_name;
    std::string v_proj_name;
    std::string in_proj_name;
    std::string out_proj_name;

public:
    MultiheadAttention(int64_t embed_dim,
                       int64_t n_head,
                       bool qkv_proj_bias        = true,
                       bool out_proj_bias        = true,
                       bool proj_in              = false,
                       std::string q_proj_name   = "q_proj",
                       std::string k_proj_name   = "k_proj",
                       std::string v_proj_name   = "v_proj",
                       std::string in_proj_name  = "in_proj",
                       std::string out_proj_name = "out_proj")
        : embed_dim(embed_dim),
          n_head(n_head),
          proj_in(proj_in),
          q_proj_name(q_proj_name),
          k_proj_name(k_proj_name),
          v_proj_name(v_proj_name),
          in_proj_name(in_proj_name),
          out_proj_name(out_proj_name) {
        if (proj_in) {
            blocks[in_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim * 3, qkv_proj_bias));
        } else {
            blocks[q_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
            blocks[k_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
            blocks[v_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
        }
        blocks[out_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, out_proj_bias));
    }

    // x: [N, n_token, embed_dim]
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* mask = nullptr) {
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks[out_proj_name]);

        ggml_tensor* q;
        ggml_tensor* k;
        ggml_tensor* v;
        if (proj_in) {
            auto in_proj = std::dynamic_pointer_cast<Linear>(blocks[in_proj_name]);
            auto qkv     = in_proj->forward(ctx, x);
            auto qkv_vec = split_qkv(ctx->ggml_ctx, qkv);
            q            = qkv_vec[0];
            k            = qkv_vec[1];
            v            = qkv_vec[2];
        } else {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks[q_proj_name]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks[k_proj_name]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks[v_proj_name]);

            q = q_proj->forward(ctx, x);
            k = k_proj->forward(ctx, x);
            v = v_proj->forward(ctx, x);
        }

        x = ggml_ext_attention_ext(ctx->ggml_ctx,
                                   ctx->backend,
                                   q,
                                   k,
                                   v,
                                   n_head,
                                   mask,
                                   false,
                                   ctx->flash_attn_enabled);  // [N, n_token, embed_dim]

        x = out_proj->forward(ctx, x);  // [N, n_token, embed_dim]
        return x;
    }
};

__STATIC_INLINE__ ggml_tensor* ggml_ext_lokr_forward(
    ggml_context* ctx,
    ggml_backend_t backend,
    ggml_tensor* h,    // Input: [q, batch] or [W, H, q, batch]
    ggml_tensor* w1,   // Outer C (Full rank)
    ggml_tensor* w1a,  // Outer A (Low rank part 1)
    ggml_tensor* w1b,  // Outer B (Low rank part 2)
    ggml_tensor* w2,   // Inner BA (Full rank)
    ggml_tensor* w2a,  // Inner A (Low rank part 1)
    ggml_tensor* w2b,  // Inner B (Low rank part 2)
    bool is_conv,
    WeightAdapter::ForwardParams::conv2d_params_t conv_params,
    float scale) {
    GGML_ASSERT((w1 != nullptr || (w1a != nullptr && w1b != nullptr)));
    GGML_ASSERT((w2 != nullptr || (w2a != nullptr && w2b != nullptr)));

    int uq = (w1 != nullptr) ? (int)w1->ne[0] : (int)w1a->ne[0];
    int up = (w1 != nullptr) ? (int)w1->ne[1] : (int)w1b->ne[1];

    int q_actual = is_conv ? (int)h->ne[2] : (int)h->ne[0];
    int vq       = q_actual / uq;

    int vp = (w2 != nullptr) ? (is_conv ? (int)w2->ne[3] : (int)w2->ne[1])
                             : (int)w2a->ne[1];
    GGML_ASSERT(q_actual == (uq * vq) && "Input dimension mismatch for LoKR split");

    ggml_tensor* hb;

    if (!is_conv) {
        int batch          = (int)h->ne[1];
        int merge_batch_uq = batch;
        int merge_batch_vp = batch;

        if (sd_backend_is(backend, "Vulkan")) {
            if (batch > 1) {
                // no access to backend here, worst case is slightly worse perfs for other backends when built alongside Vulkan backend
                int max_batch    = 65535;
                int max_batch_uq = max_batch / uq;
                merge_batch_uq   = 1;
                for (int i = max_batch_uq; i > 0; i--) {
                    if (batch % i == 0) {
                        merge_batch_uq = i;
                        break;
                    }
                }

                int max_batch_vp = max_batch / vp;
                merge_batch_vp   = 1;
                for (int i = max_batch_vp; i > 0; i--) {
                    if (batch % i == 0) {
                        merge_batch_vp = i;
                        break;
                    }
                }
            }
        }

        ggml_tensor* h_split = ggml_reshape_3d(ctx, h, vq, uq * merge_batch_uq, batch / merge_batch_uq);
        if (w2 != nullptr) {
            hb = ggml_mul_mat(ctx, w2, h_split);
        } else {
            hb = ggml_mul_mat(ctx, w2b, ggml_mul_mat(ctx, w2a, h_split));
        }

        if (batch > 1) {
            hb = ggml_reshape_3d(ctx, hb, vp, uq, batch);
        }
        ggml_tensor* hb_t = ggml_cont(ctx, ggml_transpose(ctx, hb));
        hb_t              = ggml_reshape_3d(ctx, hb_t, uq, vp * merge_batch_vp, batch / merge_batch_vp);

        ggml_tensor* hc_t;
        if (w1 != nullptr) {
            hc_t = ggml_mul_mat(ctx, w1, hb_t);
        } else {
            hc_t = ggml_mul_mat(ctx, w1b, ggml_mul_mat(ctx, w1a, hb_t));
        }

        if (batch > 1) {
            hc_t = ggml_reshape_3d(ctx, hc_t, up, vp, batch);
        }

        ggml_tensor* hc  = ggml_transpose(ctx, hc_t);
        ggml_tensor* out = ggml_reshape_2d(ctx, ggml_cont(ctx, hc), up * vp, batch);
        return ggml_scale(ctx, out, scale);
    } else {
        int batch = (int)h->ne[3];
        // 1. Reshape input: [W, H, vq*uq, batch] -> [W, H, vq, uq * batch]
        ggml_tensor* h_split = ggml_reshape_4d(ctx, h, h->ne[0], h->ne[1], vq, uq * batch);

        if (w2 != nullptr) {
            hb = ggml_ext_conv_2d(ctx, h_split, w2, nullptr,
                                  conv_params.s0,
                                  conv_params.s1,
                                  conv_params.p0,
                                  conv_params.p1,
                                  conv_params.d0,
                                  conv_params.d1,
                                  conv_params.direct,
                                  conv_params.circular_x,
                                  conv_params.circular_y,
                                  conv_params.scale);
        } else {
            // swap a and b order for conv lora
            ggml_tensor* a = w2b;
            ggml_tensor* b = w2a;

            // unpack conv2d weights if needed
            if (ggml_n_dims(a) < 4) {
                int k = (int)sqrt(a->ne[0] / h_split->ne[2]);
                GGML_ASSERT(k * k * h_split->ne[2] == a->ne[0]);
                a = ggml_reshape_4d(ctx, a, k, k, a->ne[0] / (k * k), a->ne[1]);
            } else if (a->ne[2] != h_split->ne[2]) {
                int k = (int)sqrt(a->ne[2] / h_split->ne[2]);
                GGML_ASSERT(k * k * h_split->ne[2] == a->ne[2]);
                a = ggml_reshape_4d(ctx, a, a->ne[0] * k, a->ne[1] * k, a->ne[2] / (k * k), a->ne[3]);
            }
            ggml_tensor* ha = ggml_ext_conv_2d(ctx, h_split, a, nullptr,
                                               conv_params.s0,
                                               conv_params.s1,
                                               conv_params.p0,
                                               conv_params.p1,
                                               conv_params.d0,
                                               conv_params.d1,
                                               conv_params.direct,
                                               conv_params.circular_x,
                                               conv_params.circular_y,
                                               conv_params.scale);

            // not supporting lora_mid here
            hb = ggml_ext_conv_2d(ctx,
                                  ha,
                                  b,
                                  nullptr,
                                  1,
                                  1,
                                  0,
                                  0,
                                  1,
                                  1,
                                  conv_params.direct,
                                  conv_params.circular_x,
                                  conv_params.circular_y,
                                  conv_params.scale);
        }

        // Current hb shape: [W_out, H_out, vp, uq * batch]
        int w_out = (int)hb->ne[0];
        int h_out = (int)hb->ne[1];

        // ggml_tensor* hb_cat = ggml_reshape_4d(ctx, hb, w_out , h_out , vp * uq, batch);
        // [W_out, H_out, vp * uq,  batch]
        // Now left to compute (W1 kr Id) * hb_cat == (W1 kr W2) cv h

        // merge the uq groups of size vp*w_out*h_out
        ggml_tensor* hb_merged = ggml_reshape_2d(ctx, hb, w_out * h_out * vp, uq * batch);
        ggml_tensor* hc_t;
        ggml_tensor* hb_merged_t = ggml_cont(ctx, ggml_transpose(ctx, hb_merged));
        if (w1 != nullptr) {
            // Would be great to be able to transpose w1 instead to avoid transposing both hb and hc
            hc_t = ggml_mul_mat(ctx, w1, hb_merged_t);
        } else {
            hc_t = ggml_mul_mat(ctx, w1b, ggml_mul_mat(ctx, w1a, hb_merged_t));
        }
        ggml_tensor* hc = ggml_transpose(ctx, hc_t);
        // ungroup
        ggml_tensor* out = ggml_reshape_4d(ctx, ggml_cont(ctx, hc), w_out, h_out, up * vp, batch);
        return ggml_scale(ctx, out, scale);
    }
}

#endif  // __GGML_EXTEND__HPP__
