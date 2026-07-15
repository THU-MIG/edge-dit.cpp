#include "dit_models/pipelines/flux_kontext_pipeline.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <vector>

#include "core/optimization/cache/runtime/cache_engine.hpp"
#include "dit_models/components/autoencoders/auto_encoder_kl.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"
#include "dit_models/models/flux.hpp"
#include "ggml.h"
#include "parallel/cfg_parallel.hpp"
#include "utils/util.h"

namespace {

static constexpr size_t KONTEXT_MODEL_EXAMPLE_LIMIT = 3;

std::string kontext_tensor_component_name(const std::string& name) {
    if (starts_with(name, "model.diffusion_model.")) {
        return "diffusion";
    }
    if (starts_with(name, "first_stage_model.") || starts_with(name, "vae.")) {
        return "vae";
    }
    if (starts_with(name, "text_encoders.clip_l.") || starts_with(name, "cond_stage_model.")) {
        return "clip_l";
    }
    if (starts_with(name, "text_encoders.t5xxl.")) {
        return "t5xxl";
    }
    if (starts_with(name, "text_encoders.")) {
        return "text_encoder";
    }
    return "other";
}

std::string kontext_format_type_counts(const std::map<ggml_type, uint32_t>& type_counts) {
    std::ostringstream ss;
    bool first = true;
    for (const auto& item : type_counts) {
        if (!first) {
            ss << ", ";
        }
        first = false;
        ss << ggml_type_name(item.first) << "=" << item.second;
    }
    return ss.str();
}

template <typename T>
std::string kontext_tensor_shape(const sd::Tensor<T>& tensor) {
    if (tensor.empty()) {
        return "[]";
    }
    std::ostringstream ss;
    ss << "[";
    const auto& shape = tensor.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            ss << ", ";
        }
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

bool kontext_split_tensor_chunk_base(const std::string& name, std::string* base, int* chunk_index) {
    if (ends_with(name, ".weight") || ends_with(name, ".bias")) {
        if (base != nullptr) {
            *base = name;
        }
        if (chunk_index != nullptr) {
            *chunk_index = 0;
        }
        return true;
    }

    const size_t dot = name.rfind('.');
    if (dot == std::string::npos || dot + 1 >= name.size()) {
        return false;
    }

    int value = 0;
    for (size_t i = dot + 1; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return false;
        }
        value = value * 10 + (name[i] - '0');
    }

    const std::string candidate_base = name.substr(0, dot);
    if (!ends_with(candidate_base, ".weight") && !ends_with(candidate_base, ".bias")) {
        return false;
    }

    if (base != nullptr) {
        *base = candidate_base;
    }
    if (chunk_index != nullptr) {
        *chunk_index = value;
    }
    return true;
}

bool kontext_tensor_shape_matches_storage(const ggml_tensor* expected, const TensorStorage& storage) {
    for (int i = 0; i < 4; ++i) {
        if (expected->ne[i] != storage.ne[i]) {
            return false;
        }
    }
    return true;
}

bool kontext_tensor_decl_matches_split_storage(const ggml_tensor* expected,
                                               const String2TensorStorage& storage_map,
                                               const std::string& base_name) {
    auto first_it = storage_map.find(base_name);
    if (first_it == storage_map.end()) {
        return false;
    }

    const TensorStorage& first = first_it->second;
    int concat_dim = -1;
    for (int dim = 0; dim < 4; ++dim) {
        if (expected->ne[dim] <= first.ne[dim]) {
            continue;
        }

        bool other_dims_match = true;
        for (int i = 0; i < 4; ++i) {
            if (i == dim) {
                continue;
            }
            if (expected->ne[i] != first.ne[i]) {
                other_dims_match = false;
                break;
            }
        }
        if (other_dims_match) {
            concat_dim = dim;
            break;
        }
    }

    if (concat_dim < 0) {
        return false;
    }

    int64_t concat_size = 0;
    for (int chunk = 0;; ++chunk) {
        const std::string chunk_name = chunk == 0 ? base_name : base_name + "." + std::to_string(chunk);
        auto chunk_it = storage_map.find(chunk_name);
        if (chunk_it == storage_map.end()) {
            break;
        }

        const TensorStorage& storage = chunk_it->second;
        for (int i = 0; i < 4; ++i) {
            if (i == concat_dim) {
                continue;
            }
            if (storage.ne[i] != expected->ne[i]) {
                return false;
            }
        }
        concat_size += storage.ne[concat_dim];
    }

    return concat_size == expected->ne[concat_dim];
}

int64_t kontext_offset_4d(const sd::Tensor<float>& tensor,
                          int64_t i0,
                          int64_t i1 = 0,
                          int64_t i2 = 0,
                          int64_t i3 = 0) {
    const auto& shape = tensor.shape();
    const int64_t n0 = shape.size() > 0 ? shape[0] : 1;
    const int64_t n1 = shape.size() > 1 ? shape[1] : 1;
    const int64_t n2 = shape.size() > 2 ? shape[2] : 1;
    return ((i3 * n2 + i2) * n1 + i1) * n0 + i0;
}

void kontext_set_4d(sd::Tensor<float>& tensor,
                    float value,
                    int64_t i0,
                    int64_t i1 = 0,
                    int64_t i2 = 0,
                    int64_t i3 = 0) {
    tensor.values()[static_cast<size_t>(kontext_offset_4d(tensor, i0, i1, i2, i3))] = value;
}

uint8_t kontext_float_to_u8(float value) {
    if (value <= 0.0f) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    return static_cast<uint8_t>(value * 255.0f + 0.5f);
}

void kontext_tensor_to_image_data(const sd::Tensor<float>& tensor, uint8_t* image_data) {
    const auto& shape = tensor.shape();
    const int width = static_cast<int>(shape[0]);
    const int height = static_cast<int>(shape[1]);
    const int channels = static_cast<int>(shape[2]);
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const float* src = tensor.data();

    if (channels == 3) {
        const float* c0 = src;
        const float* c1 = src + pixels;
        const float* c2 = src + pixels * 2;
        for (size_t i = 0; i < pixels; ++i) {
            image_data[i * 3 + 0] = kontext_float_to_u8(c0[i]);
            image_data[i * 3 + 1] = kontext_float_to_u8(c1[i]);
            image_data[i * 3 + 2] = kontext_float_to_u8(c2[i]);
        }
        return;
    }

    for (size_t i = 0; i < pixels; ++i) {
        for (int c = 0; c < channels; ++c) {
            image_data[i * static_cast<size_t>(channels) + static_cast<size_t>(c)] =
                kontext_float_to_u8(src[i + pixels * static_cast<size_t>(c)]);
        }
    }
}

float kontext_time_shift(float mu, float sigma, float t) {
    return std::exp(mu) / (std::exp(mu) + std::pow((1.0f / t - 1.0f), sigma));
}

float kontext_t_to_sigma(float t, float shift) {
    t = t + 1.0f;
    return kontext_time_shift(shift, 1.0f, t / 1000.0f);
}

std::vector<float> kontext_discrete_sigmas(int steps, float shift) {
    std::vector<float> result;
    if (steps <= 0) {
        return result;
    }
    if (steps == 1) {
        result.push_back(kontext_t_to_sigma(999.0f, shift));
        result.push_back(0.0f);
        return result;
    }

    const float step = 999.0f / static_cast<float>(steps - 1);
    result.reserve(static_cast<size_t>(steps) + 1);
    for (int i = 0; i < steps; ++i) {
        const float t = 999.0f - step * static_cast<float>(i);
        result.push_back(kontext_t_to_sigma(t, shift));
    }
    result.push_back(0.0f);
    return result;
}

ed_status_t kontext_tensor_to_image(const sd::Tensor<float>& tensor, ed_image_t* image) {
    if (image == nullptr || tensor.empty()) {
        return ED_STATUS_INVALID_ARGUMENT;
    }
    const auto& shape = tensor.shape();
    if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0) {
        return ED_STATUS_INVALID_ARGUMENT;
    }

    const size_t width    = static_cast<size_t>(shape[0]);
    const size_t height   = static_cast<size_t>(shape[1]);
    const size_t channels = static_cast<size_t>(shape[2]);
    uint8_t* data = static_cast<uint8_t*>(std::malloc(width * height * channels));
    if (data == nullptr) {
        return ED_STATUS_OUT_OF_MEMORY;
    }

    kontext_tensor_to_image_data(tensor, data);
    image->width    = static_cast<uint32_t>(width);
    image->height   = static_cast<uint32_t>(height);
    image->channels = static_cast<uint32_t>(channels);
    image->data     = data;
    return ED_STATUS_OK;
}

sd::Tensor<float> kontext_image_to_tensor(const ed_image_t& image,
                                          int width,
                                          int height,
                                          int channels) {
    if (image.data == nullptr || image.width == 0 || image.height == 0 || image.channels == 0 ||
        width <= 0 || height <= 0 || channels <= 0) {
        return {};
    }

    sd::Tensor<float> tensor({static_cast<int64_t>(image.width),
                              static_cast<int64_t>(image.height),
                              static_cast<int64_t>(channels),
                              1});
    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            const uint8_t* pixel = image.data + (static_cast<size_t>(y) * image.width + x) * image.channels;
            for (int c = 0; c < channels; ++c) {
                const uint32_t src_c = static_cast<uint32_t>(std::min<int>(c, static_cast<int>(image.channels) - 1));
                kontext_set_4d(tensor, pixel[src_c] / 255.0f, x, y, c, 0);
            }
        }
    }

    if (static_cast<int>(image.width) == width && static_cast<int>(image.height) == height) {
        return tensor;
    }
    return sd::ops::interpolate(tensor, {width, height, channels, 1});
}

}  // namespace

namespace edgedit {

FluxKontextPipeline::FluxKontextPipeline(SDVersion version)
    : version_(version) {
}

FluxKontextPipeline::~FluxKontextPipeline() {
    reset_flux_runner();
}

bool FluxKontextPipeline::prepare(const ed_context_params_t& params,
                                  ModelRuntime& runtime,
                                  const ModelLoader& loader,
                                  PipelineTensorRegistry& registry,
                                  std::string* error) {
    (void)params;
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();

    if (version_ != VERSION_FLUX_KONTEXT) {
        if (error != nullptr) {
            *error = "FluxKontextPipeline got non-FLUX.1-Kontext-dev model version";
        }
        return false;
    }

    build_manifest(loader);
    if (!validate(error)) {
        return false;
    }
    if (!initialize_flux_transformer_spec(loader,
                                          runtime_->backend(),
                                          runtime_->offload_params_to_cpu(),
                                          error)) {
        return false;
    }

    return prepare_flux_runtime_weights(loader,
                                        runtime_->backend(),
                                        runtime_->clip_backend(),
                                        runtime_->vae_backend(),
                                        runtime_->offload_params_to_cpu(),
                                        registry,
                                        error);
}

void FluxKontextPipeline::mark_ready() {
    const bool ok = runtime_ != nullptr &&
                    version_ == VERSION_FLUX_KONTEXT &&
                    flux_runner_ != nullptr;
    if (ok) {
        runtime_weights_loaded_ = true;
    }
    ready_ = ok;
}

void FluxKontextPipeline::reset_flux_runner() {
    conditioner_.reset();
    vae_.reset();
    flux_runner_.reset();
    flux_backend_ = nullptr;
    conditioner_backend_ = nullptr;
    vae_backend_ = nullptr;
    flux_declared_tensors_ = 0;
    runtime_weights_loaded_ = false;
    flux_missing_tensors_.clear();
    flux_shape_mismatch_tensors_.clear();
    flux_unexpected_tensors_.clear();
}

KontextPipelineComponent* FluxKontextPipeline::find_or_add_component(const std::string& name) {
    for (KontextPipelineComponent& component : components_) {
        if (component.name == name) {
            return &component;
        }
    }
    components_.push_back({});
    components_.back().name = name;
    return &components_.back();
}

bool FluxKontextPipeline::has_component(const std::string& name) const {
    for (const KontextPipelineComponent& component : components_) {
        if (component.name == name && component.tensor_count > 0) {
            return true;
        }
    }
    return false;
}

void FluxKontextPipeline::build_manifest(const ModelLoader& loader) {
    components_.clear();

    for (const auto& item : loader.get_tensor_storage_map()) {
        const TensorStorage& tensor = item.second;
        KontextPipelineComponent* component = find_or_add_component(kontext_tensor_component_name(tensor.name));
        component->tensor_count++;
        component->bytes += tensor.nbytes_to_read();
        component->type_counts[tensor.type]++;
        if (component->examples.size() < KONTEXT_MODEL_EXAMPLE_LIMIT) {
            component->examples.push_back(tensor.name);
        }
    }

    std::sort(components_.begin(), components_.end(), [](const KontextPipelineComponent& a, const KontextPipelineComponent& b) {
        return a.name < b.name;
    });

    for (const KontextPipelineComponent& component : components_) {
        LOG_INFO("model component %-12s tensors=%zu bytes=%.2fMB types=[%s]",
                 component.name.c_str(),
                 component.tensor_count,
                 component.bytes / 1024.0 / 1024.0,
                 kontext_format_type_counts(component.type_counts).c_str());
        for (const std::string& example : component.examples) {
            LOG_DEBUG("  example tensor: %s", example.c_str());
        }
    }
}

bool FluxKontextPipeline::validate(std::string* error) const {
    if (!has_component("diffusion")) {
        if (error != nullptr) {
            *error = "FLUX.1-Kontext-dev model is missing diffusion/transformer tensors";
        }
        return false;
    }
    if (!has_component("clip_l")) {
        LOG_WARN("FLUX.1-Kontext-dev manifest has no CLIP-L text encoder tensors; this is OK for transformer-only files");
    }
    if (!has_component("t5xxl")) {
        LOG_WARN("FLUX.1-Kontext-dev manifest has no T5XXL text encoder tensors; this is OK for transformer-only files");
    }
    if (!has_component("vae")) {
        LOG_WARN("FLUX.1-Kontext-dev manifest has no VAE tensors; this is OK for transformer-only files");
    }
    return true;
}

bool FluxKontextPipeline::initialize_flux_transformer_spec(const ModelLoader& loader,
                                                           ggml_backend_t backend,
                                                           bool offload_params_to_cpu,
                                                           std::string* error) {
    reset_flux_runner();

    if (backend == nullptr) {
        if (error != nullptr) {
            *error = "FluxKontextPipeline requires a non-null diffusion backend from ModelRuntime";
        }
        return false;
    }
    flux_backend_ = backend;

    try {
        flux_runner_.reset(new Flux::FluxRunner(flux_backend_,
                                                offload_params_to_cpu,
                                                loader.get_tensor_storage_map(),
                                                "model.diffusion_model",
                                                version_,
                                                false));
        if (runtime_ != nullptr) {
            flux_runner_->set_max_graph_vram_bytes(runtime_->max_graph_vram_bytes());
            flux_runner_->set_flash_attention_enabled(runtime_->flash_attention());

            auto process_group = runtime_->graph_process_group_ref();
            if (process_group != nullptr) {
                flux_runner_->set_process_group(process_group);
                LOG_INFO("flux-kontext transformer process group attached: backend=%s rank=%d world_size=%d",
                         edgedit::parallel::backend_name(process_group->backend()),
                         process_group->rank(),
                         process_group->size());
            }
        }
    } catch (const std::exception& e) {
        if (error != nullptr) {
            *error = std::string("failed to initialize FLUX.1-Kontext-dev parameter spec: ") + e.what();
        }
        reset_flux_runner();
        return false;
    } catch (...) {
        if (error != nullptr) {
            *error = "failed to initialize FLUX.1-Kontext-dev parameter spec";
        }
        reset_flux_runner();
        return false;
    }

    std::map<std::string, ggml_tensor*> declared;
    flux_runner_->get_param_tensors(declared, "model.diffusion_model");
    flux_declared_tensors_ = static_cast<int>(declared.size());

    for (const auto& item : declared) {
        auto storage_it = loader.get_tensor_storage_map().find(item.first);
        if (storage_it == loader.get_tensor_storage_map().end()) {
            flux_missing_tensors_.push_back(item.first);
            continue;
        }

        const TensorStorage& storage = storage_it->second;
        const ggml_tensor* expected = item.second;
        bool shape_matches = kontext_tensor_shape_matches_storage(expected, storage) ||
                             kontext_tensor_decl_matches_split_storage(expected,
                                                                       loader.get_tensor_storage_map(),
                                                                       item.first);
        if (!shape_matches) {
            flux_shape_mismatch_tensors_.push_back(sd_format("%s file=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64
                                                            "] flux=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]",
                                                            item.first.c_str(),
                                                            storage.ne[0],
                                                            storage.ne[1],
                                                            storage.ne[2],
                                                            storage.ne[3],
                                                            expected->ne[0],
                                                            expected->ne[1],
                                                            expected->ne[2],
                                                            expected->ne[3]));
        }
    }

    for (const auto& item : loader.get_tensor_storage_map()) {
        const std::string& name = item.first;
        if (!starts_with(name, "model.diffusion_model.")) {
            continue;
        }
        std::string split_base;
        int split_index = 0;
        if (kontext_split_tensor_chunk_base(name, &split_base, &split_index) &&
            split_index > 0 &&
            declared.find(split_base) != declared.end()) {
            continue;
        }
        if (declared.find(name) == declared.end()) {
            flux_unexpected_tensors_.push_back(name);
        }
    }

    LOG_INFO("flux-kontext transformer spec declared %d tensors; missing=%zu shape_mismatch=%zu unexpected=%zu",
             flux_declared_tensors_,
             flux_missing_tensors_.size(),
             flux_shape_mismatch_tensors_.size(),
             flux_unexpected_tensors_.size());

    const size_t preview_limit = 8;
    for (size_t i = 0; i < std::min(preview_limit, flux_missing_tensors_.size()); ++i) {
        LOG_WARN("  missing flux-kontext tensor: %s", flux_missing_tensors_[i].c_str());
    }
    for (size_t i = 0; i < std::min(preview_limit, flux_shape_mismatch_tensors_.size()); ++i) {
        LOG_WARN("  flux-kontext tensor shape mismatch: %s", flux_shape_mismatch_tensors_[i].c_str());
    }
    for (size_t i = 0; i < std::min(preview_limit, flux_unexpected_tensors_.size()); ++i) {
        LOG_WARN("  unexpected flux-kontext diffusion tensor: %s", flux_unexpected_tensors_[i].c_str());
    }

    if (!flux_missing_tensors_.empty()) {
        if (error != nullptr) {
            *error = "FLUX.1-Kontext-dev parameter spec has missing tensors; first missing tensor: " + flux_missing_tensors_.front();
        }
        return false;
    }

    return true;
}

bool FluxKontextPipeline::prepare_flux_runtime_weights(const ModelLoader& loader,
                                                       ggml_backend_t diffusion_backend,
                                                       ggml_backend_t text_backend,
                                                       ggml_backend_t vae_backend,
                                                       bool offload_params_to_cpu,
                                                       PipelineTensorRegistry& registry,
                                                       std::string* error) {
    registry.clear();
    runtime_weights_loaded_ = false;

    if (flux_runner_ == nullptr || (diffusion_backend != nullptr && flux_backend_ != diffusion_backend)) {
        if (!initialize_flux_transformer_spec(loader,
                                              diffusion_backend,
                                              offload_params_to_cpu,
                                              error)) {
            return false;
        }
    }

    if (flux_runner_ == nullptr) {
        if (error != nullptr) {
            *error = "FLUX.1-Kontext-dev parameter spec is not initialized";
        }
        return false;
    }

    if (!flux_runner_->alloc_params_buffer()) {
        if (error != nullptr) {
            *error = "failed to allocate FLUX.1-Kontext-dev transformer parameter buffer";
        }
        return false;
    }

    flux_runner_->get_param_tensors(registry.tensors(), "model.diffusion_model");

    if (has_component("clip_l") || has_component("t5xxl")) {
        if (text_backend == nullptr) {
            if (error != nullptr) {
                *error = "FluxKontextPipeline requires a non-null text encoder backend from ModelRuntime";
            }
            return false;
        }
        conditioner_backend_ = text_backend;

        conditioner_ = std::make_shared<FluxCLIPEmbedder>(conditioner_backend_,
                                                          offload_params_to_cpu,
                                                          loader.get_tensor_storage_map());

        conditioner_->alloc_params_buffer();
        conditioner_->get_param_tensors(registry.tensors());
    }

    if (has_component("vae")) {
        if (vae_backend == nullptr) {
            if (error != nullptr) {
                *error = "FluxKontextPipeline requires a non-null VAE backend from ModelRuntime";
            }
            return false;
        }
        vae_backend_ = vae_backend;

        vae_ = std::make_shared<AutoEncoderKL>(vae_backend_,
                                               offload_params_to_cpu,
                                               loader.get_tensor_storage_map(),
                                               "first_stage_model",
                                               false,
                                               false,
                                               version_);

        vae_->alloc_params_buffer();
        vae_->get_param_tensors(registry.tensors(), "first_stage_model");
    }

    registry.ignore_prefix("vae.");
    registry.ignore_prefix("cond_stage_model.");
    registry.ignore_prefix("model.diffusion_model.__x0__");
    registry.ignore_prefix("model.diffusion_model.__32x32__");
    registry.ignore_prefix("model.diffusion_model.__index_timestep_zero__");

    if (!conditioner_) {
        registry.ignore_prefix("text_encoders.");
    }

    if (!vae_) {
        registry.ignore_prefix("first_stage_model.");
    }

    return true;
}

bool FluxKontextPipeline::can_generate_image() const {
    return runtime_weights_loaded_ && flux_runner_ != nullptr && conditioner_ != nullptr && vae_ != nullptr;
}

bool FluxKontextPipeline::validate_image_params(const ed_image_generation_params_t* params,
                                                std::string* error) const {
    if (params == nullptr) {
        if (error != nullptr) {
            *error = "image generation params are null";
        }
        return false;
    }
    if (params->width <= 0 || params->height <= 0) {
        if (error != nullptr) {
            *error = "image width and height must be positive";
        }
        return false;
    }
    if (params->batch_count <= 0) {
        if (error != nullptr) {
            *error = "image batch_count must be positive";
        }
        return false;
    }
    if (params->ref_images == nullptr || params->ref_image_count <= 0) {
        if (error != nullptr) {
            *error = "FLUX.1-Kontext-dev requires an input image; use -i/--image in ed-cli";
        }
        return false;
    }
    return true;
}

bool FluxKontextPipeline::validate_video_params(const ed_video_generation_params_t* params,
                                                std::string* error) const {
    if (params == nullptr) {
        if (error != nullptr) {
            *error = "video generation params are null";
        }
        return false;
    }
    if (params->width <= 0 || params->height <= 0 || params->frames <= 0) {
        if (error != nullptr) {
            *error = "video width, height, and frames must be positive";
        }
        return false;
    }
    return true;
}

ed_status_t FluxKontextPipeline::generate_image(const ed_image_generation_params_t* params,
                                                ed_image_batch_t* out,
                                                std::string* error) {
    if (!ready_ || runtime_ == nullptr) {
        if (error != nullptr) {
            *error = "FluxKontextPipeline is not initialized";
        }
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (out == nullptr) {
        if (error != nullptr) {
            *error = "image output is null";
        }
        return ED_STATUS_INVALID_ARGUMENT;
    }
    out->images = nullptr;
    out->count  = 0;

    if (!validate_image_params(params, error)) {
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (!can_generate_image()) {
        if (error != nullptr) {
            *error = "current FLUX.1-Kontext-dev pipeline needs transformer, CLIP-L, T5XXL, and VAE weights";
        }
        return ED_STATUS_UNSUPPORTED;
    }

    const int count = params->batch_count > 0 ? params->batch_count : 1;
    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
    if (GenerationControl* control = runtime_->generation_control()) {
        control->start(count * steps);
    }
    ed_image_t* images = static_cast<ed_image_t*>(std::calloc(static_cast<size_t>(count), sizeof(ed_image_t)));
    if (images == nullptr) {
        if (error != nullptr) {
            *error = "failed to allocate image batch";
        }
        return ED_STATUS_OUT_OF_MEMORY;
    }

    for (int i = 0; i < count; ++i) {
        if (!generate_one_image(params, i, runtime_->n_threads(), &images[i], error)) {
            for (int j = 0; j <= i; ++j) {
                std::free(images[j].data);
            }
            std::free(images);
            return ED_STATUS_GENERATION_FAILED;
        }
    }

    out->images = images;
    out->count  = count;
    return ED_STATUS_OK;
}

ed_status_t FluxKontextPipeline::generate_video(const ed_video_generation_params_t* params,
                                                ed_video_t* out,
                                                std::string* error) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }
    if (!validate_video_params(params, error)) {
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (error != nullptr) {
        *error = "video generation is not implemented in FluxKontextPipeline";
    }
    return ED_STATUS_UNSUPPORTED;
}

bool FluxKontextPipeline::supports_image_generation() const {
    return ready_;
}

bool FluxKontextPipeline::supports_video_generation() const {
    return false;
}

ed_sampler_t FluxKontextPipeline::default_sample_method() const {
    return ED_SAMPLER_EULER;
}

ed_scheduler_t FluxKontextPipeline::default_scheduler(ed_sampler_t method) const {
    if (method == ED_SAMPLER_LCM || method == ED_SAMPLER_TCD) {
        return ED_SCHEDULER_LCM;
    }
    if (method == ED_SAMPLER_DDIM_TRAILING) {
        return ED_SCHEDULER_SIMPLE;
    }
    return ED_SCHEDULER_DISCRETE;
}

bool FluxKontextPipeline::generate_one_image(const ed_image_generation_params_t* params,
                                             int batch_index,
                                             int n_threads,
                                             ed_image_t* image,
                                             std::string* error) {
    if (params == nullptr || image == nullptr) {
        if (error != nullptr) {
            *error = "invalid FLUX.1-Kontext-dev generation arguments";
        }
        return false;
    }

    const int vae_scale_factor = vae_->get_scale_factor();
    if (params->width <= 0 || params->height <= 0 ||
        params->width % vae_scale_factor != 0 ||
        params->height % vae_scale_factor != 0) {
        if (error != nullptr) {
            *error = sd_format("FLUX.1-Kontext-dev image size must be positive and divisible by VAE scale factor %d",
                               vae_scale_factor);
        }
        return false;
    }

    const int latent_w   = params->width / vae_scale_factor;
    const int latent_h   = params->height / vae_scale_factor;
    const int patch_size = std::max<int>(1, flux_runner_->flux_params.patch_size);
    if (latent_w % patch_size != 0 || latent_h % patch_size != 0) {
        if (error != nullptr) {
            *error = sd_format("FLUX.1-Kontext-dev latent size %dx%d must be divisible by patch size %d",
                               latent_w,
                               latent_h,
                               patch_size);
        }
        return false;
    }

    ConditionerParams cond_params;
    cond_params.text      = params->prompt != nullptr ? params->prompt : "";
    cond_params.clip_skip = -1;
    SDCondition condition = conditioner_->get_learned_condition(n_threads, cond_params);
    if (condition.empty()) {
        if (error != nullptr) {
            *error = "FLUX.1-Kontext-dev prompt encoding returned empty condition";
        }
        return false;
    }

    const float cfg_scale = params->sample.cfg_scale > 0.0f ? params->sample.cfg_scale : 1.0f;
    SDCondition uncond;
    if (cfg_scale != 1.0f) {
        ConditionerParams uncond_params;
        uncond_params.text      = params->negative_prompt != nullptr ? params->negative_prompt : "";
        uncond_params.clip_skip = -1;
        uncond = conditioner_->get_learned_condition(n_threads, uncond_params);
        if (uncond.empty()) {
            if (error != nullptr) {
                *error = "FLUX.1-Kontext-dev negative prompt encoding returned empty condition";
            }
            return false;
        }
    }
    LOG_INFO("flux-kontext prompt encoded: cross_attn=%s vector=%s",
             kontext_tensor_shape(condition.c_crossattn).c_str(),
             kontext_tensor_shape(condition.c_vector).c_str());

    std::shared_ptr<RNG> rng = runtime_->rng_ptr();
    if (!rng) {
        if (error != nullptr) {
            *error = "FluxKontextPipeline has no RNG from ModelRuntime";
        }
        return false;
    }

    std::vector<sd::Tensor<float>> ref_latents;
    ref_latents.reserve(static_cast<size_t>(params->ref_image_count));
    const int64_t encode_start_ms = ggml_time_ms();
    for (int i = 0; i < params->ref_image_count; ++i) {
        sd::Tensor<float> ref_image = kontext_image_to_tensor(params->ref_images[i],
                                                              params->width,
                                                              params->height,
                                                              3);
        if (ref_image.empty()) {
            if (error != nullptr) {
                *error = sd_format("FLUX.1-Kontext-dev reference image %d is invalid", i);
            }
            return false;
        }
        sd::Tensor<float> encoded = vae_->encode(n_threads,
                                                 ref_image,
                                                 runtime_->vae_tiling(),
                                                 false,
                                                 false);
        if (encoded.empty()) {
            if (error != nullptr) {
                *error = sd_format("FLUX.1-Kontext-dev VAE encode failed for reference image %d", i);
            }
            return false;
        }
        sd::Tensor<float> latent = vae_->vae_output_to_latents(encoded, rng);
        latent = vae_->vae_to_diffusion_latents(latent);
        ref_latents.push_back(std::move(latent));
    }
    LOG_INFO("flux-kontext reference encoding completed: refs=%zu taking %.2fs",
             ref_latents.size(),
             (ggml_time_ms() - encode_start_ms) / 1000.0f);

    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
    float flow_shift = params->sample.flow_shift;
    if (!(flow_shift > 0.0f) || !std::isfinite(flow_shift)) {
        flow_shift = flux_runner_->flux_params.guidance_embed ? 1.15f : 1.0f;
    }
    const float distilled_guidance = params->sample.distilled_guidance != 0.0f
                                         ? params->sample.distilled_guidance
                                         : 3.5f;
    const int64_t seed = params->seed >= 0 ? params->seed : 42;
    rng->manual_seed(static_cast<uint64_t>(seed + batch_index));

    sd::Tensor<float> init_latent = sd::zeros<float>({latent_w, latent_h, 16, 1});
    sd::Tensor<float> noise = sd::Tensor<float>::randn(init_latent.shape(), rng);
    std::vector<float> sigmas = kontext_discrete_sigmas(steps, flow_shift);
    if (sigmas.size() < 2) {
        if (error != nullptr) {
            *error = "failed to create FLUX.1-Kontext-dev sigma schedule";
        }
        return false;
    }

    LOG_INFO("flux-kontext img2img: %dx%d latent=%dx%d refs=%zu steps=%d shift=%.2f guidance=%.2f cfg=%.2f seed=%" PRId64,
             params->width,
             params->height,
             latent_w,
             latent_h,
             ref_latents.size(),
             steps,
             flow_shift,
             distilled_guidance,
             cfg_scale,
             seed + batch_index);

    sd::Tensor<float> x = init_latent * (1.0f - sigmas[0]) + noise * sigmas[0];
    cache::CacheRuntime cache_runtime;
    const bool cache_use_cfg_parallel = !uncond.empty() &&
                                        parallel::cfg_parallel_available(runtime_->parallel_context());
    const bool cache_seam_available =
        !cache_use_cfg_parallel && flux_runner_->feature_cache_available();
    // Wire the device store only when the on-GPU feature-reuse path is active
    // (ED_FEATURE_CACHE_GPU on); with it off, leave the store null so a
    // device_backed slot cleanly falls back to the host declarative path.
    cache::ICacheDeviceStore* cache_store =
        (cache_seam_available && flux_runner_ != nullptr &&
         Flux::FluxRunner::feature_gpu_enabled())
            ? flux_runner_->cache_device_store()
            : nullptr;
    const bool cache_enabled =
        cache_runtime.init(params->sample, version_, sigmas, cache_seam_available, cache_store,
                           cache_use_cfg_parallel);
    // GPU DiCache (ED_DICACHE_GPU): reset per-generation persistent state and set
    // the probe depth the capture step uses to snapshot its probe residual. Read
    // the resolved depth from the engine so it stays in sync with the policy config.
    if (cache_enabled && flux_runner_ != nullptr) {
        flux_runner_->reset_dicache_gpu_states();
        flux_runner_->dicache_probe_depth_ = cache_runtime.dicache_probe_depth();
    }
    const int64_t sample_start_ms = ggml_time_ms();
    GenerationControl* control = runtime_ != nullptr ? runtime_->generation_control() : nullptr;
    for (int step = 0; step < steps; ++step) {
        if (control != nullptr && control->should_cancel()) {
            control->mark_cancelled();
            if (error != nullptr && error->empty()) {
                *error = "generation cancelled";
            }
            flux_runner_->free_compute_buffer();
            return false;
        }
        const float sigma      = sigmas[static_cast<size_t>(step)];
        const float sigma_next = sigmas[static_cast<size_t>(step + 1)];
        const float c_skip     = 1.0f;
        const float c_out      = -sigma;

        sd::Tensor<float> timesteps({1}, std::vector<float>{sigma});
        sd::Tensor<float> guidance({1}, std::vector<float>{distilled_guidance});
        sd::Tensor<float> noised_input = x;

        cache::CacheStepInfo cache_step;
        cache_step.step_index = step;
        cache_step.num_steps  = steps;
        cache_step.sigma      = sigma;
        cache_step.sigma_next = sigma_next;
        if (cache_enabled) {
            cache_runtime.begin_step(cache_step);
        }

        const bool use_cfg_parallel = !uncond.empty() &&
                                      parallel::cfg_parallel_available(runtime_->parallel_context());
        const int cfg_rank = parallel::cfg_parallel_rank(runtime_->parallel_context());

        sd::Tensor<float> model_out;
        const void* condition_key = static_cast<const void*>(&condition);
        const cache::CacheBranch condition_branch = uncond.empty() ? cache::CacheBranch::Main
                                                                   : cache::CacheBranch::Cond;

        // Cache hooks for one condition. Feature/Probe seam gated to the plain
        // path and disabled under CFG-parallel (see flux_pipeline). Kontext
        // threads ref_latents (increase_ref_index=false) through every pass.
        auto make_hooks = [&](const SDCondition& cond_in) {
            cache::CacheRunnerHooks hooks;
            hooks.input = &noised_input;
            hooks.full = [&]() {
                return flux_runner_->compute(n_threads, noised_input, timesteps,
                                             cond_in.c_crossattn, {}, cond_in.c_vector,
                                             guidance, ref_latents);
            };
            const bool seam_ok = !use_cfg_parallel && flux_runner_->feature_cache_available();
            if (seam_ok) {
                const void* branch_key = static_cast<const void*>(&cond_in);
                const bool is_probe = cache_runtime.granularity() == cache::CacheGranularity::Probe;
                const bool feature_gpu = !is_probe &&
                    cache_runtime.granularity() == cache::CacheGranularity::Feature &&
                    Flux::FluxRunner::feature_gpu_enabled();
                if (feature_gpu) {
                    hooks.substep_capture = [&](std::vector<cache::GraphExtension> exts) {
                        return flux_runner_->compute_substep_capture(
                            n_threads, noised_input, timesteps, cond_in.c_crossattn, {},
                            cond_in.c_vector, guidance, ref_latents, false, std::move(exts));
                    };
                    hooks.substep_inject_slot = [&](std::vector<cache::GraphExtension> exts) {
                        return flux_runner_->compute_substep_inject_slot(
                            n_threads, noised_input, timesteps, cond_in.c_crossattn, {},
                            cond_in.c_vector, guidance, ref_latents, false, std::move(exts));
                    };
                }
                if (cache_runtime.granularity() == cache::CacheGranularity::Probe) {
                    const bool delta_minus = cache_runtime.dicache_delta_minus();
                    hooks.substep_probe = [&, branch_key, delta_minus](int depth) {
                        return flux_runner_->compute_substep_probe(n_threads, noised_input, timesteps,
                                                                   cond_in.c_crossattn, {}, cond_in.c_vector,
                                                                   guidance, ref_latents, false, depth, branch_key,
                                                                   delta_minus);
                    };
                    if (Flux::FluxRunner::dicache_gpu_enabled()) {
                        hooks.substep_inject_gpu = [&, branch_key](std::vector<cache::GraphExtension> exts) {
                            return flux_runner_->compute_substep_inject_gpu(n_threads, noised_input, timesteps,
                                                                            cond_in.c_crossattn, {}, cond_in.c_vector,
                                                                            guidance, ref_latents, false, std::move(exts), branch_key);
                        };
                        const int probe_depth = cache_runtime.dicache_probe_depth();
                        hooks.substep_capture_probe = [&, branch_key, probe_depth]() {
                            return flux_runner_->compute_substep_capture_probe(
                                n_threads, noised_input, timesteps, cond_in.c_crossattn, {}, cond_in.c_vector,
                                guidance, ref_latents, false, probe_depth, branch_key);
                        };
                    }
                }
            }
            return hooks;
        };

        if (use_cfg_parallel) {
            const bool local_is_uncond = cfg_rank == 0;
            const SDCondition& local_condition = local_is_uncond ? uncond : condition;
            const cache::CacheBranch local_branch = local_is_uncond ? cache::CacheBranch::Uncond
                                                                    : cache::CacheBranch::Cond;
            const void* local_key = static_cast<const void*>(&local_condition);
            sd::Tensor<float> local_out = cache_enabled
                ? cache_runtime.run_branch(local_branch, local_key, make_hooks(local_condition))
                : flux_runner_->compute(n_threads, noised_input, timesteps,
                                        local_condition.c_crossattn, {}, local_condition.c_vector,
                                        guidance, ref_latents);
            std::vector<sd::Tensor<float>> gathered;
            if (local_out.empty() ||
                !parallel::cfg_all_gather(*runtime_->parallel_context(), local_out, &gathered, error) ||
                gathered.size() != 2) {
                if (error != nullptr && error->empty()) {
                    *error = sd_format("FLUX.1-Kontext-dev CFG parallel gather failed at step %d", step + 1);
                }
                flux_runner_->free_compute_buffer();
                return false;
            }
            model_out = gathered[0] + cfg_scale * (gathered[1] - gathered[0]);
        } else {
            model_out = cache_enabled
                ? cache_runtime.run_branch(condition_branch, condition_key, make_hooks(condition))
                : flux_runner_->compute(n_threads, noised_input, timesteps,
                                        condition.c_crossattn, {}, condition.c_vector,
                                        guidance, ref_latents);
        }
        if (!uncond.empty() && !use_cfg_parallel) {
            const void* uncond_key = static_cast<const void*>(&uncond);
            sd::Tensor<float> uncond_out = cache_enabled
                ? cache_runtime.run_branch(cache::CacheBranch::Uncond, uncond_key, make_hooks(uncond))
                : flux_runner_->compute(n_threads, noised_input, timesteps,
                                        uncond.c_crossattn, {}, uncond.c_vector,
                                        guidance, ref_latents);
            if (uncond_out.empty()) {
                if (error != nullptr) {
                    *error = sd_format("FLUX.1-Kontext-dev unconditional transformer compute failed at step %d", step + 1);
                }
                flux_runner_->free_compute_buffer();
                return false;
            }
            model_out = uncond_out + cfg_scale * (model_out - uncond_out);
        }
        if (model_out.empty()) {
            if (error != nullptr) {
                *error = sd_format("FLUX.1-Kontext-dev transformer compute failed at step %d", step + 1);
            }
            flux_runner_->free_compute_buffer();
            return false;
        }

        // SenCache calibration: finite-diff sensitivities on the CFG-combined
        // velocity. Two extra plain forwards/step, calibration only; off under
        // CFG-parallel. Kontext feeds timestep = sigma and threads ref_latents.
        // The policy owns the protocol; the pipeline supplies only forward_at.
        if (cache_enabled && cache_runtime.needs_calibration() && !use_cfg_parallel) {
            auto forward_at = [&](const sd::Tensor<float>& x_raw, float sigma_eval) -> sd::Tensor<float> {
                sd::Tensor<float> ts({1}, std::vector<float>{sigma_eval});
                sd::Tensor<float> cond_v = flux_runner_->compute(n_threads, x_raw, ts,
                                                                 condition.c_crossattn, {}, condition.c_vector,
                                                                 guidance, ref_latents);
                if (cond_v.empty() || uncond.empty()) {
                    return cond_v;
                }
                sd::Tensor<float> uncond_v = flux_runner_->compute(n_threads, x_raw, ts,
                                                                   uncond.c_crossattn, {}, uncond.c_vector,
                                                                   guidance, ref_latents);
                if (uncond_v.empty()) {
                    return {};
                }
                return uncond_v + cfg_scale * (cond_v - uncond_v);
            };
            cache_runtime.calibrate(condition_branch, condition_key, x, model_out, forward_at);
        }

        sd::Tensor<float> denoised = model_out * c_out + x * c_skip;
        if (sigma == 0.0f) {
            x = denoised;
        } else {
            const sd::Tensor<float> d = (x - denoised) / sigma;
            x += d * (sigma_next - sigma);
        }
        LOG_INFO("flux-kontext step %d/%d sigma=%.6f next=%.6f", step + 1, steps, sigma, sigma_next);
        if (cache_enabled) {
            cache_runtime.end_step(cache_step);
        }
        if (control != nullptr) {
            control->step_done();
        }
    }
    if (cache_enabled) {
        cache_runtime.log_summary(static_cast<size_t>(steps));
    }
    LOG_INFO("flux-kontext sampling completed, taking %.2fs", (ggml_time_ms() - sample_start_ms) / 1000.0f);
    flux_runner_->free_compute_buffer();

    if (runtime_->parallel_context() != nullptr && !runtime_->parallel_context()->is_root()) {
        return true;
    }

    sd::Tensor<float> vae_latents = vae_->diffusion_to_vae_latents(x);
    sd::Tensor<float> decoded = vae_->decode(n_threads,
                                             vae_latents,
                                             runtime_->vae_tiling(),
                                             false,
                                             false,
                                             false);
    if (decoded.empty()) {
        if (error != nullptr) {
            *error = "FLUX.1-Kontext-dev VAE decode failed";
        }
        return false;
    }

    const ed_status_t status = kontext_tensor_to_image(decoded, image);
    if (status != ED_STATUS_OK) {
        if (error != nullptr) {
            *error = status == ED_STATUS_OUT_OF_MEMORY ? "failed to allocate decoded image"
                                                       : "decoded FLUX.1-Kontext-dev tensor has invalid shape";
        }
        return false;
    }
    return true;
}

}  // namespace edgedit
