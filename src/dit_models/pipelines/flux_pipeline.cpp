#include "dit_models/pipelines/flux_pipeline.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <sstream>

#include "core/optimization/cache/runtime/cache_engine.hpp"
#include "dit_models/components/autoencoders/auto_encoder_kl.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"
#include "dit_models/models/flux.hpp"
#include "ggml.h"
#include "parallel/cfg_parallel.hpp"
#include "utils/preprocessing.hpp"
#include "utils/util.h"

static constexpr size_t ED_MODEL_EXAMPLE_LIMIT = 3;

static std::string tensor_component_name(const std::string& name) {
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

static std::string format_type_counts(const std::map<ggml_type, uint32_t>& type_counts) {
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
static std::string format_tensor_shape(const sd::Tensor<T>& tensor) {
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

static bool split_tensor_chunk_base(const std::string& name, std::string* base, int* chunk_index) {
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

static bool tensor_shape_matches_storage(const ggml_tensor* expected, const TensorStorage& storage) {
    for (int i = 0; i < 4; ++i) {
        if (expected->ne[i] != storage.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool tensor_decl_matches_split_storage(const ggml_tensor* expected,
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

static float ed_flux_time_shift(float mu, float sigma, float t) {
    return std::exp(mu) / (std::exp(mu) + std::pow((1.0f / t - 1.0f), sigma));
}

static float ed_flux_t_to_sigma(float t, float shift) {
    t = t + 1.0f;
    return ed_flux_time_shift(shift, 1.0f, t / 1000.0f);
}

static std::vector<float> ed_flux_discrete_sigmas(int steps, float shift) {
    std::vector<float> result;
    if (steps <= 0) {
        return result;
    }
    if (steps == 1) {
        result.push_back(ed_flux_t_to_sigma(999.0f, shift));
        result.push_back(0.0f);
        return result;
    }

    const float step = 999.0f / static_cast<float>(steps - 1);
    result.reserve(static_cast<size_t>(steps) + 1);
    for (int i = 0; i < steps; ++i) {
        const float t = 999.0f - step * static_cast<float>(i);
        result.push_back(ed_flux_t_to_sigma(t, shift));
    }
    result.push_back(0.0f);
    return result;
}

static ed_status_t ed_tensor_to_image(const sd::Tensor<float>& tensor, ed_image_t* image) {
    if (image == nullptr || tensor.empty()) {
        return ED_STATUS_INVALID_ARGUMENT;
    }
    const auto& shape = tensor.shape();
    if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0) {
        return ED_STATUS_INVALID_ARGUMENT;
    }

    const size_t width = static_cast<size_t>(shape[0]);
    const size_t height = static_cast<size_t>(shape[1]);
    const size_t channels = static_cast<size_t>(shape[2]);
    const size_t nbytes = width * height * channels;
    uint8_t* data = static_cast<uint8_t*>(std::malloc(nbytes));
    if (data == nullptr) {
        return ED_STATUS_OUT_OF_MEMORY;
    }

    preprocessing_tensor_frame_to_sd_image(tensor, 0, data);
    image->width = static_cast<uint32_t>(width);
    image->height = static_cast<uint32_t>(height);
    image->channels = static_cast<uint32_t>(channels);
    image->data = data;
    return ED_STATUS_OK;
}

namespace edgedit {

FluxPipeline::FluxPipeline(SDVersion version)
    : version_(version) {
}

FluxPipeline::~FluxPipeline() {
    reset_flux_runner();
}

bool FluxPipeline::prepare(const ed_context_params_t& params,
                           ModelRuntime& runtime,
                           const ModelLoader& loader,
                           PipelineTensorRegistry& registry,
                           std::string* error) {
    (void)params;
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();

    if (!ed_version_is_flux(version_) && !ed_version_is_flux2(version_)) {
        if (error != nullptr) {
            *error = "FluxPipeline got non-Flux model version";
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

void FluxPipeline::mark_ready() {
    const bool ok = runtime_ != nullptr &&
                    version_ != VERSION_COUNT &&
                    flux_runner_ != nullptr;
    if (ok) {
        runtime_weights_loaded_ = true;
    }
    ready_ = ok;
}

void FluxPipeline::reset_flux_runner() {
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

PipelineComponent* FluxPipeline::find_or_add_component(const std::string& name) {
    for (PipelineComponent& component : components_) {
        if (component.name == name) {
            return &component;
        }
    }
    components_.push_back({});
    components_.back().name = name;
    return &components_.back();
}

bool FluxPipeline::has_component(const std::string& name) const {
    for (const PipelineComponent& component : components_) {
        if (component.name == name && component.tensor_count > 0) {
            return true;
        }
    }
    return false;
}

void FluxPipeline::build_manifest(const ModelLoader& loader) {
    components_.clear();

    for (const auto& item : loader.get_tensor_storage_map()) {
        const TensorStorage& tensor = item.second;
        PipelineComponent* component = find_or_add_component(tensor_component_name(tensor.name));
        component->tensor_count++;
        component->bytes += tensor.nbytes_to_read();
        component->type_counts[tensor.type]++;
        if (component->examples.size() < ED_MODEL_EXAMPLE_LIMIT) {
            component->examples.push_back(tensor.name);
        }
    }

    std::sort(components_.begin(), components_.end(), [](const PipelineComponent& a, const PipelineComponent& b) {
        return a.name < b.name;
    });

    for (const PipelineComponent& component : components_) {
        LOG_INFO("model component %-12s tensors=%zu bytes=%.2fMB types=[%s]",
                 component.name.c_str(),
                 component.tensor_count,
                 component.bytes / 1024.0 / 1024.0,
                 format_type_counts(component.type_counts).c_str());
        for (const std::string& example : component.examples) {
            LOG_DEBUG("  example tensor: %s", example.c_str());
        }
    }
}

bool FluxPipeline::validate(std::string* error) const {
    auto has_component = [&](const std::string& name) {
        for (const PipelineComponent& component : components_) {
            if (component.name == name && component.tensor_count > 0) {
                return true;
            }
        }
        return false;
    };

    if (ed_version_is_flux(version_) || ed_version_is_flux2(version_)) {
        if (!has_component("diffusion")) {
            if (error != nullptr) {
                *error = "Flux model is missing diffusion/transformer tensors";
            }
            return false;
        }
        if (!has_component("clip_l")) {
            LOG_WARN("Flux manifest has no CLIP-L text encoder tensors; this is OK for transformer-only files");
        }
        if (!has_component("t5xxl")) {
            LOG_WARN("Flux manifest has no T5XXL text encoder tensors; this is OK for transformer-only files");
        }
        if (!has_component("vae")) {
            LOG_WARN("Flux manifest has no VAE tensors; this is OK for transformer-only files");
        }
    }

    return true;
}

bool FluxPipeline::initialize_flux_transformer_spec(const ModelLoader& loader,
                                               ggml_backend_t backend,
                                               bool offload_params_to_cpu,
                                               std::string* error) {
    reset_flux_runner();

    if (!ed_version_is_flux(version_) && !ed_version_is_flux2(version_)) {
        return true;
    }

    if (backend == nullptr) {
        if (error != nullptr) {
            *error = "FluxPipeline requires a non-null diffusion backend from ModelRuntime";
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
            const bool diffusion_flash = runtime_->flash_attention();
            flux_runner_->set_max_graph_vram_bytes(runtime_->max_graph_vram_bytes());
            flux_runner_->set_flash_attention_enabled(diffusion_flash);

            auto process_group = runtime_->graph_process_group_ref();
            if (process_group != nullptr) {
                flux_runner_->set_process_group(process_group);
                LOG_INFO("flux transformer process group attached: backend=%s rank=%d world_size=%d",
                         edgedit::parallel::backend_name(process_group->backend()),
                         process_group->rank(),
                         process_group->size());
            }
        }
    } catch (const std::exception& e) {
        if (error != nullptr) {
            *error = std::string("failed to initialize Flux parameter spec: ") + e.what();
        }
        reset_flux_runner();
        return false;
    } catch (...) {
        if (error != nullptr) {
            *error = "failed to initialize Flux parameter spec";
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
        bool shape_matches = tensor_shape_matches_storage(expected, storage) ||
                             tensor_decl_matches_split_storage(expected,
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
        if (split_tensor_chunk_base(name, &split_base, &split_index) &&
            split_index > 0 &&
            declared.find(split_base) != declared.end()) {
            continue;
        }
        if (declared.find(name) == declared.end()) {
            flux_unexpected_tensors_.push_back(name);
        }
    }

    LOG_INFO("flux transformer spec declared %d tensors; missing=%zu shape_mismatch=%zu unexpected=%zu",
             flux_declared_tensors_,
             flux_missing_tensors_.size(),
             flux_shape_mismatch_tensors_.size(),
             flux_unexpected_tensors_.size());

    const size_t preview_limit = 8;
    for (size_t i = 0; i < std::min(preview_limit, flux_missing_tensors_.size()); ++i) {
        LOG_WARN("  missing flux tensor: %s", flux_missing_tensors_[i].c_str());
    }
    for (size_t i = 0; i < std::min(preview_limit, flux_shape_mismatch_tensors_.size()); ++i) {
        LOG_WARN("  flux tensor shape mismatch: %s", flux_shape_mismatch_tensors_[i].c_str());
    }
    for (size_t i = 0; i < std::min(preview_limit, flux_unexpected_tensors_.size()); ++i) {
        LOG_WARN("  unexpected diffusion tensor: %s", flux_unexpected_tensors_[i].c_str());
    }

    if (!flux_missing_tensors_.empty()) {
        if (error != nullptr) {
            *error = "Flux parameter spec has missing tensors; first missing tensor: " + flux_missing_tensors_.front();
        }
        return false;
    }

    return true;
}

bool FluxPipeline::prepare_flux_runtime_weights(const ModelLoader& loader,
                                           ggml_backend_t diffusion_backend,
                                           ggml_backend_t text_backend,
                                           ggml_backend_t vae_backend,
                                           bool offload_params_to_cpu,
                                           PipelineTensorRegistry& registry,
                                           std::string* error) {
    if (!ed_version_is_flux(version_) && !ed_version_is_flux2(version_)) {
        return true;
    }

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
            *error = "Flux parameter spec is not initialized";
        }
        return false;
    }

    if (!flux_runner_->alloc_params_buffer()) {
        if (error != nullptr) {
            *error = "failed to allocate Flux transformer parameter buffer";
        }
        return false;
    }

    flux_runner_->get_param_tensors(registry.tensors(), "model.diffusion_model");

    if (has_component("clip_l") || has_component("t5xxl")) {
        if (text_backend == nullptr) {
            if (error != nullptr) {
                *error = "FluxPipeline requires a non-null text encoder backend from ModelRuntime";
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
                *error = "FluxPipeline requires a non-null VAE backend from ModelRuntime";
            }
            return false;
        }
        vae_backend_ = vae_backend;

        vae_ = std::make_shared<AutoEncoderKL>(vae_backend_,
                                               offload_params_to_cpu,
                                               loader.get_tensor_storage_map(),
                                               "first_stage_model",
                                               true,
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
    } else {
        registry.ignore_prefix("first_stage_model.encoder");
        registry.ignore_prefix("first_stage_model.quant");
    }

    return true;
}

bool FluxPipeline::can_generate_image() const {
    return runtime_weights_loaded_ && flux_runner_ != nullptr && conditioner_ != nullptr && vae_ != nullptr;
}

bool FluxPipeline::validate_image_params(const ed_image_generation_params_t* params, std::string* error) const {
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
    return true;
}

bool FluxPipeline::validate_video_params(const ed_video_generation_params_t* params, std::string* error) const {
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

ed_status_t FluxPipeline::generate_image(const ed_image_generation_params_t* params,
                                         ed_image_batch_t* out,
                                         std::string* error) {
    if (!ready_ || runtime_ == nullptr) {
        if (error != nullptr) {
            *error = "FluxPipeline is not initialized";
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
    out->count = 0;

    if (!validate_image_params(params, error)) {
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (!can_generate_image()) {
        if (error != nullptr) {
            *error = "current Flux pipeline needs transformer, CLIP-L, T5XXL, and VAE weights";
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
    out->count = count;
    return ED_STATUS_OK;
}

ed_status_t FluxPipeline::generate_video(const ed_video_generation_params_t* params,
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
        *error = "video generation is not implemented in FluxPipeline";
    }
    return ED_STATUS_UNSUPPORTED;
}

bool FluxPipeline::supports_image_generation() const {
    return ready_;
}

bool FluxPipeline::supports_video_generation() const {
    return false;
}

ed_sampler_t FluxPipeline::default_sample_method() const {
    return ED_SAMPLER_EULER;
}

ed_scheduler_t FluxPipeline::default_scheduler(ed_sampler_t method) const {
    if (method == ED_SAMPLER_LCM || method == ED_SAMPLER_TCD) {
        return ED_SCHEDULER_LCM;
    }
    if (method == ED_SAMPLER_DDIM_TRAILING) {
        return ED_SCHEDULER_SIMPLE;
    }
    return ED_SCHEDULER_DISCRETE;
}

bool FluxPipeline::generate_one_image(const ed_image_generation_params_t* params,
                                  int batch_index,
                                  int n_threads,
                                  ed_image_t* image,
                                  std::string* error) {
    if (params == nullptr || image == nullptr) {
        if (error != nullptr) {
            *error = "invalid Flux generation arguments";
        }
        return false;
    }
    if (!can_generate_image()) {
        if (error != nullptr) {
            *error = "full Flux runtime is not loaded; need transformer, CLIP-L, T5XXL, and VAE weights";
        }
        return false;
    }

    const int vae_scale_factor = vae_->get_scale_factor();
    if (params->width <= 0 || params->height <= 0 ||
        params->width % vae_scale_factor != 0 ||
        params->height % vae_scale_factor != 0) {
        if (error != nullptr) {
            *error = sd_format("Flux image size must be positive and divisible by VAE scale factor %d", vae_scale_factor);
        }
        return false;
    }

    const int latent_w = params->width / vae_scale_factor;
    const int latent_h = params->height / vae_scale_factor;
    const int patch_size = std::max<int>(1, flux_runner_->flux_params.patch_size);
    if (latent_w % patch_size != 0 || latent_h % patch_size != 0) {
        if (error != nullptr) {
            *error = sd_format("Flux latent size %dx%d must be divisible by patch size %d; use image dimensions divisible by %d",
                               latent_w,
                               latent_h,
                               patch_size,
                               vae_scale_factor * patch_size);
        }
        return false;
    }

    ConditionerParams cond_params;
    cond_params.text = params->prompt != nullptr ? params->prompt : "";
    cond_params.clip_skip = -1;
    SDCondition condition = conditioner_->get_learned_condition(n_threads, cond_params);
    if (condition.empty()) {
        if (error != nullptr) {
            *error = "Flux prompt encoding returned empty condition";
        }
        return false;
    }

    const float cfg_scale = params->sample.cfg_scale > 0.0f ? params->sample.cfg_scale : 1.0f;
    SDCondition uncond;
    if (cfg_scale != 1.0f) {
        ConditionerParams uncond_params;
        uncond_params.text = params->negative_prompt != nullptr ? params->negative_prompt : "";
        uncond_params.clip_skip = -1;
        uncond = conditioner_->get_learned_condition(n_threads, uncond_params);
        if (uncond.empty()) {
            if (error != nullptr) {
                *error = "Flux negative prompt encoding returned empty condition";
            }
            return false;
        }
    }
    LOG_INFO("flux prompt encoded: cross_attn=%s vector=%s",
             format_tensor_shape(condition.c_crossattn).c_str(),
             format_tensor_shape(condition.c_vector).c_str());

    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
    float flow_shift = params->sample.flow_shift;
    if (!(flow_shift > 0.0f) || !std::isfinite(flow_shift)) {
        flow_shift = flux_runner_->flux_params.guidance_embed ? 1.15f : 1.0f;
    }
    const float distilled_guidance = params->sample.distilled_guidance != 0.0f
                                         ? params->sample.distilled_guidance
                                         : 3.5f;
    const int64_t seed = params->seed >= 0 ? params->seed : 42;
    std::shared_ptr<RNG> rng = runtime_->rng_ptr();
    if (!rng) {
        if (error != nullptr) {
            *error = "FluxPipeline has no RNG from ModelRuntime";
        }
        return false;
    }
    rng->manual_seed(static_cast<uint64_t>(seed + batch_index));

    sd::Tensor<float> init_latent = sd::zeros<float>({latent_w, latent_h, 16, 1});
    sd::Tensor<float> noise = sd::Tensor<float>::randn(init_latent.shape(), rng);
    std::vector<float> sigmas = ed_flux_discrete_sigmas(steps, flow_shift);
    if (sigmas.size() < 2) {
        if (error != nullptr) {
            *error = "failed to create Flux sigma schedule";
        }
        return false;
    }

    LOG_INFO("flux txt2img: %dx%d latent=%dx%d steps=%d shift=%.2f guidance=%.2f cfg=%.2f seed=%" PRId64,
             params->width,
             params->height,
             latent_w,
             latent_h,
             steps,
             flow_shift,
             distilled_guidance,
             cfg_scale,
             seed + batch_index);

    sd::Tensor<float> x = init_latent * (1.0f - sigmas[0]) + noise * sigmas[0];
    sd::Tensor<float> denoised = x;
    cache::CacheRuntime cache_runtime;
    // The block-stack seam is usable only outside CFG-parallel (skip decisions
    // must stay in lockstep across ranks) and when the runner can cut its stack.
    const bool cfg_parallel_for_cache =
        !uncond.empty() && parallel::cfg_parallel_available(runtime_->parallel_context());
    const bool cache_seam_available =
        !cfg_parallel_for_cache && flux_runner_->feature_cache_available();
    // Wire the device store only when the on-GPU feature-reuse path is active
    // (ED_FEATURE_CACHE_GPU on); with it off, leave the store null so a
    // device_backed slot cleanly falls back to the host declarative path.
    cache::ICacheDeviceStore* cache_store =
        (cache_seam_available && flux_runner_ != nullptr &&
         Flux::FluxRunner::feature_gpu_enabled())
            ? flux_runner_->cache_device_store()
            : nullptr;
    const bool cache_enabled =
        cache_runtime.init(params->sample, version_, sigmas, cache_seam_available, cache_store);
    // GPU DiCache (ED_DICACHE_GPU): reset per-generation persistent state and set
    // the probe depth the capture step uses to snapshot its probe residual. Read
    // the resolved depth from the engine so it stays in sync with the policy's
    // config (the reference default is 1, NOT DBCache's cache_Fn_compute_blocks).
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
        const float sigma = sigmas[static_cast<size_t>(step)];
        const float sigma_next = sigmas[static_cast<size_t>(step + 1)];
        const float c_skip = 1.0f;
        const float c_out = -sigma;

        sd::Tensor<float> timesteps({1}, std::vector<float>{sigma});
        sd::Tensor<float> guidance({1}, std::vector<float>{distilled_guidance});
        sd::Tensor<float> noised_input = x;

        cache::CacheStepInfo cache_step;
        cache_step.step_index = step;
        cache_step.num_steps = steps;
        cache_step.sigma = sigma;
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

        // Build cache hooks for one condition. Feature/Probe methods (which
        // gate to the plain compute path) are disabled under CFG-parallel to
        // keep rank skip-decisions in lockstep; DiCache (Probe) is value-driven
        // so it must never run per-rank.
        auto make_hooks = [&](const SDCondition& cond_in) {
            cache::CacheRunnerHooks hooks;
            hooks.input = &noised_input;
            hooks.full = [&]() {
                return flux_runner_->compute(n_threads, noised_input, timesteps,
                                             cond_in.c_crossattn, {}, cond_in.c_vector, guidance);
            };
            const bool seam_ok = !use_cfg_parallel && flux_runner_->feature_cache_available();
            hooks.feature_supported = seam_ok;
            if (seam_ok) {
                const void* branch_key = static_cast<const void*>(&cond_in);
                // Only DiCache (Probe) uses the on-GPU probe/inject seam that a
                // branch_key drives; passing it into compute_capture for a Feature
                // method (MagCache/TaylorSeer/SenCache) would flip gpu_metric on and
                // suppress the host feature readback those methods rely on.
                const bool is_probe = cache_runtime.granularity() == cache::CacheGranularity::Probe;
                // Feature-granularity on-GPU reuse: keep the captured residual on
                // device and inject it there on skips, avoiding the ~50MB host
                // reconstruct copy + H2D upload the host inject path pays per skip.
                const bool feature_gpu = !is_probe &&
                    cache_runtime.granularity() == cache::CacheGranularity::Feature &&
                    Flux::FluxRunner::feature_gpu_enabled();
                const void* capture_key = is_probe ? branch_key : nullptr;
                if (feature_gpu) {
                    // Declarative device-slot seam (B2): the lowering hands us the
                    // slot's device tensor; capture stores the residual into it and
                    // reuse injects x_before + slot on-device (no host round-trip).
                    hooks.capture_to_slot = [&](const std::function<void*(const std::vector<int64_t>&)>& alloc_slot,
                                                int region_start, int region_end) {
                        return flux_runner_->compute_capture_to_slot(
                            n_threads, noised_input, timesteps, cond_in.c_crossattn, {},
                            cond_in.c_vector, guidance, {}, false,
                            alloc_slot, region_start, region_end);
                    };
                    hooks.inject_from_slot = [&](void* slot, int region_start, int region_end) {
                        return flux_runner_->compute_inject_from_slot(
                            n_threads, noised_input, timesteps, cond_in.c_crossattn, {},
                            cond_in.c_vector, guidance, {}, false,
                            static_cast<ggml_tensor*>(slot), region_start, region_end);
                    };
                } else {
                    hooks.capture = [&, capture_key](int region_start, int region_end) {
                        return flux_runner_->compute_capture(n_threads, noised_input, timesteps,
                                                             cond_in.c_crossattn, {}, cond_in.c_vector,
                                                             guidance, {}, false, region_start, region_end,
                                                             capture_key);
                    };
                }
                hooks.inject = [&](const sd::Tensor<float>& feat, int region_start, int region_end) {
                    return flux_runner_->compute_inject(n_threads, noised_input, timesteps,
                                                        cond_in.c_crossattn, {}, cond_in.c_vector,
                                                        guidance, {}, false, feat, region_start, region_end);
                };
                if (cache_runtime.granularity() == cache::CacheGranularity::Probe) {
                    hooks.probe = [&, branch_key](int depth) {
                        return flux_runner_->compute_probe(n_threads, noised_input, timesteps,
                                                           cond_in.c_crossattn, {}, cond_in.c_vector,
                                                           guidance, {}, false, depth, branch_key);
                    };
                    // Only wire the on-GPU inject when the model's GPU DiCache path
                    // is active. With ED_DICACHE_GPU=0, leaving inject_gpu unset lets
                    // the lowering take the declarative host probe path (the residual
                    // ring blend), instead of the legacy on-device reconstruction.
                    if (Flux::FluxRunner::dicache_gpu_enabled()) {
                        hooks.inject_gpu = [&, branch_key](float gamma, int region_start, int region_end) {
                            return flux_runner_->compute_inject_gpu(n_threads, noised_input, timesteps,
                                                                    cond_in.c_crossattn, {}, cond_in.c_vector,
                                                                    guidance, {}, false, gamma, branch_key,
                                                                    region_start, region_end);
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
                                        local_condition.c_crossattn, {}, local_condition.c_vector, guidance);
            std::vector<sd::Tensor<float>> gathered;
            if (local_out.empty() ||
                !parallel::cfg_all_gather(*runtime_->parallel_context(), local_out, &gathered, error) ||
                gathered.size() != 2) {
                if (error != nullptr && error->empty()) {
                    *error = sd_format("Flux CFG parallel gather failed at step %d", step + 1);
                }
                flux_runner_->free_compute_buffer();
                return false;
            }
            model_out = gathered[0] + cfg_scale * (gathered[1] - gathered[0]);
        } else {
            model_out = cache_enabled
                ? cache_runtime.run_branch(condition_branch, condition_key, make_hooks(condition))
                : flux_runner_->compute(n_threads, noised_input, timesteps,
                                        condition.c_crossattn, {}, condition.c_vector, guidance);
        }
        if (!uncond.empty() && !use_cfg_parallel) {
            const void* uncond_key = static_cast<const void*>(&uncond);
            sd::Tensor<float> uncond_out = cache_enabled
                ? cache_runtime.run_branch(cache::CacheBranch::Uncond, uncond_key, make_hooks(uncond))
                : flux_runner_->compute(n_threads, noised_input, timesteps,
                                        uncond.c_crossattn, {}, uncond.c_vector, guidance);
            if (uncond_out.empty()) {
                if (error != nullptr) {
                    *error = sd_format("Flux unconditional transformer compute failed at step %d", step + 1);
                }
                flux_runner_->free_compute_buffer();
                return false;
            }
            model_out = uncond_out + cfg_scale * (model_out - uncond_out);
        }
        if (model_out.empty()) {
            if (error != nullptr) {
                *error = sd_format("Flux transformer compute failed at step %d", step + 1);
            }
            flux_runner_->free_compute_buffer();
            return false;
        }

        // SenCache calibration: measure finite-difference sensitivities on the
        // CFG-combined velocity. Two extra plain forwards per step, calibration
        // only; the seam (and thus calibration) is off under CFG-parallel. The
        // policy owns the protocol; the pipeline supplies only forward_at.
        if (cache_enabled && cache_runtime.needs_calibration() && !use_cfg_parallel) {
            auto forward_at = [&](const sd::Tensor<float>& x_raw, float sigma_eval) -> sd::Tensor<float> {
                sd::Tensor<float> ts({1}, std::vector<float>{sigma_eval});
                sd::Tensor<float> cond_v = flux_runner_->compute(n_threads, x_raw, ts,
                                                                 condition.c_crossattn, {}, condition.c_vector, guidance);
                if (cond_v.empty() || uncond.empty()) {
                    return cond_v;
                }
                sd::Tensor<float> uncond_v = flux_runner_->compute(n_threads, x_raw, ts,
                                                                   uncond.c_crossattn, {}, uncond.c_vector, guidance);
                if (uncond_v.empty()) {
                    return {};
                }
                return uncond_v + cfg_scale * (cond_v - uncond_v);
            };
            cache_runtime.calibrate(condition_branch, condition_key, x, model_out, forward_at);
        }

        denoised = model_out * c_out + x * c_skip;

        if (sigma == 0.0f) {
            x = denoised;
        } else {
            const sd::Tensor<float> d = (x - denoised) / sigma;
            x += d * (sigma_next - sigma);
        }
        LOG_INFO("flux step %d/%d sigma=%.6f next=%.6f", step + 1, steps, sigma, sigma_next);
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
    const int64_t sample_end_ms = ggml_time_ms();
    LOG_INFO("flux sampling completed, taking %.2fs", (sample_end_ms - sample_start_ms) / 1000.0f);
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
            *error = "Flux VAE decode failed";
        }
        return false;
    }

    const ed_status_t status = ed_tensor_to_image(decoded, image);
    if (status != ED_STATUS_OK) {
        if (error != nullptr) {
            *error = status == ED_STATUS_OUT_OF_MEMORY ? "failed to allocate decoded image" : "decoded Flux tensor has invalid shape";
        }
        return false;
    }
    return true;
}

}  // namespace edgedit
