#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include <memory>
#include <optional>
#include <utility>

#include "backend/ggml/tensor_ggml.hpp"
#include "dit_models/models/anima.hpp"
#include "dit_models/models/ernie_image.hpp"
#include "dit_models/models/flux.hpp"
#include "dit_models/models/mmdit.hpp"
#include "dit_models/models/qwen_image.hpp"
#include "dit_models/models/unet.hpp"
#include "dit_models/models/wan.hpp"
#include "dit_models/models/ltxv.hpp"
#include "dit_models/models/z_image.hpp"
#include "optimization/cache/cache_graph_scope.hpp"
#include "parallel/process_group.hpp"

using sd::DiffusionCacheResult;

enum class MiniMaxH3ReferenceKind : int32_t {
    IMAGE,
    VIDEO,
    AUDIO,
    VIDEO_AUDIO,
};

struct MiniMaxH3ReferenceBlock {
    MiniMaxH3ReferenceKind kind = MiniMaxH3ReferenceKind::IMAGE;
    int32_t video_index         = -1;
    int32_t audio_index         = -1;
};

struct DiffusionParams {
    const sd::Tensor<float>* x                        = nullptr;
    const sd::Tensor<float>* timesteps                = nullptr;
    const sd::Tensor<float>* context                  = nullptr;
    const sd::Tensor<float>* c_concat                 = nullptr;
    const sd::Tensor<float>* y                        = nullptr;
    const sd::Tensor<int32_t>* t5_ids                 = nullptr;
    const sd::Tensor<float>* t5_weights               = nullptr;
    const sd::Tensor<float>* guidance                 = nullptr;
    const std::vector<sd::Tensor<float>>* ref_latents = nullptr;
    bool increase_ref_index                           = false;
    int num_video_frames                              = -1;
    const std::vector<sd::Tensor<float>>* controls    = nullptr;
    float control_strength                            = 0.f;
    const sd::Tensor<float>* vace_context             = nullptr;
    float vace_strength                               = 1.f;
    const std::vector<int>* skip_layers               = nullptr;
    const sd::Tensor<int32_t>* minimax_text_token_tags = nullptr;
    const sd::Tensor<int32_t>* minimax_keyframe_indices = nullptr;
    const std::vector<sd::Tensor<float>>* minimax_reference_audio_latents = nullptr;
    const std::vector<MiniMaxH3ReferenceBlock>* minimax_reference_blocks = nullptr;
    int minimax_audio_length = 0;
    float minimax_video_sigma_shift = 12.f;
    float minimax_audio_sigma_shift = 3.f;
    const sd::Tensor<float>* audio_x = nullptr;
    const sd::Tensor<float>* audio_timesteps = nullptr;
    int audio_length = 0;
    float frame_rate = 24.f;
    const sd::Tensor<float>* video_positions = nullptr;
};

template <typename T>
static inline const sd::Tensor<T>& tensor_or_empty(const sd::Tensor<T>* tensor) {
    static const sd::Tensor<T> kEmpty;
    return tensor != nullptr ? *tensor : kEmpty;
}

struct DiffusionModel {
    virtual std::string get_desc()                                               = 0;
    virtual sd::Tensor<float> compute(int n_threads,
                                      const DiffusionParams& diffusion_params)   = 0;
    virtual void alloc_params_buffer()                                           = 0;
    virtual void free_params_buffer()                                            = 0;
    virtual void free_compute_buffer()                                           = 0;
    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_buffer_size()                                      = 0;
    virtual void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter){};
    virtual void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> process_group) = 0;
    virtual int64_t get_adm_in_channels()                            = 0;
    virtual void set_flash_attention_enabled(bool enabled)           = 0;
    virtual void set_max_graph_vram_bytes(size_t max_vram_bytes)     = 0;
    virtual void set_circular_axes(bool circular_x, bool circular_y) = 0;
    // Measure the DiT compute-buffer (activation) footprint at a target latent size for
    // the auto-fit/auto-allocate scheduler. frames is the video frame count (ignored by
    // image models). Default 0 = not implemented -> caller falls back to fixed headroom.
    virtual size_t measure_compute_buffer_at(int /*latent_w*/, int /*latent_h*/, int /*frames*/) { return 0; }

    // ---- Feature/Probe cache support (Layer C runner device helpers) ----
    // A model that can capture/inject the block-stack residual overrides these.
    // The default (UNet, Anima, ...) reports no support so the CacheController
    // falls back to full compute or an Output-level policy.
    virtual bool supports_feature_cache() const { return false; }
    // True when the feature seam can run this configuration (plain compute path
    // only). Models that support feature cache forward this to their runner.
    virtual bool feature_cache_available() const { return false; }

    // Substep-path tap-driven host variants — models without a
    // device slot (Wan). capture: full forward, host residual (ModelOut-ModelIn)
    // via taps; probe: shallow prefix, host before/probe via taps. Default no-op.
    virtual DiffusionCacheResult compute_substep_capture_host(int /*n_threads*/,
                                                              const DiffusionParams& /*params*/) {
        return {};
    }
    virtual DiffusionCacheResult compute_substep_probe_host(int /*n_threads*/,
                                                            const DiffusionParams& /*params*/,
                                                            int /*probe_depth*/) {
        return {};
    }
    // Tap-driven host inject (reuse): x_before + feature over [start,end), region's
    // blocks skipped. Default no-op.
    virtual sd::Tensor<float> compute_substep_inject_host(int /*n_threads*/,
                                                          const DiffusionParams& /*params*/,
                                                          const sd::Tensor<float>& /*feature*/,
                                                          int /*region_start*/ = 0,
                                                          int /*region_end*/ = -1) {
        return {};
    }

    // ---- On-GPU device-slot cache path (MagCache). Models with a device slot
    // (Flux/Qwen/SD3) override these; the default reports no store so the
    // lowering keeps to the host path above. ----
    virtual edgedit::cache::ICacheDeviceStore* cache_device_store() { return nullptr; }
    virtual sd::Tensor<float> compute_substep_capture_slot(int /*n_threads*/,
                                                           const DiffusionParams& /*params*/,
                                                           std::vector<edgedit::cache::GraphExtension> /*exts*/) {
        return {};
    }
    virtual sd::Tensor<float> compute_substep_inject_slot(int /*n_threads*/,
                                                          const DiffusionParams& /*params*/,
                                                          std::vector<edgedit::cache::GraphExtension> /*exts*/) {
        return {};
    }

    // ---- On-GPU DiCache path (Probe granularity). Models with device DiCache
    // (Flux/Qwen/SD3) override these; the default no-ops so the lowering keeps to
    // the host DiCache path (substep_*_host). ----
    virtual sd::Tensor<float> compute_substep_capture_probe(int /*n_threads*/,
                                                            const DiffusionParams& /*params*/,
                                                            int /*probe_depth*/,
                                                            const edgedit::cache::DiCacheSlotBridge& /*bridge*/) {
        return {};
    }
    virtual DiffusionCacheResult compute_substep_probe_gpu(int /*n_threads*/,
                                                           const DiffusionParams& /*params*/,
                                                           int /*probe_depth*/,
                                                           const void* /*branch_key*/,
                                                           bool /*delta_minus*/,
                                                           const edgedit::cache::CacheOperatorRegistry& /*operators*/,
                                                           const edgedit::cache::DiCacheSlotBridge& /*bridge*/) {
        return {};
    }
    virtual sd::Tensor<float> compute_substep_inject_gpu(int /*n_threads*/,
                                                         const DiffusionParams& /*params*/,
                                                         std::vector<edgedit::cache::GraphExtension> /*exts*/,
                                                         const edgedit::cache::DiCacheSlotBridge& /*bridge*/) {
        return {};
    }
};

struct UNetModel : public DiffusionModel {
    UNetModelRunner unet;

    UNetModel(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2TensorStorage& tensor_storage_map = {},
              SDVersion version                              = VERSION_SD1)
        : unet(backend, offload_params_to_cpu, tensor_storage_map, "model.diffusion_model", version) {
    }

    std::string get_desc() override {
        return unet.get_desc();
    }

    void alloc_params_buffer() override {
        unet.alloc_params_buffer();
    }

    void free_params_buffer() override {
        unet.free_params_buffer();
    }

    void free_compute_buffer() override {
        unet.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return unet.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        unet.set_weight_adapter(adapter);
    }

    void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> process_group) override {
        unet.set_process_group(std::move(process_group));
    }

    int64_t get_adm_in_channels() override {
        return unet.unet.adm_in_channels;
    }

    void set_flash_attention_enabled(bool enabled) {
        unet.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        unet.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        unet.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_controls;
        return unet.compute(n_threads,
                            *diffusion_params.x,
                            *diffusion_params.timesteps,
                            tensor_or_empty(diffusion_params.context),
                            tensor_or_empty(diffusion_params.c_concat),
                            tensor_or_empty(diffusion_params.y),
                            diffusion_params.num_video_frames,
                            diffusion_params.controls ? *diffusion_params.controls : empty_controls,
                            diffusion_params.control_strength);
    }
};

struct MMDiTModel : public DiffusionModel {
    MMDiTRunner mmdit;

    MMDiTModel(ggml_backend_t backend,
               bool offload_params_to_cpu,
               const String2TensorStorage& tensor_storage_map = {})
        : mmdit(backend, offload_params_to_cpu, tensor_storage_map, "model.diffusion_model") {
    }

    std::string get_desc() override {
        return mmdit.get_desc();
    }

    size_t measure_compute_buffer_at(int latent_w, int latent_h, int /*frames*/) override {
        return mmdit.measure_compute_buffer_at(latent_w, latent_h);
    }

    void alloc_params_buffer() override {
        mmdit.alloc_params_buffer();
    }

    void free_params_buffer() override {
        mmdit.free_params_buffer();
    }

    void free_compute_buffer() override {
        mmdit.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        mmdit.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return mmdit.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        mmdit.set_weight_adapter(adapter);
    }

    void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> process_group) override {
        mmdit.set_process_group(std::move(process_group));
    }

    int64_t get_adm_in_channels() override {
        return 768 + 1280;
    }

    void set_flash_attention_enabled(bool enabled) {
        mmdit.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        mmdit.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        mmdit.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<int> empty_skip_layers;
        return mmdit.compute(n_threads,
                             *diffusion_params.x,
                             *diffusion_params.timesteps,
                             tensor_or_empty(diffusion_params.context),
                             tensor_or_empty(diffusion_params.y),
                             diffusion_params.skip_layers ? *diffusion_params.skip_layers : empty_skip_layers);
    }

    bool supports_feature_cache() const override { return true; }
    bool feature_cache_available() const override { return mmdit.feature_cache_available(); }

    DiffusionCacheResult compute_substep_capture_host(int n_threads, const DiffusionParams& p) override {
        return mmdit.compute_substep_capture(n_threads, *p.x, *p.timesteps,
                                             tensor_or_empty(p.context), tensor_or_empty(p.y));
    }
    DiffusionCacheResult compute_substep_probe_host(int n_threads, const DiffusionParams& p, int probe_depth) override {
        return mmdit.compute_substep_probe(n_threads, *p.x, *p.timesteps,
                                           tensor_or_empty(p.context), tensor_or_empty(p.y), probe_depth);
    }
    sd::Tensor<float> compute_substep_inject_host(int n_threads, const DiffusionParams& p,
                                                  const sd::Tensor<float>& feature,
                                                  int region_start = 0, int region_end = -1) override {
        return mmdit.compute_substep_inject(n_threads, *p.x, *p.timesteps,
                                            tensor_or_empty(p.context), tensor_or_empty(p.y),
                                            feature, region_start, region_end);
    }
    edgedit::cache::ICacheDeviceStore* cache_device_store() override {
        return mmdit.cache_device_store();
    }
    sd::Tensor<float> compute_substep_capture_slot(int n_threads, const DiffusionParams& p,
                                                   std::vector<edgedit::cache::GraphExtension> exts) override {
        return mmdit.compute_substep_capture_slot(n_threads, *p.x, *p.timesteps,
                                                  tensor_or_empty(p.context), tensor_or_empty(p.y),
                                                  std::move(exts));
    }
    sd::Tensor<float> compute_substep_inject_slot(int n_threads, const DiffusionParams& p,
                                                  std::vector<edgedit::cache::GraphExtension> exts) override {
        return mmdit.compute_substep_inject_slot(n_threads, *p.x, *p.timesteps,
                                                 tensor_or_empty(p.context), tensor_or_empty(p.y),
                                                 std::move(exts));
    }
    sd::Tensor<float> compute_substep_capture_probe(int n_threads, const DiffusionParams& p,
                                                    int probe_depth,
                                                    const edgedit::cache::DiCacheSlotBridge& bridge) override {
        return mmdit.compute_substep_capture_probe(n_threads, *p.x, *p.timesteps,
                                                   tensor_or_empty(p.context), tensor_or_empty(p.y),
                                                   probe_depth, bridge);
    }
    DiffusionCacheResult compute_substep_probe_gpu(int n_threads, const DiffusionParams& p,
                                                   int probe_depth, const void* branch_key, bool delta_minus,
                                                   const edgedit::cache::CacheOperatorRegistry& operators,
                                                   const edgedit::cache::DiCacheSlotBridge& bridge) override {
        return mmdit.compute_substep_probe(n_threads, *p.x, *p.timesteps,
                                           tensor_or_empty(p.context), tensor_or_empty(p.y),
                                           probe_depth, branch_key, delta_minus, operators, bridge);
    }
    sd::Tensor<float> compute_substep_inject_gpu(int n_threads, const DiffusionParams& p,
                                                 std::vector<edgedit::cache::GraphExtension> exts,
                                                 const edgedit::cache::DiCacheSlotBridge& bridge) override {
        return mmdit.compute_substep_inject_gpu(n_threads, *p.x, *p.timesteps,
                                                tensor_or_empty(p.context), tensor_or_empty(p.y),
                                                std::move(exts), bridge);
    }
};

struct FluxModel : public DiffusionModel {
    Flux::FluxRunner flux;

    FluxModel(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2TensorStorage& tensor_storage_map = {},
              SDVersion version                              = VERSION_FLUX,
              bool use_mask                                  = false)
        : flux(backend, offload_params_to_cpu, tensor_storage_map, "model.diffusion_model", version, use_mask) {
    }

    std::string get_desc() override {
        return flux.get_desc();
    }

    void alloc_params_buffer() override {
        flux.alloc_params_buffer();
    }

    void free_params_buffer() override {
        flux.free_params_buffer();
    }

    void free_compute_buffer() override {
        flux.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        flux.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return flux.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        flux.set_weight_adapter(adapter);
    }

    void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> process_group) override {
        flux.set_process_group(std::move(process_group));
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        flux.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        flux.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        flux.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        static const std::vector<int> empty_skip_layers;
        return flux.compute(n_threads,
                            *diffusion_params.x,
                            *diffusion_params.timesteps,
                            tensor_or_empty(diffusion_params.context),
                            tensor_or_empty(diffusion_params.c_concat),
                            tensor_or_empty(diffusion_params.y),
                            tensor_or_empty(diffusion_params.guidance),
                            diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                            diffusion_params.increase_ref_index,
                            diffusion_params.skip_layers ? *diffusion_params.skip_layers : empty_skip_layers);
    }

    bool supports_feature_cache() const override { return true; }
    bool feature_cache_available() const override { return flux.feature_cache_available(); }
};

struct AnimaModel : public DiffusionModel {
    std::string prefix;
    Anima::AnimaRunner anima;

    AnimaModel(ggml_backend_t backend,
               bool offload_params_to_cpu,
               const String2TensorStorage& tensor_storage_map = {},
               const std::string prefix                       = "model.diffusion_model")
        : prefix(prefix), anima(backend, offload_params_to_cpu, tensor_storage_map, prefix) {
    }

    std::string get_desc() override {
        return anima.get_desc();
    }

    void alloc_params_buffer() override {
        anima.alloc_params_buffer();
    }

    void free_params_buffer() override {
        anima.free_params_buffer();
    }

    void free_compute_buffer() override {
        anima.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        anima.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return anima.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        anima.set_weight_adapter(adapter);
    }

    void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> process_group) override {
        anima.set_process_group(std::move(process_group));
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        anima.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        anima.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        anima.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        return anima.compute(n_threads,
                             *diffusion_params.x,
                             *diffusion_params.timesteps,
                             tensor_or_empty(diffusion_params.context),
                             tensor_or_empty(diffusion_params.t5_ids),
                             tensor_or_empty(diffusion_params.t5_weights));
    }
};

struct WanModel : public DiffusionModel {
    std::string prefix;
    WAN::WanRunner wan;

    WanModel(ggml_backend_t backend,
             bool offload_params_to_cpu,
             const String2TensorStorage& tensor_storage_map = {},
             const std::string prefix                       = "model.diffusion_model",
             SDVersion version                              = VERSION_WAN2)
        : prefix(prefix), wan(backend, offload_params_to_cpu, tensor_storage_map, prefix, version) {
    }

    std::string get_desc() override {
        return wan.get_desc();
    }

    size_t measure_compute_buffer_at(int latent_w, int latent_h, int frames) override {
        return wan.measure_compute_buffer_at(latent_w, latent_h, frames);
    }

    int64_t num_heads() const {
        return wan.num_heads();
    }

    int64_t head_dim() const {
        return wan.head_dim();
    }

    void alloc_params_buffer() override {
        wan.alloc_params_buffer();
    }

    void free_params_buffer() override {
        wan.free_params_buffer();
    }

    void free_compute_buffer() override {
        wan.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        wan.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return wan.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        wan.set_weight_adapter(adapter);
    }

    void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> process_group) override {
        wan.set_process_group(std::move(process_group));
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        wan.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        wan.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        wan.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        return wan.compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context),
                           tensor_or_empty(diffusion_params.y),
                           tensor_or_empty(diffusion_params.c_concat),
                           sd::Tensor<float>(),
                           tensor_or_empty(diffusion_params.vace_context),
                           diffusion_params.vace_strength);
    }

    bool supports_feature_cache() const override { return true; }
    bool feature_cache_available() const override { return wan.feature_cache_available(); }

    DiffusionCacheResult compute_substep_capture_host(int n_threads, const DiffusionParams& p) override {
        return wan.compute_substep_capture(n_threads, *p.x, *p.timesteps, tensor_or_empty(p.context),
                                           tensor_or_empty(p.y), tensor_or_empty(p.c_concat));
    }
    DiffusionCacheResult compute_substep_probe_host(int n_threads, const DiffusionParams& p, int probe_depth) override {
        return wan.compute_substep_probe(n_threads, *p.x, *p.timesteps, tensor_or_empty(p.context),
                                         tensor_or_empty(p.y), tensor_or_empty(p.c_concat), probe_depth);
    }
    sd::Tensor<float> compute_substep_inject_host(int n_threads, const DiffusionParams& p,
                                                  const sd::Tensor<float>& feature,
                                                  int region_start = 0, int region_end = -1) override {
        return wan.compute_substep_inject(n_threads, *p.x, *p.timesteps, tensor_or_empty(p.context),
                                          tensor_or_empty(p.y), tensor_or_empty(p.c_concat),
                                          feature, region_start, region_end);
    }
    edgedit::cache::ICacheDeviceStore* cache_device_store() override {
        return wan.cache_device_store();
    }
    sd::Tensor<float> compute_substep_capture_slot(int n_threads, const DiffusionParams& p,
                                                   std::vector<edgedit::cache::GraphExtension> exts) override {
        return wan.compute_substep_capture_slot(n_threads, *p.x, *p.timesteps, tensor_or_empty(p.context),
                                                tensor_or_empty(p.y), tensor_or_empty(p.c_concat),
                                                std::move(exts));
    }
    sd::Tensor<float> compute_substep_inject_slot(int n_threads, const DiffusionParams& p,
                                                  std::vector<edgedit::cache::GraphExtension> exts) override {
        return wan.compute_substep_inject_slot(n_threads, *p.x, *p.timesteps, tensor_or_empty(p.context),
                                               tensor_or_empty(p.y), tensor_or_empty(p.c_concat),
                                               std::move(exts));
    }
    sd::Tensor<float> compute_substep_capture_probe(int n_threads, const DiffusionParams& p,
                                                    int probe_depth,
                                                    const edgedit::cache::DiCacheSlotBridge& bridge) override {
        return wan.compute_substep_capture_probe(n_threads, *p.x, *p.timesteps, tensor_or_empty(p.context),
                                                 tensor_or_empty(p.y), tensor_or_empty(p.c_concat),
                                                 probe_depth, bridge);
    }
    DiffusionCacheResult compute_substep_probe_gpu(int n_threads, const DiffusionParams& p,
                                                   int probe_depth, const void* branch_key, bool delta_minus,
                                                   const edgedit::cache::CacheOperatorRegistry& operators,
                                                   const edgedit::cache::DiCacheSlotBridge& bridge) override {
        return wan.compute_substep_probe(n_threads, *p.x, *p.timesteps, tensor_or_empty(p.context),
                                         tensor_or_empty(p.y), tensor_or_empty(p.c_concat),
                                         probe_depth, branch_key, delta_minus, operators, bridge);
    }
    sd::Tensor<float> compute_substep_inject_gpu(int n_threads, const DiffusionParams& p,
                                                 std::vector<edgedit::cache::GraphExtension> exts,
                                                 const edgedit::cache::DiCacheSlotBridge& bridge) override {
        return wan.compute_substep_inject_gpu(n_threads, *p.x, *p.timesteps, tensor_or_empty(p.context),
                                              tensor_or_empty(p.y), tensor_or_empty(p.c_concat),
                                              std::move(exts), bridge);
    }
};

struct LTXAVModel : public DiffusionModel {
    std::string prefix;
    LTXV::LTXAVRunner ltx;

    LTXAVModel(ggml_backend_t backend,
               bool offload_params_to_cpu,
               const String2TensorStorage& tensor_storage_map = {},
               const std::string& model_prefix = "model.diffusion_model")
        : prefix(model_prefix), ltx(backend, offload_params_to_cpu, tensor_storage_map, model_prefix) {}

    std::string get_desc() override { return ltx.get_desc(); }
    void alloc_params_buffer() override { ltx.alloc_params_buffer(); }
    void free_params_buffer() override { ltx.free_params_buffer(); }
    void free_compute_buffer() override { ltx.free_compute_buffer(); }
    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override { ltx.get_param_tensors(tensors, prefix); }
    size_t get_params_buffer_size() override { return ltx.get_params_buffer_size(); }
    bool stage_params_for_phase() { return ltx.stage_params_for_phase(); }
    void release_params_after_phase() { ltx.release_params_after_phase(); }
    void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> group) override { ltx.set_process_group(std::move(group)); }
    int64_t get_adm_in_channels() override { return 0; }
    void set_flash_attention_enabled(bool enabled) override { ltx.set_flash_attention_enabled(enabled); }
    void set_max_graph_vram_bytes(size_t bytes) override { ltx.set_max_graph_vram_bytes(bytes); }
    void set_circular_axes(bool, bool) override {}
    size_t measure_compute_buffer_at(int latent_w,
                                     int latent_h,
                                     int frames,
                                     int audio_length,
                                     int context_tokens = 256,
                                     bool conditioned = true) {
        return ltx.measure_compute_buffer_at(latent_w,
                                             latent_h,
                                             frames,
                                             audio_length,
                                             context_tokens,
                                             conditioned);
    }

    sd::Tensor<float> compute(int n_threads, const DiffusionParams& p) override {
        GGML_ASSERT(p.x != nullptr && p.timesteps != nullptr);
        return ltx.compute(n_threads,
                           *p.x,
                           *p.timesteps,
                           tensor_or_empty(p.context),
                           tensor_or_empty(p.audio_x),
                           tensor_or_empty(p.audio_timesteps),
                           p.audio_length,
                           p.frame_rate,
                           tensor_or_empty(p.video_positions));
    }
};

struct QwenImageModel : public DiffusionModel {
    std::string prefix;
    Qwen::QwenImageRunner qwen_image;

    QwenImageModel(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "model.diffusion_model",
                   SDVersion version                              = VERSION_QWEN_IMAGE,
                   bool zero_cond_t                               = false)
        : prefix(prefix), qwen_image(backend, offload_params_to_cpu, tensor_storage_map, prefix, version, zero_cond_t) {
    }

    std::string get_desc() override {
        return qwen_image.get_desc();
    }

    size_t measure_compute_buffer_at(int latent_w, int latent_h, int /*frames*/) override {
        return qwen_image.measure_compute_buffer_at(latent_w, latent_h);
    }

    void alloc_params_buffer() override {
        qwen_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        qwen_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        qwen_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        qwen_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return qwen_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        qwen_image.set_weight_adapter(adapter);
    }

    void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> process_group) override {
        qwen_image.set_process_group(std::move(process_group));
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        qwen_image.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        qwen_image.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        qwen_image.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        return qwen_image.compute(n_threads,
                                  *diffusion_params.x,
                                  *diffusion_params.timesteps,
                                  tensor_or_empty(diffusion_params.context),
                                  diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                                  true);
    }

    bool supports_feature_cache() const override { return true; }
    bool feature_cache_available() const override { return qwen_image.feature_cache_available(); }
};

struct ZImageModel : public DiffusionModel {
    std::string prefix;
    ZImage::ZImageRunner z_image;

    ZImageModel(ggml_backend_t backend,
                bool offload_params_to_cpu,
                const String2TensorStorage& tensor_storage_map = {},
                const std::string prefix                       = "model.diffusion_model",
                SDVersion version                              = VERSION_Z_IMAGE)
        : prefix(prefix), z_image(backend, offload_params_to_cpu, tensor_storage_map, prefix, version) {
    }

    std::string get_desc() override {
        return z_image.get_desc();
    }

    void alloc_params_buffer() override {
        z_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        z_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        z_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        z_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return z_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        z_image.set_weight_adapter(adapter);
    }

    void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> process_group) override {
        z_image.set_process_group(std::move(process_group));
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        z_image.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        z_image.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        z_image.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        return z_image.compute(n_threads,
                               *diffusion_params.x,
                               *diffusion_params.timesteps,
                               tensor_or_empty(diffusion_params.context),
                               diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                               true);
    }
};

struct ErnieImageModel : public DiffusionModel {
    std::string prefix;
    ErnieImage::ErnieImageRunner ernie_image;

    ErnieImageModel(ggml_backend_t backend,
                    bool offload_params_to_cpu,
                    const String2TensorStorage& tensor_storage_map = {},
                    const std::string prefix                       = "model.diffusion_model")
        : prefix(prefix), ernie_image(backend, offload_params_to_cpu, tensor_storage_map, prefix) {
    }

    std::string get_desc() override {
        return ernie_image.get_desc();
    }

    void alloc_params_buffer() override {
        ernie_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        ernie_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        ernie_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        ernie_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return ernie_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        ernie_image.set_weight_adapter(adapter);
    }

    void set_process_group(std::shared_ptr<edgedit::parallel::ProcessGroup> process_group) override {
        ernie_image.set_process_group(std::move(process_group));
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        ernie_image.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        ernie_image.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        ernie_image.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        return ernie_image.compute(n_threads,
                                   *diffusion_params.x,
                                   *diffusion_params.timesteps,
                                   tensor_or_empty(diffusion_params.context));
    }
};

#endif
