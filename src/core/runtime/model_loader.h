#ifndef ED_MODEL_LOADER_H
#define ED_MODEL_LOADER_H
#include "edge-dit.h"
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "utils/model_io/tensor_storage.h"
#include "utils/ordered_map.hpp"

enum SDVersion {
    VERSION_SD1,
    VERSION_SD1_INPAINT,
    VERSION_SD1_PIX2PIX,
    VERSION_SD1_TINY_UNET,
    VERSION_SD2,
    VERSION_SD2_INPAINT,
    VERSION_SD2_TINY_UNET,
    VERSION_SDXS_512_DS,
    VERSION_SDXS_09,
    VERSION_SDXL,
    VERSION_SDXL_INPAINT,
    VERSION_SDXL_PIX2PIX,
    VERSION_SDXL_VEGA,
    VERSION_SDXL_SSD1B,
    VERSION_SVD,
    VERSION_SD3,
    VERSION_FLUX,
    VERSION_FLUX_KONTEXT,
    VERSION_FLUX_FILL,
    VERSION_FLUX_CONTROLS,
    VERSION_FLEX_2,
    VERSION_CHROMA_RADIANCE,
    VERSION_WAN2,
    VERSION_WAN2_2_I2V,
    VERSION_WAN2_2_TI2V,
    VERSION_QWEN_IMAGE,
    VERSION_QWEN_IMAGE_EDIT,
    VERSION_MINIMAX_H3,
    VERSION_ANIMA,
    VERSION_FLUX2,
    VERSION_FLUX2_KLEIN,
    VERSION_Z_IMAGE,
    VERSION_OVIS_IMAGE,
    VERSION_ERNIE_IMAGE,
    VERSION_LTXAV,
    VERSION_COUNT,
};

enum PMVersion {
    PM_VERSION_1,
    PM_VERSION_2,
};

static inline bool ed_version_is_sd1(SDVersion version) {
    return version == VERSION_SD1 || version == VERSION_SD1_INPAINT ||
           version == VERSION_SD1_PIX2PIX || version == VERSION_SD1_TINY_UNET ||
           version == VERSION_SDXS_512_DS;
}

static inline bool ed_version_is_sd2(SDVersion version) {
    return version == VERSION_SD2 || version == VERSION_SD2_INPAINT ||
           version == VERSION_SD2_TINY_UNET || version == VERSION_SDXS_09;
}

static inline bool ed_version_is_sdxl(SDVersion version) {
    return version == VERSION_SDXL || version == VERSION_SDXL_INPAINT ||
           version == VERSION_SDXL_PIX2PIX || version == VERSION_SDXL_SSD1B ||
           version == VERSION_SDXL_VEGA;
}

static inline bool ed_version_is_unet(SDVersion version) {
    return ed_version_is_sd1(version) || ed_version_is_sd2(version) || ed_version_is_sdxl(version);
}

static inline bool ed_version_is_sd3(SDVersion version) {
    return version == VERSION_SD3;
}

static inline bool ed_version_is_flux(SDVersion version) {
    return version == VERSION_FLUX || version == VERSION_FLUX_FILL ||
           version == VERSION_FLUX_KONTEXT ||
           version == VERSION_FLUX_CONTROLS || version == VERSION_FLEX_2 ||
           version == VERSION_OVIS_IMAGE || version == VERSION_CHROMA_RADIANCE;
}

static inline bool ed_version_is_flux2(SDVersion version) {
    return version == VERSION_FLUX2 || version == VERSION_FLUX2_KLEIN;
}

static inline bool ed_version_is_wan(SDVersion version) {
    return version == VERSION_WAN2 || version == VERSION_WAN2_2_I2V || version == VERSION_WAN2_2_TI2V;
}

static inline bool ed_version_is_qwen_image(SDVersion version) {
    return version == VERSION_QWEN_IMAGE;
}

static inline bool ed_version_is_qwen_image_edit(SDVersion version) {
    return version == VERSION_QWEN_IMAGE_EDIT;
}

static inline bool ed_version_is_minimax_h3(SDVersion version) {
    return version == VERSION_MINIMAX_H3;
}

static inline bool ed_version_is_anima(SDVersion version) {
    return version == VERSION_ANIMA;
}

static inline bool ed_version_is_z_image(SDVersion version) {
    return version == VERSION_Z_IMAGE;
}

static inline bool ed_version_is_ernie_image(SDVersion version) {
    return version == VERSION_ERNIE_IMAGE;
}

static inline bool ed_version_is_ltxav(SDVersion version) {
    return version == VERSION_LTXAV;
}

static inline bool ed_version_uses_flux2_vae(SDVersion version) {
    return ed_version_is_flux2(version) || ed_version_is_ernie_image(version);
}

static inline bool ed_version_is_inpaint(SDVersion version) {
    return version == VERSION_SD1_INPAINT || version == VERSION_SD2_INPAINT ||
           version == VERSION_SDXL_INPAINT || version == VERSION_FLUX_FILL ||
           version == VERSION_FLEX_2;
}

static inline bool ed_version_is_dit(SDVersion version) {
    return ed_version_is_flux(version) || ed_version_is_flux2(version) ||
           ed_version_is_sd3(version) || ed_version_is_wan(version) ||
           ed_version_is_qwen_image(version) || ed_version_is_qwen_image_edit(version) ||
           ed_version_is_minimax_h3(version) ||
           ed_version_is_anima(version) ||
           ed_version_is_z_image(version) || ed_version_is_ernie_image(version) ||
           ed_version_is_ltxav(version);
}

static inline bool ed_version_is_unet_edit(SDVersion version) {
    return version == VERSION_SD1_PIX2PIX || version == VERSION_SDXL_PIX2PIX;
}

static inline bool ed_version_is_control(SDVersion version) {
    return version == VERSION_FLUX_CONTROLS || version == VERSION_FLEX_2;
}

using String2TensorStorage = OrderedMap<std::string, TensorStorage>;
using TensorTypeRules = std::vector<std::pair<std::string, ggml_type>>;

TensorTypeRules parse_tensor_type_rules(const std::string& tensor_type_rules);
const char* ed_version_name(SDVersion version);
SDVersion ed_version_from_name(const std::string& name);

// Detect a few-step distilled checkpoint's default step count from model/DiT
// file paths, using a reliability ladder: (1) an explicit `Nstep(s)` marker in
// the path wins for any N in [1,64]; (2) `schnell` -> 4; (3) a distill keyword
// (lightning / lightx2v / turbo / hyper / distill) with no step marker defaults
// to 8 and logs a warning. Returns 0 when no distill signal is found. Used only
// when the user did not pass an explicit --steps, so a miss just falls back to
// the base default.
int detect_distilled_default_steps(const std::vector<std::string>& file_paths,
                                   const char* diffusion_model_path);

class ModelLoader final {
public:
    using TensorMap = std::map<std::string, ggml_tensor*>;
    using IgnoreTensorSet = std::set<std::string>;

public:
    ModelLoader() = default;
    ~ModelLoader() = default;

    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    void reset();

public:
    SDVersion version() const { return version_; }
    void set_version_hint(SDVersion version) { version_ = version; }

    bool external_vae_is_invalid() const { return external_vae_is_invalid_; }
    bool use_tae() const { return use_tae_; }
    bool tae_preview_only() const { return tae_preview_only_; }
    bool use_pmid() const { return use_pmid_; }
    bool qwen_image_zero_cond_t() const { return qwen_image_zero_cond_t_; }

    String2TensorStorage& get_tensor_storage_map() {
        return tensor_storage_map_;
    }

    const String2TensorStorage& get_tensor_storage_map() const {
        return tensor_storage_map_;
    }
    const std::vector<std::string>& file_paths() const { return file_paths_; }
    const std::string& get_last_error() const { return last_error_; }

    std::vector<std::string> tensor_names() const;
    int64_t get_params_mem_size(ggml_backend_t backend,
                                ggml_type type = GGML_TYPE_COUNT) const;

public:
    bool load_model_files(const ed_context_params_t& params, std::string* error);
    bool finalize_names_and_version(std::string* error);
    bool apply_dtype_policy(const ed_context_params_t& params, std::string* error);
    bool bind_weights(const TensorMap& tensors,
                      const IgnoreTensorSet& ignore_tensors,
                      int n_threads,
                      bool use_mmap,
                      std::string* error);
    bool bind_weights(int n_threads, bool use_mmap, std::string* error);

    bool init_from_file(const std::string& file_path,
                        const std::string& prefix = "");
    void convert_tensors_name();

    bool init_from_file_and_convert_name(const std::string& file_path,
                                         const std::string& prefix = "",
                                         SDVersion version = VERSION_COUNT);

    bool init_from_gguf_file(const std::string& file_path,
                             const std::string& prefix = "");
    bool init_from_safetensors_file(const std::string& file_path,
                                    const std::string& prefix = "");
    bool init_from_safetensors_index_file(const std::string& file_path,
                                          const std::string& prefix = "");
    bool init_from_diffusers_directory(const std::string& dir_path,
                                       const std::string& prefix = "");

    // For a bare diffusers transformer file/shard-index loaded as the model body
    // (empty prefix): recover version_ from the file name ("flux") or the sibling
    // transformer/config.json, and return the component prefix ("transformer.")
    // that convert_tensor_name needs to canonicalize the DiT to
    // "model.diffusion_model.*". When a caller already passes a prefix (e.g. the
    // "--diffusion-model" path passes "model.diffusion_model."), it is returned
    // unchanged. Shared by the safetensors single-file and shard-index branches.
    std::string resolve_bare_transformer_prefix(const std::string& resolved_path,
                                                const std::string& prefix);

    // Weight binding: the original load_tensors is likewise no longer exposed to the Engine.
    bool load_tensors(on_new_tensor_cb_t on_new_tensor_cb,
                      int n_threads = 0,
                      bool use_mmap = false);

    bool load_tensors(TensorMap& tensors,
                      IgnoreTensorSet ignore_tensors = {},
                      int n_threads = 0,
                      bool use_mmap = false);

    // dtype / version / stats
    SDVersion get_ld_version();

    std::map<ggml_type, uint32_t> get_wtype_stat() const;
    std::map<ggml_type, uint32_t> get_conditioner_wtype_stat() const;
    std::map<ggml_type, uint32_t> get_diffusion_model_wtype_stat() const;
    std::map<ggml_type, uint32_t> get_vae_wtype_stat() const;

    void set_wtype_override(ggml_type wtype,
                            std::string tensor_type_rules = "");

    // Budget-aware re-quantization (auto-fit scheduler). Force every tensor whose
    // name starts with `prefix` (e.g. "model.diffusion_model") toward `dst_type`.
    // Existing tensors below the target precision remain unchanged unless
    // allow_precision_increase is set. Returns the number of tensors changed.
    size_t override_component_wtype(const std::string& prefix,
                                    ggml_type dst_type,
                                    bool allow_precision_increase = false);

    bool tensor_should_be_converted(const TensorStorage& tensor_storage,
                                    ggml_type type) const;

    // imatrix support (offline convert only).
    //
    // Installs a per-input-channel importance map keyed by CANONICAL tensor name
    // (i.e. names already run through convert_tensor_name), so it can be looked up
    // directly against the tensor names seen during load_tensors. When a quantized
    // tensor has a matching entry whose length equals its in_features (ne[0]), that
    // vector is passed to ggml_quantize_chunk instead of the all-ones placeholder,
    // minimizing quantization error on the important (high-activation) channels.
    // The map is empty by default, so on-the-fly runtime quantization is unaffected.
    void set_imatrix_map(std::map<std::string, std::vector<float>> imatrix_map) {
        imatrix_map_ = std::move(imatrix_map);
    }
    const std::map<std::string, std::vector<float>>& get_imatrix_map() const {
        return imatrix_map_;
    }

    void log_weight_stats() const;

private:
    void clear();
    void set_error(const std::string& error);
    void add_tensor_storage(const TensorStorage& tensor_storage);

    bool load_optional_file(const char* path,
                            const std::string& prefix,
                            const char* label,
                            bool required,
                            std::string* error);

    static bool non_empty(const char* path);
    static ggml_type ed_dtype_to_ggml(ed_dtype_t dtype);
    static std::string wtype_stat_to_str(const std::map<ggml_type, uint32_t>& stat);

private:
    SDVersion version_ = VERSION_COUNT;

    std::vector<std::string> file_paths_;
    String2TensorStorage tensor_storage_map_;
    std::string last_error_;

    TensorMap tensors_;
    IgnoreTensorSet ignore_tensors_;

    // Canonical-name -> per-input-channel importance vector (activation-calibrated imatrix).
    // Empty unless set_imatrix_map() was called (offline convert path only).
    std::map<std::string, std::vector<float>> imatrix_map_;

    bool external_vae_is_invalid_ = false;
    bool use_tae_ = false;
    bool tae_preview_only_ = false;
    bool use_pmid_ = false;
    bool skip_t5_ = false;
    bool qwen_image_zero_cond_t_ = false;
};

#endif
