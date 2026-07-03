#include "runtime/model_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "utils/rng_philox.hpp"
#include "utils/util.h"

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

bool device_name_matches(ggml_backend_dev_t dev, const std::string& requested) {
    if (dev == nullptr) {
        return false;
    }

    const std::string request = lowercase(requested);
    if (request == "gpu") {
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    }

    const char* name = ggml_backend_dev_name(dev);
    return name != nullptr && contains(lowercase(name), request);
}

bool is_generic_gpu_request(const std::string& requested) {
    const std::string request = lowercase(requested);
    return request == "gpu" || request == "cuda";
}

bool is_gpu_device(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return false;
    }
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
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

bool ModelRuntime::init_threads(const ed_context_params_t& params, std::string* error) {
    (void)error;
    n_threads_ = params.n_threads;
    if (n_threads_ <= 0) {
        n_threads_ = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }
    return true;
}

bool ModelRuntime::init_flags(const ed_context_params_t& params, std::string* error) {
    (void)error;
    use_mmap_ = params.use_mmap;
    offload_params_to_cpu_ = params.offload_params_to_cpu;
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
    const int gpu_device_ordinal = parallel_enabled() ? parallel_context_->local_rank() : 0;
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

    backends_.clip_backend = backends_.backend;
    if (params.keep_text_encoder_on_cpu && !ggml_backend_is_cpu(backends_.backend)) {
        backends_.clip_backend = ggml_backend_cpu_init();
        if (backends_.clip_backend == nullptr) {
            return fail(error, "failed to initialize CPU backend for text encoder");
        }
        backends_.clip_owns_backend = true;
        LOG_INFO("text encoder backend: CPU");
    }

    backends_.vae_backend = backends_.backend;
    if (params.keep_vae_on_cpu && !ggml_backend_is_cpu(backends_.backend)) {
        backends_.vae_backend = ggml_backend_cpu_init();
        if (backends_.vae_backend == nullptr) {
            return fail(error, "failed to initialize CPU backend for VAE");
        }
        backends_.vae_owns_backend = true;
        LOG_INFO("VAE backend: CPU");
    }

    backends_.control_net_backend = backends_.backend;
    if (params.keep_control_net_on_cpu && !ggml_backend_is_cpu(backends_.backend)) {
        backends_.control_net_backend = ggml_backend_cpu_init();
        if (backends_.control_net_backend == nullptr) {
            return fail(error, "failed to initialize CPU backend for ControlNet");
        }
        backends_.control_net_owns_backend = true;
        LOG_INFO("ControlNet backend: CPU");
    }

    return true;
}

void ModelRuntime::reset() {
    ready_ = false;
    rng_.reset();
    sampler_rng_.reset();
    release_backends();

    n_threads_ = 0;
    use_mmap_ = false;
    offload_params_to_cpu_ = false;
    free_params_immediately_ = false;
    max_vram_ = 0.0f;
    max_graph_vram_bytes_ = 0;
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
