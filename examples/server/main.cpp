#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "httplib.h"
#include "routes.h"
#include "runtime.h"

namespace {

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --model <model-or-diffusers-dir> [options]\n"
        "  %s --diffusion-model <path> --vae <path> --clip_l <path> --t5xxl <path> [options]\n\n"
        "  %s --diffusion-model <path> --vae <path> [--audio-vae <path>] --llm <path> [options]\n\n"
        "Server options:\n"
        "  --host <ip>               Listen host, default: 127.0.0.1\n"
        "  --port <int>              Listen port, default: 8080\n"
        "  --verbose                 Print request logs\n\n"
        "Context options:\n"
        "  --model <path>            Model or Diffusers directory\n"
        "  --diffusion-model <path>  Standalone DiT transformer weights\n"
        "  --vae <path>              Standalone VAE weights\n"
        "  --audio-vae <path>        MiniMax-H3/LTX-2.3 audio VAE weights\n"
        "  --embeddings-connectors <path>  LTX-2.3 text connector weights\n"
        "  --latent-upscaler <path>  LTX-2.3 spatial latent x2 upscaler weights\n"
        "  --clip_l <path>           CLIP-L text encoder weights\n"
        "  --clip_g <path>           CLIP-G text encoder weights\n"
        "  --t5xxl <path>            T5XXL text encoder weights\n"
        "  --llm <path>              MiniMax-H3/Qwen/LTX-2.3 text encoder weights\n"
        "  --llm-vision <path>       Optional vision-language encoder weights\n"
        "  --backend <name>          Backend: auto, cpu, cuda, vulkan, metal, gpu. Default: auto\n"
        "  --gpu                     Alias for --backend gpu\n"
        "  -t, --threads <int>       Thread count, default: auto\n"
        "  --type <dtype>            Load/on-the-fly quantization type\n"
        "  --tensor-type-rules <csv> Per-tensor mixed quantization rules\n"
        "  --offload-to-cpu          Keep all weights on CPU and stage for compute\n"
        "  --dit-offload             Stage DiT weights from CPU\n"
        "  --text-encoder-offload    Stage text encoder weights from CPU\n"
        "  --vae-offload             Stage VAE weights from CPU\n"
        "  --minimax-h3-stage-lifecycle  Release Qwen/VAE between MiniMax-H3 phases\n"
        "  --auto-allocate           Automatically place components within budget (default)\n"
        "  --no-auto-allocate        Disable automatic placement; use explicit offload flags only\n"
        "  --auto-fit                Automatically select quantization and placement\n"
        "  --max-vram <GB>           VRAM planning budget\n"
        "  --no-t5                   Skip T5 where supported\n"
        "  --no-flash-attention      Disable flash attention\n\n"
        "Default generation options:\n"
        "  -W, --width <int>         Default width, default: 1024\n"
        "  -H, --height <int>        Default height, default: 1024\n"
        "  --steps <int>             Default steps, default: 20\n"
        "  --frames <int>            Default video frames, default: 90\n"
        "                            MiniMax-H3 requires >=22 frames satisfying 17k+5\n"
        "  --fps <int>               Video response fps metadata, default: 24\n"
        "  -s, --seed <int64>        Default seed, default: -1\n"
        "  --guidance <float>        Default distilled guidance, default: 3.5\n"
        "  --cfg-scale <float>       Default CFG scale, default: 1.0\n"
        "  --flow-shift <float>      Default flow scheduler shift, default: model default\n"
        "  --cache <mode>            Default cache mode: off, easycache, ucache, dbcache, taylorseer, cache-dit\n"
        "  --help                    Show this help\n",
        prog, prog, prog);
}

bool parse_dtype(const char* text, ed_dtype_t* out) {
    const std::string v=text?text:"";
    if(v=="auto"||v=="preserve") *out=ED_DTYPE_AUTO; else if(v=="f32")*out=ED_DTYPE_F32; else if(v=="f16")*out=ED_DTYPE_F16; else if(v=="bf16")*out=ED_DTYPE_BF16; else if(v=="q4_0")*out=ED_DTYPE_Q4_0; else if(v=="q5_0")*out=ED_DTYPE_Q5_0; else if(v=="q8_0")*out=ED_DTYPE_Q8_0; else if(v=="q2_k")*out=ED_DTYPE_Q2_K; else if(v=="q3_k")*out=ED_DTYPE_Q3_K; else if(v=="q4_k")*out=ED_DTYPE_Q4_K; else if(v=="q5_k")*out=ED_DTYPE_Q5_K; else if(v=="q6_k")*out=ED_DTYPE_Q6_K; else return false; return true;
}

int parse_int(const char* text, int fallback) {
    if (text == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    return end != text ? static_cast<int>(value) : fallback;
}

int64_t parse_i64(const char* text, int64_t fallback) {
    if (text == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    return end != text ? static_cast<int64_t>(value) : fallback;
}

float parse_float(const char* text, float fallback) {
    if (text == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    const float value = std::strtof(text, &end);
    return end != text ? value : fallback;
}

bool has_text(const char* text) {
    return text != nullptr && text[0] != '\0';
}

std::string display_model_path(const ed_context_params_t& params) {
    if (has_text(params.model_path)) {
        return params.model_path;
    }
    if (has_text(params.diffusion_model_path)) {
        return params.diffusion_model_path;
    }
    return "";
}

struct Args {
    EdgeDitServerParams server;
    ed_context_params_t context = {};
    EdgeDitDefaultGenerationParams defaults;
    const char* backend = nullptr;
};

bool parse_args(int argc, char** argv, Args* args) {
    if (args == nullptr) {
        return false;
    }

    ed_context_params_init(&args->context);
    args->context.auto_allocate = true;

    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(key, "--host") == 0 || std::strcmp(key, "--listen") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->server.host = value;
        } else if (std::strcmp(key, "--port") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->server.port = parse_int(value, args->server.port);
        } else if (std::strcmp(key, "--verbose") == 0) {
            args->server.verbose = true;
        } else if (std::strcmp(key, "--model") == 0) {
            args->context.model_path = require_value(key);
        } else if (std::strcmp(key, "--diffusion-model") == 0) {
            args->context.diffusion_model_path = require_value(key);
        } else if (std::strcmp(key, "--vae") == 0) {
            args->context.vae_path = require_value(key);
        } else if (std::strcmp(key, "--audio-vae") == 0) {
            args->context.audio_vae_path = require_value(key);
        } else if (std::strcmp(key, "--embeddings-connectors") == 0) {
            args->context.embeddings_connectors_path = require_value(key);
        } else if (std::strcmp(key, "--latent-upscaler") == 0) {
            args->context.latent_upscaler_path = require_value(key);
        } else if (std::strcmp(key, "--clip_l") == 0) {
            args->context.clip_l_path = require_value(key);
        } else if (std::strcmp(key, "--clip_g") == 0) {
            args->context.clip_g_path = require_value(key);
        } else if (std::strcmp(key, "--t5xxl") == 0) {
            args->context.t5xxl_path = require_value(key);
        } else if (std::strcmp(key, "--llm") == 0) {
            args->context.llm_path = require_value(key);
        } else if (std::strcmp(key, "--llm-vision") == 0) {
            args->context.llm_vision_path = require_value(key);
        } else if (std::strcmp(key, "--backend") == 0) {
            args->backend = require_value(key);
        } else if (std::strcmp(key, "--gpu") == 0) {
            args->backend = "gpu";
        } else if (std::strcmp(key, "--threads") == 0 || std::strcmp(key, "-t") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->context.n_threads = parse_int(value, args->context.n_threads);
        } else if (std::strcmp(key, "--type") == 0 || std::strcmp(key, "--weight-type") == 0) {
            const char* value=require_value(key); if(value==nullptr||!parse_dtype(value,&args->context.weight_type)){std::fprintf(stderr,"unsupported dtype: %s\n",value?value:"");return false;}
        } else if (std::strcmp(key, "--tensor-type-rules") == 0) {
            args->context.tensor_type_rules=require_value(key);
        } else if (std::strcmp(key, "--offload-to-cpu") == 0) { args->context.offload_params_to_cpu=true;
        } else if (std::strcmp(key, "--dit-offload") == 0) { args->context.dit_offload=true;
        } else if (std::strcmp(key, "--text-encoder-offload") == 0) { args->context.text_encoder_offload=true;
        } else if (std::strcmp(key, "--vae-offload") == 0) { args->context.vae_offload=true;
        } else if (std::strcmp(key, "--minimax-h3-stage-lifecycle") == 0) { args->context.minimax_h3_stage_lifecycle=true;
        } else if (std::strcmp(key, "--auto-allocate") == 0) { args->context.auto_allocate=true;
        } else if (std::strcmp(key, "--no-auto-allocate") == 0) { args->context.auto_allocate=false;
        } else if (std::strcmp(key, "--auto-fit") == 0) { args->context.auto_fit=true;
        } else if (std::strcmp(key, "--max-vram") == 0) { const char* value=require_value(key); if(!value)return false; args->context.max_vram_gb=parse_float(value,args->context.max_vram_gb);
        } else if (std::strcmp(key, "--no-t5") == 0) { args->context.skip_t5=true;
        } else if (std::strcmp(key, "--flash-attention") == 0) { args->context.flash_attention=true;
        } else if (std::strcmp(key, "--no-flash-attention") == 0) { args->context.flash_attention=false;
        } else if (std::strcmp(key, "--width") == 0 || std::strcmp(key, "-W") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.width = parse_int(value, args->defaults.width);
        } else if (std::strcmp(key, "--height") == 0 || std::strcmp(key, "-H") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.height = parse_int(value, args->defaults.height);
        } else if (std::strcmp(key, "--steps") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.steps = parse_int(value, args->defaults.steps);
        } else if (std::strcmp(key, "--frames") == 0 || std::strcmp(key, "--video-frames") == 0) {
            const char* value=require_value(key); if(!value)return false; args->defaults.frames=parse_int(value,args->defaults.frames);
        } else if (std::strcmp(key, "--fps") == 0) {
            const char* value=require_value(key); if(!value)return false; args->defaults.fps=parse_int(value,args->defaults.fps);
        } else if (std::strcmp(key, "--seed") == 0 || std::strcmp(key, "-s") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.seed = parse_i64(value, args->defaults.seed);
        } else if (std::strcmp(key, "--guidance") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.distilled_guidance = parse_float(value, args->defaults.distilled_guidance);
        } else if (std::strcmp(key, "--cfg-scale") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.cfg_scale = parse_float(value, args->defaults.cfg_scale);
        } else if (std::strcmp(key, "--flow-shift") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.flow_shift = parse_float(value, args->defaults.flow_shift);
        } else if (std::strcmp(key, "--cache") == 0 || std::strcmp(key, "--cache-mode") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            if (!ed_cache_mode_from_string(value, &args->defaults.cache_mode)) {
                std::fprintf(stderr, "unsupported cache mode: %s\n", value);
                return false;
            }
        } else if (std::strcmp(key, "--help") == 0 || std::strcmp(key, "-h") == 0) {
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", key);
            return false;
        }
    }

    const bool has_full_model = has_text(args->context.model_path);
    const bool has_components =
        has_text(args->context.diffusion_model_path) &&
        has_text(args->context.vae_path) &&
        has_text(args->context.clip_l_path) &&
        (has_text(args->context.t5xxl_path) || args->context.skip_t5);
    const bool has_minimax_components = has_text(args->context.diffusion_model_path) && has_text(args->context.vae_path) && has_text(args->context.llm_path);

    if (!has_full_model && !has_components && !has_minimax_components) {
        std::fprintf(stderr, "--model, image components, or MiniMax-H3 --diffusion-model/--vae/--llm is required\n");
        return false;
    }
    if (args->server.port <= 0 || args->server.port > 65535) {
        std::fprintf(stderr, "port must be in 1..65535\n");
        return false;
    }
    if (args->defaults.width <= 0 || args->defaults.height <= 0) {
        std::fprintf(stderr, "default width and height must be positive\n");
        return false;
    }
    if (args->defaults.steps <= 0) {
        std::fprintf(stderr, "default steps must be positive\n");
        return false;
    }
    if (args->defaults.frames <= 0 || args->defaults.fps <= 0) { std::fprintf(stderr,"default frames and fps must be positive\n"); return false; }
    const int planning_scale = args->context.latent_upscaler_path != nullptr ? 2 : 1;
    args->context.fit_width=args->defaults.width * planning_scale;
    args->context.fit_height=args->defaults.height * planning_scale;
    args->context.fit_frames=args->defaults.frames;
    args->context.fit_fps=args->defaults.fps;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    Args args;
    if (!parse_args(argc, argv, &args)) {
        print_usage(argv[0]);
        return 1;
    }

    if (has_text(args.backend)) {
        setenv("ED_BACKEND", args.backend, 1);
    }

    std::fprintf(stderr, "loading edge-dit model: %s\n", display_model_path(args.context).c_str());
    ed_context_t* ctx = ed_create_context(&args.context);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create edge-dit context\n");
        return 2;
    }

    std::mutex ctx_mutex;
    EdgeDitServerRuntime runtime;
    runtime.ctx = ctx;
    runtime.ctx_mutex = &ctx_mutex;
    runtime.server = &args.server;
    runtime.context = &args.context;
    runtime.defaults = &args.defaults;
    runtime.display_model_path = display_model_path(args.context);

    httplib::Server server;
    server.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        std::string origin = req.get_header_value("Origin");
        if (origin.empty()) {
            origin = "*";
        }
        res.set_header("Access-Control-Allow-Origin", origin);
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    if (args.server.verbose) {
        server.set_logger([](const httplib::Request& req, const httplib::Response& res) {
            std::fprintf(stderr, "%s %s -> %d\n", req.method.c_str(), req.path.c_str(), res.status);
        });
    }

    register_edgedit_routes(server, runtime);

    std::fprintf(stderr, "edge-dit server listening on http://%s:%d\n",
                 args.server.host.c_str(),
                 args.server.port);
    const bool ok = server.listen(args.server.host, args.server.port);

    ed_free_context(ctx);
    return ok ? 0 : 3;
}
