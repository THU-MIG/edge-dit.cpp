// ed-sample: single-run text-to-image benchmark generator for edge-dit.cpp.
//
// One invocation runs ONE (model, cache-mode, prompt-file) combination: it loads
// the model once, generates every prompt in the file, and writes a run directory
// that the Python evaluation code (benchmark/) can score directly. Matrix
// expansion, multi-GPU fan-out, and quality scoring are the shell orchestrator's
// job (benchmark/run_all.sh) -- this binary is deliberately single-purpose.
//
// Output contract (mirrors BiDCache sample.py, consumed by collect_report.py):
//
//   <output_dir>/
//     imgs/img_000000.png ...            # one PNG per prompt, index-named
//     imgs_metadata/img_000000.json ...  # per-image prompt/seed/latency metadata
//     prompts.txt                        # the prompts actually used
//     image_metadata.json               # prompt_index/prompt/seeds/image_indices
//     config.json                       # the resolved sampling configuration
//     timing.json                       # e2e_time / time_wo_decoding / time_with_decoding
//
// Timing: the model is loaded once, outside the timed region. Each generation is
// timed tightly around ed_generate_image (PNG encoding happens outside it). The
// first --warmup generations are excluded from the statistics (their images are
// still written); --repeat re-runs the whole prompt slice for more timing samples
// (default 1). edge-dit fuses VAE decode into ed_generate_image, so denoise and
// decode cannot be split -- e2e_time / time_wo_decoding / time_with_decoding all
// report the same per-image wall-clock. The cache *denoise* speedup (steps
// skipped) is logged separately per run ("reused X/N steps").

#include "edge-dit.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define STB_IMAGE_IMPLEMENTATION
#include "ggml/examples/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#include "cli_common.hpp"

namespace {

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "ed-sample: single-run text-to-image benchmark generator.\n\n"
        "Usage:\n"
        "  %s --model <diffusers-dir> --prompt_file <txt> --output_dir <dir> [options]\n\n"
        "Required:\n"
        "  --model <path>            Diffusers model directory (or component flags:\n"
        "                            --diffusion-model/--vae/--clip_l/[--clip_g]/(--t5xxl|--no-t5))\n"
        "  --prompt_file <path>      Text file, one prompt per line\n"
        "  --output_dir <path>       Output run directory (created)\n\n"
        "Sampling:\n"
        "  --cache_method <mode>     off|original|easycache|ucache|dbcache|taylorseer|cache-dit|magcache|dicache|sencache\n"
        "  --width <int>             Default 1024\n"
        "  --height <int>            Default 1024\n"
        "  --num_steps <int>         Default 20\n"
        "  --seed <int>              Base seed; per-image seed = seed + image_index. Default -1\n"
        "  --guidance_scale <float>  Flux distilled guidance. Default 3.5\n"
        "  --cfg_scale <float>       Classifier-free guidance scale. Default 1.0\n"
        "  --flow_shift <float>      Flow scheduler shift. Default model default\n"
        "  --negative_prompt <text>  Negative prompt (used when cfg_scale != 1). Default empty\n"
        "  --start_index <int>       First prompt index (inclusive). Default 0\n"
        "  --end_index <int>         Last prompt index (exclusive). Default all\n"
        "  --threads <int>           CPU thread count. Default auto\n"
        "  --backend <name>          auto|cpu|cuda|vulkan|metal|gpu. Default auto\n"
        "  --no-flash-attention      Disable flash attention\n\n"
        "Timing:\n"
        "  --warmup <int>            Untimed warm-up generations (images still written). Default 1\n"
        "  --repeat <int>            Timed passes over the whole prompt slice. Default 1\n\n"
        "Cache tuning (optional; sensible per-method defaults otherwise):\n"
        "  --cache-taylor-order <int>   --cache-taylor-skip <int>\n"
        "  --cache-threshold <float>    --cache-start <float>   --cache-end <float>\n"
        "  --cache-calibrate <path>     --cache-profile <path>\n",
        prog);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::vector<std::string> read_prompts(const char* path) {
    std::vector<std::string> prompts;
    std::ifstream in(path);
    if (!in.is_open()) {
        return prompts;
    }
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                                 line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        size_t start = 0;
        while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
            ++start;
        }
        line = line.substr(start);
        if (!line.empty()) {
            prompts.push_back(line);
        }
    }
    return prompts;
}

bool write_file(const fs::path& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    out << content;
    return out.good();
}

// min/mean/median/max/stddev over a set of per-image latencies.
struct TimingStats {
    double total = 0.0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double stddev = 0.0;
};

TimingStats compute_stats(std::vector<double> samples) {
    TimingStats stats;
    if (samples.empty()) {
        return stats;
    }
    std::sort(samples.begin(), samples.end());
    stats.min = samples.front();
    stats.max = samples.back();

    double sum = 0.0;
    for (double s : samples) {
        sum += s;
    }
    stats.total = sum;
    stats.mean = sum / static_cast<double>(samples.size());

    const size_t n = samples.size();
    stats.median = (n % 2 == 1)
                       ? samples[n / 2]
                       : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);

    double var = 0.0;
    for (double s : samples) {
        const double d = s - stats.mean;
        var += d * d;
    }
    stats.stddev = std::sqrt(var / static_cast<double>(samples.size()));
    return stats;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    FluxCliArgs args;
    if (!parse_args(argc, argv, &args)) {
        print_usage(argv[0]);
        return 1;
    }
    if (args.prompt_file == nullptr || std::strlen(args.prompt_file) == 0) {
        std::fprintf(stderr, "ed-sample requires --prompt_file\n");
        print_usage(argv[0]);
        return 1;
    }
    if (args.output_dir == nullptr || std::strlen(args.output_dir) == 0) {
        std::fprintf(stderr, "ed-sample requires --output_dir\n");
        print_usage(argv[0]);
        return 1;
    }
    if (args.warmup < 0) {
        std::fprintf(stderr, "--warmup must be non-negative\n");
        return 1;
    }
    if (args.repeat < 1) {
        std::fprintf(stderr, "--repeat must be positive\n");
        return 1;
    }

    std::vector<std::string> prompts = read_prompts(args.prompt_file);
    if (prompts.empty()) {
        std::fprintf(stderr, "no prompts found in %s\n", args.prompt_file);
        return 1;
    }

    int start_index = args.start_index < 0 ? 0 : args.start_index;
    int end_index = args.end_index < 0 ? static_cast<int>(prompts.size()) : args.end_index;
    if (end_index > static_cast<int>(prompts.size())) {
        end_index = static_cast<int>(prompts.size());
    }
    if (start_index >= end_index) {
        std::fprintf(stderr, "empty prompt slice [%d, %d)\n", start_index, end_index);
        return 1;
    }

    // Distributed re-exec (mpirun) when a parallel size > 1 is requested.
    const int device_count = count_csv_values(args.devices);
    if (device_count > 0 && is_cpu_backend_name(args.backend)) {
        std::fprintf(stderr, "--devices requires a GPU backend, but --backend cpu was requested\n");
        return 1;
    }
    if (args.devices != nullptr && std::strlen(args.devices) > 0 && !is_distributed_process()) {
        setenv("CUDA_VISIBLE_DEVICES", args.devices, 1);
        if (args.backend == nullptr || std::strlen(args.backend) == 0) {
            args.backend = "gpu";
        }
    }
    const int launch_status = launch_distributed_cli(argc, argv, args, device_count);
    if (launch_status >= 0) {
        return launch_status;
    }

    if (args.backend != nullptr && std::strlen(args.backend) > 0) {
        setenv("ED_BACKEND", args.backend, 1);
    }

    const fs::path out_root(args.output_dir);
    std::error_code ec;
    fs::create_directories(out_root / "imgs", ec);
    fs::create_directories(out_root / "imgs_metadata", ec);

    // prompts.txt (all prompts, like sample.py).
    {
        std::string body;
        for (const auto& p : prompts) {
            body += p;
            body += "\n";
        }
        write_file(out_root / "prompts.txt", body);
    }

    // image_metadata.json (full list; one image per prompt).
    {
        std::string body = "[\n";
        for (size_t idx = 0; idx < prompts.size(); ++idx) {
            const int64_t seed = args.seed + static_cast<int64_t>(idx);
            body += "  {\n";
            body += "    \"prompt_index\": " + std::to_string(idx) + ",\n";
            body += "    \"prompt\": \"" + json_escape(prompts[idx]) + "\",\n";
            body += "    \"seeds\": [" + std::to_string(seed) + "],\n";
            body += "    \"image_indices\": [" + std::to_string(idx) + "]\n";
            body += idx + 1 < prompts.size() ? "  },\n" : "  }\n";
        }
        body += "]\n";
        write_file(out_root / "image_metadata.json", body);
    }

    // config.json (resolved sampling configuration).
    {
        std::string body = "{\n";
        body += "  \"model_path\": \"" + json_escape(args.model_path ? args.model_path : "") + "\",\n";
        body += "  \"cache_method\": \"" + std::string(cache_mode_name(args.cache_mode)) + "\",\n";
        body += "  \"backend\": \"" + json_escape(args.backend ? args.backend : "") + "\",\n";
        body += "  \"width\": " + std::to_string(args.width) + ",\n";
        body += "  \"height\": " + std::to_string(args.height) + ",\n";
        body += "  \"num_steps\": " + std::to_string(args.steps) + ",\n";
        body += "  \"seed\": " + std::to_string(args.seed) + ",\n";
        body += "  \"guidance_scale\": " + std::to_string(args.guidance) + ",\n";
        body += "  \"cfg_scale\": " + std::to_string(args.cfg_scale) + ",\n";
        body += "  \"warmup\": " + std::to_string(args.warmup) + ",\n";
        body += "  \"repeat\": " + std::to_string(args.repeat) + ",\n";
        body += "  \"start_index\": " + std::to_string(start_index) + ",\n";
        body += "  \"end_index\": " + std::to_string(end_index) + "\n";
        body += "}\n";
        write_file(out_root / "config.json", body);
    }

    // ---- Load the model ONCE (outside the timed loop) ----
    ed_context_params_t ctx_params;
    ed_context_params_init(&ctx_params);
    ctx_params.model_path = args.model_path;
    ctx_params.diffusion_model_path = args.diffusion_model_path;
    ctx_params.vae_path = args.vae_path;
    ctx_params.clip_l_path = args.clip_l_path;
    ctx_params.clip_g_path = args.clip_g_path;
    ctx_params.t5xxl_path = args.t5xxl_path;
    ctx_params.cfg_parallel_size = args.cfg_parallel_size;
    ctx_params.tp_parallel_size = args.tp_parallel_size;
    ctx_params.sp_parallel_size = args.sp_parallel_size;
    ctx_params.flash_attention = args.flash_attention;
    ctx_params.offload_params_to_cpu = args.offload_to_cpu;
    ctx_params.keep_text_encoder_on_cpu = args.keep_text_encoder_on_cpu;
    ctx_params.keep_vae_on_cpu = args.keep_vae_on_cpu;
    if (args.max_vram > 0.0f) {
        ctx_params.max_vram_gb = args.max_vram;
    }
    ctx_params.qwen_image_zero_cond_t = args.qwen_image_zero_cond_t;
    if (args.threads > 0) {
        ctx_params.n_threads = args.threads;
    }
    ctx_params.weight_type = args.weight_type != nullptr
                                 ? parse_weight_type(args.weight_type, nullptr)
                                 : ED_DTYPE_AUTO;
    ctx_params.tensor_type_rules = args.tensor_type_rules;
    ctx_params.skip_t5 = args.no_t5;
    if (args.vae_tiling == 1) {
        ctx_params.vae_tiling.enabled = true;
    }
    if (args.vae_tile_size > 0.0f) {
        ctx_params.vae_tiling.enabled = true;
        ctx_params.vae_tiling.rel_size_x = args.vae_tile_size;
        ctx_params.vae_tiling.rel_size_y = args.vae_tile_size;
    }

    const auto load_t0 = std::chrono::steady_clock::now();
    ed_context_t* ctx = ed_create_context(&ctx_params);
    const auto load_t1 = std::chrono::steady_clock::now();
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create edge-dit context\n");
        return 2;
    }
    const double load_seconds = std::chrono::duration<double>(load_t1 - load_t0).count();
    const bool is_root = ed_context_parallel_is_root(ctx);

    if (is_root) {
        std::printf("=== ed-sample ===\n");
        std::printf("cache mode   : %s\n", cache_mode_name(args.cache_mode));
        std::printf("resolution   : %dx%d\n", args.width, args.height);
        std::printf("steps        : %d\n", args.steps);
        std::printf("prompts      : [%d, %d) of %zu\n", start_index, end_index, prompts.size());
        std::printf("warmup/repeat: %d / %d\n", args.warmup, args.repeat);
        std::printf("model load   : %.3fs\n", load_seconds);
        std::printf("----------------------------------------\n");
        std::fflush(stdout);
    }

    // ---- Generate. warmup generations are untimed (images still written);
    //      the timed pass(es) feed the latency statistics. ----
    std::vector<double> samples;  // timed per-image latencies
    int warmed = 0;
    int generated = 0;            // distinct images written (last write wins)
    int rc = 0;

    for (int pass = 0; pass < args.repeat && rc == 0; ++pass) {
        for (int idx = start_index; idx < end_index; ++idx) {
            const std::string& prompt = prompts[idx];
            const int64_t seed = args.seed + static_cast<int64_t>(idx);
            const bool is_warmup = warmed < args.warmup;

            ed_image_generation_params_t gen;
            ed_image_generation_params_init(&gen);
            gen.prompt = prompt.c_str();
            gen.negative_prompt = args.negative_prompt;
            gen.width = args.width;
            gen.height = args.height;
            gen.seed = seed;
            gen.batch_count = 1;
            gen.sample.sampler = ED_SAMPLER_AUTO;
            gen.sample.scheduler = ED_SCHEDULER_AUTO;
            gen.sample.steps = args.steps;
            gen.sample.cfg_scale = args.cfg_scale;
            gen.sample.image_cfg_scale = 1.0f;
            gen.sample.distilled_guidance = args.guidance;
            gen.sample.flow_shift = args.flow_shift;
            apply_cache_args(args, &gen.sample);

            const auto t0 = std::chrono::steady_clock::now();
            ed_image_batch_t output = {};
            ed_status_t status = ed_generate_image(ctx, &gen, &output);
            const auto t1 = std::chrono::steady_clock::now();

            if (status != ED_STATUS_OK) {
                std::fprintf(stderr, "ed_generate_image failed for prompt %d, status=%d\n",
                             idx, static_cast<int>(status));
                const char* err = ed_get_last_error(ctx);
                if (err != nullptr && std::strlen(err) > 0) {
                    std::fprintf(stderr, "last error: %s\n", err);
                }
                ed_free_image_batch(&output);
                rc = 3;
                break;
            }

            const double seconds = std::chrono::duration<double>(t1 - t0).count();

            if (is_warmup) {
                ++warmed;
            } else {
                samples.push_back(seconds);
            }

            // Non-root ranks participate in generation but do not own the output.
            if (!is_root) {
                ed_free_image_batch(&output);
                continue;
            }

            if (output.count <= 0 || output.images == nullptr) {
                std::fprintf(stderr, "empty output for prompt %d\n", idx);
                ed_free_image_batch(&output);
                rc = 4;
                break;
            }

            char fname[32];
            std::snprintf(fname, sizeof(fname), "img_%06d.png", idx);
            const fs::path img_path = out_root / "imgs" / fname;
            if (!save_png(img_path.string().c_str(), output.images[0])) {
                std::fprintf(stderr, "failed to save %s\n", img_path.string().c_str());
                ed_free_image_batch(&output);
                rc = 5;
                break;
            }
            ++generated;

            char meta_name[40];
            std::snprintf(meta_name, sizeof(meta_name), "img_%06d.json", idx);
            std::string meta = "{\n";
            meta += "  \"image_index\": " + std::to_string(idx) + ",\n";
            meta += "  \"prompt_index\": " + std::to_string(idx) + ",\n";
            meta += "  \"image_offset\": 0,\n";
            meta += "  \"seed\": " + std::to_string(seed) + ",\n";
            meta += "  \"prompt\": \"" + json_escape(prompt) + "\",\n";
            meta += "  \"latency_seconds\": " + std::to_string(seconds) + ",\n";
            meta += "  \"warmup\": " + std::string(is_warmup ? "true" : "false") + ",\n";
            meta += "  \"filename\": \"" + std::string(fname) + "\"\n";
            meta += "}\n";
            write_file(out_root / "imgs_metadata" / meta_name, meta);

            std::printf("[ed-sample] pass %d/%d  %d/%d  seed=%lld  %.3fs%s  -> %s\n",
                        pass + 1, args.repeat, idx - start_index + 1, end_index - start_index,
                        static_cast<long long>(seed), seconds, is_warmup ? " (warmup)" : "", fname);
            std::fflush(stdout);

            ed_free_image_batch(&output);
        }
    }

    if (rc != 0) {
        ed_free_context(ctx);
        return rc;
    }

    // ---- timing.json (BiDCache-compatible keys, read by collect_report.py) ----
    // edge-dit fuses VAE decode into ed_generate_image, so e2e_time /
    // time_wo_decoding / time_with_decoding all carry the same per-image
    // wall-clock. Extra keys (min/median/stddev/samples) are ignored by the
    // collector but useful for manual inspection.
    if (is_root && !samples.empty()) {
        const TimingStats stats = compute_stats(samples);
        std::string body = "{\n";
        body += "  \"e2e_time\": {\"total\": " + std::to_string(stats.total) +
                ", \"average\": " + std::to_string(stats.mean) + "},\n";
        body += "  \"time_wo_decoding\": {\"total\": " + std::to_string(stats.total) +
                ", \"average\": " + std::to_string(stats.mean) + "},\n";
        body += "  \"time_with_decoding\": {\"total\": " + std::to_string(stats.total) +
                ", \"average\": " + std::to_string(stats.mean) + "},\n";
        body += "  \"num_images\": " + std::to_string(samples.size()) + ",\n";
        body += "  \"model_load_seconds\": " + std::to_string(load_seconds) + ",\n";
        body += "  \"e2e_seconds\": {\n";
        body += "    \"min\": " + std::to_string(stats.min) + ",\n";
        body += "    \"mean\": " + std::to_string(stats.mean) + ",\n";
        body += "    \"median\": " + std::to_string(stats.median) + ",\n";
        body += "    \"max\": " + std::to_string(stats.max) + ",\n";
        body += "    \"stddev\": " + std::to_string(stats.stddev) + "\n";
        body += "  }\n";
        body += "}\n";
        write_file(out_root / "timing.json", body);
        std::printf("[ed-sample] DONE %d images (%zu timed), avg %.3fs/img -> %s\n",
                    generated, samples.size(), stats.mean,
                    (out_root / "timing.json").string().c_str());
    } else if (is_root) {
        std::printf("[ed-sample] DONE %d images, no timed samples (warmup >= total generations)\n",
                    generated);
    }

    ed_free_context(ctx);
    return 0;
}
