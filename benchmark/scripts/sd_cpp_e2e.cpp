#include "stable-diffusion.h"
#include "media_io.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    std::string model;
    std::string diffusion_model;
    std::string vae;
    std::string clip_l;
    std::string clip_g;
    std::string t5xxl;
    std::string llm;
    std::string prompt;
    std::string negative_prompt;
    std::string output_dir;
    std::string dtype = "bf16";
    std::string backend;
    std::string params_backend;
    std::string max_vram = "0";
    std::string model_args;
    std::string sample_method;
    std::string scheduler;
    std::string task = "text-to-image";
    std::string model_family;
    std::string init_image;
    std::vector<std::string> ref_images;
    int width = 0;
    int height = 0;
    int steps = 0;
    int64_t seed = 0;
    int warmup_runs = 0;
    int measured_runs = 0;
    int qwen_image_layers = 3;
    int video_frames = 1;
    int fps = 16;
    float guidance = 3.5f;
    float cfg_scale = 1.0f;
    float flow_shift = std::numeric_limits<float>::infinity();
    bool has_flow_shift = false;
    bool flash_attn = false;
    bool diffusion_flash_attn = false;
    bool vae_tiling = false;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --prompt PROMPT --output-dir DIR --width N --height N "
        << "--steps N --seed N --warmup-runs N --measured-runs N "
        << "[--model PATH | --diffusion-model PATH] [component paths] "
        << "[--cfg-scale N] [--distilled-guidance N]\n";
    std::exit(2);
}

std::string take_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        usage(argv[0]);
    }
    ++index;
    return argv[index];
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--model") {
            args.model = take_value(i, argc, argv);
        } else if (key == "--diffusion-model") {
            args.diffusion_model = take_value(i, argc, argv);
        } else if (key == "--vae") {
            args.vae = take_value(i, argc, argv);
        } else if (key == "--clip-l") {
            args.clip_l = take_value(i, argc, argv);
        } else if (key == "--clip-g") {
            args.clip_g = take_value(i, argc, argv);
        } else if (key == "--t5xxl") {
            args.t5xxl = take_value(i, argc, argv);
        } else if (key == "--llm") {
            args.llm = take_value(i, argc, argv);
        } else if (key == "--prompt") {
            args.prompt = take_value(i, argc, argv);
        } else if (key == "--negative-prompt") {
            args.negative_prompt = take_value(i, argc, argv);
        } else if (key == "--output-dir") {
            args.output_dir = take_value(i, argc, argv);
        } else if (key == "--width") {
            args.width = std::stoi(take_value(i, argc, argv));
        } else if (key == "--height") {
            args.height = std::stoi(take_value(i, argc, argv));
        } else if (key == "--steps") {
            args.steps = std::stoi(take_value(i, argc, argv));
        } else if (key == "--seed") {
            args.seed = std::stoll(take_value(i, argc, argv));
        } else if (key == "--guidance" || key == "--distilled-guidance") {
            args.guidance = std::stof(take_value(i, argc, argv));
        } else if (key == "--cfg-scale") {
            args.cfg_scale = std::stof(take_value(i, argc, argv));
        } else if (key == "--flow-shift") {
            args.flow_shift = std::stof(take_value(i, argc, argv));
            args.has_flow_shift = true;
        } else if (key == "--dtype") {
            args.dtype = take_value(i, argc, argv);
        } else if (key == "--backend") {
            args.backend = take_value(i, argc, argv);
        } else if (key == "--params-backend") {
            args.params_backend = take_value(i, argc, argv);
        } else if (key == "--max-vram") {
            args.max_vram = take_value(i, argc, argv);
        } else if (key == "--offload-to-cpu") {
            // sd.cpp offloads params by placing them on the CPU backend.
            if (args.params_backend.empty()) {
                args.params_backend = "*=cpu";
            }
        } else if (key == "--model-args") {
            args.model_args = take_value(i, argc, argv);
        } else if (key == "--sample-method") {
            args.sample_method = take_value(i, argc, argv);
        } else if (key == "--scheduler") {
            args.scheduler = take_value(i, argc, argv);
        } else if (key == "--warmup-runs") {
            args.warmup_runs = std::stoi(take_value(i, argc, argv));
        } else if (key == "--measured-runs") {
            args.measured_runs = std::stoi(take_value(i, argc, argv));
        } else if (key == "--flash-attn") {
            args.flash_attn = true;
        } else if (key == "--diffusion-flash-attn" || key == "--diffusion-fa") {
            args.diffusion_flash_attn = true;
        } else if (key == "--vae-tiling") {
            args.vae_tiling = true;
        } else if (key == "--qwen-image-layers") {
            args.qwen_image_layers = std::stoi(take_value(i, argc, argv));
        } else if (key == "--task") {
            args.task = take_value(i, argc, argv);
        } else if (key == "--model-family") {
            args.model_family = take_value(i, argc, argv);
        } else if (key == "--init-img" || key == "--init-image") {
            args.init_image = take_value(i, argc, argv);
        } else if (key == "--ref-image") {
            args.ref_images.push_back(take_value(i, argc, argv));
        } else if (key == "--video-frames") {
            args.video_frames = std::stoi(take_value(i, argc, argv));
        } else if (key == "--fps") {
            args.fps = std::stoi(take_value(i, argc, argv));
        } else {
            std::cerr << "unknown argument: " << key << "\n";
            usage(argv[0]);
        }
    }
    if (args.prompt.empty() || args.output_dir.empty() || args.width <= 0 || args.height <= 0 ||
        args.steps <= 0 || args.measured_runs < 0 || args.warmup_runs < 0 ||
        (args.model.empty() && args.diffusion_model.empty())) {
        usage(argv[0]);
    }
    return args;
}

struct StageRecord {
    bool present = false;
    double begin_epoch_s = 0.0;
    double end_epoch_s = 0.0;
    double duration_ms = 0.0;
};

struct StageCapture {
    StageRecord encode;   // text_encoder
    StageRecord denoise;  // dit
    StageRecord decode;   // vae
};

static StageCapture g_stage_capture;
static bool g_stage_capturing = false;
// Video generation logs "sampling completed" for the denoise stage instead of the
// image path's "generating N latent images completed"; switch the trigger per task.
static bool g_video_mode = false;

double epoch_now_s() {
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Parse the "...taking X.XXs" duration (seconds) out of a log line; -1 if absent.
double parse_taking_seconds(const std::string& text) {
    const std::string key = "taking ";
    size_t pos = text.find(key);
    if (pos == std::string::npos) {
        return -1.0;
    }
    pos += key.size();
    try {
        size_t consumed = 0;
        double value = std::stod(text.substr(pos), &consumed);
        // The library may print "taking N ms" for some encode paths; normalize.
        std::string tail = text.substr(pos + consumed);
        if (tail.find("ms") != std::string::npos && tail.find('s') == tail.find("ms")) {
            return value / 1000.0;
        }
        return value;
    } catch (const std::exception&) {
        return -1.0;
    }
}

void record_stage(StageRecord& record, double seconds) {
    if (seconds < 0.0) {
        return;
    }
    const double end_s = epoch_now_s();
    record.present = true;
    record.end_epoch_s = end_s;
    record.begin_epoch_s = end_s - seconds;
    record.duration_ms = seconds * 1000.0;
    // Emit the unified epoch marker on stdout for cross-system consistency.
    const char* stage_name = &record == &g_stage_capture.encode   ? "encode"
                             : &record == &g_stage_capture.denoise ? "denoise"
                                                                   : "decode";
    std::printf("[[phase]] stage=%s event=begin t=%.6f\n", stage_name, record.begin_epoch_s);
    std::printf("[[phase]] stage=%s event=end t=%.6f\n", stage_name, record.end_epoch_s);
    std::fflush(stdout);
}

void log_callback(sd_log_level_t level, const char* text, void*) {
    if (level == SD_LOG_DEBUG) {
        return;
    }
    if (g_stage_capturing && text != nullptr) {
        const std::string line(text);
        if (line.find("get_learned_condition completed") != std::string::npos) {
            record_stage(g_stage_capture.encode, parse_taking_seconds(line));
        } else if (!g_video_mode && line.find("latent images completed") != std::string::npos) {
            record_stage(g_stage_capture.denoise, parse_taking_seconds(line));
        } else if (g_video_mode && line.find("sampling completed") != std::string::npos) {
            // Video denoise boundary (no "latent images completed" line is emitted).
            record_stage(g_stage_capture.denoise, parse_taking_seconds(line));
        } else if (line.find("decode_first_stage completed") != std::string::npos) {
            record_stage(g_stage_capture.decode, parse_taking_seconds(line));
        }
    }
    std::cerr << text;
}

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}


std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
                break;
        }
    }
    return out.str();
}

void write_png(const sd_image_t& image, const fs::path& path) {
    if (image.data == nullptr || image.width == 0 || image.height == 0 || image.channel < 3) {
        throw std::runtime_error("generated image is empty or not RGB-compatible");
    }
    fs::create_directories(path.parent_path());
    // Reuse the library media_io PNG writer (stb_image_write) so the encoding
    // matches sd-cli exactly; the ".png" suffix selects the PNG encoder.
    if (!write_image_to_file(path.string(), image.data,
                             static_cast<int>(image.width),
                             static_cast<int>(image.height),
                             static_cast<int>(image.channel))) {
        throw std::runtime_error("failed to write sample output: " + path.string());
    }
}

// Load an input/reference image into an sd_image_t (RGB). Reuses the library's
// media_io loader so decoding matches sd-cli exactly. Caller owns image.data.
sd_image_t load_input_image(const std::string& path) {
    sd_image_t image{0, 0, 3, nullptr};
    if (!load_sd_image_from_file(&image, path.c_str(), 0, 0, 3)) {
        throw std::runtime_error("failed to load input image: " + path);
    }
    return image;
}

void write_metrics(
    const fs::path& output_dir,
    double load_ms,
    const std::vector<double>& warmup_ms,
    const std::vector<double>& measured_ms,
    int steps,
    const fs::path& sample_dir,
    const StageCapture& stages) {
    fs::create_directories(output_dir);
    std::ofstream out(output_dir / "runner_metrics.json");
    if (!out) {
        throw std::runtime_error("failed to open runner_metrics.json");
    }
    auto write_array = [&](const std::vector<double>& values) {
        out << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << std::fixed << std::setprecision(3) << values[i];
        }
        out << "]";
    };
    auto write_component = [&](const char* name, const StageRecord& rec, bool trailing_comma) {
        out << "    \"" << name << "\": ";
        if (rec.present) {
            out << std::fixed << std::setprecision(3) << rec.duration_ms;
        } else {
            out << "null";
        }
        out << (trailing_comma ? ",\n" : "\n");
    };
    double mean = 0.0;
    for (double value : measured_ms) {
        mean += value;
    }
    if (!measured_ms.empty()) {
        mean /= static_cast<double>(measured_ms.size());
    }
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"metric_source\": \"stable_diffusion_cpp_c_api\",\n"
        << "  \"measurement_boundary\": \"load_once_e2e_generation_no_output_encoding\",\n"
        << "  \"load_ms\": " << std::fixed << std::setprecision(3) << load_ms << ",\n"
        << "  \"warmup_ms\": ";
    write_array(warmup_ms);
    out << ",\n  \"measured_ms\": ";
    write_array(measured_ms);
    out << ",\n  \"component_ms\": {\n";
    write_component("text_encoder", stages.encode, true);
    write_component("dit", stages.denoise, true);
    write_component("vae", stages.decode, true);
    out << "    \"per_step_avg\": ";
    if (stages.denoise.present && steps > 0) {
        out << std::fixed << std::setprecision(3) << (stages.denoise.duration_ms / static_cast<double>(steps));
    } else if (!measured_ms.empty() && steps > 0) {
        out << std::fixed << std::setprecision(3) << (mean / static_cast<double>(steps));
    } else {
        out << "null";
    }
    out << "\n  },\n";
    // Per-stage [begin, end] epoch windows for external gpu_monitor segmentation.
    out << "  \"stage_boundaries\": {\n";
    bool wrote_any = false;
    auto write_boundary = [&](const char* name, const StageRecord& rec) {
        if (!rec.present) {
            return;
        }
        if (wrote_any) {
            out << ",\n";
        }
        out << "    \"" << name << "\": ["
            << std::fixed << std::setprecision(6) << rec.begin_epoch_s << ", "
            << std::fixed << std::setprecision(6) << rec.end_epoch_s << "]";
        wrote_any = true;
    };
    write_boundary("text_encoder", stages.encode);
    write_boundary("dit", stages.denoise);
    write_boundary("vae", stages.decode);
    out << "\n  },\n"
        << "  \"sample_output_dir\": \"" << json_escape(sample_dir.string()) << "\"\n"
        << "}\n";
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        fs::path output_dir = fs::absolute(args.output_dir);
        fs::path sample_dir = output_dir / "samples" / "stable-diffusion.cpp";
        fs::create_directories(sample_dir);

        sd_set_log_callback(log_callback, nullptr);

        sd_ctx_params_t ctx_params;
        sd_ctx_params_init(&ctx_params);
        ctx_params.model_path = args.model.empty() ? nullptr : args.model.c_str();
        ctx_params.diffusion_model_path = args.diffusion_model.empty() ? nullptr : args.diffusion_model.c_str();
        ctx_params.vae_path = args.vae.empty() ? nullptr : args.vae.c_str();
        ctx_params.clip_l_path = args.clip_l.empty() ? nullptr : args.clip_l.c_str();
        ctx_params.clip_g_path = args.clip_g.empty() ? nullptr : args.clip_g.c_str();
        ctx_params.t5xxl_path = args.t5xxl.empty() ? nullptr : args.t5xxl.c_str();
        ctx_params.llm_path = args.llm.empty() ? nullptr : args.llm.c_str();
        ctx_params.backend = args.backend.empty() ? nullptr : args.backend.c_str();
        ctx_params.params_backend = args.params_backend.empty() ? nullptr : args.params_backend.c_str();
        ctx_params.max_vram = args.max_vram.empty() ? "0" : args.max_vram.c_str();
        ctx_params.model_args = args.model_args.empty() ? nullptr : args.model_args.c_str();
        ctx_params.wtype = str_to_sd_type(args.dtype.c_str());
        ctx_params.rng_type = CUDA_RNG;
        ctx_params.sampler_rng_type = CUDA_RNG;
        ctx_params.flash_attn = args.flash_attn;
        ctx_params.diffusion_flash_attn = args.diffusion_flash_attn;

        double load_start = now_ms();
        sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
        double load_ms = now_ms() - load_start;
        if (ctx == nullptr) {
            std::cerr << "new_sd_ctx failed\n";
            return 1;
        }

        std::vector<double> warmup_ms;
        std::vector<double> measured_ms;
        const bool is_video = args.task == "text-to-video";
        g_video_mode = is_video;

        // Load edit input/reference images once; sd-cli maps Kontext to --ref-image
        // and Qwen-Image-Edit to --init-img. Owned here, freed after the loop.
        sd_image_t init_image{0, 0, 3, nullptr};
        std::vector<sd_image_t> ref_images;
        if (!args.init_image.empty()) {
            init_image = load_input_image(args.init_image);
        }
        for (const auto& ref_path : args.ref_images) {
            ref_images.push_back(load_input_image(ref_path));
        }

        const int total_runs = args.warmup_runs + args.measured_runs;
        for (int index = 0; index < total_runs; ++index) {
            const bool is_warmup = index < args.warmup_runs;
            const int phase_index = is_warmup ? index : index - args.warmup_runs;

            sd_image_t* images = nullptr;
            int image_count = 0;
            if (!is_warmup) {
                g_stage_capture = StageCapture{};
                g_stage_capturing = true;
            }
            double start = now_ms();
            bool ok = false;
            if (is_video) {
                sd_vid_gen_params_t vid_params;
                sd_vid_gen_params_init(&vid_params);
                vid_params.prompt = args.prompt.c_str();
                vid_params.negative_prompt = args.negative_prompt.c_str();
                vid_params.width = args.width;
                vid_params.height = args.height;
                vid_params.seed = args.seed;
                vid_params.video_frames = args.video_frames;
                vid_params.fps = args.fps;
                vid_params.sample_params.sample_steps = args.steps;
                vid_params.sample_params.guidance.txt_cfg = args.cfg_scale;
                vid_params.sample_params.guidance.img_cfg = args.cfg_scale;
                vid_params.sample_params.guidance.distilled_guidance = args.guidance;
                if (args.has_flow_shift) {
                    vid_params.sample_params.flow_shift = args.flow_shift;
                }
                vid_params.sample_params.sample_method = args.sample_method.empty()
                    ? sd_get_default_sample_method(ctx)
                    : str_to_sample_method(args.sample_method.c_str());
                vid_params.sample_params.scheduler = args.scheduler.empty()
                    ? sd_get_default_scheduler(ctx, vid_params.sample_params.sample_method)
                    : str_to_scheduler(args.scheduler.c_str());
                if (args.vae_tiling) {
                    vid_params.vae_tiling_params.enabled = true;
                }
                sd_audio_t* audio = nullptr;
                ok = generate_video(ctx, &vid_params, &images, &image_count, &audio);
                if (audio != nullptr) {
                    free_sd_audio(audio);
                }
            } else {
                sd_img_gen_params_t gen_params;
                sd_img_gen_params_init(&gen_params);
                gen_params.prompt = args.prompt.c_str();
                gen_params.negative_prompt = args.negative_prompt.c_str();
                gen_params.width = args.width;
                gen_params.height = args.height;
                gen_params.seed = args.seed;
                gen_params.batch_count = 1;
                gen_params.qwen_image_layers = args.qwen_image_layers;
                gen_params.sample_params.sample_steps = args.steps;
                gen_params.sample_params.guidance.txt_cfg = args.cfg_scale;
                gen_params.sample_params.guidance.img_cfg = args.cfg_scale;
                gen_params.sample_params.guidance.distilled_guidance = args.guidance;
                if (args.has_flow_shift) {
                    gen_params.sample_params.flow_shift = args.flow_shift;
                }
                gen_params.sample_params.sample_method = args.sample_method.empty()
                    ? sd_get_default_sample_method(ctx)
                    : str_to_sample_method(args.sample_method.c_str());
                gen_params.sample_params.scheduler = args.scheduler.empty()
                    ? sd_get_default_scheduler(ctx, gen_params.sample_params.sample_method)
                    : str_to_scheduler(args.scheduler.c_str());
                if (args.vae_tiling) {
                    gen_params.vae_tiling_params.enabled = true;
                }
                // Image-editing inputs: qwen-edit fills init_image, Kontext fills ref_images.
                if (init_image.data != nullptr) {
                    gen_params.init_image = init_image;
                }
                if (!ref_images.empty()) {
                    gen_params.ref_images = ref_images.data();
                    gen_params.ref_images_count = static_cast<int>(ref_images.size());
                }
                ok = generate_image(ctx, &gen_params, &images, &image_count);
            }
            double elapsed = now_ms() - start;
            g_stage_capturing = false;
            if (!ok || images == nullptr || image_count <= 0) {
                free_sd_images(images, image_count);
                free_sd_ctx(ctx);
                std::cerr << (is_video ? "generate_video failed\n" : "generate_image failed\n");
                return 1;
            }
            if (is_warmup) {
                warmup_ms.push_back(elapsed);
            } else {
                measured_ms.push_back(elapsed);
                if (is_video) {
                    std::ostringstream name;
                    name << "output_" << std::setw(3) << std::setfill('0') << phase_index << ".avi";
                    fs::path video_path = sample_dir / name.str();
                    if (create_video_from_sd_images(
                            video_path.string().c_str(), images, image_count, args.fps, 90, nullptr) != 0) {
                        free_sd_images(images, image_count);
                        free_sd_ctx(ctx);
                        std::cerr << "create_video_from_sd_images failed\n";
                        return 1;
                    }
                } else {
                    std::ostringstream name;
                    name << "output_" << std::setw(3) << std::setfill('0') << phase_index << ".png";
                    write_png(images[0], sample_dir / name.str());
                }
            }
            free_sd_images(images, image_count);
            std::cout << "[sd-cpp-e2e] " << (is_warmup ? "warmup " : "measured ")
                      << phase_index << " " << std::fixed << std::setprecision(3)
                      << (elapsed / 1000.0) << "s\n";
        }

        if (init_image.data != nullptr) {
            free(init_image.data);
        }
        for (auto& ref : ref_images) {
            free(ref.data);
        }

        free_sd_ctx(ctx);
        write_metrics(output_dir, load_ms, warmup_ms, measured_ms, args.steps, sample_dir, g_stage_capture);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
}
