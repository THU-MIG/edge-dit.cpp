#include "edge-dit.h"

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>

namespace fs = std::filesystem;

inline bool load_image(const char* path, ed_image_t* image);

static uint16_t read_le16(const uint8_t* data) { return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8); }
static uint32_t read_le32(const uint8_t* data) { return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) | (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24); }

static bool load_wav(const std::string& path, std::vector<float>* samples, ed_audio_t* audio) {
    std::ifstream file(path, std::ios::binary);
    uint8_t header[12];
    if (!file.read(reinterpret_cast<char*>(header), sizeof(header)) || std::memcmp(header, "RIFF", 4) != 0 || std::memcmp(header + 8, "WAVE", 4) != 0) return false;
    uint16_t format = 0, channels = 0, bits = 0; uint32_t rate = 0, data_size = 0; std::streampos data_pos = -1;
    while (file.good()) { uint8_t chunk[8]; if (!file.read(reinterpret_cast<char*>(chunk), 8)) break; uint32_t size = read_le32(chunk + 4); auto pos = file.tellg();
        if (std::memcmp(chunk, "fmt ", 4) == 0) { std::vector<uint8_t> value(size); if (!file.read(reinterpret_cast<char*>(value.data()), size) || size < 16) return false; format = read_le16(value.data()); channels = read_le16(value.data() + 2); rate = read_le32(value.data() + 4); bits = read_le16(value.data() + 14); if (format == 0xfffe && size >= 40) format = read_le16(value.data() + 24); }
        else if (std::memcmp(chunk, "data", 4) == 0) { data_pos = pos; data_size = size; file.seekg(size, std::ios::cur); }
        else file.seekg(size, std::ios::cur); if (size & 1) file.seekg(1, std::ios::cur);
    }
    if (data_pos == std::streampos(-1) || channels == 0 || rate == 0 ||
        !((format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) ||
          (format == 3 && (bits == 32 || bits == 64)))) return false;
    const size_t bytes_per = bits / 8, count = data_size / bytes_per; samples->resize(count); file.clear(); file.seekg(data_pos);
    std::vector<uint8_t> bytes(data_size); if (!file.read(reinterpret_cast<char*>(bytes.data()), data_size)) return false;
    for (size_t index = 0; index < count; ++index) {
        const uint8_t* sample = bytes.data() + index * bytes_per;
        if (format == 3 && bits == 32) {
            float value;
            std::memcpy(&value, sample, sizeof(value));
            (*samples)[index] = std::clamp(value, -1.f, 1.f);
        } else if (format == 3) {
            double value;
            std::memcpy(&value, sample, sizeof(value));
            (*samples)[index] = std::clamp(static_cast<float>(value), -1.f, 1.f);
        } else if (bits == 8) {
            (*samples)[index] = (static_cast<float>(sample[0]) - 128.f) / 128.f;
        } else if (bits == 16) {
            (*samples)[index] = static_cast<float>(static_cast<int16_t>(read_le16(sample))) / 32768.f;
        } else if (bits == 24) {
            int32_t value = static_cast<int32_t>(sample[0]) |
                            (static_cast<int32_t>(sample[1]) << 8) |
                            (static_cast<int32_t>(sample[2]) << 16);
            if (value & 0x00800000) value |= ~0x00ffffff;
            (*samples)[index] = static_cast<float>(value) / 8388608.f;
        } else {
            (*samples)[index] = static_cast<float>(static_cast<int32_t>(read_le32(sample))) / 2147483648.f;
        }
    }
    *audio = {rate, channels, count / channels, samples->data()}; return true;
}

static bool load_images_from_dir(const std::string& directory, std::vector<ed_image_t>* frames) {
    if (!fs::is_directory(directory)) return false; std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(directory)) if (entry.is_regular_file()) { auto ext = entry.path().extension().string(); std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower); if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".webp") paths.push_back(entry.path()); }
    std::sort(paths.begin(), paths.end()); frames->resize(paths.size()); for (size_t index = 0; index < paths.size(); ++index) if (!load_image(paths[index].c_str(), &(*frames)[index])) return false; return !frames->empty();
}

#define STB_IMAGE_IMPLEMENTATION
#include "ggml/examples/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#include "cli_common.hpp"

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --model <model-or-diffusers-dir> --prompt <text> [options]\n"
        "  %s --diffusion-model <path> --vae <path> --clip_l <path> [--clip_g <path>] (--t5xxl <path> | --no-t5) --prompt <text> [options]\n"
        "  %s -M vid_gen --diffusion-model <path> --vae <path> [--audio-vae <path>] --llm <path> --prompt <text> [options]\n"
        "Options:\n"
        "  --video, -M vid_gen       Generate video frames instead of an image\n"
        "  --video-format <fmt>      Video format: auto, avi, mp4, mov, mkv, webm. Default: auto\n"
        "  -i, --image, --init-img <path>  Input image or MiniMax-H3/LTX-2.3 first frame\n"
        "  --end-img <path>          MiniMax-H3/LTX-2.3 last-frame image\n"
        "  --diffusion-model <path>  Standalone DiT transformer weights\n"
        "  --vae <path>              Standalone VAE weights\n"
        "  --audio-vae <path>        Standalone audio VAE weights (MiniMax-H3/LTX-2.3)\n"
        "  --embeddings-connectors <path>  LTX-2.3 text connector weights\n"
        "  --latent-upscaler <path> LTX-2.3 spatial latent x2 upscaler weights\n"
        "  --hires                    Run LTX-2.3 x2 latent upscale and refine pass\n"
        "  --hires-upscalers-dir <dir>  Directory used with --hires-upscaler\n"
        "  --hires-upscaler <name>   LTX upscaler name/path (stable-diffusion.cpp compatible)\n"
        "  --hires-steps <int>        LTX hires refine steps, default: 4\n"
        "  --hires-denoising-strength <float>  LTX hires strength in (0,1], default: 0.7\n"
        "  --hires-sigmas <csv>       Explicit LTX hires sigma schedule\n"
        "  --clip_l <path>           CLIP-L text encoder weights\n"
        "  --clip_g <path>           CLIP-G text encoder weights\n"
        "  --t5xxl <path>            T5XXL text encoder weights\n"
        "  --llm <path>              LLM text encoder weights (MiniMax-H3/Qwen/LTX-2.3)\n"
        "  --llm-vision <path>       Optional vision-language encoder weights\n"
        "  --negative-prompt <text>  Negative prompt text, default: empty\n"
        "  -o, --output <path>       Output image/video path, default: output.png\n"
        "  -W, --width <int>         Image width, default: 1024\n"
        "  -H, --height <int>        Image height, default: 1024\n"
        "  --frames, --video-frames <int>  Video frame count, default: 1\n"
        "  --video-duration <seconds>  MiniMax-H3 duration; aligns to >=22 frames satisfying 17k+5\n"
        "  --fps <int>               Video fps, default: 16\n"
        "  --ref-image <path>        MiniMax-H3 Ref2VA reference image; repeatable\n"
        "  --ref-image-size <mode>   Reference image sizing: match or max, default: max\n"
        "  --ref-video <path>        MiniMax-H3 Ref2VA reference video frames directory or media file; repeatable\n"
        "                            Media files are decoded with ffmpeg; embedded audio is paired automatically\n"
        "  --ref-video-audio <path>  WAV soundtrack paired with the corresponding --ref-video\n"
        "  --ref-audio <path>        Additional MiniMax-H3 Ref2VA WAV; repeatable; requires an image/video reference\n"
        "  --steps <int>             Sampling steps, default: 20\n"
        "  -s, --seed <int64>        Seed, default: -1\n"
        "  -t, --threads <int>       Thread count, default: 0\n"
        "  --guidance <float>        Flux distilled guidance, default: 3.5\n"
        "  --cfg-scale <float>       Classifier-free guidance scale, default: 1.0\n"
        "  --flow-shift <float>      Flow scheduler shift, default: model default\n"
        "  --qwen-image-zero-cond-t Enable Qwen-Image zero conditional timestep\n"
        "  --sampler <name>          Sampling method: auto, euler, res_multistep\n"
        "  --scheduler <name>        Sigma scheduler: auto, discrete, simple, ltx2\n"
        "  --cache <mode>            Cache mode: off, easycache, ucache, dbcache, taylorseer, cache-dit, magcache, dicache, sencache\n"
        "  --cache-threshold <float> EasyCache/UCache reuse threshold\n"
        "  --cache-start <float>     Cache active window start percent, default: 0.15\n"
        "  --cache-end <float>       Cache active window end percent, default: 0.95\n"
        "  --cache-error-decay <f>   UCache accumulated error decay, default: 1.0\n"
        "  --cache-relative-threshold|--cache-absolute-threshold\n"
        "                            UCache threshold scale mode, default: relative\n"
        "  --cache-no-reset-error    Keep UCache accumulated error after full compute\n"
        "  --cache-fn-blocks <int>   DBCache/CacheDiT front compute blocks, default: 8\n"
        "  --cache-bn-blocks <int>   DBCache/CacheDiT back compute blocks, default: 0\n"
        "  --cache-residual-threshold <float>\n"
        "                            Override method residual threshold; DBCache/CacheDiT default: 0.08\n"
        "  --cache-max-accumulated-residual-diff <float>\n"
        "                            Disable DBCache after accumulated diff reaches this value, -1 means unlimited\n"
        "  --cache-warmup-steps <int>\n"
        "                            Cache warmup full-compute steps, default: 8\n"
        "  --cache-max-cached-steps <int>\n"
        "                            Max cached steps, -1 means unlimited\n"
        "  --cache-max-continuous-cached-steps <int>\n"
        "                            Max continuous cached steps, -1 means unlimited\n"
        "  --cache-taylor-order <int>\n"
        "                            TaylorSeer derivative order, default: 1\n"
        "  --cache-taylor-skip <int> TaylorSeer skip interval, default: 1\n"
        "  --cache-scm-mask <csv>    Steps computation mask, e.g. 1,0,0,1\n"
        "  --cache-static-scm        Use static SCM policy for methods that support it\n"
        "  --cache-calibrate <path>  Profile a table for the selected cache method (requires a\n"
        "                            calibration-capable --cache, e.g. magcache or sencache;\n"
        "                            full-computes every step, sencache also runs extra Jacobian\n"
        "                            forwards) and write it to <path>\n"
        "  --cache-profile <path>    Load a MagCache ratio table / SenCache sensitivity profile\n"
        "                            from <path> (sencache requires this or --cache-calibrate)\n"
        "  --backend <name>          Backend: auto, cpu, cuda, vulkan, metal, gpu. Default: auto\n"
        "  --gpu                     Alias for --backend gpu\n"
        "  --devices <csv>           GPU devices for parallel workers, e.g. 0,1,2,3\n"
        "  --type <dtype>            Weight type / on-the-fly quantization when loading safetensors:\n"
        "                            preserve, f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0,\n"
        "                            q2_k, q3_k, q4_k, q5_k, q6_k. Default: preserve\n"
        "                            ('auto' remains accepted as a compatibility alias)\n"
        "  --tensor-type-rules <csv> Per-tensor quant overrides (mixed quant), e.g. \"attn=q4_0,norm=f16\"\n"
        "                            Each rule is <name-regex>=<ggml-type-name>, comma-separated\n"
        "  --no-t5                   Skip loading T5XXL text encoder (SD3 only; reduces memory, degrades prompt adherence)\n"
        "  --vae-tiling <on|off|auto>  VAE tiled decode (reduces VRAM). auto=enable on low-VRAM GPUs (<=25G); default: auto\n"
        "  --vae-tile-size <float>   VAE tile relative size, default: 5.0 (~32x32 tile). Larger = finer tiles = less VRAM\n"
        "  --offload-to-cpu          Keep model weights on CPU, copy to GPU per-compute (saves VRAM)\n"
        "  --dit-offload             Keep DiT weights on CPU, stage to GPU per step (compute on GPU)\n"
        "  --text-encoder-offload    Keep text-encoder weights on CPU, stage to GPU per encode (compute on GPU)\n"
        "  --minimax-h3-stage-lifecycle  MiniMax-H3: stage Qwen/VAE by phase and release them during DiT\n"
        "  --vae-offload             Keep VAE weights on CPU, stage to GPU per decode (compute on GPU)\n"
        "  --max-vram <GB>           Set the VRAM planning budget; with --auto-fit on\n"
        "                            single-device CUDA, also enforce a guarded allocation ceiling\n"
        "  --auto-allocate           Auto per-component placement under a VRAM planning budget\n"
        "                            = min(--max-vram, free); keeps components resident\n"
        "                            when they fit, offloads (segments) the rest\n"
        "                            (external CUDA workspaces can raise the process peak)\n"
        "  --auto-fit                Fully automatic: choose TE/DiT quantization (DiT q8_0\n"
        "                            down to q4_k) AND placement to fit the VRAM budget.\n"
        "                            Implies --auto-allocate; replans TE/DiT precision while\n"
        "                            --type still controls VAE precision. With --max-vram,\n"
        "                            oversized workloads fail safely. Off by default.\n"
        "  --flash-attention         Enable flash attention, default: on\n"
        "  --no-flash-attention      Disable flash attention\n"
        "  --rng <name>              Accepted for stable-diffusion.cpp CLI compatibility\n"
        "  -v, --verbose             Accepted compatibility flag (logging is configured externally)\n"
        "  --cfg-parallel-size <n>   Split CFG cond/uncond branches across n GPUs, currently supports 1 or 2\n"
        "  --cfg-size <n>            Alias for --cfg-parallel-size\n"
        "  --tp-size <n>             Reserved tensor parallel size, default: 1\n"
        "  --sp-size <n>             Sequence parallel size, default: 1\n"
        "  --profile-graph-cuts      Print graph-cut compute/communication timing summary\n"
        "  --profile-graph-cuts-top <n>\n"
        "                            Print top n slowest graph-cut segments, default: 8\n"
        "  --profile-graph-cuts-all-ranks\n"
        "                            Print graph-cut profile from every parallel rank\n"
        "  --help              Show this help\n",
        prog,
        prog,
        prog
    );
}

static std::string path_extension(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return "";
    }
    return lowercase(path.substr(dot));
}

static std::string replace_extension(const std::string& path, const std::string& ext) {
    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return path + ext;
    }
    return path.substr(0, dot) + ext;
}

static bool is_ffmpeg_video_ext(const std::string& ext) {
    return ext == ".mp4" || ext == ".mov" || ext == ".mkv" || ext == ".webm";
}

static bool is_supported_video_ext(const std::string& ext) {
    return ext == ".avi" || is_ffmpeg_video_ext(ext);
}

static std::string video_output_path(const char* output_path, const char* format) {
    std::string path = output_path != nullptr && output_path[0] != '\0' ? output_path : "output.avi";
    const std::string requested_format = normalized_video_format(format);

    if (requested_format != "auto") {
        return replace_extension(path, "." + requested_format);
    }

    const std::string ext = path_extension(path);
    if (!is_supported_video_ext(ext)) {
        path = replace_extension(path, ".avi");
    }
    return path;
}

static std::string find_imageio_ffmpeg_in_conda(const char* conda_prefix) {
    if (conda_prefix == nullptr || conda_prefix[0] == '\0') {
        return "";
    }

    const fs::path lib_dir = fs::path(conda_prefix) / "lib";
    std::error_code ec;
    if (!fs::is_directory(lib_dir, ec)) {
        return "";
    }

    for (const fs::directory_entry& python_entry : fs::directory_iterator(lib_dir, ec)) {
        if (ec) {
            break;
        }
        if (!python_entry.is_directory()) {
            continue;
        }
        const std::string python_dir_name = python_entry.path().filename().string();
        if (python_dir_name.rfind("python", 0) != 0) {
            continue;
        }

        const fs::path binaries_dir = python_entry.path() / "site-packages" / "imageio_ffmpeg" / "binaries";
        std::error_code bin_ec;
        if (!fs::is_directory(binaries_dir, bin_ec)) {
            continue;
        }
        for (const fs::directory_entry& ffmpeg_entry : fs::directory_iterator(binaries_dir, bin_ec)) {
            if (bin_ec) {
                break;
            }
            const std::string name = ffmpeg_entry.path().filename().string();
            if (ffmpeg_entry.is_regular_file() && name.rfind("ffmpeg-", 0) == 0) {
                return ffmpeg_entry.path().string();
            }
        }
    }
    return "";
}

static std::string find_ffmpeg_binary() {
    const char* configured = std::getenv("ED_FFMPEG");
    if (configured != nullptr && configured[0] != '\0') {
        return configured;
    }

    std::string bundled = find_imageio_ffmpeg_in_conda(std::getenv("CONDA_PREFIX"));
    if (!bundled.empty()) {
        return bundled;
    }

    return "ffmpeg";
}

struct TemporaryDirectory {
    fs::path path;
    bool keep = false;

    TemporaryDirectory() = default;
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    TemporaryDirectory(TemporaryDirectory&& other) noexcept : path(std::move(other.path)), keep(other.keep) {
        other.keep = true;
    }
    TemporaryDirectory& operator=(TemporaryDirectory&& other) noexcept {
        if (this != &other) {
            if (!keep && !path.empty()) {
                std::error_code error;
                fs::remove_all(path, error);
            }
            path = std::move(other.path);
            keep = other.keep;
            other.keep = true;
        }
        return *this;
    }

    ~TemporaryDirectory() {
        if (!keep && !path.empty()) {
            std::error_code error;
            fs::remove_all(path, error);
        }
    }
};

static bool make_temporary_directory(const std::string& prefix, TemporaryDirectory* directory) {
    if (directory == nullptr) {
        return false;
    }
    const fs::path base = fs::temp_directory_path();
    std::random_device random;
    for (int attempt = 0; attempt < 64; ++attempt) {
        const fs::path candidate = base / (prefix + std::to_string(static_cast<uint64_t>(random())) + "-" + std::to_string(attempt));
        std::error_code error;
        if (fs::create_directory(candidate, error)) {
            directory->path = candidate;
            return true;
        }
    }
    return false;
}

static bool extract_reference_video_frames(const std::string& path, const fs::path& frames_dir, int fps, int max_frames) {
    if (fps <= 0) {
        fps = 24;
    }
    const std::string pattern = (frames_dir / "frame_%06d.png").string();
    std::string command = shell_quote(find_ffmpeg_binary().c_str()) +
                          " -hide_banner -loglevel error -y -i " + shell_quote(path.c_str()) +
                          " -map 0:v:0 -an -vf " + shell_quote(("fps=" + std::to_string(fps)).c_str());
    if (max_frames > 0) {
        command += " -frames:v " + std::to_string(max_frames);
    }
    command += " " + shell_quote(pattern.c_str());
    const int status = std::system(command.c_str());
    if (status != 0) {
        std::fprintf(stderr, "ffmpeg failed while extracting reference video frames from '%s', status=%d\n", path.c_str(), status);
        return false;
    }
    return true;
}

static bool extract_reference_video_audio(const std::string& path, const fs::path& wav_path) {
    const std::string command = shell_quote(find_ffmpeg_binary().c_str()) +
                                " -hide_banner -loglevel quiet -y -i " + shell_quote(path.c_str()) +
                                " -map 0:a:0? -vn -ac 2 -ar 32000 -c:a pcm_s16le " + shell_quote(wav_path.c_str());
    const int status = std::system(command.c_str());
    if (status != 0) {
        return false;
    }
    std::error_code error;
    return fs::is_regular_file(wav_path, error) && fs::file_size(wav_path, error) > 0;
}

static bool load_reference_video(const std::string& path,
                                 int fps,
                                 int max_frames,
                                 std::vector<TemporaryDirectory>* temporary_directories,
                                 std::vector<ed_image_t>* frames,
                                 std::vector<float>* audio_samples,
                                 ed_ref_video_t* reference,
                                 bool explicit_audio_supplied) {
    if (frames == nullptr || audio_samples == nullptr || reference == nullptr) {
        return false;
    }
    if (fs::is_directory(path)) {
        if (!load_images_from_dir(path, frames)) {
            return false;
        }
        reference->frames = frames->data();
        reference->frame_count = static_cast<int>(frames->size());
        reference->fps = fps > 0 ? fps : 24;
        return true;
    }

    const std::string ext = path_extension(path);
    if (!is_supported_video_ext(ext)) {
        return false;
    }
    temporary_directories->emplace_back();
    TemporaryDirectory& directory = temporary_directories->back();
    if (!make_temporary_directory("ed-ref-video-", &directory)) {
        std::fprintf(stderr, "failed to create temporary directory for reference video '%s'\n", path.c_str());
        return false;
    }
    const fs::path frames_dir = directory.path / "frames";
    std::error_code error;
    fs::create_directory(frames_dir, error);
    if (error || !extract_reference_video_frames(path, frames_dir, fps, max_frames) || !load_images_from_dir(frames_dir.string(), frames)) {
        return false;
    }
    reference->frames = frames->data();
    reference->frame_count = static_cast<int>(frames->size());
    reference->fps = fps > 0 ? fps : 24;
    if (!explicit_audio_supplied) {
        const fs::path wav_path = directory.path / "audio.wav";
        if (extract_reference_video_audio(path, wav_path)) {
            if (!load_wav(wav_path.string(), audio_samples, &reference->audio)) {
                std::fprintf(stderr, "failed to load extracted reference video audio from '%s'\n", wav_path.c_str());
                return false;
            }
        }
    }
    return true;
}

static bool write_rgb_frame(FILE* pipe, const ed_image_t& image, std::vector<uint8_t>* scratch) {
    if (pipe == nullptr || image.data == nullptr || scratch == nullptr) {
        return false;
    }

    const size_t pixels = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
    if (image.channels == 3) {
        return std::fwrite(image.data, 1, pixels * 3, pipe) == pixels * 3;
    }

    if (image.channels != 1 && image.channels != 4) {
        std::fprintf(stderr, "unsupported video frame channel count: %u\n", image.channels);
        return false;
    }

    scratch->resize(pixels * 3);
    for (size_t i = 0; i < pixels; ++i) {
        if (image.channels == 1) {
            const uint8_t v = image.data[i];
            (*scratch)[i * 3 + 0] = v;
            (*scratch)[i * 3 + 1] = v;
            (*scratch)[i * 3 + 2] = v;
        } else {
            (*scratch)[i * 3 + 0] = image.data[i * 4 + 0];
            (*scratch)[i * 3 + 1] = image.data[i * 4 + 1];
            (*scratch)[i * 3 + 2] = image.data[i * 4 + 2];
        }
    }
    return std::fwrite(scratch->data(), 1, scratch->size(), pipe) == scratch->size();
}

static void write_u16_le(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
}

static void write_u32_le(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
}

static void patch_u32_le(std::vector<uint8_t>& out, size_t pos, uint32_t value) {
    out[pos + 0] = static_cast<uint8_t>(value & 0xff);
    out[pos + 1] = static_cast<uint8_t>((value >> 8) & 0xff);
    out[pos + 2] = static_cast<uint8_t>((value >> 16) & 0xff);
    out[pos + 3] = static_cast<uint8_t>((value >> 24) & 0xff);
}

static void write_fourcc(std::vector<uint8_t>& out, const char* fourcc) {
    out.insert(out.end(), fourcc, fourcc + 4);
}

static bool write_binary_file(const char* path, const std::vector<uint8_t>& data) {
    FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "failed to open output file: %s\n", path);
        return false;
    }
    const bool ok = std::fwrite(data.data(), 1, data.size(), file) == data.size();
    std::fclose(file);
    return ok;
}

static bool image_to_rgb(const ed_image_t& image, std::vector<uint8_t>* rgb) {
    if (image.data == nullptr || rgb == nullptr || image.width == 0 || image.height == 0) {
        return false;
    }

    const size_t pixels = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
    rgb->resize(pixels * 3);

    if (image.channels == 3) {
        std::memcpy(rgb->data(), image.data, rgb->size());
        return true;
    }
    if (image.channels == 4) {
        for (size_t i = 0; i < pixels; ++i) {
            (*rgb)[i * 3 + 0] = image.data[i * 4 + 0];
            (*rgb)[i * 3 + 1] = image.data[i * 4 + 1];
            (*rgb)[i * 3 + 2] = image.data[i * 4 + 2];
        }
        return true;
    }
    if (image.channels == 1) {
        for (size_t i = 0; i < pixels; ++i) {
            const uint8_t v = image.data[i];
            (*rgb)[i * 3 + 0] = v;
            (*rgb)[i * 3 + 1] = v;
            (*rgb)[i * 3 + 2] = v;
        }
        return true;
    }

    std::fprintf(stderr, "unsupported video frame channel count: %u\n", image.channels);
    return false;
}

struct AviIndexEntry {
    char fourcc[4];
    uint32_t flags = 0;
    uint32_t offset = 0;
    uint32_t size = 0;
};

static bool save_mjpg_avi(const char* path, const ed_video_t& video, int fps, int quality) {
    if (path == nullptr || video.frames == nullptr || video.frame_count <= 0 || fps <= 0) {
        return false;
    }

    const ed_image_t& first = video.frames[0];
    if (first.data == nullptr || first.width == 0 || first.height == 0) {
        return false;
    }

    const uint32_t width = first.width;
    const uint32_t height = first.height;
    const uint32_t frame_count = static_cast<uint32_t>(video.frame_count);
    const int jpg_quality = quality < 1 ? 1 : (quality > 100 ? 100 : quality);

    std::vector<uint8_t> avi;
    avi.reserve(static_cast<size_t>(width) * height * 3 * video.frame_count / 4);

    write_fourcc(avi, "RIFF");
    const size_t riff_size_pos = avi.size();
    write_u32_le(avi, 0);
    write_fourcc(avi, "AVI ");

    write_fourcc(avi, "LIST");
    write_u32_le(avi, 4 + 8 + 56 + 8 + 4 + 8 + 56 + 8 + 40);
    write_fourcc(avi, "hdrl");

    write_fourcc(avi, "avih");
    write_u32_le(avi, 56);
    write_u32_le(avi, static_cast<uint32_t>(1000000 / fps));
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0x110);
    write_u32_le(avi, frame_count);
    write_u32_le(avi, 0);
    write_u32_le(avi, 1);
    write_u32_le(avi, width * height * 3);
    write_u32_le(avi, width);
    write_u32_le(avi, height);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);

    write_fourcc(avi, "LIST");
    write_u32_le(avi, 4 + 8 + 56 + 8 + 40);
    write_fourcc(avi, "strl");

    write_fourcc(avi, "strh");
    write_u32_le(avi, 56);
    write_fourcc(avi, "vids");
    write_fourcc(avi, "MJPG");
    write_u32_le(avi, 0);
    write_u16_le(avi, 0);
    write_u16_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 1);
    write_u32_le(avi, static_cast<uint32_t>(fps));
    write_u32_le(avi, 0);
    write_u32_le(avi, frame_count);
    write_u32_le(avi, width * height * 3);
    write_u32_le(avi, 0xffffffffu);
    write_u32_le(avi, 0);
    write_u16_le(avi, 0);
    write_u16_le(avi, 0);
    write_u16_le(avi, 0);
    write_u16_le(avi, 0);

    write_fourcc(avi, "strf");
    write_u32_le(avi, 40);
    write_u32_le(avi, 40);
    write_u32_le(avi, width);
    write_u32_le(avi, height);
    write_u16_le(avi, 1);
    write_u16_le(avi, 24);
    write_fourcc(avi, "MJPG");
    write_u32_le(avi, width * height * 3);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);

    write_fourcc(avi, "LIST");
    const size_t movi_size_pos = avi.size();
    write_u32_le(avi, 0);
    write_fourcc(avi, "movi");

    std::vector<AviIndexEntry> index;
    index.reserve(static_cast<size_t>(video.frame_count));
    std::vector<uint8_t> rgb;
    std::vector<uint8_t> jpg;

    for (int i = 0; i < video.frame_count; ++i) {
        const ed_image_t& frame = video.frames[i];
        if (frame.width != width || frame.height != height || !image_to_rgb(frame, &rgb)) {
            std::fprintf(stderr, "video frame %d has invalid or inconsistent data\n", i);
            return false;
        }

        jpg.clear();
        auto write_jpg = [](void* context, void* data, int size) {
            auto* buffer = static_cast<std::vector<uint8_t>*>(context);
            const uint8_t* bytes = static_cast<const uint8_t*>(data);
            buffer->insert(buffer->end(), bytes, bytes + size);
        };
        if (!stbi_write_jpg_to_func(write_jpg,
                                    &jpg,
                                    static_cast<int>(width),
                                    static_cast<int>(height),
                                    3,
                                    rgb.data(),
                                    jpg_quality)) {
            std::fprintf(stderr, "failed to encode AVI frame %d as JPEG\n", i);
            return false;
        }

        AviIndexEntry entry{};
        std::memcpy(entry.fourcc, "00dc", 4);
        entry.flags = 0x10;
        // idx1 offsets are relative to the "movi" FourCC. movi_size_pos points at
        // the LIST size field, so the first media chunk starts four bytes later.
        entry.offset = static_cast<uint32_t>(avi.size() - (movi_size_pos + 4));
        entry.size = static_cast<uint32_t>(jpg.size());

        write_fourcc(avi, "00dc");
        write_u32_le(avi, entry.size);
        avi.insert(avi.end(), jpg.begin(), jpg.end());
        if (jpg.size() % 2 != 0) {
            avi.push_back(0);
        }
        index.push_back(entry);
    }

    patch_u32_le(avi, movi_size_pos, static_cast<uint32_t>(avi.size() - movi_size_pos - 4));

    write_fourcc(avi, "idx1");
    write_u32_le(avi, static_cast<uint32_t>(index.size() * 16));
    for (const AviIndexEntry& entry : index) {
        write_fourcc(avi, entry.fourcc);
        write_u32_le(avi, entry.flags);
        write_u32_le(avi, entry.offset);
        write_u32_le(avi, entry.size);
    }

    patch_u32_le(avi, riff_size_pos, static_cast<uint32_t>(avi.size() - riff_size_pos - 4));
    return write_binary_file(path, avi);
}

static bool save_ffmpeg_video(const char* path, const ed_video_t& video, int fps) {
    if (path == nullptr || video.frames == nullptr || video.frame_count <= 0) {
        return false;
    }
    if (fps <= 0) {
        fps = 16;
    }

    const ed_image_t& first = video.frames[0];
    if (first.data == nullptr || first.width == 0 || first.height == 0) {
        return false;
    }

    for (int i = 0; i < video.frame_count; ++i) {
        const ed_image_t& frame = video.frames[i];
        if (frame.data == nullptr || frame.width != first.width || frame.height != first.height) {
            std::fprintf(stderr, "video frame %d has inconsistent dimensions\n", i);
            return false;
        }
    }

    std::signal(SIGPIPE, SIG_IGN);

    char cmd[4096];
    const std::string ffmpeg_path = find_ffmpeg_binary();
    const std::string quoted_ffmpeg = shell_quote(ffmpeg_path.c_str());
    const std::string quoted_path = shell_quote(path);
    const std::string ext = path_extension(path);
    const char* codec_args = ext == ".webm"
                                 ? "-an -c:v libvpx-vp9 -crf 18 -b:v 0 -pix_fmt yuv420p"
                                 : "-an -c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p";
    const char* mux_args = ext == ".webm" ? "" : "-movflags +faststart";
    std::snprintf(cmd,
                  sizeof(cmd),
                  "%s -hide_banner -loglevel error -y "
                  "-f rawvideo -pix_fmt rgb24 -s %ux%u -r %d -i - "
                  "%s %s %s",
                  quoted_ffmpeg.c_str(),
                  first.width,
                  first.height,
                  fps,
                  codec_args,
                  mux_args,
                  quoted_path.c_str());

    FILE* pipe = popen(cmd, "w");
    if (pipe == nullptr) {
        std::fprintf(stderr, "failed to start ffmpeg: %s\n", std::strerror(errno));
        return false;
    }

    std::vector<uint8_t> scratch;
    bool ok = true;
    for (int i = 0; i < video.frame_count; ++i) {
        if (!write_rgb_frame(pipe, video.frames[i], &scratch)) {
            ok = false;
            break;
        }
    }

    const int status = pclose(pipe);
    if (!ok || status != 0) {
        if (status == 32512) {
            std::fprintf(stderr, "ffmpeg was not found; install ffmpeg, add it to PATH, or set ED_FFMPEG\n");
        } else {
            std::fprintf(stderr, "ffmpeg failed while writing video, status=%d\n", status);
        }
        return false;
    }
    return true;
}

static bool save_video(const char* path, const ed_video_t& video, int fps) {
    const std::string ext = path_extension(path != nullptr ? path : "");
    if (ext == ".avi") {
        return save_mjpg_avi(path, video, fps, 95);
    }
    if (is_ffmpeg_video_ext(ext)) {
        return save_ffmpeg_video(path, video, fps);
    }
    std::fprintf(stderr, "unsupported video extension: %s\n", ext.c_str());
    return false;
}

static bool save_wav(const char* path, const ed_video_t& video) {
    if (path == nullptr || video.audio == nullptr || video.audio_sample_count <= 0 ||
        video.audio_channels <= 0 || video.audio_sample_rate <= 0) {
        return false;
    }
    const uint32_t channels = static_cast<uint32_t>(video.audio_channels);
    const uint32_t sample_rate = static_cast<uint32_t>(video.audio_sample_rate);
    const uint32_t sample_count = static_cast<uint32_t>(video.audio_sample_count);
    const uint32_t data_size = sample_count * channels * sizeof(int16_t);
    FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        return false;
    }
    auto put_u16 = [&](uint16_t value) { return std::fwrite(&value, sizeof(value), 1, file) == 1; };
    auto put_u32 = [&](uint32_t value) { return std::fwrite(&value, sizeof(value), 1, file) == 1; };
    const bool header_ok = std::fwrite("RIFF", 1, 4, file) == 4 && put_u32(36 + data_size) &&
                           std::fwrite("WAVEfmt ", 1, 8, file) == 8 && put_u32(16) && put_u16(1) &&
                           put_u16(static_cast<uint16_t>(channels)) && put_u32(sample_rate) &&
                           put_u32(sample_rate * channels * sizeof(int16_t)) &&
                           put_u16(static_cast<uint16_t>(channels * sizeof(int16_t))) && put_u16(16) &&
                           std::fwrite("data", 1, 4, file) == 4 && put_u32(data_size);
    bool samples_ok = header_ok;
    for (uint32_t sample = 0; samples_ok && sample < sample_count; ++sample) {
        for (uint32_t channel = 0; channel < channels; ++channel) {
            const float value = std::clamp(video.audio[sample * channels + channel], -1.0f, 1.0f);
            const int16_t pcm = static_cast<int16_t>(std::lround(value * 32767.0f));
            samples_ok = std::fwrite(&pcm, sizeof(pcm), 1, file) == 1;
            if (!samples_ok) break;
        }
    }
    return std::fclose(file) == 0 && samples_ok;
}

static bool save_video_with_audio(const char* path, const ed_video_t& video, int fps, bool* audio_muxed) {
    if (audio_muxed != nullptr) {
        *audio_muxed = false;
    }
    if (video.audio == nullptr || video.audio_sample_count <= 0) {
        return save_video(path, video, fps);
    }
    const fs::path output(path);
    const std::string ext = path_extension(path);
    const fs::path video_tmp = output.string() + ".video.tmp" + (ext == ".avi" ? ".avi" : ext);
    const fs::path wav_path = output.string() + ".wav";
    if (!save_video(video_tmp.c_str(), video, fps) || !save_wav(wav_path.c_str(), video)) {
        std::error_code error;
        fs::remove(video_tmp, error);
        return false;
    }
    const char* audio_codec = ext == ".avi" ? "pcm_s16le" : (ext == ".webm" ? "libopus" : "aac");
    const std::string command = shell_quote(find_ffmpeg_binary().c_str()) +
                                " -hide_banner -loglevel error -y -i " + shell_quote(video_tmp.c_str()) +
                                " -i " + shell_quote(wav_path.c_str()) +
                                " -c:v copy -c:a " + audio_codec + " " + shell_quote(path);
    const int status = std::system(command.c_str());
    std::error_code error;
    if (status != 0) {
        fs::remove(output, error);
        fs::rename(video_tmp, output, error);
        if (error) {
            std::fprintf(stderr, "ffmpeg failed while muxing audio, status=%d; video remains at %s and WAV at %s\n",
                         status, video_tmp.c_str(), wav_path.c_str());
            return false;
        }
        std::fprintf(stderr, "ffmpeg failed while muxing audio, status=%d; saved AVI and WAV sidecar at %s\n",
                     status, wav_path.c_str());
        return true;
    }
    fs::remove(video_tmp, error);
    if (audio_muxed != nullptr) {
        *audio_muxed = true;
    }
    return true;
}

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

    std::string prompt_file_contents;
    if ((args.prompt == nullptr || args.prompt[0] == '\0') &&
        args.prompt_file != nullptr && args.prompt_file[0] != '\0') {
        std::ifstream prompt_file(args.prompt_file, std::ios::binary);
        if (!prompt_file.is_open()) {
            std::fprintf(stderr, "failed to open prompt file: %s\n", args.prompt_file);
            return 1;
        }
        std::ostringstream prompt_stream;
        prompt_stream << prompt_file.rdbuf();
        prompt_file_contents = prompt_stream.str();
        if (prompt_file_contents.empty()) {
            std::fprintf(stderr, "prompt file is empty: %s\n", args.prompt_file);
            return 1;
        }
        args.prompt = prompt_file_contents.c_str();
    }

    if (args.backend != nullptr && std::strlen(args.backend) > 0) {
        setenv("ED_BACKEND", args.backend, 1);
    }
    if (args.profile_graph_cuts) {
        setenv("ED_PROFILE_GRAPH_CUTS", "1", 1);
        std::string top = std::to_string(args.profile_graph_cuts_top);
        setenv("ED_PROFILE_GRAPH_CUTS_TOP", top.c_str(), 1);
        if (args.profile_graph_cuts_all_ranks) {
            setenv("ED_PROFILE_GRAPH_CUTS_ALL_RANKS", "1", 1);
        }
    }

    ed_context_params_t ctx_params;
    ed_context_params_init(&ctx_params);

    const std::string latent_upscaler_path = resolve_ltx_latent_upscaler_path(args);
    if (args.hires && latent_upscaler_path.empty()) {
        std::fprintf(stderr,
                     "--hires requires --latent-upscaler, or --hires-upscalers-dir with --hires-upscaler\n");
        return 1;
    }

    ctx_params.model_path = args.model_path;
    ctx_params.diffusion_model_path = args.diffusion_model_path;
    ctx_params.vae_path = args.vae_path;
    ctx_params.audio_vae_path = args.audio_vae_path;
    ctx_params.embeddings_connectors_path = args.embeddings_connectors_path;
    ctx_params.latent_upscaler_path = latent_upscaler_path.empty() ? nullptr : latent_upscaler_path.c_str();
    ctx_params.clip_l_path = args.clip_l_path;
    ctx_params.clip_g_path = args.clip_g_path;
    ctx_params.t5xxl_path = args.t5xxl_path;
    ctx_params.llm_path = args.llm_path;
    ctx_params.llm_vision_path = args.llm_vision_path;
    ctx_params.cfg_parallel_size = args.cfg_parallel_size;
    ctx_params.tp_parallel_size = args.tp_parallel_size;
    ctx_params.sp_parallel_size = args.sp_parallel_size;
    ctx_params.flash_attention = args.flash_attention;
    ctx_params.offload_params_to_cpu = args.offload_to_cpu;
    ctx_params.dit_offload = args.dit_offload;
    ctx_params.text_encoder_offload = args.text_encoder_offload;
    ctx_params.minimax_h3_stage_lifecycle = args.minimax_h3_stage_lifecycle;
    ctx_params.auto_allocate = args.auto_allocate;
    ctx_params.auto_fit = args.auto_fit;
    // For auto-allocate/auto-fit compute-buffer measurement: use the requested
    // generation size as the target resolution so the resident-headroom estimate
    // reflects the real activation footprint. Harmless when neither mode is on.
    ctx_params.fit_width = args.hires ? args.width * 2 : args.width;
    ctx_params.fit_height = args.hires ? args.height * 2 : args.height;
    ctx_params.fit_frames = resolve_video_fit_frames(args);
    ctx_params.fit_fps = args.fps;
    ctx_params.vae_offload = args.vae_offload;
    if (args.max_vram > 0.0f) {
        ctx_params.max_vram_gb = args.max_vram;
    }
    ctx_params.qwen_image_zero_cond_t = args.qwen_image_zero_cond_t;

    if (args.threads > 0) {
        ctx_params.n_threads = args.threads;
    }

    /*
     * During Flux bring-up, let the internals auto-detect dtype / sampler / scheduler first.
     * If your model is a quantized GGUF, you can also specify it manually here:
     *   ctx_params.weight_type = ED_DTYPE_Q8_0;
     * The command-line --type / --tensor-type-rules will override the defaults here,
     * and trigger online quantization when loading safetensors.
     */
    ctx_params.weight_type = args.weight_type != nullptr
                                 ? parse_weight_type(args.weight_type, nullptr)
                                 : ED_DTYPE_AUTO;
    ctx_params.tensor_type_rules = args.tensor_type_rules;
    ctx_params.skip_t5 = args.no_t5;
    if (args.vae_tiling == 1) {
        ctx_params.vae_tiling.enabled = true;
    } else if (args.vae_tiling == 0) {
        ctx_params.vae_tiling.force_disable = true;  // explicit off: suppress low-VRAM auto-enable
    }
    if (args.vae_tile_size > 0.0f) {
        ctx_params.vae_tiling.enabled = true;
        ctx_params.vae_tiling.rel_size_x = args.vae_tile_size;
        ctx_params.vae_tiling.rel_size_y = args.vae_tile_size;
    }

    using ed_wall_clock = std::chrono::steady_clock;
    auto ed_wall_ms = [](ed_wall_clock::time_point a, ed_wall_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double ed_wall_load = 0, ed_wall_gen = 0, ed_wall_save = 0, ed_wall_free = 0;

    auto ed_wall_t0 = ed_wall_clock::now();
    ed_context_t* ctx = ed_create_context(&ctx_params);
    ed_wall_load = ed_wall_ms(ed_wall_t0, ed_wall_clock::now());
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create edge-dit context\n");
        return 2;
    }

    if (args.video) {
        int output_fps = args.fps;
        const char* pipeline_name = ed_context_pipeline_name(ctx);
        const bool is_minimax_h3 = pipeline_name != nullptr && std::strcmp(pipeline_name, "minimax-h3") == 0;
        if (args.video_duration_explicit && !is_minimax_h3) {
            std::fprintf(stderr, "--video-duration is currently supported only by MiniMax-H3\n");
            ed_free_context(ctx);
            return 1;
        }
        if (is_minimax_h3 && output_fps != 24) {
            std::fprintf(stderr, "MiniMax-H3 uses 24 fps; overriding requested fps %d\n", output_fps);
            output_fps = 24;
        }
        const int generation_frames = resolve_video_frames(args, is_minimax_h3);
        if (args.video_duration_explicit) {
            std::fprintf(stderr,
                         "video duration %.3fs resolved to %d frames (%.3fs at %d fps)%s\n",
                         args.video_duration,
                         generation_frames,
                         static_cast<double>(generation_frames) / output_fps,
                         output_fps,
                         is_minimax_h3 ? "; MiniMax-H3 requires at least 22 frames satisfying 17k+5" : "");
        }
        ed_video_generation_params_t gen_params;
        ed_video_generation_params_init(&gen_params);
        ed_image_t init_image = {};
        ed_image_t end_image = {};
        std::vector<ed_image_t> ref_images(args.ref_image_paths.size());
        std::vector<std::vector<ed_image_t>> ref_video_frames(args.ref_video_paths.size());
        std::vector<std::vector<float>> ref_video_audio_samples(args.ref_video_paths.size());
        std::vector<ed_ref_video_t> ref_videos(args.ref_video_paths.size());
        std::vector<TemporaryDirectory> ref_video_temp_directories;
        std::vector<std::vector<float>> ref_audio_samples(args.ref_audio_paths.size());
        std::vector<ed_audio_t> ref_audios(args.ref_audio_paths.size());
        auto free_video_references = [&]() {
            for (auto& frames : ref_video_frames) {
                for (ed_image_t& frame : frames) {
                    ed_free_image(&frame);
                }
            }
        };

        gen_params.prompt = args.prompt;
        gen_params.negative_prompt = args.negative_prompt;
        gen_params.width = args.width;
        gen_params.height = args.height;
        gen_params.frames = generation_frames;
        gen_params.fps = output_fps;
        gen_params.hires_enabled = args.hires;
        gen_params.hires_steps = args.hires_steps;
        gen_params.hires_denoising_strength = args.hires_denoising_strength;
        gen_params.hires_sigmas = args.hires_sigmas.empty() ? nullptr : args.hires_sigmas.data();
        gen_params.hires_sigmas_count = static_cast<int>(args.hires_sigmas.size());
        gen_params.seed = args.seed;
        gen_params.ref_image_size = args.ref_image_size;
        gen_params.sample.sampler = args.sampler;
        gen_params.sample.scheduler = args.scheduler;
        gen_params.sample.steps = args.steps;
        gen_params.sample.cfg_scale = args.cfg_scale;
        gen_params.sample.image_cfg_scale = 1.0f;
        gen_params.sample.distilled_guidance = args.guidance;
        gen_params.sample.flow_shift = args.flow_shift;
        apply_cache_args(args, &gen_params.sample);
        if (args.image_path != nullptr && std::strlen(args.image_path) > 0) {
            if (!load_image(args.image_path, &init_image)) {
                ed_free_context(ctx);
                return 6;
            }
            gen_params.init_image = &init_image;
        }
        if (args.end_image_path != nullptr && std::strlen(args.end_image_path) > 0) {
            if (!load_image(args.end_image_path, &end_image)) {
                ed_free_image(&init_image);
                ed_free_context(ctx);
                return 6;
            }
            gen_params.end_image = &end_image;
        }
        for (size_t index = 0; index < args.ref_image_paths.size(); ++index) {
            if (!load_image(args.ref_image_paths[index].c_str(), &ref_images[index])) {
                for (ed_image_t& image : ref_images) ed_free_image(&image);
                ed_free_image(&init_image);
                ed_free_image(&end_image);
                ed_free_context(ctx);
                return 6;
            }
        }
        if (!ref_images.empty()) {
            gen_params.ref_images = ref_images.data();
            gen_params.ref_image_count = static_cast<int>(ref_images.size());
        }
        for (size_t index = 0; index < args.ref_audio_paths.size(); ++index) {
            if (!load_wav(args.ref_audio_paths[index], &ref_audio_samples[index], &ref_audios[index])) {
                std::fprintf(stderr, "failed to load reference WAV '%s'\n", args.ref_audio_paths[index].c_str());
                for (ed_image_t& image : ref_images) ed_free_image(&image);
                free_video_references();
                ed_free_image(&init_image); ed_free_image(&end_image); ed_free_context(ctx); return 6;
            }
        }
        if (!ref_audios.empty()) { gen_params.ref_audios = ref_audios.data(); gen_params.ref_audio_count = static_cast<int>(ref_audios.size()); }
        for (size_t index = 0; index < args.ref_video_paths.size(); ++index) {
            const bool explicit_audio_supplied = index < args.ref_video_audio_paths.size();
            if (!load_reference_video(args.ref_video_paths[index],
                                      output_fps,
                                      generation_frames,
                                      &ref_video_temp_directories,
                                      &ref_video_frames[index],
                                      &ref_video_audio_samples[index],
                                      &ref_videos[index],
                                      explicit_audio_supplied)) {
                std::fprintf(stderr, "failed to load reference video from '%s'\n", args.ref_video_paths[index].c_str());
                for (ed_image_t& image : ref_images) ed_free_image(&image);
                free_video_references();
                ed_free_image(&init_image); ed_free_image(&end_image); ed_free_context(ctx); return 6;
            }
            if (explicit_audio_supplied && !load_wav(args.ref_video_audio_paths[index], &ref_video_audio_samples[index], &ref_videos[index].audio)) {
                std::fprintf(stderr, "failed to load reference video WAV '%s'\n", args.ref_video_audio_paths[index].c_str());
                for (ed_image_t& image : ref_images) ed_free_image(&image);
                free_video_references();
                ed_free_image(&init_image); ed_free_image(&end_image); ed_free_context(ctx); return 6;
            }
        }
        if (!ref_videos.empty()) { gen_params.ref_videos = ref_videos.data(); gen_params.ref_video_count = static_cast<int>(ref_videos.size()); }

        ed_video_t output;
        auto ed_wall_gen0 = ed_wall_clock::now();
        ed_status_t status = ed_generate_video(ctx, &gen_params, &output);
        ed_wall_gen = ed_wall_ms(ed_wall_gen0, ed_wall_clock::now());
        if (status != ED_STATUS_OK) {
            std::fprintf(stderr, "ed_generate_video failed, status=%d\n", static_cast<int>(status));
            const char* err = ed_get_last_error(ctx);
            if (err != nullptr && std::strlen(err) > 0) {
                std::fprintf(stderr, "last error: %s\n", err);
            }
            ed_free_image(&init_image);
            ed_free_image(&end_image);
            for (ed_image_t& image : ref_images) ed_free_image(&image);
            free_video_references();
            ed_free_context(ctx);
            return 3;
        }

        if (!ed_context_parallel_is_root(ctx)) {
            ed_free_video(&output);
            free_video_references();
            for (ed_image_t& image : ref_images) ed_free_image(&image);
            ed_free_image(&init_image);
            ed_free_image(&end_image);
            ed_free_context(ctx);
            return 0;
        }

        if (output.frame_count <= 0 || output.frames == nullptr) {
            std::fprintf(stderr, "generation succeeded but video output is empty\n");
            free_video_references();
            for (ed_image_t& image : ref_images) ed_free_image(&image);
            ed_free_image(&init_image);
            ed_free_image(&end_image);
            ed_free_context(ctx);
            return 4;
        }

        const std::string output_path = video_output_path(args.output_path, args.video_format);
        auto ed_wall_save0 = ed_wall_clock::now();
        bool audio_muxed = false;
        bool ed_save_ok = save_video_with_audio(output_path.c_str(), output, output_fps, &audio_muxed);
        ed_wall_save = ed_wall_ms(ed_wall_save0, ed_wall_clock::now());
        if (!ed_save_ok) {
            std::fprintf(stderr, "failed to save output video: %s\n", output_path.c_str());
            ed_free_video(&output);
            free_video_references();
            for (ed_image_t& image : ref_images) ed_free_image(&image);
            ed_free_image(&init_image);
            ed_free_image(&end_image);
            ed_free_context(ctx);
            return 5;
        }

        if (output.audio != nullptr && output.audio_sample_count > 0 && !audio_muxed) {
            std::printf("saved video to %s with WAV sidecar %s.wav\n", output_path.c_str(), output_path.c_str());
        } else {
            std::printf("saved video%s to %s\n", audio_muxed ? " with audio" : "", output_path.c_str());
        }

        ed_free_video(&output);
        ed_free_image(&init_image);
        ed_free_image(&end_image);
        for (ed_image_t& image : ref_images) ed_free_image(&image);
        free_video_references();
    } else {
        ed_image_generation_params_t gen_params;
        ed_image_generation_params_init(&gen_params);
        ed_image_t input_image = {};
        bool has_input_image = false;
        if (args.image_path != nullptr && std::strlen(args.image_path) > 0) {
            if (!load_image(args.image_path, &input_image)) {
                ed_free_context(ctx);
                return 6;
            }
            has_input_image = true;
            gen_params.init_image = &input_image;
            gen_params.ref_images = &input_image;
            gen_params.ref_image_count = 1;
        }

        gen_params.prompt = args.prompt;
        gen_params.negative_prompt = args.negative_prompt;

        gen_params.width = args.width;
        gen_params.height = args.height;
        gen_params.seed = args.seed;
        gen_params.batch_count = 1;

        /*
         * Flux / DiT testing tips:
         * - use AUTO for sampler/scheduler, letting the internals pick based on the model
         * - set cfg_scale to 1.0 to avoid the traditional CFG negative-conditioning branch
         * - use the common Flux value of 3.5 for distilled_guidance
         */
        gen_params.sample.sampler = args.sampler;
        gen_params.sample.scheduler = args.scheduler;
        gen_params.sample.steps = args.steps;
        gen_params.sample.cfg_scale = args.cfg_scale;
        gen_params.sample.image_cfg_scale = 1.0f;
        gen_params.sample.distilled_guidance = args.guidance;
        gen_params.sample.flow_shift = args.flow_shift;
        apply_cache_args(args, &gen_params.sample);

        ed_image_batch_t output;
        ed_status_t status = ed_generate_image(ctx, &gen_params, &output);

        if (status != ED_STATUS_OK) {
            std::fprintf(stderr, "ed_generate_image failed, status=%d\n", static_cast<int>(status));

            const char* err = ed_get_last_error(ctx);
            if (err != nullptr && std::strlen(err) > 0) {
                std::fprintf(stderr, "last error: %s\n", err);
            }

            ed_free_context(ctx);
            if (has_input_image) {
                ed_free_image(&input_image);
            }
            return 3;
        }

        if (!ed_context_parallel_is_root(ctx)) {
            ed_free_image_batch(&output);
            ed_free_context(ctx);
            if (has_input_image) {
                ed_free_image(&input_image);
            }
            return 0;
        }

        if (output.count <= 0 || output.images == nullptr) {
            std::fprintf(stderr, "generation succeeded but output is empty\n");
            ed_free_context(ctx);
            if (has_input_image) {
                ed_free_image(&input_image);
            }
            return 4;
        }

        if (!save_png(args.output_path, output.images[0])) {
            std::fprintf(stderr, "failed to save output image: %s\n", args.output_path);
            ed_free_image_batch(&output);
            ed_free_context(ctx);
            if (has_input_image) {
                ed_free_image(&input_image);
            }
            return 5;
        }

        std::printf("saved image to %s\n", args.output_path);

        ed_free_image_batch(&output);
        if (has_input_image) {
            ed_free_image(&input_image);
        }
    }
    auto ed_wall_free0 = ed_wall_clock::now();
    ed_free_context(ctx);
    ed_wall_free = ed_wall_ms(ed_wall_free0, ed_wall_clock::now());

    std::printf("[ED_WALL] load=%.0fms gen=%.0fms save=%.0fms free=%.0fms (sum=%.0fms)\n",
                ed_wall_load, ed_wall_gen, ed_wall_save, ed_wall_free,
                ed_wall_load + ed_wall_gen + ed_wall_save + ed_wall_free);

    return 0;
}
