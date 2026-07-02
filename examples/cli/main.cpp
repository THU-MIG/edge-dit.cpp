#include "edge-dit.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <algorithm>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define STB_IMAGE_IMPLEMENTATION
#include "ggml/examples/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --model <model-or-diffusers-dir> --prompt <text> [options]\n"
        "  %s --diffusion-model <path> --vae <path> --clip_l <path> [--clip_g <path>] (--t5xxl <path> | --no-t5) --prompt <text> [options]\n"
        "Options:\n"
        "  --video                   Generate video frames instead of an image\n"
        "  --video-format <fmt>      Video format: auto, avi, mp4, mov, mkv, webm. Default: auto\n"
        "  -i, --image <path>        Input image for image-edit models such as Qwen-Image-Edit\n"
        "  --diffusion-model <path>  Standalone DiT transformer weights\n"
        "  --vae <path>              Standalone VAE weights\n"
        "  --clip_l <path>           CLIP-L text encoder weights\n"
        "  --clip_g <path>           CLIP-G text encoder weights\n"
        "  --t5xxl <path>            T5XXL text encoder weights\n"
        "  --negative-prompt <text>  Negative prompt text, default: empty\n"
        "  -o, --output <path>       Output image/video path, default: output.png\n"
        "  -W, --width <int>         Image width, default: 1024\n"
        "  -H, --height <int>        Image height, default: 1024\n"
        "  --frames <int>            Video frame count, default: 1\n"
        "  --fps <int>               Video fps, default: 16\n"
        "  --steps <int>             Sampling steps, default: 20\n"
        "  -s, --seed <int64>        Seed, default: -1\n"
        "  -t, --threads <int>       Thread count, default: 0\n"
        "  --guidance <float>        Flux distilled guidance, default: 3.5\n"
        "  --cfg-scale <float>       Classifier-free guidance scale, default: 1.0\n"
        "  --flow-shift <float>      Flow scheduler shift, default: model default\n"
        "  --cache <mode>            Cache mode: off, easycache, ucache, dbcache, taylorseer, cache-dit\n"
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
        "                            DBCache residual diff threshold, default: 0.08\n"
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
        "  --backend <name>          Backend: auto, cpu, cuda, gpu. Default: auto\n"
        "  --gpu                     Alias for --backend gpu\n"
        "  --devices <csv>           GPU devices for parallel workers, e.g. 0,1,2,3\n"
        "  --type <dtype>            Weight type / on-the-fly quantization when loading safetensors:\n"
        "                            f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k. Default: auto\n"
        "  --tensor-type-rules <csv> Per-tensor quant overrides (mixed quant), e.g. \"attn=q4_0,norm=f16\"\n"
        "                            Each rule is <name-regex>=<ggml-type-name>, comma-separated\n"
        "  --no-t5                   Skip loading T5XXL text encoder (SD3 only; reduces memory, degrades prompt adherence)\n"
        "  --vae-tiling              Enable VAE tiled decode (reduces VRAM, default: off)\n"
        "  --vae-tile-size <float>   VAE tile relative size, default: 2.0 (2x2 grid). Larger = finer tiles = less VRAM\n"
        "  --offload-to-cpu          Keep model weights on CPU, copy to GPU per-compute (saves VRAM)\n"
        "  --keep-text-encoder-on-cpu  Run text encoder on CPU backend\n"
        "  --keep-vae-on-cpu         Run VAE on CPU backend\n"
        "  --max-vram <GB>           Limit VRAM usage for compute graphs (e.g. 8.0)\n"
        "  --flash-attention         Enable flash attention, default: on\n"
        "  --no-flash-attention      Disable flash attention\n"
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
        prog
    );
}

static bool save_png(const char* path, const ed_image_t& image) {
    if (path == nullptr || image.data == nullptr) {
        return false;
    }

    if (image.channels == 0 || image.channels > 4) {
        std::fprintf(stderr, "unsupported channel count: %u\n", image.channels);
        return false;
    }

    return stbi_write_png(
        path,
        static_cast<int>(image.width),
        static_cast<int>(image.height),
        static_cast<int>(image.channels),
        image.data,
        0,
        nullptr
    ) != 0;
}

static bool load_image(const char* path, ed_image_t* image) {
    if (path == nullptr || image == nullptr) {
        return false;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = stbi_load(path, &width, &height, &channels, 3);
    if (data == nullptr || width <= 0 || height <= 0) {
        std::fprintf(stderr, "failed to load input image '%s': %s\n",
                     path,
                     stbi_failure_reason() != nullptr ? stbi_failure_reason() : "unknown error");
        if (data != nullptr) {
            stbi_image_free(data);
        }
        return false;
    }

    const size_t nbytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    uint8_t* owned = static_cast<uint8_t*>(std::malloc(nbytes));
    if (owned == nullptr) {
        stbi_image_free(data);
        std::fprintf(stderr, "failed to allocate input image buffer\n");
        return false;
    }
    std::memcpy(owned, data, nbytes);
    stbi_image_free(data);

    image->width = static_cast<uint32_t>(width);
    image->height = static_cast<uint32_t>(height);
    image->channels = 3;
    image->data = owned;
    return true;
}

static std::string shell_quote(const char* value) {
    std::string quoted = "'";
    const char* text = value != nullptr ? value : "";
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p == '\'') {
            quoted += "'\\''";
        } else {
            quoted += *p;
        }
    }
    quoted += "'";
    return quoted;
}

static std::string lowercase(std::string value) {
    for (char& c : value) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return value;
}

static int env_int_value(const char* name, int fallback = 0) {
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

static bool is_distributed_process() {
    int world_size = env_int_value("WORLD_SIZE", 1);
    world_size = env_int_value("OMPI_COMM_WORLD_SIZE", world_size);
    world_size = env_int_value("MV2_COMM_WORLD_SIZE", world_size);
    world_size = env_int_value("SLURM_NTASKS", world_size);
    world_size = env_int_value("PMI_SIZE", world_size);
    return world_size > 1;
}

static int count_csv_values(const char* csv) {
    if (csv == nullptr || csv[0] == '\0') {
        return 0;
    }

    int count = 0;
    bool in_value = false;
    for (const char* p = csv; *p != '\0'; ++p) {
        if (*p == ',') {
            if (in_value) {
                ++count;
                in_value = false;
            }
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(*p))) {
            in_value = true;
        }
    }
    return in_value ? count + 1 : count;
}

static std::vector<std::string> split_csv_values(const char* csv) {
    std::vector<std::string> values;
    if (csv == nullptr || csv[0] == '\0') {
        return values;
    }

    std::string current;
    auto flush = [&]() {
        size_t begin = 0;
        while (begin < current.size() &&
               std::isspace(static_cast<unsigned char>(current[begin]))) {
            ++begin;
        }
        size_t end = current.size();
        while (end > begin &&
               std::isspace(static_cast<unsigned char>(current[end - 1]))) {
            --end;
        }
        if (end > begin) {
            values.emplace_back(current.substr(begin, end - begin));
        }
        current.clear();
    };

    for (const char* p = csv; *p != '\0'; ++p) {
        if (*p == ',') {
            flush();
        } else {
            current.push_back(*p);
        }
    }
    flush();
    return values;
}

static bool is_cpu_backend_name(const char* backend) {
    return lowercase(backend != nullptr ? backend : "") == "cpu";
}

static int parse_int_value(const char* text, int fallback = 0) {
    if (text == nullptr) {
        return fallback;
    }

    while (std::isspace(static_cast<unsigned char>(*text))) {
        ++text;
    }

    bool negative = false;
    if (*text == '+' || *text == '-') {
        negative = *text == '-';
        ++text;
    }

    int value = 0;
    bool any = false;
    while (*text >= '0' && *text <= '9') {
        any = true;
        value = value * 10 + (*text - '0');
        ++text;
    }
    if (!any) {
        return fallback;
    }
    return negative ? -value : value;
}

static int64_t parse_i64_value(const char* text, int64_t fallback = 0) {
    if (text == nullptr) {
        return fallback;
    }

    while (std::isspace(static_cast<unsigned char>(*text))) {
        ++text;
    }

    bool negative = false;
    if (*text == '+' || *text == '-') {
        negative = *text == '-';
        ++text;
    }

    int64_t value = 0;
    bool any = false;
    while (*text >= '0' && *text <= '9') {
        any = true;
        value = value * 10 + (*text - '0');
        ++text;
    }
    if (!any) {
        return fallback;
    }
    return negative ? -value : value;
}

static float parse_float_value(const char* text, float fallback = 0.0f) {
    if (text == nullptr) {
        return fallback;
    }

    bool negative = false;
    while (std::isspace(static_cast<unsigned char>(*text))) {
        ++text;
    }
    if (*text == '+' || *text == '-') {
        negative = *text == '-';
        ++text;
    }

    double value = 0.0;
    bool any = false;
    while (*text >= '0' && *text <= '9') {
        any = true;
        value = value * 10.0 + static_cast<double>(*text - '0');
        ++text;
    }

    if (*text == '.') {
        ++text;
        double scale = 0.1;
        while (*text >= '0' && *text <= '9') {
            any = true;
            value += static_cast<double>(*text - '0') * scale;
            scale *= 0.1;
            ++text;
        }
    }

    if (!any) {
        return fallback;
    }

    if (*text == 'e' || *text == 'E') {
        ++text;
        const int exp = parse_int_value(text, 0);
        value *= std::pow(10.0, static_cast<double>(exp));
    }

    if (negative) {
        value = -value;
    }
    return static_cast<float>(value);
}

static ed_cache_mode_t parse_cache_mode(const char* text, bool* ok) {
    if (ok != nullptr) {
        *ok = true;
    }
    std::string mode = lowercase(text != nullptr ? text : "off");
    for (char& c : mode) {
        if (c == '_' || c == '.') {
            c = '-';
        }
    }

    if (mode == "off" || mode == "none" || mode == "disabled" || mode == "disable" || mode == "0") {
        return ED_CACHE_DISABLED;
    }
    if (mode == "easycache" || mode == "easy") {
        return ED_CACHE_EASYCACHE;
    }
    if (mode == "ucache" || mode == "u") {
        return ED_CACHE_UCACHE;
    }
    if (mode == "dbcache" || mode == "db") {
        return ED_CACHE_DBCACHE;
    }
    if (mode == "taylorseer" || mode == "taylor" || mode == "taylor-seer") {
        return ED_CACHE_TAYLORSEER;
    }
    if (mode == "cache-dit" || mode == "cachedit") {
        return ED_CACHE_CACHE_DIT;
    }

    if (ok != nullptr) {
        *ok = false;
    }
    return ED_CACHE_DISABLED;
}

static ed_dtype_t parse_weight_type(const char* text, bool* ok) {
    if (ok != nullptr) {
        *ok = true;
    }
    std::string type = lowercase(text != nullptr ? text : "");
    for (char& c : type) {
        if (c == '-' || c == '.') {
            c = '_';
        }
    }

    if (type == "auto" || type.empty()) {
        return ED_DTYPE_AUTO;
    }
    if (type == "f32" || type == "fp32") {
        return ED_DTYPE_F32;
    }
    if (type == "f16" || type == "fp16") {
        return ED_DTYPE_F16;
    }
    if (type == "bf16") {
        return ED_DTYPE_BF16;
    }
    if (type == "q4_0") {
        return ED_DTYPE_Q4_0;
    }
    if (type == "q4_1") {
        return ED_DTYPE_Q4_1;
    }
    if (type == "q5_0") {
        return ED_DTYPE_Q5_0;
    }
    if (type == "q5_1") {
        return ED_DTYPE_Q5_1;
    }
    if (type == "q8_0") {
        return ED_DTYPE_Q8_0;
    }
    if (type == "q2_k") {
        return ED_DTYPE_Q2_K;
    }
    if (type == "q3_k") {
        return ED_DTYPE_Q3_K;
    }
    if (type == "q4_k") {
        return ED_DTYPE_Q4_K;
    }
    if (type == "q5_k") {
        return ED_DTYPE_Q5_K;
    }
    if (type == "q6_k") {
        return ED_DTYPE_Q6_K;
    }

    if (ok != nullptr) {
        *ok = false;
    }
    return ED_DTYPE_AUTO;
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

static std::string normalized_video_format(const char* format) {
    std::string value = format != nullptr && format[0] != '\0' ? format : "auto";
    value = lowercase(value);
    if (!value.empty() && value[0] == '.') {
        value.erase(value.begin());
    }
    return value;
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

    bundled = find_imageio_ffmpeg_in_conda("/export/home/liuyiming54/miniconda3/envs/hicache");
    if (!bundled.empty()) {
        return bundled;
    }

    return "ffmpeg";
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
        entry.offset = static_cast<uint32_t>(avi.size());
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

struct FluxCliArgs {
    const char* model_path = nullptr;
    const char* diffusion_model_path = nullptr;
    const char* vae_path = nullptr;
    const char* clip_l_path = nullptr;
    const char* clip_g_path = nullptr;
    const char* t5xxl_path = nullptr;
    const char* prompt = nullptr;
    const char* negative_prompt = nullptr;
    const char* image_path = nullptr;
    const char* output_path = "output.png";
    const char* video_format = nullptr;
    const char* backend = nullptr;
    const char* devices = nullptr;
    const char* weight_type = nullptr;
    const char* tensor_type_rules = nullptr;
    int cfg_parallel_size = 1;
    int tp_parallel_size = 1;
    int sp_parallel_size = 1;
    int profile_graph_cuts_top = 8;

    bool video = false;
    bool profile_graph_cuts = false;
    bool profile_graph_cuts_all_ranks = false;
    bool flash_attention = true;
    int width = 1024;
    int height = 1024;
    int frames = 1;
    int fps = 16;
    int steps = 20;
    int threads = 0;

    int64_t seed = -1;
    float guidance = 3.5f;
    float cfg_scale = 1.0f;
    float flow_shift = 0.0f;

    ed_cache_mode_t cache_mode = ED_CACHE_DISABLED;
    float cache_reuse_threshold = std::numeric_limits<float>::infinity();
    float cache_start_percent = 0.15f;
    float cache_end_percent = 0.95f;
    float cache_error_decay_rate = 1.0f;
    bool cache_use_relative_threshold = true;
    bool cache_reset_error_on_compute = true;
    int cache_Fn_compute_blocks = 8;
    int cache_Bn_compute_blocks = 0;
    float cache_residual_diff_threshold = 0.08f;
    float cache_max_accumulated_residual_diff = -1.0f;
    int cache_max_warmup_steps = 8;
    int cache_max_cached_steps = -1;
    int cache_max_continuous_cached_steps = -1;
    int cache_taylorseer_n_derivatives = 1;
    int cache_taylorseer_skip_interval = 1;
    const char* cache_scm_mask = nullptr;
    bool cache_scm_policy_dynamic = true;
    bool no_t5 = false;
    int vae_tiling = -1;  // -1=default(off), 1=on
    float vae_tile_size = 0.0f;
    bool offload_to_cpu = false;
    bool keep_text_encoder_on_cpu = false;
    bool keep_vae_on_cpu = false;
    float max_vram = 0.0f;
};

static bool parse_args(int argc, char** argv, FluxCliArgs* args) {
    if (args == nullptr) {
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];

        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(key, "--video") == 0) {
            args->video = true;
        } else if (std::strcmp(key, "--video-format") == 0) {
            args->video_format = require_value(key);
        } else if (std::strcmp(key, "--model") == 0) {
            args->model_path = require_value(key);
        } else if (std::strcmp(key, "--diffusion-model") == 0) {
            args->diffusion_model_path = require_value(key);
        } else if (std::strcmp(key, "--vae") == 0) {
            args->vae_path = require_value(key);
        } else if (std::strcmp(key, "--clip_l") == 0) {
            args->clip_l_path = require_value(key);
        } else if (std::strcmp(key, "--clip_g") == 0) {
            args->clip_g_path = require_value(key);
        } else if (std::strcmp(key, "--t5xxl") == 0) {
            args->t5xxl_path = require_value(key);
        } else if (std::strcmp(key, "--prompt") == 0 || std::strcmp(key, "-p") == 0) {
            args->prompt = require_value(key);
        } else if (std::strcmp(key, "--output") == 0 || std::strcmp(key, "-o") == 0) {
            args->output_path = require_value(key);
        } else if (std::strcmp(key, "--width") == 0 || std::strcmp(key, "-W") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->width = parse_int_value(v, args->width);
        } else if (std::strcmp(key, "--height") == 0 || std::strcmp(key, "-H") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->height = parse_int_value(v, args->height);
        } else if (std::strcmp(key, "--frames") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->frames = parse_int_value(v, args->frames);
        } else if (std::strcmp(key, "--fps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->fps = parse_int_value(v, args->fps);
        } else if (std::strcmp(key, "--steps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->steps = parse_int_value(v, args->steps);
        } else if (std::strcmp(key, "--threads") == 0 || std::strcmp(key, "-t") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->threads = parse_int_value(v, args->threads);
        } else if (std::strcmp(key, "--seed") == 0 || std::strcmp(key, "-s") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->seed = parse_i64_value(v, args->seed);
        } else if (std::strcmp(key, "--guidance") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->guidance = parse_float_value(v, args->guidance);
        } else if (std::strcmp(key, "--cfg-scale") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cfg_scale = parse_float_value(v, args->cfg_scale);
        } else if (std::strcmp(key, "--flow-shift") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->flow_shift = parse_float_value(v, args->flow_shift);
        } else if (std::strcmp(key, "--cache") == 0 || std::strcmp(key, "--cache-mode") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            bool ok = false;
            args->cache_mode = parse_cache_mode(v, &ok);
            if (!ok) {
                std::fprintf(stderr, "unsupported cache mode: %s\n", v);
                return false;
            }
        } else if (std::strcmp(key, "--cache-threshold") == 0 ||
                   std::strcmp(key, "--cache-reuse-threshold") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_reuse_threshold = parse_float_value(v, args->cache_reuse_threshold);
        } else if (std::strcmp(key, "--cache-start") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_start_percent = parse_float_value(v, args->cache_start_percent);
        } else if (std::strcmp(key, "--cache-end") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_end_percent = parse_float_value(v, args->cache_end_percent);
        } else if (std::strcmp(key, "--cache-error-decay") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_error_decay_rate = parse_float_value(v, args->cache_error_decay_rate);
        } else if (std::strcmp(key, "--cache-relative-threshold") == 0) {
            args->cache_use_relative_threshold = true;
        } else if (std::strcmp(key, "--cache-absolute-threshold") == 0) {
            args->cache_use_relative_threshold = false;
        } else if (std::strcmp(key, "--cache-no-reset-error") == 0) {
            args->cache_reset_error_on_compute = false;
        } else if (std::strcmp(key, "--cache-reset-error") == 0) {
            args->cache_reset_error_on_compute = true;
        } else if (std::strcmp(key, "--cache-fn-blocks") == 0 ||
                   std::strcmp(key, "--cache-Fn-compute-blocks") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_Fn_compute_blocks = parse_int_value(v, args->cache_Fn_compute_blocks);
        } else if (std::strcmp(key, "--cache-bn-blocks") == 0 ||
                   std::strcmp(key, "--cache-Bn-compute-blocks") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_Bn_compute_blocks = parse_int_value(v, args->cache_Bn_compute_blocks);
        } else if (std::strcmp(key, "--cache-residual-threshold") == 0 ||
                   std::strcmp(key, "--cache-residual-diff-threshold") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_residual_diff_threshold = parse_float_value(v, args->cache_residual_diff_threshold);
        } else if (std::strcmp(key, "--cache-max-accumulated-residual-diff") == 0 ||
                   std::strcmp(key, "--cache-max-accumulated-residual-diff-threshold") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_max_accumulated_residual_diff =
                parse_float_value(v, args->cache_max_accumulated_residual_diff);
        } else if (std::strcmp(key, "--cache-warmup-steps") == 0 ||
                   std::strcmp(key, "--cache-max-warmup-steps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_max_warmup_steps = parse_int_value(v, args->cache_max_warmup_steps);
        } else if (std::strcmp(key, "--cache-max-cached-steps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_max_cached_steps = parse_int_value(v, args->cache_max_cached_steps);
        } else if (std::strcmp(key, "--cache-max-continuous-cached-steps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_max_continuous_cached_steps = parse_int_value(v, args->cache_max_continuous_cached_steps);
        } else if (std::strcmp(key, "--cache-taylor-order") == 0 ||
                   std::strcmp(key, "--cache-taylorseer-n-derivatives") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_taylorseer_n_derivatives = parse_int_value(v, args->cache_taylorseer_n_derivatives);
        } else if (std::strcmp(key, "--cache-taylor-skip") == 0 ||
                   std::strcmp(key, "--cache-taylorseer-skip-interval") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cache_taylorseer_skip_interval = parse_int_value(v, args->cache_taylorseer_skip_interval);
        } else if (std::strcmp(key, "--cache-scm-mask") == 0) {
            args->cache_scm_mask = require_value(key);
        } else if (std::strcmp(key, "--cache-static-scm") == 0) {
            args->cache_scm_policy_dynamic = false;
        } else if (std::strcmp(key, "--cache-dynamic-scm") == 0) {
            args->cache_scm_policy_dynamic = true;
        } else if (std::strcmp(key, "--backend") == 0) {
            args->backend = require_value(key);
        } else if (std::strcmp(key, "--gpu") == 0) {
            args->backend = "gpu";
        } else if (std::strcmp(key, "--devices") == 0 ||
                   std::strcmp(key, "--gpus") == 0) {
            args->devices = require_value(key);
        } else if (std::strcmp(key, "--type") == 0 ||
                   std::strcmp(key, "--weight-type") == 0) {
            args->weight_type = require_value(key);
        } else if (std::strcmp(key, "--tensor-type-rules") == 0) {
            args->tensor_type_rules = require_value(key);
        } else if (std::strcmp(key, "--no-t5") == 0) {
            args->no_t5 = true;
        } else if (std::strcmp(key, "--vae-tiling") == 0) {
            args->vae_tiling = 1;
        } else if (std::strcmp(key, "--vae-tile-size") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->vae_tile_size = parse_float_value(v, 0.0f);
        } else if (std::strcmp(key, "--offload-to-cpu") == 0) {
            args->offload_to_cpu = true;
        } else if (std::strcmp(key, "--keep-text-encoder-on-cpu") == 0) {
            args->keep_text_encoder_on_cpu = true;
        } else if (std::strcmp(key, "--keep-vae-on-cpu") == 0) {
            args->keep_vae_on_cpu = true;
        } else if (std::strcmp(key, "--max-vram") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->max_vram = parse_float_value(v, 0.0f);
        } else if (std::strcmp(key, "--flash-attention") == 0 ||
                   std::strcmp(key, "--flash-attn") == 0) {
            args->flash_attention = true;
        } else if (std::strcmp(key, "--no-flash-attention") == 0 ||
                   std::strcmp(key, "--no-flash-attn") == 0) {
            args->flash_attention = false;
        } else if (std::strcmp(key, "--cfg-parallel-size") == 0 ||
                   std::strcmp(key, "--cfg-size") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->cfg_parallel_size = parse_int_value(v, args->cfg_parallel_size);
        } else if (std::strcmp(key, "--tp-size") == 0 ||
                   std::strcmp(key, "--tensor-parallel-size") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->tp_parallel_size = parse_int_value(v, args->tp_parallel_size);
        } else if (std::strcmp(key, "--sp-size") == 0 ||
                   std::strcmp(key, "--sequence-parallel-size") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->sp_parallel_size = parse_int_value(v, args->sp_parallel_size);
        } else if (std::strcmp(key, "--profile-graph-cuts") == 0) {
            args->profile_graph_cuts = true;
        } else if (std::strcmp(key, "--profile-graph-cuts-top") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->profile_graph_cuts_top = parse_int_value(v, args->profile_graph_cuts_top);
        } else if (std::strcmp(key, "--profile-graph-cuts-all-ranks") == 0) {
            args->profile_graph_cuts = true;
            args->profile_graph_cuts_all_ranks = true;
        } else if (std::strcmp(key, "--help") == 0 || std::strcmp(key, "-h") == 0) {
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", key);
            return false;
        }
    }

    const bool has_full_model = args->model_path != nullptr && std::strlen(args->model_path) > 0;
    const bool has_components =
        args->diffusion_model_path != nullptr && std::strlen(args->diffusion_model_path) > 0 &&
        args->vae_path != nullptr && std::strlen(args->vae_path) > 0 &&
        args->clip_l_path != nullptr && std::strlen(args->clip_l_path) > 0 &&
        (args->no_t5 || (args->t5xxl_path != nullptr && std::strlen(args->t5xxl_path) > 0));

    if (!has_full_model && !has_components) {
        std::fprintf(stderr, "--model or the full --diffusion-model/--vae/--clip_l/(--t5xxl or --no-t5) set is required\n");
        return false;
    }

    if (args->prompt == nullptr || std::strlen(args->prompt) == 0) {
        std::fprintf(stderr, "--prompt is required\n");
        return false;
    }

    if (args->width <= 0 || args->height <= 0) {
        std::fprintf(stderr, "width and height must be positive\n");
        return false;
    }

    if (args->frames <= 0) {
        std::fprintf(stderr, "frames must be positive\n");
        return false;
    }

    if (args->fps <= 0) {
        std::fprintf(stderr, "fps must be positive\n");
        return false;
    }

    const std::string video_format = normalized_video_format(args->video_format);
    if (video_format != "auto" &&
        video_format != "avi" &&
        video_format != "mp4" &&
        video_format != "mov" &&
        video_format != "mkv" &&
        video_format != "webm") {
        std::fprintf(stderr, "unsupported video format: %s\n", video_format.c_str());
        return false;
    }

    if (args->steps <= 0) {
        std::fprintf(stderr, "steps must be positive\n");
        return false;
    }

    if (args->cache_start_percent < 0.0f || args->cache_start_percent > 1.0f ||
        args->cache_end_percent < 0.0f || args->cache_end_percent > 1.0f ||
        args->cache_start_percent >= args->cache_end_percent) {
        std::fprintf(stderr, "cache window must satisfy 0 <= start < end <= 1\n");
        return false;
    }

    if (args->cache_reuse_threshold < 0.0f && !std::isinf(args->cache_reuse_threshold)) {
        std::fprintf(stderr, "cache threshold must be non-negative\n");
        return false;
    }

    if (args->cache_error_decay_rate < 0.0f || args->cache_error_decay_rate > 1.0f) {
        std::fprintf(stderr, "cache error decay must be in [0, 1]\n");
        return false;
    }

    if (args->cache_Fn_compute_blocks < 0 || args->cache_Bn_compute_blocks < 0) {
        std::fprintf(stderr, "cache block counts must be non-negative\n");
        return false;
    }

    if (args->cache_residual_diff_threshold < 0.0f) {
        std::fprintf(stderr, "cache residual threshold must be non-negative\n");
        return false;
    }

    if (args->cache_max_accumulated_residual_diff < -1.0f) {
        std::fprintf(stderr, "cache max accumulated residual diff must be >= -1\n");
        return false;
    }

    if (args->cache_max_warmup_steps < 0) {
        std::fprintf(stderr, "cache warmup steps must be non-negative\n");
        return false;
    }

    if (args->cache_taylorseer_n_derivatives < 1) {
        std::fprintf(stderr, "cache Taylor order must be >= 1\n");
        return false;
    }

    if (args->cache_taylorseer_skip_interval < 0) {
        std::fprintf(stderr, "cache Taylor skip interval must be non-negative\n");
        return false;
    }

    if (args->cfg_parallel_size != 1 && args->cfg_parallel_size != 2) {
        std::fprintf(stderr, "cfg parallel size currently supports 1 or 2\n");
        return false;
    }

    if (args->tp_parallel_size != 1) {
        std::fprintf(stderr, "tensor parallel is not implemented yet; use --tp-size 1\n");
        return false;
    }

    if (args->sp_parallel_size <= 0) {
        std::fprintf(stderr, "--sp-size must be positive\n");
        return false;
    }

    if (args->profile_graph_cuts_top < 0) {
        std::fprintf(stderr, "--profile-graph-cuts-top must be non-negative\n");
        return false;
    }

    if (args->weight_type != nullptr) {
        bool ok = false;
        parse_weight_type(args->weight_type, &ok);
        if (!ok) {
            std::fprintf(stderr, "unsupported weight type: %s\n", args->weight_type);
            return false;
        }
    }

    return true;
}

static void apply_cache_args(const FluxCliArgs& args, ed_sample_params_t* sample) {
    if (sample == nullptr) {
        return;
    }

    sample->cache_mode = args.cache_mode;
    sample->cache_reuse_threshold = args.cache_reuse_threshold;
    sample->cache_start_percent = args.cache_start_percent;
    sample->cache_end_percent = args.cache_end_percent;
    sample->cache_error_decay_rate = args.cache_error_decay_rate;
    sample->cache_use_relative_threshold = args.cache_use_relative_threshold;
    sample->cache_reset_error_on_compute = args.cache_reset_error_on_compute;
    sample->cache_Fn_compute_blocks = args.cache_Fn_compute_blocks;
    sample->cache_Bn_compute_blocks = args.cache_Bn_compute_blocks;
    sample->cache_residual_diff_threshold = args.cache_residual_diff_threshold;
    sample->cache_max_accumulated_residual_diff = args.cache_max_accumulated_residual_diff;
    sample->cache_max_warmup_steps = args.cache_max_warmup_steps;
    sample->cache_max_cached_steps = args.cache_max_cached_steps;
    sample->cache_max_continuous_cached_steps = args.cache_max_continuous_cached_steps;
    sample->cache_taylorseer_n_derivatives = args.cache_taylorseer_n_derivatives;
    sample->cache_taylorseer_skip_interval = args.cache_taylorseer_skip_interval;
    sample->cache_scm_mask = args.cache_scm_mask;
    sample->cache_scm_policy_dynamic = args.cache_scm_policy_dynamic;
}

static int requested_parallel_size(const FluxCliArgs& args) {
    return std::max({1, args.cfg_parallel_size, args.tp_parallel_size, args.sp_parallel_size});
}

static int launch_distributed_cli(int argc, char** argv, const FluxCliArgs& args, int device_count) {
    const int parallel_size = requested_parallel_size(args);
    if (parallel_size <= 1 || is_distributed_process()) {
        return -1;
    }

    if (args.devices == nullptr || std::strlen(args.devices) == 0) {
        std::fprintf(stderr, "parallel generation requires --devices with at least %d GPU ids\n", parallel_size);
        return 1;
    }
    if (device_count != parallel_size) {
        std::fprintf(stderr,
                     "parallel size (%d) must match --devices count (%d)\n",
                     parallel_size,
                     device_count);
        return 1;
    }

    const std::vector<std::string> devices = split_csv_values(args.devices);
    if (static_cast<int>(devices.size()) != parallel_size) {
        std::fprintf(stderr,
                     "parallel size (%d) must match parsed --devices count (%zu)\n",
                     parallel_size,
                     devices.size());
        return 1;
    }

    std::string argv_cmd;
    for (int i = 0; i < argc; ++i) {
        argv_cmd += " ";
        argv_cmd += shell_quote(argv[i]);
    }

    std::string cmd = "mpirun";
    for (int rank = 0; rank < parallel_size; ++rank) {
        if (rank > 0) {
            cmd += " :";
        }
        cmd += " -np 1 env ED_CLI_LAUNCHED=1 ED_CLI_SINGLE_VISIBLE_DEVICE=1 CUDA_VISIBLE_DEVICES=";
        cmd += shell_quote(devices[static_cast<size_t>(rank)].c_str());
        cmd += argv_cmd;
    }

    std::fprintf(stderr,
                 "launching %d parallel workers on devices %s\n",
                 parallel_size,
                 args.devices);
    const int status = std::system(cmd.c_str());
    if (status == -1) {
        std::fprintf(stderr, "failed to launch distributed workers with mpirun\n");
        return 127;
    }
    return status == 0 ? 0 : 1;
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

    if (args.threads > 0) {
        ctx_params.n_threads = args.threads;
    }

    /*
     * Flux 测试阶段先让内部自动识别 dtype / sampler / scheduler。
     * 如果你的模型是量化 GGUF，也可以在这里手动指定：
     *   ctx_params.weight_type = ED_DTYPE_Q8_0;
     * 命令行 --type / --tensor-type-rules 会覆盖这里的默认值，
     * 并在加载 safetensors 时触发在线量化。
     */
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

    ed_context_t* ctx = ed_create_context(&ctx_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create edge-dit context\n");
        return 2;
    }

    if (args.video) {
        ed_video_generation_params_t gen_params;
        ed_video_generation_params_init(&gen_params);

        gen_params.prompt = args.prompt;
        gen_params.negative_prompt = args.negative_prompt;
        gen_params.width = args.width;
        gen_params.height = args.height;
        gen_params.frames = args.frames;
        gen_params.seed = args.seed;
        gen_params.sample.sampler = ED_SAMPLER_AUTO;
        gen_params.sample.scheduler = ED_SCHEDULER_AUTO;
        gen_params.sample.steps = args.steps;
        gen_params.sample.cfg_scale = args.cfg_scale;
        gen_params.sample.image_cfg_scale = 1.0f;
        gen_params.sample.distilled_guidance = args.guidance;
        gen_params.sample.flow_shift = args.flow_shift;
        apply_cache_args(args, &gen_params.sample);

        ed_video_t output;
        ed_status_t status = ed_generate_video(ctx, &gen_params, &output);
        if (status != ED_STATUS_OK) {
            std::fprintf(stderr, "ed_generate_video failed, status=%d\n", static_cast<int>(status));
            const char* err = ed_get_last_error(ctx);
            if (err != nullptr && std::strlen(err) > 0) {
                std::fprintf(stderr, "last error: %s\n", err);
            }
            ed_free_context(ctx);
            return 3;
        }

        if (!ed_context_parallel_is_root(ctx)) {
            ed_free_video(&output);
            ed_free_context(ctx);
            return 0;
        }

        if (output.frame_count <= 0 || output.frames == nullptr) {
            std::fprintf(stderr, "generation succeeded but video output is empty\n");
            ed_free_context(ctx);
            return 4;
        }

        const std::string output_path = video_output_path(args.output_path, args.video_format);
        if (!save_video(output_path.c_str(), output, args.fps)) {
            std::fprintf(stderr, "failed to save output video: %s\n", output_path.c_str());
            ed_free_video(&output);
            ed_free_context(ctx);
            return 5;
        }

        std::printf("saved video to %s\n", output_path.c_str());

        ed_free_video(&output);
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
        }

        gen_params.prompt = args.prompt;
        gen_params.negative_prompt = args.negative_prompt;

        gen_params.width = args.width;
        gen_params.height = args.height;
        gen_params.seed = args.seed;
        gen_params.batch_count = 1;

        /*
         * Flux / DiT 测试建议：
         * - sampler/scheduler 用 AUTO，让内部根据模型选择
         * - cfg_scale 设为 1.0，避免传统 CFG 负条件分支
         * - distilled_guidance 用 Flux 常见值 3.5
         */
        gen_params.sample.sampler = ED_SAMPLER_AUTO;
        gen_params.sample.scheduler = ED_SCHEDULER_AUTO;
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
    ed_free_context(ctx);

    return 0;
}
