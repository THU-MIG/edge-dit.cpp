#include "utils/util.h"

#include <algorithm>
#include <codecvt>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <locale>
#include <sstream>
#include <thread>

#include "utils/tensor.hpp"

bool ends_with(const std::string& str, const std::string& ending) {
    return str.size() >= ending.size() &&
           str.compare(str.size() - ending.size(), ending.size(), ending) == 0;
}

bool starts_with(const std::string& str, const std::string& start) {
    return str.size() >= start.size() && str.compare(0, start.size(), start) == 0;
}

bool contains(const std::string& str, const std::string& substr) {
    return str.find(substr) != std::string::npos;
}

std::string sd_format(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    const int needed = std::vsnprintf(nullptr, 0, fmt, args_copy);
    va_end(args_copy);

    std::string result;
    if (needed > 0) {
        result.resize(static_cast<size_t>(needed));
        std::vsnprintf(result.data(), result.size() + 1, fmt, args);
    }
    va_end(args);
    return result;
}

std::u32string utf8_to_utf32(const std::string& utf8_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.from_bytes(utf8_str);
}

std::string utf32_to_utf8(const std::u32string& utf32_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.to_bytes(utf32_str);
}

std::u32string unicode_value_to_utf32(int unicode_value) {
    return {static_cast<char32_t>(unicode_value)};
}

std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
    // Minimal parser for the first Flux path. Weight syntax can be expanded later;
    // returning one segment keeps the tokenizer contract identical.
    if (text.empty()) {
        return {};
    }
    return {{text, 1.0f}};
}

sd::Tensor<float> clip_preprocess(const sd::Tensor<float>& image, int target_width, int target_height) {
    if (image.empty() || target_width <= 0 || target_height <= 0 || image.dim() < 3) {
        return {};
    }

    const int64_t src_c = image.shape()[2];
    const int64_t frames = image.dim() >= 4 ? image.shape()[3] : 1;
    if (src_c < 3) {
        return {};
    }

    sd::Tensor<float> resized = image;
    if (image.shape()[0] != target_width || image.shape()[1] != target_height) {
        resized = sd::ops::interpolate(image,
                                       {target_width, target_height, image.shape()[2], frames},
                                       sd::ops::InterpolateMode::Bicubic,
                                       false,
                                       true);
    }

    static constexpr float mean[3] = {0.48145466f, 0.45782750f, 0.40821073f};
    static constexpr float std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
    sd::Tensor<float> output({target_width, target_height, 3, frames});
    for (int64_t f = 0; f < frames; ++f) {
        for (int y = 0; y < target_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                for (int64_t c = 0; c < 3; ++c) {
                    output.index(x, y, c, f) = (resized.index(x, y, c, f) - mean[c]) / std[c];
                }
            }
        }
    }
    return output;
}

void replace_all_chars(std::string& str, char target, char replacement) {
    std::replace(str.begin(), str.end(), target, replacement);
}

int round_up_to(int value, int base) {
    if (base <= 0) {
        return value;
    }
    return ((value + base - 1) / base) * base;
}

int ed_get_num_physical_cores() {
    const unsigned int n = std::thread::hardware_concurrency();
    return n == 0 ? 4 : static_cast<int>(n);
}

bool file_exists(const std::string& filename) {
    std::error_code ec;
    return std::filesystem::is_regular_file(filename, ec);
}

bool is_directory(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

std::string path_join(const std::string& p1, const std::string& p2) {
    if (p1.empty()) {
        return p2;
    }
    if (p2.empty()) {
        return p1;
    }
    return (std::filesystem::path(p1) / p2).string();
}

std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        result.push_back(item);
    }
    return result;
}

void pretty_bytes_progress(int step, int steps, uint64_t bytes_processed, float elapsed_seconds) {
    if (steps <= 0) {
        return;
    }
    const double mb = static_cast<double>(bytes_processed) / (1024.0 * 1024.0);
    std::fprintf(stderr, "\rloading tensors %d/%d %.2fMB %.2fs", step, steps, mb, elapsed_seconds);
    if (step >= steps) {
        std::fprintf(stderr, "\n");
    }
}

void pretty_progress(int step, int steps, float time) {
    if (steps <= 0 || step == 0) {
        return;
    }
    const char* unit = "s/it";
    float speed = time;
    if (speed < 1.0f && speed > 0.0f) {
        speed = 1.0f / speed;
        unit = "it/s";
    }
    std::fprintf(stderr, "\rprogress %d/%d %.2f%s", step, steps, speed, unit);
    if (step >= steps) {
        std::fprintf(stderr, "\n");
    }
}

void log_printf(ed_log_level_t level, const char* file, int line, const char* format, ...) {
    const char* level_name = "debug";
    switch (level) {
        case ED_LOG_INFO: level_name = "info"; break;
        case ED_LOG_WARN: level_name = "warn"; break;
        case ED_LOG_ERROR: level_name = "error"; break;
        case ED_LOG_DEBUG:
        default: break;
    }

    std::fprintf(stderr, "%s:%d [%s] ", file, line, level_name);
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

std::string trim(const std::string& s) {
    const char* ws = " \t\n\r\f\v";
    const size_t begin = s.find_first_not_of(ws);
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = s.find_last_not_of(ws);
    return s.substr(begin, end - begin + 1);
}

std::unique_ptr<MmapWrapper> MmapWrapper::create(const std::string& filename) {
    static constexpr size_t max_fallback_mmap_size = 256ull * 1024ull * 1024ull;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size < 0) {
        return nullptr;
    }
    if (static_cast<uint64_t>(size) > max_fallback_mmap_size) {
        return nullptr;
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!data.empty()) {
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!file) {
            return nullptr;
        }
    }
    return std::unique_ptr<MmapWrapper>(new MmapWrapper(std::move(data)));
}

bool MmapWrapper::copy_data(void* buf, size_t n, size_t offset) const {
    if (buf == nullptr || offset > data_.size() || n > data_.size() - offset) {
        return false;
    }
    std::memcpy(buf, data_.data() + offset, n);
    return true;
}

bool sd_backend_is(ggml_backend_t backend, const std::string& name) {
    if (backend == nullptr) {
        return false;
    }
    const char* backend_name = ggml_backend_name(backend);
    return backend_name != nullptr && contains(backend_name, name);
}
