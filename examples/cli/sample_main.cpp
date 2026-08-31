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

uint16_t read_le16(const uint8_t* data) { return uint16_t(data[0]) | (uint16_t(data[1]) << 8); }
uint32_t read_le32(const uint8_t* data) { return uint32_t(data[0]) | (uint32_t(data[1]) << 8) | (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24); }

bool load_wav(const std::string& path, std::vector<float>* samples, ed_audio_t* audio) {
    std::ifstream file(path, std::ios::binary); uint8_t header[12];
    if (!file.read(reinterpret_cast<char*>(header), 12) || std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4)) return false;
    uint16_t format = 0, channels = 0, bits = 0; uint32_t rate = 0, data_size = 0; std::streampos data_pos = -1;
    while (file.good()) { uint8_t chunk[8]; if (!file.read(reinterpret_cast<char*>(chunk), 8)) break; uint32_t size = read_le32(chunk + 4); auto pos = file.tellg();
        if (!std::memcmp(chunk, "fmt ", 4)) { std::vector<uint8_t> v(size); if (!file.read(reinterpret_cast<char*>(v.data()), size) || size < 16) return false; format=read_le16(v.data()); channels=read_le16(v.data()+2); rate=read_le32(v.data()+4); bits=read_le16(v.data()+14); }
        else if (!std::memcmp(chunk, "data", 4)) { data_pos=pos; data_size=size; file.seekg(size, std::ios::cur); } else file.seekg(size, std::ios::cur); if (size & 1) file.seekg(1, std::ios::cur);
    }
    if (data_pos == std::streampos(-1) || channels == 0 || rate == 0 || !((format == 1 && bits == 16) || (format == 3 && bits == 32))) return false;
    const size_t bytes_per = bits / 8, values = data_size / bytes_per; std::vector<uint8_t> raw(data_size); file.clear(); file.seekg(data_pos); if (!file.read(reinterpret_cast<char*>(raw.data()), data_size)) return false; samples->resize(values);
    for (size_t i=0;i<values;++i) { if (format == 3) std::memcpy(&(*samples)[i], raw.data()+i*4, 4); else (*samples)[i]=float(int16_t(read_le16(raw.data()+i*2)))/32768.f; }
    *audio = {rate, channels, values / channels, samples->data()}; return true;
}

bool save_wav(const std::string& path, const ed_video_t& video) {
    if (!video.audio || video.audio_sample_count <= 0 || video.audio_channels <= 0 || video.audio_sample_rate <= 0) return true;
    std::ofstream out(path, std::ios::binary); if (!out) return false; const uint32_t values=video.audio_sample_count*video.audio_channels, data_size=values*2, riff_size=36+data_size;
    auto w16=[&](uint16_t v){ char b[2]={char(v),char(v>>8)}; out.write(b,2); }; auto w32=[&](uint32_t v){ char b[4]={char(v),char(v>>8),char(v>>16),char(v>>24)}; out.write(b,4); };
    out.write("RIFF",4); w32(riff_size); out.write("WAVEfmt ",8); w32(16); w16(1); w16(video.audio_channels); w32(video.audio_sample_rate); w32(video.audio_sample_rate*video.audio_channels*2); w16(video.audio_channels*2); w16(16); out.write("data",4); w32(data_size);
    for (uint32_t i=0;i<values;++i) w16(uint16_t(int16_t(std::lround(std::clamp(video.audio[i],-1.f,1.f)*32767.f)))); return bool(out);
}

bool load_frame_directory(const std::string& path, std::vector<ed_image_t>* frames) {
    if (!fs::is_directory(path)) return false; std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(path)) if (entry.is_regular_file()) { std::string ext=entry.path().extension().string(); std::transform(ext.begin(),ext.end(),ext.begin(),::tolower); if (ext==".png"||ext==".jpg"||ext==".jpeg"||ext==".bmp"||ext==".webp") files.push_back(entry.path()); }
    std::sort(files.begin(),files.end()); frames->resize(files.size()); for(size_t i=0;i<files.size();++i) if(!load_image(files[i].c_str(),&(*frames)[i])) return false; return !frames->empty();
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "ed-sample: single-run text-to-image benchmark generator.\n\n"
        "Usage:\n"
        "  %s --model <diffusers-dir> --prompt_file <txt> --output_dir <dir> [options]\n\n"
        "Required:\n"
        "  --model <path>            Diffusers model directory (or component flags:\n"
        "                            --diffusion-model/--vae/--clip_l/[--clip_g]/(--t5xxl|--no-t5),\n"
        "                            or video --diffusion-model/--vae/[--audio-vae]/--llm)\n"
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
        "  --sampler <name>          Sampling method: auto, euler, res_multistep\n"
        "  --scheduler <name>        Sigma scheduler: auto, discrete, simple, ltx2\n"
        "  --negative_prompt <text>  Negative prompt (used when cfg_scale != 1). Default empty\n"
        "  --image <path>            Input image or MiniMax-H3/LTX-2.3 first frame\n"
        "  --end-img <path>          MiniMax-H3/LTX-2.3 last frame\n"
        "  --embeddings-connectors <path>  LTX-2.3 text connector weights\n"
        "  --latent-upscaler <path>  LTX-2.3 spatial latent x2 upscaler weights\n"
        "  --hires                   Run LTX-2.3 x2 latent upscale and refine pass\n"
        "  --hires-upscalers-dir <dir>  Directory used with --hires-upscaler\n"
        "  --hires-upscaler <name>  LTX upscaler name/path\n"
        "  --hires-steps <int>       LTX hires refine steps. Default 4\n"
        "  --hires-denoising-strength <float>  LTX hires strength. Default 0.7\n"
        "  --hires-sigmas <csv>      Explicit LTX hires sigma schedule\n"
        "  --ref-image <path>        MiniMax-H3 Ref2VA image; repeatable\n"
        "  --ref-image-size <mode>   max or match\n"
        "  --ref-video <dir>         Ref2VA numbered frame directory; repeatable\n"
        "  --ref-video-audio <wav>   WAV paired by position with --ref-video\n"
        "  --ref-audio <wav>         Extra Ref2VA audio; repeatable\n"
        "  --video, -M vid_gen       Generate video frames (calls ed_generate_video; writes .avi)\n"
        "  --frames, --video-frames <int>  Video frame count (with --video). Default 1\n"
        "  --video-duration <seconds>  MiniMax-H3 duration; aligns to >=22 frames satisfying 17k+5\n"
        "  --fps <int>               Video fps (MiniMax-H3 is forced to 24). Default 16\n"
        "  --start_index <int>       First prompt index (inclusive). Default 0\n"
        "  --end_index <int>         Last prompt index (exclusive). Default all\n"
        "  --threads <int>           CPU thread count. Default auto\n"
        "  --backend <name>          auto|cpu|cuda|vulkan|metal|gpu. Default auto\n"
        "  --type <dtype>            Global weight policy. Default preserve ('auto' is an alias)\n"
        "  --no-flash-attention      Disable flash attention\n\n"
        "Memory / offload:\n"
        "  --offload-to-cpu          Full offload: all weights on CPU, staged to GPU per compute\n"
        "  --dit-offload             DiT weights on CPU, staged to GPU per step (compute on GPU)\n"
        "  --text-encoder-offload    Text-encoder weights on CPU, staged to GPU per encode\n"
        "  --minimax-h3-stage-lifecycle  MiniMax-H3: stage Qwen/VAE by phase and release them during DiT\n"
        "  --vae-offload             VAE weights on CPU, staged to GPU per decode\n"
        "  --max-vram <GB>           Compute-graph budget; with --auto-fit on single-device CUDA, also a guarded ceiling\n"
        "  --auto-allocate           Auto per-component placement under a VRAM planning budget\n"
        "  --auto-fit                Replan TE/DiT quant + placement; --type still controls VAE precision\n"
        "  --vae-tiling <on|off|auto>  VAE tiled decode (reduces VRAM)\n\n"
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

// ---- Video (MJPG AVI) output. edge-dit benchmarks write .avi; the container is
//      built in-memory here so no ffmpeg dependency is needed (mirrors ed-cli). ----
void write_u16_le(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
}

void write_u32_le(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
}

void patch_u32_le(std::vector<uint8_t>& out, size_t pos, uint32_t value) {
    out[pos + 0] = static_cast<uint8_t>(value & 0xff);
    out[pos + 1] = static_cast<uint8_t>((value >> 8) & 0xff);
    out[pos + 2] = static_cast<uint8_t>((value >> 16) & 0xff);
    out[pos + 3] = static_cast<uint8_t>((value >> 24) & 0xff);
}

void write_fourcc(std::vector<uint8_t>& out, const char* fourcc) {
    out.insert(out.end(), fourcc, fourcc + 4);
}

bool image_to_rgb(const ed_image_t& image, std::vector<uint8_t>* rgb) {
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

bool save_mjpg_avi(const char* path, const ed_video_t& video, int fps, int quality) {
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
        if (!stbi_write_jpg_to_func(write_jpg, &jpg,
                                    static_cast<int>(width), static_cast<int>(height),
                                    3, rgb.data(), jpg_quality)) {
            std::fprintf(stderr, "failed to encode AVI frame %d as JPEG\n", i);
            return false;
        }
        AviIndexEntry entry{};
        std::memcpy(entry.fourcc, "00dc", 4);
        entry.flags = 0x10;
        entry.offset = static_cast<uint32_t>(avi.size() - (movi_size_pos + 8));
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

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        std::fprintf(stderr, "failed to open output video file: %s\n", path);
        return false;
    }
    out.write(reinterpret_cast<const char*>(avi.data()), static_cast<std::streamsize>(avi.size()));
    return out.good();
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

    const auto load_t0 = std::chrono::steady_clock::now();
    ed_context_t* ctx = ed_create_context(&ctx_params);
    const auto load_t1 = std::chrono::steady_clock::now();
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create edge-dit context\n");
        return 2;
    }
    const double load_seconds = std::chrono::duration<double>(load_t1 - load_t0).count();
    const bool is_root = ed_context_parallel_is_root(ctx);
    int output_fps = args.fps;
    const char* pipeline_name = ed_context_pipeline_name(ctx);
    const bool is_minimax_h3 = pipeline_name != nullptr && std::strcmp(pipeline_name, "minimax-h3") == 0;
    if (args.video && args.video_duration_explicit && !is_minimax_h3) {
        std::fprintf(stderr, "--video-duration is currently supported only by MiniMax-H3\n");
        ed_free_context(ctx);
        return 1;
    }
    if (args.video && is_minimax_h3 && output_fps != 24) {
        std::fprintf(stderr, "MiniMax-H3 uses 24 fps; overriding requested fps %d\n", output_fps);
        output_fps = 24;
    }
    const int generation_frames = resolve_video_frames(args, is_minimax_h3);
    if (args.video && args.video_duration_explicit) {
        std::fprintf(stderr,
                     "video duration %.3fs resolved to %d frames (%.3fs at %d fps)%s\n",
                     args.video_duration,
                     generation_frames,
                     static_cast<double>(generation_frames) / output_fps,
                     output_fps,
                     is_minimax_h3 ? "; MiniMax-H3 requires at least 22 frames satisfying 17k+5" : "");
    }

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

    ed_image_t input_image = {};
    ed_image_t end_image = {};
    bool has_input_image = false;
    bool has_end_image = false;
    if (args.image_path != nullptr && std::strlen(args.image_path) > 0) {
        if (!load_image(args.image_path, &input_image)) {
            ed_free_context(ctx);
            return 6;
        }
        has_input_image = true;
    }
    if (args.end_image_path != nullptr && std::strlen(args.end_image_path) > 0) {
        if (!load_image(args.end_image_path, &end_image)) { if (has_input_image) ed_free_image(&input_image); ed_free_context(ctx); return 6; }
        has_end_image = true;
    }
    std::vector<ed_image_t> ref_images(args.ref_image_paths.size());
    std::vector<std::vector<ed_image_t>> ref_video_frames(args.ref_video_paths.size());
    std::vector<std::vector<float>> ref_video_audio_data(args.ref_video_paths.size());
    std::vector<ed_ref_video_t> ref_videos(args.ref_video_paths.size());
    std::vector<std::vector<float>> ref_audio_data(args.ref_audio_paths.size());
    std::vector<ed_audio_t> ref_audios(args.ref_audio_paths.size());
    auto free_inputs = [&]() {
        if (has_input_image) ed_free_image(&input_image);
        if (has_end_image) ed_free_image(&end_image);
        for (auto& image : ref_images) ed_free_image(&image);
        for (auto& frames : ref_video_frames) for (auto& frame : frames) ed_free_image(&frame);
    };
    for (size_t i=0;i<ref_images.size();++i) if (!load_image(args.ref_image_paths[i].c_str(), &ref_images[i])) { free_inputs(); ed_free_context(ctx); return 6; }
    for (size_t i=0;i<ref_audios.size();++i) if (!load_wav(args.ref_audio_paths[i], &ref_audio_data[i], &ref_audios[i])) { std::fprintf(stderr,"failed to load reference WAV '%s'\n",args.ref_audio_paths[i].c_str()); free_inputs(); ed_free_context(ctx); return 6; }
    for (size_t i=0;i<ref_videos.size();++i) {
        if (!load_frame_directory(args.ref_video_paths[i], &ref_video_frames[i])) { std::fprintf(stderr,"ed-sample --ref-video currently expects a numbered frame directory: %s\n",args.ref_video_paths[i].c_str()); free_inputs(); ed_free_context(ctx); return 6; }
        ref_videos[i].frames=ref_video_frames[i].data(); ref_videos[i].frame_count=static_cast<int>(ref_video_frames[i].size()); ref_videos[i].fps=output_fps;
        if (i < args.ref_video_audio_paths.size() && !load_wav(args.ref_video_audio_paths[i], &ref_video_audio_data[i], &ref_videos[i].audio)) { std::fprintf(stderr,"failed to load reference-video WAV '%s'\n",args.ref_video_audio_paths[i].c_str()); free_inputs(); ed_free_context(ctx); return 6; }
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

            // Timing is identical for image and video: the model is loaded once
            // (outside this loop) and each generation call is tightly wrapped, so
            // video shares the exact same measurement boundary as t2i/edit
            // (load excluded, output encoding excluded).
            double seconds = 0.0;
            ed_image_batch_t output = {};
            ed_video_t video_output = {};

            if (args.video) {
                ed_video_generation_params_t vgen;
                ed_video_generation_params_init(&vgen);
                vgen.prompt = prompt.c_str();
                vgen.negative_prompt = args.negative_prompt;
                if (has_input_image) {
                    vgen.init_image = &input_image;
                }
                if (has_end_image) vgen.end_image = &end_image;
                if (!ref_images.empty()) { vgen.ref_images=ref_images.data(); vgen.ref_image_count=static_cast<int>(ref_images.size()); vgen.ref_image_size=args.ref_image_size; }
                if (!ref_videos.empty()) { vgen.ref_videos=ref_videos.data(); vgen.ref_video_count=static_cast<int>(ref_videos.size()); }
                if (!ref_audios.empty()) { vgen.ref_audios=ref_audios.data(); vgen.ref_audio_count=static_cast<int>(ref_audios.size()); }
                vgen.width = args.width;
                vgen.height = args.height;
                vgen.frames = generation_frames;
                vgen.fps = output_fps;
                vgen.hires_enabled = args.hires;
                vgen.hires_steps = args.hires_steps;
                vgen.hires_denoising_strength = args.hires_denoising_strength;
                vgen.hires_sigmas = args.hires_sigmas.empty() ? nullptr : args.hires_sigmas.data();
                vgen.hires_sigmas_count = static_cast<int>(args.hires_sigmas.size());
                vgen.seed = seed;
                vgen.sample.sampler = args.sampler;
                vgen.sample.scheduler = args.scheduler;
                vgen.sample.steps = args.steps;
                vgen.sample.cfg_scale = args.cfg_scale;
                vgen.sample.image_cfg_scale = 1.0f;
                vgen.sample.distilled_guidance = args.guidance;
                vgen.sample.flow_shift = args.flow_shift;
                apply_cache_args(args, &vgen.sample);

                const auto t0 = std::chrono::steady_clock::now();
                ed_status_t status = ed_generate_video(ctx, &vgen, &video_output);
                const auto t1 = std::chrono::steady_clock::now();
                if (status != ED_STATUS_OK) {
                    std::fprintf(stderr, "ed_generate_video failed for prompt %d, status=%d\n",
                                 idx, static_cast<int>(status));
                    const char* err = ed_get_last_error(ctx);
                    if (err != nullptr && std::strlen(err) > 0) {
                        std::fprintf(stderr, "last error: %s\n", err);
                    }
                    ed_free_video(&video_output);
                    rc = 3;
                    break;
                }
                seconds = std::chrono::duration<double>(t1 - t0).count();
            } else {
                ed_image_generation_params_t gen;
                ed_image_generation_params_init(&gen);
                gen.prompt = prompt.c_str();
                gen.negative_prompt = args.negative_prompt;
                if (has_input_image) {
                    gen.init_image = &input_image;
                    gen.ref_images = &input_image;
                    gen.ref_image_count = 1;
                }
                gen.width = args.width;
                gen.height = args.height;
                gen.seed = seed;
                gen.batch_count = 1;
                gen.sample.sampler = args.sampler;
                gen.sample.scheduler = args.scheduler;
                gen.sample.steps = args.steps;
                gen.sample.cfg_scale = args.cfg_scale;
                gen.sample.image_cfg_scale = 1.0f;
                gen.sample.distilled_guidance = args.guidance;
                gen.sample.flow_shift = args.flow_shift;
                apply_cache_args(args, &gen.sample);

                const auto t0 = std::chrono::steady_clock::now();
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
                seconds = std::chrono::duration<double>(t1 - t0).count();
            }

            if (is_warmup) {
                ++warmed;
            } else {
                samples.push_back(seconds);
            }

            // Non-root ranks participate in generation but do not own the output.
            if (!is_root) {
                ed_free_image_batch(&output);
                ed_free_video(&video_output);
                continue;
            }

            char fname[32];
            const fs::path out_path = out_root / "imgs";
            bool save_ok = false;
            if (args.video) {
                if (video_output.frame_count <= 0 || video_output.frames == nullptr) {
                    std::fprintf(stderr, "empty video output for prompt %d\n", idx);
                    ed_free_video(&video_output);
                    rc = 4;
                    break;
                }
                std::snprintf(fname, sizeof(fname), "vid_%06d.avi", idx);
                save_ok = save_mjpg_avi((out_path / fname).string().c_str(),
                                        video_output, output_fps, 95);
                if (save_ok && video_output.audio != nullptr && video_output.audio_sample_count > 0) {
                    save_ok = save_wav((out_path / (std::string(fname) + ".wav")).string(), video_output);
                }
            } else {
                if (output.count <= 0 || output.images == nullptr) {
                    std::fprintf(stderr, "empty output for prompt %d\n", idx);
                    ed_free_image_batch(&output);
                    rc = 4;
                    break;
                }
                std::snprintf(fname, sizeof(fname), "img_%06d.png", idx);
                save_ok = save_png((out_path / fname).string().c_str(), output.images[0]);
            }
            if (!save_ok) {
                std::fprintf(stderr, "failed to save %s\n", (out_path / fname).string().c_str());
                ed_free_image_batch(&output);
                ed_free_video(&video_output);
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
            ed_free_video(&video_output);
        }
    }

    if (rc != 0) {
        free_inputs();
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

    free_inputs();
    ed_free_context(ctx);
    return 0;
}
