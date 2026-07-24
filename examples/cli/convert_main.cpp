// ed-convert: offline model quantizer / format converter for edge-dit.cpp.
//
// Quantizes an fp16/bf16 model to a target ggml type once and writes a single
// GGUF (or safetensors) file, so subsequent inference can load the pre-quantized
// weights directly and skip the (very slow) on-the-fly CPU quantization that
// happens every time a diffusers/safetensors model is loaded with --type.
//
// This mirrors stable-diffusion.cpp's `-M convert` workflow. Example:
//
//   ed-convert -m /models/sd3-medium -o sd3-q8.gguf --type q8_0
//   ed-sample  -m sd3-q8.gguf --prompt_file prompts.txt ...   // fast load
//
// The heavy lifting lives in src/utils/model_io/convert.cpp; this file is only
// argument parsing plus a call into convert().

#include "edge-dit.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "utils/model_io/convert.h"

namespace {

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --model <model-or-diffusers-dir> --output <out.gguf> [--type <dtype>] [options]\n"
        "\n"
        "Converts / quantizes a model once and writes it to disk so inference can\n"
        "load pre-quantized weights directly (skips on-the-fly quantization).\n"
        "\n"
        "Options:\n"
        "  -m, --model <path>        Input model: diffusers dir, .safetensors, shard index, or .gguf (required)\n"
        "  -o, --output <path>       Output path; .safetensors -> safetensors, else GGUF (required)\n"
        "      --vae <path>          Optional external VAE weights, merged under the vae. prefix\n"
        "      --type <dtype>        Target weight type: f32, f16, bf16, q4_0, q4_1, q5_0, q5_1,\n"
        "                            q8_0, q2_k, q3_k, q4_k, q5_k, q6_k. Default: q8_0\n"
        "      --tensor-type-rules <csv>\n"
        "                            Per-tensor quant overrides, e.g. \"attn=q4_0,norm=f16\"\n"
        "      --awq <path>          Optional AWQ calibration GGUF (per-input-channel importance,\n"
        "                            produced by tools/awq/calibrate.py). When set, quantization\n"
        "                            weights toward the important channels instead of the uniform\n"
        "                            default -- improves low-bit (q4_k) quality.\n"
        "      --raw-names           Keep raw (pre-canonicalization) tensor names. Default is to\n"
        "                            canonicalize, which is REQUIRED for a reusable GGUF (a GGUF\n"
        "                            has no config, so the loader recovers the version from names)\n"
        "  -h, --help                Show this help\n",
        prog);
}

// Local dtype string -> ed_dtype_t parser. Kept independent of cli_common.hpp so
// this tool does not pull in the stb image / sampling machinery.
bool parse_dtype(const char* text, ed_dtype_t* out) {
    std::string type = text != nullptr ? text : "";
    for (char& c : type) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        } else if (c == '-' || c == '.') {
            c = '_';
        }
    }

    if (type == "f32" || type == "fp32") { *out = ED_DTYPE_F32; return true; }
    if (type == "f16" || type == "fp16") { *out = ED_DTYPE_F16; return true; }
    if (type == "bf16")                  { *out = ED_DTYPE_BF16; return true; }
    if (type == "q4_0")                  { *out = ED_DTYPE_Q4_0; return true; }
    if (type == "q4_1")                  { *out = ED_DTYPE_Q4_1; return true; }
    if (type == "q5_0")                  { *out = ED_DTYPE_Q5_0; return true; }
    if (type == "q5_1")                  { *out = ED_DTYPE_Q5_1; return true; }
    if (type == "q8_0")                  { *out = ED_DTYPE_Q8_0; return true; }
    if (type == "q2_k")                  { *out = ED_DTYPE_Q2_K; return true; }
    if (type == "q3_k")                  { *out = ED_DTYPE_Q3_K; return true; }
    if (type == "q4_k")                  { *out = ED_DTYPE_Q4_K; return true; }
    if (type == "q5_k")                  { *out = ED_DTYPE_Q5_K; return true; }
    if (type == "q6_k")                  { *out = ED_DTYPE_Q6_K; return true; }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model_path        = nullptr;
    const char* output_path       = nullptr;
    const char* vae_path          = nullptr;
    const char* tensor_type_rules = nullptr;
    const char* imatrix_path      = nullptr;
    ed_dtype_t  output_type       = ED_DTYPE_Q8_0;
    bool        convert_name      = true;

    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];

        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(key, "--help") == 0 || std::strcmp(key, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(key, "--model") == 0 || std::strcmp(key, "--model_path") == 0 ||
                   std::strcmp(key, "-m") == 0) {
            model_path = require_value(key);
            if (!model_path) return 1;
        } else if (std::strcmp(key, "--output") == 0 || std::strcmp(key, "-o") == 0) {
            output_path = require_value(key);
            if (!output_path) return 1;
        } else if (std::strcmp(key, "--vae") == 0) {
            vae_path = require_value(key);
            if (!vae_path) return 1;
        } else if (std::strcmp(key, "--type") == 0 || std::strcmp(key, "--weight-type") == 0) {
            const char* v = require_value(key);
            if (!v) return 1;
            if (!parse_dtype(v, &output_type)) {
                std::fprintf(stderr, "unsupported weight type: %s\n", v);
                return 1;
            }
        } else if (std::strcmp(key, "--tensor-type-rules") == 0) {
            tensor_type_rules = require_value(key);
            if (!tensor_type_rules) return 1;
        } else if (std::strcmp(key, "--awq") == 0) {
            imatrix_path = require_value(key);
            if (!imatrix_path) return 1;
        } else if (std::strcmp(key, "--raw-names") == 0) {
            convert_name = false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", key);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path == nullptr || std::strlen(model_path) == 0) {
        std::fprintf(stderr, "--model is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (output_path == nullptr || std::strlen(output_path) == 0) {
        std::fprintf(stderr, "--output is required\n");
        print_usage(argv[0]);
        return 1;
    }

    std::fprintf(stderr, "converting '%s' -> '%s' (type=%d)\n", model_path, output_path, (int)output_type);

    if (!convert(model_path, vae_path, output_path, output_type, tensor_type_rules, convert_name, imatrix_path)) {
        std::fprintf(stderr, "conversion failed\n");
        return 2;
    }

    std::fprintf(stderr, "conversion succeeded: %s\n", output_path);
    return 0;
}
