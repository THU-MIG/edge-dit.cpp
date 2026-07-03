#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                                          \
    do {                                                                                          \
        cudaError_t _status = (expr);                                                             \
        if (_status != cudaSuccess) {                                                             \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_status) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));        \
        }                                                                                         \
    } while (0)

struct Options {
    int rows = 24 * 4352;
    int cols = 128;
    int warmup = 20;
    int iters = 300;
    float eps = 1.0e-6f;
};

void usage(const char * argv0) {
    std::cerr << "Usage: " << argv0 << " [options]\n"
              << "  --rows N    number of rows, default 24*4352\n"
              << "  --cols N    hidden/head dim, default 128; use 0 to sweep small-D cases\n"
              << "  --warmup N  warmup iterations, default 20\n"
              << "  --iters N   measured iterations, default 300\n"
              << "  --eps F     RMSNorm epsilon, default 1e-6\n";
}

int parse_int(const char * s) {
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0) {
        throw std::runtime_error(std::string("invalid positive integer: ") + s);
    }
    return static_cast<int>(v);
}

int parse_non_negative_int(const char * s) {
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 0) {
        throw std::runtime_error(std::string("invalid non-negative integer: ") + s);
    }
    return static_cast<int>(v);
}

float parse_float(const char * s) {
    char * end = nullptr;
    float v = std::strtof(s, &end);
    if (end == s || *end != '\0' || v < 0.0f) {
        throw std::runtime_error(std::string("invalid non-negative float: ") + s);
    }
    return v;
}

Options parse_args(int argc, char ** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--rows") == 0) {
            opt.rows = parse_int(need_value("--rows"));
        } else if (std::strcmp(argv[i], "--cols") == 0) {
            opt.cols = parse_non_negative_int(need_value("--cols"));
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            opt.warmup = parse_int(need_value("--warmup"));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opt.iters = parse_int(need_value("--iters"));
        } else if (std::strcmp(argv[i], "--eps") == 0) {
            opt.eps = parse_float(need_value("--eps"));
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error(std::string("unknown option: ") + argv[i]);
        }
    }
    return opt;
}

template <typename T>
struct DeviceBuffer {
    T * ptr = nullptr;
    size_t count = 0;

    explicit DeviceBuffer(size_t n = 0) : count(n) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;
};

__inline__ __device__ float warp_reduce_sum_lane0(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__inline__ __device__ float warp_reduce_sum_all(float v) {
    v = warp_reduce_sum_lane0(v);
    return __shfl_sync(0xffffffffu, v, 0);
}

template <int block_size>
__inline__ __device__ float block_reduce_sum(float v) {
    static_assert(block_size % 32 == 0, "block size must be a multiple of warp size");
    __shared__ float warp_sums[block_size / 32];
    __shared__ float block_sum;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_reduce_sum_lane0(v);
    if (lane == 0) {
        warp_sums[warp] = v;
    }
    __syncthreads();
    v = threadIdx.x < block_size / 32 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        v = warp_reduce_sum_lane0(v);
        if (lane == 0) {
            block_sum = v;
        }
    }
    __syncthreads();
    return block_sum;
}

template <int block_size>
__global__ void rms_norm_mul_generic_kernel(const float * x, const float * mul, float * y, int cols, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * x_row = x + static_cast<int64_t>(row) * cols;
    float * y_row = y + static_cast<int64_t>(row) * cols;

    float ss = 0.0f;
    for (int col = tid; col < cols; col += block_size) {
        const float v = x_row[col];
        ss += v * v;
    }
    ss = block_reduce_sum<block_size>(ss);
    const float scale = rsqrtf(ss / cols + eps);

    for (int col = tid; col < cols; col += block_size) {
        y_row[col] = x_row[col] * scale * mul[col % cols];
    }
}

__global__ void rms_norm_mul_small_d_warp_kernel(const float * x, const float * mul, float * y, int rows, int cols, float eps) {
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp >= rows) {
        return;
    }

    const int lane = threadIdx.x & 31;
    const float * x_row = x + static_cast<int64_t>(global_warp) * cols;
    float * y_row = y + static_cast<int64_t>(global_warp) * cols;

    float ss = 0.0f;
    for (int col = lane; col < cols; col += 32) {
        const float v = x_row[col];
        ss += v * v;
    }
    ss = warp_reduce_sum_all(ss);
    const float scale = rsqrtf(ss / static_cast<float>(cols) + eps);

    for (int col = lane; col < cols; col += 32) {
        y_row[col] = x_row[col] * scale * mul[col];
    }
}

void cpu_rms_norm_mul(const std::vector<float> & x,
                      const std::vector<float> & mul,
                      std::vector<float> & y,
                      int rows,
                      int cols,
                      float eps) {
    for (int row = 0; row < rows; ++row) {
        const int64_t base = static_cast<int64_t>(row) * cols;
        double ss = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double v = x[base + col];
            ss += v * v;
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(ss / cols) + eps);
        for (int col = 0; col < cols; ++col) {
            y[base + col] = x[base + col] * scale * mul[col];
        }
    }
}

struct ErrorStats {
    double mae = 0.0;
    double rmse = 0.0;
    float max_abs = 0.0f;
};

ErrorStats compare(const std::vector<float> & got, const std::vector<float> & ref) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float d = std::abs(got[i] - ref[i]);
        sum_abs += d;
        sum_sq += static_cast<double>(d) * d;
        max_abs = std::max(max_abs, d);
    }
    return {sum_abs / got.size(), std::sqrt(sum_sq / got.size()), max_abs};
}

float pseudo_uniform(uint32_t & state, float lo, float hi) {
    state = state * 1664525u + 1013904223u;
    const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777215.0f);
    return lo + (hi - lo) * u;
}

template <typename Fn>
float benchmark_cuda(Fn && fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

void print_result(const std::string & name, float ms, const ErrorStats & err) {
    std::cout << std::left << std::setw(28) << name
              << " avg_ms=" << std::right << std::setw(9) << std::fixed << std::setprecision(4) << ms
              << "  mae=" << std::scientific << std::setprecision(3) << err.mae
              << "  rmse=" << err.rmse
              << "  max_abs=" << err.max_abs << std::defaultfloat << "\n";
}

void run_case(const Options & opt) {
    const int64_t n = static_cast<int64_t>(opt.rows) * opt.cols;

    std::vector<float> h_x(n);
    std::vector<float> h_mul(opt.cols);
    std::vector<float> h_ref(n);
    std::vector<float> h_out(n);

    uint32_t rng = 1234 + static_cast<uint32_t>(opt.cols);
    for (float & v : h_x) {
        v = pseudo_uniform(rng, -1.0f, 1.0f);
    }
    for (float & v : h_mul) {
        v = pseudo_uniform(rng, 0.8f, 1.2f);
    }
    cpu_rms_norm_mul(h_x, h_mul, h_ref, opt.rows, opt.cols, opt.eps);

    DeviceBuffer<float> d_x(n);
    DeviceBuffer<float> d_mul(opt.cols);
    DeviceBuffer<float> d_y(n);
    CUDA_CHECK(cudaMemcpy(d_x.ptr, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mul.ptr, h_mul.data(), opt.cols * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "rows=" << opt.rows
              << " cols=" << opt.cols
              << " eps=" << opt.eps
              << " warmup=" << opt.warmup
              << " iters=" << opt.iters << "\n";

    auto run_generic = [&]() {
        rms_norm_mul_generic_kernel<256><<<opt.rows, 256>>>(d_x.ptr, d_mul.ptr, d_y.ptr, opt.cols, opt.eps);
    };
    float generic_ms = benchmark_cuda(run_generic, opt.warmup, opt.iters);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_y.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    print_result("ggml-like-rmsnorm-mul", generic_ms, compare(h_out, h_ref));

    if (opt.cols <= 256) {
        const int threads = 256;
        const int warps_per_block = threads / 32;
        const int blocks = (opt.rows + warps_per_block - 1) / warps_per_block;
        auto run_small_d = [&]() {
            rms_norm_mul_small_d_warp_kernel<<<blocks, threads>>>(d_x.ptr, d_mul.ptr, d_y.ptr, opt.rows, opt.cols, opt.eps);
        };
        float small_d_ms = benchmark_cuda(run_small_d, opt.warmup, opt.iters);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_y.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
        print_result("small-d-warp", small_d_ms, compare(h_out, h_ref));
    } else {
        std::cout << "small-d-warp skipped: --cols must be <= 256\n";
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        Options opt = parse_args(argc, argv);
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "device=" << prop.name << "\n";

        if (opt.cols == 0) {
            for (int cols : {32, 64, 80, 96, 128, 160, 192, 256}) {
                Options case_opt = opt;
                case_opt.cols = cols;
                run_case(case_opt);
            }
        } else {
            run_case(opt);
        }

        std::cout << "This benchmark compares ggml-like fused RMSNorm+scale with the generic small-D warp path.\n";

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
