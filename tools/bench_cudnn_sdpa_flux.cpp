#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

namespace {

#define CUDA_CHECK(expr)                                                                          \
    do {                                                                                          \
        cudaError_t _status = (expr);                                                             \
        if (_status != cudaSuccess) {                                                             \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_status) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));        \
        }                                                                                         \
    } while (0)

#define CUDNN_CHECK(expr)                                                                            \
    do {                                                                                             \
        cudnnStatus_t _status = (expr);                                                              \
        if (_status != CUDNN_STATUS_SUCCESS) {                                                       \
            throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(_status) +  \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));           \
        }                                                                                            \
    } while (0)

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;

struct Options {
    int64_t b = 1;
    int64_t h = 24;
    int64_t s = 4352;
    int64_t d = 128;
    int warmup = 10;
    int iters = 200;
    bool bshd = false;
    bool bf16 = true;
    bool generate_stats = false;
    float scale = 0.0f;
};

void usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --b N             batch size, default 1\n"
        << "  --h N             heads, default 24\n"
        << "  --s N             sequence length, default 4352\n"
        << "  --d N             head dim, default 128\n"
        << "  --warmup N        warmup iterations, default 10\n"
        << "  --iters N         measured iterations, default 200\n"
        << "  --layout bhsd|bshd physical layout, default bhsd\n"
        << "  --dtype bf16|f16  IO dtype, default bf16\n"
        << "  --stats 0|1       generate softmax stats, default 0\n"
        << "  --scale F         attention scale, default 1/sqrt(d)\n";
}

int64_t parse_i64(const char * s) {
    char * end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0') {
        throw std::runtime_error(std::string("invalid integer: ") + s);
    }
    return static_cast<int64_t>(v);
}

int parse_int(const char * s) {
    return static_cast<int>(parse_i64(s));
}

float parse_float(const char * s) {
    char * end = nullptr;
    float v = std::strtof(s, &end);
    if (end == s || *end != '\0') {
        throw std::runtime_error(std::string("invalid float: ") + s);
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
        if (std::strcmp(argv[i], "--b") == 0) {
            opt.b = parse_i64(need_value("--b"));
        } else if (std::strcmp(argv[i], "--h") == 0) {
            opt.h = parse_i64(need_value("--h"));
        } else if (std::strcmp(argv[i], "--s") == 0) {
            opt.s = parse_i64(need_value("--s"));
        } else if (std::strcmp(argv[i], "--d") == 0) {
            opt.d = parse_i64(need_value("--d"));
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            opt.warmup = parse_int(need_value("--warmup"));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opt.iters = parse_int(need_value("--iters"));
        } else if (std::strcmp(argv[i], "--layout") == 0) {
            std::string v = need_value("--layout");
            if (v == "bshd") {
                opt.bshd = true;
            } else if (v == "bhsd") {
                opt.bshd = false;
            } else {
                throw std::runtime_error("layout must be bhsd or bshd");
            }
        } else if (std::strcmp(argv[i], "--dtype") == 0) {
            std::string v = need_value("--dtype");
            if (v == "bf16") {
                opt.bf16 = true;
            } else if (v == "f16") {
                opt.bf16 = false;
            } else {
                throw std::runtime_error("dtype must be bf16 or f16");
            }
        } else if (std::strcmp(argv[i], "--stats") == 0) {
            opt.generate_stats = parse_int(need_value("--stats")) != 0;
        } else if (std::strcmp(argv[i], "--scale") == 0) {
            opt.scale = parse_float(need_value("--scale"));
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error(std::string("unknown option: ") + argv[i]);
        }
    }
    if (opt.scale == 0.0f) {
        opt.scale = 1.0f / std::sqrt(static_cast<float>(opt.d));
    }
    return opt;
}

struct CudnnHandle {
    cudnnHandle_t handle = nullptr;
    CudnnHandle() { CUDNN_CHECK(cudnnCreate(&handle)); }
    ~CudnnHandle() {
        if (handle != nullptr) {
            cudnnDestroy(handle);
        }
    }
};

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

void fill_bytes(void * ptr, size_t bytes) {
    CUDA_CHECK(cudaMemset(ptr, 0x3c, bytes));
}

std::vector<int64_t> bh_stride(const Options & opt) {
    if (!opt.bshd) {
        return {opt.h * opt.s * opt.d, opt.s * opt.d, opt.d, 1};
    }
    return {opt.s * opt.h * opt.d, opt.d, opt.h * opt.d, 1};
}

std::shared_ptr<fe::graph::Graph> create_graph(const Options & opt, fe::DataType_t dtype) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(dtype)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const auto stride = bh_stride(opt);
    const std::vector<int64_t> dim = {opt.b, opt.h, opt.s, opt.d};

    auto q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim(dim)
                               .set_stride(stride));
    auto k = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim(dim)
                               .set_stride(stride));
    auto v = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim(dim)
                               .set_stride(stride));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flux_sdpa")
                            .set_generate_stats(opt.generate_stats)
                            .set_attn_scale(opt.scale);

    auto [o, stats] = graph->sdpa(q, k, v, sdpa_options);
    (void)stats;
    o->set_output(true).set_uid(O_UID).set_dim(dim).set_stride(stride);

    return graph;
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Options opt = parse_args(argc, argv);

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::cout << "bench_cudnn_sdpa_flux\n"
                  << "device=" << device << " " << prop.name << " cc=" << prop.major << "." << prop.minor << "\n"
                  << "cudnn_version=" << cudnnGetVersion() << "\n"
                  << "shape B,H,S,D=" << opt.b << "," << opt.h << "," << opt.s << "," << opt.d << "\n"
                  << "layout=" << (opt.bshd ? "bshd" : "bhsd") << " dtype=" << (opt.bf16 ? "bf16" : "f16")
                  << " scale=" << opt.scale << " warmup=" << opt.warmup << " iters=" << opt.iters << "\n";

        CudnnHandle handle;
        const fe::DataType_t dtype = opt.bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF;

        const size_t elements = static_cast<size_t>(opt.b * opt.h * opt.s * opt.d);
        const size_t element_size = opt.bf16 ? sizeof(__nv_bfloat16) : sizeof(half);
        DeviceBuffer<unsigned char> q(elements * element_size);
        DeviceBuffer<unsigned char> k(elements * element_size);
        DeviceBuffer<unsigned char> v(elements * element_size);
        DeviceBuffer<unsigned char> o(elements * element_size);
        fill_bytes(q.ptr, q.count);
        fill_bytes(k.ptr, k.count);
        fill_bytes(v.ptr, v.count);
        CUDA_CHECK(cudaMemset(o.ptr, 0, o.count));

        auto build_start = std::chrono::steady_clock::now();
        auto graph = create_graph(opt, dtype);
        auto status = graph->build(handle.handle, {fe::HeurMode_t::A});
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN graph build failed: " + status.get_message());
        }
        auto build_end = std::chrono::steady_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();

        int64_t workspace_size = 0;
        status = graph->get_workspace_size(workspace_size);
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN get_workspace_size failed: " + status.get_message());
        }
        DeviceBuffer<unsigned char> workspace(static_cast<size_t>(std::max<int64_t>(workspace_size, 1)));

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack = {
            {Q_UID, q.ptr},
            {K_UID, k.ptr},
            {V_UID, v.ptr},
            {O_UID, o.ptr},
        };

        for (int i = 0; i < opt.warmup; ++i) {
            status = graph->execute(handle.handle, variant_pack, workspace.ptr);
            if (!status.is_good()) {
                throw std::runtime_error("cuDNN graph execute failed during warmup: " + status.get_message());
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            status = graph->execute(handle.handle, variant_pack, workspace.ptr);
            if (!status.is_good()) {
                throw std::runtime_error("cuDNN graph execute failed: " + status.get_message());
            }
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        const float total_ms = elapsed_ms(start, stop);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        std::cout << std::fixed << std::setprecision(4)
                  << "build_ms=" << build_ms << "\n"
                  << "workspace_bytes=" << workspace_size << "\n"
                  << "total_execute_ms=" << total_ms << "\n"
                  << "mean_execute_ms=" << (total_ms / std::max(1, opt.iters)) << "\n"
                  << "edge_ggml_attention_reference_ms=3.2600\n"
                  << "diffusers_pytorch_flash_reference_ms=0.8220\n";

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

