#include "parallel/process_group.hpp"

#include "parallel/backends/cpu/cpu_process_group.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#ifdef ED_ENABLE_NCCL
#include "parallel/backends/nccl/nccl_process_group.hpp"
#endif

namespace edgedit::parallel {
namespace {

class CompletedWork final : public Work {
public:
    void wait() override {}

    bool is_completed() const override {
        return true;
    }
};

} // namespace

std::unique_ptr<Work> ProcessGroup::all_reduce_async(const Buffer& input,
                                                     const Buffer& output,
                                                     ReduceOp op) {
    all_reduce(input, output, op);
    return std::make_unique<CompletedWork>();
}

std::unique_ptr<Work> ProcessGroup::all_gather_async(const Buffer& input,
                                                     const Buffer& output) {
    all_gather(input, output);
    return std::make_unique<CompletedWork>();
}

std::unique_ptr<Work> ProcessGroup::all_gather_async_on_stream(const Buffer& input,
                                                               const Buffer& output,
                                                               void* stream) {
    (void)stream;
    return all_gather_async(input, output);
}

std::unique_ptr<Work> ProcessGroup::all_to_all_async(const Buffer& input,
                                                     const Buffer& output,
                                                     size_t count_per_peer) {
    all_to_all(input, output, count_per_peer);
    return std::make_unique<CompletedWork>();
}

std::unique_ptr<Work> ProcessGroup::all_to_all_async_on_stream(const Buffer& input,
                                                               const Buffer& output,
                                                               size_t count_per_peer,
                                                               void* stream) {
    (void)stream;
    return all_to_all_async(input, output, count_per_peer);
}

std::unique_ptr<Work> ProcessGroup::broadcast_async(const Buffer& buffer,
                                                    int root) {
    broadcast(buffer, root);
    return std::make_unique<CompletedWork>();
}

void ProcessGroup::warmup() {}

const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::kNone:
            return "none";
        case Backend::kCpu:
            return "cpu";
        case Backend::kNccl:
            return "nccl";
    }
    return "unknown";
}

const char* dtype_name(DataType type) {
    switch (type) {
        case DataType::kFloat32:
            return "float32";
        case DataType::kFloat16:
            return "float16";
        case DataType::kBFloat16:
            return "bfloat16";
        case DataType::kInt32:
            return "int32";
        case DataType::kInt64:
            return "int64";
        case DataType::kUInt8:
            return "uint8";
    }
    return "unknown";
}

size_t dtype_size(DataType type) {
    switch (type) {
        case DataType::kFloat32:
            return sizeof(float);
        case DataType::kFloat16:
            return 2;
        case DataType::kBFloat16:
            return 2;
        case DataType::kInt32:
            return sizeof(int32_t);
        case DataType::kInt64:
            return sizeof(int64_t);
        case DataType::kUInt8:
            return sizeof(uint8_t);
    }
    throw std::invalid_argument("unsupported parallel data type");
}

Backend parse_backend(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (lower.empty() || lower == "none" || lower == "disabled") {
        return Backend::kNone;
    }
    if (lower == "cpu") {
        return Backend::kCpu;
    }
    if (lower == "nccl") {
        return Backend::kNccl;
    }
    throw std::invalid_argument("unknown parallel backend: " + name);
}

std::unique_ptr<ProcessGroup> create_process_group(const ParallelConfig& config) {
    if (config.world_size <= 0) {
        throw std::invalid_argument("parallel world_size must be positive");
    }
    if (config.rank < 0 || config.rank >= config.world_size) {
        throw std::invalid_argument("parallel rank must be in [0, world_size)");
    }

    switch (config.backend) {
        case Backend::kNone:
        case Backend::kCpu:
            return std::make_unique<CpuProcessGroup>(config);
        case Backend::kNccl:
#ifdef ED_ENABLE_NCCL
            return std::make_unique<NcclProcessGroup>(config);
#else
            throw std::runtime_error("NCCL backend requested, but edge-dit was built without ED_ENABLE_NCCL");
#endif
    }
    throw std::invalid_argument("unsupported parallel backend");
}

} // namespace edgedit::parallel
