#include "parallel/backends/nccl/nccl_process_group.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef ED_ENABLE_MPI
#include <mpi.h>
#endif

namespace edgedit::parallel {
namespace {

void check_cuda(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + " failed: " + cudaGetErrorString(status));
    }
}

void check_nccl(ncclResult_t status, const char* expr) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(expr) + " failed: " + ncclGetErrorString(status));
    }
}

class NcclWork final : public Work {
public:
    explicit NcclWork(cudaEvent_t event)
        : event_(event) {}

    ~NcclWork() override {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
            event_ = nullptr;
        }
    }

    NcclWork(const NcclWork&) = delete;
    NcclWork& operator=(const NcclWork&) = delete;

    void wait() override {
        if (event_ == nullptr || waited_) {
            return;
        }
        check_cuda(cudaEventSynchronize(event_), "cudaEventSynchronize");
        waited_ = true;
    }

    bool is_completed() const override {
        if (event_ == nullptr || waited_) {
            return true;
        }

        cudaError_t status = cudaEventQuery(event_);
        if (status == cudaSuccess) {
            waited_ = true;
            return true;
        }
        if (status == cudaErrorNotReady) {
            return false;
        }

        throw std::runtime_error(
            std::string("cudaEventQuery failed: ") + cudaGetErrorString(status)
        );
    }

private:
    cudaEvent_t event_ = nullptr;
    mutable bool waited_ = false;
};

ncclDataType_t to_nccl_dtype(DataType type) {
    switch (type) {
        case DataType::kFloat32:
            return ncclFloat32;
        case DataType::kFloat16:
            return ncclFloat16;
        case DataType::kBFloat16:
            return ncclBfloat16;
        case DataType::kInt32:
            return ncclInt32;
        case DataType::kInt64:
            return ncclInt64;
        case DataType::kUInt8:
            return ncclUint8;
    }
    throw std::invalid_argument("unsupported NCCL data type");
}

ncclRedOp_t to_nccl_reduce_op(ReduceOp op) {
    switch (op) {
        case ReduceOp::kSum:
            return ncclSum;
        case ReduceOp::kMax:
            return ncclMax;
        case ReduceOp::kMin:
            return ncclMin;
    }
    throw std::invalid_argument("unsupported NCCL reduce op");
}

int env_int(const char* name, int fallback) {
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

int infer_local_rank(int fallback) {
    fallback = env_int("LOCAL_RANK", fallback);
    fallback = env_int("OMPI_COMM_WORLD_LOCAL_RANK", fallback);
    fallback = env_int("MV2_COMM_WORLD_LOCAL_RANK", fallback);
    fallback = env_int("SLURM_LOCALID", fallback);
    fallback = env_int("PMI_LOCAL_RANK", fallback);
    return fallback;
}

int infer_global_rank(int fallback) {
    fallback = env_int("RANK", fallback);
    fallback = env_int("OMPI_COMM_WORLD_RANK", fallback);
    fallback = env_int("MV2_COMM_WORLD_RANK", fallback);
    fallback = env_int("SLURM_PROCID", fallback);
    fallback = env_int("PMI_RANK", fallback);
    return fallback;
}

int infer_world_size(int fallback) {
    fallback = env_int("WORLD_SIZE", fallback);
    fallback = env_int("OMPI_COMM_WORLD_SIZE", fallback);
    fallback = env_int("MV2_COMM_WORLD_SIZE", fallback);
    fallback = env_int("SLURM_NTASKS", fallback);
    fallback = env_int("PMI_SIZE", fallback);
    return fallback;
}

#ifdef ED_ENABLE_MPI
struct MpiRuntime {
    MpiRuntime() {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            int provided = 0;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);
            owned = true;
        }
    }

    ~MpiRuntime() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (owned && !finalized) {
            MPI_Finalize();
        }
    }

    bool owned = false;
};

MpiRuntime& mpi_runtime() {
    static MpiRuntime runtime;
    return runtime;
}
#endif

} // namespace

NcclProcessGroup::NcclProcessGroup(const ParallelConfig& config) : config_(config) {
#ifdef ED_ENABLE_MPI
    mpi_runtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &config_.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &config_.world_size);
    config_.local_rank = infer_local_rank(config_.local_rank);
#else
    config_.rank       = infer_global_rank(config_.rank);
    config_.world_size = infer_world_size(config_.world_size);
    config_.local_rank = infer_local_rank(config_.local_rank);
#endif

    if (config_.world_size <= 0) {
        throw std::invalid_argument("NCCL world_size must be positive");
    }
    if (config_.rank < 0 || config_.rank >= config_.world_size) {
        throw std::invalid_argument("NCCL rank must be in [0, world_size)");
    }

    check_cuda(cudaSetDevice(config_.device), "cudaSetDevice");
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    init_unique_id();
}

NcclProcessGroup::~NcclProcessGroup() {
    if (comm_ != nullptr) {
        ncclCommDestroy(comm_);
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

Backend NcclProcessGroup::backend() const {
    return Backend::kNccl;
}

int NcclProcessGroup::rank() const {
    return config_.rank;
}

int NcclProcessGroup::size() const {
    return config_.world_size;
}

int NcclProcessGroup::local_rank() const {
    return config_.local_rank;
}

bool NcclProcessGroup::enabled() const {
    return config_.world_size > 1;
}

void NcclProcessGroup::warmup() {
    if (!enabled()) {
        return;
    }

    set_device();

    const size_t count_per_peer = 1;
    const size_t world_count = static_cast<size_t>(config_.world_size) * count_per_peer;
    float* d_in = nullptr;
    float* d_out = nullptr;

    check_cuda(cudaMalloc(&d_in, world_count * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMalloc(&d_out, world_count * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMemsetAsync(d_in, 0, world_count * sizeof(float), stream_), "cudaMemsetAsync");
    check_cuda(cudaMemsetAsync(d_out, 0, world_count * sizeof(float), stream_), "cudaMemsetAsync");

    Buffer one_in{d_in, count_per_peer, DataType::kFloat32, config_.device};
    Buffer one_out{d_out, count_per_peer, DataType::kFloat32, config_.device};
    Buffer many_in{d_in, world_count, DataType::kFloat32, config_.device};
    Buffer many_out{d_out, world_count, DataType::kFloat32, config_.device};

    all_reduce(one_in, one_out, ReduceOp::kSum);
    all_gather(one_in, many_out);
    all_to_all(many_in, many_out, count_per_peer);
    broadcast(many_in, 0);

    check_cuda(cudaFree(d_in), "cudaFree");
    check_cuda(cudaFree(d_out), "cudaFree");
}

void NcclProcessGroup::barrier() {
    set_device();
    int send = config_.rank;
    int recv = 0;
    Buffer in{&send, 1, DataType::kInt32, config_.device};
    Buffer out{&recv, 1, DataType::kInt32, config_.device};

    int* d_in  = nullptr;
    int* d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, sizeof(int)), "cudaMalloc");
    check_cuda(cudaMalloc(&d_out, sizeof(int)), "cudaMalloc");
    check_cuda(cudaMemcpy(d_in, &send, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    in.data  = d_in;
    out.data = d_out;
    all_reduce(in, out, ReduceOp::kSum);
    check_cuda(cudaMemcpy(&recv, d_out, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    check_cuda(cudaFree(d_in), "cudaFree");
    check_cuda(cudaFree(d_out), "cudaFree");
}

std::unique_ptr<Work> NcclProcessGroup::record_work(cudaStream_t stream) {
    cudaEvent_t event = nullptr;
    check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
               "cudaEventCreateWithFlags");
    check_cuda(cudaEventRecord(event, stream), "cudaEventRecord");
    return std::make_unique<NcclWork>(event);
}

std::unique_ptr<Work> NcclProcessGroup::record_work() {
    return record_work(stream_);
}

std::unique_ptr<Work> NcclProcessGroup::all_reduce_async(
    const Buffer& input,
    const Buffer& output,
    ReduceOp op
) {
    set_device();
    check_buffer(input);
    check_buffer(output);

    if (input.count != output.count || input.type != output.type) {
        throw std::invalid_argument("NCCL all_reduce input and output buffers must match");
    }

    check_nccl(ncclAllReduce(input.data,
                             output.data,
                             input.count,
                             to_nccl_dtype(input.type),
                             to_nccl_reduce_op(op),
                             comm_,
                             stream_),
               "ncclAllReduce");

    return record_work();
}

void NcclProcessGroup::all_reduce(
    const Buffer& input,
    const Buffer& output,
    ReduceOp op
) {
    auto work = all_reduce_async(input, output, op);
    work->wait();
}

std::unique_ptr<Work> NcclProcessGroup::all_gather_async(
    const Buffer& input,
    const Buffer& output
) {
    set_device();
    check_buffer(input);
    check_buffer(output);

    if (input.type != output.type ||
        output.count != input.count * static_cast<size_t>(config_.world_size)) {
        throw std::invalid_argument("NCCL all_gather output must hold world_size copies of input");
    }

    check_nccl(ncclAllGather(input.data,
                             output.data,
                             input.count,
                             to_nccl_dtype(input.type),
                             comm_,
                             stream_),
               "ncclAllGather");

    return record_work();
}

std::unique_ptr<Work> NcclProcessGroup::all_gather_async_on_stream(
    const Buffer& input,
    const Buffer& output,
    void* stream
) {
    set_device();
    check_buffer(input);
    check_buffer(output);

    cudaStream_t comm_stream = reinterpret_cast<cudaStream_t>(stream);
    if (comm_stream == nullptr) {
        comm_stream = stream_;
    }

    if (input.type != output.type ||
        output.count != input.count * static_cast<size_t>(config_.world_size)) {
        throw std::invalid_argument("NCCL all_gather output must hold world_size copies of input");
    }

    check_nccl(ncclAllGather(input.data,
                             output.data,
                             input.count,
                             to_nccl_dtype(input.type),
                             comm_,
                             comm_stream),
               "ncclAllGather");

    return record_work(comm_stream);
}

void NcclProcessGroup::all_gather(
    const Buffer& input,
    const Buffer& output
) {
    auto work = all_gather_async(input, output);
    work->wait();
}

std::unique_ptr<Work> NcclProcessGroup::all_to_all_async(
    const Buffer& input,
    const Buffer& output,
    size_t count_per_peer
) {
    set_device();
    check_buffer(input);
    check_buffer(output);

    const size_t total_count = count_per_peer * static_cast<size_t>(config_.world_size);
    if (input.type != output.type ||
        input.count != total_count ||
        output.count != total_count) {
        throw std::invalid_argument(
            "NCCL all_to_all input and output must contain count_per_peer * world_size items"
        );
    }

    const size_t item_size  = dtype_size(input.type);
    const size_t chunk_size = count_per_peer * item_size;
    const uint8_t* send     = reinterpret_cast<const uint8_t*>(input.data);
    uint8_t* recv           = reinterpret_cast<uint8_t*>(output.data);

    check_nccl(ncclGroupStart(), "ncclGroupStart");
    for (int peer = 0; peer < config_.world_size; ++peer) {
        check_nccl(ncclSend(send + static_cast<size_t>(peer) * chunk_size,
                            count_per_peer,
                            to_nccl_dtype(input.type),
                            peer,
                            comm_,
                            stream_),
                   "ncclSend");

        check_nccl(ncclRecv(recv + static_cast<size_t>(peer) * chunk_size,
                            count_per_peer,
                            to_nccl_dtype(output.type),
                            peer,
                            comm_,
                            stream_),
                   "ncclRecv");
    }
    check_nccl(ncclGroupEnd(), "ncclGroupEnd");

    return record_work();
}

std::unique_ptr<Work> NcclProcessGroup::all_to_all_async_on_stream(
    const Buffer& input,
    const Buffer& output,
    size_t count_per_peer,
    void* stream
) {
    set_device();
    check_buffer(input);
    check_buffer(output);

    cudaStream_t comm_stream = reinterpret_cast<cudaStream_t>(stream);
    if (comm_stream == nullptr) {
        comm_stream = stream_;
    }

    const size_t total_count = count_per_peer * static_cast<size_t>(config_.world_size);
    if (input.type != output.type ||
        input.count != total_count ||
        output.count != total_count) {
        throw std::invalid_argument(
            "NCCL all_to_all input and output must contain count_per_peer * world_size items"
        );
    }

    const size_t item_size  = dtype_size(input.type);
    const size_t chunk_size = count_per_peer * item_size;
    const uint8_t* send     = reinterpret_cast<const uint8_t*>(input.data);
    uint8_t* recv           = reinterpret_cast<uint8_t*>(output.data);

    check_nccl(ncclGroupStart(), "ncclGroupStart");
    for (int peer = 0; peer < config_.world_size; ++peer) {
        check_nccl(ncclSend(send + static_cast<size_t>(peer) * chunk_size,
                            count_per_peer,
                            to_nccl_dtype(input.type),
                            peer,
                            comm_,
                            comm_stream),
                   "ncclSend");

        check_nccl(ncclRecv(recv + static_cast<size_t>(peer) * chunk_size,
                            count_per_peer,
                            to_nccl_dtype(output.type),
                            peer,
                            comm_,
                            comm_stream),
                   "ncclRecv");
    }
    check_nccl(ncclGroupEnd(), "ncclGroupEnd");

    return record_work(comm_stream);
}

void NcclProcessGroup::all_to_all(
    const Buffer& input,
    const Buffer& output,
    size_t count_per_peer
) {
    auto work = all_to_all_async(input, output, count_per_peer);
    work->wait();
}

std::unique_ptr<Work> NcclProcessGroup::broadcast_async(
    const Buffer& buffer,
    int root
) {
    set_device();
    check_buffer(buffer);

    if (root < 0 || root >= config_.world_size) {
        throw std::invalid_argument("NCCL broadcast root is out of range");
    }

    check_nccl(ncclBroadcast(buffer.data,
                             buffer.data,
                             buffer.count,
                             to_nccl_dtype(buffer.type),
                             root,
                             comm_,
                             stream_),
               "ncclBroadcast");

    return record_work();
}

void NcclProcessGroup::broadcast(
    const Buffer& buffer,
    int root
) {
    auto work = broadcast_async(buffer, root);
    work->wait();
}

void NcclProcessGroup::init_unique_id() {
    ncclUniqueId id;
    if (config_.rank == 0) {
        check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    }
#ifdef ED_ENABLE_MPI
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
#else
    if (config_.world_size != 1) {
        throw std::runtime_error("distributed NCCL requires ED_ENABLE_MPI=ON for unique id bootstrap");
    }
#endif
    check_nccl(ncclCommInitRank(&comm_, config_.world_size, id, config_.rank), "ncclCommInitRank");
}

void NcclProcessGroup::set_device() const {
    check_cuda(cudaSetDevice(config_.device), "cudaSetDevice");
}

void NcclProcessGroup::check_buffer(const Buffer& buffer) const {
    if (buffer.data == nullptr && buffer.count != 0) {
        throw std::invalid_argument("NCCL buffer data is null");
    }
}

} // namespace edgedit::parallel
