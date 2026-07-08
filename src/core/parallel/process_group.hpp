#ifndef __ED_PARALLEL_PROCESS_GROUP_HPP__
#define __ED_PARALLEL_PROCESS_GROUP_HPP__

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace edgedit::parallel {

enum class Backend {
    kNone,
    kCpu,
    kNccl,
};

enum class DataType {
    kFloat32,
    kFloat16,
    kBFloat16,
    kInt32,
    kInt64,
    kUInt8,
};

enum class ReduceOp {
    kSum,
    kMax,
    kMin,
};

struct Buffer {
    void* data    = nullptr;
    size_t count  = 0;
    DataType type = DataType::kFloat32;
    int device    = -1;
};

struct ParallelConfig {
    Backend backend       = Backend::kNone;
    int rank              = 0;
    int world_size        = 1;
    int local_rank        = 0;
    int device            = 0;
    std::string store_path;
    int cfg_parallel_size = 1;
    int tp_parallel_size  = 1;
    int sp_parallel_size  = 1;
};

// 异步通信句柄：
// NCCL 后端里 wait() 才真正同步；CPU 后端默认 async 会退化成同步完成。
class Work {
public:
    virtual ~Work() = default;

    virtual void wait() = 0;
    virtual bool is_completed() const = 0;
};

class ProcessGroup {
public:
    virtual ~ProcessGroup() = default;

    virtual Backend backend() const = 0;
    virtual int rank() const        = 0;
    virtual int size() const        = 0;
    virtual int local_rank() const  = 0;
    virtual bool enabled() const    = 0;

    virtual void warmup();
    virtual void barrier() = 0;
    
    virtual std::unique_ptr<Work> all_reduce_async(const Buffer& input,
                                                   const Buffer& output,
                                                   ReduceOp op);

    virtual std::unique_ptr<Work> all_gather_async(const Buffer& input,
                                                   const Buffer& output);

    virtual std::unique_ptr<Work> all_gather_async_on_stream(const Buffer& input,
                                                             const Buffer& output,
                                                             void* stream);

    virtual std::unique_ptr<Work> all_to_all_async(const Buffer& input,
                                                   const Buffer& output,
                                                   size_t count_per_peer);

    virtual std::unique_ptr<Work> all_to_all_async_on_stream(const Buffer& input,
                                                             const Buffer& output,
                                                             size_t count_per_peer,
                                                             void* stream);

    virtual std::unique_ptr<Work> broadcast_async(const Buffer& buffer,
                                                  int root);
    virtual void all_reduce(const Buffer& input,
                            const Buffer& output,
                            ReduceOp op) = 0;

    virtual void all_gather(const Buffer& input,
                            const Buffer& output) = 0;

    virtual void all_to_all(const Buffer& input,
                            const Buffer& output,
                            size_t count_per_peer) = 0;

    virtual void broadcast(const Buffer& buffer,
                           int root) = 0;
};

const char* backend_name(Backend backend);
const char* dtype_name(DataType type);
size_t dtype_size(DataType type);
Backend parse_backend(const std::string& name);

std::unique_ptr<ProcessGroup> create_process_group(const ParallelConfig& config);

} // namespace edgedit::parallel

#endif // __ED_PARALLEL_PROCESS_GROUP_HPP__
