#ifndef __ED_PARALLEL_CONTEXT_HPP__
#define __ED_PARALLEL_CONTEXT_HPP__

#include "parallel/process_group.hpp"

#include <memory>

namespace edgedit::parallel {

class ParallelContext {
public:
    ParallelContext(std::unique_ptr<ProcessGroup> group,
                    int cfg_parallel_size = 1,
                    int tp_parallel_size = 1,
                    int sp_parallel_size = 1,
                    int device = 0);

    bool enabled() const;
    bool is_root() const;
    int rank() const;
    int local_rank() const;
    int device() const;
    int world_size() const;
    int cfg_parallel_size() const;
    int tp_parallel_size() const;
    int sp_parallel_size() const;
    Backend backend() const;

    ProcessGroup& world_group();
    const ProcessGroup& world_group() const;

    ProcessGroup& group();
    const ProcessGroup& group() const;

private:
    std::unique_ptr<ProcessGroup> group_;
    int cfg_parallel_size_ = 1;
    int tp_parallel_size_ = 1;
    int sp_parallel_size_ = 1;
    int device_ = 0;
};

std::unique_ptr<ParallelContext> create_parallel_context(const ParallelConfig& config);

} // namespace edgedit::parallel

#endif // __ED_PARALLEL_CONTEXT_HPP__
