#include "parallel/parallel_context.hpp"

#include <stdexcept>

namespace edgedit::parallel {

namespace {

int normalize_parallel_size(int value) {
    return value > 0 ? value : 1;
}

} // namespace

ParallelContext::ParallelContext(std::unique_ptr<ProcessGroup> group,
                                 int cfg_parallel_size,
                                 int tp_parallel_size,
                                 int sp_parallel_size,
                                 int device)
    : group_(std::move(group)),
      cfg_parallel_size_(normalize_parallel_size(cfg_parallel_size)),
      tp_parallel_size_(normalize_parallel_size(tp_parallel_size)),
      sp_parallel_size_(normalize_parallel_size(sp_parallel_size)),
      device_(device >= 0 ? device : 0) {
    if (!group_) {
        throw std::invalid_argument("parallel context requires a process group");
    }
}

bool ParallelContext::enabled() const {
    return group_->enabled();
}

bool ParallelContext::is_root() const {
    return rank() == 0;
}

int ParallelContext::rank() const {
    return group_->rank();
}

int ParallelContext::local_rank() const {
    return group_->local_rank();
}

int ParallelContext::device() const {
    return device_;
}

int ParallelContext::world_size() const {
    return group_->size();
}

int ParallelContext::cfg_parallel_size() const {
    return cfg_parallel_size_;
}

int ParallelContext::tp_parallel_size() const {
    return tp_parallel_size_;
}

int ParallelContext::sp_parallel_size() const {
    return sp_parallel_size_;
}

Backend ParallelContext::backend() const {
    return group_->backend();
}

ProcessGroup& ParallelContext::world_group() {
    return *group_;
}

const ProcessGroup& ParallelContext::world_group() const {
    return *group_;
}

ProcessGroup& ParallelContext::group() {
    return *group_;
}

const ProcessGroup& ParallelContext::group() const {
    return *group_;
}

std::unique_ptr<ParallelContext> create_parallel_context(const ParallelConfig& config) {
    return std::make_unique<ParallelContext>(create_process_group(config),
                                             config.cfg_parallel_size,
                                             config.tp_parallel_size,
                                             config.sp_parallel_size,
                                             config.device);
}

} // namespace edgedit::parallel
