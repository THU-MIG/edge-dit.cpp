#include "parallel/cfg_parallel.hpp"

#include <cstring>
#include <stdexcept>

#ifdef ED_ENABLE_NCCL
#include <cuda_runtime.h>
#endif

namespace edgedit::parallel {
namespace {

#ifdef ED_ENABLE_NCCL
void check_cuda(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + " failed: " + cudaGetErrorString(status));
    }
}
#endif

} // namespace

bool cfg_parallel_available(const ParallelContext* context) {
    return context != nullptr &&
           context->enabled() &&
           context->cfg_parallel_size() == 2 &&
           context->world_size() == 2;
}

int cfg_parallel_rank(const ParallelContext* context) {
    return context != nullptr ? context->rank() : 0;
}

int cfg_parallel_world_size(const ParallelContext* context) {
    return context != nullptr ? context->world_size() : 1;
}

bool cfg_all_gather(ParallelContext& context,
                    const sd::Tensor<float>& local,
                    std::vector<sd::Tensor<float>>* gathered,
                    std::string* error) {
    if (gathered == nullptr) {
        if (error != nullptr) {
            *error = "cfg_all_gather got null output";
        }
        return false;
    }
    if (local.empty()) {
        if (error != nullptr) {
            *error = "cfg_all_gather got empty local tensor";
        }
        return false;
    }
    if (context.world_size() <= 1) {
        *gathered = {local};
        return true;
    }

    const size_t local_count = static_cast<size_t>(local.numel());
    std::vector<float> all(local_count * static_cast<size_t>(context.world_size()), 0.0f);

    if (context.backend() == Backend::kNccl) {
#ifdef ED_ENABLE_NCCL
        float* d_local = nullptr;
        float* d_all = nullptr;
        const size_t local_bytes = local_count * sizeof(float);
        const size_t all_bytes = all.size() * sizeof(float);
        try {
            check_cuda(cudaSetDevice(context.device()), "cudaSetDevice");
            check_cuda(cudaMalloc(&d_local, local_bytes), "cudaMalloc");
            check_cuda(cudaMalloc(&d_all, all_bytes), "cudaMalloc");
            check_cuda(cudaMemcpy(d_local, local.data(), local_bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
            context.world_group().all_gather(Buffer{d_local, local_count, DataType::kFloat32, context.device()},
                                             Buffer{d_all, all.size(), DataType::kFloat32, context.device()});
            check_cuda(cudaMemcpy(all.data(), d_all, all_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
        } catch (...) {
            if (d_local != nullptr) {
                cudaFree(d_local);
            }
            if (d_all != nullptr) {
                cudaFree(d_all);
            }
            throw;
        }
        cudaFree(d_local);
        cudaFree(d_all);
#else
        if (error != nullptr) {
            *error = "cfg_all_gather requested NCCL but edge-dit was built without NCCL";
        }
        return false;
#endif
    } else {
        context.world_group().all_gather(Buffer{const_cast<float*>(local.data()), local_count, DataType::kFloat32, -1},
                                         Buffer{all.data(), all.size(), DataType::kFloat32, -1});
    }

    gathered->clear();
    gathered->reserve(static_cast<size_t>(context.world_size()));
    for (int rank = 0; rank < context.world_size(); ++rank) {
        sd::Tensor<float> tensor(local.shape());
        std::memcpy(tensor.data(),
                    all.data() + static_cast<size_t>(rank) * local_count,
                    local_count * sizeof(float));
        gathered->push_back(std::move(tensor));
    }
    return true;
}

} // namespace edgedit::parallel
