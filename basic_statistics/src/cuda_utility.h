#pragma once
#ifndef CUDA_UTILITY_LIXIONG_H
#define CUDA_UTILITY_LIXIONG_H

#include "../include/basic_statistics.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_device_runtime_api.h>

namespace basic_statistics {

inline cudaError_t
checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

inline void
Basic_Statistics::allocate_cuda_memory(precision** ptr, UInt n, UInt gpu_idx)
{
    if (n_gpu > 1)
        checkCuda(cudaSetDevice(gpu_idx));
    UInt size_byte = n * precision_byte;
    checkCuda(cudaMalloc((void**)ptr, size_byte));
}

inline void
Basic_Statistics::allocate_cuda_memory(precision** ptr, UInt n)
{
    UInt size_byte = n * precision_byte;
    checkCuda(cudaMalloc((void**)ptr, size_byte));
}

inline void
Basic_Statistics::host_copy_to_gpu(const precision* host_ptr,
                                   precision*       gpu_ptr,
                                   UInt             n,
                                   UInt             gpu_idx)
{
    if (n_gpu > 1)
        checkCuda(cudaSetDevice(gpu_idx));
    UInt size_byte = n * precision_byte;
    checkCuda(cudaMemcpyAsync(gpu_ptr, host_ptr, size_byte, cudaMemcpyHostToDevice));
}

inline void
Basic_Statistics::gpu_copy_to_host(const precision* gpu_ptr,
                                   precision*       host_ptr,
                                   UInt             n,
                                   UInt             gpu_idx)
{
    if (n_gpu > 1)
        cudaSetDevice(gpu_idx);
    UInt size_byte = n * precision_byte;
    checkCuda(cudaMemcpyAsync(host_ptr, gpu_ptr, size_byte, cudaMemcpyDeviceToHost));
}

inline void
Basic_Statistics::gpu_copy_to_host(const precision* gpu_ptr, precision* host_ptr, UInt n)
{
    UInt size_byte = n * precision_byte;
    checkCuda(cudaMemcpyAsync(host_ptr, gpu_ptr, size_byte, cudaMemcpyDeviceToHost));
}

inline void
Basic_Statistics::host_copy_to_gpu(const precision* host_ptr, precision* gpu_ptr, UInt n)
{
    UInt size_byte = n * precision_byte;
    checkCuda(cudaMemcpyAsync(gpu_ptr, host_ptr, size_byte, cudaMemcpyHostToDevice));
}




}









#endif
