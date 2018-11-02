#include "../include/basic_statistics.h"
#include "cuda_parameter.h"
#include "cuda_utility.h"
#include <stdexcept>

using namespace basic_statistics;

Basic_Statistics::Basic_Statistics(const precision *M, UInt n_row, UInt n_col)
    : M(M)
    , n_row(n_row)
    , n_col(n_col)
    , mean_calculated(false)
    , var_calculated(false)
{
    // only use cpu for small matrix
    if (n_row * n_col < 500 or n_col < 50)
    {
        cpu_only = true;
        n_gpu = 0;
    } else
    {
        cpu_only = false;
    }

    // divide workload between devices
    divide_workload();
    assert(n_gpu > 0 || cpu_only);
    gpu_M = vector<precision*>(n_gpu, nullptr);
    gpu_mean_memory = vector<precision*>(n_gpu, nullptr);
    gpu_var_memory = vector<precision*>(n_gpu, nullptr);
    gpu_multiplier = vector<precision*>(n_gpu, nullptr);

    cpu_M = nullptr;
    cpu_mean_memory = nullptr;
    cpu_var_memory = nullptr;
    cpu_cov_memory = nullptr;
    cpu_multiplier = nullptr;

    // load data to each device
    load_data();

}

void Basic_Statistics::get_gpu_info()
{
    checkCuda(cudaGetDeviceCount(&n_gpu));
    assert(n_gpu > 0);

    GPU_DRAM_size = vector<uint_fast64_t>(n_gpu);
    GPU_memory_bandwidth = vector<UInt>(n_gpu);
    GPU_max_gridsize = vector<uint_fast64_t>(n_gpu);

    for (int i = 0; i < n_gpu; ++i)
    {
        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceProperties(&prop, i));

        GPU_DRAM_size[i] = prop.totalGlobalMem;
        GPU_memory_bandwidth[i] = prop.memoryClockRate * prop.memoryBusWidth / 8000;
        GPU_max_gridsize[i] = prop.maxGridSize[0];
    }
}

void Basic_Statistics::get_gpu_maximum_column()
{
    get_gpu_info();
    UInt system_reserve = 500 * 1024 * 1024;   // reserve 500 MB for system usage
    GPU_maximum_column = vector<UInt>(n_gpu);
    for (UInt i = 0; i < n_gpu; ++i)
    {
        if (GPU_DRAM_size[i] <= system_reserve + n_row * precision_byte)
        {
            GPU_maximum_column[i] = 0;
        }
        else
        {
            uint_fast64_t temp = (GPU_DRAM_size[i] - system_reserve) / precision_byte - n_row;
            GPU_maximum_column[i] = temp / (n_row + 2);
            GPU_maximum_column[i] = std::min((uint_fast64_t)GPU_maximum_column[i],  GPU_max_gridsize[i]);
        }
    }
}

UInt Basic_Statistics::gpu_maximum_column()
{
    get_gpu_maximum_column();
    return sum_vector(GPU_maximum_column);
}

void Basic_Statistics::divide_workload()
{
    UInt column_assigned = 0;
    if (!cpu_only)
    {
        get_gpu_info();
        get_gpu_maximum_column();

        vector<UInt> ideal_workload(n_gpu, 0);
        UInt total_gpu_multiprocess = sum_vector(GPU_memory_bandwidth);
        for (UInt i = 0; i < n_gpu; ++i)
        {
            if (i != n_gpu - 1)
            {
                ideal_workload[i] = n_col * GPU_memory_bandwidth[i] / total_gpu_multiprocess;
            }
            else
            {
                ideal_workload[i] = n_col - sum_vector(ideal_workload);
            }
        }

        gpu_column_start = vector<UInt>(n_gpu, 0);
        gpu_column_length = vector<UInt>(n_gpu, 0);

        for (UInt i = 0; i < n_gpu; ++i)
        {
            gpu_column_start[i] = column_assigned;
            gpu_column_length[i] = std::min((UInt)ideal_workload[i], (UInt)GPU_maximum_column[i]);
            column_assigned += gpu_column_length[i];
        }
    }

    cpu_column_start = column_assigned;
    cpu_column_length = n_col - sum_vector(gpu_column_length);
    //if (cpu_column_length > 0) printf("part of the work (%d columns)is done on cpu.\n", cpu_column_length);
}

void Basic_Statistics::load_data()
{
    if (!cpu_only)
    {
        for (int i = 0; i < n_gpu; ++i)
        {
            if (n_gpu > 1)  checkCuda(cudaSetDevice(i));

            UInt length = n_row * gpu_column_length[i];
            precision ** memory = gpu_M.data() + i;
            allocate_cuda_memory(memory, length);

            const precision * M_address = M + n_row * gpu_column_start[i];
            host_copy_to_gpu(M_address, gpu_M[i], length);
        }
    }

    cpu_M = M + cpu_column_start * n_row;
}

Basic_Statistics::~Basic_Statistics()
{
    if (!cpu_only)
    {
        for (UInt id = 0; id < n_gpu; ++id)
        {
            if (n_gpu > 1)  checkCuda(cudaSetDevice(id));
            if (gpu_M[id] != nullptr) cudaFree(gpu_M[id]);
            if (gpu_mean_memory[id] != nullptr) cudaFree(gpu_mean_memory[id]);
            if (gpu_var_memory[id] != nullptr) cudaFree(gpu_var_memory[id]);
            if (gpu_multiplier[id] != nullptr) cudaFree(gpu_multiplier[id]);
        }
    }


    if (cpu_mean_memory != nullptr) delete [] cpu_mean_memory;
    if (cpu_var_memory != nullptr) delete [] cpu_var_memory;
    if (cpu_cov_memory != nullptr) delete [] cpu_cov_memory;
}
