#include "cuda_utility.h"

using namespace basic_statistics;

void Basic_Statistics::mean(precision *mean_output)
{
    assert(mean_output != nullptr);
    if (!mean_calculated) calculate_mean();

    if (!cpu_only)
    {
        for (UInt i = 0; i < n_gpu; ++i)
        {
            if (n_gpu > 1) checkCuda(cudaSetDevice(i));
            gpu_copy_to_host(gpu_mean_memory[i],
                             mean_output + gpu_column_start[i],
                             gpu_column_length[i]);
        }

        for (UInt id = 0; id < n_gpu; ++id)
        {
            if (n_gpu > 1) checkCuda(cudaSetDevice(id));
            checkCuda(cudaDeviceSynchronize());
        }
    }


    if (cpu_column_length > 0)
    {
        #pragma omp simd
        for (UInt ci = 0; ci < cpu_column_length; ++ci)
        {
            mean_output[cpu_column_start + ci] = cpu_mean_memory[ci];
        }
    }
}

void Basic_Statistics::var(precision *var_output)
{
    assert(var_output != nullptr);
    if (!var_calculated) calculate_var();

    if (!cpu_only)
    {
        for (UInt i = 0; i < n_gpu; ++i)
        {
            if (n_gpu > 1) checkCuda(cudaSetDevice(i));
            gpu_copy_to_host(gpu_var_memory[i],
                             var_output + gpu_column_start[i],
                             gpu_column_length[i]);
        }

        for (UInt id = 0; id < n_gpu; ++id)
        {
            if (n_gpu > 1) checkCuda(cudaSetDevice(id));
            checkCuda(cudaDeviceSynchronize());
        }
    }


    if (cpu_column_length > 0)
    {
        #pragma omp simd
        for (UInt ci = 0; ci < cpu_column_length; ++ci)
        {
            var_output[cpu_column_start + ci] = cpu_var_memory[ci];
        }
    }
}

void Basic_Statistics::cov(precision *cov_ptr)
{
    assert(cov_ptr != nullptr);
    calculate_cov();

    for (UInt i = 0; i < n_col * n_col; ++i)
    {
        cov_ptr[i] = cpu_cov_memory[i];
    }
}

void Basic_Statistics::apply_multiplier(const precision *multiplier)
{
    // apply multiplier to gpu
    if (!cpu_only)
    {
        for (UInt i = 0; i < n_gpu; ++i)
        {
            if (n_gpu > 1) checkCuda(cudaSetDevice(i));
            if (gpu_multiplier[i] == nullptr)
            {
                allocate_cuda_memory(&(gpu_multiplier[i]), n_row);
            }
            host_copy_to_gpu(multiplier, gpu_multiplier[i], n_row);
        }
    }

    // apply multiplier to cpu
    cpu_multiplier = multiplier;

    mean_calculated = false;
    var_calculated = false;

}

void Basic_Statistics::cov_within_group(precision *cov_ptr, UInt group_size)
{
    calculate_cov_group(cov_ptr, group_size);
}
