#include "../include/basic_statistics.h"
#include "cuda_kernel_function.cuh"
#include "cuda_utility.h"

using namespace basic_statistics;

void Basic_Statistics::calculate_mean()
{
    if (!mean_calculated)
    {
        if(!cpu_only)
        {
            for (UInt id = 0; id < n_gpu; ++id)
            {
                gpu_mean(id);
            }
        }

        if (cpu_column_length > 0)
        {
            for (UInt id = 0; id < n_gpu; ++id)
            {
                if (n_gpu > 1)  checkCuda(cudaSetDevice(id));
                cudaDeviceSynchronize();
            }
            cpu_mean();
        }

        mean_calculated = true;
    }
}

void Basic_Statistics::gpu_mean(UInt id)
{

    if (GPU_maximum_column[id] > 0)
    {
        if (n_gpu > 1)  checkCuda(cudaSetDevice(id));
        if (gpu_mean_memory[id] == nullptr)
        {
            allocate_cuda_memory(&(gpu_mean_memory[id]), gpu_column_length[id]);
        }
        if (gpu_multiplier[id] == nullptr)
        {
            precision * memory;
            allocate_cuda_memory(&memory, n_row);
            gpu_multiplier[id] = memory;
            fill<precision>(gpu_multiplier[id], 1, n_row, id);
        }

        rowV_multiply_Mat<precision>(gpu_multiplier[id],
                          gpu_M[id],
                          n_row,
                          gpu_column_length[id],
                          gpu_mean_memory[id],
                          1.0 / n_row);
    }
}

void Basic_Statistics::cpu_mean()
{
    if (cpu_column_length > 0)
    {
        if (cpu_mean_memory == nullptr)
        {
            cpu_mean_memory = new precision[cpu_column_length];
        }
        if (cpu_multiplier == nullptr)
        {
            #pragma omp parallel for
            for (UInt ci = 0; ci < cpu_column_length; ++ci)
            {
                const precision * M_ci = cpu_M + ci * n_row;
                precision mean_ci = 0;
                #pragma omp simd reduction(+:mean_ci)
                for (UInt ri = 0; ri < n_row; ++ri)
                {
                    mean_ci += M_ci[ri];
                }
                cpu_mean_memory[ci] = mean_ci;
            }

        } else
        {
            #pragma omp parallel for
            for (UInt ci = 0; ci < cpu_column_length; ++ci)
            {
                const precision * M_ci = cpu_M + ci * n_row;
                precision mean_ci = 0;
                #pragma omp simd reduction(+:mean_ci)
                for (UInt ri = 0; ri < n_row; ++ri)
                {
                    mean_ci += M_ci[ri] * cpu_multiplier[ri];
                }
                cpu_mean_memory[ci] = mean_ci;
            }
        }

        #pragma omp parallel for simd
        for (UInt ci = 0; ci < cpu_column_length; ++ci)
        {
            cpu_mean_memory[ci] /= n_row;
        }
    }
}

// ===============================================================================
// Calculate Variance
// ===============================================================================

void Basic_Statistics::calculate_var()
{
    if (!var_calculated)
    {
        if (!mean_calculated) calculate_mean();

        if(!cpu_only)
        {
            for (UInt id = 0; id < n_gpu; ++id)
            {
                gpu_var(id);
            }
        }

        if (cpu_column_length > 0)
        {
            for (UInt id = 0; id < n_gpu; ++id)
            {
                if (n_gpu > 1)  checkCuda(cudaSetDevice(id));
                cudaDeviceSynchronize();
            }
            cpu_var();
        }

        var_calculated = true;
    }
}


void Basic_Statistics::gpu_var(UInt id)
{
    if (GPU_maximum_column[id] > 0)
    {
        if (n_gpu > 1) checkCuda(cudaSetDevice(id));
        if (gpu_var_memory[id] == nullptr)
        {
            precision * memory;
            allocate_cuda_memory(&memory, gpu_column_length[id]);
            gpu_var_memory[id] = memory;
        }
        if (gpu_multiplier[id] == nullptr)
        {
            precision * memory;
            allocate_cuda_memory(&memory, n_row);
            gpu_multiplier[id] = memory;
            fill<precision>(gpu_multiplier[id], 1, n_row, id);
        }

        rowV_multiply_Mat_sq<precision>(gpu_multiplier[id],
                          gpu_M[id],
                          n_row,
                          gpu_column_length[id],
                          gpu_var_memory[id],
                          1.0 / n_row);

        X_minus_Y_square(gpu_var_memory[id], gpu_mean_memory[id], gpu_column_length[id], id);
    }
}

void Basic_Statistics::cpu_var()
{
    if (cpu_column_length > 0)
    {
        if (cpu_var_memory == nullptr)
        {
            cpu_var_memory = new precision [cpu_column_length];
        }
        if (cpu_multiplier == nullptr)
        {
            #pragma omp parallel for
            for (UInt ci = 0; ci < cpu_column_length; ++ci)
            {
                const precision * M_ci = cpu_M + ci * n_row;
                precision square_mean_ci = 0;
                #pragma omp simd reduction(+:square_mean_ci)
                for (UInt ri = 0; ri < n_row; ++ri)
                {
                    square_mean_ci += M_ci[ri] * M_ci[ri];
                }
                cpu_var_memory[ci] = square_mean_ci / n_row - cpu_mean_memory[ci] * cpu_mean_memory[ci];

            }
        } else
        {
            #pragma omp parallel for
            for (UInt ci = 0; ci < cpu_column_length; ++ci)
            {
                const precision * M_ci = cpu_M + ci * n_row;
                precision square_mean_ci = 0;
                #pragma omp simd reduction(+:square_mean_ci)
                for (UInt ri = 0; ri < n_row; ++ri)
                {
                    square_mean_ci += M_ci[ri] * M_ci[ri] * cpu_multiplier[ri];
                }
                cpu_var_memory[ci] = square_mean_ci / n_row - cpu_mean_memory[ci] * cpu_mean_memory[ci];
            }
        }
    }
}

// ===============================================================================
// Calculate Covariance
// ===============================================================================
void Basic_Statistics::calculate_cov()
{
    // Currently, only use CPU to calculate covariance and ignore mulitplier
    assert(cpu_multiplier == nullptr);

    // first calculate mean and variance
    precision * var_pt = new precision[n_col];
    var(var_pt);
    precision * mean_pt = new precision[n_col];
    mean(mean_pt);

    if (cpu_cov_memory == nullptr) cpu_cov_memory = new precision[n_col * n_col];

    #pragma omp parallel for schedule(dynamic)
    for (UInt c1 = 0; c1 < n_col; ++c1)
    {
        precision mean_c1 = mean_pt[c1];
        cpu_cov_memory[c1 + n_col * c1] = var_pt[c1];
        const precision * M_c1 = M + c1 * n_row;


        for (UInt c2 = 0; c2 < c1; ++c2)
        {
            precision mean_c2 = mean_pt[c2];
            const precision * M_c2 = M + c2 * n_row;

            precision cov_c1_c2 = 0;
            for (UInt i = 0; i < n_row; ++i)
            {
                cov_c1_c2 += M_c1[i] * M_c2[i];
            }
            cov_c1_c2 = cov_c1_c2 / n_row - mean_c1 * mean_c2;
            cpu_cov_memory[c1 + n_col * c2] = cov_c1_c2;
            cpu_cov_memory[c2 + n_col * c1] = cov_c1_c2;
        }
    }
    delete [] var_pt;
    delete [] mean_pt;
}


// =====================================================================
// Calculate Covariance matrix within Group
void Basic_Statistics::calculate_cov_group(precision * cov_ptr, UInt group_size)
{
    if (group_size == 0) throw std::runtime_error("group size should be positive.");
    if (n_col % group_size != 0) throw std::runtime_error("we should have n_col % group_size == 0.");

    // TO DO: relax the following restriction
    assert(n_gpu == 1);
    assert(cpu_column_length == 0);

    vector<precision> mean_M(n_col);
    vector<precision> var_M(n_col);
    mean(mean_M.data());
    var(var_M.data());

    UInt n_group = n_col / group_size;

    precision * gpu_cov_temp;
    allocate_cuda_memory(&gpu_cov_temp, n_col);
    precision * cov_temp = new precision[n_col];

    UInt n_cov_within_group = group_size * group_size;


    for (UInt i = 0; i + 1 < group_size; ++i)
    {
        rowV_multiply_Mat_group_offset<precision>(gpu_multiplier[0],
                                       gpu_M[0],
                                       n_row,
                                       gpu_column_length[0],
                                       gpu_cov_temp,
                                       group_size,
                                       i,
                                       1.0 / n_row);
        gpu_copy_to_host(gpu_cov_temp, cov_temp, n_col);
        checkCuda(cudaDeviceSynchronize());

        for (UInt n = 0; n < n_group; ++n)
        {
            for (UInt j = i + 1; j < group_size; ++j)
            {
                cov_ptr[n * n_cov_within_group + i + j * group_size] = cov_temp[j + n * group_size] - mean_M[j + n * group_size] * mean_M[i + n * group_size];
                cov_ptr[n * n_cov_within_group + j + i * group_size] = cov_ptr[n * n_cov_within_group + i + j * group_size];
            }
        }
    }

    for (UInt n = 0; n < n_group; ++n)
    {
        for (UInt i = 0; i < group_size; ++i)
        {
            cov_ptr[n * n_cov_within_group + i + i * group_size] = var_M[i + n * group_size];
        }
    }

    delete [] cov_temp;
    checkCuda(cudaFree(gpu_cov_temp));
}

