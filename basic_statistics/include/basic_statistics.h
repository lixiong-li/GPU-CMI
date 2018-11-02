#pragma once

#ifndef BASIC_STATISTICS_v1_0_H
#define BASIC_STATISTICS_v1_0_H

#include <chrono>
#include <cstdint>
#include <vector>

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <stdio.h>

namespace basic_statistics
{

#ifdef USE_32_BIT_FLOAT
using precision = float;
#else
using precision = double;
#endif
constexpr unsigned precision_byte = sizeof(precision);

// name convension
using UInt = uint_fast32_t;
using std::nth_element;
using std::vector;

class Basic_Statistics
{
  public:
    // --------------------------------------
    // Constructor and desctructor
    // --------------------------------------
    Basic_Statistics(const precision* M, UInt n_row, UInt n_col);
    template<class MAT>
    Basic_Statistics(const MAT& M)
      : Basic_Statistics(M.data(), M.rows(), M.cols())
    {
    }

    ~Basic_Statistics();

    // --------------------------------------
    // Interface
    // --------------------------------------
    void mean(precision* mean_ptr);

    template<class VECTOR>
    void mean(VECTOR& vec)
    {
        assert(vec.size() == n_col);
        mean(vec.data());
    }

    void var(precision* var_ptr);
    template<class VECTOR>
    void var(VECTOR& vec)
    {
        assert(vec.size() == n_col);
        var(vec.data());
    }

    void cov(precision* cov_ptr);
    template<class VECTOR>
    void cov(VECTOR& mat)
    {
        assert(mat.size() == n_col * n_col);
        cov(mat.data());
    }

    void cov_within_group(precision * cov_ptr, UInt group_size);

    void apply_multiplier(const precision* multiplier);

    template<class VECTOR>
    void apply_multiplier(const VECTOR& multiplier)
    {
        assert(multiplier.size() == n_row);
        const precision* ptr = multiplier.data();
        apply_multiplier(ptr);
    }

  private:
    // ************************************************************
    //               IMPLIMENTATION DETAILS
    // ************************************************************

    // -------------------------------------
    // Pointers which hold the data
    // -------------------------------------
    const precision* M;
    UInt             n_row;
    UInt             n_col;

    // Pointers which refer to GPU memory
    vector<precision*> gpu_M;

    vector<precision*> gpu_mean_memory;
    vector<precision*> gpu_var_memory;

    vector<precision*> gpu_multiplier;

    vector<UInt> gpu_column_start;
    vector<UInt> gpu_column_length;

    // Pointers refering to CPU memory
    const precision* cpu_M;

    precision* cpu_mean_memory;
    precision* cpu_var_memory;
    precision* cpu_cov_memory;

    const precision* cpu_multiplier;

    UInt cpu_column_start;
    UInt cpu_column_length;

    // Internal State
    bool mean_calculated;
    bool var_calculated;
    bool cpu_only;

    // --------------------------------------
    // Cuda Property
    // --------------------------------------
    int        n_gpu;

    // Properties of the GPU
    vector<uint_fast64_t> GPU_memory_bandwidth;
    vector<uint_fast64_t> GPU_DRAM_size; // in byte
    vector<uint_fast64_t> GPU_max_gridsize;
    vector<UInt>          GPU_maximum_column;

    // --------------------------------------
    // Cuda State
    // --------------------------------------
    //vector<cudaStream_t> cuda_stream;

    // --------------------------------------
    // Calculation
    // --------------------------------------
    void calculate_mean();
    void gpu_mean(UInt gpu_idx);
    void cpu_mean();

    void calculate_var();
    void gpu_var(UInt gpu_idx);
    void cpu_var();

    void calculate_cov();

    void calculate_cov_group(precision * cov_ptr, UInt group_size);
    void gpu_cov_group(UInt group);

    // --------------------------------------
    // Auxiliary Function
    // --------------------------------------
    void divide_workload();
    void load_data();

    void get_gpu_info();
    void get_gpu_maximum_column();
    UInt gpu_maximum_column();

    inline void allocate_cuda_memory(precision** ptr, UInt n, UInt gpu_idx);
    inline void allocate_cuda_memory(precision** ptr, UInt n);
    inline void host_copy_to_gpu(const precision* host_ptr,
                                 precision*       gpu_ptr,
                                 UInt             n,
                                 UInt             gpu_idx);
    inline void host_copy_to_gpu(const precision* host_ptr, precision* gpu_ptr, UInt n);
    inline void gpu_copy_to_host(const precision* gpu_ptr,
                                 precision*       host_ptr,
                                 UInt             n,
                                 UInt             gpu_idx);
    inline void gpu_copy_to_host(const precision* gpu_ptr, precision* host_ptr, UInt n);
};

// ----------------------------------------------------
// Other auxiliary functions
// ----------------------------------------------------

template<typename T>
inline T
sum_vector(const vector<T>& v)
{
    T sum_v = 0;

    for (UInt i = 0; i < v.size(); ++i)
    {
        sum_v += v[i];
    }
    return sum_v;
}

// compuate quantile of a vector V
// return min(v in V:  #(V elements <= v) / #(V elements) >= alpha)
template<class V>
precision
quantile(const V& v, UInt length, precision alpha)
{
    assert(alpha >= 0);
    assert(alpha <= 1);

    precision* v_sorted = new precision[length];
#pragma omp parallel for simd
    for (UInt i = 0; i < length; ++i)
    {
        v_sorted[i] = v[i];
    }

    UInt pos = std::ceil(alpha * length) - 1;
    pos      = pos <= length - 1 ? pos : length - 1;

    nth_element(v_sorted, v_sorted + pos, v_sorted + length);

    precision q = v_sorted[pos];

    delete[] v_sorted;
    return q;
}
}

#endif // BASIC_STATISTICS_v1_0_H
