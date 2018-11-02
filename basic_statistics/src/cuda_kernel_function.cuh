#pragma once

#include "../include/basic_statistics.h"
#include "device_launch_parameters.h"
#include "cuda_parameter.h"
#include <iostream>

namespace basic_statistics {

// fill X with constant 'v'
template <typename T>
__global__
void kernel_fill(T* __restrict__ X,
          const T v,
          const unsigned int n)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride)
    {
      X[i] = v;
    }
}

template<typename T>
void fill(T* __restrict__ X,
          const T v,
          const unsigned int n,
          unsigned int id)
{
    kernel_fill<<<grid_size, block_size>>>(X, v, n);
}


// calculate Y = X^2
template <typename T>
__global__
void kernel_X_square(const T* __restrict__ X,
                     T* __restrict__ Y,
                     const unsigned int n)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride)
    {
      Y[i] = X[i] * X[i];
    }
}

template<typename T>
void X_square(const T* __restrict__ X,
              T* __restrict__ Y,
              const unsigned int n,
              const unsigned int gpu_id)
{
    kernel_X_square<<<grid_size, block_size>>>(X, Y, n);
}


// calculate X - Y^2 and store it in X
template <typename T>
__global__
void kernel_X_minus_Y_square(T* __restrict__ X,
                             const T* __restrict__ Y,
                             const unsigned int n)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride)
    {
      X[i] -= Y[i] * Y[i];
    }
}

template <typename T>
void X_minus_Y_square(T* __restrict__ X,
                      const T* __restrict__ Y,
                      const unsigned int n,
                      const unsigned int id)
{
    kernel_X_minus_Y_square<<<grid_size, block_size>>>(X, Y, n);
}


// calculate Y = alpha * rowV * M, where rowV is a row vector and M is a matrix
// and alpha is a scalar
template<typename T>
__global__
void kernel_rowV_multiply_Mat(const T* __restrict__ rowV,
                              const T* __restrict__ M,
                              const unsigned int M_row,
                              T* __restrict__ Y,
                              const T alpha)
{
    __shared__ T sdata[block_size];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    const unsigned int col_offset = blockIdx.x * M_row;

//        sdata[tid] = 0;
//        for (unsigned int i = tid; i < M_row; i += stride)
//        {
//            sdata[tid] += rowV[i] * M[i + col_offset];
//        }

//     to increase memroy bandwidth efficienty of the above code, unroll the for loop

    sdata[tid] = 0;
    unsigned int i = tid;
    while (i + stride < M_row )
    {
        sdata[tid] += rowV[i] * M[i + col_offset]
                + rowV[i + stride] * M[i + stride + col_offset];
        i += 2 * stride;
    }
    if (i < M_row) sdata[tid] += rowV[i] * M[i + col_offset];
    __syncthreads();

    // unroll parallel reduction algorithm
    if (block_size >= 1024) { if (tid < 512) {sdata[tid] += sdata[tid + 512];} __syncthreads();}
    if (block_size >= 512) { if (tid < 256) {sdata[tid] += sdata[tid + 256];} __syncthreads();}
    if (block_size >= 256) { if (tid < 128) {sdata[tid] += sdata[tid + 128];} __syncthreads();}
    if (block_size >= 128) { if (tid < 64) {sdata[tid] += sdata[tid + 64];} __syncthreads();}
    if (block_size >= 64) { if (tid < 32) {sdata[tid] += sdata[tid + 32];} __syncthreads();}
    if (block_size >= 32) { if (tid < 16) {sdata[tid] += sdata[tid + 16];} __syncthreads();}
    if (block_size >= 16) { if (tid < 8) {sdata[tid] += sdata[tid + 8];} __syncthreads();}
    if (block_size >= 8) { if (tid < 4) {sdata[tid] += sdata[tid + 4];} __syncthreads();}
    if (block_size >= 4) { if (tid < 2) {sdata[tid] += sdata[tid + 2];} __syncthreads();}
    if (block_size >= 2) { if (tid < 1) {sdata[tid] += sdata[tid + 1];} __syncthreads();}

    if (tid == 0) Y[blockIdx.x] = alpha * sdata[0];
}

template <typename T>
void rowV_multiply_Mat(const T* __restrict__ rowV,
                       const T* __restrict__ M,
                       const unsigned int M_row,
                       const unsigned int M_col,
                       T* __restrict__ Y,
                       const T alpha)
{
    kernel_rowV_multiply_Mat<T><<<M_col, block_size>>>(rowV, M, M_row, Y, alpha);
}


// calculate Y = alpha * rowV * (M .* M), where rowV is a row vector and M is a matrix
// and alpha is a scalar
template<typename T>
__global__
void kernel_rowV_multiply_Mat_sq(const T* __restrict__ rowV,
                              const T* __restrict__ M,
                              const unsigned int M_row,
                              T* __restrict__ Y,
                              const T alpha)
{
    __shared__ T sdata[block_size];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    const unsigned int col_offset = blockIdx.x * M_row;

//        sdata[tid] = 0;
//        for (unsigned int i = tid; i < M_row; i += stride)
//        {
//            sdata[tid] += rowV[i] * M[i + col_offset] * M[i + col_offset];
//        }

    // to increase memroy bandwidth efficienty of the above code, unroll the for loop

    sdata[tid] = 0;
    unsigned int i = tid;
    while (i +  stride < M_row)
    {
        sdata[tid] += rowV[i] * M[i + col_offset] *  M[i + col_offset]
                + rowV[i + stride] * M[i + stride + col_offset] * M[i + stride + col_offset];
        i += 2 * stride;
    }
    if (i < M_row) sdata[tid] += rowV[i] * M[i + col_offset] * M[i + col_offset];
    __syncthreads();

    // unroll parallel reduction algorithm
    if (block_size >= 1024) { if (tid < 512) {sdata[tid] += sdata[tid + 512];} __syncthreads();}
    if (block_size >= 512) { if (tid < 256) {sdata[tid] += sdata[tid + 256];} __syncthreads();}
    if (block_size >= 256) { if (tid < 128) {sdata[tid] += sdata[tid + 128];} __syncthreads();}
    if (block_size >= 128) { if (tid < 64) {sdata[tid] += sdata[tid + 64];} __syncthreads();}
    if (block_size >= 64) { if (tid < 32) {sdata[tid] += sdata[tid + 32];} __syncthreads();}
    if (block_size >= 32) { if (tid < 16) {sdata[tid] += sdata[tid + 16];} __syncthreads();}
    if (block_size >= 16) { if (tid < 8) {sdata[tid] += sdata[tid + 8];} __syncthreads();}
    if (block_size >= 8) { if (tid < 4) {sdata[tid] += sdata[tid + 4];} __syncthreads();}
    if (block_size >= 4) { if (tid < 2) {sdata[tid] += sdata[tid + 2];} __syncthreads();}
    if (block_size >= 2) { if (tid < 1) {sdata[tid] += sdata[tid + 1];} __syncthreads();}

    if (tid == 0) Y[blockIdx.x] = alpha * sdata[0];
}

template <typename T>
void rowV_multiply_Mat_sq(const T* __restrict__ rowV,
                       const T* __restrict__ M,
                       const unsigned int M_row,
                       const unsigned int M_col,
                       T* __restrict__ Y,
                       const T alpha)
{
    kernel_rowV_multiply_Mat_sq<T><<<M_col, block_size>>>(rowV, M, M_row, Y, alpha);
}

// calculate Y = alpha * rowV * (M .* M[:, within_group_idx]), where rowV is a row vector and M is a matrix
// and alpha is a scalar
template<typename T>
__global__
void kernel_rowV_multiply_Mat_group_offset(const T* __restrict__ rowV,
                                        const T* __restrict__ M,
                                        const unsigned int M_row,
                                        T* __restrict__ Y,
                                        const unsigned int group_size,
                                        const unsigned int within_group_idx,
                                        const T alpha)
{
    __shared__ T sdata[block_size];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    const unsigned int col_offset = blockIdx.x * M_row;

    const unsigned int group_idx = blockIdx.x / group_size;
    const unsigned int group_col_offset = (group_idx * group_size + within_group_idx) * M_row;

    sdata[tid] = 0;
    if (blockIdx.x % group_size > within_group_idx)
    {
        unsigned int i = tid;
        while (i +  stride < M_row)
        {
            sdata[tid] += rowV[i] * M[i + col_offset] *  M[i + group_col_offset]
                    + rowV[i + stride] * M[i + stride + col_offset] * M[i + stride + group_col_offset];
            i += 2 * stride;
        }
        if (i < M_row) sdata[tid] += rowV[i] * M[i + col_offset] * M[i + group_col_offset];
        __syncthreads();

        // unroll parallel reduction algorithm
        if (block_size >= 1024) { if (tid < 512) {sdata[tid] += sdata[tid + 512];} __syncthreads();}
        if (block_size >= 512) { if (tid < 256) {sdata[tid] += sdata[tid + 256];} __syncthreads();}
        if (block_size >= 256) { if (tid < 128) {sdata[tid] += sdata[tid + 128];} __syncthreads();}
        if (block_size >= 128) { if (tid < 64) {sdata[tid] += sdata[tid + 64];} __syncthreads();}
        if (block_size >= 64) { if (tid < 32) {sdata[tid] += sdata[tid + 32];} __syncthreads();}
        if (block_size >= 32) { if (tid < 16) {sdata[tid] += sdata[tid + 16];} __syncthreads();}
        if (block_size >= 16) { if (tid < 8) {sdata[tid] += sdata[tid + 8];} __syncthreads();}
        if (block_size >= 8) { if (tid < 4) {sdata[tid] += sdata[tid + 4];} __syncthreads();}
        if (block_size >= 4) { if (tid < 2) {sdata[tid] += sdata[tid + 2];} __syncthreads();}
        if (block_size >= 2) { if (tid < 1) {sdata[tid] += sdata[tid + 1];} __syncthreads();}
    }

    if (tid == 0) Y[blockIdx.x] = alpha * sdata[0];
}


template <typename T>
void rowV_multiply_Mat_group_offset(const T* __restrict__ rowV,
                                   const T* __restrict__ M,
                                   const unsigned int M_row,
                                   const unsigned int M_col,
                                   T* __restrict__ Y,
                                    const unsigned int group_size,
                                    const unsigned int within_group_idx,
                                   const T alpha)
{
    if (within_group_idx >= group_size) throw std::runtime_error("within_group_idx should be less than group_size");
    if (group_size > M_col) throw std::runtime_error("group_size should be less than M_col");

    kernel_rowV_multiply_Mat_group_offset<T><<<M_col, block_size>>>(rowV, M, M_row, Y, group_size, within_group_idx, alpha);
}

}

