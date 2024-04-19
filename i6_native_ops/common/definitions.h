#ifndef _COMMON_DEFINITIONS_H
#define _COMMON_DEFINITIONS_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

#define Ndarray torch::Tensor
#define Ndarray_DEV_DATA(x) ((float*)(x).data_ptr())
#define Ndarray_DEV_DATA_int32(x) ((int32_t*)(x).data_ptr())
#define Ndarray_DEV_DATA_int32_scalar(x) (x).scalar<int32>()()
#define Ndarray_HOST_DIMS(x) ((x).sizes())
#define Ndarray_DIMS(x) ((x).sizes())
#define Ndarray_DIMS Ndarray_HOST_DIMS
#define Ndarray_NDIM(x) (x).ndimension()
#define Ndarray_dtype_size(x) torch::elementSize((x).scalar_type())
typedef long long Ndarray_DIM_Type;
#define Ndarray_SIZE(x) (x).numel()
#define Ndarray_STRIDE(x, dim) ((x).stride(dim))

#define CUDA_CUR_STREAM (0)  // default stream

#define DEF_KERNEL __global__
#define DEV_FUNC __device__
#define HOST_FUNC __host__

#define elem_atomic_add(x, v) atomicAdd(x, v)
#define elem_atomic_min(x, v) atomicMin(x, v)
#define elem_atomic_cas(a, c, v) atomicCAS(a, c, v)

#define int_as_float __int_as_float
#define float_as_int __float_as_int

#define INF_F CUDART_INF_F
#define NAN_F CUDART_NAN_F

#define Ndarray_memcpy(y, x, size) \
    (cudaMemcpyAsync(y, x, size, cudaMemcpyDeviceToDevice, CUDA_CUR_STREAM))
#define Ndarray_memset(s, c, size) (cudaMemsetAsync(s, c, size, CUDA_CUR_STREAM))

#define DIM_GRID 128
#define DIM_BLOCK 512

// <<<DimGrid,DimBlock,ShmemSize|0,Stream|0>>>.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration
#define start_dev_kernel(kernel, args) (kernel<<<DIM_GRID, DIM_BLOCK, 0, CUDA_CUR_STREAM>>> args);
#define start_dev_kernel2(kernel, dim_grid, dim_block, shared_size, args) \
    (kernel<<<dim_grid, dim_block, shared_size, CUDA_CUR_STREAM>>> args);

#define DEF_SHARED(type, name) extern __shared__ type name[];

static const char* _cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

static void _cudaHandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("NativeOp: CUDA runtime error: '%s' in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}

static void _cudaHandleError(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("NativeOp: cuBLAS runtime error: '%s' in %s at line %d\n", _cudaGetErrorEnum(status),
               file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(status) (_cudaHandleError(status, __FILE__, __LINE__))
#define HANDLE_LAST_ERROR() (HANDLE_ERROR(cudaGetLastError()))

long Ndarray_get_n_total_elements(Ndarray& a) {
    long c = 1;
    for (long i = 0; i < Ndarray_NDIM(a); ++i)
        c *= Ndarray_DIMS(a)[i];
    return c;
}

void _Ndarray_set_zero(Ndarray& a) {
    long size = Ndarray_get_n_total_elements(a) * Ndarray_dtype_size(a);
    Ndarray_memset(Ndarray_DEV_DATA(a), 0, size);
}
#define Ndarray_set_zero _Ndarray_set_zero

#define assert_cmp(a, cmp, b)                  \
    if (!((a)cmp(b))) {                        \
        printf("NativeOp assertion failed: "); \
        printf("%s %s %s, ", #a, #cmp, #b);    \
        printf(_format_for_type(a), a);        \
        printf(" " #cmp " ");                  \
        printf(_format_for_type(b), b);        \
        printf("\n");                          \
        assert((a)cmp(b));                     \
    }

template<typename T>
DEV_FUNC HOST_FUNC const char* _format_for_type(const T&) {
    printf("ERROR: _format_for_type(%s) not implemented, aborting\n", typeid(T).name());
    assert(0);
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const char&) {
    return "%c";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const unsigned char&) {
    return "%u";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const short&) {
    return "%hi";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const unsigned short&) {
    return "%hu";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const int&) {
    return "%i";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const unsigned int&) {
    return "%u";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const long&) {
    return "%li";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const unsigned long&) {
    return "%lu";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const long long&) {
    return "%lli";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const unsigned long long&) {
    return "%llu";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const float&) {
    return "%f";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const double&) {
    return "%f";
}
template<>
DEV_FUNC HOST_FUNC const char* _format_for_type(const long double&) {
    return "%Lf";
}

static inline void* device_malloc(size_t size) {
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}
static inline void device_free(void* ptr) {
    cudaFree(ptr);
}

#endif // _COMMON_DEFINITIONS_H