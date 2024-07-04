#ifndef MONOTONIC_RNNT_REDUCE_H
#define MONOTONIC_RNNT_REDUCE_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "rnnt_helper.h"
#include "status.h"

const int warp_size = 32;

/*
 *  Template structure for a parallel reduction within a CUDA thread block
 *  Template Parameters:
 *      NT: Number of threads per block
 *      T: Data type of the elements
 *      Rop: Binary associative reduction operation
 */
template <int NT, typename T, typename Rop>
struct CTAReduce {
    struct Storage {
        T shared[NT];
    };

    __device__ static T

    /*
     *  Reduce the elements within a CUDA block using shared memory and shuffle operations.
     *  Parameters:
     *      tid: Thread ID within the block
     *      x: Initial value of the reduction for the thread
     *      storage: Shared memory storage for the reduction
     *      count: Number of elements to reduce
     *      g: Binary reduction function
     *  Compute output = g(x[0], g(x[1], g(x[2], ..., g(x[count-2], x[count-1]))))
     *  Requires that the reduce operation g is associative, i.e. g(a, g(b, c)) = g(g(a, b), c)
     *  Examples could be sum, product or maximum of all elements
     */
    reduce(int tid, T x, Storage &storage, int count, Rop g) {
        T *s = storage.shared;
        s[tid] = x;
        __syncthreads();

        // Fold the data in half with each pass.
#pragma unroll
        for (int offset = NT / 2; offset >= warp_size; offset /= 2) {
            if (tid + offset < count && tid < offset) {
                // Read from the right half and store to the left half.
                x = g(x, s[offset + tid]);
                s[tid] = x;
            }
            __syncthreads();
        }

        T shuffle;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
#if CUDART_VERSION < 9000
            shuff = __shfl_down(x, offset);
#else
            shuffle = __shfl_down_sync(0xFFFFFFFF, x, offset);
#endif
            if (tid + offset < count && tid < offset) x = g(x, shuffle);
        }
        return x;
    }
};

/*
 *  Reduce rows of a 2D matrix using a specified operation and store the result in an output array.
 *  Parameters:
 *      f: Unary operation applied to each element before reduction.
 *      g: Binary operation used for reduction
 *      acts: Input matrix
 *      output: Output vector where results are stored
 *      num_rows: Number of rows in the matrix
 *
 *  Returns:
 *      For each column j, output[j] = g(f(acts[0, j]), g(f(acts[1, j]), ...))
 */
template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_rows(Iop f, Rop g, const T *const acts, T *output, int num_rows) {
    typedef CTAReduce<NT, T, Rop> R;
    __shared__ typename R::Storage storage;

    int tid = static_cast<int>(threadIdx.x);
    int idx = tid;
    int col = static_cast<int>(blockIdx.x);
    T curr;

    // Each block works on a column
    if (idx < num_rows) {
        curr = f(acts[col * num_rows + idx]);
    }
    idx += NT;

    while (idx < num_rows) {
        curr = g(curr, f(acts[col * num_rows + idx]));
        idx += NT;
    }

    // Sum thread-totals over the CTA.
    curr = R::reduce(tid, curr, storage, num_rows, g);

    // Store result in out
    if (tid == 0) output[col] = curr;
}

/*
 * Similar to `reduce_rows` but each element is first subtracted by the current element from the output array
 * before the unary operation `f` is applied.
 * For each column j, output[j] = -output[j] - log(g(f(acts[0, j] - output[j]), g(f(acts[1, j] - output[j]), ...)))
 */
template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_minus(Iop f, Rop g, const T *const acts, T *output, int num_rows) {
    typedef CTAReduce<NT, T, Rop> R;
    __shared__ typename R::Storage storage;

    int tid = static_cast<int>(threadIdx.x);
    int idx = tid;
    int col = static_cast<int>(blockIdx.x);
    T curr;
    T max = output[col];

    // Each block works on a column
    if (idx < num_rows) {
        curr = f(acts[col * num_rows + idx] - max);
    }
    idx += NT;

    while (idx < num_rows) {
        curr = g(curr, f(acts[col * num_rows + idx] - max));
        idx += NT;
    }

    // Sum thread-totals over the CTA.
    curr = R::reduce(tid, curr, storage, num_rows, g);

    // Store result in out
    if (tid == 0) output[col] = -max - log(curr);
}

/*
 *  Helper struct to execute the kernel functions with appropriate grid and block dimensions
 */
struct ReduceHelper {
    template <typename T, typename Iof, typename Rof>
    static void impl(Iof f, Rof g, const T *const acts, T *output, int num_rows, int num_cols, bool minus,
                     cudaStream_t stream) {
        if (minus) {
            reduce_minus<128><<<num_cols, 128, 0, stream>>>(f, g, acts, output, num_rows);
        } else {
            reduce_rows<128><<<num_cols, 128, 0, stream>>>(f, g, acts, output, num_rows);
        }
    }
};

/*
 *  General reduction interface that can apply any operation
 */
template <typename T, typename Iof, typename Rof>
RNNTStatus reduce(Iof f, Rof g, const T *const acts, T *output, int rows, int cols, bool minus, cudaStream_t stream) {
    ReduceHelper::impl(f, g, acts, output, rows, cols, minus, stream);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return RNNT_STATUS_EXECUTION_FAILED;

    return RNNT_STATUS_SUCCESS;
}

/*
 *  Applies the exponential function to each element before reducing by addition, i.e.
 *  output[j] = sum_i exp(acts[i, j])
 */
template <typename T>
RNNTStatus reduce_exp(const T *const acts, T *get_denom, int rows, int cols, bool minus, cudaStream_t stream) {
    return reduce(rnnt_helper::exponential<T>(), rnnt_helper::add<T>(), acts, get_denom, rows, cols, minus, stream);
}

/*
 *  Finds the maximum value in each column
 *  output[j] = max_i exp(acts[i, j])
 */
template <typename T>
RNNTStatus reduce_max(const T *const acts, T *get_denom, int rows, int cols, bool minus, cudaStream_t stream) {
    return reduce(rnnt_helper::identity<T>(), rnnt_helper::maximum<T>(), acts, get_denom, rows, cols, minus, stream);
}

#endif  // MONOTONIC_RNNT_REDUCE_H
