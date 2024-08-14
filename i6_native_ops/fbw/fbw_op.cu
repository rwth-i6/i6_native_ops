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

#include <DebugOptions.h>

namespace py = pybind11;

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
    TORCH_CHECK(err == cudaSuccess, "NativeOp: CUDA runtime error: ", cudaGetErrorString(err), " in ", file, " at line ", line);
}

static void _cudaHandleError(cublasStatus_t status, const char* file, int line) {
    TORCH_CHECK(err == cudaSuccess, "NativeOp: cuBLAS runtime error: ", cudaGetErrorString(err), " in ", file, " at line ", line);
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

DEF_KERNEL
void set_start_states(float* states, unsigned* start_states) {
    unsigned state_idx = start_states[blockIdx.x * blockDim.x + threadIdx.x];
    states[state_idx]  = 0.0;
}

DEF_KERNEL
void fill_array(float* array, float value, unsigned size) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = value;
    }
}

DEF_KERNEL
void remove_inf(float* array, unsigned size) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = fminf(array[idx], 1e32);
    }
}

DEV_FUNC
float prob_add(float a, float b) {
    float diff = a - b;
    if (isnan(diff)) {
        return INF_F;
    }
    else {
        return -log1pf(expf(-fabsf(diff))) + fminf(a, b);
    }
}

DEV_FUNC
void atomic_prob_add(float* a, float b) {
    int* addr = (int*)a;
    int  old  = float_as_int(*a);
    int  assumed;
    do {
        assumed = old;
        old     = elem_atomic_cas(addr, assumed, float_as_int(prob_add(int_as_float(old), b)));
    } while (old != assumed);
}

DEF_KERNEL
void init_bwd_state_buffer(float* states, unsigned* end_states, unsigned t, unsigned max_t,
                           unsigned* seq_lens) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t + 1 == seq_lens[idx]) {
        unsigned state_idx = end_states[idx];
        states[state_idx]  = 0.0;
    }
}

DEF_KERNEL
void next_frame(bool fwd, unsigned num_edges, unsigned num_emissions, unsigned* sequence_idxs,
                unsigned* from_buffer, unsigned* to_buffer, float* weight_buffer,
                unsigned* emission_idxs, float* prev_frame, float* next_frame, float* am_scores,
                float* edge_buffer) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) {
        return;
    }

    unsigned from     = from_buffer[idx];
    float    prev_val = prev_frame[from];
    if (isinf(prev_val)) {
        edge_buffer[idx] = INF_F;
        return;
    }

    unsigned to           = to_buffer[idx];
    unsigned emission_idx = emission_idxs[idx];
    float    edge_weight  = weight_buffer[idx];
    unsigned sequence_idx = sequence_idxs[idx];

    float val = prev_val + edge_weight + am_scores[sequence_idx * num_emissions + emission_idx];

    if (fwd) {
        edge_buffer[idx] += val;
    }
    else {
        edge_buffer[idx] += prev_val;
    }
    atomic_prob_add(next_frame + to, val);
}

DEF_KERNEL
void normalize(float* buffer, unsigned* sequence_idxs, unsigned num_edges, unsigned num_seqs,
               float* sum_output) {
    DEF_SHARED(float, sum);

    buffer += blockIdx.x * num_edges;

    for (unsigned s = 0u; s < num_seqs; s++) {
        sum[s] = INF_F;
    }

    for (unsigned e = 0u; e < num_edges; e++) {
        unsigned s = sequence_idxs[e];
        sum[s]     = prob_add(sum[s], buffer[e]);
    }

    for (unsigned s = 0ul; s < num_seqs; s++) {
        if (isinf(sum[s])) {
            // if the frame is empty (happens due to batching of seqs with
            // unequal length), set it to 0
            sum_output[blockIdx.x * num_seqs + s] = 0.0;
        }
        else {
            sum_output[blockIdx.x * num_seqs + s] = sum[s];
        }
    }

    for (unsigned e = 0u; e < num_edges; e++) {
        unsigned s = sequence_idxs[e];
        buffer[e] -= sum[s];
    }
}

DEF_KERNEL
void compute_result(float* edge_buffer, float* out, unsigned* emission_idxs,
                    unsigned* sequence_idxs, unsigned frame_stride, unsigned seq_stride,
                    unsigned num_frames, unsigned num_seqs, unsigned num_edges) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frames * num_edges) {
        return;
    }

    unsigned e_idx        = idx % num_edges;
    unsigned frame        = idx / num_edges;
    unsigned emission_idx = emission_idxs[e_idx];
    unsigned seq_idx      = sequence_idxs[e_idx];
    float    score        = edge_buffer[idx];

    atomic_prob_add(out + frame * frame_stride + seq_idx * seq_stride + emission_idx, score);
}

void write_alignment_to_file(float* d_state_buffer, unsigned* d_seq_lens, unsigned* d_start_states,
                             unsigned* d_end_states, float pruning, unsigned n_frames,
                             unsigned n_seqs, unsigned n_states, unsigned batch_idx) {
    std::vector<float>    state_buffer((n_frames + 1u) * n_states);
    std::vector<unsigned> seq_lens(n_seqs);
    std::vector<unsigned> start_states(n_seqs);
    std::vector<unsigned> end_states(n_seqs);

    HANDLE_ERROR(cudaMemcpy(state_buffer.data(), d_state_buffer,
                            state_buffer.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(start_states.data(), d_start_states,
                            start_states.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(end_states.data(), d_end_states, end_states.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(seq_lens.data(), d_seq_lens, seq_lens.size() * sizeof(unsigned),
                            cudaMemcpyDeviceToHost));

    for (unsigned seq = 0u; seq < n_seqs; seq++) {
        std::stringstream filename;
        filename << "alignment.dump." << batch_idx << '.' << seq;
        std::ofstream out(filename.str().c_str(), std::ios::out | std::ios::trunc);
        for (unsigned t = 0u; t < n_frames; t++) {
            if (t > 0u && t >= seq_lens[seq]) {
                break;
            }
            float sum = std::numeric_limits<float>::infinity();
            for (unsigned s = start_states[seq]; s <= end_states[seq]; s++) {
                const float val  = state_buffer[t * n_states + s];
                float       diff = val - sum;
                if (!isnan(diff)) {
                    sum = -log1p(exp(-abs(diff))) + fminf(sum, val);
                }
            }
            for (unsigned s = start_states[seq]; s <= end_states[seq]; s++) {
                const float val = state_buffer[t * n_states + s] - sum;
                if (val <= pruning) {
                    out << t << ' ' << (s - start_states[seq]) << ' ' << val << '\n';
                }
            }
        }
    }
}

void write_output_to_file(float* d_out, unsigned* d_seq_lens, float pruning, unsigned n_frames,
                          unsigned n_seqs, unsigned n_emissions, unsigned batch_idx) {
    std::vector<float>    buffer(n_frames * n_seqs * n_emissions);
    std::vector<unsigned> seq_lens(n_seqs);

    HANDLE_ERROR(cudaMemcpy(buffer.data(), d_out, buffer.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(seq_lens.data(), d_seq_lens, seq_lens.size() * sizeof(unsigned),
                            cudaMemcpyDeviceToHost));

    for (unsigned seq = 0u; seq < n_seqs; seq++) {
        std::stringstream filename;
        filename << "target.dump." << batch_idx << '.' << seq;
        std::ofstream out(filename.str().c_str(), std::ios::out | std::ios::trunc);
        for (unsigned t = 0u; t < n_frames; t++) {
            if (t > 0u && t >= seq_lens[seq]) {
                break;
            }
            for (unsigned e = 0u; e < n_emissions; e++) {
                const float val = buffer[t * n_seqs * n_emissions + seq * n_emissions + e];
                if (val <= pruning) {
                    out << t << ' ' << e << ' ' << val << '\n';
                }
            }
        }
    }
}

std::vector<torch::Tensor> fbw_cuda(torch::Tensor& am_scores, torch::Tensor& edges,
                                    torch::Tensor& weights, torch::Tensor& start_end_states,
                                    torch::Tensor& seq_lens, unsigned n_states,
                                    DebugOptions debug_options) {
    auto          options    = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor out        = torch::zeros_like(am_scores, options);
    torch::Tensor sum_output = torch::zeros({am_scores.size(0), am_scores.size(1)}, options);

    assert_cmp(Ndarray_DIMS(am_scores)[0], ==, Ndarray_DIMS(out)[0]);
    assert_cmp(Ndarray_DIMS(am_scores)[1], ==, Ndarray_DIMS(out)[1]);
    assert_cmp(Ndarray_DIMS(am_scores)[2], ==, Ndarray_DIMS(out)[2]);
    assert_cmp(Ndarray_DIMS(am_scores)[1], ==, Ndarray_DIMS(start_end_states)[1]);
    assert_cmp(Ndarray_DIMS(start_end_states)[0], ==, 2);

    assert_cmp(Ndarray_DIMS(sum_output)[0], ==, Ndarray_DIMS(am_scores)[0]);
    assert_cmp(Ndarray_DIMS(sum_output)[1], ==, Ndarray_DIMS(am_scores)[1]);

    bool            dump_alignment = debug_options.dump_alignment;
    bool            dump_output    = debug_options.dump_output;
    unsigned        dump_every     = debug_options.dump_every;
    static unsigned batch_idx      = 0u;
    float           pruning        = debug_options.pruning;

    unsigned* d_from          = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges) +
                                                   0 * Ndarray_STRIDE(edges, 0));
    unsigned* d_to            = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges) +
                                                 1 * Ndarray_STRIDE(edges, 0));
    unsigned* d_emission_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges) +
                                                            2 * Ndarray_STRIDE(edges, 0));
    unsigned* d_sequence_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges) +
                                                            3 * Ndarray_STRIDE(edges, 0));
    float*    d_weights       = Ndarray_DEV_DATA(weights);
    float*    d_am_scores     = Ndarray_DEV_DATA(am_scores);

    unsigned* d_start_states = reinterpret_cast<unsigned*>(
            Ndarray_DEV_DATA_int32(start_end_states) + 0 * Ndarray_STRIDE(start_end_states, 0));
    unsigned* d_end_states = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(start_end_states) +
                                                         1 * Ndarray_STRIDE(start_end_states, 0));
    unsigned* d_seq_lens   = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(seq_lens));
    float*    d_out        = Ndarray_DEV_DATA(out);
    float*    d_sum_output = Ndarray_DEV_DATA(sum_output);

    unsigned n_frames    = Ndarray_DIMS(am_scores)[0];
    unsigned n_seqs      = Ndarray_DIMS(am_scores)[1];
    unsigned n_emissions = Ndarray_DIMS(am_scores)[2];
    unsigned n_edges     = Ndarray_DIMS(edges)[1];
    unsigned n_threads   = 1024u;
    unsigned n_blocks    = (n_edges + n_threads - 1) / n_threads;

    unsigned frame_stride    = Ndarray_STRIDE(am_scores, 0);
    unsigned sequence_stride = Ndarray_STRIDE(am_scores, 1);

    assert_cmp(n_frames, >, 0);
    assert_cmp(n_states, >, 0);
    cudaDeviceSynchronize();

    // initialize buffers
    float* d_state_buffer_prev = reinterpret_cast<float*>(device_malloc(n_states * sizeof(float)));
    float* d_state_buffer_next = reinterpret_cast<float*>(device_malloc(n_states * sizeof(float)));
    float* d_edge_buffer =
            reinterpret_cast<float*>(device_malloc(n_edges * n_frames * sizeof(float)));
    if (!d_edge_buffer) {
        HANDLE_LAST_ERROR();
        abort();
    }  // error should have been set in device_malloc
    unsigned n_fill_blocks = (n_edges * n_frames + n_threads - 1u) / n_threads;
    start_dev_kernel2(fill_array, n_fill_blocks, n_threads, 0,
                      (d_edge_buffer, 0.0, n_edges * n_frames));
    HANDLE_LAST_ERROR();

    // initialize the state buffer
    n_fill_blocks = (n_states + n_threads - 1u) / n_threads;
    start_dev_kernel2(fill_array, n_fill_blocks, n_threads, 0,
                      (d_state_buffer_prev, std::numeric_limits<float>::infinity(), n_states));
    HANDLE_LAST_ERROR();
    start_dev_kernel2(set_start_states, 1, n_seqs, 0, (d_state_buffer_prev, d_start_states));
    HANDLE_LAST_ERROR();

    // initialize full state buffer (only used to dump the alignment)
    float* d_state_buffer_all = NULL;
    if (dump_alignment && batch_idx % dump_every == 0) {
        d_state_buffer_all =
                reinterpret_cast<float*>(device_malloc(n_states * (n_frames + 1u) * sizeof(float)));
        if (!d_state_buffer_all) {
            HANDLE_LAST_ERROR();
            abort();
        }  // error should have been set in device_malloc
        Ndarray_memcpy(d_state_buffer_all, d_state_buffer_prev, n_states * sizeof(float));
        HANDLE_LAST_ERROR();
    }

    // fwd pass
    for (unsigned t = 0u; t < n_frames; t++) {
        start_dev_kernel2(fill_array, n_fill_blocks, n_threads, 0,
                          (d_state_buffer_next, std::numeric_limits<float>::infinity(), n_states));
        HANDLE_LAST_ERROR();
        start_dev_kernel2(next_frame, n_blocks, n_threads, 0,
                          (true, n_edges, sequence_stride, d_sequence_idxs, d_from, d_to, d_weights,
                           d_emission_idxs, d_state_buffer_prev, d_state_buffer_next,
                           d_am_scores + t * frame_stride, d_edge_buffer + t * n_edges));
        HANDLE_LAST_ERROR();
        if (dump_alignment && batch_idx % dump_every == 0) {
            Ndarray_memcpy(d_state_buffer_all + (t + 1u) * n_states, d_state_buffer_next,
                           n_states * sizeof(float));
            HANDLE_LAST_ERROR();
        }
        std::swap(d_state_buffer_prev, d_state_buffer_next);
    }

    // bwd pass
    start_dev_kernel2(fill_array, n_fill_blocks, n_threads, 0,
                      (d_state_buffer_prev, std::numeric_limits<float>::infinity(), n_states));
    HANDLE_LAST_ERROR();
    for (unsigned t = n_frames; t > 0; t--) {
        start_dev_kernel2(init_bwd_state_buffer, 1, n_seqs, 0,
                          (d_state_buffer_prev, d_end_states, t - 1, n_frames - 1, d_seq_lens));
        HANDLE_LAST_ERROR();
        start_dev_kernel2(fill_array, n_fill_blocks, n_threads, 0,
                          (d_state_buffer_next, std::numeric_limits<float>::infinity(), n_states));
        HANDLE_LAST_ERROR();
        start_dev_kernel2(
                next_frame, n_blocks, n_threads, 0,
                (false, n_edges, sequence_stride, d_sequence_idxs, d_to, d_from, d_weights,
                 d_emission_idxs, d_state_buffer_prev, d_state_buffer_next,
                 d_am_scores + (t - 1) * frame_stride, d_edge_buffer + (t - 1) * n_edges));
        HANDLE_LAST_ERROR();
        std::swap(d_state_buffer_prev, d_state_buffer_next);
    }

    // normalize at each time frame
    start_dev_kernel2(normalize, n_frames, 1, n_seqs * sizeof(float),
                      (d_edge_buffer, d_sequence_idxs, n_edges, n_seqs, d_sum_output));
    HANDLE_LAST_ERROR();

    // dump alignment
    if (dump_alignment && batch_idx % dump_every == 0) {
        write_alignment_to_file(d_state_buffer_all, d_seq_lens, d_start_states, d_end_states,
                                pruning, n_frames, n_seqs, n_states, batch_idx);
    }

    n_fill_blocks = (n_frames * n_seqs * n_emissions + n_threads - 1u) / n_threads;
    start_dev_kernel2(
            fill_array, n_fill_blocks, n_threads, 0,
            (d_out, std::numeric_limits<float>::infinity(), n_frames * n_seqs * n_emissions));
    HANDLE_LAST_ERROR();

    frame_stride    = Ndarray_STRIDE(out, 0);
    sequence_stride = Ndarray_STRIDE(out, 1);
    n_blocks        = (n_frames * n_edges + n_threads - 1u) / n_threads;
    start_dev_kernel2(compute_result, n_blocks, n_threads, 0,
                      (d_edge_buffer, d_out, d_emission_idxs, d_sequence_idxs, frame_stride,
                       sequence_stride, n_frames, n_seqs, n_edges));
    HANDLE_LAST_ERROR();

    if (dump_output && batch_idx % dump_every == 0) {
        write_output_to_file(d_out, d_seq_lens, pruning, n_frames, n_seqs, n_emissions, batch_idx);
    }

    device_free(d_edge_buffer);
    if (d_state_buffer_all != NULL) {
        device_free(d_state_buffer_all);
    }
    batch_idx++;

    return {out, sum_output};
}
