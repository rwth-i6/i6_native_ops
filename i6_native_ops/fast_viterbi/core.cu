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

#include <../common/definitions.h>

struct __attribute__((__packed__)) IdxAndVal {
    int idx;
    float val;
};

DEV_FUNC
void select_max(IdxAndVal* a, IdxAndVal b) {
    // fast path
    if(b.val < a->val)
        return;
    // Maybe we could use double compare-and-swap (https://stackoverflow.com/questions/55941382/).
    // But not sure how.
    // So instead, we use double-wide compare-and-swap.
    union U {
        IdxAndVal s;
        unsigned long long int v64;
    };
    while(true) {
        U prev;
        prev.s = *a;
        if(b.val < prev.s.val)
            return;
        if(b.val == prev.s.val && b.idx >= prev.s.idx)
            return;
        U updated;
        updated.s = b;

        U old;
        old.v64 = elem_atomic_cas((unsigned long long int*) a, prev.v64, updated.v64);
        if(old.v64 == prev.v64)
            return;
        // Not the same, so repeat.
    }
}

DEF_KERNEL
void init_buffer
(
    int n_time,
    int n_states, // for the whole batch
    IdxAndVal* buffer // (time+1,n_states), states for the whole batch
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while(idx < (n_time + 1) * n_states) {
        buffer[idx].val = -INF_F;
        buffer[idx].idx = -1;
        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL
void init_first_frame
(
    int n_batch,
    int n_states, // for the whole batch
    IdxAndVal* frame, // (n_states,), states for the whole batch
    const int32_t* d_start_states // (n_batch,)
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while(idx < n_batch) {
        int state_idx = d_start_states[idx];
        frame[state_idx].val = 0;
        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL
void next_frame
(
    int n_time,
    int n_states,
    int n_edges,
    int n_classes,
    int t,
    const float* d_am_scores,
    const int32_t* d_am_seq_len,
    const IdxAndVal* prev_frame,
    IdxAndVal* frame,
    const int32_t* d_edge_from,
    const int32_t* d_edge_to,
    const int32_t* d_edge_emission_idx,
    const int32_t* d_edge_seq_idx,
    const float* d_edge_weights,
    const int32_t* d_end_states // (n_batch,)
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while(idx < n_edges) {
        int from_idx = d_edge_from[idx];
        //assert_cmp(0, <=, from_idx); assert_cmp(from_idx, <, n_states);

        int seq_idx = d_edge_seq_idx[idx];
        if(t < d_am_seq_len[seq_idx]) {
            float prev_val = prev_frame[from_idx].val;
            int emission_idx = d_edge_emission_idx[idx];
            //assert_cmp(0, <=, emission_idx); assert_cmp(emission_idx, <, n_classes);
            int to_idx = d_edge_to[idx];
            //assert_cmp(0, <=, to_idx); assert_cmp(to_idx, <, n_states);
            IdxAndVal candidate;
            candidate.val = prev_val + d_edge_weights[idx] + d_am_scores[seq_idx * n_classes + emission_idx];
            candidate.idx = idx;
            select_max(&frame[to_idx], candidate);
        }

        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL
void select_scores
(
    int n_batch,
    int n_states,
    int buffer_stride,
    const IdxAndVal* buffer,
    const int32_t* d_am_seq_len, // (n_batch,)
    const int32_t* d_end_states, // (n_batch,)
    float* d_score // (n_batch,)
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while(idx < n_batch) {
        const IdxAndVal* last_frame = buffer + d_am_seq_len[idx] * buffer_stride;
        int end_state_idx = d_end_states[idx];
        d_score[idx] = last_frame[end_state_idx].val;

        idx += gridDim.x * blockDim.x;
    }
}

DEF_KERNEL
void select_best_path
(
    int n_batch,
    int n_states,
    int n_edges,
    int t,
    int32_t* cur_state, // (n_batch,)
    const IdxAndVal* frame,
    const int32_t* d_am_seq_len,
    const int32_t* d_edge_from,
    const int32_t* d_edge_to,
    const int32_t* d_edge_emission_idx,
    int32_t* output
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while(idx < n_batch) {
        if(t < d_am_seq_len[idx]) {
        int state_idx = cur_state[idx];
        //assert_cmp(0, <=, state_idx); assert_cmp(state_idx, <, n_states);
        int edge_idx = frame[state_idx].idx;
        if(edge_idx >= 0) {
            //assert_cmp(0, <=, edge_idx); assert_cmp(edge_idx, <, n_edges);
            //assert_cmp(state_idx, ==, d_edge_to[edge_idx]);
            cur_state[idx] = d_edge_from[edge_idx];
            output[idx] = d_edge_emission_idx[edge_idx];
        }
        else  // no path found
            output[idx] = 0;
        }
        else {
        output[idx] = 0;
        }
        idx += gridDim.x * blockDim.x;
    }
}

std::vector<torch::Tensor> fast_viterbi_cuda(torch::Tensor& am_scores, torch::Tensor& edges,
                                             torch::Tensor& weights, torch::Tensor& start_end_states,
                                             torch::Tensor& seq_lens, unsigned n_states) {
    using namespace std;

    assert_cmp(Ndarray_NDIM(am_scores), ==, 3);
    assert_cmp(Ndarray_NDIM(seq_lens), ==, 1);
    assert_cmp(Ndarray_NDIM(edges), ==, 2);
    assert_cmp(Ndarray_NDIM(weights), ==, 1);
    assert_cmp(Ndarray_NDIM(start_end_states), ==, 2);
    int n_time = Ndarray_DIMS(am_scores)[0];
    int n_batch = Ndarray_DIMS(am_scores)[1];
    int n_classes = Ndarray_DIMS(am_scores)[2];
    assert_cmp(Ndarray_DIMS(am_scores)[0], ==, n_time);
    assert_cmp(Ndarray_DIMS(am_scores)[1], ==, n_batch);
    assert_cmp(Ndarray_DIMS(am_scores)[2], ==, n_classes);
    assert_cmp(Ndarray_DIMS(seq_lens)[0], ==, n_batch);
    int n_edges = Ndarray_DIMS(edges)[1];
    assert_cmp(Ndarray_DIMS(edges)[0], ==, 4);
    assert_cmp(Ndarray_DIMS(edges)[1], ==, n_edges);
    assert_cmp(Ndarray_DIMS(weights)[0], ==, n_edges);
    assert_cmp(Ndarray_DIMS(start_end_states)[0], ==, 2);
    assert_cmp(Ndarray_DIMS(start_end_states)[1], ==, n_batch);

    auto options         = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor output = torch::zeros({n_time, n_batch}, options.dtype(torch::kInt32));
    torch::Tensor score  = torch::zeros({n_batch}, options);

    int32_t* d_edge_from = Ndarray_DEV_DATA_int32(edges) + 0 * Ndarray_STRIDE(edges, 0);
    int32_t* d_edge_to = Ndarray_DEV_DATA_int32(edges) + 1 * Ndarray_STRIDE(edges, 0);
    int32_t* d_edge_emission_idx = Ndarray_DEV_DATA_int32(edges) + 2 * Ndarray_STRIDE(edges, 0);
    int32_t* d_edge_seq_idx = Ndarray_DEV_DATA_int32(edges) + 3 * Ndarray_STRIDE(edges, 0);
    float* d_edge_weights = Ndarray_DEV_DATA(weights);
    float* d_am_scores = Ndarray_DEV_DATA(am_scores);
    int am_scores_stride = Ndarray_STRIDE(am_scores, 0);
    int32_t* d_am_seq_len = Ndarray_DEV_DATA_int32(seq_lens);
    int32_t* d_start_states = Ndarray_DEV_DATA_int32(start_end_states) + 0 * Ndarray_STRIDE(start_end_states, 0);
    int32_t* d_end_states = Ndarray_DEV_DATA_int32(start_end_states) + 1 * Ndarray_STRIDE(start_end_states, 0);
    int32_t* d_output = Ndarray_DEV_DATA_int32(output);
    int output_stride = Ndarray_STRIDE(output, 0);
    float* d_score = Ndarray_DEV_DATA(score);

    IdxAndVal* d_buffer = (IdxAndVal*) device_malloc((n_time + 1) * n_states * sizeof(IdxAndVal));
    int buffer_stride = n_states;
    start_dev_kernel(init_buffer, (n_time, n_states, d_buffer));
    start_dev_kernel(init_first_frame, (n_batch, n_states, d_buffer, d_start_states));
    HANDLE_LAST_ERROR();

    for(int t = 0; t < n_time; ++t) {
        start_dev_kernel(next_frame, (
            n_time,
            n_states,
            n_edges,
            n_classes,
            t,
            d_am_scores + t * am_scores_stride,
            d_am_seq_len,
            d_buffer + t * buffer_stride,
            d_buffer + (t + 1) * buffer_stride,
            d_edge_from,
            d_edge_to,
            d_edge_emission_idx,
            d_edge_seq_idx,
            d_edge_weights,
            d_end_states
        ));
    }
    HANDLE_LAST_ERROR();

    start_dev_kernel(select_scores, (
        n_batch,
        n_states,
        buffer_stride,
        d_buffer,
        d_am_seq_len,
        d_end_states,
        d_score // out
    ));

    int32_t* d_cur_state = (int32_t*) device_malloc(n_batch * sizeof(int32_t));
    Ndarray_memcpy(d_cur_state, d_end_states, n_batch * sizeof(int32_t));

    for(int t = n_time - 1; t >= 0; --t) {
        start_dev_kernel(select_best_path, (
            n_batch,
            n_states,
            n_edges,
            t,
            d_cur_state,
            d_buffer + (t + 1) * buffer_stride,
            d_am_seq_len,
            d_edge_from,
            d_edge_to,
            d_edge_emission_idx,
            d_output + t * output_stride // out
        ));
    }
    HANDLE_LAST_ERROR();

    device_free(d_cur_state);
    device_free(d_buffer);

    return {output, score};
}