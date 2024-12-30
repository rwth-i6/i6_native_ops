#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cooperative_groups.h>
//#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <torch/extension.h>

#include <DebugOptions.h>
#include <../common/returnn_definitions.h>

namespace cg = cooperative_groups;

constexpr unsigned max_edge_threads = 1024;

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
void compute_edge_merging_vectors(unsigned n_seqs,
                                  unsigned n_threads_intra_seq,
                                  unsigned const* edges,
                                  unsigned const* edge_offsets,
                                  unsigned* merge_edge_offsets,
                                  unsigned* merge_edge_idxs,
                                  unsigned* merge_edge_targets) {
    __shared__ unsigned local_edges_full_buffer[max_edge_threads];

    unsigned seq = blockIdx.x * blockDim.y + threadIdx.y;
    if (seq > n_seqs) {
        return;
    }

    unsigned buffer_start = threadIdx.y * n_threads_intra_seq;
    for (unsigned start_edge = edge_offsets[seq]; start_edge < edge_offsets[seq+1]; start_edge += n_threads_intra_seq) {
        unsigned end_edge = min(start_edge + n_threads_intra_seq, edge_offsets[seq+1]);
        thrust::sequence(thrust::device, merge_edge_idxs + start_edge, merge_edge_idxs + end_edge, threadIdx.y * n_threads_intra_seq);

        unsigned  num_edges_to_process = end_edge - start_edge;
        unsigned* local_edges = local_edges_full_buffer + buffer_start;

        thrust::copy(thrust::device, edges + start_edge, edges + end_edge, local_edges);
        thrust::sort_by_key(thrust::device, local_edges, local_edges + num_edges_to_process, merge_edge_idxs + start_edge);
        thrust::reduce_by_key(thrust::device,
                              local_edges,
                              local_edges + num_edges_to_process,
                              thrust::make_constant_iterator(1),
                              merge_edge_targets + start_edge,
                              merge_edge_offsets + start_edge);
        merge_edge_offsets[start_edge] += start_edge;
        thrust::inclusive_scan(thrust::device, merge_edge_offsets + start_edge, merge_edge_offsets + end_edge, merge_edge_offsets + start_edge);
    }
}

DEF_KERNEL
void set_start_states(float* states, unsigned* start_states) {
    unsigned state_idx = start_states[blockIdx.x * blockDim.x + threadIdx.x];
    states[state_idx]  = 0.0;
}

template<bool fwd>
DEF_KERNEL
void baum_welch(unsigned num_frames, unsigned num_seq, unsigned num_emissions, unsigned num_edges,
                unsigned const* state_offsets, unsigned const* edge_offsets, unsigned const* seq_lens,
                unsigned const* from_buffer, unsigned const* to_buffer, float const* weight_buffer, unsigned const* emission_idxs, unsigned const* init_states,
                float* prev_states, float* next_states, float const* am_scores, float* edge_buffer,
                unsigned const* merge_edge_offsets, unsigned const* merge_edge_idxs, unsigned const* merge_edge_targets,
                float* state_buffer_all) {
    __shared__ float edge_values[max_edge_threads];

    unsigned seq = blockIdx.x * blockDim.y + threadIdx.y;
    if (seq > num_seq) {
        return;
    }

    unsigned start_time = fwd ? 0 : seq_lens[seq];
    unsigned end_time   = fwd ? seq_lens[seq] : 0;
    int      time_step  = fwd ? 1 : -1;

    unsigned max_inner_loop_size = 1;
    for (unsigned s = blockIdx.x * blockDim.y; s < (blockIdx.x + 1) * blockDim.y; s++) {
        unsigned loop_size = (edge_offsets[s+1] - edge_offsets[s] + blockDim.x - 1) / blockDim.x;
        max_inner_loop_size = max(max_inner_loop_size, loop_size);
    }

    for (unsigned t = start_time; t != end_time; t += time_step) {
        float const* cur_frame_am_scores = am_scores + (t - (fwd ? 0 : 1)) * num_seq * num_emissions + seq * num_emissions;
        float* cur_frame_edge_buffer = edge_buffer + (t - (fwd ? 0 : 1)) * num_edges;

        if (fwd and threadIdx.x == 0 and t == 0) {
            prev_states[init_states[seq]] = 0.0;
        }
        if (not fwd and threadIdx.x == 0 and t == seq_lens[seq]) {
            prev_states[init_states[seq]] = 0.0;
        }

        for (unsigned state = state_offsets[seq] + threadIdx.x; state < state_offsets[seq+1]; state += blockDim.x) {
            next_states[state] = INF_F;
        }

        __syncthreads();  // synchronize to ensure that prev/next_states has been set by all threads

        for (unsigned l = 0; l < max_inner_loop_size; l++) {
            unsigned edge = edge_offsets[seq] + l * blockDim.x + threadIdx.x;
            bool do_work = edge < edge_offsets[seq+1];

            if (do_work) {
                unsigned from     = from_buffer[edge];
                float    prev_val = prev_states[from];

                float val = INF_F;
                if (not isinf(prev_val)) {
                    unsigned emission_idx = emission_idxs[edge];
                    float    edge_weight  = weight_buffer[edge];

                    val = prev_val + edge_weight + cur_frame_am_scores[emission_idx];
                }

                edge_values[threadIdx.y * blockDim.x + threadIdx.x] = val;
                if (fwd) {
                    cur_frame_edge_buffer[edge] = val;
                }
                else {
                    cur_frame_edge_buffer[edge] += prev_val;
                }
            }

            __syncthreads();  // synchronize to ensure that edge_values has been set by all threads

            if (do_work) {
                unsigned merge_offset_start = threadIdx.x == 0 ? (edge_offsets[seq] + l * blockDim.x) : merge_edge_offsets[edge-1];
                unsigned merge_offset_end   = merge_edge_offsets[edge];
                if (merge_offset_start < merge_offset_end) {
                    float sum = edge_values[merge_edge_idxs[merge_offset_start]];
                    for (unsigned e = merge_offset_start + 1; e < merge_offset_end; e++) {
                        sum = prob_add(sum, edge_values[merge_edge_idxs[e]]);
                    }
                    unsigned target_state = merge_edge_targets[edge];
                    next_states[target_state] = prob_add(next_states[target_state], sum);
                }
            }

            __syncthreads();  // synchronize to ensure that next_states has been set by all threads
        }

        if (state_buffer_all != nullptr) {
            __syncthreads();
            unsigned num_states = state_offsets[num_seq];
            for (unsigned state = state_offsets[seq] + threadIdx.x; state < state_offsets[seq+1]; state += blockDim.x) {
                unsigned idx = t * num_states + state;
                float sum = fwd ? prev_states[state] : (state_buffer_all[idx] + prev_states[state]);
                state_buffer_all[idx] = sum;
            }
        }

        float* tmp = next_states;
        next_states = prev_states;
        prev_states = tmp;
    }

    if (state_buffer_all != nullptr) {
        __syncthreads();
        unsigned num_states = state_offsets[num_seq];
        for (unsigned state = state_offsets[seq] + threadIdx.x; state < state_offsets[seq+1]; state += blockDim.x) {
            unsigned idx = end_time * num_states + state;
            float sum = fwd ? prev_states[state] : (state_buffer_all[idx] + prev_states[state]);
            state_buffer_all[idx] = sum;
        }
    }
}

template<bool fwd>
DEF_KERNEL
void baum_welch_v2(unsigned num_frames, unsigned num_seq, unsigned num_emissions, unsigned num_edges,
                   unsigned const* state_offsets, unsigned const* edge_offsets, unsigned const* seq_lens,
                   unsigned const* from_buffer, unsigned const* to_buffer, float const* weight_buffer, unsigned const* emission_idxs, unsigned const* init_states,
                   float* prev_states, float* next_states, float const* am_scores, float* edge_buffer,
                   float* state_buffer_all) {
    unsigned seq = blockIdx.x * blockDim.y + threadIdx.y;
    if (seq >= num_seq) {
        return;
    }

    unsigned start_time = fwd ? 0 : seq_lens[seq];
    unsigned end_time   = fwd ? seq_lens[seq] : 0;
    int      time_step  = fwd ? 1 : -1;

    unsigned max_inner_loop_size = 1;
    for (unsigned s = blockIdx.x * blockDim.y; s < (blockIdx.x + 1) * blockDim.y and s < num_seq; s++) {
        unsigned loop_size = (edge_offsets[s+1] - edge_offsets[s] + blockDim.x - 1) / blockDim.x;
        max_inner_loop_size = max(max_inner_loop_size, loop_size);
    }

    if (fwd and threadIdx.x == 0) {
        prev_states[init_states[seq]] = 0.0;
    }

    for (unsigned t = start_time; t != end_time; t += time_step) {
        float const* cur_frame_am_scores = am_scores + (t - (fwd ? 0 : 1)) * num_seq * num_emissions + seq * num_emissions;
        float* cur_frame_edge_buffer = edge_buffer + (t - (fwd ? 0 : 1)) * num_edges;

        if (not fwd and threadIdx.x == 0 and t == seq_lens[seq]) {
            prev_states[init_states[seq]] = 0.0;
        }

        for (unsigned state = state_offsets[seq] + threadIdx.x; state < state_offsets[seq+1]; state += blockDim.x) {
            next_states[state] = INF_F;
        }

        __syncthreads();  // synchronize to ensure that prev/next_states has been set by all threads

        for (unsigned l = 0; l < max_inner_loop_size; l++) {
            unsigned edge = edge_offsets[seq] + l * blockDim.x + threadIdx.x;
            bool do_work = edge < edge_offsets[seq+1];

            if (do_work) {
                unsigned from     = from_buffer[edge];
                unsigned to       = to_buffer[edge];
                float    prev_val = prev_states[from];

                float val = INF_F;
                if (not isinf(prev_val)) {
                    unsigned emission_idx = emission_idxs[edge];
                    float    edge_weight  = weight_buffer[edge];

                    val = prev_val + edge_weight + cur_frame_am_scores[emission_idx];
                }

                atomic_prob_add(next_states + to, val);
                if (fwd) {
                    cur_frame_edge_buffer[edge] = val;
                }
                else {
                    cur_frame_edge_buffer[edge] += prev_val;
                }
            }

            __syncthreads();  // synchronize to ensure that next_states has been set by all threads
        }

        if (state_buffer_all != nullptr) {
            unsigned num_states = state_offsets[num_seq];
            for (unsigned state = state_offsets[seq] + threadIdx.x; state < state_offsets[seq+1]; state += blockDim.x) {
                unsigned idx = t * num_states + state;
                float sum = fwd ? prev_states[state] : (state_buffer_all[idx] + prev_states[state]);
                state_buffer_all[idx] = sum;
            }
        }

        float* tmp = next_states;
        next_states = prev_states;
        prev_states = tmp;
    }

    if (state_buffer_all != nullptr) {
        __syncthreads();
        unsigned num_states = state_offsets[num_seq];
        for (unsigned state = state_offsets[seq] + threadIdx.x; state < state_offsets[seq+1]; state += blockDim.x) {
            unsigned idx = end_time * num_states + state;
            float sum = fwd ? prev_states[state] : (state_buffer_all[idx] + prev_states[state]);
            state_buffer_all[idx] = sum;
        }
    }
}

DEF_KERNEL
void normalize(unsigned num_seqs, unsigned num_edges, unsigned const* seq_lens, float* edge_buffer, unsigned const* edge_offsets, float* sum_output) {
    __shared__ float local_sum[max_edge_threads];

    unsigned seq = blockIdx.x * blockDim.y + threadIdx.y;
    if (seq >= num_seqs) {
        return;
    }

    unsigned time = blockIdx.y;

    edge_buffer += time * num_edges;

    float sum = INF_F;

    if (time < seq_lens[seq]) {
        for (unsigned e = edge_offsets[seq] + threadIdx.x; e < edge_offsets[seq+1]; e += blockDim.x) {
            sum = prob_add(sum, edge_buffer[e]);
        }
    }

    unsigned edge_idx = threadIdx.y * blockDim.x + threadIdx.x;
    local_sum[edge_idx] = sum;

    __syncthreads();

    for (unsigned offset = blockDim.x / 2; offset >= 1; offset /= 2) {
        if (edge_idx < offset) {
            local_sum[edge_idx] = prob_add(local_sum[edge_idx], local_sum[edge_idx + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (isinf(local_sum[edge_idx])) {
            // if the frame is empty (happens due to batching of seqs with
            // unequal length), set it to 0
            sum_output[time * num_seqs + seq] = 0.0;
        }
        else {
            sum_output[time * num_seqs + seq] = local_sum[edge_idx];
        }
    }

    __syncthreads();

    sum = sum_output[time * num_seqs + seq];
    for (unsigned e = edge_offsets[seq] + threadIdx.x; e < edge_offsets[seq+1]; e += blockDim.x) {
        edge_buffer[e] -= sum;
    }
}

DEF_KERNEL
void compute_result(float const* edge_buffer, float* out, unsigned const* emission_idxs,
                    unsigned const* seq_lens, unsigned const* edge_offsets,
                    unsigned frame_stride, unsigned seq_stride,
                    unsigned num_edges, unsigned num_seqs) {
    unsigned seq = blockIdx.x * blockDim.y + threadIdx.y;
    if (seq >= num_seqs) {
        return;
    }

    unsigned time = blockIdx.y;

    if (time >= seq_lens[seq]) {
        return;
    }

    edge_buffer += time * num_edges;
    out += time * frame_stride + seq * seq_stride;

    for (unsigned e = edge_offsets[seq] + threadIdx.x; e < edge_offsets[seq+1]; e += blockDim.x) {
        unsigned emission_idx = emission_idxs[e];
        float    score        = edge_buffer[e];

        atomic_prob_add(out + emission_idx, score);
    }
}

void write_merge_edges_to_file(std::string const& filename,
                               thrust::device_vector<unsigned>& merge_edge_offsets,
                               thrust::device_vector<unsigned>& merge_edge_idxs,
                               thrust::device_vector<unsigned>& merge_edge_targets) {
    thrust::host_vector<unsigned> meo = merge_edge_offsets;
    thrust::host_vector<unsigned> mei = merge_edge_idxs;
    thrust::host_vector<unsigned> met = merge_edge_targets;
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::trunc);

    out << "idx:               ";
    for (size_t i = 0ul; i < meo.size(); i++) {
        out << " " << std::setw(4) << i;
    }
    out << std::endl;

    out << "merge_edge_offsets:";
    for (unsigned v: meo) {
        out << " " << std::setw(4) << v;
    }
    out << std::endl;
    out << "merge_edge_idxs:   ";
    for (unsigned v: mei) {
        out << " " << std::setw(4) << v;
    }
    out << std::endl;
    out << "merge_edge_targets:";
    for (unsigned v: met) {
        out << " " << std::setw(4) << v;
    }
    out << std::endl;
}

void write_alignment_to_file(std::string const& filename, float* d_state_buffer,
                             unsigned* d_seq_lens, unsigned* d_start_states,
                             unsigned* d_end_states, float pruning, unsigned n_frames,
                             unsigned n_seqs, unsigned n_states, unsigned batch_idx, bool normalize) {
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
        std::stringstream ss;
        ss << filename << batch_idx << '.' << seq;
        std::ofstream out(ss.str().c_str(), std::ios::out | std::ios::trunc);
        for (unsigned t = 0u; t <= n_frames; t++) {
            if (t > 0u && t > seq_lens[seq]) {
                break;
            }
            float sum = 0.0;
            if (normalize) {
                sum = std::numeric_limits<float>::infinity();
                for (unsigned s = start_states[seq]; s <= end_states[seq]; s++) {
                    const float val  = state_buffer[t * n_states + s];
                    float       diff = val - sum;
                    if (!isnan(diff)) {
                        sum = -log1p(exp(-abs(diff))) + fminf(sum, val);
                    }
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
    size_t                        out_size = n_frames * n_seqs * n_emissions;
    thrust::host_vector<float>    buffer(out_size);
    thrust::host_vector<unsigned> seq_lens(n_seqs);

    thrust::copy(thrust::device_ptr<float>(d_out), thrust::device_ptr<float>(d_out + out_size), buffer.begin());
    thrust::copy(thrust::device_ptr<unsigned>(d_seq_lens), thrust::device_ptr<unsigned>(d_seq_lens + n_seqs), seq_lens.begin());

    for (unsigned seq = 0u; seq < n_seqs; seq++) {
        std::stringstream filename;
        filename << "target.dump_v2." << batch_idx << '.' << seq;
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

std::vector<torch::Tensor> fbw2_cuda(torch::Tensor& num_states, torch::Tensor& num_edges, torch::Tensor& seq_lens,
                                     torch::Tensor& am_scores, torch::Tensor& edges, torch::Tensor& weights, torch::Tensor& start_end_states,
                                     DebugOptionsV2 debug_options) {
    auto          options    = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor out        = torch::zeros_like(am_scores, options);
    torch::Tensor sum_output = torch::zeros({am_scores.size(0), am_scores.size(1)}, options);

    // make sure we have the right shapes
    assert_cmp(Ndarray_DIMS(out)[0], ==, Ndarray_DIMS(am_scores)[0]);
    assert_cmp(Ndarray_DIMS(out)[1], ==, Ndarray_DIMS(am_scores)[1]);
    assert_cmp(Ndarray_DIMS(out)[2], ==, Ndarray_DIMS(am_scores)[2]);
    assert_cmp(Ndarray_DIMS(sum_output)[0], ==, Ndarray_DIMS(am_scores)[0]);
    assert_cmp(Ndarray_DIMS(sum_output)[1], ==, Ndarray_DIMS(am_scores)[1]);

    assert_cmp(Ndarray_DIMS(start_end_states)[0], ==, 2);
    assert_cmp(Ndarray_DIMS(start_end_states)[1], ==, Ndarray_DIMS(am_scores)[1]);
    assert_cmp(Ndarray_DIMS(num_edges)[0], ==, Ndarray_DIMS(am_scores)[1]);
    assert_cmp(Ndarray_DIMS(num_states)[0], ==, Ndarray_DIMS(am_scores)[1]);

    // debugging options
    static unsigned batch_idx = 0u;
    bool dump_edges = debug_options.dump_edges and batch_idx % debug_options.dump_every == 0;
    bool dump_alignment = debug_options.dump_alignment and batch_idx % debug_options.dump_every == 0;
    bool dump_output = debug_options.dump_output and batch_idx % debug_options.dump_every == 0;

    // raw data pointers
    unsigned* d_from          = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges) + 0 * Ndarray_STRIDE(edges, 0));
    unsigned* d_to            = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges) + 1 * Ndarray_STRIDE(edges, 0));
    unsigned* d_emission_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges) + 2 * Ndarray_STRIDE(edges, 0));
    unsigned* h_num_states    = Ndarray_DEV_DATA_uint32(num_states);
    unsigned* h_num_edges     = Ndarray_DEV_DATA_uint32(num_edges);
    float*    d_weights       = Ndarray_DEV_DATA(weights);
    float*    d_am_scores     = Ndarray_DEV_DATA(am_scores);

    unsigned* d_start_states = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(start_end_states) + 0 * Ndarray_STRIDE(start_end_states, 0));
    unsigned* d_end_states   = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(start_end_states) + 1 * Ndarray_STRIDE(start_end_states, 0));
    unsigned* d_seq_lens     = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(seq_lens));
    float*    d_out          = Ndarray_DEV_DATA(out);
    float*    d_sum_output   = Ndarray_DEV_DATA(sum_output);

    unsigned n_frames    = Ndarray_DIMS(am_scores)[0];
    unsigned n_seqs      = Ndarray_DIMS(am_scores)[1];
    unsigned n_emissions = Ndarray_DIMS(am_scores)[2];
    unsigned n_edges     = Ndarray_DIMS(edges)[1];

    assert_cmp(n_frames, >, 0);

    unsigned frame_stride    = Ndarray_STRIDE(am_scores, 0);
    unsigned sequence_stride = Ndarray_STRIDE(am_scores, 1);

    // calculate state offsets per seq
    thrust::host_vector<unsigned> state_offsets_host(n_seqs + 1, 0);
    thrust::inclusive_scan(h_num_states, h_num_states + n_seqs, state_offsets_host.begin() + 1);
    thrust::device_vector<unsigned> state_offsets = state_offsets_host;

    unsigned n_states = state_offsets_host.back();
    assert_cmp(n_states, >, 0);

    // calculate edge offsets per seq
    thrust::host_vector<unsigned> edge_offsets_host(n_seqs + 1, 0);
    thrust::inclusive_scan(h_num_edges, h_num_edges + n_seqs, edge_offsets_host.begin() + 1);
    thrust::device_vector<unsigned> edge_offsets = edge_offsets_host;

    // calculate size of fwb kernel blocks / grid
    unsigned max_edges_per_seq = *std::max_element(h_num_edges, h_num_edges + n_seqs);

    // we divide each block into smaller pieces to accommodate more sequences within one block
    unsigned n_threads_intra_seq = max_edge_threads;
    while (n_threads_intra_seq / 2 > max_edges_per_seq) {
        n_threads_intra_seq /= 2;
    }
    unsigned n_threads_inter_seq = std::min(max_edge_threads / n_threads_intra_seq, n_seqs);
    dim3     block_size_fbw(n_threads_intra_seq, n_threads_inter_seq);

    unsigned n_blocks_fbw = (n_seqs + n_threads_inter_seq - 1) / n_threads_inter_seq;

    // compute edge merging vectors
    thrust::device_vector<unsigned> merge_edge_offsets;
    thrust::device_vector<unsigned> merge_edge_idxs;
    thrust::device_vector<unsigned> merge_edge_targets;
    if (debug_options.explicit_merge) {
        merge_edge_offsets.resize(n_edges, 0);
        merge_edge_idxs.resize(n_edges, 0);
        merge_edge_targets.resize(n_edges, 0);
        start_dev_kernel2(compute_edge_merging_vectors, n_blocks_fbw, dim3(1, n_threads_inter_seq), 0,
                          (n_seqs, n_threads_intra_seq, d_to, edge_offsets.data().get(), merge_edge_offsets.data().get(), merge_edge_idxs.data().get(), merge_edge_targets.data().get()));
        HANDLE_LAST_ERROR();

        if (dump_edges) {
            write_merge_edges_to_file("fwd.edges.dump_v2." + std::to_string(batch_idx), merge_edge_offsets, merge_edge_idxs, merge_edge_targets);
        }
    }

    // initialize buffers
    thrust::device_vector<float> state_buffer_prev(n_states, std::numeric_limits<float>::infinity());
    thrust::device_vector<float> state_buffer_next(n_states, std::numeric_limits<float>::infinity());
    thrust::device_vector<float> state_buffer_all;
    thrust::device_vector<float> edge_buffer(n_edges * n_frames, 0.0f);

    // initialize full state buffer (only used to dump the alignment)
    if (dump_alignment) {
        state_buffer_all.resize(n_states * (n_frames + 1u));
        thrust::copy(thrust::device, state_buffer_prev.begin(), state_buffer_prev.end(), state_buffer_all.begin());
    }

    // fwd pass
    if (debug_options.explicit_merge) {
        start_dev_kernel2(baum_welch<true>, n_blocks_fbw, block_size_fbw, 0,
                          (n_frames, n_seqs, n_emissions, n_edges,
                           state_offsets.data().get(), edge_offsets.data().get(), d_seq_lens,
                           d_from, d_to, d_weights, d_emission_idxs, d_start_states,
                           state_buffer_prev.data().get(), state_buffer_next.data().get(), d_am_scores, edge_buffer.data().get(),
                           merge_edge_offsets.data().get(), merge_edge_idxs.data().get(), merge_edge_targets.data().get(),
                           dump_alignment ? state_buffer_all.data().get() : nullptr));
    }
    else {
        start_dev_kernel2(baum_welch_v2<true>, n_blocks_fbw, block_size_fbw, 0,
                          (n_frames, n_seqs, n_emissions, n_edges,
                           state_offsets.data().get(), edge_offsets.data().get(), d_seq_lens,
                           d_from, d_to, d_weights, d_emission_idxs, d_start_states,
                           state_buffer_prev.data().get(), state_buffer_next.data().get(), d_am_scores, edge_buffer.data().get(),
                           dump_alignment ? state_buffer_all.data().get() : nullptr));
    }
    HANDLE_LAST_ERROR();

    // dump alignment
    if (dump_alignment) {
        write_alignment_to_file("fwd.alignment.dump_v2.", state_buffer_all.data().get(), d_seq_lens, d_start_states, d_end_states,
                                debug_options.pruning, n_frames, n_seqs, n_states, batch_idx, false);
    }

    // bwd pass

    // sort again based on from edge
    if (debug_options.explicit_merge) {
        thrust::fill(merge_edge_offsets.begin(), merge_edge_offsets.end(), 0);
        start_dev_kernel2(compute_edge_merging_vectors, n_blocks_fbw, dim3(1, n_threads_inter_seq), 0,
                          (n_seqs, n_threads_intra_seq, d_from, edge_offsets.data().get(), merge_edge_offsets.data().get(), merge_edge_idxs.data().get(), merge_edge_targets.data().get()));
        HANDLE_LAST_ERROR();

        if (dump_edges) {
            write_merge_edges_to_file("bwd.edges.dump_v2." + std::to_string(batch_idx), merge_edge_offsets, merge_edge_idxs, merge_edge_targets);
        }
    }

    thrust::fill(state_buffer_prev.begin(), state_buffer_prev.end(), std::numeric_limits<float>::infinity());
    if (debug_options.explicit_merge) {
        start_dev_kernel2(baum_welch<false>, n_blocks_fbw, block_size_fbw, 0,
                          (n_frames, n_seqs, n_emissions, n_edges,
                           state_offsets.data().get(), edge_offsets.data().get(), d_seq_lens,
                           d_to, d_from, d_weights, d_emission_idxs, d_end_states,
                           state_buffer_prev.data().get(), state_buffer_next.data().get(), d_am_scores, edge_buffer.data().get(),
                           merge_edge_offsets.data().get(), merge_edge_idxs.data().get(), merge_edge_targets.data().get(),
                           dump_alignment ? state_buffer_all.data().get() : nullptr));
    }
    else {
        start_dev_kernel2(baum_welch_v2<false>, n_blocks_fbw, block_size_fbw, 0,
                          (n_frames, n_seqs, n_emissions, n_edges,
                           state_offsets.data().get(), edge_offsets.data().get(), d_seq_lens,
                           d_to, d_from, d_weights, d_emission_idxs, d_end_states,
                           state_buffer_prev.data().get(), state_buffer_next.data().get(), d_am_scores, edge_buffer.data().get(),
                           dump_alignment ? state_buffer_all.data().get() : nullptr));
    }
    HANDLE_LAST_ERROR();

    // dump alignment
    if (dump_alignment) {
        write_alignment_to_file("alignment.dump_v2.", state_buffer_all.data().get(), d_seq_lens, d_start_states, d_end_states,
                                debug_options.pruning, n_frames, n_seqs, n_states, batch_idx, false);
        write_alignment_to_file("norm.alignment.dump_v2.", state_buffer_all.data().get(), d_seq_lens, d_start_states, d_end_states,
                                debug_options.pruning, n_frames, n_seqs, n_states, batch_idx, true);
    }

    // normalize at each time frame
    start_dev_kernel2(normalize, dim3(n_blocks_fbw, n_frames), block_size_fbw, 0,
                      (n_seqs, n_edges, d_seq_lens, edge_buffer.data().get(), edge_offsets.data().get(), d_sum_output));
    HANDLE_LAST_ERROR();

    thrust::fill(thrust::device, d_out, d_out + n_frames * n_seqs * n_emissions, std::numeric_limits<float>::infinity());

    frame_stride    = Ndarray_STRIDE(out, 0);
    sequence_stride = Ndarray_STRIDE(out, 1);
    start_dev_kernel2(compute_result, dim3(n_blocks_fbw, n_frames), block_size_fbw, 0,
                      (edge_buffer.data().get(), d_out, d_emission_idxs, d_seq_lens, edge_offsets.data().get(),
                       frame_stride, sequence_stride, n_edges, n_seqs));
    HANDLE_LAST_ERROR();
    
    if (dump_output) {
        write_output_to_file(d_out, d_seq_lens, debug_options.pruning, n_frames, n_seqs, n_emissions, batch_idx);
    }

    batch_idx++;

    return {out, sum_output};
}
