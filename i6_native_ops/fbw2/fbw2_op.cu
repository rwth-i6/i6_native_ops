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

constexpr unsigned max_threads_per_block = 1024;

/**
 * Device function to add two probabilities in a numerically stable manner in neg-log space.
 */
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

/**
 * Device function to add two probabilities in a numerically stable manner in neg-log space. The sum is stored
 * atomically at the address pointed to by a.
 */
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

/**
 * Compute edge merging vectors. Merging is performed by the thread that process a given edge. It will loop over a fixed
 * set of edges and compute the sum of their probabilities and write them into the target state so there is no conflict
 * between two or more threads wanting to write into the same location.
 * Kernel launch config: (y=#FRAMES, x=#INTER_SEQ) x (y=#INTRA_SEQ, x=1)
 *
 * @param num_seqs: Number of sequences = #SEQ
 * @param n_threads_intra_seq: Number of threads for a sequence within a block
 * @param edges: Target states for each edge, Shape: #EDGES (edges are flattened over all sequences)
 * @param edge_offsets: Offsets of edges in each sequence, exclusive prefix sum over number of edges per seq,
                        Shape: #SEQ + 1
 * @param merge_edge_offsets: last edge (exclusive) to merge. So each thread will merge edges in the interval
                              [merge_edge_offsets[my_edge-1], merge_edge_offsets[my_edge]). The intervals here do not
                              refer to the edges directly, but to the indices stored in merge_edge_idxs. The first
                              thread starts at edge 0. Shape: #EDGES
 * @param merge_edge_idxs: The actual indices of the edges to be merged. Shape #EDGES
 * @param merge_edge_targets: The target state for the sum. Shape #EDGES
 */
DEF_KERNEL
void compute_edge_merging_vectors(unsigned num_seqs,
                                  unsigned num_threads_intra_seq,
                                  unsigned const* edges,
                                  unsigned const* edge_offsets,
                                  unsigned* merge_edge_offsets,
                                  unsigned* merge_edge_idxs,
                                  unsigned* merge_edge_targets) {
    __shared__ unsigned local_edges_full_buffer[max_threads_per_block];

    unsigned seq = blockIdx.x * blockDim.y + threadIdx.y;
    if (seq > num_seqs) {
        return;
    }

    unsigned buffer_start = threadIdx.y * num_threads_intra_seq;
    for (unsigned start_edge = edge_offsets[seq]; start_edge < edge_offsets[seq+1]; start_edge += num_threads_intra_seq) {
        unsigned end_edge = min(start_edge + num_threads_intra_seq, edge_offsets[seq+1]);
        thrust::sequence(thrust::device, merge_edge_idxs + start_edge, merge_edge_idxs + end_edge, threadIdx.y * num_threads_intra_seq);

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

/**
 * The main kernel computing per edge gammas. The template parameter controls which direction is being run. This variant
 * uses explicit merging.
 * Kernel launch config: (x=#INTER_SEQ) x (y=#INTRA_SEQ, x=#THREADS)
 *
 * @param num_frames: The maximum number of time frames for any sequence.
 * @param num_seqs: Number of sequences = #SEQ
 * @param num_emissions: Number of emissions / labels
 * @param num_edges: Number of edges, should be the same as edge_offsets[-1]
 * @param state_offsets: Offsets for the states belonging to one sequence. Exclusive prefix sum over number of states
                         per seq. Shape #SEQ + 1
 * @param edge_offsets: Offsets of edges in each sequence, exclusive prefix sum over number of edges per seq,
                        Shape: #SEQ + 1
 * @param seq_lens: The sequence lengths. Shape: #SEQ
 * @param from_buffer: The source states of all edges. Shape #EDGES
 * @param to_buffer: The target states of all edges. Shape #EDGES
 * @param weight_buffer: The weight of each edge. Shape #EDGES
 * @param emission_idxs: emission/label index of each edge, Shape: #EDGES
 * @param init_states: The initial state for each sequence. Shape: #SEQ
 * @param final_states: The final state for each sequence. Shape: #SEQ
 * @param prev_states: An intermediate buffer to be used by this kernel to store the state probabilities. Shape: #STATES
 * @param next_states: An intermediate buffer to be used by this kernel to store the state probabilities. Shape: #STATES
 * @param am_scores: The emission probabilities per time/seq/emission. Shape: #TIME x #SEQ x #LABEL
 * @param edge_buffer: Buffer of edge gammas. Shape: #FRAMES x #EDGES (edges are flattened over all sequences)
 * @param norm_factors: Optional buffer of normalization values. If not set the user is expected to call the normalize
                        kernel after the backward pass separately. If set the forward pass will write the state value of
                        the final state in this vector and the backward pass will use it to directly normalize the
                        output. Shape: #SEQ
 * @param merge_edge_offsets: last edge (exclusive) to merge. So each thread will merge edges in the interval
                              [merge_edge_offsets[my_edge-1], merge_edge_offsets[my_edge]). The intervals here do not
                              refer to the edges directly, but to the indices stored in merge_edge_idxs. The first
                              thread starts at edge 0. Shape: #EDGES
 * @param merge_edge_idxs: The actual indices of the edges to be merged. Shape #EDGES
 * @param merge_edge_targets: The target state for the sum. Shape #EDGES
 * @param state_buffer_all: Optional buffer to store the per state values for all time frames. Used for debugging.
                            Shape: #FRAMES + 1 x #STATES
 */
template<bool fwd>
DEF_KERNEL
void baum_welch(unsigned num_frames, unsigned num_seqs, unsigned num_emissions, unsigned num_edges,
                unsigned const* state_offsets, unsigned const* edge_offsets, unsigned const* seq_lens,
                unsigned const* from_buffer, unsigned const* to_buffer, float const* weight_buffer, unsigned const* emission_idxs,
                unsigned const* init_states, unsigned const* final_states,
                float* prev_states, float* next_states, float const* am_scores, float* edge_buffer, float* norm_factors,
                unsigned const* merge_edge_offsets, unsigned const* merge_edge_idxs, unsigned const* merge_edge_targets,
                float* state_buffer_all) {
    __shared__ float edge_values[max_threads_per_block];

    unsigned seq = blockIdx.x * blockDim.y + threadIdx.y;
    if (seq > num_seqs) {
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

    float norm_factor = 0.0;
    if (not fwd and norm_factors != nullptr) {
        norm_factor = norm_factors[seq];
    }

    for (unsigned t = start_time; t != end_time; t += time_step) {
        float const* cur_frame_am_scores = am_scores + (t - (fwd ? 0 : 1)) * num_seqs * num_emissions + seq * num_emissions;
        float* cur_frame_edge_buffer = edge_buffer + (t - (fwd ? 0 : 1)) * num_edges;

        if (fwd and threadIdx.x == 0 and t == 0) {
            prev_states[init_states[seq]] = 0.0;
        }
        if (not fwd and threadIdx.x == 0 and t == seq_lens[seq]) {
            prev_states[final_states[seq]] = 0.0;
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

                    val = prev_val + edge_weight + cur_frame_am_scores[emission_idx] - norm_factor;
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
            unsigned num_states = state_offsets[num_seqs];
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

    if (fwd and norm_factors != nullptr and threadIdx.x == 0) {
        norm_factors[seq] = prev_states[final_states[seq]];
    }

    if (state_buffer_all != nullptr) {
        __syncthreads();
        unsigned num_states = state_offsets[num_seqs];
        for (unsigned state = state_offsets[seq] + threadIdx.x; state < state_offsets[seq+1]; state += blockDim.x) {
            unsigned idx = end_time * num_states + state;
            float sum = fwd ? prev_states[state] : (state_buffer_all[idx] + prev_states[state]);
            state_buffer_all[idx] = sum;
        }
    }
}
/**
 * The main kernel computing per edge gammas. The template parameter controls which direction is being run. This variant
 * computes the sum for all incoming edges into a state by atomic additions.
 * Kernel launch config: (x=#INTER_SEQ) x (y=#INTRA_SEQ, x=#THREADS)
 *
 * @param num_frames: The maximum number of time frames for any sequence.
 * @param num_seqs: Number of sequences = #SEQ
 * @param num_emissions: Number of emissions / labels
 * @param num_edges: Number of edges, should be the same as edge_offsets[-1]
 * @param state_offsets: Offsets for the states belonging to one sequence. Exclusive prefix sum over number of states
                         per seq. Shape #SEQ + 1
 * @param edge_offsets: Offsets of edges in each sequence, exclusive prefix sum over number of edges per seq,
                        Shape: #SEQ + 1
 * @param seq_lens: The sequence lengths. Shape: #SEQ
 * @param from_buffer: The source states of all edges. Shape #EDGES
 * @param to_buffer: The target states of all edges. Shape #EDGES
 * @param weight_buffer: The weight of each edge. Shape #EDGES
 * @param emission_idxs: emission/label index of each edge, Shape: #EDGES
 * @param init_states: The initial state for each sequence. Shape: #SEQ
 * @param final_states: The final state for each sequence. Shape: #SEQ
 * @param prev_states: An intermediate buffer to be used by this kernel to store the state probabilities. Shape: #STATES
 * @param next_states: An intermediate buffer to be used by this kernel to store the state probabilities. Shape: #STATES
 * @param am_scores: The emission probabilities per time/seq/emission. Shape: #TIME x #SEQ x #LABEL
 * @param edge_buffer: Buffer of edge gammas. Shape: #FRAMES x #EDGES (edges are flattened over all sequences)
 * @param norm_factors: Optional buffer of normalization values. If not set the user is expected to call the normalize
                        kernel after the backward pass separately. If set the forward pass will write the state value of
                        the final state in this vector and the backward pass will use it to directly normalize the
                        output. Shape: #SEQ
 * @param state_buffer_all: Optional buffer to store the per state values for all time frames. Used for debugging.
                            Shape: #FRAMES + 1 x #STATES
 */

template<bool fwd>
DEF_KERNEL
void baum_welch_v2(unsigned num_frames, unsigned num_seqs, unsigned num_emissions, unsigned num_edges,
                   unsigned const* state_offsets, unsigned const* edge_offsets, unsigned const* seq_lens,
                   unsigned const* from_buffer, unsigned const* to_buffer, float const* weight_buffer, unsigned const* emission_idxs,
                   unsigned const* init_states, unsigned const* final_states, unsigned const* final_state_offsets,
                   float* prev_states, float* next_states, float const* am_scores, float* edge_buffer, float* norm_factors,
                   float* state_buffer_all) {
    unsigned seq = blockIdx.x * blockDim.y + threadIdx.y;
    if (seq >= num_seqs) {
        return;
    }

    unsigned start_time = fwd ? 0 : seq_lens[seq];
    unsigned end_time   = fwd ? seq_lens[seq] : 0;
    int      time_step  = fwd ? 1 : -1;

    unsigned max_inner_loop_size = 1;
    for (unsigned s = blockIdx.x * blockDim.y; s < (blockIdx.x + 1) * blockDim.y and s < num_seqs; s++) {
        unsigned loop_size = (edge_offsets[s+1] - edge_offsets[s] + blockDim.x - 1) / blockDim.x;
        max_inner_loop_size = max(max_inner_loop_size, loop_size);
    }

    if (fwd and threadIdx.x == 0) {
        prev_states[init_states[seq]] = 0.0;
    }

    float norm_factor = 0.0;
    if (not fwd and norm_factors != nullptr) {
        norm_factor = norm_factors[seq];
    }

    for (unsigned t = start_time; t != end_time; t += time_step) {
        float const* cur_frame_am_scores = am_scores + (t - (fwd ? 0 : 1)) * num_seqs * num_emissions + seq * num_emissions;
        float* cur_frame_edge_buffer = edge_buffer + (t - (fwd ? 0 : 1)) * num_edges;

        if (not fwd and threadIdx.x == 0 and t == seq_lens[seq]) {
            for (unsigned fs = final_state_offsets[seq]; fs < final_state_offsets[seq+1]; fs++) {
                prev_states[final_states[fs]] = 0.0;
            }
        }

        for (unsigned state = state_offsets[seq] + threadIdx.x; state < state_offsets[seq+1]; state += blockDim.x) {
            next_states[state] = INF_F;
        }

        __syncthreads();  // synchronize to ensure that prev/next_states has been set by all threads

        for (unsigned l = 0; l < max_inner_loop_size; l++) {
            unsigned edge = edge_offsets[seq] + l * blockDim.x + threadIdx.x;

            if (edge < edge_offsets[seq+1]) {
                unsigned from     = from_buffer[edge];
                unsigned to       = to_buffer[edge];
                float    prev_val = prev_states[from];
                float    val      = INF_F;

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
                    float gamma = cur_frame_edge_buffer[edge] + prev_val - norm_factor;
                    cur_frame_edge_buffer[edge] = gamma;
                }
            }

            __syncthreads();  // synchronize to ensure that next_states has been set by all threads
        }

        if (state_buffer_all != nullptr) {
            unsigned num_states = state_offsets[num_seqs];
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

    if (fwd and norm_factors != nullptr and threadIdx.x == 0) {
        norm_factors[seq] = prev_states[final_states[seq]];
    }

    if (state_buffer_all != nullptr) {
        __syncthreads();
        unsigned num_states = state_offsets[num_seqs];
        for (unsigned state = state_offsets[seq] + threadIdx.x; state < state_offsets[seq+1]; state += blockDim.x) {
            unsigned idx = end_time * num_states + state;
            float sum = fwd ? prev_states[state] : (state_buffer_all[idx] + prev_states[state]);
            state_buffer_all[idx] = sum;
        }
    }
}

/**
 * Compute per frame and per sequence normalization values by summing all edge probabilities (in neg-log space) for a
 * given frame/seq. Then divide the edge probabilities by this normalization factors (subtraction in neg-log space).
 * Kernel launch config: (y=#FRAMES, x=#INTER_SEQ) x (y=#INTRA_SEQ, x=#THREADS)
 *
 * @param num_seqs: Number of sequences = #SEQ
 * @param num_edges: Number of edges, should be the same as edge_offsets[-1]
 * @param seq_lens: The sequence lengths. Shape: #SEQ
 * @param edge_buffer: Buffer of edge gammas. Shape: #FRAMES x #EDGES (edges are flattened over all sequences)
 * @param edge_offsets: Offsets of edges in each sequence, exclusive prefix sum over number of edges per seq,
                        Shape: #SEQ + 1
 * @param sum_output: output buffer for the normalization factors. Shape: #SEQ x #FRAMES
 */
DEF_KERNEL
void normalize(unsigned num_seqs, unsigned num_edges, unsigned const* seq_lens, float* edge_buffer, unsigned const* edge_offsets, float* sum_output) {
    __shared__ float local_sum[max_threads_per_block];

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
        sum_output[seq * gridDim.y + time] = local_sum[edge_idx];
    }

    sum = local_sum[threadIdx.y * blockDim.x];  // edge_idx for threadIdx.x == 0
    for (unsigned e = edge_offsets[seq] + threadIdx.x; e < edge_offsets[seq+1]; e += blockDim.x) {
        edge_buffer[e] -= sum;
    }
}

/**
 * A simple mean reduction kernel that takes per frame normalization values and computes the average
 * over the time taking into account the sequence length.
 * Kernel launch config: (x=#SEQ) x (x=#THREADS)
 *
 * @param num_frames: The maximum number of time frames for any sequence.
 * @param seq_lens: The sequence lengths. Shape: #SEQ
 * @param per_frame_norm: The norm for each sequence at each frame. Shape: #SEQ x #FRAMES
 * @param loss: Output buffer for the per seq normalization value which is also the loss. Shape: #SEQ
 */
DEF_KERNEL
void average_norm(unsigned num_frames, unsigned const* seq_lens, float const* per_frame_norm, float* loss) {
    __shared__ float local_sum[max_threads_per_block];

    const unsigned seq = blockIdx.x;

    float sum = 0.0;
    for (unsigned t = threadIdx.x; t < seq_lens[seq]; t += blockDim.x) {
        sum += per_frame_norm[seq * num_frames + t];
    }
    //printf("seq: %d thread: %d sum: %.4f\n", seq, threadIdx.x, sum);
    local_sum[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned offset = blockDim.x / 2; offset >= 1; offset /= 2) {
        if (threadIdx.x < offset and (threadIdx.x + offset) < seq_lens[seq]) {
            local_sum[threadIdx.x] += local_sum[threadIdx.x + offset];
        }
        if (threadIdx.x < offset) {
            //printf("seq: %d offset: %d thread: %d sum: %.4f\n", seq, offset, threadIdx.x, local_sum[threadIdx.x]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        loss[seq] = local_sum[0] / seq_lens[seq];
        //printf("seq: %d sum: %.4f\n", seq, loss[seq]);
    }
}

/**
 * Compute the gamma values on output label level from the per edge gammas. For that we sum over all edges
 * that have the same label.
 * Kernel launch config: (y=#FRAMES, x=#INTER_SEQ) x (y=#INTRA_SEQ, x=#THREADS)
 *
 * @param edge_buffer: Buffer of edge gammas. Shape: #FRAMES x #EDGES (edges are flattened over all sequences)
 * @param out: buffer of output label gammas, Shape: #FRAMES x #SEQ x #LABEL
 * @param emission_idxs: emission/label index of each edge, Shape: #EDGES
 * @param seq_lens: length of each sequence, Shape: #SEQ
 * @param edge_offsets: Offsets of edges in each sequence, exclusive prefix sum over number of edges per seq,
                        Shape: #SEQ + 1
 * @param frame_stride: Stride between two different time frames in the output buffer, typically #SEQ * #LABEL
 * @param seq_stride: Stride between two different sequences in the output buffer, typically #LABEL
 * @param num_edges: Number of edges, should be the same as edge_offsets[-1]
 * @param num_seqs: Number of sequences = #SEQ
 */
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
                                     torch::Tensor& am_scores, torch::Tensor& edges, torch::Tensor& weights,
                                     torch::Tensor& start_states, torch::Tensor& end_states,
                                     torch::Tensor& end_state_offsets,
                                     DebugOptionsV2 debug_options) {
    // am_scores is a [T, B, F] tensor

    assert_cmp(Ndarray_DIMS(start_states)[0], ==, Ndarray_DIMS(am_scores)[1]);
    // assert_cmp(Ndarray_DIMS(end_states)[0], ==, 1);
    // assert_cmp(Ndarray_DIMS(end_states)[1], ==, Ndarray_DIMS(am_scores)[1]);
    assert_cmp(Ndarray_DIMS(num_edges)[0], ==, Ndarray_DIMS(am_scores)[1]);
    assert_cmp(Ndarray_DIMS(num_states)[0], ==, Ndarray_DIMS(am_scores)[1]);

    auto          options = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor out     = torch::zeros_like(am_scores, options);
    torch::Tensor loss    = torch::zeros({am_scores.size(1)}, options);

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

    unsigned* d_start_states      = Ndarray_DEV_DATA_uint32(start_states);
    unsigned* d_end_states        = Ndarray_DEV_DATA_uint32(end_states);
    unsigned* d_end_state_offsets = Ndarray_DEV_DATA_uint32(end_state_offsets);
    unsigned* d_seq_lens          = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(seq_lens));
    float*    d_out               = Ndarray_DEV_DATA(out);
    float*    d_loss              = Ndarray_DEV_DATA(loss);

    unsigned n_frames    = Ndarray_DIMS(am_scores)[0];
    unsigned n_seqs      = Ndarray_DIMS(am_scores)[1];
    unsigned n_emissions = Ndarray_DIMS(am_scores)[2];
    unsigned n_edges     = Ndarray_DIMS(edges)[1];

    assert_cmp(n_frames, >, 0);

    unsigned frame_stride    = Ndarray_STRIDE(out, 0);
    unsigned sequence_stride = Ndarray_STRIDE(out, 1);

    // fill output tensor
    thrust::fill(thrust::device, d_out, d_out + n_frames * n_seqs * n_emissions, std::numeric_limits<float>::infinity());

    // calculate state offsets per seq
    thrust::host_vector<unsigned> state_offsets_host(n_seqs + 1, 0);
    thrust::inclusive_scan(h_num_states, h_num_states + n_seqs, state_offsets_host.begin() + 1);
    thrust::device_vector<unsigned> state_offsets = state_offsets_host;

    unsigned n_states = state_offsets_host.back();
    assert_cmp(n_states, >, 0);

    // calculate edge offsets per seq
    thrust::host_vector<unsigned> edge_offsets_host(n_seqs + 1, 0);
    thrust::inclusive_scan(h_num_edges, h_num_edges + n_seqs, edge_offsets_host.begin() + 1);
    thrust::device_vector<unsigned> d_edge_offsets = edge_offsets_host;

    // if not end state offsets were provided we assume that there is one end state per sequence
    if (not end_state_offsets.defined() or not end_state_offsets.numel()) {
        thrust::device_vector<unsigned> d_end_state_offsets_;
        thrust::sequence(d_end_state_offsets_.begin(), d_end_state_offsets_.end());
        d_end_state_offsets = d_end_state_offsets_.data().get();
    }

    // calculate size of fwb kernel blocks / grid
    unsigned max_edges_per_seq = *std::max_element(h_num_edges, h_num_edges + n_seqs);

    // we divide each block into smaller pieces to accommodate more sequences within one block
    unsigned n_threads_intra_seq = max_threads_per_block;
    while (n_threads_intra_seq / 2 > max_edges_per_seq) {
        n_threads_intra_seq /= 2;
    }
    unsigned n_threads_inter_seq = std::min(max_threads_per_block / n_threads_intra_seq, n_seqs);
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
                          (n_seqs, n_threads_intra_seq, d_to, d_edge_offsets.data().get(), merge_edge_offsets.data().get(), merge_edge_idxs.data().get(), merge_edge_targets.data().get()));
        HANDLE_LAST_ERROR();

        if (dump_edges) {
            write_merge_edges_to_file("fwd.edges.dump_v2." + std::to_string(batch_idx), merge_edge_offsets, merge_edge_idxs, merge_edge_targets);
        }
    }

    thrust::device_vector<float> per_frame_norm;
    if (debug_options.per_frame_norm) {
        per_frame_norm.resize(n_frames * n_seqs);
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
                           state_offsets.data().get(), d_edge_offsets.data().get(), d_seq_lens,
                           d_from, d_to, d_weights, d_emission_idxs,
                           d_start_states, d_end_states,
                           state_buffer_prev.data().get(), state_buffer_next.data().get(), d_am_scores, edge_buffer.data().get(),
                           debug_options.per_frame_norm ? nullptr : d_loss,
                           merge_edge_offsets.data().get(), merge_edge_idxs.data().get(), merge_edge_targets.data().get(),
                           dump_alignment ? state_buffer_all.data().get() : nullptr));
    }
    else {
        start_dev_kernel2(baum_welch_v2<true>, n_blocks_fbw, block_size_fbw, 0,
                          (n_frames, n_seqs, n_emissions, n_edges,
                           state_offsets.data().get(), d_edge_offsets.data().get(), d_seq_lens,
                           d_from, d_to, d_weights, d_emission_idxs, d_start_states, d_end_states, d_end_state_offsets,
                           state_buffer_prev.data().get(), state_buffer_next.data().get(), d_am_scores, edge_buffer.data().get(),
                           debug_options.per_frame_norm ? nullptr : d_loss,
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
                          (n_seqs, n_threads_intra_seq, d_from, d_edge_offsets.data().get(), merge_edge_offsets.data().get(), merge_edge_idxs.data().get(), merge_edge_targets.data().get()));
        HANDLE_LAST_ERROR();

        if (dump_edges) {
            write_merge_edges_to_file("bwd.edges.dump_v2." + std::to_string(batch_idx), merge_edge_offsets, merge_edge_idxs, merge_edge_targets);
        }
    }

    thrust::fill(state_buffer_prev.begin(), state_buffer_prev.end(), std::numeric_limits<float>::infinity());
    if (debug_options.explicit_merge) {
        start_dev_kernel2(baum_welch<false>, n_blocks_fbw, block_size_fbw, 0,
                          (n_frames, n_seqs, n_emissions, n_edges,
                           state_offsets.data().get(), d_edge_offsets.data().get(), d_seq_lens,
                           d_to, d_from, d_weights, d_emission_idxs,
                           d_start_states, d_end_states,
                           state_buffer_prev.data().get(), state_buffer_next.data().get(), d_am_scores, edge_buffer.data().get(),
                           debug_options.per_frame_norm ? nullptr : d_loss,
                           merge_edge_offsets.data().get(), merge_edge_idxs.data().get(), merge_edge_targets.data().get(),
                           dump_alignment ? state_buffer_all.data().get() : nullptr));
    }
    else {
        start_dev_kernel2(baum_welch_v2<false>, n_blocks_fbw, block_size_fbw, 0,
                          (n_frames, n_seqs, n_emissions, n_edges,
                           state_offsets.data().get(), d_edge_offsets.data().get(), d_seq_lens,
                           d_to, d_from, d_weights, d_emission_idxs,
                           d_start_states, d_end_states, d_end_state_offsets,
                           state_buffer_prev.data().get(), state_buffer_next.data().get(), d_am_scores, edge_buffer.data().get(),
                           debug_options.per_frame_norm ? nullptr : d_loss,
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

    if (debug_options.per_frame_norm) {
        // normalize at each time frame
        start_dev_kernel2(normalize, dim3(n_blocks_fbw, n_frames), block_size_fbw, 0,
                          (n_seqs, n_edges, d_seq_lens, edge_buffer.data().get(), d_edge_offsets.data().get(), per_frame_norm.data().get()));
        HANDLE_LAST_ERROR();
        // average the loss over all frames
        unsigned average_threads = max_threads_per_block;
        while (average_threads / 2 >= n_frames) {
            average_threads /= 2;
        }
        start_dev_kernel2(average_norm, n_seqs, average_threads, 0,
                          (n_frames, d_seq_lens, per_frame_norm.data().get(), d_loss));
        HANDLE_LAST_ERROR();
    }

    start_dev_kernel2(compute_result, dim3(n_blocks_fbw, n_frames), block_size_fbw, 0,
                      (edge_buffer.data().get(), d_out, d_emission_idxs, d_seq_lens, d_edge_offsets.data().get(),
                       frame_stride, sequence_stride, n_edges, n_seqs));
    HANDLE_LAST_ERROR();
    
    if (dump_output) {
        write_output_to_file(d_out, d_seq_lens, debug_options.pruning, n_frames, n_seqs, n_emissions, batch_idx);
    }

    batch_idx++;

    return {out, loss};
}
