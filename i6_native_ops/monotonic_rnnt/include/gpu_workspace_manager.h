#ifndef MONOTONIC_RNNT_GPU_WORKSPACE_MANAGER_H
#define MONOTONIC_RNNT_GPU_WORKSPACE_MANAGER_H

#include <cstdio>
#ifdef DEBUG_SPACE
#include <stdio.h>
#endif

#include <algorithm>
#include <vector>

#include "status.h"
#include "workspace_manager.h"

template <typename dtype>
class GpuRNNTWorkspaceManager : public RNNTWorkspaceManager {
   public:
    /** For a given set of minibatch sample shapes manager the required
     *  workspace. Can calculate required size for all variables and
     *  performs structuring and access handling inside the allocated space
     *  after it is passed. Also handles memory access for activations and
     *labels.
     *
     * \param [in]  acts 1-D flattened array containing all the model logits in
     *packed row-major order \param [in]  labels 1-D flattened array containing
     *all the labels in non-packed row-major order \param [in]  B Number of
     *examples in minibatch. \param [in]  T Number of time steps for each
     *minibatch sample \param [in]  S Number of labels for each minibatch sample
     * \param [in]  V alphabet size including blank
     *
     * \return Status information
     **/
    explicit GpuRNNTWorkspaceManager(const dtype *const acts, const int *const labels, const int B, const int *T,
                                     const int *S, const int V)
        : B_h(B),
          B(nullptr),
          T(T),
          S(S),
          S_max(nullptr),
          T_max(nullptr),
          V_h(V),
          V(nullptr),
          min_allowed_s(nullptr),
          max_allowed_s(nullptr),
          acts(acts),
          labels(labels),
          denom_start_indices(nullptr),
          var_start_offsets(nullptr),
          alphas(nullptr),
          betas(nullptr),
          denom(nullptr),
          dtype_size_(sizeof(dtype)) {}

    GpuRNNTWorkspaceManager(const GpuRNNTWorkspaceManager &) = delete;

    ~GpuRNNTWorkspaceManager() override = default;

    void *workspace_;  // device

    const int B_h;  // host
    const int V_h;  // host

    const int *T;  // device
    const int *S;  // device
    int *B;        // device
    int *V;        // device

    const dtype *const acts;  // device
    const int *const labels;  // device

    dtype *denom;   // workspace
    dtype *alphas;  // workspace
    dtype *betas;   // workspace

    int *min_allowed_s;  // workspace
    int *max_allowed_s;  // workspace

    int *denom_start_indices;  // workspace
    int *var_start_offsets;    // workspace

    int *S_max;  // workspace
    int *T_max;  // workspace

    dtype *ll_forward;   // workspace
    dtype *ll_backward;  // workspace

    [[nodiscard]] std::vector<int> T_host() const {
        std::vector<int> T_h(B_h);
        cudaMemcpy(T_h.data(), T, B_h * sizeof(int), cudaMemcpyDeviceToHost);
        return T_h;
    }
    [[nodiscard]] std::vector<int> S_host() const {
        std::vector<int> S_h(B_h);
        cudaMemcpy(S_h.data(), S, B_h * sizeof(int), cudaMemcpyDeviceToHost);
        return S_h;
    }
    [[nodiscard]] inline int B_host() const { return B_h; }

    [[nodiscard]] inline int V_host() const { return V_h; }

    [[nodiscard]] int num_denoms() const {
        auto T_h = T_host();
        auto S_h = S_host();

        int result = 0;
        for (int b = 0; b < B_h; ++b) {
            result += T_h[b] * (S_h[b] + 1);
        }
        return result;
    }

    [[nodiscard]] int num_fwd_bwd_var_positions() const {
        auto T_h = T_host();
        auto S_h = S_host();

        int fwd_bwd_var_positions = 0;
        for (int b = 0; b < B_h; ++b) {
            fwd_bwd_var_positions += T_h[b] * (S_h[b] + 1);
        }

        return fwd_bwd_var_positions;
    }

    [[nodiscard]] std::vector<int> var_start_offsets_host() const {
        std::vector<int> var_start_offsets_h(B_h);
        cudaMemcpy(var_start_offsets_h.data(), var_start_offsets, sizeof(int) * var_start_offsets_h.size(),
                   cudaMemcpyDeviceToHost);
        return var_start_offsets_h;
    }

    [[nodiscard]] std::vector<dtype> acts_host() const {
        std::vector<dtype> acts_h(num_denoms() * V_host());
        cudaMemcpy(acts_h.data(), acts, dtype_size_ * acts_h.size(), cudaMemcpyDeviceToHost);
        return acts_h;
    }

    [[nodiscard]] std::vector<dtype> denom_host() const {
        std::vector<dtype> denom_h(num_denoms());
        cudaMemcpy(denom_h.data(), denom, dtype_size_ * denom_h.size(), cudaMemcpyDeviceToHost);
        return denom_h;
    }

    [[nodiscard]] std::vector<dtype> alphas_host() const {
        std::vector<dtype> alphas_h(num_fwd_bwd_var_positions());
        cudaMemcpy(alphas_h.data(), alphas, dtype_size_ * alphas_h.size(), cudaMemcpyDeviceToHost);
        return alphas_h;
    }

    [[nodiscard]] std::vector<dtype> betas_host() const {
        std::vector<dtype> betas_h(num_fwd_bwd_var_positions());
        cudaMemcpy(betas_h.data(), betas, dtype_size_ * betas_h.size(), cudaMemcpyDeviceToHost);
        return betas_h;
    }

    [[nodiscard]] int S_max_host() const {
        int S_max_h;
        cudaMemcpy(&S_max_h, S_max, sizeof(int), cudaMemcpyDeviceToHost);
        return S_max_h;
    }

    [[nodiscard]] int T_max_host() const {
        int T_max_h;
        cudaMemcpy(&T_max_h, T_max, sizeof(int), cudaMemcpyDeviceToHost);
        return T_max_h;
    }

    [[nodiscard]] std::vector<int> min_allowed_s_host() const {
        std::vector<int> min_allowed_s_h(T_max_host() * B_host());
        cudaMemcpy(min_allowed_s_h.data(), min_allowed_s, sizeof(int) * min_allowed_s_h.size(), cudaMemcpyDeviceToHost);
        return min_allowed_s_h;
    }

    [[nodiscard]] std::vector<int> max_allowed_s_host() const {
        std::vector<int> max_allowed_s_h(T_max_host() * B_host());
        cudaMemcpy(max_allowed_s_h.data(), max_allowed_s, sizeof(int) * max_allowed_s_h.size(), cudaMemcpyDeviceToHost);
        return max_allowed_s_h;
    }

    [[nodiscard]] std::vector<dtype> ll_forward_host() const {
        std::vector<dtype> ll_forward_h(B_h);
        cudaMemcpy(ll_forward_h.data(), ll_forward, dtype_size_ * B_h, cudaMemcpyDeviceToHost);
        return ll_forward_h;
    }

    [[nodiscard]] std::vector<dtype> ll_backward_host() const {
        std::vector<dtype> ll_backward_h(B_h);
        cudaMemcpy(ll_backward_h.data(), ll_backward, dtype_size_ * B_h, cudaMemcpyDeviceToHost);
        return ll_backward_h;
    }

    void restrict_to_alignment(const int *const alignments, int max_shift, int blank_idx) {
        auto T_h = T_host();
        int B_h = B_host();
        int T_max_h = T_max_host();

        std::vector<int> alignments_h(B_h * T_max_h);
        std::vector<int> min_allowed_s_h(B_h * T_max_h);
        std::vector<int> max_allowed_s_h(B_h * T_max_h);

        cudaMemcpy(alignments_h.data(), alignments, sizeof(int) * alignments_h.size(), cudaMemcpyDeviceToHost);

        for (int b = 0; b < B_h; ++b) {
            std::vector<int> s_index_mapping(T_h[b] + 1, 0);
            for (int t = 0; t < T_h[b]; ++t) {
                if (alignments_h[b * T_max_h + t] == blank_idx) {
                    s_index_mapping[t + 1] = s_index_mapping[t];
                } else {
                    s_index_mapping[t + 1] = s_index_mapping[t] + 1;
                }
            }
            for (int t = 0; t < T_h[b]; ++t) {
                min_allowed_s_h[b * T_max_h + t] = s_index_mapping[std::max(0, t + 1 - max_shift)];
                max_allowed_s_h[b * T_max_h + t] = s_index_mapping[std::min(T_h[b], t + 1 + max_shift)];
            }
        }

        cudaMemcpy(min_allowed_s, min_allowed_s_h.data(), sizeof(int) * min_allowed_s_h.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(max_allowed_s, max_allowed_s_h.data(), sizeof(int) * max_allowed_s_h.size(), cudaMemcpyHostToDevice);
    }

    /**
     * Calculate required memory for denominator, alphas and betas.
     * This memory needs to be allocated externally.
     *
     * \param [out] size_bytes Pointer to a scalar where the memory
     *              requirement in bytes will be placed.
     **/
    RNNTStatus get_workspace_size(size_t *size_bytes) const {
        auto T_h = T_host();
        auto S_h = S_host();

        if (B_h <= 0) {
            return RNNT_STATUS_INVALID_VALUE;
        }
        for (int b = 0; b < B_h; ++b) {
            if (T_h[b] <= 0 || S_h[b] < 0 || T_h[b] < S_h[b]) {
                return RNNT_STATUS_INVALID_VALUE;
            }
        }
        int T_max_h = *std::max_element(T_h.begin(), T_h.end());

        *size_bytes = dtype_size_ * num_denoms()                       // denom
                      + 2 * dtype_size_ * num_fwd_bwd_var_positions()  // alpha+beta
                      + 2 * B_h * sizeof(int)                          // var_start_offsets + denom_start_indices
                      + 2 * B_h * dtype_size_                          // ll_forward + ll_backward
                      + 2 * B_h * T_max_h * sizeof(int)                // min/max allowed s
                      + 4 * sizeof(int);                               // B, V, S_max, T_max

#ifdef DEBUG_SPACE
        printf("Reserve %.3f mb of memory for computations\n", static_cast<float>(*size_bytes) / 1e6);
#endif

        return RNNT_STATUS_SUCCESS;
    }

    void set_workspace(void *workspace) {
        workspace_ = workspace;

        auto T_h = T_host();
        auto S_h = S_host();

        int var_start_offsets_host[B_h + 1];
        var_start_offsets_host[0] = 0;
        for (int b = 1; b <= B_h; ++b) {
            var_start_offsets_host[b] = var_start_offsets_host[b - 1] + T_h[b - 1] * (S_h[b - 1] + 1);
        }

        int denom_start_indices_host[B_h + 1];
        denom_start_indices_host[0] = 0;
        for (int b = 1; b <= B_h; ++b) {
            denom_start_indices_host[b] = denom_start_indices_host[b - 1] + T_h[b - 1] * (S_h[b - 1] + 1);
        }

        size_t current_offset = 0ul;

        denom = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * denom_start_indices_host[B_h];

        alphas = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * var_start_offsets_host[B_h];

        betas = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * var_start_offsets_host[B_h];

        denom_start_indices = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += B_h * sizeof(int);
        cudaMemcpy(denom_start_indices, denom_start_indices_host, B_h * sizeof(int), cudaMemcpyHostToDevice);

        var_start_offsets = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += B_h * sizeof(int);
        cudaMemcpy(var_start_offsets, var_start_offsets_host, B_h * sizeof(int), cudaMemcpyHostToDevice);

        ll_forward = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * B_h;

        ll_backward = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * B_h;

        B = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += sizeof(int);
        cudaMemcpy(B, &B_h, sizeof(int), cudaMemcpyHostToDevice);

        V = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += sizeof(int);
        cudaMemcpy(V, &V_h, sizeof(int), cudaMemcpyHostToDevice);

        int S_max_h = *std::max_element(S_h.begin(), S_h.end());
        S_max = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += sizeof(int);
        cudaMemcpy(S_max, &S_max_h, sizeof(int), cudaMemcpyHostToDevice);

        int T_max_h = *std::max_element(T_h.begin(), T_h.end());
        T_max = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += sizeof(int);
        cudaMemcpy(T_max, &T_max_h, sizeof(int), cudaMemcpyHostToDevice);

        std::vector<int> min_allowed_s_h(B_h * T_max_h, 0);
        min_allowed_s = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += sizeof(int) * min_allowed_s_h.size();
        cudaMemcpy(min_allowed_s, min_allowed_s_h.data(), sizeof(int) * min_allowed_s_h.size(), cudaMemcpyHostToDevice);

        std::vector<int> max_allowed_s_h(B_h * T_max_h);
        for (int b = 0; b < B_h; ++b) {
            std::fill_n(max_allowed_s_h.begin() + b * T_max_h, T_max_h, S_h[b]);
        }
        max_allowed_s = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += sizeof(int) * max_allowed_s_h.size();
        cudaMemcpy(max_allowed_s, max_allowed_s_h.data(), sizeof(int) * max_allowed_s_h.size(), cudaMemcpyHostToDevice);
    }

    RNNTStatus create_workspace() {
        size_t gpu_bytes;
        auto status = get_workspace_size(&gpu_bytes);
        if (status == RNNT_STATUS_SUCCESS) {
            void *workspace;
            cudaMalloc(&workspace, gpu_bytes);
            set_workspace(workspace);
        }
        return status;
    }

    void free_workspace() { cudaFree(workspace_); }

   private:
    size_t dtype_size_;  // host
};

#endif  // MONOTONIC_RNNT_GPU_WORKSPACE_MANAGER_H
