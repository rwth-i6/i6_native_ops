#ifndef MONOTONIC_RNNT_GPU_RNNT_H
#define MONOTONIC_RNNT_GPU_RNNT_H

#if defined(DEBUG_TIME) || defined(DEBUG_LOG_SOFTMAX) || defined(DEBUG_FWDBWD) || defined(DEBUG_GRADS)
#include <stdio.h>
#endif

#ifdef DEBUG_TIME
#include <chrono>
#endif

#include "gpu_rnnt_kernel.h"
#include "gpu_workspace_manager.h"
#include "reduce.h"

template <typename ProbT>
class GpuRNNTComputer {
   public:
    // Noncopyable
    GpuRNNTComputer(GpuRNNTWorkspaceManager<ProbT> &workspace_manager, int blank, CUstream stream)
        : workspace_manager_(workspace_manager), blank_(blank), stream_(stream) {}

    GpuRNNTComputer(const GpuRNNTComputer &) = delete;

    GpuRNNTComputer &operator=(const GpuRNNTComputer &) = delete;

    RNNTStatus cost_and_grad(ProbT *costs, ProbT *grads) {
        int B = workspace_manager_.B_host();
        auto T = workspace_manager_.T_host();
        auto S = workspace_manager_.S_host();
        int V = workspace_manager_.V_host();
        int S_max = workspace_manager_.S_max_host();
        int T_max = workspace_manager_.T_max_host();
        auto min_allowed_s = workspace_manager_.min_allowed_s_host();
        auto max_allowed_s = workspace_manager_.max_allowed_s_host();

        bool training = (grads != nullptr);

        // denom

#ifdef DEBUG_TIME
        auto start = std::chrono::high_resolution_clock::now();
#endif
        setup_log_softmax_denom();
#ifdef DEBUG_TIME
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printf("DEBUG: log_softmax denom %.2f ms\n", elapsed.count() * 1000);
        start = std::chrono::high_resolution_clock::now();
#endif

#ifdef DEBUG_LOG_SOFTMAX
        auto cpu_acts = workspace_manager_.acts_host();
        auto cpu_denoms = workspace_manager_.denom_host();
        int denom_start_indices_host[B];
        denom_start_indices_host[0] = 0;
        for (int b = 1; b < B; ++b) {
            denom_start_indices_host[b] = denom_start_indices_host[b - 1] + T[b - 1] * (S[b - 1] + 1);
        }
        printf("gpu acts and denoms\n");
        for (int b = 0; b < B; b++) {
            printf("b = %d\n", b);
            for (int t = 0; t < 1; t++) {
                printf("  t = %d\n", t);
                for (int s = 0; s <= 0; s++) {
                    printf("    s = %d\n      ", s);
                    int denom_idx = denom_start_indices_host[b] + t * (S[b] + 1) + s;
                    for (int v = 0; v < V; v++) {
                        printf("%.4f ", cpu_acts[denom_idx * V + v]);
                    }
                    printf("=> %.4f;\n", cpu_denoms[denom_idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("gpu probs\n");
        for (int b = 0; b < B; b++) {
            printf("b = %d\n", b);
            for (int t = 0; t < 1; t++) {
                printf("  t = %d\n", t);
                for (int s = 0; s <= 0; s++) {
                    printf("    s = %d\n      ", s);
                    int denom_idx = denom_start_indices_host[b] + t * (S[b] + 1) + s;
                    for (int v = 0; v < V; v++) {
                        printf("%.4f ", exp(cpu_acts[denom_idx * V + v] + cpu_denoms[denom_idx]));
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
        }
#endif

        // alphas

#ifdef USE_NAIVE_KERNEL
        compute_alphas_kernel_naive<ProbT><<<1, B, 0, stream_>>>(
            workspace_manager_.acts, workspace_manager_.denom, workspace_manager_.alphas, workspace_manager_.ll_forward,
            workspace_manager_.T, workspace_manager_.S, workspace_manager_.V, workspace_manager_.labels,
            workspace_manager_.var_start_offsets, workspace_manager_.denom_start_indices, workspace_manager_.S_max,
            workspace_manager_.T_max, workspace_manager_.min_allowed_s, workspace_manager_.max_allowed_s, blank_);
#else
        compute_alphas_kernel<ProbT><<<B, S_max + 1, 0, stream_>>>(
            workspace_manager_.acts, workspace_manager_.denom, workspace_manager_.alphas, workspace_manager_.ll_forward,
            workspace_manager_.T, workspace_manager_.S, workspace_manager_.V, workspace_manager_.labels,
            workspace_manager_.var_start_offsets, workspace_manager_.denom_start_indices, workspace_manager_.S_max,
            workspace_manager_.T_max, workspace_manager_.min_allowed_s, workspace_manager_.max_allowed_s, blank_);
#endif
#ifdef DEBUG_TIME
        cudaStreamSynchronize(stream_);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printf("DEBUG: compute_alphas_kernel %.2f ms\n", elapsed.count() * 1000);
#endif
#ifdef DEBUG_FWDBWD
        auto alphas = workspace_manager_.alphas_host();
        auto var_start_offsets = workspace_manager_.var_start_offsets_host();
        for (int b = 0; b < B; b++) {
            printf("gpu alphas (b = %d, T = %d, S = %d):\n", b, T[b], S[b]);
            float *alphas_b = alphas.data() + var_start_offsets[b];
            for (int s = S[b]; s >= 0; --s) {
                for (int t = -1; t < T[b]; ++t) {
                    printf("%.2f ", alpha(alphas_b, t, s, T[b], S[b], min_allowed_s.data() + b * T_max,
                                          max_allowed_s.data() + b * T_max));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("forward likelihood\n");
        auto ll_forward = workspace_manager_.ll_forward_host();
        for (int b = 0; b < B; ++b) {
            printf("%.2f ", ll_forward[b]);
        }
        printf("\n\n");
#endif
        if (training) {
            // betas
#ifdef DEBUG_TIME
            start = std::chrono::high_resolution_clock::now();
#endif
#ifdef USE_NAIVE_KERNEL
            compute_betas_kernel_naive<ProbT><<<1, B, 0, stream_>>>(
                workspace_manager_.acts, workspace_manager_.denom, workspace_manager_.betas,
                workspace_manager_.ll_backward, workspace_manager_.T, workspace_manager_.S, workspace_manager_.V,
                workspace_manager_.labels, workspace_manager_.var_start_offsets, workspace_manager_.denom_start_indices,
                workspace_manager_.S_max, workspace_manager_.T_max, workspace_manager_.min_allowed_s,
                workspace_manager_.max_allowed_s, blank_);
#else
            compute_betas_kernel<ProbT><<<B, S_max + 1, 0, stream_>>>(
                workspace_manager_.acts, workspace_manager_.denom, workspace_manager_.betas,
                workspace_manager_.ll_backward, workspace_manager_.T, workspace_manager_.S, workspace_manager_.V,
                workspace_manager_.labels, workspace_manager_.var_start_offsets, workspace_manager_.denom_start_indices,
                workspace_manager_.S_max, workspace_manager_.T_max, workspace_manager_.min_allowed_s,
                workspace_manager_.max_allowed_s, blank_);
#endif
#ifdef DEBUG_TIME
            cudaStreamSynchronize(stream_);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            printf("DEBUG: compute_betas_kernel %.2f ms\n", elapsed.count() * 1000);
#endif
#ifdef DEBUG_FWDBWD
            auto betas = workspace_manager_.betas_host();
            for (int b = 0; b < B; b++) {
                printf("gpu betas (b = %d, T = %d, S = %d):\n", b, T[b], S[b]);
                float *betas_b = betas.data() + var_start_offsets[b];
                for (int s = S[b]; s >= 0; --s) {
                    for (int t = 0; t <= T[b]; ++t) {
                        printf("%.2f ", beta(betas_b, t, s, T[b], S[b], min_allowed_s.data() + b * T_max,
                                             max_allowed_s.data() + b * T_max));
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("backward likelihood\n");
            auto ll_backward = workspace_manager_.ll_backward_host();
            for (int b = 0; b < B; ++b) {
                printf("%.2f ", ll_backward[b]);
            }
            printf("\n\n");
#endif

            // gradient
#ifdef DEBUG_TIME
            start = std::chrono::high_resolution_clock::now();
#endif
            compute_grad_kernel<128, ProbT><<<workspace_manager_.num_denoms(), 128, 0, stream_>>>(
                grads, workspace_manager_.acts, workspace_manager_.denom, workspace_manager_.alphas,
                workspace_manager_.betas, workspace_manager_.ll_forward, workspace_manager_.B, workspace_manager_.T,
                workspace_manager_.S, workspace_manager_.labels, workspace_manager_.var_start_offsets,
                workspace_manager_.denom_start_indices, workspace_manager_.S_max, workspace_manager_.T_max,
                workspace_manager_.min_allowed_s, workspace_manager_.max_allowed_s, workspace_manager_.V, blank_);
#ifdef DEBUG_TIME
            cudaStreamSynchronize(stream_);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            printf("DEBUG: compute_grad_kernel %.2f ms\n", elapsed.count() * 1000);
#endif
#ifdef DEBUG_GRADS
            std::vector<ProbT> cpu_grads(workspace_manager_.num_denoms() * V);
            cudaMemcpy(cpu_grads.data(), grads, sizeof(ProbT) * cpu_grads.size(), cudaMemcpyDeviceToHost);

            printf("gpu grads\n");
            int grad_idx = 0;
            for (int b = 0; b < B; ++b) {
                printf("b = %d\n", b);
                for (int t = 0; t < T[b]; ++t) {
                    printf("  t = %d\n", t);
                    for (int s = 0; s <= S[b]; ++s) {
                        printf("    s = %d\n      ", s);
                        for (int v = 0; v < V; ++v) {
                            printf("%.4f ", cpu_grads[grad_idx]);
                            grad_idx += 1;
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                printf("\n");
            }
#endif
        }

        // cost
        cudaMemcpy(costs, workspace_manager_.ll_forward, sizeof(ProbT) * B, cudaMemcpyDeviceToHost);
        for (int b = 0; b < B; ++b) {
            costs[b] = -costs[b];
        }
        return RNNT_STATUS_SUCCESS;
    }
    RNNTStatus cost(ProbT *costs) { return cost_and_grad(costs, nullptr); }

   private:
    GpuRNNTWorkspaceManager<ProbT> &workspace_manager_;
    int blank_;
    CUstream stream_;

    void setup_log_softmax_denom() {
        const int num_denoms = workspace_manager_.num_denoms();
        const int V = workspace_manager_.V_host();

        reduce_max(workspace_manager_.acts, workspace_manager_.denom, V, num_denoms, false, stream_);
        reduce_exp(workspace_manager_.acts, workspace_manager_.denom, V, num_denoms, true, stream_);
    }
};

#endif  // MONOTONIC_RNNT_GPU_RNNT_H
