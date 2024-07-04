#ifndef MONOTONIC_RNNT_CPU_WORKSPACE_MANAGER_H
#define MONOTONIC_RNNT_CPU_WORKSPACE_MANAGER_H

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

#include "status.h"
#include "workspace_manager.h"

template <typename dtype>
class CpuRNNTWorkspaceManager : public RNNTWorkspaceManager {
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

    explicit CpuRNNTWorkspaceManager(const dtype *const acts, const int *const labels, const int B, const int *T,
                                     const int *S, const int V)
        : T_(T),
          S_(S),
          B_(B),
          V_(V),
          acts_(acts),
          labels_(labels),
          alphas_(B),
          betas_(B),
          denom_(B),
          dtype_size_(sizeof(dtype)),
          act_start_indices_(B),
          S_max_(*std::max_element(S, S + B)),
          workspace_(nullptr) {
        act_start_indices_[0] = 0;
        for (int b = 1ul; b < B_; ++b) {
            act_start_indices_[b] = act_start_indices_[b - 1] + T_[b - 1] * (S_[b - 1] + 1) * V_;
        }

        min_allowed_s_.reserve(B_);
        max_allowed_s_.reserve(B_);
        for (int b = 0ul; b < B_; ++b) {
            min_allowed_s_.push_back(std::vector<int>(T_[b], 0));
            max_allowed_s_.push_back(std::vector<int>(T_[b], S_[b]));
        }
    }

    CpuRNNTWorkspaceManager(const CpuRNNTWorkspaceManager &) = delete;

    ~CpuRNNTWorkspaceManager() override = default;

    [[nodiscard]] inline int T(int b) const { return T_[b]; }

    [[nodiscard]] inline int S(int b) const { return S_[b]; }

    [[nodiscard]] inline int alpha_s_min(int b, int t) const {
        return std::max(min_allowed_s_[b][t], t - (T_[b] - 1 - S_[b]));
    }

    [[nodiscard]] inline int alpha_s_max(int b, int t) const { return std::min(max_allowed_s_[b][t], t + 1); }

    [[nodiscard]] inline int beta_s_min(int b, int t) const {
        if (t == 0) {
            return 0;
        }
        return std::max(min_allowed_s_[b][t - 1], t - (T_[b] - S_[b]));
    }

    [[nodiscard]] inline int beta_s_max(int b, int t) const {
        if (t == 0) {
            return 0;
        }

        return std::min(max_allowed_s_[b][t - 1], t);
    }

    [[nodiscard]] inline int B() const { return B_; }

    [[nodiscard]] inline int V() const { return V_; }

    /**
     * Calculate required memory for denominator, alphas and betas.
     * This memory needs to be allocated externally.
     *
     * \param [out] size_bytes Pointer to a scalar where the memory
     *              requirement in bytes will be placed.
     **/
    RNNTStatus get_workspace_size(size_t *size_bytes) const {
        if (B_ <= 0) {
            return RNNT_STATUS_INVALID_VALUE;
        }
        for (int b = 0; b < B_; ++b) {
            if (T_[b] <= 0 || S_[b] < 0 || T_[b] < S_[b]) {
                return RNNT_STATUS_INVALID_VALUE;
            }
        }

        *size_bytes = calc_total_denom_space() + 2 * calc_total_fwdbwd_var_space_();
#ifdef DEBUG_SPACE
        printf("Reserve %.3f mb of memory for computations\n", static_cast<float>(*size_bytes) / 1e6);
#endif

        return RNNT_STATUS_SUCCESS;
    }

    int label(int b, int s) const {
        assert(0 <= b);
        assert(b <= B_);
        assert(0 <= s);
        assert(s < S_[b]);
        return labels_[b * S_max_ + s];
    }

    [[nodiscard]] int act_index(int b, int t, int s, int v) const {
        assert(0 <= b);
        assert(b <= B_);
        assert(0 <= t);
        assert(t < T_[b]);
        assert(0 <= s);
        assert(s <= S_[b]);
        assert(0 <= v);
        assert(v <= V_);
        return act_start_indices_[b] + (t * (S_[b] + 1) + s) * V_ + v;
    }

    [[nodiscard]] inline dtype act(int b, int t, int s, int v) const { return acts_[act_index(b, t, s, v)]; }

    void set_denom(int b, int t, int s, dtype value) {
        assert(0 <= b);
        assert(b < B_);
        assert(0 <= t);
        assert(t < T_[b]);
        assert(0 <= s);
        assert(s <= S_[b]);
        denom_[b][t * (S_[b] + 1) + s] = value;
    }

    dtype &get_denom(int b, int t, int s) {
        assert(0 <= b);
        assert(b < B_);
        assert(0 <= t);
        assert(t < T_[b]);
        assert(0 <= s);
        assert(s <= S_[b]);
        return denom_[b][t * (S_[b] + 1) + s];
    }

    void inline set_alpha(int b, int t, int s, dtype value) { alphas_[b][alpha_idx_(b, t, s)] = value; }

    dtype get_alpha(int b, int t, int s) const {
        // Note: t = -1 and s = -1 are allowed to be used as virtual starts
        // They correspond to constants
        if (s == -1) {
            return -std::numeric_limits<dtype>::infinity();
        }

        if (t == -1) {
            return s == 0 ? 0 : -std::numeric_limits<dtype>::infinity();
        }

        if (s < min_allowed_s_[b][t] || s > max_allowed_s_[b][t]) {
            return -std::numeric_limits<dtype>::infinity();
        }

        if (s > t + 1 || S_[b] - s > T_[b] - 1 - t) {
            return -std::numeric_limits<dtype>::infinity();
        }

        return alphas_[b][alpha_idx_(b, t, s)];
    }

    void inline set_beta(int b, int t, int s, dtype value) { betas_[b][beta_idx_(b, t, s)] = value; }

    dtype get_beta(int b, int t, int s) {
        // Note: t = T and s = S+1 are allowed to be used as virtual starts
        // They correspond to constants
        if (s == S_[b] + 1) {
            return -std::numeric_limits<dtype>::infinity();
        }

        if (t == T_[b]) {
            return s == S_[b] ? 0 : -std::numeric_limits<dtype>::infinity();
        }

        if (t > 0 && (s < min_allowed_s_[b][t - 1] || s > max_allowed_s_[b][t - 1])) {
            return -std::numeric_limits<dtype>::infinity();
        }

        if (s > t || S_[b] - s - 1 > T_[b] - 1 - t) {
            return -std::numeric_limits<dtype>::infinity();
        }

        return betas_[b][beta_idx_(b, t, s)];
    }

    void restrict_to_alignment(const int *const alignments, int max_shift, int blank_idx) {
        int T_max = *(std::max_element(T_, T_ + B_));

        for (int b = 0; b < B_; ++b) {
            std::vector<int> s_index_mapping(T_[b] + 1, 0);
            for (int t = 0; t < T_[b]; ++t) {
                if (alignments[b * T_max + t] == blank_idx) {
                    s_index_mapping[t + 1] = s_index_mapping[t];
                } else {
                    s_index_mapping[t + 1] = s_index_mapping[t] + 1;
                }
            }
            for (int t = 0; t < T_[b]; ++t) {
                min_allowed_s_[b][t] = s_index_mapping[std::max(0, t + 1 - max_shift)];
                max_allowed_s_[b][t] = s_index_mapping[std::min(T_[b], t + 1 + max_shift)];
            }
        }
    }

    void set_workspace(void *workspace) {
        size_t current_offset = 0ul;

        for (int b = 0ul; b < B_; ++b) {
            denom_[b] = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
            current_offset += calc_denom_space_(b);

            alphas_[b] = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
            current_offset += calc_fwdbwd_var_space_(b);

            betas_[b] = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
            current_offset += calc_fwdbwd_var_space_(b);
        }
    }

    RNNTStatus create_workspace() {
        size_t size_bytes;
        RNNTStatus status = get_workspace_size(&size_bytes);
        if (status == RNNT_STATUS_SUCCESS) {
            workspace_ = malloc(size_bytes);
            set_workspace(workspace_);
        }

        return status;
    }

    void free_workspace() { free(workspace_); }

   private:
    void *workspace_;

    const int *T_;
    const int *S_;
    const int S_max_;
    const int B_;
    const int V_;
    size_t dtype_size_;

    const dtype *const acts_;
    const int *const labels_;

    std::vector<int> act_start_indices_;

    std::vector<dtype *> denom_;
    std::vector<dtype *> alphas_;
    std::vector<dtype *> betas_;

    std::vector<std::vector<int>> min_allowed_s_;
    std::vector<std::vector<int>> max_allowed_s_;

    [[nodiscard]] inline size_t calc_denom_space_(int b) const { return dtype_size_ * T_[b] * (S_[b] + 1); }

    [[nodiscard]] size_t calc_total_denom_space() const {
        size_t result = 0ul;
        for (int b = 0ul; b < B_; ++b) {
            result += calc_denom_space_(b);
        }
        return result;
    }

    [[nodiscard]] size_t calc_fwdbwd_var_space_(int b) const {
        // alphas & betas
        // For example with T = 7, S = 3, the shape of the grid of
        // reachable states for the forward variables is
        // . . # # # # #
        //  . # # # # # .
        //  # # # # # . .
        //  # # # # . . .
        // which decomposes to S + 1 rows of length T + 1 - S each minus 1 state
        // in the last row. This is symmetric for the betas
        return dtype_size_ * ((T_[b] + 1 - S_[b]) * (S_[b] + 1) - 1);
    }

    [[nodiscard]] size_t calc_total_fwdbwd_var_space_() const {
        size_t result = 0ul;
        for (int b = 0ul; b < B_; ++b) {
            result += calc_fwdbwd_var_space_(b);
        }
        return result;
    }

    [[nodiscard]] int alpha_idx_(int b, int t, int s) const {
        // Shape in batch sample with T_[b] = 7, S_[b] = 3 is
        // . . # # # # #
        // . # # # # # .
        // # # # # # . .
        // # # # # . . .
        // alphas are saved row-wise and packed
        // The overall index is
        // sum of rows below current label index + t - right shift of current row
        // e.g. alphas[t = 1, s = 1] is at index 5 in the array or alphas[t = 2, s = 3]
        // is at index 14 or alphas[t = 5, s = 2] is at index 13

        // Shape in batch sample with T_[b] = 7, S_[b] = 5 is
        // . . . . # # #
        // . . . # # # .
        // . . # # # . .
        // . # # # . . .
        // # # # . . . .
        // # # . . . . .
        // alphas are saved column-wise and packed
        // The overall index is
        // sum of columns to the left of current timestep + s - a height offset
        // inside the right triangle e.g. alphas[t = 1, s = 1] is at index 3 in
        // the array or alphas[t = 2, s = 3] is at index 8 or alphas[t = 5, s =
        // 2] is at index 16

        assert(0 <= b);
        assert(b < B_);
        assert(0 <= t);
        assert(t < T_[b]);
        assert(0 <= s);
        assert(s <= S_[b]);
        assert(s <= t + 1);
        assert(T_[b] - 1 - t >= S_[b] - s);

        // each row has length T + 1 - S except the bottom row which has one less element
        int sum_below_row = s > 0 ? s * (T_[b] + 1 - S_[b]) - 1 : 0;

        // The bottom two rows both have no right shift
        int right_shift = s > 0 ? s - 1 : 0;

        return sum_below_row + t - right_shift;
    }

    [[nodiscard]] int beta_idx_(int b, int t, int s) const {
        // This is symmetric to the alpha indices just with flipped axes
        return alpha_idx_(b, T_[b] - 1 - t, S_[b] - s);
    }
};

#endif  // MONOTONIC_RNNT_CPU_WORKSPACE_MANAGER_H
