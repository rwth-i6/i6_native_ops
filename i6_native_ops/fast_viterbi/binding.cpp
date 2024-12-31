#include <vector>

#include <torch/extension.h>

#include "../common/returnn_definitions.h"

namespace py = pybind11;

std::vector<torch::Tensor> fast_viterbi_cuda(torch::Tensor& am_scores, torch::Tensor& edges,
                                             torch::Tensor& weights, torch::Tensor& start_end_states,
                                             torch::Tensor& seq_lens, unsigned n_states);

std::vector<torch::Tensor> fast_viterbi(torch::Tensor& am_scores, torch::Tensor& edges,
                                        torch::Tensor& weights, torch::Tensor& start_end_states,
                                        torch::Tensor& seq_lens, unsigned num_states) {
    CHECK_INPUT(am_scores, torch::kFloat32);
    CHECK_INPUT(edges, torch::kInt32);
    CHECK_INPUT(weights, torch::kFloat32);
    CHECK_INPUT(start_end_states, torch::kInt32);
    CHECK_INPUT(seq_lens, torch::kInt32);

    auto outputs = fast_viterbi_cuda(am_scores, edges, weights, start_end_states, seq_lens, num_states);

    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_viterbi", &fast_viterbi, "Fast Viterbi CUDA routine");
}
