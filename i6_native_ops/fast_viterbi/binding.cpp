#include <vector>

#include <torch/extension.h>

namespace py = pybind11;

std::vector<torch::Tensor> fast_viterbi_cuda(torch::Tensor& am_scores, torch::Tensor& edges,
                                             torch::Tensor& weights, torch::Tensor& start_end_states,
                                             torch::Tensor& seq_lens, unsigned n_states);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fast_viterbi(torch::Tensor& am_scores, torch::Tensor& edges,
                                        torch::Tensor& weights, torch::Tensor& start_end_states,
                                        torch::Tensor& seq_lens, unsigned num_states) {
    CHECK_INPUT(am_scores);
    CHECK_INPUT(edges);
    CHECK_INPUT(weights);
    CHECK_INPUT(start_end_states);
    CHECK_INPUT(seq_lens);

    auto outputs = fast_viterbi_cuda(am_scores, edges, weights, start_end_states, seq_lens, num_states);

    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_viterbi", &fast_viterbi, "Fast Viterbi CUDA routine");
}
