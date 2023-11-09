#include <vector>

#include <torch/extension.h>

namespace py = pybind11;

std::vector<torch::Tensor> fbw_cuda(std::vector<torch::Tensor> torch_inputs);

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fbw(torch::Tensor &am_scores, torch::Tensor &edges,
                               torch::Tensor &weights,
                               torch::Tensor &start_end_states,
                               torch::Tensor &index,
                               torch::Tensor &state_buffer) {
    CHECK_INPUT(am_scores);
    CHECK_INPUT(edges);
    CHECK_INPUT(weights);
    CHECK_INPUT(start_end_states);
    CHECK_INPUT(index);
    CHECK_INPUT(state_buffer);

    auto outputs = fbw_cuda(
        {am_scores, edges, weights, start_end_states, index, state_buffer});

    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fbw", &fbw, "Fast Baum-Welch CUDA routine");
}
