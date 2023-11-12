#include <vector>

#include <torch/extension.h>

#include <DebugOptions.h>

namespace py = pybind11;

std::vector<torch::Tensor> fbw_cuda(torch::Tensor& am_scores, torch::Tensor& edges,
                                    torch::Tensor& weights, torch::Tensor& start_end_states,
                                    torch::Tensor& seq_lens, unsigned n_states,
                                    DebugOptions debug_options);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fbw(torch::Tensor& am_scores, torch::Tensor& edges,
                               torch::Tensor& weights, torch::Tensor& start_end_states,
                               torch::Tensor& seq_lens, unsigned num_states,
                               DebugOptions debug_options = DebugOptions()) {
    CHECK_INPUT(am_scores);
    CHECK_INPUT(edges);
    CHECK_INPUT(weights);
    CHECK_INPUT(start_end_states);
    CHECK_INPUT(seq_lens);

    auto outputs = fbw_cuda(am_scores, edges, weights, start_end_states, seq_lens, num_states,
                            debug_options);

    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fbw", &fbw, "Fast Baum-Welch CUDA routine");

    py::class_<DebugOptions>(m, "DebugOptions")
            .def(py::init<>())
            .def_readwrite("dump_alignment", &DebugOptions::dump_alignment)
            .def_readwrite("dump_output", &DebugOptions::dump_output)
            .def_readwrite("dump_every", &DebugOptions::dump_every)
            .def_readwrite("pruning", &DebugOptions::pruning);
}
