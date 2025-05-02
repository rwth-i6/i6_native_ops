#include <vector>

#include <torch/extension.h>

#include <DebugOptions.h>

namespace py = pybind11;

std::vector<torch::Tensor> fbw2_cuda(torch::Tensor& num_states, torch::Tensor& num_edges, torch::Tensor& seq_lens,
                                     torch::Tensor& am_scores, torch::Tensor& edges, torch::Tensor& weights, torch::Tensor& start_end_states,
                                     DebugOptionsV2 debug_options);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_HOST(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fbw2(torch::Tensor& num_states, torch::Tensor& num_edges, torch::Tensor& seq_lens,
                                torch::Tensor& am_scores, torch::Tensor& edges, torch::Tensor& weights, torch::Tensor& start_end_states,
                                DebugOptionsV2 debug_options = DebugOptionsV2()) {
    CHECK_HOST(num_states);
    CHECK_HOST(num_edges);
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(am_scores);
    CHECK_INPUT(edges);
    CHECK_INPUT(weights);
    CHECK_INPUT(start_end_states);

    auto outputs = fbw2_cuda(num_states, num_edges, seq_lens,
                             am_scores, edges, weights, start_end_states,
                             debug_options);

    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fbw2", &fbw2, "Fast Baum-Welch CUDA routine version 2");

    py::class_<DebugOptionsV2>(m, "DebugOptionsV2")
            .def(py::init<>())
            .def_readwrite("dump_edges", &DebugOptionsV2::dump_edges)
            .def_readwrite("dump_alignment", &DebugOptionsV2::dump_alignment)
            .def_readwrite("dump_output", &DebugOptionsV2::dump_output)
            .def_readwrite("dump_every", &DebugOptionsV2::dump_every)
            .def_readwrite("pruning", &DebugOptionsV2::pruning)
            .def_readwrite("explicit_merge", &DebugOptionsV2::explicit_merge)
            .def_readwrite("per_frame_norm", &DebugOptionsV2::per_frame_norm);
}
