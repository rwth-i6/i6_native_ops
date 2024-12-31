#ifdef RNNT_ENABLE_GPU

#include <ATen/cuda/CUDAContext.h>

#include "gpu_rnnt.h"
#include "gpu_workspace_manager.h"

#endif

#include <torch/extension.h>

#include "cpu_rnnt.h"
#include "cpu_workspace_manager.h"
#include "options.h"

int cpu_monotonic_rnnt(torch::Tensor& acts, torch::Tensor& labels, torch::Tensor& input_lengths,
                       torch::Tensor& label_lengths, torch::Tensor& costs, torch::Tensor& grads, int blank_label,
                       int num_threads) {
    TORCH_CHECK(acts.type().scalarType() == torch::ScalarType::Float);
    TORCH_CHECK(labels.type().scalarType() == torch::ScalarType::Int ||
                labels.type().scalarType() == torch::ScalarType::Long);
    TORCH_CHECK(input_lengths.type().scalarType() == torch::ScalarType::Int ||
                input_lengths.type().scalarType() == torch::ScalarType::Long);
    TORCH_CHECK(label_lengths.type().scalarType() == torch::ScalarType::Int ||
                label_lengths.type().scalarType() == torch::ScalarType::Long);

    int B = labels.size(0);
    int V = acts.size(1);

    RNNTOptions options;
    options.loc = RNNT_CPU;
    options.blank_label = blank_label;
    options.num_threads = num_threads;

    auto labels_int = labels.to(torch::kInt32);
    auto input_lengths_int = input_lengths.to(torch::kInt32);
    auto label_lengths_int = label_lengths.to(torch::kInt32);

    CpuRNNTWorkspaceManager<float> workspace_manager(acts.data_ptr<float>(), labels_int.data_ptr<int>(), B,
                                                     input_lengths_int.data_ptr<int>(),
                                                     label_lengths_int.data_ptr<int>(), V);
    auto rnnt_status = workspace_manager.create_workspace();

    TORCH_CHECK(rnnt_status == RNNT_STATUS_SUCCESS, "cpu_rnnt error in create_workspace");

    CpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.num_threads);

    rnnt_status = rnnt_computer.cost_and_grad(costs.data_ptr<float>(), grads.data_ptr<float>());
    TORCH_CHECK(rnnt_status == RNNT_STATUS_SUCCESS, "cpu_rnnt error in rnnt_computer");

    workspace_manager.free_workspace();

    return rnnt_status;
}

int cpu_monotonic_rnnt_align_restrict(torch::Tensor& acts, torch::Tensor& labels, torch::Tensor& input_lengths,
                                      torch::Tensor& label_lengths, torch::Tensor& alignment,
                                      int max_shift_from_alignment, torch::Tensor& costs, torch::Tensor& grads,
                                      int blank_label, int num_threads) {
    TORCH_CHECK(acts.type().scalarType() == torch::ScalarType::Float);
    TORCH_CHECK(labels.type().scalarType() == torch::ScalarType::Int ||
                labels.type().scalarType() == torch::ScalarType::Long);
    TORCH_CHECK(input_lengths.type().scalarType() == torch::ScalarType::Int ||
                input_lengths.type().scalarType() == torch::ScalarType::Long);
    TORCH_CHECK(label_lengths.type().scalarType() == torch::ScalarType::Int ||
                label_lengths.type().scalarType() == torch::ScalarType::Long);

    int B = labels.size(0);
    int V = acts.size(1);

    RNNTOptions options;
    options.loc = RNNT_CPU;
    options.blank_label = blank_label;
    options.num_threads = num_threads;

    auto labels_int = labels.to(torch::kInt32);
    auto input_lengths_int = input_lengths.to(torch::kInt32);
    auto label_lengths_int = label_lengths.to(torch::kInt32);

    CpuRNNTWorkspaceManager<float> workspace_manager(acts.data_ptr<float>(), labels_int.data_ptr<int>(), B,
                                                     input_lengths_int.data_ptr<int>(),
                                                     label_lengths_int.data_ptr<int>(), V);
    auto rnnt_status = workspace_manager.create_workspace();

    workspace_manager.restrict_to_alignment(alignment.data_ptr<int>(), max_shift_from_alignment, blank_label);

    TORCH_CHECK(rnnt_status == RNNT_STATUS_SUCCESS, "cpu_rnnt error in create_workspace");

    CpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.num_threads);

    rnnt_status = rnnt_computer.cost_and_grad(costs.data_ptr<float>(), grads.data_ptr<float>());
    TORCH_CHECK(rnnt_status == RNNT_STATUS_SUCCESS, "cpu_rnnt error in rnnt_computer");

    workspace_manager.free_workspace();

    return rnnt_status;
}

#ifdef RNNT_ENABLE_GPU

int gpu_monotonic_rnnt(torch::Tensor& acts, torch::Tensor& labels, torch::Tensor& input_lengths,
                       torch::Tensor& label_lengths, torch::Tensor& costs, torch::Tensor& grads, int blank_label,
                       int num_threads) {
    TORCH_CHECK(acts.type().scalarType() == torch::ScalarType::Float);
    TORCH_CHECK(acts.type().is_cuda(), "acts must be a CUDA tensor");
    TORCH_CHECK(labels.type().scalarType() == torch::ScalarType::Int ||
                labels.type().scalarType() == torch::ScalarType::Long);
    TORCH_CHECK(labels.type().is_cuda(), "labels must be a CUDA tensor");
    TORCH_CHECK(input_lengths.type().scalarType() == torch::ScalarType::Int ||
                input_lengths.type().scalarType() == torch::ScalarType::Long);
    TORCH_CHECK(input_lengths.type().is_cuda(), "input_lengths must be a CUDA tensor");
    TORCH_CHECK(label_lengths.type().scalarType() == torch::ScalarType::Int ||
                label_lengths.type().scalarType() == torch::ScalarType::Long);
    TORCH_CHECK(label_lengths.type().is_cuda(), "label_lengths must be a CUDA tensor");

    int B = labels.size(0);
    int V = acts.size(1);

    RNNTOptions options;
    options.loc = RNNT_GPU;
    options.blank_label = blank_label;
    options.stream = at::cuda::getCurrentCUDAStream();
    options.num_threads = num_threads;

    auto labels_int = labels.to(torch::kInt32);
    auto input_lengths_int = input_lengths.to(torch::kInt32);
    auto label_lengths_int = label_lengths.to(torch::kInt32);

    GpuRNNTWorkspaceManager<float> workspace_manager(acts.data_ptr<float>(), labels_int.data_ptr<int>(), B,
                                                     input_lengths_int.data_ptr<int>(),
                                                     label_lengths_int.data_ptr<int>(), V);

    auto rnnt_status = workspace_manager.create_workspace();

    TORCH_CHECK(rnnt_status == RNNT_STATUS_SUCCESS, "gpu_rnnt error in create_workspace");

    GpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.stream);
    rnnt_status = rnnt_computer.cost_and_grad(costs.data_ptr<float>(), grads.data_ptr<float>());
    TORCH_CHECK(rnnt_status == RNNT_STATUS_SUCCESS, "gpu_rnnt error in rnnt_computer");

    workspace_manager.free_workspace();

    return rnnt_status;
}

int gpu_monotonic_rnnt_align_restrict(torch::Tensor& acts, torch::Tensor& labels, torch::Tensor& input_lengths,
                                      torch::Tensor& label_lengths, torch::Tensor& alignment,
                                      int max_shift_from_alignment, torch::Tensor& costs, torch::Tensor& grads,
                                      int blank_label, int num_threads) {
    TORCH_CHECK(acts.type().scalarType() == torch::ScalarType::Float);
    TORCH_CHECK(acts.type().is_cuda(), "acts must be a CUDA tensor");
    TORCH_CHECK(labels.type().scalarType() == torch::ScalarType::Int ||
                labels.type().scalarType() == torch::ScalarType::Long);
    TORCH_CHECK(labels.type().is_cuda(), "labels must be a CUDA tensor");
    TORCH_CHECK(input_lengths.type().scalarType() == torch::ScalarType::Int ||
                input_lengths.type().scalarType() == torch::ScalarType::Long);
    TORCH_CHECK(input_lengths.type().is_cuda(), "input_lengths must be a CUDA tensor");
    TORCH_CHECK(label_lengths.type().scalarType() == torch::ScalarType::Int ||
                label_lengths.type().scalarType() == torch::ScalarType::Long);
    TORCH_CHECK(label_lengths.type().is_cuda(), "label_lengths must be a CUDA tensor");

    int B = labels.size(0);
    int V = acts.size(1);

    RNNTOptions options;
    options.loc = RNNT_GPU;
    options.blank_label = blank_label;
    options.stream = at::cuda::getCurrentCUDAStream();
    options.num_threads = num_threads;

    auto labels_int = labels.to(torch::kInt32);
    auto input_lengths_int = input_lengths.to(torch::kInt32);
    auto label_lengths_int = label_lengths.to(torch::kInt32);

    GpuRNNTWorkspaceManager<float> workspace_manager(acts.data_ptr<float>(), labels_int.data_ptr<int>(), B,
                                                     input_lengths_int.data_ptr<int>(),
                                                     label_lengths_int.data_ptr<int>(), V);

    auto rnnt_status = workspace_manager.create_workspace();

    workspace_manager.restrict_to_alignment(alignment.data_ptr<int>(), max_shift_from_alignment, blank_label);

    TORCH_CHECK(rnnt_status == RNNT_STATUS_SUCCESS, "gpu_rnnt error in create_workspace");

    GpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.stream);
    rnnt_status = rnnt_computer.cost_and_grad(costs.data_ptr<float>(), grads.data_ptr<float>());
    TORCH_CHECK(rnnt_status == RNNT_STATUS_SUCCESS, "gpu_rnnt error in rnnt_computer");

    workspace_manager.free_workspace();

    return rnnt_status;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_monotonic_rnnt", &cpu_monotonic_rnnt, "Monotonic RNNT CPU version");
    m.def("cpu_monotonic_rnnt_align_restrict", &cpu_monotonic_rnnt_align_restrict,
          "Alignment-restricted monotonic RNNT CPU version");
#ifdef RNNT_ENABLE_GPU
    m.def("gpu_monotonic_rnnt", &gpu_monotonic_rnnt, "Monotonic RNNT GPU version");
    m.def("gpu_monotonic_rnnt_align_restrict", &gpu_monotonic_rnnt_align_restrict,
          "Alignment-restricted monotonic RNNT GPU version");
#endif
}
