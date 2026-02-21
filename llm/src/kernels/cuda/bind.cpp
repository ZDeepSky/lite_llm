
#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor silu_cuda(torch::Tensor x);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("silu", &silu_cuda, "SiLU activation (CUDA)");
}
