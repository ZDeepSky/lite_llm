
#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor silu_cuda(torch::Tensor x);
torch::Tensor transpose_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("silu_cuda", &silu_cuda, "SiLU activation (CUDA)");
    module.def("transpose_cuda", &transpose_cuda, "transpose activation (CUDA)");
}
