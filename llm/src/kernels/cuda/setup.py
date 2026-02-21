from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="silu_cuda_kernel", #对应pybind11的名字
    ext_modules=[
        CUDAExtension(
            name="silu_cuda",          # 模块名字，对应 import
            sources=["./bind.cpp",
                      "./silu_cuda.cu"], # 源文件路径
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
