from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT_PATH = ".."
KERNEL_CUDA_PATH = ROOT_PATH+"/src/kernels/cuda"


setup(
    name="cuda_kernel", #对应pybind11的名字
    ext_modules=[
        CUDAExtension(
            name="self_kernels",          # 模块名字，对应 import
            sources=[
                        KERNEL_CUDA_PATH+"/bind.cpp",
                        KERNEL_CUDA_PATH+"/silu.cu",
                        KERNEL_CUDA_PATH+"/transpose.cu",
                    ], # 源文件路径
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
