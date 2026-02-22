import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))# 加入llm作为path目录

import torch
from src.kernels.gemm import gemm

def test_gemm():
    a_fp32 = torch.randn((128,32),dtype=torch.float32,device ="cuda")
    b_fp32 = torch.randn((128,32),dtype=torch.float32,device ="cuda")
    c_torch = a_fp32@b_fp32.T
    c_triton = gemm(a_fp32, b_fp32)
    print(f"maxabs={torch.abs(c_torch-c_triton).max()}")
    print(f"{torch.allclose(c_torch, c_triton, atol = 1e-5)}")
    print(f"maxabs={torch.abs(c_torch-c_triton).max()}")
    assert(torch.allclose(c_torch, c_triton, atol = 1e-1))
