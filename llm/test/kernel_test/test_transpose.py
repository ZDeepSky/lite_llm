
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output")))
import self_kernels
import torch
from src.kernels.transpose import transpose




def test_transpose():
    x = torch.randn((16,64,64), device="cuda")
    y_torch = x.transpose(-1,-2)
    y_triton = transpose(x)
    y_cuda =  self_kernels.transpose_cuda(x)

    assert(torch.allclose(y_torch, y_triton, atol=1e-7))
    assert(torch.allclose(y_torch, y_cuda, atol=1e-7))