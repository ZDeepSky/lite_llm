

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output")))
import self_kernels
import torch
from src.kernels.silu import silu

def test_silu():
    x = torch.randn(10000, device='cuda')
    output_triton = silu(x)
    output_torch = torch.nn.functional.silu(x)
    output_cuda = self_kernels.silu_cuda(x)
    assert torch.allclose(output_triton, output_torch, atol=1e-7)
    assert torch.allclose(output_cuda, output_torch, atol=1e-7)

