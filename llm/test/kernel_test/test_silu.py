

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import torch
from src.kernels.silu import silu

def test_silu():
    x = torch.randn(10000, device='cuda')
    output_triton = silu(x)
    output_torch = torch.nn.functional.silu(x)
    assert torch.allclose(output_triton, output_torch, atol=1e-7)

