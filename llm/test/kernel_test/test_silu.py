
import torch
from llm.code.kernels.silu import silu

def test_silu():
    x = torch.randn(10000, device='cuda')
    output_triton = silu(x)
    output_torch = torch.nn.functional.silu(x)
    assert torch.allclose(output_triton, output_torch, atol=1e-7)
