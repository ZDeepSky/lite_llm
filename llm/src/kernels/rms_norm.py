
import triton
import torch
import triton.language as tl
import torch.nn.functional as F
from typing import Optional

@triton.jit
def rmsnorm_kernel(x_ptr, y_ptr,r_ptr,w_ptr,B,S,N,
            x_stride_b, x_stride_s, x_stride_n,
            y_stride_b, y_stride_s, y_stride_n,
            r_stride_b, r_stride_s, r_stride_n,
            w_stride_n,
            has_r,
            BLOCK_SIZE:tl.constexpr):

    pid = tl.program_id(axis=0)
 # 提取 start信息：B和S的信息
    batch = pid//S
    seq = pid%S

    start_x   = batch*x_stride_b+seq*x_stride_s
    col =  tl.arange(0, BLOCK_SIZE)
    offsets_x =  start_x +col*x_stride_n

    mask = col<N

    x = tl.load(x_ptr+offsets_x, mask=mask)
    if has_r:
        start_r = batch*r_stride_b+seq*r_stride_s
        offsets_r =  start_r + col*r_stride_n
        r = tl.load(r_ptr + offsets_r, mask=mask)
        x = x + r


    val = (tl.sum(x*x, axis=0)+1e-5)/N
    rms = 1.0/(tl.sqrt(val))
    w = tl.load(w_ptr+col*w_stride_n, mask=mask)
    y = x*rms*w

    start_y = batch*y_stride_b+seq*y_stride_s
    offsets_y = start_y + col*y_stride_n
    tl.store(y_ptr+offsets_y,y,  mask=mask)


def rms_norm(x:torch.tensor, r: Optional[torch.tensor], w: torch.tensor):
    # x为三维 B S N
    # RMS_NORM 必须每个block加载一整行数据
    B,S,N = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE=triton.next_power_of_2(N)
    grid = (B*S,)
    r_stride_b, r_stride_s, r_stride_n = 0,0,0
    has_r = False
    if r is not None:
        has_r = True
        r_stride_b, r_stride_s, r_stride_n = r.stride()

    rmsnorm_kernel[grid](x,y,r,w, B,S,N,
            *x.stride(),
            *y.stride(),
            r_stride_b,
            r_stride_s,
            r_stride_n,
            *w.stride(),
            has_r,
            BLOCK_SIZE
            )


    return y


def python_skip_rmsnorm(x, r, w, eps=1e-5):
    # x, r: (B, N)
    x = x + r
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x / torch.sqrt(var + eps)
    return (x_normed * w)


if __name__=="__main__":
    x = torch.randn((2,4,8), device="cuda", dtype=torch.float32)
    r = torch.randn((2,4,8), device="cuda", dtype=torch.float32)
    w = torch.ones((8), device="cuda", dtype=torch.float32)

    y_triton = rms_norm(x,r,w)
    y_torch = python_skip_rmsnorm(x,r,w)

    print(f"{y_torch}")
    print(f"{y_triton}")
    print(f"{torch.allclose(y_torch, y_triton, atol=1e-5)}")





