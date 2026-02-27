import triton
import torch
import triton.language as tl
import torch.profiler as profiler


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE:tl.constexpr):
    #pid 获取
    pid = tl.program_id(axis=0)
    # start获取
    start = pid * BLOCK_SIZE
    #offset 每个block加载的数据位置
    offsets = start+tl.arange(0,BLOCK_SIZE)
# 计算mask
    mask = offsets < n

    x = tl.load(x_ptr+offsets,mask=mask)
    sigmod_x = 1/(1+tl.exp(-x))
    out = x*sigmod_x
    tl.store(out_ptr+offsets,out,mask=mask)



def silu(x:torch.tensor):
    output = torch.empty_like(x)
    n_element = x.numel()

    BLOCK_SIZE = 1024
    grid = lambda meta:(triton.cdiv(n_element, meta["BLOCK_SIZE"]),)
    silu_kernel[grid](x,output,n_element, BLOCK_SIZE=BLOCK_SIZE)
    return output

if __name__=="__main__":
    x = torch.randn(10000, device='cuda')

    output_triton = silu(x)
    output_torch = torch.nn.functional.silu(x)
    with profiler.profile( activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                           record_shapes=True, profile_memory=True, with_stack=True
                             ) as prof:
        for _ in range(10):
            output_cuda = silu_cuda.silu(x)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(torch.allclose(output_triton,output_torch,atol=1e-7))
    print(torch.allclose(output_cuda,output_torch,atol=1e-7))