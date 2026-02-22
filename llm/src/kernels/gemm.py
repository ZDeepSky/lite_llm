import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M,N,K,
                BLOCK_SIZE_M:tl.constexpr,
                BLOCK_SIZE_N:tl.constexpr,
                BLOCK_SIZE_K:tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    start_m = pid_m*BLOCK_SIZE_M
    start_n = pid_n*BLOCK_SIZE_N


    offset_m = start_m + tl.arange(0,BLOCK_SIZE_M)
    offset_n = start_n + tl.arange(0,BLOCK_SIZE_N)
    offset_k =  tl.arange(0,BLOCK_SIZE_K)

    a_temp_ptr = a_ptr + offset_m[:,None]*K+ offset_k[None, :]
    b_temp_ptr = b_ptr + offset_n[:,None]*K+ offset_k[None, :]
    # b 需要转置

    k = tl.cdiv(K, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(k):
        mask_a = (offset_m[:,None]<M) & (offset_k[None, :]<(K-i*BLOCK_SIZE_K))
        a = tl.load(a_temp_ptr, mask=mask_a,other=0.0)

        mask_b = (offset_n[:,None]<N) & (offset_k[None, :]<(K-i*BLOCK_SIZE_K))
        b = tl.load(b_temp_ptr, mask=mask_b, other=0.0)
        #这里转置
        b = tl.trans(b)

        acc+=tl.dot(a,b)

        a_temp_ptr+=BLOCK_SIZE_K
        b_temp_ptr+=BLOCK_SIZE_K

    c = acc.to(c_ptr.dtype.element_ty)


    c_ptrs = c_ptr + offset_m[:,None]*N + offset_n[None,:]
    mask_c = (offset_m[:,None]<M) &(offset_n[None,:]<N)

    tl.store(c_ptrs,c, mask=mask_c)

def gemm(a:torch.tensor, b:torch.tensor):
    assert a.is_contiguous() and b.is_contiguous()
    M = a.size(0)
    N = b.size(0)
    K = a.size(-1)
    c = torch.empty((M,N), device=a.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 16

    grid = lambda meta:(triton.cdiv(M,meta["BLOCK_SIZE_M"]),
        triton.cdiv(N,meta["BLOCK_SIZE_N"]),)


    gemm_kernel[grid](a,b,c,M,N,K, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K)

    return c


if __name__=="__main__":
    a_fp32 = torch.randn((128,32),dtype=torch.float32,device ="cuda")
    b_fp32 = torch.randn((128,32),dtype=torch.float32,device ="cuda")
    # a_fp32 = torch.ones((4,8),dtype=torch.float32,device ="cuda")
    # b_fp32 = torch.ones((4,8),dtype=torch.float32,device ="cuda")

    c_torch = a_fp32@b_fp32.T
    c_triton = gemm(a_fp32, b_fp32)
    print(f"{c_torch}")
    print(f"{c_triton}")
    diff = (c_torch - c_triton).abs()
    print("Max absolute difference:", diff.max().item())
    print("Mean absolute difference:", diff.mean().item())
    print(f"{torch.allclose(c_torch, c_triton, atol = 1e-1)}")