import torch
import triton
import triton.language as tl


#分块矩阵乘法
@triton.jit
def gemm_kernel(A_ptr, B_ptr, C_ptr,M,N,K, BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N:tl.constexpr, BLOCK_SIZE_K:tl.constexpr):
    pid_0 = tl.program_id(axis=0)
    pid_1 = tl.program_id(axis=1)
    #row start and col start

    start_m = pid_0 * BLOCK_SIZE_M
    start_n = pid_1 * BLOCK_SIZE_N

    offsets_m = start_m+tl.arange(0,BLOCK_SIZE_M)# (BLOCK_SIZE_M, 1)
    offsets_n = start_n+tl.arange(0,BLOCK_SIZE_N) # (1, BLOCK_SIZE)

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),dtype=tl.float32)
    #分块矩阵乘法
    for k in range(0, K, BLOCK_SIZE_K):
        offset_k = k + tl.arange(0,BLOCK_SIZE_K)
        mask_mk =(offset_k[None,:]<K)&(offsets_m[:,None]<M)
        #offsets_m[:,None]*K 为M,K维度矩阵下每行起始位置的实际偏移
        a_mk = tl.load(A_ptr+offsets_m[:,None]*K+ offset_k[None,:],mask=mask_mk)

        mask_nk = (offset_k[:,None]<K)&(offsets_n[None,:]<N)

        #  offset_k[:,None]*N 就是K,N维度矩阵下每行起始位置的实际偏移
        b_nk = tl.load(B_ptr + offsets_n[None,:]+ offset_k[:,None]*N,mask = mask_nk)

        c=tl.dot(a_mk,b_nk,acc=c)

    # c的块是BLOCK_SIZE_M，BLOCK_SIZE_N，
    # 加载 offsets_m行 offsets_n列 行起始位置是offsets_m*N
    mask_c = (offsets_m[:,None]<M) & (offsets_n[None,:]<N)

    tl.device_print("c",c)
    tl.device_print("offsets_m",offsets_m[:,None]*N+offsets_n[None,:])
    tl.store(C_ptr+offsets_m[:,None]*N+offsets_n[None,:], c, mask=mask_c)




#A(batch, m,k) B(k,n) -> C(batch, m,n)
def gemm(A: torch.tensor, B: torch.tensor):
    output_shape = (A.shape[:-1])
    A = A.view(-1, A.shape[-1])
    print(f"{A.shape}")
    m,k = A.shape
    n = B.shape[1]
    print(f"output_shape:{output_shape}")
    C = torch.empty((m,n), device=A.device)

    BLOCK_SIZE_M=16
    BLOCK_SIZE_N=16
    BLOCK_SIZE_K=16

    grid = lambda meta:(triton.cdiv(m, meta['BLOCK_SIZE_M']), triton.cdiv(n, meta['BLOCK_SIZE_N']),)
    gemm_kernel[grid](A,B,C,m,n,k,BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K)

    return C.view((*output_shape,n))


if __name__=="__main__":
    A=torch.randn((2,128,128), device='cuda')
    B=torch.randn((128,64), device='cuda')
    # A=torch.ones((1,2,16), device='cuda', dtype=torch.float32)  # 简化为1×2×3
    # B=torch.ones((16,4), device='cuda', dtype=torch.float32)   # 3×4

    C_triton = gemm(A,B)
    C_torch = A@B
    print(f"{C_torch}")
    print(f"{C_triton}")
    print(f"{C_torch.shape} {C_triton.shape}")
    print(torch.allclose(C_triton,C_torch,atol=1e-))
    diff = (C_triton - C_torch).abs()
    print("Max absolute difference:", diff.max().item())
    print("Mean absolute difference:", diff.mean().item())