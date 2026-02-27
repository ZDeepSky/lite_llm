import triton
import triton.language as tl
import torch


@triton.jit
def transpose_kernel2(
    x_ptr,
    y_ptr,
    B,
    M,
    N,
    stride_B,
    stride_M,
    stride_N,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    # 每個 block 的起始位置
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    start_b = pid_b * BLOCK_SIZE_B

    offset_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offset_n = start_n + tl.arange(0, BLOCK_SIZE_N)

    for b in range(0, BLOCK_SIZE_B):
        offset_b = (start_b + b) * stride_B

        x_offset = offset_b + offset_m[:, None] * stride_M + offset_n[None, :]
        mask_x = (offset_m[:, None] < M) & (offset_n[None, :] < N)

        x = tl.load(x_ptr + x_offset, mask=mask_x, other=1.0)

        y = tl.trans(x)

        y_offset = offset_b + offset_n[:, None] * M + offset_m[None, :]
        mask_y = (offset_n[:, None] < N) & (offset_m[None, :] < M)

        tl.store(y_ptr + y_offset, y, mask=mask_y)


def transpose(x: torch.Tensor):
    origin_shape = x.shape
    # 除最后两个维度外，将其他维度压缩到一个维度
    x = x.view(-1, origin_shape[-2], origin_shape[-1])
    B, M, N = x.shape

    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 4
    BLOCK_SIZE_B = 4

    y = torch.empty((B, N, M), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        triton.cdiv(B, BLOCK_SIZE_B),
    )

    transpose_kernel2[grid](
        x,
        y,
        B,
        M,
        N,
        *x.stride(),
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    new_shape = (*origin_shape[:-2], origin_shape[-1], origin_shape[-2])
    # print(f"new_shape: {new_shape}")

    return y.view(new_shape)


if __name__ == "__main__":
    x = torch.randn((2, 2, 14, 20), device="cuda", dtype=torch.float32)
    y_triton = transpose(x)
    y_torch = x.transpose(-1, -2)

    # print(f"{x}")
    # print(f"{y_triton}")
    # print(f"{y_torch}")
    print(f"{torch.allclose(y_triton, y_torch, atol=1e-7)}")