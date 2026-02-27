#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_B 4  // 每block batch 數

__global__ void transpose_kernel(
    const float* x_ptr,
    float* y_ptr,
    int B,
    int M,
    int N,
    int stride_B,
    int stride_M,
    int stride_N)
{
    // block 內
    int m_local = threadIdx.x;
    int n_local = threadIdx.y;
    int b_local = threadIdx.z;

    // block
    int m_block = blockIdx.x * TILE_M;
    int n_block = blockIdx.y * TILE_N;
    int b_block = blockIdx.z * TILE_B;

    // 全局
    int m = m_local + m_block;
    int n = n_local + n_block;
    int b = b_local + b_block;

    if (b < B && m < M && n < N)
    {
        int x_index = b * stride_B + m * stride_M + n * stride_N;
        // (b, n, m) → (b * N * M + n * M + m)
        int y_index = b * (N * M) + n * M + m;

        y_ptr[y_index] = x_ptr[x_index];
    }
}

torch::Tensor transpose_cuda(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda(),    "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only support float32");

    int B = x.size(0);
    int M = x.size(1);
    int N = x.size(2);

    // 输出 (B, N, M)
    torch::Tensor y = torch::empty({B, N, M}, x.options());

    int stride_B = x.stride(0);
    int stride_M = x.stride(1);
    int stride_N = x.stride(2);

    // block 大小：TILE_M × TILE_N × TILE_B 的元素
    dim3 blockDim(TILE_M, TILE_N, TILE_B);

    // grid 大小：向上取整
    dim3 gridDim(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N,
        (B + TILE_B - 1) / TILE_B
    );

    const float* x_ptr = x.data_ptr<float>();
    float*       y_ptr = y.data_ptr<float>();

    transpose_kernel<<<gridDim, blockDim>>>(
        x_ptr, y_ptr, B, M, N,
        stride_B, stride_M, stride_N
    );

    return y;
}