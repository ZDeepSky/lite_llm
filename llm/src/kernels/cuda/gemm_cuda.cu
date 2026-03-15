#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 16

__global__ void gemm_cuda_kernel(float *x, float *y, float* z,
                            int M, int N, int K)
{
    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_N][BLOCK_SIZE_K];

    // 当前处理的行和列
    int x_row = blockIdx.x*BLOCK_SIZE_M + threadIdx.x;
    int y_row = blockIdx.y*BLOCK_SIZE_N + threadIdx.y;

    //triton就是操作一个block cuda则是一个线程

    float acc = 0.0;

    // 加载m,k  n,k
    int max_k_block = (K+BLOCK_SIZE_K-1)/BLOCK_SIZE_K;
    for(int k_block = 0; k_block<max_k_block; k_block++)
    {
        //加载m,k, n,k
        for (int k = 0; k < BLOCK_SIZE_K; ++k)
        {
            int curKindex = k+k_block*BLOCK_SIZE_K;
            As[threadIdx.x][k] = ((x_row<M)&&(curKindex<K))? A[x_row*K+curKindex]:0.0;
            Bs[threadIdx.y][k] = ((y_row<N)&&(curKindex<K))? B[y_row*K+curKindex]:0.0;
        }
        // 这里代码没有优化 为啥，threadIdx.x，threadIdx.y 对应的线程 还是在重复的读取global，没有减少访存的读取
        // 不是用了shared mem 就能提升性能的
    }
    __syncthreads();
    for (int k = 0; k < BLOCK_SIZE_K; ++k)
    {
        acc+=As[threadIdx.x][k]*Bs[threadIdx.y][k];
    }

    __syncthreads();

    if (x_row < M && y_row < N)
        z[x_row*N + y_row] = acc;

}


torch::Tensor gemm_cuda(torch::Tensor x, torch:Tensor y)
{
    int M = x.size(0);
    int K = x.size(1);
    int N = y.size(0);

    torch::Tensor output =



    dim3 grid();
}
