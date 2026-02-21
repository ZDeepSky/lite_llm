#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


#define THREAD_PRE_BLOCK 256
__global__ void silu_cuda_kernel(float *x, float *y, int num)
{
    __shared__ float shardMem[THREAD_PRE_BLOCK];

    int thread_idx = threadIdx.x+ blockIdx.x*blockDim.x;// 这里使用共享内存的优化有限

    if(thread_idx<num)
    {
        shardMem[threadIdx.x] = x[thread_idx];
    }
    __syncthreads();


    shardMem[threadIdx.x] = shardMem[threadIdx.x]/(1.0f+__expf(-shardMem[threadIdx.x]));
    y[thread_idx] = shardMem[threadIdx.x];

}




torch::Tensor silu_cuda(torch::Tensor x)
{
    torch::Tensor y = torch::empty_like(x);
    int num = x.numel();
    dim3 grid((num+THREAD_PRE_BLOCK-1)/THREAD_PRE_BLOCK,1);
    dim3 block(THREAD_PRE_BLOCK,1);
    silu_cuda_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(),num);
    return y;
}



