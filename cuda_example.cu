#include <iostream>
#include <math.h>
#include <stdio.h>
#include <chrono>

__global__ void init_cuda(int n, float *A, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {
    for (int j = 0; j < n; j++)
    {
      A[i * n + j] = float(i % 5) - float(j % 7) * 0.3f;
    }
    x[i] = float(i % 3 - 1.2f);
    y[i] = 0.0f;
  }
}

void init_cpu(int n, float *A, float *x, float *y)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      A[i * n + j] = float(i % 5) - float(j % 7) * 0.3f;
    }
    x[i] = float(i % 3 - 1.2f);
    y[i] = 0.0f;
  }
}

__global__ void mul_gpu(int n, float const *__restrict__ A, float const *__restrict__ x, float *__restrict__ y)
{
}

void mul_cpu(int n, float *A, float *x, float *y)
{
  for (int i = 0; i < n; i++)
  {
    float sum = 0.0f;
    for (int j = 0; j < n; j++)
    {
      sum += A[i * n + j] * x[j];
    }
    y[i] = sum;
  }
}

int main(void)
{
  int N = 256;
  uint32_t num_iterations = 10;

  // GPU version
  {
    float *A, *x, *y;
    cudaMallocManaged(&A, N * N * sizeof(float));
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    int block_size = 192;
    int num_blocks = (N + block_size - 1) / block_size;

    init_cuda<<<num_blocks, block_size>>>(N, A, x, y);
    mul_gpu<<<num_blocks, block_size>>>(N, A, x, y);
    cudaDeviceSynchronize();

    float sum_y = 0.0f;
    for (int i = 0; i < N; i++)
    {
      sum_y += y[i];
    }
    std::cout << "GPU result: y[0] = " << y[0] << ", sum(y) = " << sum_y << std::endl;

    cudaMemPrefetchAsync(A, N * N * sizeof(float), 0);
    cudaMemPrefetchAsync(x, N * sizeof(float), 0);
    cudaMemPrefetchAsync(y, N * sizeof(float), 0);

    printf("Starting GPU benchmark\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++)
    {
      mul_gpu<<<num_blocks, block_size>>>(N, A, x, y);
      cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("%f ms per iteration\n", elapsed.count() / num_iterations * 1e3f);

    cudaFree(A);
    cudaFree(x);
    cudaFree(y);
  }

  // CPU version
  {
    float *A = (float *)malloc(N * N * sizeof(float));
    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));

    init_cpu(N, A, x, y);

    mul_cpu(N, A, x, y);

    float sum_y = 0.0f;
    for (int i = 0; i < N; i++)
    {
      sum_y += y[i];
    }
    std::cout << "CPU result: y[0] = " << y[0] << ", sum(y) = " << sum_y << std::endl;

    printf("Starting CPU benchmark\n");

    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++)
    {
      mul_cpu(N, A, x, y);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("%f ms per iteration\n", elapsed.count() / num_iterations * 1e3f);

    free(A);
    free(x);
    free(y);
  }

  return 0;
}