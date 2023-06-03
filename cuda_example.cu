#include <iostream>
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <cassert>

#define CUDA_CHECK(err)                                                         \
  do                                                                            \
  {                                                                             \
    cudaError_t err_ = (err);                                                   \
    if (err_ != cudaSuccess)                                                    \
    {                                                                           \
      fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, \
              cudaGetErrorString(err_));                                        \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

__global__ void
init_cuda(int n, float *A, float *x, float *y)
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
  __shared__ float x_shared[4096];
  for (int i = threadIdx.x; i < n; i += blockDim.x)
  {
    x_shared[i] = x[i];
  }

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float sum = 0.0f;
    for (int j = 0; j < n; j++)
    {
      sum += A[i * n + j] * x_shared[j];
    }
    y[i] = sum;
  }
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
  int N = 4096;
  uint32_t num_iterations = 10;

  // GPU version
  {
    float *A, *x, *y;
    CUDA_CHECK(cudaMallocManaged(&A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&x, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&y, N * sizeof(float)));

    assert(N % 2 == 0);
    int block_size = 32;
    int grid_size = (N + block_size - 1) / block_size;

    init_cuda<<<grid_size, block_size>>>(N, A, x, y);
    mul_gpu<<<grid_size, block_size>>>(N, A, x, y);
    CUDA_CHECK(cudaDeviceSynchronize());

    float sum_y = 0.0f;
    for (int i = 0; i < N; i++)
    {
      sum_y += y[i];
    }
    std::cout << "GPU result: y[0] = " << y[0] << ", sum(y) = " << sum_y << std::endl;

    CUDA_CHECK(cudaMemPrefetchAsync(A, N * N * sizeof(float), 0));
    CUDA_CHECK(cudaMemPrefetchAsync(x, N * sizeof(float), 0));
    CUDA_CHECK(cudaMemPrefetchAsync(y, N * sizeof(float), 0));

    printf("Starting GPU benchmark\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++)
    {
      mul_gpu<<<grid_size, block_size>>>(N, A, x, y);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("%f ms per iteration\n", elapsed.count() / num_iterations * 1e3f);

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
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