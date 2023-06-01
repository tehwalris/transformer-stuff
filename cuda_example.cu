#include <iostream>
#include <math.h>
#include <stdio.h>
#include <chrono>

__global__ void init_cuda(int n, float *x, float *y, float *z)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
    z[i] = 0.0f;
  }
}

void init_cpu(int n, float *x, float *y, float *z)
{
  for (int i = 0; i < n; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
    z[i] = 0.0f;
  }
}

// function to add the elements of two arrays
__global__ void add_gpu(int n, float const *__restrict__ x, float const *__restrict__ y, float *__restrict__ z)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    z[i] = x[i] + y[i];
  }
}

void add_cpu(int n, float *x, float *y, float *z)
{
  for (int i = 0; i < n; i++)
  {
    z[i] = x[i] + y[i];
  }
}

int main(void)
{
  int N = 1 << 26;
  uint32_t num_iterations = 10;

  // GPU version
  {
    float *x, *y, *z;
    cudaMalloc(&x, N * sizeof(float));
    cudaMalloc(&y, N * sizeof(float));
    cudaMalloc(&z, N * sizeof(float));

    int block_size = 192;
    int num_blocks = (N + block_size - 1) / block_size;

    init_cuda<<<num_blocks, block_size>>>(N, x, y, z);

    add_gpu<<<num_blocks, block_size>>>(N, x, y, z);
    cudaDeviceSynchronize();

    // // Check for errors (all values should be 3.0f)
    // float maxError = 0.0f;
    // for (int i = 0; i < N; i++)
    // {
    //   maxError = fmax(maxError, fabs(z[i] - 3.0f));
    // }
    // std::cout << "Max error: " << maxError << std::endl;

    printf("Starting GPU benchmark\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++)
    {
      add_gpu<<<num_blocks, block_size>>>(N, x, y, z);
      cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("%f ms per iteration\n", elapsed.count() / num_iterations * 1e3f);

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
  }

  {
    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));
    float *z = (float *)malloc(N * sizeof(float));

    init_cpu(N, x, y, z);

    // CPU version
    {
      add_cpu(N, x, y, z);

      float maxError = 0.0f;
      for (int i = 0; i < N; i++)
      {
        maxError = fmax(maxError, fabs(z[i] - 3.0f));
      }
      std::cout << "Max error: " << maxError << std::endl;

      printf("Starting CPU benchmark\n");

      auto start = std::chrono::high_resolution_clock::now();
      for (uint32_t i = 0; i < num_iterations; i++)
      {
        add_cpu(N, x, y, z);
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      printf("%f ms per iteration\n", elapsed.count() / num_iterations * 1e3f);

      free(x);
      free(y);
      free(z);
    }
  }

  return 0;
}