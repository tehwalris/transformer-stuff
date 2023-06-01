#include <iostream>
#include <math.h>
#include <stdio.h>
#include <chrono>

// function to add the elements of two arrays
__global__ void add_gpu(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {
    y[i] = x[i] + y[i];
  }
}

void add_cpu(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
  {
    y[i] = x[i] + y[i];
  }
}

int main(void)
{
  int N = int(1e8);
  uint32_t num_iterations = 10;

  float *x, *y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // GPU version
  {
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;

    add_gpu<<<num_blocks, block_size>>>(N, x, y);
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
      maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    printf("Starting GPU benchmark\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++)
    {
      add_gpu<<<num_blocks, block_size>>>(N, x, y);
      cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("%f ms per iteration\n", elapsed.count() / num_iterations * 1e3f);
  }

  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // CPU version
  {
    add_cpu(N, x, y);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
      maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    printf("Starting CPU benchmark\n");

    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++)
    {
      add_cpu(N, x, y);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("%f ms per iteration\n", elapsed.count() / num_iterations * 1e3f);
  }

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}