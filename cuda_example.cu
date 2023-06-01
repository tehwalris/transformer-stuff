#include <iostream>
#include <math.h>
#include <stdio.h>
#include <chrono>

// function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
  {
    y[i] = x[i] + y[i];
  }
}

int main(void)
{
  int N = 1 << 20; // 1M elements

  float *x, *y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  add<<<1, 256>>>(N, x, y);
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  printf("Starting benchmark\n");
  uint32_t num_iterations = 30;
  auto start = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < num_iterations; i++)
  {
    add<<<1, 1>>>(N, x, y);
    cudaDeviceSynchronize();
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  printf("%f ms per iteration\n", elapsed.count() / num_iterations * 1e3f);

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}