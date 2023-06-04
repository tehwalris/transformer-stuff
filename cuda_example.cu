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

void fill_rand_char4(int n, char4 *arr)
{
  union int_char4
  {
    int i;
    char4 c;
  };

  assert(n % 4 == 0);
  for (int i = 0; i < n; i += 4)
  {
    int_char4 temp;
    temp.i = rand();
    // HACK because rand() only generates non-negative numbers
    temp.i *= temp.i & 1 ? 1 : -1;
    arr[i / 4] = temp.c;
  }
}

float rand_float_neg_1_1()
{
  return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

void init_cpu(int n, char4 *A, float *A_scale, char4 *x, float *x_scale, float *y)
{
  fill_rand_char4(n * n, A);
  for (int i = 0; i < n; i++)
  {
    A_scale[i] = rand_float_neg_1_1();
  }
  fill_rand_char4(n, x);
  *x_scale = rand_float_neg_1_1();
  for (int i = 0; i < n; i++)
  {
    y[i] = 0.0f;
  }
}

__global__ void mul_gpu(int n, char4 const *__restrict__ A, char4 const *__restrict__ x, float *__restrict__ y)
{
  for (int i_row = blockIdx.y * blockDim.y + threadIdx.y; i_row < n; i_row += blockDim.y * gridDim.y)
  {
    int sum = 0;
    for (int i_col = threadIdx.x; i_col < n / 4; i_col += blockDim.x)
    {
      sum = __dp4a(A[(i_row * n) / 4 + i_col], x[i_col], sum);
    }
    atomicAdd(&y[i_row], float(sum));
  }
}

void mul_cpu(int n, char4 *A, char4 *x, float *y)
{
  for (int i = 0; i < n; i++)
  {
    int sum = 0;
    for (int j = 0; j < n; j += 4)
    {
      sum += int(A[(i * n + j) / 4].x) * int(x[j / 4].x);
      sum += int(A[(i * n + j) / 4].y) * int(x[j / 4].y);
      sum += int(A[(i * n + j) / 4].z) * int(x[j / 4].z);
      sum += int(A[(i * n + j) / 4].w) * int(x[j / 4].w);
    }
    y[i] = float(sum);
  }
}

__global__ void post_mul_scale_gpu(int n, float *y, float *A_scale, float x_scale)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    y[i] *= A_scale[i] * x_scale;
  }
}

void post_mul_scale_cpu(int n, float *y, float *A_scale, float x_scale)
{
  for (int i = 0; i < n; i++)
  {
    y[i] *= A_scale[i] * x_scale;
  }
}

int main(void)
{
  srand(0);

  int N = 4096;
  uint32_t num_iterations = 10;

  char4 *A;
  float *A_scale;
  char4 *x;
  float x_scale;
  float *y;
  assert(N % 4 == 0);
  CUDA_CHECK(cudaMallocManaged(&A, N * N / 4 * sizeof(char4)));
  CUDA_CHECK(cudaMallocManaged(&A_scale, N * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&x, N / 4 * sizeof(char4)));
  CUDA_CHECK(cudaMallocManaged(&y, N * sizeof(float)));

  init_cpu(N, A, A_scale, x, &x_scale, y);

  mul_cpu(N, A, x, y);
  post_mul_scale_cpu(N, y, A_scale, x_scale);

  // GPU version
  {
    for (int i = 0; i < N; i++)
    {
      y[i] = 0.0f;
    }

    assert(N % 4 == 0);
    dim3 block_size_mul(32, 8);
    dim3 grid_size_mul(1, (N + block_size_mul.y - 1) / block_size_mul.y);

    int block_size_scale(256);
    int grid_size_scale((N + block_size_scale - 1) / block_size_scale);

    mul_gpu<<<grid_size_mul, block_size_mul>>>(N, A, x, y);
    post_mul_scale_gpu<<<grid_size_scale, block_size_scale>>>(N, y, A_scale, x_scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    float sum_y = 0.0f;
    for (int i = 0; i < N; i++)
    {
      sum_y += y[i];
    }
    std::cout << "GPU result: y[0] = " << y[0] << ", sum(y) = " << sum_y << std::endl;

    CUDA_CHECK(cudaMemPrefetchAsync(A, N * N / 4 * sizeof(char4), 0));
    CUDA_CHECK(cudaMemPrefetchAsync(x, N / 4 * sizeof(char4), 0));
    CUDA_CHECK(cudaMemPrefetchAsync(y, N * sizeof(float), 0));

    printf("Starting GPU benchmark\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++)
    {
      mul_gpu<<<grid_size_mul, block_size_mul>>>(N, A, x, y);
      post_mul_scale_gpu<<<grid_size_scale, block_size_scale>>>(N, y, A_scale, x_scale);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("%f ms per iteration\n", elapsed.count() / num_iterations * 1e3f);
  }

  // CPU version
  {
    for (int i = 0; i < N; i++)
    {
      y[i] = 0.0f;
    }

    mul_cpu(N, A, x, y);
    post_mul_scale_cpu(N, y, A_scale, x_scale);

    CUDA_CHECK(cudaMemPrefetchAsync(A, N * N / 4 * sizeof(char4), cudaCpuDeviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(x, N / 4 * sizeof(char4), cudaCpuDeviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(y, N * sizeof(float), cudaCpuDeviceId));

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
      post_mul_scale_cpu(N, y, A_scale, x_scale);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("%f ms per iteration\n", elapsed.count() / num_iterations * 1e3f);
  }

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));

  return 0;
}