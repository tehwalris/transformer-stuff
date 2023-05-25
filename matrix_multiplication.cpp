#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <immintrin.h>
#include <fmaintrin.h>
#include <chrono>

const uint32_t n_hidden = 4096, n_context = 2048, n_layers = 32, n_heads = 32;
const uint32_t cache_line_bytes = 64;

float vector_dot_product_baseline(uint32_t n, float *va, float *vb)
{
  float sum = 0.0;
  for (uint32_t i = 0; i < n; i++)
  {
    sum += va[i] * vb[i];
  }
  return sum;
}

float _mm256_add_elements_ps(__m256 v)
{
  float sum = 0.0;
  for (uint32_t i = 0; i < 8; i++)
  {
    sum += ((float *)&v)[i];
  }
  return sum;
}

float vector_dot_product_fast_n_hidden(float *va, float *vb)
{
  assert(n_hidden % 8 == 0);

  __m256 sum = _mm256_setzero_ps();
  for (uint32_t i = 0; i < n_hidden; i += 8)
  {
    __m256 a = _mm256_load_ps(va + i);
    __m256 b = _mm256_load_ps(vb + i);
    sum = _mm256_fmadd_ps(a, b, sum);
  }
  return _mm256_add_elements_ps(sum);
}

void softmax(uint32_t n, float *v)
{
  // TODO check implementation (Copilot generated it)

  float max = *std::max_element(v, v + n);
  float sum = 0.0;
  for (uint32_t i = 0; i < n; i++)
  {
    v[i] = std::exp(v[i] - max);
    sum += v[i];
  }
  for (uint32_t i = 0; i < n; i++)
  {
    v[i] /= sum;
  }
}

void step_baseline(uint32_t new_i, float *new_q, float *new_k, float *new_v,
                   float *cache_k, float *cache_v,
                   float dot_product_scale, float *temp_dot_product,
                   float *out)
{
  // Copy the new KV to the cache
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    cache_k[new_i * n_hidden + i] = new_k[i];
    cache_v[new_i * n_hidden + i] = new_v[i];
  }

  // Calculate the dot product with each cached K
  for (uint32_t i = 0; i < n_context; i++)
  {
    if (i <= new_i)
    {
      temp_dot_product[i] = dot_product_scale * vector_dot_product_baseline(n_hidden, new_q, &cache_k[i * n_hidden]);
    }
    else
    {
      temp_dot_product[i] = 0.0f;
    }
  }

  softmax(n_context, temp_dot_product);

  // Calculate the weighted sum of the cached V
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    out[i] = 0.0f;
  }
  for (uint32_t i_context = 0; i_context < n_context; i_context++)
  {
    float weight = temp_dot_product[i_context];
    for (uint32_t i_hidden = 0; i_hidden < n_hidden; i_hidden++)
    {
      out[i_hidden] += weight * cache_v[i_context * n_hidden + i_hidden];
    }
  }
}

void step_fast(uint32_t new_i, float *new_q, float *new_k, float *new_v,
               float *cache_k, float *cache_v,
               float dot_product_scale, float *temp_dot_product,
               float *out)
{
  // Copy the new KV to the cache
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    cache_k[new_i * n_hidden + i] = new_k[i];
    cache_v[new_i * n_hidden + i] = new_v[i];
  }

  // Calculate the dot product with each cached K
  for (uint32_t i = 0; i <= new_i; i++)
  {
    temp_dot_product[i] = dot_product_scale * vector_dot_product_fast_n_hidden(new_q, &cache_k[i * n_hidden]);
  }

  softmax(n_context, temp_dot_product);

  // Calculate the weighted sum of the cached V
  for (uint32_t offset = 0; offset < n_hidden; offset += 16)
  {
    __m256 sum_0 = _mm256_setzero_ps();
    __m256 sum_1 = _mm256_setzero_ps();
    for (uint32_t i_context = 0; i_context <= new_i; i_context++)
    {
      float weight = temp_dot_product[i_context];
      __m256 v_0 = _mm256_load_ps(&cache_v[i_context * n_hidden + offset]);
      __m256 v_1 = _mm256_load_ps(&cache_v[i_context * n_hidden + offset + 8]);
      sum_0 = _mm256_fmadd_ps(_mm256_set1_ps(weight), v_0, sum_0);
      sum_1 = _mm256_fmadd_ps(_mm256_set1_ps(weight), v_1, sum_1);
    }
    _mm256_store_ps(&out[offset], sum_0);
    _mm256_store_ps(&out[offset], sum_1);
  }
}

float rand_float_neg_1_1()
{
  return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

int main()
{
  srand(0);

  assert(cache_line_bytes % (8 * sizeof(float)) == 0);
  assert(n_context % cache_line_bytes == 0);
  assert(n_hidden % cache_line_bytes == 0);

  // allocate aligned to cache lines
  float *input_q = (float *)aligned_alloc(cache_line_bytes, n_context * n_hidden * sizeof(float));
  float *input_k = (float *)aligned_alloc(cache_line_bytes, n_context * n_hidden * sizeof(float));
  float *input_v = (float *)aligned_alloc(cache_line_bytes, n_context * n_hidden * sizeof(float));

  float *output_before_projection = (float *)aligned_alloc(cache_line_bytes, n_context * n_hidden * sizeof(float));

  float *cache_k = (float *)aligned_alloc(cache_line_bytes, n_layers * n_context * n_hidden * sizeof(float));
  float *cache_v = (float *)aligned_alloc(cache_line_bytes, n_layers * n_context * n_hidden * sizeof(float));

  float *temp_dot_product = (float *)aligned_alloc(cache_line_bytes, n_context * sizeof(float));

  for (uint32_t i = 0; i < n_layers * n_context * n_hidden; i++)
  {
    cache_k[i] = 0.0f;
    cache_v[i] = 0.0f;
  }

  for (uint32_t i = 0; i < n_context * n_hidden; i++)
  {
    input_q[i] = rand_float_neg_1_1();
    input_k[i] = rand_float_neg_1_1();
    input_v[i] = rand_float_neg_1_1();
  }

  uint32_t test_size = n_context * n_hidden;
  uint32_t repetitions = 500;
  uint32_t warmup_repetitions = 10;

  {
    auto start = std::chrono::high_resolution_clock::now();
    float sum = 0.0f;

    for (uint32_t i = 0; i < repetitions + warmup_repetitions; i++)
    {
      if (i == warmup_repetitions)
      {
        start = std::chrono::high_resolution_clock::now();
      }

      for (uint32_t j = 0; j < test_size; j++)
      {
        sum += input_q[j];
      }
    }

    printf("v1 sum: %f\n", sum);
    auto end = std::chrono::high_resolution_clock::now();
    printf("time per iteration: %fns\n", double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / double(repetitions));
    fflush(stdout);
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    float sum = 0.0f;

    for (uint32_t i = 0; i < repetitions + warmup_repetitions; i++)
    {
      if (i == warmup_repetitions)
      {
        start = std::chrono::high_resolution_clock::now();
      }

      __m256 sum_0 = _mm256_setzero_ps();
      __m256 sum_1 = _mm256_setzero_ps();
      for (uint32_t j = 0; j < test_size; j += 16)
      {
        __m256 v_0 = _mm256_load_ps(&input_q[j]);
        __m256 v_1 = _mm256_load_ps(&input_q[j + 8]);
        sum_0 = _mm256_add_ps(sum_0, v_0);
        sum_1 = _mm256_add_ps(sum_1, v_1);
      }
      sum += _mm256_add_elements_ps(_mm256_add_ps(sum_0, sum_1));
    }

    printf("v2 sum: %f\n", sum);
    auto end = std::chrono::high_resolution_clock::now();
    printf("time per iteration: %fns\n", double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / double(repetitions));
    fflush(stdout);
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    float sum = 0.0f;

    for (uint32_t i = 0; i < repetitions + warmup_repetitions; i++)
    {
      if (i == warmup_repetitions)
      {
        start = std::chrono::high_resolution_clock::now();
      }

      __m256 sum_0 = _mm256_setzero_ps();
      __m256 sum_1 = _mm256_setzero_ps();
      __m256 sum_2 = _mm256_setzero_ps();
      __m256 sum_3 = _mm256_setzero_ps();
      for (uint32_t j = 0; j < test_size; j += 32)
      {
        __m256 v_0 = _mm256_load_ps(&input_q[j]);
        __m256 v_1 = _mm256_load_ps(&input_q[j + 8]);
        __m256 v_2 = _mm256_load_ps(&input_q[j + 16]);
        __m256 v_3 = _mm256_load_ps(&input_q[j + 24]);
        sum_0 = _mm256_add_ps(sum_0, v_0);
        sum_1 = _mm256_add_ps(sum_1, v_1);
        sum_2 = _mm256_add_ps(sum_2, v_2);
        sum_3 = _mm256_add_ps(sum_3, v_3);
      }
      sum += _mm256_add_elements_ps(_mm256_add_ps(_mm256_add_ps(sum_0, sum_1), _mm256_add_ps(sum_2, sum_3)));
    }

    printf("v3 sum: %f\n", sum);
    auto end = std::chrono::high_resolution_clock::now();
    printf("time per iteration: %fns\n", double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / double(repetitions));
    fflush(stdout);
  }

  float dot_product_scale = 1.0f / sqrtf((float)n_hidden / (float)n_heads);

  auto start_10 = std::chrono::high_resolution_clock::now();
  for (uint32_t i_context = 0; i_context < n_context; i_context++)
  {
    if (i_context % 10 == 0 && i_context != 0)
    {
      auto now = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = now - start_10;
      printf(" time for last 10 of %d iterations: %fs; %fus per KV pair and layer; %fMB KV cache per layer\n",
             i_context,
             float(elapsed.count()),
             1e6f * float(elapsed.count()) / 10.0f / float(i_context) / float(n_layers),
             2 * i_context * n_hidden * sizeof(float) / 1e6f);
      start_10 = now;
    }

    printf(".");
    fflush(stdout);
    for (uint32_t i_layer = 0; i_layer < n_layers; i_layer++)
    {
      // HACK the inputs should be different for each layer
      step_fast(i_context, &input_q[i_context * n_hidden], &input_k[i_context * n_hidden], &input_v[i_context * n_hidden],
                &cache_k[i_layer * n_context * n_hidden], &cache_v[i_layer * n_context * n_hidden],
                dot_product_scale, temp_dot_product,
                &output_before_projection[i_context * n_hidden]);
    }
  }
  printf("\n");

  printf("output_before_projection[0 * n_hidden + 0]: %f\n", output_before_projection[0 * n_hidden + 0]);
  printf("output_before_projection[0 * n_hidden + 1]: %f\n", output_before_projection[0 * n_hidden + 1]);
  printf("output_before_projection[20 * n_hidden + 6]: %f\n", output_before_projection[20 * n_hidden + 6]);
  printf("output_before_projection[(n_context - 1) * n_hidden + (n_hidden - 1)]: %f\n", output_before_projection[(n_context - 1) * n_hidden + (n_hidden - 1)]);

  return 0;
}