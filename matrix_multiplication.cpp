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

float _mm256_reduce_add_ps(__m256 v)
{
  float sum = 0.0;
  for (uint32_t i = 0; i < 8; i++)
  {
    sum += ((float *)&v)[i];
  }
  return sum;
}

template <uint32_t n>
float vector_dot_product_fast(float *va, float *vb)
{
  assert(n % 16 == 0);

  __m256 sum_0 = _mm256_setzero_ps();
  __m256 sum_1 = _mm256_setzero_ps();
  for (uint32_t i = 0; i < n; i += 16)
  {
    __m256 a_0 = _mm256_load_ps(va + i);
    __m256 b_0 = _mm256_load_ps(vb + i);
    __m256 a_1 = _mm256_load_ps(va + i + 8);
    __m256 b_1 = _mm256_load_ps(vb + i + 8);
    sum_0 = _mm256_fmadd_ps(a_0, b_0, sum_0);
    sum_1 = _mm256_fmadd_ps(a_1, b_1, sum_1);
  }
  return _mm256_reduce_add_ps(_mm256_add_ps(sum_0, sum_1));
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
  for (uint32_t i_context = 0; i_context < n_context; i_context++)
  {
    if (i_context <= new_i)
    {
      for (uint32_t i_head = 0; i_head < n_heads; i_head++)
      {
        uint32_t head_offset = i_head * (n_hidden / n_heads);
        temp_dot_product[i_context * n_heads + i_head] = dot_product_scale * vector_dot_product_baseline(n_hidden / n_heads, &new_q[head_offset], &cache_k[i_context * n_hidden + head_offset]);
      }
    }
    else
    {
      for (uint32_t i_head = 0; i_head < n_heads; i_head++)
      {
        temp_dot_product[i_context * n_heads + i_head] = 0.0f;
      }
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
    for (uint32_t i_head = 0; i_head < n_heads; i_head++)
    {
      float weight = temp_dot_product[i_context * n_heads + i_head];
      for (uint32_t i_hidden = 0; i_hidden < n_hidden / n_heads; i_hidden++)
      {
        uint32_t offset = i_head * (n_hidden / n_heads) + i_hidden;
        out[offset] += weight * cache_v[i_context * n_hidden + offset];
      }
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
  for (uint32_t i_context = 0; i_context <= new_i; i_context++)
  {
    for (uint32_t i_head = 0; i_head < n_heads; i_head++)
    {
      uint32_t head_offset = i_head * (n_hidden / n_heads);
      temp_dot_product[i_context * n_heads + i_head] = dot_product_scale * vector_dot_product_fast<n_hidden / n_heads>(&new_q[head_offset], &cache_k[i_context * n_hidden + head_offset]);
    }
  }

  softmax(n_context, temp_dot_product);

  // Calculate the weighted sum of the cached V
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    out[i] = 0.0f;
  }
  for (uint32_t i_context = 0; i_context <= new_i; i_context++)
  {
    for (uint32_t i_head = 0; i_head < n_heads; i_head++)
    {
      __m256 weight = _mm256_set1_ps(temp_dot_product[i_context * n_heads + i_head]);
      for (uint32_t i_hidden = 0; i_hidden < n_hidden / n_heads; i_hidden += 16)
      {
        uint32_t offset = i_head * (n_hidden / n_heads) + i_hidden;
        __m256 v_0 = _mm256_load_ps(&cache_v[i_context * n_hidden + offset]);
        __m256 v_1 = _mm256_load_ps(&cache_v[i_context * n_hidden + offset + 8]);
        __m256 out_0 = _mm256_load_ps(&out[offset]);
        __m256 out_1 = _mm256_load_ps(&out[offset + 8]);
        out_0 = _mm256_fmadd_ps(weight, v_0, out_0);
        out_1 = _mm256_fmadd_ps(weight, v_1, out_1);
        _mm256_store_ps(&out[offset], out_0);
        _mm256_store_ps(&out[offset + 8], out_1);
      }
    }
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
  assert(n_hidden % cache_line_bytes == 0);
  assert(n_hidden % n_heads == 0);

  // allocate aligned to cache lines
  float *input_q = (float *)aligned_alloc(cache_line_bytes, n_context * n_hidden * sizeof(float));
  float *input_k = (float *)aligned_alloc(cache_line_bytes, n_context * n_hidden * sizeof(float));
  float *input_v = (float *)aligned_alloc(cache_line_bytes, n_context * n_hidden * sizeof(float));

  float *output_before_projection = (float *)aligned_alloc(cache_line_bytes, n_context * n_hidden * sizeof(float));

  float *cache_k = (float *)aligned_alloc(cache_line_bytes, n_layers * n_context * n_hidden * sizeof(float));
  float *cache_v = (float *)aligned_alloc(cache_line_bytes, n_layers * n_context * n_hidden * sizeof(float));

  float *temp_dot_product = (float *)aligned_alloc(cache_line_bytes, n_context * n_heads * sizeof(float));

  for (int i = 0; i < n_layers * n_context * n_hidden; i++)
  {
    cache_k[i] = 0.0f;
    cache_v[i] = 0.0f;
  }

  for (int i = 0; i < n_context * n_hidden; i++)
  {
    input_q[i] = rand_float_neg_1_1();
    input_k[i] = rand_float_neg_1_1();
    input_v[i] = rand_float_neg_1_1();
  }

  float dot_product_scale = 1.0f / sqrtf((float)n_hidden / (float)n_heads);

  uint32_t timing_group_size = 50;
  auto start_group = std::chrono::high_resolution_clock::now();
  for (int i_context = 0; i_context < 5; i_context++)
  {
    if (i_context % 50 == 0 && i_context != 0)
    {
      auto now = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = now - start_group;
      printf(" time for last 50 of %d iterations: %fs; %fus per KV pair and layer; %fMB KV cache per layer\n",
             i_context,
             float(elapsed.count()),
             1e6f * float(elapsed.count()) / float(timing_group_size) / float(i_context) / float(n_layers),
             2 * i_context * n_hidden * sizeof(float) / 1e6f);
      start_group = now;
    }

    printf(".");
    fflush(stdout);
    for (int i_layer = 0; i_layer < n_layers; i_layer++)
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
  printf("output_before_projection[0 * n_hidden + (n_hidden - 1)]: %f\n", output_before_projection[0 * n_hidden + (n_hidden - 1)]);
  printf("output_before_projection[20 * n_hidden + 6]: %f\n", output_before_projection[20 * n_hidden + 6]);
  printf("output_before_projection[(n_context - 1) * n_hidden + (n_hidden - 1)]: %f\n", output_before_projection[(n_context - 1) * n_hidden + (n_hidden - 1)]);

  return 0;
}