#include <iostream>
#include <algorithm>
#include <cmath>

uint32_t n_hidden = 4096, n_context = 512, n_layers = 32;

float vector_dot_product(uint32_t n, float *va, float *vb)
{
  float sum = 0.0;
  for (uint32_t i = 0; i < n; i++)
  {
    sum += va[i] * vb[i];
  }
  return sum;
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

void step(uint32_t new_i, float *new_q, float *new_k, float *new_v,
          float *cache_k, float *cache_v,
          float *temp_dot_product,
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
      temp_dot_product[i] = vector_dot_product(n_hidden, new_q, &cache_k[i * n_hidden]);
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

float rand_float_neg_1_1()
{
  return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

int main()
{
  srand(0);

  float *input_q = new float[n_context * n_hidden];
  float *input_k = new float[n_context * n_hidden];
  float *input_v = new float[n_context * n_hidden];

  float *output_before_projection = new float[n_context * n_hidden];

  float *cache_k = new float[n_context * n_hidden];
  float *cache_v = new float[n_context * n_hidden];

  float *temp_dot_product = new float[n_context];

  for (int i = 0; i < n_context * n_hidden; i++)
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

  for (int i = 0; i < n_context; i++)
  {
    printf("i: %d\n", i);
    step(i, &input_q[i * n_hidden], &input_k[i * n_hidden], &input_v[i * n_hidden],
         cache_k, cache_v,
         temp_dot_product,
         &output_before_projection[i * n_hidden]);
  }

  return 0;
}