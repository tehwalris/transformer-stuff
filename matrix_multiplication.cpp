#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <immintrin.h>
#include <fmaintrin.h>
#include <f16cintrin.h>
#include <chrono>

const uint32_t n_hidden = 4096, n_context = 2048, n_layers = 32, n_heads = 32, n_ff_multiple = 256, n_vocab = 32000;
const uint32_t cache_line_bytes = 64;

const uint32_t n_ff = ((2 * (4 * n_hidden) / 3 + n_ff_multiple - 1) / n_ff_multiple) * n_ff_multiple;
const float dot_product_scale = 1.0f / std::sqrt(n_hidden / n_heads);

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

template <uint32_t n>
void rms_norm(float *in, float *out)
{
  float eps = 1e-6;

  float sum = 0.0f;
  for (uint32_t i = 0; i < n; i++)
  {
    sum += in[i] * in[i];
  }

  float scale = 1.0f / std::sqrt(sum / float(n) + eps);
  for (uint32_t i = 0; i < n; i++)
  {
    out[i] = in[i] * scale;
  }
}

float silu(float x)
{
  return x / (1.0f + expf(-x));
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

static inline float fp32_from_bits(uint32_t w)
{
  union
  {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f)
{
  union
  {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

static inline float compute_fp16_to_fp32(uint16_t h)
{
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
  const float exp_scale = 0x1.0p-112f;
  const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign |
                          (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

static inline uint16_t compute_fp32_to_fp16(float f)
{
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000))
  {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

static float table_f32_f16[1 << 16];

void init_table_f32_f16()
{
  for (int i = 0; i < (1 << 16); ++i)
  {
    table_f32_f16[i] = compute_fp16_to_fp32(i);
  }
}

// Same format that llama.cpp uses
#define QK8_0 32
typedef struct
{
  uint16_t d;       // delta (fp16)
  int8_t qs[QK8_0]; // quants
} block_q8_0;

static void from_quantized_block_q8_0(block_q8_0 *quantized, float *output)
{
  float d = table_f32_f16[quantized->d];
  for (int j = 0; j < QK8_0; ++j)
  {
    output[j] = float(quantized->qs[j]) * d;
  }
}

// Multiply an m x n matrix with an n element vector and store the result in an m element vector.
// The matrix is stored in row-major order.
template <uint32_t m, uint32_t n>
void matrix_vector_multiply_quantized(block_q8_0 *mat_in, float *vec_in, float *vec_out)
{
  assert(n % QK8_0 == 0);
  for (uint32_t i_row = 0; i_row < m; i_row++)
  {
    float sum = 0.0;
    for (uint32_t i_col = 0; i_col < n; i_col += QK8_0)
    {
      float block[QK8_0];
      from_quantized_block_q8_0((&mat_in[(i_row * n + i_col) / QK8_0]), block);
      for (uint32_t i_offset = 0; i_offset < QK8_0; i_offset++)
      {
        sum += block[i_offset] * vec_in[i_col + i_offset];
      }
    }
    vec_out[i_row] = sum;
  }
}

void apply_rope(uint32_t new_i, float *vec)
{
  assert(n_hidden % 2 == 0);

  float theta_scale = powf(10000.0, -2.0f / float(n_hidden));

  float theta = float(new_i);
  for (uint32_t i = 0; i < n_hidden; i += 2)
  {
    float cos_theta = std::cos(theta);
    float sin_theta = std::sin(theta);

    theta *= theta_scale;

    float old_0 = vec[i];
    float old_1 = vec[i + 1];

    vec[i] = old_0 * cos_theta - old_1 * sin_theta;
    vec[i + 1] = old_0 * sin_theta + old_1 * cos_theta;
  }
}

void attention_baseline(uint32_t new_i, float *new_q, float *new_k, float *new_v,
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

void attention_fast(uint32_t new_i, float *new_q, float *new_k, float *new_v,
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

struct TransformerLayerWeights
{
  block_q8_0 *q;         // n_hidden * n_hidden
  block_q8_0 *k;         // n_hidden * n_hidden
  block_q8_0 *v;         // n_hidden * n_hidden
  block_q8_0 *o;         // n_hidden * n_hidden
  block_q8_0 *l1;        // n_ff * n_hidden
  block_q8_0 *l2;        // n_ff * n_hidden
  block_q8_0 *l3;        // n_hidden * n_ff
  float *attention_norm; // n_hidden
  float *ff_norm;        // n_hidden
};

struct TransformerWholeWeights
{
  TransformerLayerWeights *layers; // n_layers
  block_q8_0 *token_embeddings;    // n_vocab * n_hidden
  float *model_norm;               // n_hidden
  block_q8_0 *output_layer;        // n_hidden * n_vocab
};

struct TempBaseline
{
  float *embedding_0;      // n_hidden
  float *embedding_1;      // n_hidden
  float *dot_product;      // n_context
  float *norm_residual;    // n_hidden
  float *attention_result; // n_hidden
  float *q;                // n_hidden
  float *k;                // n_hidden
  float *v;                // n_hidden
  float *o;                // n_hidden
  float *l1;               // n_ff
  float *l2;               // n_ff
  float *l3;               // n_hidden
  float *model_norm;       // n_hidden
};

void transformer_layer_baseline(uint32_t new_i, float *new_hidden,
                                TransformerLayerWeights &w,
                                float *cache_k, float *cache_v,
                                TempBaseline &temp,
                                float *out)
{
  // Norm before attention
  rms_norm<n_hidden>(new_hidden, temp.norm_residual);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    temp.norm_residual[i] *= w.attention_norm[i];
  }

  // Compute Q, K, V
  matrix_vector_multiply_quantized<n_hidden, n_hidden>(w.q, new_hidden, temp.q);
  matrix_vector_multiply_quantized<n_hidden, n_hidden>(w.k, new_hidden, temp.k);
  matrix_vector_multiply_quantized<n_hidden, n_hidden>(w.v, new_hidden, temp.v);

  // Apply RoPE
  apply_rope(new_i, temp.q);
  apply_rope(new_i, temp.k);

  // Attention
  attention_baseline(new_i, temp.q, temp.k, temp.v, cache_k, cache_v, temp.dot_product, temp.attention_result);

  // Projection and residual
  matrix_vector_multiply_quantized<n_hidden, n_hidden>(w.o, temp.attention_result, temp.o);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    temp.o[i] += temp.norm_residual[i];
  }

  // Norm before feed forward
  rms_norm<n_hidden>(temp.o, temp.norm_residual);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    temp.norm_residual[i] *= w.ff_norm[i];
  }

  // Feed forward
  matrix_vector_multiply_quantized<n_ff, n_hidden>(w.l1, temp.o, temp.l1);
  matrix_vector_multiply_quantized<n_ff, n_hidden>(w.l3, temp.o, temp.l3);
  for (uint32_t i = 0; i < n_ff; i++)
  {
    temp.l3[i] *= silu(temp.l1[i]);
  }
  matrix_vector_multiply_quantized<n_hidden, n_ff>(w.l2, temp.l3, temp.l2);

  // Residual
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    out[i] = temp.l2[i] + temp.norm_residual[i];
  }
}

void transformer_whole_baseline(uint32_t new_i, uint32_t new_token,
                                TransformerWholeWeights &w,
                                float *cache_k, float *cache_v,
                                TempBaseline &temp,
                                float *out)
{
  float *embedding_in = temp.embedding_0;
  float *embedding_out = temp.embedding_1;
  float *last_layer_embedding_out = nullptr;

  assert(n_hidden % QK8_0 == 0);
  for (uint32_t i = 0; i < n_hidden; i += QK8_0)
  {
    float block[QK8_0];
    from_quantized_block_q8_0(&w.token_embeddings[(new_token * n_hidden + i) / QK8_0], block);
    for (uint32_t i_offset = 0; i_offset < QK8_0; i_offset++)
    {
      embedding_in[i + i_offset] = block[i_offset];
    }
  }

  for (uint32_t i_layer; i_layer < n_layers; i_layer++)
  {
    transformer_layer_baseline(new_i, embedding_in, w.layers[i_layer], cache_k, cache_v, temp, embedding_out);

    last_layer_embedding_out = embedding_out;
    float *temp = embedding_in;
    embedding_in = embedding_out;
    embedding_out = temp;
  }

  rms_norm<n_hidden>(last_layer_embedding_out, temp.model_norm);
  matrix_vector_multiply_quantized<n_vocab, n_hidden>(w.output_layer, temp.model_norm, out);
}

float rand_float_neg_1_1()
{
  return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

template <uint32_t n>
void fill_rand_int8(int8_t *arr)
{
  assert(n % 4 == 0);
  for (uint32_t i = 0; i < n; i += 4)
  {
    *((int32_t *)(void *)(arr + i)) = rand();
  }
}

void rand_block_q8_0(block_q8_0 *block, float scale)
{
  block->d = compute_fp32_to_fp16(scale);
  fill_rand_int8<QK8_0>(block->qs);
}

void fill_rand_block_q8_0(block_q8_0 *block, uint32_t n, float scale)
{
  assert(n % QK8_0 == 0);
  for (uint32_t i_block = 0; i_block < n / QK8_0; i_block++)
  {
    rand_block_q8_0(&block[i_block], scale);
  }
}

int main()
{
  srand(0);

  init_table_f32_f16();

  assert(cache_line_bytes % (8 * sizeof(float)) == 0);
  assert(n_context % cache_line_bytes == 0);
  assert(n_hidden % cache_line_bytes == 0);
  assert(n_hidden % cache_line_bytes == 0);
  assert(n_hidden % n_heads == 0);
  assert(n_hidden % QK8_0 == 0);
  assert(n_context % QK8_0 == 0);

  TransformerWholeWeights weights;
  weights.layers = (TransformerLayerWeights *)aligned_alloc(cache_line_bytes, n_layers * sizeof(TransformerLayerWeights));
  for (uint32_t i = 0; i < n_layers; i++)
  {
    weights.layers[i].q = (block_q8_0 *)aligned_alloc(cache_line_bytes, n_hidden * n_hidden / QK8_0 * sizeof(block_q8_0));
    weights.layers[i].k = (block_q8_0 *)aligned_alloc(cache_line_bytes, n_hidden * n_hidden / QK8_0 * sizeof(block_q8_0));
    weights.layers[i].v = (block_q8_0 *)aligned_alloc(cache_line_bytes, n_hidden * n_hidden / QK8_0 * sizeof(block_q8_0));
    weights.layers[i].o = (block_q8_0 *)aligned_alloc(cache_line_bytes, n_hidden * n_hidden / QK8_0 * sizeof(block_q8_0));
    weights.layers[i].l1 = (block_q8_0 *)aligned_alloc(cache_line_bytes, n_ff * n_hidden / QK8_0 * sizeof(block_q8_0));
    weights.layers[i].l2 = (block_q8_0 *)aligned_alloc(cache_line_bytes, n_ff * n_hidden / QK8_0 * sizeof(block_q8_0));
    weights.layers[i].l3 = (block_q8_0 *)aligned_alloc(cache_line_bytes, n_hidden * n_ff / QK8_0 * sizeof(block_q8_0));
    weights.layers[i].attention_norm = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
    weights.layers[i].ff_norm = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  }
  weights.token_embeddings = (block_q8_0 *)aligned_alloc(cache_line_bytes, n_vocab * n_hidden / QK8_0 * sizeof(block_q8_0));
  weights.model_norm = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  weights.output_layer = (block_q8_0 *)aligned_alloc(cache_line_bytes, n_vocab * n_hidden / QK8_0 * sizeof(block_q8_0));

  float weight_scale = 1.0f / sqrtf(n_hidden);
  for (uint32_t i = 0; i < n_layers; i++)
  {
    fill_rand_block_q8_0(weights.layers[i].q, n_hidden * n_hidden, weight_scale);
    fill_rand_block_q8_0(weights.layers[i].k, n_hidden * n_hidden, weight_scale);
    fill_rand_block_q8_0(weights.layers[i].v, n_hidden * n_hidden, weight_scale);
    fill_rand_block_q8_0(weights.layers[i].o, n_hidden * n_hidden, weight_scale);
    fill_rand_block_q8_0(weights.layers[i].l1, n_ff * n_hidden, weight_scale);
    fill_rand_block_q8_0(weights.layers[i].l1, n_ff * n_hidden, weight_scale);
    fill_rand_block_q8_0(weights.layers[i].l1, n_hidden * n_ff, weight_scale);
    for (uint32_t j = 0; j < n_hidden; j++)
    {
      weights.layers[i].attention_norm[j] = rand_float_neg_1_1();
      weights.layers[i].ff_norm[j] = rand_float_neg_1_1();
    }
  }
  fill_rand_block_q8_0(weights.token_embeddings, n_vocab * n_hidden, weight_scale);
  fill_rand_block_q8_0(weights.output_layer, n_vocab * n_hidden, weight_scale);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    weights.model_norm[i] = rand_float_neg_1_1();
  }

  TempBaseline temp;
  temp.embedding_0 = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp.embedding_1 = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp.dot_product = (float *)aligned_alloc(cache_line_bytes, n_context * sizeof(float));
  temp.norm_residual = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp.attention_result = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp.q = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp.k = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp.v = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp.o = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp.l1 = (float *)aligned_alloc(cache_line_bytes, n_ff * sizeof(float));
  temp.l2 = (float *)aligned_alloc(cache_line_bytes, n_ff * sizeof(float));
  temp.l3 = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp.model_norm = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));

  float *token_probs = (float *)aligned_alloc(cache_line_bytes, n_vocab * sizeof(float));

  float *cache_k = (float *)aligned_alloc(cache_line_bytes, n_layers * n_context * n_hidden * sizeof(float));
  float *cache_v = (float *)aligned_alloc(cache_line_bytes, n_layers * n_context * n_hidden * sizeof(float));
  for (int i = 0; i < n_layers * n_context * n_hidden; i++)
  {
    cache_k[i] = 0.0f;
    cache_v[i] = 0.0f;
  }

  printf("Initialized\n");
  fflush(stdout);

  uint32_t last_token = 1;

  uint32_t timing_group_size = 50;
  auto start_group = std::chrono::high_resolution_clock::now();
  for (int i_context = 0; i_context < n_context; i_context++)
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

    transformer_whole_baseline(i_context, last_token, weights, cache_k, cache_v, temp, token_probs);

    last_token = uint32_t(std::max_element(token_probs, token_probs + n_vocab) - token_probs);
    printf("%d ", last_token);
    fflush(stdout);
  }
  printf("\n");

  return 0;
}