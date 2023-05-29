#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <immintrin.h>
#include <fmaintrin.h>
#include <f16cintrin.h>
#include <chrono>
#include <llama.h>

const uint32_t n_hidden = 4096, n_context = 2048, n_layers = 32, n_heads = 32, n_ff_multiple = 256, n_vocab = 32000;
const uint32_t cache_line_bytes = 64;

const uint32_t n_ff = ((2 * (4 * n_hidden) / 3 + n_ff_multiple - 1) / n_ff_multiple) * n_ff_multiple;
const float dot_product_scale = 1.0f / std::sqrt(float(n_hidden) / float(n_heads));

static float table_f32_f16[1 << 16];

float vector_dot_product_baseline(uint32_t n, float *va, float *vb)
{
  float sum = 0.0;
  for (uint32_t i = 0; i < n; i++)
  {
    sum += va[i] * vb[i];
  }
  return sum;
}

inline float _mm256_reduce_add_ps_float(__m256 vec)
{
  __attribute__((aligned(32))) float values[8];
  _mm256_store_ps(values, vec);

  float sum = 0.0;
  for (uint32_t i = 0; i < 8; i++)
  {
    sum += values[i];
  }
  return sum;
}

inline int32_t _mm256i_reduce_add_int16_t_int32_t(__m256i vec)
{
  __attribute__((aligned(32))) int16_t values[16];
  _mm256_store_si256((__m256i *)values, vec);

  int32_t sum = 0;
  for (uint32_t i = 0; i < 16; i++)
  {
    sum += int32_t(values[i]);
  }
  return sum;
}

inline __m256 _mm256i_reduce_add_int16_t_float(__m256i vec)
{
  // from ggml sum_i16_pairs_float
  const __m256i summed_pairs = _mm256_madd_epi16(_mm256_set1_epi16(1), vec);
  return _mm256_cvtepi32_ps(summed_pairs);
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
  return _mm256_reduce_add_ps_float(_mm256_add_ps(sum_0, sum_1));
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

// Same format that llama.cpp uses
#define QK8_0 32
typedef struct
{
  uint16_t d;       // delta (fp16)
  int8_t qs[QK8_0]; // quants
} block_q8_0;

inline void from_quantized_block_q8_0(block_q8_0 *quantized, float *output)
{
  float d = table_f32_f16[quantized->d];
  for (int j = 0; j < QK8_0; ++j)
  {
    output[j] = float(quantized->qs[j]) * d;
  }
}

inline float to_quantized_block_q8_0_raw(float *input, int8_t *quantized)
{
  float d = 0.0;
  for (int i = 0; i < QK8_0; ++i)
  {
    d = std::max(d, std::abs(input[i]));
  }

  d /= float((1 << 7) - 1);
  float id = d ? 1.0f / d : 0.0f;

  for (int i = 0; i < QK8_0; ++i)
  {
    quantized[i] = int8_t(std::round(input[i] * id));
  }

  return d;
}

inline void to_quantized_block_q8_0(float *input, block_q8_0 *quantized)
{
  float d = to_quantized_block_q8_0_raw(input, quantized->qs);
  quantized->d = ggml_fp32_to_fp16(d);
}

float dot_product_block_q8_0_baseline(block_q8_0 *a, block_q8_0 *b)
{
  float sum = 0.0;
  for (uint32_t i = 0; i < QK8_0; i++)
  {
    sum += float(a->qs[i]) * float(b->qs[i]);
  }
  return sum * ggml_fp16_to_fp32(a->d) * ggml_fp16_to_fp32(b->d);
}

inline __m256 dot_product_block_q8_0_fast(block_q8_0 *a, int8_t *b_qs, float b_d, __m256 out_accumulator)
{
  __m256i qa = _mm256_loadu_si256((__m256i *)a->qs);
  __m256i qb = _mm256_load_si256((__m256i *)b_qs);

  __m256i abs_a = _mm256_sign_epi8(qa, qa);
  __m256i b_times_sign_a = _mm256_sign_epi8(qb, qa);

  __m256i summed_products = _mm256_maddubs_epi16(abs_a, b_times_sign_a);
  __m256i summed_product_pairs = _mm256_madd_epi16(_mm256_set1_epi16(1), summed_products);
  __m256 sum = _mm256_cvtepi32_ps(summed_product_pairs);

  __m256 scale = _mm256_set1_ps(table_f32_f16[a->d] * b_d);
  return _mm256_fmadd_ps(sum, scale, out_accumulator);
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
      from_quantized_block_q8_0(&mat_in[(i_row * n + i_col) / QK8_0], block);
      for (uint32_t i_offset = 0; i_offset < QK8_0; i_offset++)
      {
        sum += block[i_offset] * vec_in[i_col + i_offset];
      }
    }
    vec_out[i_row] = sum;
  }
}

template <uint32_t m, uint32_t n>
void matrix_vector_multiply_quantized_fast(block_q8_0 *mat_in, float *vec_in,
                                           float *vec_out,
                                           int8_t *temp_vec_in_quantized_qs, float *temp_vec_in_quantized_d)
{
  assert(n % QK8_0 == 0);

  for (uint32_t i_block = 0; i_block < n / QK8_0; i_block++)
  {
    temp_vec_in_quantized_d[i_block] = to_quantized_block_q8_0_raw(&vec_in[i_block * QK8_0], &temp_vec_in_quantized_qs[i_block * QK8_0]);
  }

  for (uint32_t i_row = 0; i_row < m; i_row++)
  {
    __m256 sum = _mm256_setzero_ps();
    for (uint32_t i_block = 0; i_block < n / QK8_0; i_block++)
    {
      sum = dot_product_block_q8_0_fast(&mat_in[(i_row * n) / QK8_0 + i_block], &temp_vec_in_quantized_qs[i_block * QK8_0], temp_vec_in_quantized_d[i_block], sum);
    }
    vec_out[i_row] = _mm256_reduce_add_ps_float(sum);
  }
}

void apply_rope(uint32_t new_i, float *vec)
{
  assert(n_hidden % n_heads == 0);
  assert((n_hidden / n_heads) % 2 == 0);

  float theta_scale = powf(10000.0, -2.0f / (float(n_hidden) / float(n_heads)));

  float theta = float(new_i);
  for (uint32_t i = 0; i < n_hidden; i += 2)
  {
    if (i % (n_hidden / n_heads) == 0)
    {
      // RoPE is applied separately to each head
      theta = float(new_i);
    }

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

  // Calculate the dot product with each cached K (per head)
  for (uint32_t i_context = 0; i_context <= new_i; i_context++)
  {
    for (uint32_t i_head = 0; i_head < n_heads; i_head++)
    {
      uint32_t head_offset = i_head * (n_hidden / n_heads);
      temp_dot_product[i_head * n_context + i_context] = dot_product_scale * vector_dot_product_baseline(n_hidden / n_heads, &new_q[head_offset], &cache_k[i_context * n_hidden + head_offset]);
    }
  }

  for (uint32_t i_head = 0; i_head < n_heads; i_head++)
  {
    softmax(new_i + 1, &temp_dot_product[i_head * n_context]);
  }

  // Calculate the weighted sum of the cached V
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    out[i] = 0.0f;
  }
  for (uint32_t i_context = 0; i_context <= new_i; i_context++)
  {
    for (uint32_t i_head = 0; i_head < n_heads; i_head++)
    {
      float weight = temp_dot_product[i_head * n_context + i_context];
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
      temp_dot_product[i_head * n_context + i_context] = dot_product_scale * vector_dot_product_fast<n_hidden / n_heads>(&new_q[head_offset], &cache_k[i_context * n_hidden + head_offset]);
    }
  }

  for (uint32_t i_head = 0; i_head < n_heads; i_head++)
  {
    softmax(new_i + 1, &temp_dot_product[i_head * n_context]);
  }

  // Calculate the weighted sum of the cached V
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    out[i] = 0.0f;
  }
  for (uint32_t i_context = 0; i_context <= new_i; i_context++)
  {
    for (uint32_t i_head = 0; i_head < n_heads; i_head++)
    {
      __m256 weight = _mm256_set1_ps(temp_dot_product[i_head * n_context + i_context]);
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

struct TempFast
{
  float *embedding_0;        // n_hidden
  float *embedding_1;        // n_hidden
  float *dot_product;        // n_heads * n_context
  float *norm_residual;      // n_hidden
  float *attention_result;   // n_hidden
  float *q;                  // n_hidden
  float *k;                  // n_hidden
  float *v;                  // n_hidden
  float *o;                  // n_hidden
  float *l1;                 // n_ff
  float *l2;                 // n_hidden
  float *l3;                 // n_ff
  float *model_norm;         // n_hidden
  block_q8_0 *quantized_vec; // n_ff
  int8_t *quantized_vec_qs;  // n_ff
  float *quantized_vec_d;    // n_ff
};

void transformer_layer_fast(uint32_t new_i, float *new_hidden,
                            TransformerLayerWeights &w,
                            float *cache_k, float *cache_v,
                            TempFast &temp,
                            float *out)
{
  // Norm before attention
  rms_norm<n_hidden>(new_hidden, temp.norm_residual);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    temp.norm_residual[i] *= w.attention_norm[i];
  }

  // Compute Q, K, V
  matrix_vector_multiply_quantized_fast<n_hidden, n_hidden>(w.q, temp.norm_residual, temp.q, temp.quantized_vec_qs, temp.quantized_vec_d);
  matrix_vector_multiply_quantized_fast<n_hidden, n_hidden>(w.k, temp.norm_residual, temp.k, temp.quantized_vec_qs, temp.quantized_vec_d);
  matrix_vector_multiply_quantized_fast<n_hidden, n_hidden>(w.v, temp.norm_residual, temp.v, temp.quantized_vec_qs, temp.quantized_vec_d);

  // Apply RoPE
  apply_rope(new_i, temp.q);
  apply_rope(new_i, temp.k);

  // Attention
  attention_fast(new_i, temp.q, temp.k, temp.v,
                 cache_k, cache_v,
                 temp.dot_product,
                 temp.attention_result);

  // Projection and residual
  matrix_vector_multiply_quantized_fast<n_hidden, n_hidden>(w.o, temp.attention_result, temp.o, temp.quantized_vec_qs, temp.quantized_vec_d);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    temp.o[i] += new_hidden[i];
  }

  // Norm before feed forward
  rms_norm<n_hidden>(temp.o, temp.norm_residual);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    temp.norm_residual[i] *= w.ff_norm[i];
  }

  // Feed forward
  matrix_vector_multiply_quantized_fast<n_ff, n_hidden>(w.l1, temp.norm_residual, temp.l1, temp.quantized_vec_qs, temp.quantized_vec_d);
  matrix_vector_multiply_quantized_fast<n_ff, n_hidden>(w.l3, temp.norm_residual, temp.l3, temp.quantized_vec_qs, temp.quantized_vec_d);

  for (uint32_t i = 0; i < n_ff; i++)
  {
    temp.l3[i] *= silu(temp.l1[i]);
  }
  matrix_vector_multiply_quantized_fast<n_hidden, n_ff>(w.l2, temp.l3, temp.l2, temp.quantized_vec_qs, temp.quantized_vec_d);

  // Residual
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    out[i] = temp.l2[i] + temp.o[i];
  }
}

void transformer_whole_fast(uint32_t new_i, uint32_t new_token,
                            TransformerWholeWeights &w,
                            float *cache_k, float *cache_v,
                            TempFast &temp,
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

  for (uint32_t i_layer = 0; i_layer < n_layers; i_layer++)
  {
    transformer_layer_fast(new_i, embedding_in,
                           w.layers[i_layer],
                           &cache_k[i_layer * n_context * n_hidden], &cache_v[i_layer * n_context * n_hidden],
                           temp,
                           embedding_out);

    last_layer_embedding_out = embedding_out;
    float *temp = embedding_in;
    embedding_in = embedding_out;
    embedding_out = temp;
  }

  rms_norm<n_hidden>(last_layer_embedding_out, temp.model_norm);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    temp.model_norm[i] *= w.model_norm[i];
  }
  matrix_vector_multiply_quantized_fast<n_vocab, n_hidden>(w.output_layer, temp.model_norm, out, temp.quantized_vec_qs, temp.quantized_vec_d);
}

struct TempBaseline
{
  float *embedding_0;      // n_hidden
  float *embedding_1;      // n_hidden
  float *dot_product;      // n_heads * n_context
  float *norm_residual;    // n_hidden
  float *attention_result; // n_hidden
  float *q;                // n_hidden
  float *k;                // n_hidden
  float *v;                // n_hidden
  float *o;                // n_hidden
  float *l1;               // n_ff
  float *l2;               // n_hidden
  float *l3;               // n_ff
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
  matrix_vector_multiply_quantized<n_hidden, n_hidden>(w.q, temp.norm_residual, temp.q);
  matrix_vector_multiply_quantized<n_hidden, n_hidden>(w.k, temp.norm_residual, temp.k);
  matrix_vector_multiply_quantized<n_hidden, n_hidden>(w.v, temp.norm_residual, temp.v);

  // Apply RoPE
  apply_rope(new_i, temp.q);
  apply_rope(new_i, temp.k);

  // Attention
  attention_baseline(new_i, temp.q, temp.k, temp.v,
                     cache_k, cache_v,
                     temp.dot_product,
                     temp.attention_result);

  // Projection and residual
  matrix_vector_multiply_quantized<n_hidden, n_hidden>(w.o, temp.attention_result, temp.o);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    temp.o[i] += new_hidden[i];
  }

  // Norm before feed forward
  rms_norm<n_hidden>(temp.o, temp.norm_residual);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    temp.norm_residual[i] *= w.ff_norm[i];
  }

  // Feed forward
  matrix_vector_multiply_quantized<n_ff, n_hidden>(w.l1, temp.norm_residual, temp.l1);
  matrix_vector_multiply_quantized<n_ff, n_hidden>(w.l3, temp.norm_residual, temp.l3);

  for (uint32_t i = 0; i < n_ff; i++)
  {
    temp.l3[i] *= silu(temp.l1[i]);
  }
  matrix_vector_multiply_quantized<n_hidden, n_ff>(w.l2, temp.l3, temp.l2);

  // Residual
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    out[i] = temp.l2[i] + temp.o[i];
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

  for (uint32_t i_layer = 0; i_layer < n_layers; i_layer++)
  {
    transformer_layer_baseline(new_i, embedding_in,
                               w.layers[i_layer],
                               &cache_k[i_layer * n_context * n_hidden], &cache_v[i_layer * n_context * n_hidden],
                               temp,
                               embedding_out);

    last_layer_embedding_out = embedding_out;
    float *temp = embedding_in;
    embedding_in = embedding_out;
    embedding_out = temp;
  }

  rms_norm<n_hidden>(last_layer_embedding_out, temp.model_norm);
  for (uint32_t i = 0; i < n_hidden; i++)
  {
    temp.model_norm[i] *= w.model_norm[i];
  }
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
  block->d = ggml_fp32_to_fp16(scale);
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

float *float_ptr_from_ggml_tensor(ggml_tensor *t)
{
  assert(t->type == GGML_TYPE_F32);
  return (float *)t->data;
}

block_q8_0 *block_q8_0_ptr_from_ggml_tensor(ggml_tensor *t)
{
  assert(t->type == GGML_TYPE_Q8_0);
  return (block_q8_0 *)t->data;
}

llama_context *load_llama_model(char *model_path, TransformerWholeWeights *weights)
{
  auto lparams = llama_context_default_params();
  lparams.n_ctx = n_context;
  lparams.n_gpu_layers = 0;
  lparams.seed = 0;
  lparams.f16_kv = false;
  lparams.use_mmap = true;
  lparams.use_mlock = false;
  lparams.logits_all = false;
  lparams.embedding = false;
  lparams.vocab_only = false;
  lparams.model_ony = false;

  llama_context *lctx = llama_init_from_file(model_path, lparams);
  llama_model *model = llama_get_model(lctx);

  assert(model->hparams.n_vocab == n_vocab);
  assert(model->hparams.n_ctx == n_context);
  assert(model->hparams.n_embd == n_hidden);
  assert(model->hparams.n_mult == n_ff_multiple);
  assert(model->hparams.n_head == n_heads);
  assert(model->hparams.n_layer == n_layers);

  weights->layers = new TransformerLayerWeights[n_layers];
  for (uint32_t i_layer = 0; i_layer < n_layers; i_layer++)
  {
    TransformerLayerWeights &out_layer = weights->layers[i_layer];
    llama_layer &in_layer = model->layers[i_layer];

    out_layer.attention_norm = float_ptr_from_ggml_tensor(in_layer.attention_norm);
    out_layer.ff_norm = float_ptr_from_ggml_tensor(in_layer.ffn_norm);

    out_layer.q = block_q8_0_ptr_from_ggml_tensor(in_layer.wq);
    out_layer.k = block_q8_0_ptr_from_ggml_tensor(in_layer.wk);
    out_layer.v = block_q8_0_ptr_from_ggml_tensor(in_layer.wv);
    out_layer.o = block_q8_0_ptr_from_ggml_tensor(in_layer.wo);

    out_layer.l1 = block_q8_0_ptr_from_ggml_tensor(in_layer.w1);
    out_layer.l2 = block_q8_0_ptr_from_ggml_tensor(in_layer.w2);
    out_layer.l3 = block_q8_0_ptr_from_ggml_tensor(in_layer.w3);
  }

  weights->token_embeddings = block_q8_0_ptr_from_ggml_tensor(model->tok_embeddings);
  weights->model_norm = float_ptr_from_ggml_tensor(model->norm);
  weights->output_layer = block_q8_0_ptr_from_ggml_tensor(model->output);

  return lctx;
}

float thing_baseline(uint32_t n, int8_t *a, float *b)
{
  float sum = 0.0f;
  for (uint32_t i = 0; i < n; i++)
  {
    sum += float(a[i]) * b[i];
  }
  return sum;
}

template <uint32_t n>
float thing_fast(int8_t *a, float *b_shuffled)
{
  assert(n % 32 == 0);

  __m256 sum = _mm256_setzero_ps();
  for (uint32_t i = 0; i < n / 32; i++)
  {
    __m256i va_epi8 = _mm256_load_si256(&((__m256i *)a)[i]);

    // Per 128 bit lane: Load the first 4 int8 values from va_epi8 into the MSB of 4 int32 values of va_epi32_0
    __m256i shuffle_mask_0 = _mm256_set_epi32(
        0x03FFFFFF, 0x02FFFFFF, 0x01FFFFFF, 0x00FFFFFF,
        0x03FFFFFF, 0x02FFFFFF, 0x01FFFFFF, 0x00FFFFFF);
    __m256i va_epi32_0 = _mm256_shuffle_epi8(va_epi8, shuffle_mask_0);
    // Sign extend each int32 to value to correctly represent the int8 value
    va_epi32_0 = _mm256_srai_epi32(va_epi32_0, 24);
    __m256 va_ps_0 = _mm256_cvtepi32_ps(va_epi32_0);
    __m256 vb_0 = _mm256_load_ps(&b_shuffled[i * 32 + 0]);
    sum = _mm256_fmadd_ps(va_ps_0, vb_0, sum);

    __m256i shuffle_mask_1 = _mm256_set_epi32(
        0x07FFFFFF, 0x06FFFFFF, 0x05FFFFFF, 0x04FFFFFF,
        0x07FFFFFF, 0x06FFFFFF, 0x05FFFFFF, 0x04FFFFFF);
    __m256i va_epi32_1 = _mm256_shuffle_epi8(va_epi8, shuffle_mask_1);
    va_epi32_1 = _mm256_srai_epi32(va_epi32_1, 24);
    __m256 va_ps_1 = _mm256_cvtepi32_ps(va_epi32_1);
    __m256 vb_1 = _mm256_load_ps(&b_shuffled[i * 32 + 8]);
    sum = _mm256_fmadd_ps(va_ps_1, vb_1, sum);

    __m256i shuffle_mask_2 = _mm256_set_epi32(
        0x0BFFFFFF, 0x0AFFFFFF, 0x09FFFFFF, 0x08FFFFFF,
        0x0BFFFFFF, 0x0AFFFFFF, 0x09FFFFFF, 0x08FFFFFF);
    __m256i va_epi32_2 = _mm256_shuffle_epi8(va_epi8, shuffle_mask_2);
    va_epi32_2 = _mm256_srai_epi32(va_epi32_2, 24);
    __m256 va_ps_2 = _mm256_cvtepi32_ps(va_epi32_2);
    __m256 vb_2 = _mm256_load_ps(&b_shuffled[i * 32 + 16]);
    sum = _mm256_fmadd_ps(va_ps_2, vb_2, sum);

    __m256i shuffle_mask_3 = _mm256_set_epi32(
        0x0FFFFFFF, 0x0EFFFFFF, 0x0DFFFFFF, 0x0CFFFFFF,
        0x0FFFFFFF, 0x0EFFFFFF, 0x0DFFFFFF, 0x0CFFFFFF);
    __m256i va_epi32_3 = _mm256_shuffle_epi8(va_epi8, shuffle_mask_3);
    va_epi32_3 = _mm256_srai_epi32(va_epi32_3, 24);
    __m256 va_ps_3 = _mm256_cvtepi32_ps(va_epi32_3);
    __m256 vb_3 = _mm256_load_ps(&b_shuffled[i * 32 + 24]);
    sum = _mm256_fmadd_ps(va_ps_3, vb_3, sum);

    // __attribute__((aligned(32))) uint32_t debug_int[8];
    // _mm256_store_si256((__m256i *)debug_int, va_epi32_1);
    // __attribute__((aligned(32))) float debug_float[8];
    // _mm256_store_ps(debug_float, vb_1);
    // printf("va_ep32_1: ");
    // for (uint32_t j = 0; j < 8; j++)
    // {
    //   printf("%8x ", debug_int[j]);
    // }
    // printf("\n");
    // printf("vb_1:      ");
    // for (uint32_t j = 0; j < 8; j++)
    // {
    //   printf("%f ", debug_float[j]);
    // }
    // printf("\n");
    // printf("a:         ");
    // for (uint32_t j = 0; j < 32; j++)
    // {
    //   if (j != 0 && j % 8 == 0)
    //   {
    //     printf("\n           ");
    //   }
    //   printf("%8x ", a[i * 32 + j]);
    // }
    // printf("\n");
    // printf("b:         ");
    // for (uint32_t j = 0; j < 32; j++)
    // {
    //   if (j != 0 && j % 8 == 0)
    //   {
    //     printf("\n           ");
    //   }
    //   printf("%f ", b[i * 32 + j]);
    // }
    // printf("\n");
    // return 0.0f;
  }
  return _mm256_reduce_add_ps_float(sum);
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    printf("usage: %s <model_path>\n", argv[0]);
    exit(1);
  }
  char *model_path = argv[1];

  srand(0);

  assert(cache_line_bytes % (8 * sizeof(float)) == 0);
  assert(n_context % cache_line_bytes == 0);
  assert(n_hidden % cache_line_bytes == 0);
  assert(n_hidden % cache_line_bytes == 0);
  assert(n_hidden % n_heads == 0);
  assert(n_hidden % QK8_0 == 0);
  assert(n_context % QK8_0 == 0);
  assert(n_ff > n_hidden);
  assert(n_ff % QK8_0 == 0);

  // TempBaseline temp_baseline;
  // temp_baseline.embedding_0 = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  // temp_baseline.embedding_1 = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  // temp_baseline.dot_product = (float *)aligned_alloc(cache_line_bytes, n_heads * n_context * sizeof(float));
  // temp_baseline.norm_residual = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  // temp_baseline.attention_result = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  // temp_baseline.q = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  // temp_baseline.k = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  // temp_baseline.v = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  // temp_baseline.o = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  // temp_baseline.l1 = (float *)aligned_alloc(cache_line_bytes, n_ff * sizeof(float));
  // temp_baseline.l2 = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  // temp_baseline.l3 = (float *)aligned_alloc(cache_line_bytes, n_ff * sizeof(float));
  // temp_baseline.model_norm = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));

  TempFast temp_fast;
  temp_fast.embedding_0 = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp_fast.embedding_1 = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp_fast.dot_product = (float *)aligned_alloc(cache_line_bytes, n_heads * n_context * sizeof(float));
  temp_fast.norm_residual = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp_fast.attention_result = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp_fast.q = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp_fast.k = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp_fast.v = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp_fast.o = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp_fast.l1 = (float *)aligned_alloc(cache_line_bytes, n_ff * sizeof(float));
  temp_fast.l2 = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp_fast.l3 = (float *)aligned_alloc(cache_line_bytes, n_ff * sizeof(float));
  temp_fast.model_norm = (float *)aligned_alloc(cache_line_bytes, n_hidden * sizeof(float));
  temp_fast.quantized_vec = (block_q8_0 *)aligned_alloc(cache_line_bytes, (n_ff / QK8_0) * sizeof(block_q8_0));
  temp_fast.quantized_vec_qs = (int8_t *)aligned_alloc(cache_line_bytes, n_ff * sizeof(int8_t));
  temp_fast.quantized_vec_d = (float *)aligned_alloc(cache_line_bytes, n_ff * sizeof(float));

  {
    const uint32_t n = n_hidden;
    uint32_t num_iterations = 1e5;

    int8_t *a = (int8_t *)aligned_alloc(cache_line_bytes, n * sizeof(int8_t));
    float *b = (float *)aligned_alloc(cache_line_bytes, n * sizeof(float));
    float *b_shuffled = (float *)aligned_alloc(cache_line_bytes, n * sizeof(float));

    fill_rand_int8<n>(a);
    for (uint32_t i = 0; i < n; i++)
    {
      b[i] = (float)rand_float_neg_1_1();
    }

    /*
    0 1 2 3 - 16 17 18 19
    4 5 6 7 - 20 21 22 23
    8 9 10 11 - 24 25 26 27
    12 13 14 15 - 28 29 30 31
    */
    assert(n % 32 == 0);
    for (uint32_t i = 0; i < n; i += 32)
    {
      for (uint32_t j = 0; j < 4; j++)
      {
        for (uint32_t k = 0; k < 4; k++)
        {
          b_shuffled[i + j * 8 + k + 0] = b[i + j * 4 + k + 0];
          b_shuffled[i + j * 8 + k + 4] = b[i + j * 4 + k + 16];
        }
      }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    volatile float result_baseline = 0.0f, result_fast = 0.0f;

    start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++)
    {
      result_fast += thing_fast<n>(a, b_shuffled);
    }
    end = std::chrono::high_resolution_clock::now();
    {
      std::chrono::duration<double> elapsed = end - start;
      printf("thing_fast: %f us per iteration; result %f\n", elapsed.count() / num_iterations * 1e6f, result_fast);
    }

    start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++)
    {
      result_baseline += thing_baseline(n, a, b);
    }
    end = std::chrono::high_resolution_clock::now();
    {
      std::chrono::duration<double> elapsed = end - start;
      printf("thing_baseline: %f us per iteration; result %f\n", elapsed.count() / num_iterations * 1e6f, result_baseline);
    }

    return 0;
  }

  llama_init_backend();
  TransformerWholeWeights weights;
  llama_context *lctx = load_llama_model(model_path, &weights);

  printf("Loaded model\n");
  fflush(stdout);

  for (uint16_t i = 0; true; i++)
  {
    table_f32_f16[i] = ggml_fp16_to_fp32(i);
    if (i == UINT16_MAX)
    {
      break;
    }
  }

  float *token_probs = (float *)aligned_alloc(cache_line_bytes, n_vocab * sizeof(float));

  float *cache_k = (float *)aligned_alloc(cache_line_bytes, n_layers * n_context * n_hidden * sizeof(float));
  float *cache_v = (float *)aligned_alloc(cache_line_bytes, n_layers * n_context * n_hidden * sizeof(float));
  for (uint32_t i = 0; i < n_layers * n_context * n_hidden; i++)
  {
    cache_k[i] = 0.0f;
    cache_v[i] = 0.0f;
  }

  printf("Initialized\n");
  fflush(stdout);

  const uint32_t max_input_tokens = 1000;
  std::string input_string = " How are you";
  llama_token input_tokens[max_input_tokens];
  uint32_t n_input_tokens = llama_tokenize(lctx, input_string.c_str(), input_tokens, max_input_tokens, true);
  assert(n_input_tokens >= 1);

  for (uint32_t i = 0; i < n_input_tokens; i++)
  {
    printf("\"%s\" ", llama_token_to_str(lctx, input_tokens[i]));
  }
  printf("\n");
  fflush(stdout);

  llama_token last_token = input_tokens[0];

  uint32_t timing_group_size = 5;
  auto start_group = std::chrono::high_resolution_clock::now();
  for (uint32_t i_context = 0; i_context < n_context; i_context++)
  {
    if (i_context % timing_group_size == 0 && i_context != 0)
    {
      auto now = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = now - start_group;
      printf("\n=== time for last %d of %d iterations: %fs ===\n",
             timing_group_size,
             i_context,
             float(elapsed.count()));
      start_group = now;
    }

    llama_token input_token = i_context < n_input_tokens ? input_tokens[i_context] : last_token;
    printf("%s", llama_token_to_str(lctx, input_token));
    fflush(stdout);

    // transformer_whole_baseline(i_context, uint32_t(input_token), weights, cache_k, cache_v, temp_baseline, token_probs);
    transformer_whole_fast(i_context, uint32_t(input_token), weights, cache_k, cache_v, temp_fast, token_probs);
    llama_token output_token = llama_token(std::max_element(token_probs, token_probs + n_vocab) - token_probs);

    last_token = output_token;
  }
  printf("\n");

  return 0;
}