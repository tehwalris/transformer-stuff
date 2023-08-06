#include <cassert>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include "baseline.h"

namespace cml
{
  namespace baseline
  {

    const uint32_t cache_line_bytes = 64;

    void rms_norm(uint32_t n, const float *in, float *out)
    {
      float eps = 5e-6;

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

    float vector_dot_product(uint32_t n, float *va, float *vb)
    {
      float sum = 0.0;
      for (uint32_t i = 0; i < n; i++)
      {
        sum += va[i] * vb[i];
      }
      return sum;
    }

    void fp32s_from_fp16s(uint32_t n, const uint16_t *in, float *out)
    {
      assert(n % 16 == 0); // TODO: support non-multiple of 16
      for (uint32_t i = 0; i < n; i += 16)
      {
        __m256i v_f16 = _mm256_load_si256((__m256i *)(in + i));
        __m128i v_f16_0 = _mm256_extracti128_si256(v_f16, 0);
        __m128i v_f16_1 = _mm256_extracti128_si256(v_f16, 1);
        __m256 v_f32_0 = _mm256_cvtph_ps(v_f16_0);
        __m256 v_f32_1 = _mm256_cvtph_ps(v_f16_1);
        _mm256_store_ps(out + i, v_f32_0);
        _mm256_store_ps(out + i + 8, v_f32_1);
      }
    }

    float fp32_from_fp16(uint16_t in)
    {
      return _cvtsh_ss(in);
    }

    void unquantize_row(const GPTQMatrix &mat, uint32_t i_row, float *out)
    {
      for (uint32_t i_col = 0; i_col < mat.cols; i_col++)
      {
        uint32_t i_qweight = (i_col / 8) * mat.rows + i_row;
        uint32_t i_group = mat.g_idx[i_col];
        uint32_t i_qzeros = i_group * (mat.rows / 8) + (i_row / 8);
        uint32_t i_scales = i_group * mat.rows + i_row;

        float scale = fp32_from_fp16(mat.scales[i_scales]);

        uint32_t zero_quant_group = mat.qzeros[i_qzeros];
        uint32_t zero_quant = (zero_quant_group >> (4 * (i_row % 8))) & 0xf;
        float zero = float(zero_quant + 1) * scale;

        uint32_t weight_quant_group = mat.qweight[i_qweight];
        uint32_t weight_quant = (weight_quant_group >> (4 * (i_col % 8))) & 0xf;
        float weight = float(weight_quant) * scale;

        out[i_col] = weight - zero;
      }
    }

    void matrix_vector_multiply_gptq(const GPTQMatrix &mat, float *temp_row, float *vec_in, float *vec_out)
    {
      for (uint32_t i_row = 0; i_row < mat.rows; i_row++)
      {
        unquantize_row(mat, i_row, temp_row);
        vec_out[i_row] = vector_dot_product(mat.cols, temp_row, vec_in);
      }
    }

    void matrix_vector_multiply(uint32_t m, uint32_t n, float *mat_in, float *vec_in, float *vec_out)
    {
      for (uint32_t i_row = 0; i_row < m; i_row++)
      {
        vec_out[i_row] = vector_dot_product(n, mat_in + i_row * n, vec_in);
      }
    }

    void apply_rope(const LlamaHyperparams &params, uint32_t position, float *vec)
    {
      assert(params.n_hidden % params.n_heads == 0);
      assert((params.n_hidden / params.n_heads) % 2 == 0);

      float theta_scale = powf(10000.0, -2.0f / (float(params.n_hidden) / float(params.n_heads)));

      float theta = float(position);
      for (uint32_t i = 0; i < params.n_hidden; i += 2)
      {
        if (i % (params.n_hidden / params.n_heads) == 0)
        {
          // RoPE is applied separately to each head
          theta = float(position);
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

    void attention(const LlamaHyperparams &params,
                   uint32_t n_path, const uint32_t *path,
                   uint32_t new_i, float *new_q, float *new_k, float *new_v,
                   float *cache_k, float *cache_v,
                   float *temp_dot_product,
                   float *out)
    {
      // Copy the new KV to the cache
      for (uint32_t i = 0; i < params.n_hidden; i++)
      {
        cache_k[new_i * params.n_hidden + i] = new_k[i];
        cache_v[new_i * params.n_hidden + i] = new_v[i];
      }

      // Calculate the dot product with each cached K (per head)
      const float dot_product_scale = 1.0f / std::sqrt(float(params.n_hidden / params.n_heads));
      for (uint32_t i_path = 0; i_path < n_path; i_path++)
      {
        uint32_t i_context = path[i_path];
        for (uint32_t i_head = 0; i_head < params.n_heads; i_head++)
        {
          uint32_t head_offset = i_head * (params.n_hidden / params.n_heads);
          temp_dot_product[i_head * params.n_context + i_path] = dot_product_scale * vector_dot_product(params.n_hidden / params.n_heads, &new_q[head_offset], &cache_k[i_context * params.n_hidden + head_offset]);
        }
      }

      for (uint32_t i_head = 0; i_head < params.n_heads; i_head++)
      {
        softmax(n_path, &temp_dot_product[i_head * params.n_context]);
      }

      // Calculate the weighted sum of the cached V
      for (uint32_t i = 0; i < params.n_hidden; i++)
      {
        out[i] = 0.0f;
      }
      for (uint32_t i_path = 0; i_path < n_path; i_path++)
      {
        uint32_t i_context = path[i_path];
        for (uint32_t i_head = 0; i_head < params.n_heads; i_head++)
        {
          float weight = temp_dot_product[i_head * params.n_context + i_path];
          for (uint32_t i_hidden = 0; i_hidden < params.n_hidden / params.n_heads; i_hidden++)
          {
            uint32_t offset = i_head * (params.n_hidden / params.n_heads) + i_hidden;
            out[offset] += weight * cache_v[i_context * params.n_hidden + offset];
          }
        }
      }
    }

    float *aligned_alloc_floats(uint32_t n)
    {
      return (float *)aligned_alloc(cache_line_bytes, n * sizeof(float));
    }

    GPTQMatrix copy_gptq_matrix(const GPTQMatrix &old_mat)
    {
      assert(old_mat.block_size % 8 == 0);
      assert(old_mat.rows % old_mat.block_size == 0);
      assert(old_mat.cols % old_mat.block_size == 0);

      GPTQMatrix new_mat;
      new_mat.rows = old_mat.rows;
      new_mat.cols = old_mat.cols;
      new_mat.block_size = old_mat.block_size;

      size_t qweight_bytes = (old_mat.cols / 8) * old_mat.rows * sizeof(uint32_t);
      size_t qzeros_bytes = (old_mat.cols / old_mat.block_size) * (old_mat.rows / 8) * sizeof(uint32_t);
      size_t scales_bytes = (old_mat.cols / old_mat.block_size) * old_mat.rows * sizeof(uint16_t);
      size_t g_idx_bytes = old_mat.cols * sizeof(uint32_t);

      uint32_t *qweight = (uint32_t *)malloc(qweight_bytes);
      uint32_t *qzeros = (uint32_t *)malloc(qzeros_bytes);
      uint16_t *scales = (uint16_t *)malloc(scales_bytes);
      uint32_t *g_idx = (uint32_t *)malloc(g_idx_bytes);

      memcpy(qweight, old_mat.qweight, qweight_bytes);
      memcpy(qzeros, old_mat.qzeros, qzeros_bytes);
      memcpy(scales, old_mat.scales, scales_bytes);
      memcpy(g_idx, old_mat.g_idx, g_idx_bytes);

      new_mat.qweight = qweight;
      new_mat.qzeros = qzeros;
      new_mat.scales = scales;
      new_mat.g_idx = g_idx;

      return new_mat;
    }

    void free_gptq_matrix(GPTQMatrix &mat)
    {
      free(mat.qweight);
      free(mat.qzeros);
      free(mat.scales);
      free(mat.g_idx);

      mat.qweight = nullptr;
      mat.qzeros = nullptr;
      mat.scales = nullptr;
      mat.g_idx = nullptr;
    }

    struct Weights
    {
      GPTQMatrix q;          // n_hidden * n_hidden
      GPTQMatrix k;          // n_hidden * n_hidden
      GPTQMatrix v;          // n_hidden * n_hidden
      GPTQMatrix o;          // n_hidden * n_hidden
      GPTQMatrix l1;         // n_ff * n_hidden
      GPTQMatrix l2;         // n_hidden * n_ff
      GPTQMatrix l3;         // n_ff * n_hidden
      float *attention_norm; // n_hidden
      float *ff_norm;        // n_hidden
    };

    struct Temp
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
      float *unquantized_row;  // n_ff
    };

    struct State
    {
      float *cache_k; // n_cache * n_hidden
      float *cache_v; // n_cache * n_hidden
      uint32_t new_i;
    };

    class LlamaLayer : public SimpleTransformerLayer
    {
    public:
      LlamaLayer(const LlamaGPTQLayerWeights *loader_weights, LlamaHyperparams params, uint32_t n_cache) : params(params), n_cache(n_cache)
      {
        auto get_weight_matrix = [&](const GPTQMatrix &mat, const std::vector<uint32_t> &shape)
        {
          assert(shape.size() == 2);
          assert(mat.rows == shape[0]);
          assert(mat.cols == shape[1]);
          return copy_gptq_matrix(mat);
        };

        auto get_1d_weights = [&](const uint16_t *values, uint32_t n)
        {
          float *copied_values = aligned_alloc_floats(n);
          fp32s_from_fp16s(n, values, copied_values);
          return copied_values;
        };

        weights.q = get_weight_matrix(loader_weights->self_attn_q_proj, {params.n_hidden, params.n_hidden});
        weights.k = get_weight_matrix(loader_weights->self_attn_k_proj, {params.n_hidden, params.n_hidden});
        weights.v = get_weight_matrix(loader_weights->self_attn_v_proj, {params.n_hidden, params.n_hidden});
        weights.o = get_weight_matrix(loader_weights->self_attn_o_proj, {params.n_hidden, params.n_hidden});
        weights.l1 = get_weight_matrix(loader_weights->mlp_gate_proj, {params.n_ff, params.n_hidden});
        weights.l2 = get_weight_matrix(loader_weights->mlp_down_proj, {params.n_hidden, params.n_ff});
        weights.l3 = get_weight_matrix(loader_weights->mlp_up_proj, {params.n_ff, params.n_hidden});
        weights.attention_norm = get_1d_weights(loader_weights->input_layernorm, params.n_hidden);
        weights.ff_norm = get_1d_weights(loader_weights->post_attention_layernorm, params.n_hidden);

        temp.embedding_0 = aligned_alloc_floats(params.n_hidden);
        temp.embedding_1 = aligned_alloc_floats(params.n_hidden);
        temp.dot_product = aligned_alloc_floats(params.n_heads * params.n_context);
        temp.norm_residual = aligned_alloc_floats(params.n_hidden);
        temp.attention_result = aligned_alloc_floats(params.n_hidden);
        temp.q = aligned_alloc_floats(params.n_hidden);
        temp.k = aligned_alloc_floats(params.n_hidden);
        temp.v = aligned_alloc_floats(params.n_hidden);
        temp.o = aligned_alloc_floats(params.n_hidden);
        temp.l1 = aligned_alloc_floats(params.n_ff);
        temp.l2 = aligned_alloc_floats(params.n_hidden);
        temp.l3 = aligned_alloc_floats(params.n_ff);
        temp.model_norm = aligned_alloc_floats(params.n_hidden);
        temp.unquantized_row = aligned_alloc_floats(params.n_ff);

        state.cache_k = aligned_alloc_floats(n_cache * params.n_hidden);
        state.cache_v = aligned_alloc_floats(n_cache * params.n_hidden);
        state.new_i = 0;
      }

      LlamaLayer(const LlamaLayer &) = delete;

      virtual ~LlamaLayer()
      {
        free_gptq_matrix(weights.q);
        free_gptq_matrix(weights.k);
        free_gptq_matrix(weights.v);
        free_gptq_matrix(weights.o);
        free_gptq_matrix(weights.l1);
        free_gptq_matrix(weights.l2);
        free_gptq_matrix(weights.l3);
        free(weights.attention_norm);
        free(weights.ff_norm);

        free(temp.embedding_0);
        free(temp.embedding_1);
        free(temp.dot_product);
        free(temp.norm_residual);
        free(temp.attention_result);
        free(temp.q);
        free(temp.k);
        free(temp.v);
        free(temp.o);
        free(temp.l1);
        free(temp.l2);
        free(temp.l3);
        free(temp.model_norm);
        free(temp.unquantized_row);

        free(state.cache_k);
        free(state.cache_v);
      }

      virtual void forward(const int n_in, const float *hidden_in, const int n_out, float *hidden_out, const uint32_t n_path, const uint32_t *path) override
      {
        assert(uint32_t(n_in) == params.n_hidden);
        assert(uint32_t(n_out) == params.n_hidden);
        assert(state.new_i < n_cache);
        assert(n_path > 0);
        assert(n_path <= state.new_i + 1);
        assert(n_path <= params.n_context);
        assert(path[n_path - 1] == state.new_i);

        // Norm before attention
        rms_norm(params.n_hidden, hidden_in, temp.norm_residual);
        for (uint32_t i = 0; i < params.n_hidden; i++)
        {
          temp.norm_residual[i] *= weights.attention_norm[i];
        }

        // Compute Q, K, V
        matrix_vector_multiply_gptq(weights.q, temp.unquantized_row, temp.norm_residual, temp.q);
        matrix_vector_multiply_gptq(weights.k, temp.unquantized_row, temp.norm_residual, temp.k);
        matrix_vector_multiply_gptq(weights.v, temp.unquantized_row, temp.norm_residual, temp.v);

        // Apply RoPE
        apply_rope(params, n_path - 1, temp.q);
        apply_rope(params, n_path - 1, temp.k);

        // Attention
        attention(params,
                  n_path, path,
                  state.new_i, temp.q, temp.k, temp.v,
                  state.cache_k, state.cache_v,
                  temp.dot_product,
                  temp.attention_result);

        // Projection and residual
        matrix_vector_multiply_gptq(weights.o, temp.unquantized_row, temp.attention_result, temp.o);
        for (uint32_t i = 0; i < params.n_hidden; i++)
        {
          temp.o[i] += hidden_in[i];
        }

        // Norm before feed forward
        rms_norm(params.n_hidden, temp.o, temp.norm_residual);
        for (uint32_t i = 0; i < params.n_hidden; i++)
        {
          temp.norm_residual[i] *= weights.ff_norm[i];
        }

        // Feed forward
        matrix_vector_multiply_gptq(weights.l1, temp.unquantized_row, temp.norm_residual, temp.l1);
        matrix_vector_multiply_gptq(weights.l3, temp.unquantized_row, temp.norm_residual, temp.l3);

        for (uint32_t i = 0; i < params.n_ff; i++)
        {
          temp.l3[i] *= silu(temp.l1[i]);
        }
        matrix_vector_multiply_gptq(weights.l2, temp.unquantized_row, temp.l3, temp.l2);

        // Residual
        for (uint32_t i = 0; i < params.n_hidden; i++)
        {
          hidden_out[i] = temp.l2[i] + temp.o[i];
        }

        state.new_i++;
      }

      virtual uint32_t next_i() const override
      {
        return state.new_i;
      }

      virtual void retain(const uint32_t n_retain, const uint32_t *retain) override
      {
        assert(n_retain <= state.new_i);
        assert(n_retain <= n_cache);

        float *old_cache_k = state.cache_k;
        float *old_cache_v = state.cache_v;

        state.cache_k = aligned_alloc_floats(n_cache * params.n_hidden);
        state.cache_v = aligned_alloc_floats(n_cache * params.n_hidden);

        for (uint32_t i = 0; i < n_retain; i++)
        {
          memcpy(&state.cache_k[i * params.n_hidden], &old_cache_k[retain[i] * params.n_hidden], params.n_hidden * sizeof(float));
          memcpy(&state.cache_v[i * params.n_hidden], &old_cache_v[retain[i] * params.n_hidden], params.n_hidden * sizeof(float));
        }

        free(old_cache_k);
        free(old_cache_v);

        state.new_i = n_retain;
      }

    private:
      LlamaHyperparams params;
      uint32_t n_cache;
      Weights weights;
      Temp temp;
      State state;
    };

    class LlamaFinalLayer : public SimpleTransformerLayer
    {
    public:
      LlamaFinalLayer(const LlamaFinalLayerWeights *loader_weights, LlamaHyperparams params)
      {
        n_hidden = params.n_hidden;
        n_vocab = params.n_vocab;

        temp_model_norm = aligned_alloc_floats(n_hidden);

        float *data_temp;

        weights_model_norm = aligned_alloc_floats(n_hidden);
        fp32s_from_fp16s(n_hidden, loader_weights->norm, weights_model_norm);

        weights_output_layer = aligned_alloc_floats(n_vocab * n_hidden);
        fp32s_from_fp16s(n_vocab * n_hidden, loader_weights->lm_head, weights_output_layer);

        new_i = 0;
      }

      LlamaFinalLayer(const LlamaFinalLayer &) = delete;

      virtual ~LlamaFinalLayer()
      {
        free(temp_model_norm);
        free(weights_model_norm);
        free(weights_output_layer);
      }

      virtual void forward(const int n_in, const float *hidden_in, const int n_out, float *hidden_out, const uint32_t n_path, const uint32_t *path) override
      {
        assert(uint32_t(n_in) == n_hidden);
        assert(uint32_t(n_out) == n_vocab);
        assert(n_path > 0);
        assert(n_path <= new_i + 1);
        assert(path[n_path - 1] == new_i);

        // Norm before output layer
        rms_norm(n_hidden, hidden_in, temp_model_norm);
        for (uint32_t i = 0; i < n_hidden; i++)
        {
          temp_model_norm[i] *= weights_model_norm[i];
        }

        // Output layer
        matrix_vector_multiply(n_vocab, n_hidden, weights_output_layer, temp_model_norm, hidden_out);

        new_i++;
      }

      virtual uint32_t next_i() const override
      {
        return new_i;
      }

      virtual void retain(const uint32_t n_retain, const uint32_t *retain) override
      {
        assert(n_retain <= new_i);
        new_i = n_retain;
      }

    private:
      uint32_t n_hidden;
      uint32_t n_vocab;
      uint32_t new_i;
      float *temp_model_norm;
      float *weights_model_norm;
      float *weights_output_layer;
    };

    __attribute__((visibility("default")))
    SimpleTransformerLayer *
    create_llama_layer_gptq(const LlamaGPTQLayerWeights *loader_weights, LlamaHyperparams params, uint32_t n_cache)
    {
      return new LlamaLayer(loader_weights, params, n_cache);
    }

    __attribute__((visibility("default")))
    SimpleTransformerLayer *
    create_llama_final_layer(const LlamaFinalLayerWeights *loader_weights, LlamaHyperparams params)
    {
      return new LlamaFinalLayer(loader_weights, params);
    }
  };
};