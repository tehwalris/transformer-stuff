#include <cassert>
#include <cmath>
#include <algorithm>
#include "baseline.h"

namespace cml
{
  namespace baseline
  {

    const uint32_t n_ff_multiple = 256;
    const uint32_t cache_line_bytes = 64;

    struct Hyperparams
    {
      uint32_t n_hidden;
      uint32_t n_context;
      uint32_t n_heads;
      uint32_t n_ff;
    };

    void rms_norm(uint32_t n, const float *in, float *out)
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

    float vector_dot_product(uint32_t n, float *va, float *vb)
    {
      float sum = 0.0;
      for (uint32_t i = 0; i < n; i++)
      {
        sum += va[i] * vb[i];
      }
      return sum;
    }

    void matrix_vector_multiply(uint32_t m, uint32_t n, float *mat_in, float *vec_in, float *vec_out)
    {
      for (uint32_t i_row = 0; i_row < m; i_row++)
      {
        vec_out[i_row] = vector_dot_product(n, mat_in + i_row * n, vec_in);
      }
    }

    void apply_rope(const Hyperparams &params, uint32_t position, float *vec)
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

    void attention(const Hyperparams &params,
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

    struct Weights
    {
      float *q;              // n_hidden * n_hidden
      float *k;              // n_hidden * n_hidden
      float *v;              // n_hidden * n_hidden
      float *o;              // n_hidden * n_hidden
      float *l1;             // n_ff * n_hidden
      float *l2;             // n_ff * n_hidden
      float *l3;             // n_hidden * n_ff
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
    };

    struct State
    {
      float *cache_k; // n_context * n_hidden
      float *cache_v; // n_context * n_hidden
      uint32_t new_i;
    };

    class LlamaLayer : public SimpleTransformerLayer
    {
    public:
      LlamaLayer(SimpleLlamaModelLoader *loader, uint32_t layer_index)
      {
        llama_hparams *hparams = loader->get_hparams();
        params.n_hidden = hparams->n_embd;
        params.n_context = hparams->n_ctx;
        params.n_heads = hparams->n_head;
        params.n_ff = ((2 * (4 * params.n_hidden) / 3 + n_ff_multiple - 1) / n_ff_multiple) * n_ff_multiple;

        assert(layer_index < hparams->n_layer);

        auto get_weights = [&](const std::string &short_name, const std::vector<uint32_t> &shape)
        {
          std::string name = "layers." + std::to_string(layer_index) + "." + short_name;

          uint32_t num_elements = 1;
          for (uint32_t v : shape)
          {
            num_elements *= v;
          }

          float *data = aligned_alloc_floats(num_elements);
          float *data_temp = loader->get_tensor_float(name, shape);
          memcpy(data, data_temp, num_elements * sizeof(float));
          return data;
        };

        weights.q = get_weights("attention.wq.weight", {params.n_hidden, params.n_hidden});
        weights.k = get_weights("attention.wk.weight", {params.n_hidden, params.n_hidden});
        weights.v = get_weights("attention.wv.weight", {params.n_hidden, params.n_hidden});
        weights.o = get_weights("attention.wo.weight", {params.n_hidden, params.n_hidden});
        weights.l1 = get_weights("feed_forward.w1.weight", {params.n_ff, params.n_hidden});
        weights.l2 = get_weights("feed_forward.w2.weight", {params.n_hidden, params.n_ff});
        weights.l3 = get_weights("feed_forward.w3.weight", {params.n_ff, params.n_hidden});
        weights.attention_norm = get_weights("attention_norm.weight", {params.n_hidden});
        weights.ff_norm = get_weights("ffn_norm.weight", {params.n_hidden});

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

        state.cache_k = aligned_alloc_floats(params.n_context * params.n_hidden);
        state.cache_v = aligned_alloc_floats(params.n_context * params.n_hidden);
        state.new_i = 0;
      }

      LlamaLayer(const LlamaLayer &) = delete;

      virtual ~LlamaLayer()
      {
        free(weights.q);
        free(weights.k);
        free(weights.v);
        free(weights.o);
        free(weights.l1);
        free(weights.l2);
        free(weights.l3);
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

        free(state.cache_k);
        free(state.cache_v);
      }

      virtual void forward(const int n_in, const float *hidden_in, const int n_out, float *hidden_out, const uint32_t n_path, const uint32_t *path) override
      {
        assert(uint32_t(n_in) == params.n_hidden);
        assert(uint32_t(n_out) == params.n_hidden);
        assert(state.new_i < params.n_context);
        assert(n_path > 0);
        assert(n_path <= state.new_i + 1);
        assert(path[n_path - 1] == state.new_i);

        // Norm before attention
        rms_norm(params.n_hidden, hidden_in, temp.norm_residual);
        for (uint32_t i = 0; i < params.n_hidden; i++)
        {
          temp.norm_residual[i] *= weights.attention_norm[i];
        }

        // Compute Q, K, V
        matrix_vector_multiply(params.n_hidden, params.n_hidden, weights.q, temp.norm_residual, temp.q);
        matrix_vector_multiply(params.n_hidden, params.n_hidden, weights.k, temp.norm_residual, temp.k);
        matrix_vector_multiply(params.n_hidden, params.n_hidden, weights.v, temp.norm_residual, temp.v);

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
        matrix_vector_multiply(params.n_hidden, params.n_hidden, weights.o, temp.attention_result, temp.o);
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
        matrix_vector_multiply(params.n_ff, params.n_hidden, weights.l1, temp.norm_residual, temp.l1);
        matrix_vector_multiply(params.n_ff, params.n_hidden, weights.l3, temp.norm_residual, temp.l3);

        for (uint32_t i = 0; i < params.n_ff; i++)
        {
          temp.l3[i] *= silu(temp.l1[i]);
        }
        matrix_vector_multiply(params.n_hidden, params.n_ff, weights.l2, temp.l3, temp.l2);

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

      virtual void reset() override
      {
        state.new_i = 0;
      }

    private:
      Hyperparams params;
      Weights weights;
      Temp temp;
      State state;
    };

    class LlamaFinalLayer : public SimpleTransformerLayer
    {
    public:
      LlamaFinalLayer(SimpleLlamaModelLoader *loader)
      {
        llama_hparams *hparams = loader->get_hparams();
        n_hidden = hparams->n_embd;
        n_vocab = hparams->n_vocab;

        temp_model_norm = aligned_alloc_floats(n_hidden);

        float *data_temp;

        weights_model_norm = aligned_alloc_floats(n_hidden);
        data_temp = loader->get_tensor_float("norm.weight", {n_hidden});
        memcpy(weights_model_norm, data_temp, n_hidden * sizeof(float));

        weights_output_layer = aligned_alloc_floats(n_vocab * n_hidden);
        data_temp = loader->get_tensor_float("output.weight", {n_vocab, n_hidden});
        memcpy(weights_output_layer, data_temp, n_vocab * n_hidden * sizeof(float));
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

      virtual void reset() override
      {
        new_i = 0;
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
    create_llama_layer(SimpleLlamaModelLoader *loader, uint32_t layer_index)
    {
      return new LlamaLayer(loader, layer_index);
    }

    __attribute__((visibility("default")))
    SimpleTransformerLayer *
    create_llama_final_layer(SimpleLlamaModelLoader *loader)
    {
      return new LlamaFinalLayer(loader);
    }
  };
};