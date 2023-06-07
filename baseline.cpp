#include "baseline.h"
#include <cassert>

const uint32_t n_ff_multiple = 256;
const uint32_t cache_line_bytes = 64;

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

float *aligned_alloc_floats(uint32_t n)
{
  return (float *)aligned_alloc(cache_line_bytes, n * sizeof(float));
}

class BaselineLlamaLayer : public SimpleTransformerLayer
{
public:
  BaselineLlamaLayer(SimpleLlamaModelLoader *loader, uint32_t layer_index)
  {
    llama_hparams *hparams = loader->get_hparams();
    n_hidden = hparams->n_embd;
    n_context = hparams->n_ctx;
    n_heads = hparams->n_head;
    n_ff = ((2 * (4 * n_hidden) / 3 + n_ff_multiple - 1) / n_ff_multiple) * n_ff_multiple;

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

    weights.q = get_weights("attention.wq.weight", {n_hidden, n_hidden});
    weights.k = get_weights("attention.wk.weight", {n_hidden, n_hidden});
    weights.v = get_weights("attention.wv.weight", {n_hidden, n_hidden});
    weights.o = get_weights("attention.wo.weight", {n_hidden, n_hidden});
    weights.l1 = get_weights("feed_forward.w1.weight", {n_ff, n_hidden});
    weights.l2 = get_weights("feed_forward.w2.weight", {n_hidden, n_ff});
    weights.l3 = get_weights("feed_forward.w3.weight", {n_ff, n_hidden});
    weights.attention_norm = get_weights("attention_norm.weight", {n_hidden});
    weights.ff_norm = get_weights("ffn_norm.weight", {n_hidden});

    temp.embedding_0 = aligned_alloc_floats(n_hidden);
    temp.embedding_1 = aligned_alloc_floats(n_hidden);
    temp.dot_product = aligned_alloc_floats(n_heads * n_context);
    temp.norm_residual = aligned_alloc_floats(n_hidden);
    temp.attention_result = aligned_alloc_floats(n_hidden);
    temp.q = aligned_alloc_floats(n_hidden);
    temp.k = aligned_alloc_floats(n_hidden);
    temp.v = aligned_alloc_floats(n_hidden);
    temp.o = aligned_alloc_floats(n_hidden);
    temp.l1 = aligned_alloc_floats(n_ff);
    temp.l2 = aligned_alloc_floats(n_hidden);
    temp.l3 = aligned_alloc_floats(n_ff);
    temp.model_norm = aligned_alloc_floats(n_hidden);
  }

  BaselineLlamaLayer(const BaselineLlamaLayer &) = delete;

  virtual ~BaselineLlamaLayer()
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
  }

  virtual void forward(int n, float *hidden_in, float *hidden_out) override
  {
    // TODO
  }

  virtual void reset() override
  {
    // TODO
  }

private:
  uint32_t n_hidden;
  uint32_t n_context;
  uint32_t n_heads;
  uint32_t n_ff;

  Weights weights;
  Temp temp;
};

SimpleTransformerLayer *create_baseline_llama_layer(SimpleLlamaModelLoader *loader, uint32_t layer_index)
{
  return new BaselineLlamaLayer(loader, layer_index);
}