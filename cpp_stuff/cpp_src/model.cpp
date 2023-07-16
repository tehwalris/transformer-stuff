#include "model.h"

namespace cml
{
  void simple_transformer_layer_delete(SimpleTransformerLayer *layer)
  {
    delete layer;
  }

  void simple_transformer_layer_forward(SimpleTransformerLayer *layer, const int n_in, const float *hidden_in, const int n_out, float *hidden_out, const uint32_t n_path, const uint32_t *path)
  {
    layer->forward(n_in, hidden_in, n_out, hidden_out, n_path, path);
  }

  uint32_t simple_transformer_layer_next_i(const SimpleTransformerLayer *layer)
  {
    return layer->next_i();
  }

  void simple_transformer_layer_retain(SimpleTransformerLayer *layer, const uint32_t n_retain, const uint32_t *retain)
  {
    layer->retain(n_retain, retain);
  }
}