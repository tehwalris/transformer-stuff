#include "model.h"

namespace cml
{
  void simple_transformer_layer_delete(SimpleTransformerLayer *layer)
  {
    delete layer;
  }

  void simple_transformer_layer_forward(SimpleTransformerLayer *layer, const int n_in, const float *hidden_in, const int n_out, float *hidden_out)
  {
    layer->forward(n_in, hidden_in, n_out, hidden_out);
  }

  void simple_transformer_layer_reset(SimpleTransformerLayer *layer)
  {
    layer->reset();
  }
}