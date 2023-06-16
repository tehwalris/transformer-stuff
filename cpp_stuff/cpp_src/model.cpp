#include "model.h"

namespace cml
{
  void simple_transformer_layer_delete(SimpleTransformerLayer *layer)
  {
    delete layer;
  }

  void simple_transformer_layer_forward(SimpleTransformerLayer *layer, int n, float *hidden_in, float *hidden_out)
  {
    layer->forward(n, hidden_in, hidden_out);
  }

  void simple_transformer_layer_reset(SimpleTransformerLayer *layer)
  {
    layer->reset();
  }
}