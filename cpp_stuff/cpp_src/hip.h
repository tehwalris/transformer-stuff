#pragma once

#include "model.h"
#include "loading.h"

namespace cml
{
  namespace hip
  {

    SimpleTransformerLayer *create_llama_layer(SimpleLlamaModelLoader *loader, uint32_t layer_index, uint32_t n_cache);

  };
};