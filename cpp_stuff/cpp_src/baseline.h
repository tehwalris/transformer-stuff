#pragma once

#include "model.h"
#include "loading.h"

namespace cml
{
  namespace baseline
  {

    SimpleTransformerLayer *create_llama_layer(SimpleLlamaModelLoader *loader, uint32_t layer_index, uint32_t n_cache);

    SimpleTransformerLayer *create_llama_final_layer(SimpleLlamaModelLoader *loader);

  };
};