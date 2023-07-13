#pragma once

#include "model.h"
#include "loading.h"

namespace cml
{
  namespace cuda
  {

    SimpleTransformerLayer *create_llama_layer(SimpleLlamaModelLoader *loader, uint32_t layer_index);

    SimpleTransformerLayer *create_llama_final_layer(SimpleLlamaModelLoader *loader);

  };
};