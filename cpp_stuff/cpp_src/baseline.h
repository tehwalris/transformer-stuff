#pragma once

#include "model.h"
#include "loading.h"

namespace cml
{
  namespace baseline
  {

    SimpleTransformerLayer *create_llama_layer_gptq(const LlamaGPTQLayerWeights *loader_weights, const llama_hparams *hparams, uint32_t n_cache);

    SimpleTransformerLayer *create_llama_final_layer(SimpleLlamaModelLoader *loader);

  };
};