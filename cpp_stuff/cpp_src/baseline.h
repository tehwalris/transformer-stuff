#pragma once

#include "model.h"
#include "loading.h"

namespace cml
{
  namespace baseline
  {

    SimpleTransformerLayer *create_llama_layer_gptq(const LlamaGPTQLayerWeights *loader_weights, LlamaHyperparams params, uint32_t n_cache);

    SimpleTransformerLayer *create_llama_final_layer(const LlamaFinalLayerWeights *loader_weights, LlamaHyperparams params);

  };
};