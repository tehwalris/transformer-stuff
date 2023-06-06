#pragma once

#include "model.h"
#include "loading.h"

SimpleTransformerLayer *create_baseline_llama_layer(SimpleLlamaModelLoader *loader, uint32_t layer_index);