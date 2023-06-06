#pragma once

#include <string>
#include <vector>
#include "llama.h"

class SimpleLlamaModelLoader
{
public:
  SimpleLlamaModelLoader(const std::string &fname_base);
  ~SimpleLlamaModelLoader();

  float *get_tensor_float(const std::string &name, const std::vector<uint32_t> &shape);
  llama_hparams *get_hparams();

private:
  llama_model_loader *loader;
  uint8_t *loading_buffer;
  float *conversion_buffer;
};