#pragma once

#include <string>
#include <vector>
#include "llama.h"

class simple_llama_model_loader
{
public:
  simple_llama_model_loader(const std::string &fname_base);
  ~simple_llama_model_loader();

  float *get_tensor_float(const std::string &name, const std::vector<uint32_t> &shape);
  llama_hparams *get_hparams();

private:
  llama_model_loader *loader;
  std::vector<uint8_t> loading_buffer;
  std::vector<uint8_t> conversion_buffer;
};