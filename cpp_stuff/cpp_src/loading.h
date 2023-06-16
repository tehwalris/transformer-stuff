#pragma once

#include <string>
#include <vector>
#include "llama.h"

namespace cml
{

  class SimpleLlamaModelLoader
  {
  public:
    SimpleLlamaModelLoader(const char *fname_base);
    SimpleLlamaModelLoader(const SimpleLlamaModelLoader &) = delete;
    ~SimpleLlamaModelLoader();

    float *get_tensor_float(const std::string &name, const std::vector<uint32_t> &shape);
    llama_hparams *get_hparams() const;

  private:
    llama_model_loader *loader;
    uint8_t *loading_buffer;
    float *conversion_buffer;
  };

};