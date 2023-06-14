#pragma once

#include <string>
#include <vector>

namespace cml
{

  class SimpleLlamaModelLoader
  {
  public:
    SimpleLlamaModelLoader(const char *fname_base);
    SimpleLlamaModelLoader(const std::string &fname_base);
    SimpleLlamaModelLoader(const SimpleLlamaModelLoader &) = delete;
    ~SimpleLlamaModelLoader();

    float *get_tensor_float(const std::string &name, const std::vector<uint32_t> &shape);

  private:
    uint8_t *loading_buffer;
    float *conversion_buffer;
  };

};