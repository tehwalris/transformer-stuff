#include <cassert>
#include <immintrin.h>
#include <iostream>
#include "example.h"

namespace cml
{

  const uint32_t cache_line_bytes = 64;

  SimpleLlamaModelLoader::SimpleLlamaModelLoader(const char *fname_base)
  {
    std::string std_string_fname_base(fname_base); // TODO pass to new_llama_model_loader directly
    loading_buffer = nullptr;
    conversion_buffer = nullptr;

    std::cout << "DEBUG SimpleLlamaModelLoader(" << std_string_fname_base << ")" << std::endl;
  }

  SimpleLlamaModelLoader::~SimpleLlamaModelLoader()
  {
    free(loading_buffer);
    free(conversion_buffer);
  }

  float *SimpleLlamaModelLoader::get_tensor_float(const std::string &name, const std::vector<uint32_t> &shape)
  {
    assert(shape.size() >= 1 && shape.size() <= 4);

    free(loading_buffer);
    loading_buffer = nullptr;
    free(conversion_buffer);
    conversion_buffer = nullptr;

    return nullptr;
  }
};