#include <cassert>
#include <immintrin.h>
#include <iostream>
#include "loading.h"

namespace cml
{

  const uint32_t cache_line_bytes = 64;

  SimpleLlamaModelLoader::SimpleLlamaModelLoader(const char *fname_base)
  {
    loader = new_llama_model_loader(std::string(fname_base));
    loading_buffer = nullptr;
    conversion_buffer = nullptr;
  }

  SimpleLlamaModelLoader::~SimpleLlamaModelLoader()
  {
    delete_llama_model_loader(loader);
    free(loading_buffer);
    free(conversion_buffer);
  }

  float *SimpleLlamaModelLoader::get_tensor_float(const std::string &name, const std::vector<uint32_t> &shape)
  {
    assert(shape.size() >= 1 && shape.size() <= 4);
    const std::vector<uint32_t> &actual_shape = llama_model_loader_get_tensor_shape(loader, name.c_str());
    assert(actual_shape.size() <= 4);
    assert(shape.size() == actual_shape.size());

    for (size_t i = 0; i < actual_shape.size(); i++)
    {
      // ggml defines tensor shapes with the contiguous dimension first, but we want the contiguous dimension last
      assert(actual_shape[i] == shape[shape.size() - i - 1]);
    }
    uint64_t num_elements = 1;
    for (uint32_t v : actual_shape)
    {
      num_elements *= uint64_t(v);
    }

    free(loading_buffer);
    loading_buffer = nullptr;
    free(conversion_buffer);
    conversion_buffer = nullptr;

    ggml_type type = llama_model_loader_get_tensor_type(loader, name.c_str());
    if (type == GGML_TYPE_F32)
    {
      loading_buffer = (uint8_t *)aligned_alloc(cache_line_bytes, num_elements * sizeof(float));
    }
    else if (type == GGML_TYPE_F16)
    {
      loading_buffer = (uint8_t *)aligned_alloc(cache_line_bytes, num_elements * sizeof(uint16_t));
    }
    else
    {
      std::cerr << "Unsupported type: " << type << " for tensor " << name << std::endl;
      assert(false);
      return nullptr;
    }

    llama_model_loader_get_tensor_data(loader, name.c_str(), loading_buffer);

    if (type == GGML_TYPE_F32)
    {
      return (float *)loading_buffer;
    }
    else if (type == GGML_TYPE_F16)
    {
      assert(num_elements % 16 == 0); // TODO: support non-multiple of 16
      conversion_buffer = (float *)aligned_alloc(cache_line_bytes, num_elements * sizeof(float));
      for (uint64_t i = 0; i < num_elements; i += 16)
      {
        __m256i v_f16 = _mm256_load_si256((__m256i *)(loading_buffer + i * sizeof(uint16_t)));
        __m128i v_f16_0 = _mm256_extracti128_si256(v_f16, 0);
        __m128i v_f16_1 = _mm256_extracti128_si256(v_f16, 1);
        __m256 v_f32_0 = _mm256_cvtph_ps(v_f16_0);
        __m256 v_f32_1 = _mm256_cvtph_ps(v_f16_1);
        _mm256_store_ps(conversion_buffer + i, v_f32_0);
        _mm256_store_ps(conversion_buffer + i + 8, v_f32_1);
      }
      return conversion_buffer;
    }
    else
    {
      std::cerr << "Unsupported type: " << type << " for tensor " << name << std::endl;
      assert(false);
      return nullptr;
    }
  }

  llama_hparams *SimpleLlamaModelLoader::get_hparams() const
  {
    return llama_model_loader_get_hparams(loader);
  }

};