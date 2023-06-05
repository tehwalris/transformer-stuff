#include <cassert>
#include <immintrin.h>
#include "loading.h"

simple_llama_model_loader::simple_llama_model_loader(const std::string &fname_base)
{
  loader = new_llama_model_loader(fname_base);
}

simple_llama_model_loader::~simple_llama_model_loader()
{
  delete_llama_model_loader(loader);
}

float *simple_llama_model_loader::get_tensor_float(const std::string &name, const std::vector<uint32_t> &shape)
{
  assert(shape.size() == 4);
  const std::vector<uint32_t> &actual_shape = llama_model_loader_get_tensor_shape(loader, name.c_str());
  assert(actual_shape.size() == 4);
  for (int i = 0; i < 4; i++)
  {
    assert(actual_shape[i] == shape[i]);
  }
  uint64_t num_elements = uint64_t(shape[0]) * uint64_t(shape[1]) * uint64_t(shape[2]) * uint64_t(shape[3]);

  ggml_type type = llama_model_loader_get_tensor_type(loader, name.c_str());
  if (type == GGML_TYPE_F32)
  {
    conversion_buffer.resize(num_elements * sizeof(float));
  }
  else if (type == GGML_TYPE_F16)
  {
    conversion_buffer.resize(num_elements * sizeof(uint16_t));
  }
  else
  {
    assert(false);
    return nullptr;
  }

  llama_model_loader_get_tensor_data(loader, name.c_str(), conversion_buffer.data());

  if (type == GGML_TYPE_F32)
  {
    return (float *)loading_buffer.data();
  }
  else if (type == GGML_TYPE_F16)
  {
    conversion_buffer.resize(num_elements * sizeof(float));
    uint16_t *src = (uint16_t *)loading_buffer.data();
    float *dst = (float *)conversion_buffer.data();
    for (uint64_t i = 0; i < num_elements; i++)
    {
      dst[i] = _cvtsh_ss(src[i]);
    }
  }
  else
  {
    assert(false);
    return nullptr;
  }
}

llama_hparams *simple_llama_model_loader::get_hparams()
{
  return llama_model_loader_get_hparams(loader);
}