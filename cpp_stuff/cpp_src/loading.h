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

  struct GPTQMatrix
  {
    uint32_t rows;
    uint32_t cols;
    uint32_t block_size;

    // This always uses 4 bits per quantized value. We pack 8 values into a
    // uint32_t. That is the reason for "/ 8". Shapes are in [cols][rows] order,
    // but the in-memory layout is row-major.

    uint32_t *qweight; // Packed 4 bit signed integers (uint32_t shape: [cols / 8][rows])
    uint32_t *qzeros;  // Packed 4 bit signed integers (uint32_t shape: [cols / block_size][rows / 8])
    uint16_t *scales;  // 16 bit floats (uint16_t shape: [cols / 8][rows])
  };

  struct LlamaGPTQLayerWeights
  {
    uint16_t *input_layernorm;
    GPTQMatrix self_attn_q_proj;
    GPTQMatrix self_attn_k_proj;
    GPTQMatrix self_attn_v_proj;
    GPTQMatrix self_attn_o_proj;
    uint16_t *post_attention_layernorm;
    GPTQMatrix mlp_up_proj;
    GPTQMatrix mlp_gate_proj;
    GPTQMatrix mlp_down_proj;
  };

};