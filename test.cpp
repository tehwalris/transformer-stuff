#include <iostream>
#include <vector>
#include "loading.h"
#include "baseline.h"
#include "cuda.h"

namespace cml
{

  float rand_float_neg_1_1()
  {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
  }

};

using namespace cml;

int main(int argc, char **argv)
{
  srand(0);

  if (argc != 2)
  {
    printf("usage: %s <model_path>\n", argv[0]);
    exit(1);
  }
  char *model_path = argv[1];

  SimpleLlamaModelLoader loader(model_path);

  uint32_t n_hidden = loader.get_hparams()->n_embd;
  std::vector<float> hidden_in(n_hidden);
  std::vector<float> hidden_out_baseline(n_hidden);
  std::vector<float> hidden_out_cuda(n_hidden);

  SimpleTransformerLayer *baseline_layer = baseline::create_llama_layer(&loader, 0);
  SimpleTransformerLayer *cuda_layer = cuda::create_llama_layer(&loader, 0);

  for (uint32_t i_context = 0; i_context < 10; i_context++)
  {
    std::cout << "i_context = " << i_context << std::endl;
    for (float &v : hidden_in)
    {
      v = rand_float_neg_1_1();
    }

    std::fill(hidden_out_baseline.begin(), hidden_out_baseline.end(), 0.0f);
    std::fill(hidden_out_cuda.begin(), hidden_out_cuda.end(), 0.0f);

    baseline_layer->forward(n_hidden, hidden_in.data(), hidden_out_baseline.data());
    cuda_layer->forward(n_hidden, hidden_in.data(), hidden_out_cuda.data());

    for (uint32_t i_hidden : {0u, 1u, n_hidden - 2, n_hidden - 1})
    {
      std::cout << "hidden_out_baseline[" << i_hidden << "] = " << hidden_out_baseline[i_hidden] << std::endl;
      std::cout << "hidden_out_cuda[" << i_hidden << "] = " << hidden_out_cuda[i_hidden] << std::endl;
    }

    float hidden_out_sum_baseline = 0.0f;
    for (float v : hidden_out_baseline)
    {
      hidden_out_sum_baseline += v;
    }
    std::cout << "hidden_out_sum_baseline = " << hidden_out_sum_baseline << std::endl;

    float hidden_out_sum_cuda = 0.0f;
    for (float v : hidden_out_cuda)
    {
      hidden_out_sum_cuda += v;
    }
    std::cout << "hidden_out_sum_cuda = " << hidden_out_sum_cuda << std::endl;

    std::cout << std::endl;
  }
}