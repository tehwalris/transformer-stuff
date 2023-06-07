#include <iostream>
#include <vector>
#include "loading.h"
#include "baseline.h"

float rand_float_neg_1_1()
{
  return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

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
  for (float &v : hidden_in)
  {
    v = 15.0f * rand_float_neg_1_1();
  }

  SimpleTransformerLayer *baseline_layer = create_baseline_llama_layer(&loader, 0);
  baseline_layer->forward(n_hidden, hidden_in.data(), hidden_out_baseline.data());

  for (uint32_t i : {0u, 1u, n_hidden - 2, n_hidden - 1})
  {
    std::cout << "hidden_out_baseline[" << i << "] =" << hidden_out_baseline[i] << std::endl;
  }
  float hidden_out_sum_baseline = 0.0f;
  for (float v : hidden_out_baseline)
  {
    hidden_out_sum_baseline += v;
  }
  std::cout << "hidden_out_sum_baseline = " << hidden_out_sum_baseline << std::endl;
}