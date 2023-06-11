#include <iostream>
#include <vector>
#include <chrono>
// #include "loading.h"
// #include "baseline.h"
// #include "cuda.h"
// #include "hip.h"
#include "fill_copy_sequence.h"

// namespace cml
// {
//
//   float rand_float_neg_1_1()
//   {
//     return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
//   }
//
// };
//
// using namespace cml;
//
// void benchmark_layer(SimpleTransformerLayer *layer, const std::string &layer_name, uint32_t n_hidden, std::vector<float> &hidden_in, std::vector<float> &hidden_out_baseline, uint32_t benchmark_iterations)
// {
//   std::cout << "Benchmarking " << layer_name << std::endl;
//   layer->reset();
//
//   std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
//
//   for (uint32_t i = 0; i < benchmark_iterations; i++)
//   {
//     layer->forward(n_hidden, hidden_in.data(), hidden_out_baseline.data());
//   }
//
//   std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
//
//   std::chrono::duration<double> elapsed = end - start;
//   std::cout << "Layer took " << elapsed.count() / benchmark_iterations * 1e3 << " ms per iteration" << std::endl;
// }

int main(int argc, char **argv)
{

  uint32_t n_hidden = 4096;

  std::vector<float> hidden_in(n_hidden);

  CarrierInterface *carrier = create_carrier(n_hidden);
  carrier->hidden_in_thing(n_hidden, hidden_in.data());
}

// int main_old(int argc, char **argv)
// {
//   srand(0);
//   if (argc != 2)
//
//   {
//     printf("usage: %s <model_path>\n", argv[0]);
//     exit(1);
//   }
//   char *model_path = argv[1];
//
//   SimpleLlamaModelLoader loader(model_path);
//
//   uint32_t n_hidden = loader.get_hparams()->n_embd;
//   std::vector<float> hidden_in(n_hidden);
//   std::vector<float> hidden_out_baseline(n_hidden);
//   std::vector<float> hidden_out_cuda(n_hidden);
//   std::vector<float> hidden_out_hip(n_hidden);
//
//   SimpleTransformerLayer *baseline_layer = baseline::create_llama_layer(&loader, 0);
//   SimpleTransformerLayer *cuda_layer = cuda::create_llama_layer(&loader, 0);
//   SimpleTransformerLayer *hip_layer = hip::create_llama_layer(&loader, 0);
//
//   for (uint32_t i_context = 0; i_context < 10; i_context++)
//   {
//     std::cout << "i_context = " << i_context << std::endl;
//     for (float &v : hidden_in)
//     {
//       v = rand_float_neg_1_1();
//     }
//
//     std::fill(hidden_out_baseline.begin(), hidden_out_baseline.end(), 0.0f);
//     std::fill(hidden_out_cuda.begin(), hidden_out_cuda.end(), 0.0f);
//     std::fill(hidden_out_hip.begin(), hidden_out_hip.end(), 0.0f);
//
//     baseline_layer->forward(n_hidden, hidden_in.data(), hidden_out_baseline.data());
//     cuda_layer->forward(n_hidden, hidden_in.data(), hidden_out_cuda.data());
//     hip_layer->forward(n_hidden, hidden_in.data(), hidden_out_hip.data());
//
//     for (uint32_t i_hidden : {0u, 1u, n_hidden - 2, n_hidden - 1})
//     {
//       std::cout << "hidden_out_baseline[" << i_hidden << "] = " << hidden_out_baseline[i_hidden] << std::endl;
//       std::cout << "hidden_out_cuda[" << i_hidden << "] = " << hidden_out_cuda[i_hidden] << std::endl;
//       std::cout << "hidden_out_hip[" << i_hidden << "] = " << hidden_out_hip[i_hidden] << std::endl;
//     }
//
//     float hidden_out_sum_baseline = 0.0f;
//     for (float v : hidden_out_baseline)
//     {
//       hidden_out_sum_baseline += v;
//     }
//     std::cout << "hidden_out_sum_baseline = " << hidden_out_sum_baseline << std::endl;
//
//     float hidden_out_sum_cuda = 0.0f;
//     for (float v : hidden_out_cuda)
//     {
//       hidden_out_sum_cuda += v;
//     }
//     std::cout << "hidden_out_sum_cuda = " << hidden_out_sum_cuda << std::endl;
//
//     float hidden_out_sum_hip = 0.0f;
//     for (float v : hidden_out_hip)
//     {
//       hidden_out_sum_hip += v;
//     }
//     std::cout << "hidden_out_sum_hip = " << hidden_out_sum_hip << std::endl;
//
//     std::cout << std::endl;
//   }
//
//   uint32_t benchmark_iterations = 128;
//   benchmark_layer(hip_layer, "hip_layer", n_hidden, hidden_in, hidden_out_hip, benchmark_iterations);
//   benchmark_layer(cuda_layer, "cuda_layer", n_hidden, hidden_in, hidden_out_cuda, benchmark_iterations);
//   benchmark_layer(baseline_layer, "baseline_layer", n_hidden, hidden_in, hidden_out_baseline, benchmark_iterations);
//
//   delete baseline_layer;
//   delete cuda_layer;
//   delete hip_layer;
// }