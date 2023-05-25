#include <iostream>
#include <algorithm>

uint32_t n_hidden = 4096, n_context = 512, n_layers = 32;

int main()
{

  float *input_q = new float[n_context * n_hidden];
  float *input_k = new float[n_context * n_hidden];
  float *input_v = new float[n_context * n_hidden];

  float *output_before_projection = new float[n_context * n_hidden];

  return 0;
}