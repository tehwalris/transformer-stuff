#include <cuda_runtime.h>
#include <cassert>
#include "cuda.h"

#define CUDA_CHECK(err)                                                         \
  do                                                                            \
  {                                                                             \
    cudaError_t err_ = (err);                                                   \
    if (err_ != cudaSuccess)                                                    \
    {                                                                           \
      fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, \
              cudaGetErrorString(err_));                                        \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

namespace cml
{
  namespace cuda
  {
    const uint32_t n_ff_multiple = 256;
    const uint32_t cache_line_bytes = 64;

    struct Hyperparams
    {
      uint32_t n_hidden;
      uint32_t n_context;
      uint32_t n_heads;
      uint32_t n_ff;
    };

    __global__ void mul_gpu(int n, char4 const *__restrict__ A, char4 const *__restrict__ x, float *__restrict__ y)
    {
      for (int i_row = blockIdx.y * blockDim.y + threadIdx.y; i_row < n; i_row += blockDim.y * gridDim.y)
      {
        int sum = 0;
        for (int i_col = threadIdx.x; i_col < n / 4; i_col += blockDim.x)
        {
          sum = __dp4a(A[(i_row * n) / 4 + i_col], x[i_col], sum);
        }
        atomicAdd(&y[i_row], float(sum));
      }
    }

    __global__ void post_mul_scale_gpu(int n, float *y, float *A_scale, float x_scale)
    {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
      {
        y[i] *= A_scale[i] * x_scale;
      }
    }

    struct QuantizedMatrix
    {
      uint32_t n;
      uint32_t m;
      char4 *data;
      float *scale;
    };

    struct Weights
    {
      QuantizedMatrix q;     // n_hidden * n_hidden
      QuantizedMatrix k;     // n_hidden * n_hidden
      QuantizedMatrix v;     // n_hidden * n_hidden
      QuantizedMatrix o;     // n_hidden * n_hidden
      QuantizedMatrix l1;    // n_ff * n_hidden
      QuantizedMatrix l2;    // n_ff * n_hidden
      QuantizedMatrix l3;    // n_hidden * n_ff
      float *attention_norm; // n_hidden
      float *ff_norm;        // n_hidden
    };

    struct Temp
    {
    };

    struct State
    {
      float *cache_k; // n_context * n_hidden
      float *cache_v; // n_context * n_hidden
      uint32_t new_i;
    };

    class LlamaLayer : public SimpleTransformerLayer
    {
    public:
      LlamaLayer(SimpleLlamaModelLoader *loader, uint32_t layer_index)
      {
        llama_hparams *hparams = loader->get_hparams();
        params.n_hidden = hparams->n_embd;
        params.n_context = hparams->n_ctx;
        params.n_heads = hparams->n_head;
        params.n_ff = ((2 * (4 * params.n_hidden) / 3 + n_ff_multiple - 1) / n_ff_multiple) * n_ff_multiple;

        assert(layer_index < hparams->n_layer);

        auto get_weights = [&](const std::string &short_name, const std::vector<uint32_t> &shape)
        {
          std::string name = "layers." + std::to_string(layer_index) + "." + short_name;

          uint32_t num_elements = 1;
          for (uint32_t v : shape)
          {
            num_elements *= v;
          }

          float *data = aligned_alloc_floats(num_elements);
          float *data_temp = loader->get_tensor_float(name, shape);
          memcpy(data, data_temp, num_elements * sizeof(float));
          return data;
        };

        weights.q = get_weights("attention.wq.weight", {params.n_hidden, params.n_hidden});
        weights.k = get_weights("attention.wk.weight", {params.n_hidden, params.n_hidden});
        weights.v = get_weights("attention.wv.weight", {params.n_hidden, params.n_hidden});
        weights.o = get_weights("attention.wo.weight", {params.n_hidden, params.n_hidden});
        weights.l1 = get_weights("feed_forward.w1.weight", {params.n_ff, params.n_hidden});
        weights.l2 = get_weights("feed_forward.w2.weight", {params.n_hidden, params.n_ff});
        weights.l3 = get_weights("feed_forward.w3.weight", {params.n_ff, params.n_hidden});
        weights.attention_norm = get_weights("attention_norm.weight", {params.n_hidden});
        weights.ff_norm = get_weights("ffn_norm.weight", {params.n_hidden});

        state.cache_k = aligned_alloc_floats(params.n_context * params.n_hidden);
        state.cache_v = aligned_alloc_floats(params.n_context * params.n_hidden);
        state.new_i = 0;
      }

      LlamaLayer(const LlamaLayer &) = delete;

      virtual ~LlamaLayer()
      {
        free(weights.q);
        free(weights.k);
        free(weights.v);
        free(weights.o);
        free(weights.l1);
        free(weights.l2);
        free(weights.l3);
        free(weights.attention_norm);
        free(weights.ff_norm);

        free(state.cache_k);
        free(state.cache_v);
      }

      virtual void forward(int n, float *hidden_in, float *hidden_out) override
      {
        assert(uint32_t(n) == params.n_hidden);
        assert(state.new_i < params.n_context);

        // TODO

        state.new_i++;
      }

      virtual void reset() override
      {
        state.new_i = 0;
      }

    private:
      Hyperparams params;
      Weights weights;
      Temp temp;
      State state;
    };

    SimpleTransformerLayer *create_llama_layer(SimpleLlamaModelLoader *loader, uint32_t layer_index)
    {
      return new LlamaLayer(loader, layer_index);
    }
  };
};