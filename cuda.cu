#include <cuda_runtime.h>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
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

    struct ScaledMulFunctor
    {
      const float scale;

      ScaledMulFunctor(float scale) : scale(scale) {}

      __host__ __device__ float operator()(const float &x, const float &y) const
      {
        return scale * x * y;
      }
    };

    struct QuantizedMatrix
    {
      uint32_t n;
      uint32_t m;
      char4 *data = nullptr;
      float *scale = nullptr;

      QuantizedMatrix() = default;

      QuantizedMatrix(uint32_t n, uint32_t m) : n(n), m(m)
      {
        CUDA_CHECK(cudaMallocManaged(&data, n * (m / 4) * sizeof(char4)));
        CUDA_CHECK(cudaMallocManaged(&scale, n * sizeof(float)));
      }

      QuantizedMatrix(const QuantizedMatrix &) = delete;

      QuantizedMatrix(QuantizedMatrix &&other) : n(other.n), m(other.m), data(other.data), scale(other.scale)
      {
        other.data = nullptr;
        other.scale = nullptr;
      }

      ~QuantizedMatrix()
      {
        if (data != nullptr)
        {
          CUDA_CHECK(cudaFree(data));
        }
        if (scale != nullptr)
        {
          CUDA_CHECK(cudaFree(scale));
        }
      }

      QuantizedMatrix &operator=(QuantizedMatrix &&other) noexcept
      {
        if (this != &other)
        {
          n = other.n;
          m = other.m;
          data = other.data;
          scale = other.scale;
          other.data = nullptr;
          other.scale = nullptr;
        }
        return *this;
      }

      void fill_from_unquantized(const float *unquantized)
      {
        for (uint32_t i = 0; i < n; i++)
        {
          float max_abs = 0;
          for (uint32_t j = 0; j < m; j++)
          {
            float abs = fabs(unquantized[i * m + j]);
            if (abs > max_abs)
            {
              max_abs = abs;
            }
          }

          float scale = max_abs / 127.0f;
          this->scale[i] = scale;
          float inv_scale = 1.0f / scale;

          for (uint32_t j = 0; j < m; j += 4)
          {
            this->data[(i * m + j) / 4].x = (int8_t)(unquantized[i * m + j + 0] * inv_scale);
            this->data[(i * m + j) / 4].y = (int8_t)(unquantized[i * m + j + 1] * inv_scale);
            this->data[(i * m + j) / 4].z = (int8_t)(unquantized[i * m + j + 2] * inv_scale);
            this->data[(i * m + j) / 4].w = (int8_t)(unquantized[i * m + j + 3] * inv_scale);
          }
        }
      }
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
      thrust::device_vector<float> hidden_in;     // n_hidden
      thrust::device_vector<float> hidden_out;    // n_hidden
      thrust::device_vector<float> norm_residual; // n_hidden
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

        auto get_quantized_matrix = [&](const std::string &short_name, uint32_t n, uint32_t m)
        {
          std::string name = "layers." + std::to_string(layer_index) + "." + short_name;

          assert(m % 4 == 0);

          QuantizedMatrix q(n, m);

          q.fill_from_unquantized(loader->get_tensor_float(name, {n, m}));

          return q;
        };

        auto get_vector = [&](const std::string &short_name, uint32_t n)
        {
          std::string name = "layers." + std::to_string(layer_index) + "." + short_name;

          float *data;
          CUDA_CHECK(cudaMallocManaged(&data, n * sizeof(float)));

          float *data_temp = loader->get_tensor_float(name, {n});
          memcpy(data, data_temp, n * sizeof(float));

          return data;
        };

        weights.q = get_quantized_matrix("attention.wq.weight", params.n_hidden, params.n_hidden);
        weights.k = get_quantized_matrix("attention.wk.weight", params.n_hidden, params.n_hidden);
        weights.v = get_quantized_matrix("attention.wv.weight", params.n_hidden, params.n_hidden);
        weights.o = get_quantized_matrix("attention.wo.weight", params.n_hidden, params.n_hidden);
        weights.l1 = get_quantized_matrix("feed_forward.w1.weight", params.n_ff, params.n_hidden);
        weights.l2 = get_quantized_matrix("feed_forward.w2.weight", params.n_hidden, params.n_ff);
        weights.l3 = get_quantized_matrix("feed_forward.w3.weight", params.n_ff, params.n_hidden);
        weights.attention_norm = get_vector("attention_norm.weight", params.n_hidden);
        weights.ff_norm = get_vector("ffn_norm.weight", params.n_hidden);

        temp.hidden_in.resize(params.n_hidden);
        temp.hidden_out.resize(params.n_hidden);
        temp.norm_residual.resize(params.n_hidden);

        CUDA_CHECK(cudaMallocManaged(&state.cache_k, params.n_context * params.n_hidden * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&state.cache_v, params.n_context * params.n_hidden * sizeof(float)));
        state.new_i = 0;

        CUDA_CHECK(cudaDeviceSynchronize());
      }

      LlamaLayer(const LlamaLayer &) = delete;

      virtual ~LlamaLayer()
      {
        free(weights.attention_norm);
        free(weights.ff_norm);

        free(state.cache_k);
        free(state.cache_v);
      }

      virtual void forward(int n, float *hidden_in, float *hidden_out) override
      {
        assert(uint32_t(n) == params.n_hidden);
        assert(state.new_i < params.n_context);

        const float eps = 1e-6f;

        thrust::copy(hidden_in, hidden_in + n, temp.hidden_in.begin());

        // Norm before attention
        float hidden_in_sq_norm = thrust::inner_product(temp.hidden_in.begin(), temp.hidden_in.end(), temp.hidden_in.begin(), 0.0f);
        thrust::transform(temp.hidden_in.begin(), temp.hidden_in.end(),
                          thrust::device_ptr<float>(weights.attention_norm),
                          temp.norm_residual.begin(),
                          ScaledMulFunctor(1.0f / std::sqrt(hidden_in_sq_norm / float(n) + eps)));

        dim3 block_size_mul(32, 8);
        dim3 grid_size_mul(1, (params.n_hidden + block_size_mul.y - 1) / block_size_mul.y);

        int block_size_scale(256);
        int grid_size_scale((params.n_hidden + block_size_scale - 1) / block_size_scale);

        // TODO

        // TODO remove
        thrust::copy(temp.norm_residual.begin(), temp.norm_residual.end(), hidden_out);

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