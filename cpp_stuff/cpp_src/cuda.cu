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

    struct Hyperparams
    {
      uint32_t n_hidden;
      uint32_t n_context;
      uint32_t n_heads;
      uint32_t n_ff;
    };

    __global__ void mul_gpu(int n_rows, int n_cols, char4 const *__restrict__ A, char4 const *__restrict__ x, float *__restrict__ y)
    {
      for (int i_row = blockIdx.y * blockDim.y + threadIdx.y; i_row < n_rows; i_row += blockDim.y * gridDim.y)
      {
        int sum = 0;
        for (int i_col = threadIdx.x; i_col < n_cols / 4; i_col += blockDim.x)
        {
          sum = __dp4a(A[(i_row * n_cols) / 4 + i_col], x[i_col], sum);
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

    __global__ void quantize_gpu(int n, float *input, char4 *output, float quantize_scale)
    {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4; i += blockDim.x * gridDim.x)
      {
        output[i].x = (int8_t)(rint(input[i * 4 + 0] * quantize_scale));
        output[i].y = (int8_t)(rint(input[i * 4 + 1] * quantize_scale));
        output[i].z = (int8_t)(rint(input[i * 4 + 2] * quantize_scale));
        output[i].w = (int8_t)(rint(input[i * 4 + 3] * quantize_scale));
      }
    }

    __global__ void rope_gpu(uint32_t n_hidden, uint32_t n_heads, uint32_t new_i, float *vec)
    {
      for (uint32_t i_head = blockIdx.y * blockDim.y + threadIdx.y; i_head < n_heads; i_head += blockDim.y * gridDim.y)
      {
        for (uint32_t i_in_head = 2 * (blockIdx.x * blockDim.x + threadIdx.x); i_in_head < n_hidden / n_heads; i_in_head += 2 * blockDim.x * gridDim.x)
        {
          float theta = float(new_i) * powf(10000.0f, -float(i_in_head) / (float(n_hidden) / float(n_heads)));
          float cos_theta = cosf(theta);
          float sin_theta = sinf(theta);

          uint32_t i_0 = i_head * (n_hidden / n_heads) + i_in_head;
          uint32_t i_1 = i_0 + 1;

          float old_0 = vec[i_0];
          float old_1 = vec[i_1];

          vec[i_0] = old_0 * cos_theta - old_1 * sin_theta;
          vec[i_1] = old_0 * sin_theta + old_1 * cos_theta;
        }
      }
    }

    __global__ void attention_dot_gpu(uint32_t n_hidden, uint32_t n_heads, uint32_t new_i, float *cache_k, float *new_q, float *attention)
    {
      uint32_t i_context = blockIdx.z;
      uint32_t i_head = blockIdx.y * blockDim.y + threadIdx.y;
      if (i_head < n_heads)
      {
        float sum = 0.0f;
        for (uint32_t i_in_head = threadIdx.x; i_in_head < n_hidden / n_heads; i_in_head += blockDim.x)
        {
          sum += cache_k[i_context * n_hidden + i_head * (n_hidden / n_heads) + i_in_head] * new_q[i_head * (n_hidden / n_heads) + i_in_head];
        }
        atomicAdd(&attention[i_head * (new_i + 1) + i_context], sum);
      }
    }

    __global__ void attention_softmax_gpu(uint32_t new_i, float *attention, float dot_product_scale, float *sum)
    {
      __shared__ float sum_shared[32];
      float sum_local = 0.0f;

      uint32_t i_head = blockIdx.y;
      uint32_t i_context = blockIdx.x * blockDim.x + threadIdx.x;
      if (i_context <= new_i)
      {
        float value = expf(dot_product_scale * attention[i_head * (new_i + 1) + i_context]);
        attention[i_head * (new_i + 1) + i_context] = value;
        sum_local = value;
      }

      sum_shared[threadIdx.x] = sum_local;
      __syncthreads();

      for (uint32_t s = 16; s > 0; s >>= 1)
      {
        if (threadIdx.x < s)
        {
          sum_shared[threadIdx.x] += sum_shared[threadIdx.x + s];
        }
        __syncthreads();
      }

      if (threadIdx.x == 0)
      {
        atomicAdd(sum + i_head, sum_shared[0]);
      }
    }

    __global__ void attention_sum_gpu(uint32_t n_hidden, uint32_t n_heads, uint32_t new_i, float *cache_v, float *attention, float *sum, float *out)
    {
      uint32_t i_head = blockIdx.y;
      uint32_t i_in_head = blockIdx.x * blockDim.x + threadIdx.x;

      if (i_in_head < n_hidden / n_heads)
      {
        float sum_local = 0.0f;
        for (uint32_t i_context = 0; i_context <= new_i; i_context++)
        {
          sum_local += attention[i_head * (new_i + 1) + i_context] / sum[i_head] * cache_v[i_context * n_hidden + i_head * (n_hidden / n_heads) + i_in_head];
        }
        out[i_head * (n_hidden / n_heads) + i_in_head] = sum_local;
      }
    }

    template <typename T>
    T ceil_div(T a, T b)
    {
      return (a + b - 1) / b;
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

    template <typename T>
    struct AbsoluteValueFunctor
    {
      __host__ __device__ T operator()(const T &x) const
      {
        return fabsf(x);
      }
    };

    template <typename T>
    struct SiluMultiplyFunctor
    {
      __host__ __device__ T operator()(const T &x, const T &y) const
      {
        return (x * y) / (1.0f + expf(-y));
      }
    };

    float quantize_gpu_full(thrust::device_vector<float> &unquantized, thrust::device_vector<char4> &quantized)
    {
      uint32_t n = uint32_t(unquantized.size());

      const int block_size(256);
      const int grid_size(ceil_div<uint32_t>(n, 4 * block_size));

      float abs_max = thrust::transform_reduce(unquantized.begin(), unquantized.end(),
                                               AbsoluteValueFunctor<float>(),
                                               0.0f,
                                               thrust::maximum<float>());
      float unquantize_scale = abs_max / 127.0f;
      float quantize_scale = 1.0f / unquantize_scale;
      quantize_gpu<<<grid_size, block_size>>>(n, unquantized.data().get(), quantized.data().get(), quantize_scale);

      return unquantize_scale;
    }

    void rms_norm_gpu_full(thrust::device_vector<float> &in, thrust::device_vector<float> &out, thrust::device_ptr<float> weights)
    {
      assert(in.size() == out.size());

      const float eps = 1e-6f;

      float sq_norm = thrust::inner_product(in.begin(), in.end(), in.begin(), 0.0f);
      thrust::transform(in.begin(), in.end(),
                        weights,
                        out.begin(),
                        ScaledMulFunctor(1.0f / std::sqrt(sq_norm / float(in.size()) + eps)));
    }

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
      thrust::device_vector<float> hidden_in;                  // n_hidden
      thrust::device_vector<float> hidden_out;                 // n_hidden
      thrust::device_vector<float> norm_residual;              // n_hidden
      thrust::device_vector<char4> norm_residual_quantized;    // n_hidden
      thrust::device_vector<float> q;                          // n_hidden
      thrust::device_vector<float> k;                          // n_hidden
      thrust::device_vector<float> v;                          // n_hidden
      thrust::device_vector<float> o;                          // n_hidden
      thrust::device_vector<float> attention;                  // n_context * n_heads
      thrust::device_vector<float> attention_sum;              // n_context
      thrust::device_vector<float> attention_result;           // n_hidden
      thrust::device_vector<char4> attention_result_quantized; // n_hidden
      thrust::device_vector<float> l1;                         // n_ff
      thrust::device_vector<float> l2;                         // n_hidden
      thrust::device_vector<float> l3;                         // n_ff
      thrust::device_vector<char4> l3_quantized;               // n_ff
    };

    struct State
    {
      thrust::device_vector<float> cache_k; // n_context * n_hidden
      thrust::device_vector<float> cache_v; // n_context * n_hidden
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
        assert(params.n_hidden % 4 == 0);
        assert(params.n_ff % 4 == 0);
        assert(params.n_hidden % params.n_heads == 0);
        assert((params.n_hidden / params.n_heads) % 2 == 0);

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
        temp.norm_residual_quantized.resize(params.n_hidden / 4);
        temp.q.resize(params.n_hidden);
        temp.k.resize(params.n_hidden);
        temp.v.resize(params.n_hidden);
        temp.o.resize(params.n_hidden);
        temp.attention.resize(params.n_context * params.n_heads);
        temp.attention_sum.resize(params.n_context);
        temp.attention_result.resize(params.n_hidden);
        temp.attention_result_quantized.resize(params.n_hidden / 4);
        temp.l1.resize(params.n_ff);
        temp.l2.resize(params.n_hidden);
        temp.l3.resize(params.n_ff);
        temp.l3_quantized.resize(params.n_ff / 4);

        state.cache_k.resize(params.n_context * params.n_hidden);
        state.cache_v.resize(params.n_context * params.n_hidden);
        state.new_i = 0;

        CUDA_CHECK(cudaDeviceSynchronize());
      }

      LlamaLayer(const LlamaLayer &) = delete;

      virtual ~LlamaLayer()
      {
        cudaFree(weights.attention_norm);
        cudaFree(weights.ff_norm);
      }

      virtual void forward(const int n_in, const float *hidden_in, const int n_out, float *hidden_out) override
      {
        assert(uint32_t(n_in) == params.n_hidden);
        assert(uint32_t(n_out) == params.n_hidden);
        assert(state.new_i < params.n_context);

        const dim3 block_size_mul_n_hidden(32, 8);
        const dim3 grid_size_mul_n_hidden(1, ceil_div<uint32_t>(params.n_hidden, block_size_mul_n_hidden.y));

        const dim3 block_size_mul_n_ff(32, 8);
        const dim3 grid_size_mul_n_ff(1, ceil_div<uint32_t>(params.n_ff, block_size_mul_n_ff.y));

        const int block_size_scale(256);
        const int grid_size_scale(ceil_div<uint32_t>(params.n_hidden, block_size_scale));

        const int block_size_quantize(256);
        const int grid_size_quantize(ceil_div<uint32_t>(params.n_hidden, block_size_quantize));

        const dim3 block_size_rope(64, 1);
        const dim3 grid_size_rope(ceil_div<uint32_t>(params.n_hidden / params.n_heads / 2, block_size_rope.x), params.n_heads);

        const dim3 block_size_attention_dot(32, 8, 1);
        const dim3 grid_size_attention_dot(1,
                                           ceil_div<uint32_t>(params.n_heads, block_size_attention_dot.y),
                                           state.new_i + 1);

        const dim3 block_size_attention_softmax(32, 1);
        const dim3 grid_size_attention_softmax(ceil_div<uint32_t>(state.new_i + 1, block_size_attention_softmax.x),
                                               params.n_heads);

        const dim3 block_size_attention_sum(32, 1);
        const dim3 grid_size_attention_sum(ceil_div<uint32_t>(params.n_hidden / params.n_heads, block_size_attention_sum.x),
                                           ceil_div<uint32_t>(params.n_heads, block_size_attention_sum.y));

        // Copy to GPU
        thrust::copy(hidden_in, hidden_in + params.n_hidden, temp.hidden_in.begin());

        // Zero accumulators
        thrust::fill(temp.q.begin(), temp.q.end(), 0.0f);
        thrust::fill(temp.k.begin(), temp.k.end(), 0.0f);
        thrust::fill(temp.v.begin(), temp.v.end(), 0.0f);
        thrust::fill(temp.o.begin(), temp.o.end(), 0.0f);
        thrust::fill(temp.attention.begin(), temp.attention.end(), 0.0f);
        thrust::fill(temp.attention_sum.begin(), temp.attention_sum.end(), 0.0f);
        thrust::fill(temp.l1.begin(), temp.l1.end(), 0.0f);
        thrust::fill(temp.l2.begin(), temp.l2.end(), 0.0f);
        thrust::fill(temp.l3.begin(), temp.l3.end(), 0.0f);

        // Norm before attention
        rms_norm_gpu_full(temp.hidden_in, temp.norm_residual, thrust::device_ptr<float>(weights.attention_norm));

        // Compute Q, K, V
        float qkv_unquantize_scale = quantize_gpu_full(temp.norm_residual, temp.norm_residual_quantized);
        mul_gpu<<<grid_size_mul_n_hidden, block_size_mul_n_hidden>>>(params.n_hidden, params.n_hidden, weights.q.data, temp.norm_residual_quantized.data().get(), temp.q.data().get());
        post_mul_scale_gpu<<<grid_size_scale, block_size_scale>>>(params.n_hidden, temp.q.data().get(), weights.q.scale, qkv_unquantize_scale);
        mul_gpu<<<grid_size_mul_n_hidden, block_size_mul_n_hidden>>>(params.n_hidden, params.n_hidden, weights.k.data, temp.norm_residual_quantized.data().get(), temp.k.data().get());
        post_mul_scale_gpu<<<grid_size_scale, block_size_scale>>>(params.n_hidden, temp.k.data().get(), weights.k.scale, qkv_unquantize_scale);
        mul_gpu<<<grid_size_mul_n_hidden, block_size_mul_n_hidden>>>(params.n_hidden, params.n_hidden, weights.v.data, temp.norm_residual_quantized.data().get(), temp.v.data().get());
        post_mul_scale_gpu<<<grid_size_scale, block_size_scale>>>(params.n_hidden, temp.v.data().get(), weights.v.scale, qkv_unquantize_scale);

        // Apply RoPE
        rope_gpu<<<grid_size_rope, block_size_rope>>>(params.n_hidden, params.n_heads, state.new_i, temp.q.data().get());
        rope_gpu<<<grid_size_rope, block_size_rope>>>(params.n_hidden, params.n_heads, state.new_i, temp.k.data().get());

        // Copy the new KV to the cache
        thrust::copy(temp.k.begin(), temp.k.end(), state.cache_k.begin() + state.new_i * params.n_hidden);
        thrust::copy(temp.v.begin(), temp.v.end(), state.cache_v.begin() + state.new_i * params.n_hidden);

        // Calculate the dot product with each cached K (per head)
        attention_dot_gpu<<<grid_size_attention_dot, block_size_attention_dot>>>(params.n_hidden, params.n_heads, state.new_i, state.cache_k.data().get(), temp.q.data().get(), temp.attention.data().get());

        // Softmax (except divide)
        float dot_product_scale = 1.0f / std::sqrt(float(params.n_hidden / params.n_heads));
        attention_softmax_gpu<<<grid_size_attention_softmax, block_size_attention_softmax>>>(state.new_i, temp.attention.data().get(), dot_product_scale, temp.attention_sum.data().get());

        // Sum V weighted by softmax attention
        attention_sum_gpu<<<grid_size_attention_sum, block_size_attention_sum>>>(params.n_hidden, params.n_heads, state.new_i, state.cache_v.data().get(), temp.attention.data().get(), temp.attention_sum.data().get(), temp.attention_result.data().get());

        // Projection and residual
        float projection_unquantize_scale = quantize_gpu_full(temp.attention_result, temp.attention_result_quantized);
        mul_gpu<<<grid_size_mul_n_hidden, block_size_mul_n_hidden>>>(params.n_hidden, params.n_hidden, weights.o.data, temp.attention_result_quantized.data().get(), temp.o.data().get());
        post_mul_scale_gpu<<<grid_size_scale, block_size_scale>>>(params.n_hidden, temp.o.data().get(), weights.o.scale, projection_unquantize_scale);
        thrust::transform(temp.o.begin(), temp.o.end(), temp.hidden_in.begin(), temp.o.begin(), thrust::plus<float>());

        // Norm before feed forward
        rms_norm_gpu_full(temp.o, temp.norm_residual, thrust::device_ptr<float>(weights.ff_norm));

        // Feed forward (up)
        float ff_up_unquantize_scale = quantize_gpu_full(temp.norm_residual, temp.norm_residual_quantized);
        mul_gpu<<<grid_size_mul_n_ff, block_size_mul_n_ff>>>(params.n_ff, params.n_hidden, weights.l1.data, temp.norm_residual_quantized.data().get(), temp.l1.data().get());
        post_mul_scale_gpu<<<grid_size_scale, block_size_scale>>>(params.n_ff, temp.l1.data().get(), weights.l1.scale, ff_up_unquantize_scale);
        mul_gpu<<<grid_size_mul_n_ff, block_size_mul_n_ff>>>(params.n_ff, params.n_hidden, weights.l3.data, temp.norm_residual_quantized.data().get(), temp.l3.data().get());
        post_mul_scale_gpu<<<grid_size_scale, block_size_scale>>>(params.n_ff, temp.l3.data().get(), weights.l3.scale, ff_up_unquantize_scale);

        // Feed forward (silu)
        thrust::transform(temp.l3.begin(), temp.l3.end(), temp.l1.begin(), temp.l3.begin(), SiluMultiplyFunctor<float>());

        // Feed forward (down)
        float ff_down_unquantize_scale = quantize_gpu_full(temp.l3, temp.l3_quantized);
        mul_gpu<<<grid_size_mul_n_hidden, block_size_mul_n_hidden>>>(params.n_hidden, params.n_ff, weights.l2.data, temp.l3_quantized.data().get(), temp.l2.data().get());
        post_mul_scale_gpu<<<grid_size_scale, block_size_scale>>>(params.n_hidden, temp.l2.data().get(), weights.l2.scale, ff_down_unquantize_scale);

        // Residual
        thrust::transform(temp.l2.begin(), temp.l2.end(), temp.o.begin(), temp.l2.begin(), thrust::plus<float>());

        // Copy from GPU
        thrust::copy(temp.l2.begin(), temp.l2.end(), hidden_out);

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

    __attribute__((visibility("default")))
    SimpleTransformerLayer *
    create_llama_layer(SimpleLlamaModelLoader *loader, uint32_t layer_index)
    {
      return new LlamaLayer(loader, layer_index);
    }
  };
};