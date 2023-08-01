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

    __global__ void attention_dot_gpu(uint32_t n_hidden, uint32_t n_heads, uint32_t n_path, const uint32_t *path, float *cache_k, float *new_q, float *attention)
    {
      uint32_t i_path = blockIdx.z;
      uint32_t i_head = blockIdx.y * blockDim.y + threadIdx.y;
      if (i_head < n_heads)
      {
        uint32_t i_context = path[i_path];
        float sum = 0.0f;
        for (uint32_t i_in_head = threadIdx.x; i_in_head < n_hidden / n_heads; i_in_head += blockDim.x)
        {
          sum += cache_k[i_context * n_hidden + i_head * (n_hidden / n_heads) + i_in_head] * new_q[i_head * (n_hidden / n_heads) + i_in_head];
        }
        atomicAdd(&attention[i_head * n_path + i_path], sum);
      }
    }

    __global__ void attention_softmax_gpu(uint32_t n_path, float *attention, float dot_product_scale, float *sum)
    {
      __shared__ float sum_shared[32];
      float sum_local = 0.0f;

      uint32_t i_head = blockIdx.y;
      uint32_t i_path = blockIdx.x * blockDim.x + threadIdx.x;
      if (i_path < n_path)
      {
        float value = expf(dot_product_scale * attention[i_head * n_path + i_path]);
        attention[i_head * n_path + i_path] = value;
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

    __global__ void attention_sum_gpu(uint32_t n_hidden, uint32_t n_heads, uint32_t n_path, const uint32_t *path, float *cache_v, float *attention, float *sum, float *out)
    {
      uint32_t i_head = blockIdx.y;
      uint32_t i_in_head = blockIdx.x * blockDim.x + threadIdx.x;

      if (i_in_head < n_hidden / n_heads)
      {
        float sum_local = 0.0f;
        for (uint32_t i_path = 0; i_path < n_path; i_path++)
        {
          uint32_t i_context = path[i_path];
          sum_local += attention[i_head * n_path + i_path] / sum[i_head] * cache_v[i_context * n_hidden + i_head * (n_hidden / n_heads) + i_in_head];
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

    __global__ void mul_gpu_new(const int n_rows, const int n_cols, const int block_size, const uint32_t *__restrict__ qweight, const uint32_t *__restrict__ qzeros, const half *__restrict__ scales, const float *__restrict__ x, float *__restrict__ y)
    {
      for (int i_row = blockIdx.y * blockDim.y + threadIdx.y; i_row < n_rows; i_row += blockDim.y * gridDim.y)
      {
        float sum = 0;
        for (int i_col = threadIdx.x; i_col < n_cols; i_col += blockDim.x)
        {
          uint32_t i_qweight = (i_col / 8) * n_rows + i_row;
          uint32_t i_qzeros = (i_col / block_size) * (n_rows / 8) + (i_row / 8);
          uint32_t i_scales = (i_col / block_size) * n_rows + i_row;

          float scale = __half2float(scales[i_scales]);

          uint32_t zero_quant_group = qzeros[i_qzeros];
          uint32_t zero_quant = (zero_quant_group >> (4 * (i_row % 8))) & 0xf;
          float zero = float(zero_quant + 1) * scale;

          uint32_t weight_quant_group = qweight[i_qweight];
          uint32_t weight_quant = (weight_quant_group >> (4 * (i_col % 8))) & 0xf;
          float weight = float(weight_quant) * scale;

          sum += (weight - zero) * x[i_col];
        }
        atomicAdd(&y[i_row], float(sum));
      }
    }

    struct GPTQMatrixGPU
    {
      int rows;
      int cols;
      int block_size;

      uint32_t *qweight;
      uint32_t *qzeros;
      half *scales;
    };

    void mul_gpu_full(const GPTQMatrixGPU &A, const float *__restrict__ x, float *__restrict__ y)
    {
      assert(A.rows % A.block_size == 0);
      assert(A.cols % A.block_size == 0);

      const dim3 block_size(32, 8);
      const dim3 grid_size(1, ceil_div<uint32_t>(A.rows, block_size.y));

      mul_gpu_new<<<grid_size, block_size>>>(A.rows, A.cols, A.block_size, A.qweight, A.qzeros, A.scales, x, y);
    }

    GPTQMatrixGPU copy_gptq_matrix_gpu(const GPTQMatrix &old_mat)
    {
      assert(old_mat.block_size % 8 == 0);
      assert(old_mat.rows % old_mat.block_size == 0);
      assert(old_mat.cols % old_mat.block_size == 0);

      GPTQMatrixGPU new_mat;
      new_mat.rows = int(old_mat.rows);
      new_mat.cols = int(old_mat.cols);
      new_mat.block_size = int(old_mat.block_size);

      size_t qweight_bytes = (old_mat.cols / 8) * old_mat.rows * sizeof(uint32_t);
      size_t qzeros_bytes = (old_mat.cols / old_mat.block_size) * (old_mat.rows / 8) * sizeof(uint32_t);
      size_t scales_bytes = (old_mat.cols / old_mat.block_size) * old_mat.rows * sizeof(half);

      CUDA_CHECK(cudaMalloc(&new_mat.qweight, qweight_bytes));
      CUDA_CHECK(cudaMalloc(&new_mat.qzeros, qzeros_bytes));
      CUDA_CHECK(cudaMalloc(&new_mat.scales, scales_bytes));

      CUDA_CHECK(cudaMemcpy(new_mat.qweight, old_mat.qweight, qweight_bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(new_mat.qzeros, old_mat.qzeros, qzeros_bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(new_mat.scales, old_mat.scales, scales_bytes, cudaMemcpyHostToDevice));

      return new_mat;
    }

    void free_gptq_matrix_gpu(GPTQMatrixGPU &mat)
    {
      CUDA_CHECK(cudaFree(mat.qweight));
      CUDA_CHECK(cudaFree(mat.qzeros));
      CUDA_CHECK(cudaFree(mat.scales));

      mat.qweight = nullptr;
      mat.qzeros = nullptr;
      mat.scales = nullptr;
    }

    struct Weights
    {
      GPTQMatrixGPU q;       // n_hidden * n_hidden
      GPTQMatrixGPU k;       // n_hidden * n_hidden
      GPTQMatrixGPU v;       // n_hidden * n_hidden
      GPTQMatrixGPU o;       // n_hidden * n_hidden
      GPTQMatrixGPU l1;      // n_ff * n_hidden
      GPTQMatrixGPU l2;      // n_hidden * n_ff
      GPTQMatrixGPU l3;      // n_ff * n_hidden
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
      thrust::device_vector<uint32_t> path;                    // n_context
    };

    struct State
    {
      thrust::device_vector<float> cache_k; // n_cache * n_hidden
      thrust::device_vector<float> cache_v; // n_cache * n_hidden
      uint32_t new_i;
    };

    class LlamaLayer : public SimpleTransformerLayer
    {
    public:
      LlamaLayer(const LlamaGPTQLayerWeights *loader_weights, LlamaHyperparams params, uint32_t n_cache) : params(params), n_cache(n_cache)
      {
        assert(params.n_hidden % 4 == 0);
        assert(params.n_ff % 4 == 0);
        assert(params.n_hidden % params.n_heads == 0);
        assert((params.n_hidden / params.n_heads) % 2 == 0);
        assert(params.gptq_block_size % 8 == 0);
        assert(params.n_hidden % params.gptq_block_size == 0);
        assert(params.n_ff % params.gptq_block_size == 0);
        assert(params.n_vocab % params.gptq_block_size == 0);

        auto get_weight_matrix = [&](const GPTQMatrix &mat, const std::vector<uint32_t> &shape)
        {
          assert(shape.size() == 2);
          assert(mat.rows == shape[0]);
          assert(mat.cols == shape[1]);
          return copy_gptq_matrix_gpu(mat);
        };

        auto get_1d_weights = [&](const uint16_t *values, uint32_t n)
        {
          assert(n % 2 == 0);

          float *converted_values = new float[n];
          for (uint32_t i = 0; i < n; i++)
          {
            converted_values[i] = __half2float(values[i]);
          }

          float *copied_values;
          CUDA_CHECK(cudaMalloc(&copied_values, n * sizeof(float)));
          CUDA_CHECK(cudaMemcpy(copied_values, converted_values, n * sizeof(float), cudaMemcpyHostToDevice));

          delete[] converted_values;

          return copied_values;
        };

        weights.q = get_weight_matrix(loader_weights->self_attn_q_proj, {params.n_hidden, params.n_hidden});
        weights.k = get_weight_matrix(loader_weights->self_attn_k_proj, {params.n_hidden, params.n_hidden});
        weights.v = get_weight_matrix(loader_weights->self_attn_v_proj, {params.n_hidden, params.n_hidden});
        weights.o = get_weight_matrix(loader_weights->self_attn_o_proj, {params.n_hidden, params.n_hidden});
        weights.l1 = get_weight_matrix(loader_weights->mlp_gate_proj, {params.n_ff, params.n_hidden});
        weights.l2 = get_weight_matrix(loader_weights->mlp_down_proj, {params.n_hidden, params.n_ff});
        weights.l3 = get_weight_matrix(loader_weights->mlp_up_proj, {params.n_ff, params.n_hidden});
        weights.attention_norm = get_1d_weights(loader_weights->input_layernorm, params.n_hidden);
        weights.ff_norm = get_1d_weights(loader_weights->post_attention_layernorm, params.n_hidden);

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
        temp.path.resize(params.n_context);

        state.cache_k.resize(n_cache * params.n_hidden);
        state.cache_v.resize(n_cache * params.n_hidden);
        state.new_i = 0;

        CUDA_CHECK(cudaDeviceSynchronize());
      }

      LlamaLayer(const LlamaLayer &) = delete;

      virtual ~LlamaLayer()
      {
        CUDA_CHECK(cudaFree(weights.attention_norm));
        CUDA_CHECK(cudaFree(weights.ff_norm));
      }

      virtual void forward(const int n_in, const float *hidden_in, const int n_out, float *hidden_out, const uint32_t n_path, const uint32_t *path) override
      {
        assert(uint32_t(n_in) == params.n_hidden);
        assert(uint32_t(n_out) == params.n_hidden);
        assert(state.new_i < n_cache);
        assert(n_path > 0);
        assert(n_path <= state.new_i + 1);
        assert(n_path <= params.n_context);
        assert(path[n_path - 1] == state.new_i);

        const dim3 block_size_mul_n_hidden(32, 8);
        const dim3 grid_size_mul_n_hidden(1, ceil_div<uint32_t>(params.n_hidden, block_size_mul_n_hidden.y));

        const dim3 block_size_mul_n_ff(32, 8);
        const dim3 grid_size_mul_n_ff(1, ceil_div<uint32_t>(params.n_ff, block_size_mul_n_ff.y));

        const int block_size_scale(256);
        const int grid_size_scale(ceil_div<uint32_t>(params.n_hidden, block_size_scale));

        const dim3 block_size_rope(64, 1);
        const dim3 grid_size_rope(ceil_div<uint32_t>(params.n_hidden / params.n_heads / 2, block_size_rope.x), params.n_heads);

        const dim3 block_size_attention_dot(32, 8, 1);
        const dim3 grid_size_attention_dot(1,
                                           ceil_div<uint32_t>(params.n_heads, block_size_attention_dot.y),
                                           n_path);

        const dim3 block_size_attention_softmax(32, 1);
        const dim3 grid_size_attention_softmax(ceil_div<uint32_t>(n_path, block_size_attention_softmax.x),
                                               params.n_heads);

        const dim3 block_size_attention_sum(32, 1);
        const dim3 grid_size_attention_sum(ceil_div<uint32_t>(params.n_hidden / params.n_heads, block_size_attention_sum.x),
                                           ceil_div<uint32_t>(params.n_heads, block_size_attention_sum.y));

        // Copy to GPU
        thrust::copy(hidden_in, hidden_in + params.n_hidden, temp.hidden_in.begin());
        thrust::copy(path, path + n_path, temp.path.begin());

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
        mul_gpu_full(weights.q, temp.norm_residual.data().get(), temp.q.data().get());
        mul_gpu_full(weights.k, temp.norm_residual.data().get(), temp.k.data().get());
        mul_gpu_full(weights.v, temp.norm_residual.data().get(), temp.v.data().get());

        // Apply RoPE
        rope_gpu<<<grid_size_rope, block_size_rope>>>(params.n_hidden, params.n_heads, n_path - 1, temp.q.data().get());
        rope_gpu<<<grid_size_rope, block_size_rope>>>(params.n_hidden, params.n_heads, n_path - 1, temp.k.data().get());

        // Copy the new KV to the cache
        thrust::copy(temp.k.begin(), temp.k.end(), state.cache_k.begin() + state.new_i * params.n_hidden);
        thrust::copy(temp.v.begin(), temp.v.end(), state.cache_v.begin() + state.new_i * params.n_hidden);

        // Calculate the dot product with each cached K (per head)
        attention_dot_gpu<<<grid_size_attention_dot, block_size_attention_dot>>>(params.n_hidden, params.n_heads, n_path, temp.path.data().get(), state.cache_k.data().get(), temp.q.data().get(), temp.attention.data().get());

        // Softmax (except divide)
        float dot_product_scale = 1.0f / std::sqrt(float(params.n_hidden / params.n_heads));
        attention_softmax_gpu<<<grid_size_attention_softmax, block_size_attention_softmax>>>(n_path, temp.attention.data().get(), dot_product_scale, temp.attention_sum.data().get());

        // Sum V weighted by softmax attention
        attention_sum_gpu<<<grid_size_attention_sum, block_size_attention_sum>>>(params.n_hidden, params.n_heads, n_path, temp.path.data().get(), state.cache_v.data().get(), temp.attention.data().get(), temp.attention_sum.data().get(), temp.attention_result.data().get());

        // Projection and residual
        mul_gpu_full(weights.o, temp.attention_result.data().get(), temp.o.data().get());
        thrust::transform(temp.o.begin(), temp.o.end(), temp.hidden_in.begin(), temp.o.begin(), thrust::plus<float>());

        // Norm before feed forward
        rms_norm_gpu_full(temp.o, temp.norm_residual, thrust::device_ptr<float>(weights.ff_norm));

        // Feed forward (up)
        mul_gpu_full(weights.l1, temp.norm_residual.data().get(), temp.l1.data().get());
        mul_gpu_full(weights.l3, temp.norm_residual.data().get(), temp.l3.data().get());

        // Feed forward (silu)
        thrust::transform(temp.l3.begin(), temp.l3.end(), temp.l1.begin(), temp.l3.begin(), SiluMultiplyFunctor<float>());

        // Feed forward (down)
        mul_gpu_full(weights.l2, temp.l3.data().get(), temp.l2.data().get());

        // Residual
        thrust::transform(temp.l2.begin(), temp.l2.end(), temp.o.begin(), temp.l2.begin(), thrust::plus<float>());

        // Copy from GPU
        thrust::copy(temp.l2.begin(), temp.l2.end(), hidden_out);

        state.new_i++;
      }

      virtual uint32_t next_i() const override
      {
        return state.new_i;
      }

      virtual void retain(const uint32_t n_retain, const uint32_t *retain) override
      {
        assert(n_retain <= state.new_i);
        assert(n_retain <= n_cache);

        thrust::device_vector<float> old_cache_k(n_cache * params.n_hidden);
        thrust::device_vector<float> old_cache_v(n_cache * params.n_hidden);

        old_cache_k.swap(state.cache_k);
        old_cache_v.swap(state.cache_v);

        for (uint32_t i = 0; i < n_retain; i++)
        {
          thrust::copy(old_cache_k.begin() + retain[i] * params.n_hidden, old_cache_k.begin() + (retain[i] + 1) * params.n_hidden, state.cache_k.begin() + i * params.n_hidden);
          thrust::copy(old_cache_v.begin() + retain[i] * params.n_hidden, old_cache_v.begin() + (retain[i] + 1) * params.n_hidden, state.cache_v.begin() + i * params.n_hidden);
        }

        state.new_i = n_retain;
      }

    private:
      LlamaHyperparams params;
      uint32_t n_cache;
      Weights weights;
      Temp temp;
      State state;
    };

    class LlamaFinalLayer : public SimpleTransformerLayer
    {
    public:
      LlamaFinalLayer(SimpleLlamaModelLoader *loader)
      {
        llama_hparams *hparams = loader->get_hparams();
        n_hidden = hparams->n_embd;
        n_vocab = hparams->n_vocab;

        assert(n_hidden % 4 == 0);

        CUDA_CHECK(cudaMallocManaged(&weights_model_norm, n_hidden * sizeof(float)));
        float *data_temp = loader->get_tensor_float("norm.weight", {n_hidden});
        memcpy(weights_model_norm, data_temp, n_hidden * sizeof(float));

        // weights_output_layer = QuantizedMatrix(n_vocab, n_hidden);
        // weights_output_layer.fill_from_unquantized(loader->get_tensor_float("output.weight", {n_vocab, n_hidden}));

        temp_hidden_in.resize(n_hidden);
        temp_hidden_out.resize(n_vocab);
        temp_model_norm.resize(n_hidden);
        temp_model_norm_quantized.resize(n_hidden);

        CUDA_CHECK(cudaDeviceSynchronize());
      }

      LlamaFinalLayer(const LlamaFinalLayer &) = delete;

      virtual ~LlamaFinalLayer()
      {
        CUDA_CHECK(cudaFree(weights_model_norm));
      }

      virtual void forward(const int n_in, const float *hidden_in, const int n_out, float *hidden_out, const uint32_t n_path, const uint32_t *path) override
      {
        assert(uint32_t(n_in) == n_hidden);
        assert(uint32_t(n_out) == n_vocab);
        assert(n_path > 0);
        assert(n_path <= new_i + 1);
        assert(path[n_path - 1] == new_i);

        const dim3 block_size_mul_n_vocab(32, 8);
        const dim3 grid_size_mul_n_vocab(1, ceil_div<uint32_t>(n_vocab, block_size_mul_n_vocab.y));

        const int block_size_scale(256);
        const int grid_size_scale(ceil_div<uint32_t>(n_hidden, block_size_scale));

        // Copy to GPU
        thrust::copy(hidden_in, hidden_in + n_hidden, temp_hidden_in.begin());

        // Zero accumulators
        thrust::fill(temp_hidden_out.begin(), temp_hidden_out.end(), 0.0f);

        // Norm before output layer
        // rms_norm_gpu_full(temp_hidden_in, temp_model_norm, thrust::device_ptr<float>(weights_model_norm));

        // Output layer
        // float unquantize_scale = quantize_gpu_full(temp_model_norm, temp_model_norm_quantized);
        // mul_gpu<<<grid_size_mul_n_vocab, block_size_mul_n_vocab>>>(n_vocab, n_hidden, weights_output_layer.data, temp_model_norm_quantized.data().get(), temp_hidden_out.data().get());
        // post_mul_scale_gpu<<<grid_size_scale, block_size_scale>>>(n_vocab, temp_hidden_out.data().get(), weights_output_layer.scale, unquantize_scale);

        // Copy from GPU
        thrust::copy(temp_hidden_out.begin(), temp_hidden_out.end(), hidden_out);

        new_i++;
      }

      virtual uint32_t next_i() const override
      {
        return new_i;
      }

      virtual void retain(const uint32_t n_retain, const uint32_t *retain) override
      {
        assert(n_retain <= new_i);
        new_i = n_retain;
      }

    private:
      uint32_t n_hidden;
      uint32_t n_vocab;
      uint32_t new_i;
      thrust::device_vector<float> temp_hidden_in;
      thrust::device_vector<float> temp_hidden_out;
      thrust::device_vector<float> temp_model_norm;
      thrust::device_vector<char4> temp_model_norm_quantized;
      float *weights_model_norm;
      // QuantizedMatrix weights_output_layer;
    };

    __attribute__((visibility("default")))
    SimpleTransformerLayer *
    create_llama_layer_gptq(const LlamaGPTQLayerWeights *loader_weights, LlamaHyperparams params, uint32_t n_cache)
    {
      return new LlamaLayer(loader_weights, params, n_cache);
    }

    __attribute__((visibility("default")))
    SimpleTransformerLayer *
    create_llama_final_layer(SimpleLlamaModelLoader *loader)
    {
      return new LlamaFinalLayer(loader);
    }
  };
};