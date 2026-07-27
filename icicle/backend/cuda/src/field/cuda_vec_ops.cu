#include <cuda.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "icicle/errors.h"
#include "icicle/backend/vec_ops_backend.h"
#include "gpu-utils/error_handler.h"
#include "error_translation.h"
#include "gpu-utils/utils.h"

#include "icicle/fields/field_config.h"
using namespace field_config;

#define MAX_THREADS_PER_BLOCK 256

template <typename E, void (*Kernel)(const E*, const E*, uint64_t, E*)>
cudaError_t vec_op(
  const E* a, const E* b, uint64_t size_a, uint64_t size_b, const VecOpsConfig& config, E* result, uint64_t size_res)
{
  CHK_INIT_IF_RETURN();

  size_a *= config.batch_size;
  size_b *= config.batch_size;
  size_res *= config.batch_size;

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  a = config.is_a_on_device ? a : allocate_and_copy_to_device(a, size_a * sizeof(E), cuda_stream);
  b = config.is_b_on_device ? b : allocate_and_copy_to_device(b, size_b * sizeof(E), cuda_stream);
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(size_res * sizeof(E), cuda_stream);

  // Call the kernel to perform element-wise operation
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (size_res + num_threads - 1) / num_threads;
  Kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(a, b, size_res, d_result);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, size_res * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)a, cuda_stream)); }
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)b, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E, void (*Kernel)(const E*, const E*, uint64_t, uint64_t, bool, E*)>
cudaError_t vec_scalar_op(
  const E* scalar,
  const E* vec,
  uint64_t scalar_size,
  uint64_t vec_size,
  const VecOpsConfig& config,
  E* result,
  uint64_t res_size)
{
  CHK_INIT_IF_RETURN();

  res_size *= config.batch_size;

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  // scalar is always a single element (not an array)
  scalar = config.is_a_on_device ? scalar : allocate_and_copy_to_device(scalar, sizeof(E), cuda_stream);
  vec = config.is_b_on_device ? vec
                              : allocate_and_copy_to_device(vec, vec_size * config.batch_size * sizeof(E), cuda_stream);
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(res_size * sizeof(E), cuda_stream);

  // Call the kernel to perform element-wise operation
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (res_size + num_threads - 1) / num_threads;
  Kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
    scalar, vec, vec_size, config.batch_size, config.columns_batch, d_result);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, res_size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)scalar, cuda_stream)); }
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)vec, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E, void (*Kernel)(const E*, uint64_t, uint64_t, bool, E*)>
cudaError_t vec_unary_op(const E* input, uint64_t size, const VecOpsConfig& config, E* result)
{
  CHK_INIT_IF_RETURN();

  uint64_t total_size = size * config.batch_size;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  input = config.is_a_on_device ? input : allocate_and_copy_to_device(input, total_size * sizeof(E), cuda_stream);
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(total_size * sizeof(E), cuda_stream);

  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (total_size + num_threads - 1) / num_threads;
  Kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(input, size, config.batch_size, config.columns_batch, d_result);

  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, total_size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)input, cuda_stream)); }

  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E>
__global__ void batch_inverse_ranges_kernel(const E* input, E* output, uint64_t total_size, uint64_t range_size)
{
  uint64_t range_idx = blockIdx.x;
  uint64_t start = range_idx * range_size;
  if (start >= total_size || threadIdx.x != 0) { return; }

  const uint64_t end = min(start + range_size, total_size);
  output[start] = input[start];
  for (uint64_t i = start + 1; i < end; ++i) {
    output[i] = output[i - 1] * input[i];
  }

  E acc = E::inverse(output[end - 1]);
  for (uint64_t i = end - 1; i > start; --i) {
    output[i] = acc * output[i - 1];
    acc = acc * input[i];
  }
  output[start] = acc;
}

template <typename E>
__global__ void batch_inverse_ranges_inplace_kernel(E* input_output, uint64_t total_size, uint64_t range_size)
{
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  E* original = reinterpret_cast<E*>(shared_bytes);

  uint64_t range_idx = blockIdx.x;
  uint64_t start = range_idx * range_size;
  if (start >= total_size || threadIdx.x != 0) { return; }

  const uint64_t end = min(start + range_size, total_size);
  const uint64_t range_len = end - start;

  for (uint64_t i = 0; i < range_len; ++i) {
    original[i] = input_output[start + i];
  }

  input_output[start] = original[0];
  for (uint64_t i = 1; i < range_len; ++i) {
    input_output[start + i] = input_output[start + i - 1] * original[i];
  }

  E acc = E::inverse(input_output[end - 1]);
  for (uint64_t i = range_len - 1; i > 0; --i) {
    input_output[start + i] = acc * input_output[start + i - 1];
    acc = acc * original[i];
  }
  input_output[start] = acc;
}

template <typename E>
cudaError_t vec_batch_inverse_op(const E* input, uint64_t size, const VecOpsConfig& config, E* result)
{
  CHK_INIT_IF_RETURN();

  uint64_t total_size = size * config.batch_size;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
  if (total_size == 0) { return cudaSuccess; }

  const E* d_input =
    config.is_a_on_device ? input : allocate_and_copy_to_device(input, total_size * sizeof(E), cuda_stream);
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(total_size * sizeof(E), cuda_stream);

  if (config.is_a_on_device && config.is_result_on_device && input == result) {
    constexpr uint64_t max_shared_bytes = 32 * 1024;
    uint64_t range_size = std::max<uint64_t>(1, max_shared_bytes / sizeof(E));
    uint64_t num_ranges = (total_size + range_size - 1) / range_size;
    batch_inverse_ranges_inplace_kernel<<<num_ranges, 32, range_size * sizeof(E), cuda_stream>>>(
      d_result, total_size, range_size);
  } else {
    uint64_t num_ranges = std::min<uint64_t>(total_size, 4096);
    uint64_t range_size = (total_size + num_ranges - 1) / num_ranges;
    batch_inverse_ranges_kernel<<<num_ranges, 32, 0, cuda_stream>>>(d_input, d_result, total_size, range_size);
  }

  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, total_size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)d_input, cuda_stream)); }

  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

/*============================== add ==============================*/
template <typename E>
__global__ void add_kernel(const E* element_vec1, const E* element_vec2, uint64_t size, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { result[tid] = element_vec1[tid] + element_vec2[tid]; }
}

template <typename E>
eIcicleError
add_cuda(const Device& device, const E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, add_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

/*============================== sub ==============================*/
template <typename E>
__global__ void sub_kernel(const E* element_vec1, const E* element_vec2, uint64_t size, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { result[tid] = element_vec1[tid] - element_vec2[tid]; }
}

template <typename E>
eIcicleError
sub_cuda(const Device& device, const E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, sub_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

/*============================== mul ==============================*/
template <typename E>
__global__ void mul_kernel(const E* vec_a, const E* vec_b, uint64_t size, E* result)
{
  uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) { result[tid] = vec_a[tid] * vec_b[tid]; }
}

template <typename E>
eIcicleError
mul_cuda(const Device& device, const E* vec_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, mul_kernel>(vec_a, vec_b, size, size, config, result, size);
  return translateCudaError(err);
}

/*============================== shift ==============================*/
template <typename E>
__global__ void shift_kernel(const E* input, uint64_t vec_size, uint64_t nof_vecs, bool columns_batch, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t total_size = vec_size * nof_vecs;
  if (tid >= total_size || vec_size == 0) { return; }

  uint64_t src_idx = 0;
  if (columns_batch) {
    const uint64_t batch_idx = tid % nof_vecs;
    const uint64_t vec_idx = tid / nof_vecs;
    const uint64_t src_vec_idx = (vec_idx + 1) % vec_size;
    src_idx = src_vec_idx * nof_vecs + batch_idx;
  } else {
    const uint64_t batch_idx = tid / vec_size;
    const uint64_t vec_idx = tid % vec_size;
    const uint64_t src_vec_idx = (vec_idx + 1) % vec_size;
    src_idx = batch_idx * vec_size + src_vec_idx;
  }

  result[tid] = input[src_idx];
}

template <typename E>
eIcicleError shift_cuda(const Device& device, const E* input, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_unary_op<E, shift_kernel>(input, size, config, result);
  return translateCudaError(err);
}

/*============================== batch inverse ==============================*/
template <typename E>
eIcicleError
batch_inverse_cuda(const Device& device, const E* input, uint64_t size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_batch_inverse_op<E>(input, size, config, result);
  return translateCudaError(err);
}

template <typename E>
__global__ void mul_scalar_kernel(
  const E* scalar, const E* element_vec, uint64_t vec_size, uint64_t nof_vecs, bool columns_batch, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < vec_size * nof_vecs) {
    // scalar is always a single element, use it for all vector elements
    E scalar_val = *scalar;
    result[tid] = element_vec[tid] * scalar_val;
  }
}

template <typename E>
eIcicleError mul_scalar_cuda(
  const Device& device, const E* scalar_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  // scalar_a is always a single element (not an array)
  cudaError_t err = vec_scalar_op<E, mul_scalar_kernel>(scalar_a, vec_b, 1, size, config, result, size);
  return translateCudaError(err);
}

template <typename E>
__global__ void add_scalar_kernel(
  const E* scalar, const E* element_vec, uint64_t vec_size, uint64_t nof_vecs, bool columns_batch, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < vec_size * nof_vecs) {
    // scalar is always a single element, use it for all vector elements
    E scalar_val = *scalar;
    result[tid] = element_vec[tid] + scalar_val;
  }
}

template <typename E>
eIcicleError add_scalar_cuda(
  const Device& device, const E* scalar_a, const E* vec_b, uint64_t size, const VecOpsConfig& config, E* result)
{
  // scalar_a is always a single element (not an array)
  cudaError_t err = vec_scalar_op<E, add_scalar_kernel>(scalar_a, vec_b, 1, size, config, result, size);
  return translateCudaError(err);
}

/*============================== merge shards bit-reverse ==============================*/
template <typename E>
__global__ void merge_shards_bit_reverse_kernel(E* const* shards, uint64_t rho, uint64_t n, uint64_t mm, E* result)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t total_size = rho * n;
  if (tid >= total_size) { return; }

  uint64_t shard_idx = tid % rho;
  uint64_t j = tid / rho;
  uint64_t dst_idx = (__brevll(tid) >> mm);
  result[dst_idx] = shards[shard_idx][j];
}

template <typename E>
cudaError_t merge_shards_bit_reverse_cuda(
  E* const* shards, uint64_t rho, uint64_t n, uint64_t mm, const VecOpsConfig& config, E* result)
{
  CHK_INIT_IF_RETURN();

  if (shards == nullptr || result == nullptr || rho == 0 || n == 0) { return cudaErrorInvalidValue; }

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
  uint64_t total_size = rho * n;

  E** d_shards = nullptr;
  CHK_IF_RETURN(cudaMallocAsync((void**)&d_shards, sizeof(E*) * rho, cuda_stream));
  CHK_IF_RETURN(cudaMemcpyAsync(d_shards, shards, sizeof(E*) * rho, cudaMemcpyHostToDevice, cuda_stream));

  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (total_size + num_threads - 1) / num_threads;
  merge_shards_bit_reverse_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(d_shards, rho, n, mm, result);

  CHK_IF_RETURN(cudaFreeAsync(d_shards, cuda_stream));

  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));
  return CHK_LAST();
}

template <typename E>
eIcicleError merge_shards_bit_reverse_cuda_backend(
  const Device& device,
  const E* const* shards,
  uint64_t rho,
  uint64_t n,
  uint64_t mm,
  const VecOpsConfig& config,
  E* result)
{
  (void)device;
  cudaError_t err = merge_shards_bit_reverse_cuda<E>((E* const*)shards, rho, n, mm, config, result);
  return translateCudaError(err);
}

/*********************************** POLY EVAL AT ***********************************/
template <typename E>
__global__ void poly_eval_chunks_kernel(
  const E* coeffs,
  uint64_t n,
  const E* point,
  uint64_t chunk_size,
  uint64_t mm,
  bool bit_reversed,
  E* chunk_vals,
  E* chunk_pows)
{
  const uint64_t chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t num_chunks = (n + chunk_size - 1) / chunk_size;
  if (chunk_idx >= num_chunks) { return; }

  const uint64_t start = chunk_idx * chunk_size;
  const uint64_t end = ((start + chunk_size) < n) ? (start + chunk_size) : n;

  E acc = E::zero();
  E pow = E::one();
  const E x = *point;
  for (uint64_t i = start; i < end; ++i) {
    const uint64_t src_idx = bit_reversed ? (__brevll(i) >> mm) : i;
    acc = acc + coeffs[src_idx] * pow;
    pow = pow * x;
  }
  chunk_vals[chunk_idx] = acc;
  chunk_pows[chunk_idx] = pow;
}

template <typename E>
__global__ void poly_eval_merge_chunks_kernel(const E* chunk_vals, const E* chunk_pows, uint64_t num_chunks, E* output)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
  E acc = E::zero();
  for (int64_t c = (int64_t)num_chunks - 1; c >= 0; --c) {
    acc = acc * chunk_pows[c] + chunk_vals[c];
  }
  output[0] = acc;
}

template <typename E>
cudaError_t poly_eval_at_cuda(
  const E* coeffs, uint64_t n, uint64_t mm, bool bit_reversed, const E* point, const VecOpsConfig& config, E* output)
{
  CHK_INIT_IF_RETURN();
  if (n == 0 || coeffs == nullptr || point == nullptr || output == nullptr) { return cudaErrorInvalidValue; }
  if (config.batch_size != 1 || config.columns_batch) { return cudaErrorInvalidValue; }
  if (bit_reversed && ((n & (n - 1)) != 0)) { return cudaErrorInvalidValue; }

  constexpr uint64_t chunk_size = 1024;
  const uint64_t num_chunks = (n + chunk_size - 1) / chunk_size;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  struct poly_eval_workspace_t {
    E* chunk_vals = nullptr;
    E* chunk_pows = nullptr;
    uint64_t capacity = 0;
  };

  static std::mutex workspace_mu;
  static std::unordered_map<void*, poly_eval_workspace_t> workspace_by_stream;

  E* d_chunk_vals = nullptr;
  E* d_chunk_pows = nullptr;
  {
    std::lock_guard<std::mutex> guard(workspace_mu);
    void* stream_key = reinterpret_cast<void*>(cuda_stream);
    auto& ws = workspace_by_stream[stream_key];

    if (ws.capacity < num_chunks) {
      E* new_chunk_vals = nullptr;
      E* new_chunk_pows = nullptr;

      cudaError_t alloc_vals_err = cudaMalloc((void**)&new_chunk_vals, sizeof(E) * num_chunks);
      if (alloc_vals_err != cudaSuccess) { return alloc_vals_err; }
      cudaError_t alloc_pows_err = cudaMalloc((void**)&new_chunk_pows, sizeof(E) * num_chunks);
      if (alloc_pows_err != cudaSuccess) {
        cudaFree(new_chunk_vals);
        return alloc_pows_err;
      }

      // Resize path is expected to happen rarely (typically once per stream).
      // cudaFree may synchronize, but this avoids per-call malloc/free overhead.
      if (ws.chunk_vals != nullptr) {
        cudaError_t free_vals_err = cudaFree(ws.chunk_vals);
        if (free_vals_err != cudaSuccess) {
          cudaFree(new_chunk_vals);
          cudaFree(new_chunk_pows);
          return free_vals_err;
        }
      }
      if (ws.chunk_pows != nullptr) {
        cudaError_t free_pows_err = cudaFree(ws.chunk_pows);
        if (free_pows_err != cudaSuccess) {
          cudaFree(new_chunk_vals);
          cudaFree(new_chunk_pows);
          return free_pows_err;
        }
      }

      ws.chunk_vals = new_chunk_vals;
      ws.chunk_pows = new_chunk_pows;
      ws.capacity = num_chunks;
    }

    d_chunk_vals = ws.chunk_vals;
    d_chunk_pows = ws.chunk_pows;
  }

  const uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  const uint64_t num_blocks = (num_chunks + num_threads - 1) / num_threads;
  poly_eval_chunks_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
    coeffs, n, point, chunk_size, mm, bit_reversed, d_chunk_vals, d_chunk_pows);
  poly_eval_merge_chunks_kernel<<<1, 1, 0, cuda_stream>>>(d_chunk_vals, d_chunk_pows, num_chunks, output);

  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));
  return CHK_LAST();
}

template <typename E>
eIcicleError poly_eval_at_cuda_backend(
  const Device& device, const E* coeffs, uint64_t n, const E* point, const VecOpsConfig& config, E* output)
{
  (void)device;
  cudaError_t err = poly_eval_at_cuda<E>(coeffs, n, 0, false, point, config, output);
  return translateCudaError(err);
}

template <typename E>
eIcicleError poly_eval_at_bit_reverse_cuda_backend(
  const Device& device, const E* coeffs, uint64_t n, uint64_t mm, const E* point, const VecOpsConfig& config, E* output)
{
  (void)device;
  cudaError_t err = poly_eval_at_cuda<E>(coeffs, n, mm, true, point, config, output);
  return translateCudaError(err);
}

/*********************************** DIVIDE BY (X-A) ***********************************/
template <typename E>
__global__ void divide_by_x_minus_a_kernel(const E* coeffs, uint64_t n, const E* point, E* output)
{
  if (blockIdx.x != 0 || threadIdx.x != 0 || n < 2) { return; }
  output[n - 2] = coeffs[n - 1];
  for (int64_t i = (int64_t)n - 2; i >= 1; --i) {
    output[i - 1] = coeffs[i] + output[i] * (*point);
  }
}

template <typename E>
__global__ void divide_by_x_minus_a_init_affine_kernel(const E* coeffs, uint64_t m, const E* point, E* a, E* b)
{
  const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= m) { return; }

  a[tid] = *point;
  b[tid] = coeffs[tid + 1];
}

template <typename E>
__global__ void
divide_by_x_minus_a_compose_affine_kernel(const E* a_in, const E* b_in, uint64_t m, uint64_t stride, E* a_out, E* b_out)
{
  const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= m) { return; }

  const E a0 = a_in[tid];
  const E b0 = b_in[tid];
  if (tid + stride < m) {
    const E a1 = a_in[tid + stride];
    const E b1 = b_in[tid + stride];
    a_out[tid] = a0 * a1;
    b_out[tid] = b0 + a0 * b1;
  } else {
    a_out[tid] = a0;
    b_out[tid] = b0;
  }
}

template <typename E>
__global__ void divide_by_x_minus_a_finalize_kernel(const E* coeffs, const E* a, const E* b, uint64_t n, E* output)
{
  const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t m = n - 2;
  if (tid > m) { return; }

  const E tail = coeffs[n - 1];
  if (tid == m) {
    output[m] = tail;
    return;
  }

  output[tid] = b[tid] + a[tid] * tail;
}

template <typename E>
cudaError_t divide_by_x_minus_a_cuda(const E* coeffs, uint64_t n, const E* point, const VecOpsConfig& config, E* output)
{
  CHK_INIT_IF_RETURN();
  if (n < 2 || coeffs == nullptr || point == nullptr || output == nullptr) { return cudaErrorInvalidValue; }
  if (config.batch_size != 1 || config.columns_batch) { return cudaErrorInvalidValue; }

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
  constexpr uint64_t PARALLEL_THRESHOLD = 64;
  if (n <= PARALLEL_THRESHOLD) {
    divide_by_x_minus_a_kernel<<<1, 1, 0, cuda_stream>>>(coeffs, n, point, output);
  } else {
    const uint64_t m = n - 2;
    E* a0 = allocate_on_device<E>(m * sizeof(E), cuda_stream);
    E* b0 = allocate_on_device<E>(m * sizeof(E), cuda_stream);
    E* a1 = allocate_on_device<E>(m * sizeof(E), cuda_stream);
    E* b1 = allocate_on_device<E>(m * sizeof(E), cuda_stream);

    const uint64_t num_threads = MAX_THREADS_PER_BLOCK;
    const uint64_t num_blocks_m = (m + num_threads - 1) / num_threads;

    divide_by_x_minus_a_init_affine_kernel<<<num_blocks_m, num_threads, 0, cuda_stream>>>(coeffs, m, point, a0, b0);

    E* a_in = a0;
    E* b_in = b0;
    E* a_out = a1;
    E* b_out = b1;
    for (uint64_t stride = 1; stride < m; stride <<= 1) {
      divide_by_x_minus_a_compose_affine_kernel<<<num_blocks_m, num_threads, 0, cuda_stream>>>(
        a_in, b_in, m, stride, a_out, b_out);
      std::swap(a_in, a_out);
      std::swap(b_in, b_out);
    }

    const uint64_t out_size = n - 1;
    const uint64_t num_blocks_out = (out_size + num_threads - 1) / num_threads;
    divide_by_x_minus_a_finalize_kernel<<<num_blocks_out, num_threads, 0, cuda_stream>>>(coeffs, a_in, b_in, n, output);

    CHK_IF_RETURN(cudaFreeAsync(a0, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(b0, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(a1, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(b1, cuda_stream));
  }

  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));
  return CHK_LAST();
}

template <typename E>
eIcicleError divide_by_x_minus_a_cuda_backend(
  const Device& device, const E* coeffs, uint64_t n, const E* point, const VecOpsConfig& config, E* output)
{
  (void)device;
  cudaError_t err = divide_by_x_minus_a_cuda<E>(coeffs, n, point, config, output);
  return translateCudaError(err);
}

/*********************************** SUPPORT IDENTITY ***********************************/
template <typename E>
__device__ E pow_u64(E base, uint64_t exp)
{
  E result = E::one();
  while (exp > 0) {
    if (exp & 1) { result = result * base; }
    base = base * base;
    exp >>= 1;
  }
  return result;
}

template <typename E>
__global__ void
support_identity_kernel(const E* omega, const E* shift, uint64_t size_poly, uint64_t nb_copies, E* output)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t total_size = size_poly * nb_copies;
  if (tid >= total_size) { return; }

  const uint64_t copy_idx = tid / size_poly;
  const uint64_t i = tid - copy_idx * size_poly;

  const E omega_pow = pow_u64(*omega, i);
  const E shift_pow = pow_u64(*shift, copy_idx);
  output[tid] = omega_pow * shift_pow;
}

template <typename E>
cudaError_t support_identity_cuda(
  const E* omega, const E* shift, uint64_t size_poly, uint64_t nb_copies, const VecOpsConfig& config, E* output)
{
  CHK_INIT_IF_RETURN();
  if (omega == nullptr || shift == nullptr || output == nullptr) { return cudaErrorInvalidValue; }
  if (size_poly != 0 && nb_copies > (UINT64_MAX / size_poly)) { return cudaErrorInvalidValue; }
  const uint64_t total_size = size_poly * nb_copies;
  if (total_size == 0) { return cudaSuccess; }

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (total_size + num_threads - 1) / num_threads;
  support_identity_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(omega, shift, size_poly, nb_copies, output);

  if (!config.is_async) { return CHK_STICKY(cudaStreamSynchronize(cuda_stream)); }
  return CHK_LAST();
}

template <typename E>
eIcicleError support_identity_cuda_backend(
  const Device& device,
  const E* omega,
  const E* shift,
  uint64_t size_poly,
  uint64_t nb_copies,
  const VecOpsConfig& config,
  E* output)
{
  (void)device;
  cudaError_t err = support_identity_cuda<E>(omega, shift, size_poly, nb_copies, config, output);
  return translateCudaError(err);
}

/*********************************** GATHER BY INDICES ***********************************/
template <typename E>
__global__ void
gather_by_indices_kernel(const E* input, uint64_t input_size, const int64_t* indices, uint64_t size, E* output)
{
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) { return; }

  const int64_t idx = indices[tid];
  if (idx < 0 || static_cast<uint64_t>(idx) >= input_size) {
    output[tid] = E::zero();
    return;
  }
  output[tid] = input[idx];
}

template <typename E>
cudaError_t gather_by_indices_cuda(
  const E* input, uint64_t input_size, const int64_t* indices, uint64_t size, const VecOpsConfig& config, E* output)
{
  CHK_INIT_IF_RETURN();
  if (input == nullptr || indices == nullptr || output == nullptr) { return cudaErrorInvalidValue; }
  if (size == 0) { return cudaSuccess; }

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
  uint64_t num_threads = MAX_THREADS_PER_BLOCK;
  uint64_t num_blocks = (size + num_threads - 1) / num_threads;
  gather_by_indices_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(input, input_size, indices, size, output);

  if (!config.is_async) { return CHK_STICKY(cudaStreamSynchronize(cuda_stream)); }
  return CHK_LAST();
}

template <typename E>
eIcicleError gather_by_indices_cuda_backend(
  const Device& device,
  const E* input,
  uint64_t input_size,
  const int64_t* indices,
  uint64_t size,
  const VecOpsConfig& config,
  E* output)
{
  (void)device;
  cudaError_t err = gather_by_indices_cuda<E>(input, input_size, indices, size, config, output);
  return translateCudaError(err);
}

/************************************ REGISTRATION ************************************/

REGISTER_VECTOR_ADD_BACKEND("CUDA", add_cuda<scalar_t>);
REGISTER_VECTOR_SUB_BACKEND("CUDA", sub_cuda<scalar_t>);
REGISTER_VECTOR_MUL_BACKEND("CUDA", mul_cuda<scalar_t>);
REGISTER_VECTOR_SHIFT_BACKEND("CUDA", shift_cuda<scalar_t>);
REGISTER_VECTOR_BATCH_INVERSE_BACKEND("CUDA", batch_inverse_cuda<scalar_t>);
REGISTER_SCALAR_MUL_VEC_BACKEND("CUDA", (mul_scalar_cuda<scalar_t>));
REGISTER_SCALAR_ADD_VEC_BACKEND("CUDA", (add_scalar_cuda<scalar_t>));
REGISTER_MERGE_SHARDS_BIT_REVERSE_BACKEND("CUDA", (merge_shards_bit_reverse_cuda_backend<scalar_t>));
REGISTER_POLY_EVAL_AT_BACKEND("CUDA", (poly_eval_at_cuda_backend<scalar_t>));
REGISTER_POLY_EVAL_AT_BIT_REVERSE_BACKEND("CUDA", (poly_eval_at_bit_reverse_cuda_backend<scalar_t>));
REGISTER_DIVIDE_BY_X_MINUS_A_BACKEND("CUDA", (divide_by_x_minus_a_cuda_backend<scalar_t>));
REGISTER_SUPPORT_IDENTITY_BACKEND("CUDA", (support_identity_cuda_backend<scalar_t>));
REGISTER_GATHER_BY_INDICES_BACKEND("CUDA", (gather_by_indices_cuda_backend<scalar_t>));
