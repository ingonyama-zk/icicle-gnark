

#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <atomic>
#include <iostream>
#include <unordered_map>
#include <type_traits>

#include "icicle/backend/ntt_backend.h"
#include "icicle/device.h"
#include "icicle/config_extension.h"
#include "error_translation.h"

#include "gpu-utils/sharedmem.h"
#include "utils/utils_kernels.h"
#include "gpu-utils/device_context.h"
#include "gpu-utils/error_handler.h"
#include "mixed_radix_ntt.cuh"
#include "ntt/cuda_ntt_config.cuh"
#include "icicle/utils/log.h"
#include "icicle/utils/utils.h"

#include "icicle/fields/field_config.h"
using namespace field_config;

#include <mutex>

#ifdef CURVE_ID
  #include "icicle/curves/curve_config.h"
using namespace curve_config;
  #define IS_ECNTT std::is_same_v<E, projective_t>
#else
  #define IS_ECNTT false
#endif

namespace ntt {

  namespace {
    // TODO: Set MAX THREADS based on GPU arch
    const uint32_t MAX_NUM_THREADS = 512; // TODO: hotfix - should be 1024, currently limits shared memory size
    const uint32_t MAX_THREADS_BATCH = 512;
    const uint32_t MAX_THREADS_BATCH_ECNTT =
      128; // TODO: hardcoded - allows (2^18 x 64) ECNTT for sm86, decrease this to allow larger ecntt length, batch
    // size limited by on-device memory
    const uint32_t MAX_SHARED_MEM_ELEMENT_SIZE = 32; // TODO: occupancy calculator, hardcoded for sm_86..sm_89
    const uint32_t MAX_SHARED_MEM = MAX_SHARED_MEM_ELEMENT_SIZE * MAX_NUM_THREADS;
    static inline std::mutex ntt_profile_log_mutex;

    inline bool is_env_flag_enabled(const char* key)
    {
      const char* value = std::getenv(key);
      if (!value || value[0] == '\0') return false;
      const char c = value[0];
      return !(c == '0' || c == 'f' || c == 'F' || c == 'n' || c == 'N');
    }

    inline bool is_ntt_full_profile_enabled()
    {
      static const bool enabled = is_env_flag_enabled("ICICLE_NTT_PROFILE_FULL");
      return enabled;
    }

    inline bool is_arbitrary_coset_profile_enabled()
    {
      static const bool enabled = true;
      return enabled;
    }

    template <typename S>
    __device__ __forceinline__ S pow_u64(S base, uint64_t exp)
    {
      S result = S::one();
      while (exp > 0) {
        if (exp & 1) result = result * base;
        base = base * base;
        exp >>= 1;
      }
      return result;
    }

    template <typename S>
    __global__ void generate_arbitrary_coset_powers_kernel(S* out, uint64_t size, S coset_gen)
    {
      const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
      const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
      if (tid >= size) return;

      // Each thread computes g^tid once, then advances by multiplying with g^stride.
      S cur = pow_u64(coset_gen, tid);
      const S step = pow_u64(coset_gen, stride);
      for (uint64_t i = tid; i < size; i += stride) {
        out[i] = cur;
        cur = cur * step;
      }
    }

    template <typename E>
    __global__ void reverse_order_kernel(const E* arr, E* arr_reversed, uint32_t n, uint32_t logn, uint32_t batch_size)
    {
      int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (threadId < n * batch_size) {
        int idx = threadId % n;
        int batch_idx = threadId / n;
        int idx_reversed = __brev(idx) >> (32 - logn);

        if (arr == arr_reversed) { // for in-place (when pointers arr==arr_reversed)
          if (idx < idx_reversed) {
            E val = arr[batch_idx * n + idx];
            arr_reversed[batch_idx * n + idx] = arr[batch_idx * n + idx_reversed];
            arr_reversed[batch_idx * n + idx_reversed] = val;
          }
        } else
          arr_reversed[batch_idx * n + idx_reversed] = arr[batch_idx * n + idx];
      }
    }

    /**
     * Bit-reverses a batch of input arrays out-of-place inside GPU.
     * for example: on input array ([a[0],a[1],a[2],a[3]], 4, 2) it returns
     * [a[0],a[3],a[2],a[1]] (elements at indices 3 and 1 swhich places).
     * @param arr_in batch of arrays of some object of type T. Should be on GPU.
     * @param n length of `arr`.
     * @param logn log(n).
     * @param batch_size the size of the batch.
     * @param arr_out buffer of the same size as `arr_in` on the GPU to write the bit-permuted array into.
     */
    template <typename E>
    void reverse_order_batch(
      const E* arr_in, uint32_t n, uint32_t logn, uint32_t batch_size, cudaStream_t stream, E* arr_out)
    {
      int number_of_threads = MAX_THREADS_BATCH;
      int number_of_blocks = (n * batch_size + number_of_threads - 1) / number_of_threads;
      reverse_order_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(arr_in, arr_out, n, logn, batch_size);
    }

    /**
     * Bit-reverses an input array out-of-place inside GPU.
     * for example: on array ([a[0],a[1],a[2],a[3]], 4, 2) it returns
     * [a[0],a[3],a[2],a[1]] (elements at indices 3 and 1 swhich places).
     * @param arr_in array of some object of type T of size which is a power of 2. Should be on GPU.
     * @param n length of `arr`.
     * @param logn log(n).
     * @param arr_out buffer of the same size as `arr_in` on the GPU to write the bit-permuted array into.
     */
    template <typename E>
    void reverse_order(const E* arr_in, uint32_t n, uint32_t logn, cudaStream_t stream, E* arr_out)
    {
      reverse_order_batch(arr_in, n, logn, 1, stream, arr_out);
    }

    /**
     * Cooley-Tuckey NTT.
     * NOTE! this function assumes that d_twiddles are located in the device memory.
     * @param arr_in input array of type E (elements).
     * @param n length of d_arr.
     * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power
     of 2).
     * @param n_twiddles length of twiddles, should be negative for intt.
     * @param max_task max count of parallel tasks.
     * @param s log2(n) loop index.
     * @param arr_out buffer for the output.
     */
    template <typename E, typename S>
    __global__ void ntt_template_kernel_shared_rev(
      const E* __restrict__ arr_in,
      int n,
      const S* __restrict__ r_twiddles,
      int n_twiddles,
      int max_task,
      int ss,
      int logn,
      E* __restrict__ arr_out)
    {
      SharedMemory<E> smem;
      E* arr = smem.getPointer();

      uint32_t task = blockIdx.x;
      uint32_t loop_limit = blockDim.x;
      uint32_t chunks = n / (loop_limit * 2);
      uint32_t offset = (task / chunks) * n;
      if (task < max_task) {
        // flattened loop allows parallel processing
        uint32_t l = threadIdx.x;

        if (l < loop_limit) {
          UNROLL
          for (; ss < logn; ss++) {
            int s = logn - ss - 1;
            bool is_beginning = ss == 0;
            bool is_end = ss == (logn - 1);

            uint32_t ntw_i = task % chunks;

            uint32_t n_twiddles_div = n_twiddles >> (s + 1);

            uint32_t shift_s = 1 << s;
            uint32_t shift2_s = 1 << (s + 1);

            l = ntw_i * loop_limit + l; // to l from chunks to full

            uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
            uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
            uint32_t oij = i + j;
            uint32_t k = oij + shift_s;

            S tw = *(r_twiddles + (int)(j * n_twiddles_div));

            E u = is_beginning ? arr_in[offset + oij] : arr[oij];
            E v = is_beginning ? arr_in[offset + k] : arr[k];
            if (is_end) {
              arr_out[offset + oij] = u + v;
              arr_out[offset + k] = tw * (u - v);
            } else {
              arr[oij] = u + v;
              arr[k] = tw * (u - v);
            }

            __syncthreads();
          }
        }
      }
    }

    /**
     * Cooley-Tuckey NTT.
     * NOTE! this function assumes that d_twiddles are located in the device memory.
     * @param arr_in input array of type E (elements).
     * @param n length of d_arr.
     * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power
     of 2).
     * @param n_twiddles length of twiddles, should be negative for intt.
     * @param max_task max count of parallel tasks.
     * @param s log2(n) loop index.
     * @param arr_out buffer for the output.
     */
    template <typename E, typename S>
    __global__ void ntt_template_kernel_shared(
      const E* __restrict__ arr_in,
      int n,
      const S* __restrict__ r_twiddles,
      int n_twiddles,
      int max_task,
      int s,
      int logn,
      E* __restrict__ arr_out)
    {
      SharedMemory<E> smem;
      E* arr = smem.getPointer();

      uint32_t task = blockIdx.x;
      uint32_t loop_limit = blockDim.x;
      uint32_t chunks = n / (loop_limit * 2);
      uint32_t offset = (task / chunks) * n;
      if (task < max_task) {
        // flattened loop allows parallel processing
        uint32_t l = threadIdx.x;

        if (l < loop_limit) {
          UNROLL
          for (; s < logn; s++) // TODO: this loop also can be unrolled
          {
            uint32_t ntw_i = task % chunks;

            uint32_t n_twiddles_div = n_twiddles >> (s + 1);

            uint32_t shift_s = 1 << s;
            uint32_t shift2_s = 1 << (s + 1);

            l = ntw_i * loop_limit + l; // to l from chunks to full

            uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
            uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
            uint32_t oij = i + j;
            uint32_t k = oij + shift_s;
            S tw = *(r_twiddles + (int)(j * n_twiddles_div));

            E u = s == 0 ? arr_in[offset + oij] : arr[oij];
            E v = s == 0 ? arr_in[offset + k] : arr[k];
            v = tw * v;
            if (s == (logn - 1)) {
              arr_out[offset + oij] = u + v;
              arr_out[offset + k] = u - v;
            } else {
              arr[oij] = u + v;
              arr[k] = u - v;
            }

            __syncthreads();
          }
        }
      }
    }

    /**
     * Cooley-Tukey NTT.
     * NOTE! this function assumes that d_twiddles are located in the device memory.
     * @param arr input array of type E (elements).
     * @param n length of d_arr.
     * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power
     of 2).
     * @param n_twiddles length of twiddles, should be negative for intt.
     * @param max_task max count of parallel tasks.
     * @param s log2(n) loop index.
     */
    template <typename E, typename S>
    __global__ void
    ntt_template_kernel(const E* arr_in, int n, S* twiddles, int n_twiddles, int max_task, int s, bool rev, E* arr_out)
    {
      int task = blockIdx.x;
      int chunks = n / (blockDim.x * 2);

      if (task < max_task) {
        // flattened loop allows parallel processing
        uint32_t l = threadIdx.x;
        uint32_t loop_limit = blockDim.x;

        if (l < loop_limit) {
          uint32_t ntw_i = task % chunks;

          uint32_t shift_s = 1 << s;
          uint32_t shift2_s = 1 << (s + 1);
          uint32_t n_twiddles_div = n_twiddles >> (s + 1);

          l = ntw_i * blockDim.x + l; // to l from chunks to full

          uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
          uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
          uint32_t k = i + j + shift_s;

          S tw = *(twiddles + (int)(j * n_twiddles_div));

          uint32_t offset = (task / chunks) * n;
          E u = arr_in[offset + i + j];
          E v = arr_in[offset + k];
          if (!rev) v = tw * v;
          arr_out[offset + i + j] = u + v;
          v = u - v;
          arr_out[offset + k] = rev ? tw * v : v;
        }
      }
    }

    /**
     * NTT/INTT inplace batch
     * Note: this function does not perform any bit-reverse permutations on its inputs or outputs.
     * @param d_input Input array
     * @param n Size of `d_input`
     * @param d_twiddles Twiddles
     * @param n_twiddles Size of `d_twiddles`
     * @param batch_size The size of the batch; the length of `d_inout` is `n` * `batch_size`.
     * @param inverse true for iNTT
     * @param coset should be array of length n or a nullptr if NTT is not computed on a coset
     * @param stream CUDA stream
     * @param is_async if false, perform sync of the supplied CUDA stream at the end of processing
     * @param d_output Output array
     */
    template <typename E, typename S>
    cudaError_t ntt_inplace_batch_template(
      const E* d_input,
      int n,
      S* d_twiddles,
      int n_twiddles,
      int batch_size,
      int logn,
      bool inverse,
      bool dit,
      S* arbitrary_coset,
      int coset_gen_index,
      cudaStream_t stream,
      E* d_output)
    {
      CHK_INIT_IF_RETURN();

      bool is_shared_mem_enabled = sizeof(E) <= MAX_SHARED_MEM_ELEMENT_SIZE;
      const int log2_shmem_elems = is_shared_mem_enabled ? int(log(int(MAX_SHARED_MEM / sizeof(E))) / log(2)) : logn;
      int max_threads_batch = IS_ECNTT ? MAX_THREADS_BATCH_ECNTT : MAX_THREADS_BATCH;
      int num_threads = max(min(min(n / 2, max_threads_batch), 1 << (log2_shmem_elems - 1)), 1);
      const int chunks = max(int((n / 2) / num_threads), 1);
      const int total_tasks = batch_size * chunks;
      int num_blocks = total_tasks;
      const int shared_mem = 2 * num_threads * sizeof(E); // TODO: calculator, as shared mem size may be more efficient
                                                          // less then max to allow more concurrent blocks on SM
      const int logn_shmem = is_shared_mem_enabled ? int(log(2 * num_threads) / log(2))
                                                   : 0; // TODO: shared memory support only for types <= 32 bytes
      // Note: for ecntt we limit block size (=#threads per block) since otherwise it doesn't fit the SM resources.
      int num_threads_coset = max(min(n / 2, IS_ECNTT ? MAX_THREADS_BATCH_ECNTT : MAX_NUM_THREADS), 1);
      int num_blocks_coset = (n * batch_size + num_threads_coset - 1) / num_threads_coset;

      if (inverse) {
        d_twiddles = d_twiddles + n_twiddles;
        n_twiddles = -n_twiddles;
      }

      bool is_on_coset = (coset_gen_index != 0) || arbitrary_coset;
      bool direct_coset = (!inverse && is_on_coset);
      if (direct_coset)
        utils_internal::BatchMulKernel<E, S><<<num_blocks_coset, num_threads_coset, 0, stream>>>(
          d_input, n, batch_size, arbitrary_coset ? arbitrary_coset : d_twiddles, arbitrary_coset ? 1 : coset_gen_index,
          n_twiddles, logn, dit, d_output);

      if (dit) {
        if (is_shared_mem_enabled)
          ntt_template_kernel_shared<<<num_blocks, num_threads, shared_mem, stream>>>(
            direct_coset ? d_output : d_input, 1 << logn_shmem, d_twiddles, n_twiddles, total_tasks, 0, logn_shmem,
            d_output);

        for (int s = logn_shmem; s < logn; s++) // TODO: this loop also can be unrolled
        {
          ntt_template_kernel<E, S><<<num_blocks, num_threads, 0, stream>>>(
            (direct_coset || (s > 0)) ? d_output : d_input, n, d_twiddles, n_twiddles, total_tasks, s, false, d_output);
        }
      } else {
        for (int s = logn - 1; s >= logn_shmem; s--) // TODO: this loop also can be unrolled
        {
          ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(
            (direct_coset || (s < logn - 1)) ? d_output : d_input, n, d_twiddles, n_twiddles, total_tasks, s, true,
            d_output);
        }

        if (is_shared_mem_enabled)
          ntt_template_kernel_shared_rev<<<num_blocks, num_threads, shared_mem, stream>>>(
            (direct_coset || (logn > logn_shmem)) ? d_output : d_input, 1 << logn_shmem, d_twiddles, n_twiddles,
            total_tasks, 0, logn_shmem, d_output);
      }

      if (inverse) {
        if (is_on_coset)
          utils_internal::BatchMulKernel<E, S><<<num_blocks_coset, num_threads_coset, 0, stream>>>(
            d_output, n, batch_size, arbitrary_coset ? arbitrary_coset : d_twiddles,
            arbitrary_coset ? 1 : -coset_gen_index, -n_twiddles, logn, !dit, d_output);

        utils_internal::NormalizeKernel<E, S>
          <<<num_blocks_coset, num_threads_coset, 0, stream>>>(d_output, S::inv_log_size(logn), n * batch_size);
      }

      return CHK_LAST();
    }

  } // namespace

  template <typename S, typename E>
  cudaError_t ntt_cuda(const E* input, int size, NTTDir dir, const NTTConfig<S>& config, int device_id, E* output);

  /**
   * @struct Domain
   * Struct containing information about the domain on which (i)NTT is evaluated i.e. twiddle factors.
   * Twiddle factors are private, static and can only be set using [init_domain](@ref init_domain) function.
   * The internal representation of twiddles is prone to change in accordance with changing [NTT](@ref NTT)
   algorithm.
   * @tparam S The type of twiddle factors \f$ \{ \omega^i \} \f$. Must be a field.
   */
  template <typename S>
  class Domain
  {
    // Mutex for protecting access to the domain/device container array
    static inline std::mutex device_domain_mutex;
    // The domain-per-device container - assumption is init_domain is called once per device per program.

    int max_size = 0;
    int max_log_size = 0;
    S* twiddles = nullptr;
    bool initialized = false; // protection for multi-threaded case
    std::unordered_map<S, int> coset_index = {};
    std::unordered_map<S, S*> arbitrary_coset_cache = {};

    S* internal_twiddles = nullptr; // required by mixed-radix NTT
    S* basic_twiddles = nullptr;    // required by mixed-radix NTT

    // mixed-radix NTT supports a fast-twiddle option at the cost of additional 4N memory (where N is max NTT size)
    S* fast_external_twiddles = nullptr;     // required by mixed-radix NTT (fast-twiddles mode)
    S* fast_internal_twiddles = nullptr;     // required by mixed-radix NTT (fast-twiddles mode)
    S* fast_basic_twiddles = nullptr;        // required by mixed-radix NTT (fast-twiddles mode)
    S* fast_external_twiddles_inv = nullptr; // required by mixed-radix NTT (fast-twiddles mode)
    S* fast_internal_twiddles_inv = nullptr; // required by mixed-radix NTT (fast-twiddles mode)
    S* fast_basic_twiddles_inv = nullptr;    // required by mixed-radix NTT (fast-twiddles mode)

  public:
    static inline Domain domains_for_devices[device_context::MAX_DEVICES] = {};

    static cudaError_t init_domain(S primitive_root, device_context::DeviceContext& ctx, bool fast_tw);
    static cudaError_t release_domain(device_context::DeviceContext& ctx);
    static S get_root_of_unity_from_domain(uint64_t logn, device_context::DeviceContext& ctx);

    template <typename U, typename E>
    friend cudaError_t
    ntt_cuda<U, E>(const E* input, int size, NTTDir dir, const NTTConfig<U>& config, int device_id, E* output);
  };

  template <typename S>
  cudaError_t Domain<S>::init_domain(S primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode)
  {
    CHK_INIT_IF_RETURN();

    Domain& domain = domains_for_devices[ctx.device_id];
    const bool profile_ntt_full = is_ntt_full_profile_enabled();
    const auto init_total_start =
      profile_ntt_full ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool fast_twiddles_requested = fast_twiddles_mode;
    bool initialized_now = false;
    double twiddles_gen_ms = 0.0;
    double fast_twiddles_ms = 0.0;
    double sync_ms = 0.0;
    double coset_map_ms = 0.0;

    // only generate twiddles if they haven't been generated yet
    // please note that this offers just basic thread-safety,
    // it's assumed a singleton (non-enforced) that is supposed
    // to be initialized once per device per program lifetime
    if (!domain.initialized) {
      // Mutex is automatically released when lock goes out of scope, even in case of exceptions
      std::lock_guard<std::mutex> lock(Domain<S>::device_domain_mutex);
      // double check locking
      if (domain.initialized) return CHK_LAST(); // another thread is already initializing the domain

      initialized_now = true;
      bool found_logn = false;
      S omega = primitive_root;
      unsigned omegas_count = S::get_omegas_count();
      for (int i = 0; i < omegas_count; i++) {
        omega = S::sqr(omega);
        if (!found_logn) {
          ++domain.max_log_size;
          found_logn = omega == S::one();
          if (found_logn) break;
        }
      }

      domain.max_size = (int)pow(2, domain.max_log_size);
      if (omega != S::one()) {
        THROW_ICICLE_ERR(
          eIcicleError::INVALID_ARGUMENT, "Primitive root provided to the InitDomain function is not in the subgroup");
      }

      // allocate and calculate twiddles on GPU
      // Note: radix-2 INTT needs ONE in last element (in addition to first element), therefore have n+1 elements
      // Managed allocation allows host to read the elements (logn) without copying all (n) TFs back to host
      const auto twiddles_gen_start =
        profile_ntt_full ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      CHK_IF_RETURN(cudaMallocManaged(&domain.twiddles, (domain.max_size + 1) * sizeof(S)));
      CHK_IF_RETURN(mxntt::generate_external_twiddles_generic(
        primitive_root, domain.twiddles, domain.internal_twiddles, domain.basic_twiddles, domain.max_log_size,
        ctx.stream));
      if (profile_ntt_full) {
        twiddles_gen_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - twiddles_gen_start).count();
      }

      fast_twiddles_mode &= domain.max_log_size >= 4; // fast twiddles cannot be built for smaller domain
      if (fast_twiddles_mode) {
        const auto fast_twiddles_start =
          profile_ntt_full ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // generating fast-twiddles (note that this cost 4N additional memory)
        CHK_IF_RETURN(cudaMallocAsync(&domain.fast_external_twiddles, domain.max_size * sizeof(S) * 2, ctx.stream));
        CHK_IF_RETURN(cudaMallocAsync(&domain.fast_external_twiddles_inv, domain.max_size * sizeof(S) * 2, ctx.stream));

        // fast-twiddles forward NTT
        CHK_IF_RETURN(mxntt::generate_external_twiddles_fast_twiddles_mode(
          primitive_root, domain.fast_external_twiddles, domain.fast_internal_twiddles, domain.fast_basic_twiddles,
          domain.max_log_size, ctx.stream));

        // fast-twiddles inverse NTT
        S primitive_root_inv;
        CHK_IF_RETURN(cudaMemcpyAsync(
          &primitive_root_inv, &domain.twiddles[domain.max_size - 1], sizeof(S), cudaMemcpyDeviceToHost, ctx.stream));
        CHK_IF_RETURN(mxntt::generate_external_twiddles_fast_twiddles_mode(
          primitive_root_inv, domain.fast_external_twiddles_inv, domain.fast_internal_twiddles_inv,
          domain.fast_basic_twiddles_inv, domain.max_log_size, ctx.stream));
        if (profile_ntt_full) {
          fast_twiddles_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fast_twiddles_start).count();
        }
      }
      const auto sync_start =
        profile_ntt_full ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      CHK_IF_RETURN(cudaStreamSynchronize(ctx.stream));
      if (profile_ntt_full) {
        sync_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync_start).count();
      }

      const bool is_map_only_powers_of_primitive_root = true;
      const auto coset_map_start =
        profile_ntt_full ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      if (is_map_only_powers_of_primitive_root) {
        // populate the coset_index map. Note that only powers of the primitive-root are stored (1, PR, PR^2, PR ^ 4, PR
        // ^ 8 etc.)
        domain.coset_index[S::one()] = 0;
        for (int i = 0; i < domain.max_log_size; ++i) {
          const int index = (int)pow(2, i);
          domain.coset_index[domain.twiddles[index]] = index;
        }
      } else {
        // populate all values
        for (int i = 0; i < domain.max_size; ++i) {
          domain.coset_index[domain.twiddles[i]] = i;
        }
      }
      if (profile_ntt_full) {
        coset_map_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - coset_map_start).count();
      }
      domain.initialized = true;
    }

    if (profile_ntt_full) {
      const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - init_total_start).count();
      std::lock_guard<std::mutex> lock(ntt_profile_log_mutex);
      std::cerr << "[ICICLE][NTT_PROFILE_FULL] init_domain"
                << " device_id=" << ctx.device_id << " initialized_now=" << (initialized_now ? "1" : "0")
                << " max_size=" << domain.max_size << " max_log_size=" << domain.max_log_size
                << " fast_twiddles_requested=" << (fast_twiddles_requested ? "1" : "0")
                << " fast_twiddles_built=" << ((domain.fast_external_twiddles != nullptr) ? "1" : "0")
                << " twiddles_gen_ms=" << twiddles_gen_ms << " fast_twiddles_ms=" << fast_twiddles_ms
                << " stream_sync_ms=" << sync_ms << " coset_map_ms=" << coset_map_ms << " total_ms=" << total_ms
                << std::endl;
    }

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t Domain<S>::release_domain(device_context::DeviceContext& ctx)
  {
    CHK_INIT_IF_RETURN();

    Domain& domain = domains_for_devices[ctx.device_id];

    domain.max_size = 0;
    domain.max_log_size = 0;
    domain.twiddles = nullptr; // allocated via cudaMallocManaged(...) so released without calling cudaFree(...)
    CHK_IF_RETURN(cudaFreeAsync(domain.internal_twiddles, ctx.stream));
    domain.internal_twiddles = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.basic_twiddles, ctx.stream));
    domain.basic_twiddles = nullptr;
    domain.coset_index.clear();
    for (auto it = domain.arbitrary_coset_cache.begin(); it != domain.arbitrary_coset_cache.end(); ++it) {
      CHK_IF_RETURN(cudaFreeAsync(it->second, ctx.stream));
    }
    domain.arbitrary_coset_cache.clear();

    CHK_IF_RETURN(cudaFreeAsync(domain.fast_external_twiddles, ctx.stream));
    domain.fast_external_twiddles = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.fast_internal_twiddles, ctx.stream));
    domain.fast_internal_twiddles = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.fast_basic_twiddles, ctx.stream));
    domain.fast_basic_twiddles = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.fast_external_twiddles_inv, ctx.stream));
    domain.fast_external_twiddles_inv = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.fast_internal_twiddles_inv, ctx.stream));
    domain.fast_internal_twiddles_inv = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.fast_basic_twiddles_inv, ctx.stream));
    domain.fast_basic_twiddles_inv = nullptr;
    domain.initialized = false;

    return CHK_LAST();
  }

  template <typename S>
  S Domain<S>::get_root_of_unity_from_domain(uint64_t logn, device_context::DeviceContext& ctx)
  {
    Domain& domain = domains_for_devices[ctx.device_id];
    if (logn > domain.max_log_size) {
      std::ostringstream oss;
      oss << "NTT log_size=" << logn << " is too large for the domain (logsize=" << domain.max_log_size
          << "). Consider generating your domain with a higher order root of unity.\n ";
      THROW_ICICLE_ERR(eIcicleError::INVALID_ARGUMENT, oss.str().c_str());
    }
    const size_t twiddles_idx = 1ULL << (domain.max_log_size - logn);
    return domain.twiddles[twiddles_idx];
  }

  template <typename S>
  static bool is_choosing_radix2_algorithm(int logn, int batch_size, const NTTConfig<S>& config)
  {
    const bool is_mixed_radix_alg_supported = (logn > 3 && logn != 7);
    if (!is_mixed_radix_alg_supported && config.columns_batch) {
      THROW_ICICLE_ERR(eIcicleError::INVALID_ARGUMENT, "columns batch is not supported for given NTT size");
    }

    const NttAlgorithm ntt_alg = get_ntt_alg_from_config(config.ext);
    const bool is_user_selected_radix2_alg = ntt_alg == NttAlgorithm::Radix2;
    const bool is_force_radix2 = !is_mixed_radix_alg_supported || is_user_selected_radix2_alg;
    if (is_force_radix2) return true;

    const bool is_user_selected_mixed_radix_alg = ntt_alg == NttAlgorithm::MixedRadix;
    if (is_user_selected_mixed_radix_alg) return false;
    if (config.columns_batch) return false; // radix2 does not currently support columns batch mode.

    // Heuristic to automatically select an algorithm
    // Note that generally the decision depends on {logn, batch, ordering, inverse, coset, in-place, coeff-field} and
    // the specific GPU.
    // the following heuristic is a simplification based on measurements. Users can try both and select the algorithm
    // based on the specific case via the 'NTTConfig.ntt_algorithm' field

    if (logn >= 16) return false; // mixed-radix is typically faster in those cases
    if (logn <= 11) return true;  //  radix-2 is typically faster for batch<=256 in those cases
    const int log_batch = (int)log2(batch_size);
    return (logn + log_batch <= 18); // almost the cutoff point where both are equal
  }

  template <typename S, typename E>
  cudaError_t radix2_ntt(
    const E* d_input,
    E* d_output,
    S* twiddles,
    int ntt_size,
    int max_size,
    int batch_size,
    bool is_inverse,
    Ordering ordering,
    S* arbitrary_coset,
    int coset_gen_index,
    cudaStream_t cuda_stream)
  {
    CHK_INIT_IF_RETURN();

    const int logn = int(log2(ntt_size));

    bool dit = true;
    bool reverse_input = false;
    switch (ordering) {
    case Ordering::kNN:
      reverse_input = true;
      break;
    case Ordering::kNR:
    case Ordering::kNM:
      dit = false;
      break;
    case Ordering::kRR:
      reverse_input = true;
      dit = false;
      break;
    case Ordering::kRN:
    case Ordering::kMN:
      reverse_input = false;
    }

    if (reverse_input) reverse_order_batch(d_input, ntt_size, logn, batch_size, cuda_stream, d_output);

    CHK_IF_RETURN(ntt_inplace_batch_template(
      reverse_input ? d_output : d_input, ntt_size, twiddles, max_size, batch_size, logn, is_inverse, dit,
      arbitrary_coset, coset_gen_index, cuda_stream, d_output));

    return CHK_LAST();
  }

  template <typename S, typename E>
  cudaError_t ntt_cuda(const E* input, int size, NTTDir dir, const NTTConfig<S>& config, int device_id, E* output)
  {
    CHK_INIT_IF_RETURN();

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
    Domain<S>& domain = Domain<S>::domains_for_devices[device_id];
    const bool profile_ntt_full = is_ntt_full_profile_enabled();
    using profile_clock = std::chrono::steady_clock;
    const auto total_start = profile_ntt_full ? profile_clock::now() : profile_clock::time_point{};
    static std::atomic<uint64_t> ntt_profile_call_counter{0};
    uint64_t profile_call_id = 0;
    if (profile_ntt_full) {
      profile_call_id = ntt_profile_call_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      if (profile_call_id == 1) {
        std::lock_guard<std::mutex> lock(ntt_profile_log_mutex);
        std::cerr << "[ICICLE][NTT_PROFILE_FULL] enabled via ICICLE_NTT_PROFILE_FULL"
                  << " (per-call ntt_cuda and init_domain timings)" << std::endl;
      }
    }

    double input_prepare_ms = 0.0;
    double output_prepare_ms = 0.0;
    double coset_prepare_ms = 0.0;
    double ntt_execute_ms = 0.0;
    double output_copy_ms = 0.0;
    double cleanup_ms = 0.0;
    double stream_sync_ms = 0.0;
    bool arbitrary_coset_used = false;
    bool arbitrary_coset_cache_hit = false;
    bool used_radix2_algorithm = true;
    bool used_fast_twiddles = false;

    if (size > domain.max_size) {
      std::ostringstream oss;
      oss << "NTT size=" << size << " is too large for the domain (size=" << domain.max_size
          << "). Consider generating your domain with a higher order root of unity.\n ";
      THROW_ICICLE_ERR(eIcicleError::INVALID_ARGUMENT, oss.str().c_str());
    }

    int logn = int(log2(size));
    const bool is_size_power_of_two = size == (1 << logn);
    if (!is_size_power_of_two) {
      std::ostringstream oss;
      oss << "NTT size=" << size << " is not supported since it is not a power of two.\n";
      THROW_ICICLE_ERR(eIcicleError::INVALID_ARGUMENT, oss.str().c_str());
    }

    size_t batch_size = config.batch_size;
    size_t input_size_bytes = (size_t)size * batch_size * sizeof(E);
    bool are_inputs_on_device = config.are_inputs_on_device;
    bool are_outputs_on_device = config.are_outputs_on_device;
    auto emit_profile = [&](const char* sync_mode) {
      if (!profile_ntt_full) return;
      const double total_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - total_start).count();
      std::lock_guard<std::mutex> lock(ntt_profile_log_mutex);
      std::cerr << "[ICICLE][NTT_PROFILE_FULL] ntt_cuda"
                << " call_id=" << profile_call_id << " device_id=" << device_id
                << " dir=" << (dir == NTTDir::kInverse ? "inverse" : "forward") << " size=" << size
                << " batch=" << batch_size << " ordering=" << static_cast<uint32_t>(config.ordering)
                << " in_dev=" << (are_inputs_on_device ? "1" : "0")
                << " out_dev=" << (are_outputs_on_device ? "1" : "0")
                << " arbitrary_coset_used=" << (arbitrary_coset_used ? "1" : "0")
                << " arbitrary_coset_cache_hit=" << (arbitrary_coset_cache_hit ? "1" : "0")
                << " radix2=" << (used_radix2_algorithm ? "1" : "0")
                << " fast_twiddles=" << (used_fast_twiddles ? "1" : "0") << " input_prepare_ms=" << input_prepare_ms
                << " output_prepare_ms=" << output_prepare_ms << " coset_prepare_ms=" << coset_prepare_ms
                << " ntt_execute_ms=" << ntt_execute_ms << " output_copy_ms=" << output_copy_ms
                << " cleanup_ms=" << cleanup_ms << " stream_sync_ms=" << stream_sync_ms << " total_ms=" << total_ms
                << " sync_mode=" << sync_mode << std::endl;
    };

    const E* d_input;
    E* d_allocated_input = nullptr;
    if (are_inputs_on_device) {
      d_input = input;
    } else {
      const auto step_start = profile_ntt_full ? profile_clock::now() : profile_clock::time_point{};
      CHK_IF_RETURN(cudaMallocAsync(&d_allocated_input, input_size_bytes, cuda_stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_allocated_input, input, input_size_bytes, cudaMemcpyHostToDevice, cuda_stream));
      d_input = d_allocated_input;
      if (profile_ntt_full) {
        input_prepare_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - step_start).count();
      }
    }
    E* d_output;
    E* d_allocated_output = nullptr;
    if (are_outputs_on_device) {
      d_output = output;
    } else {
      const auto step_start = profile_ntt_full ? profile_clock::now() : profile_clock::time_point{};
      CHK_IF_RETURN(cudaMallocAsync(&d_allocated_output, input_size_bytes, cuda_stream));
      d_output = d_allocated_output;
      if (profile_ntt_full) {
        output_prepare_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - step_start).count();
      }
    }

    S* coset = nullptr;
    int coset_index = 0;
    const auto coset_start = profile_ntt_full ? profile_clock::now() : profile_clock::time_point{};
    try {
      coset_index = domain.coset_index.at(config.coset_gen);
    } catch (...) {
      arbitrary_coset_used = true;
      const bool profile_arbitrary_coset = is_arbitrary_coset_profile_enabled();
      const auto profile_start = profile_arbitrary_coset ? profile_clock::now() : profile_clock::time_point{};
      bool cache_hit = false;

      // Fallback for arbitrary cosets: cache [1, g, g^2, ...] on device and reuse it across calls.
      const S coset_gen = (dir == NTTDir::kInverse) ? S::inverse(config.coset_gen) : config.coset_gen;
      {
        std::lock_guard<std::mutex> lock(Domain<S>::device_domain_mutex);
        auto it = domain.arbitrary_coset_cache.find(coset_gen);
        if (it != domain.arbitrary_coset_cache.end()) {
          coset = it->second;
          cache_hit = true;
        }
      }

      if (!coset) {
        const size_t coset_size = static_cast<size_t>(domain.max_size);
        const size_t coset_bytes = coset_size * sizeof(S);
        S* d_coset = nullptr;
        CHK_IF_RETURN(cudaMallocAsync(&d_coset, coset_bytes, cuda_stream));
        const uint32_t threads = 256;
        uint64_t blocks64 = (coset_size + threads - 1) / threads;
        if (blocks64 > 4096ULL) blocks64 = 4096ULL;
        const uint32_t blocks = static_cast<uint32_t>(blocks64);
        generate_arbitrary_coset_powers_kernel<<<blocks, threads, 0, cuda_stream>>>(
          d_coset, static_cast<uint64_t>(coset_size), coset_gen);
        CHK_IF_RETURN(CHK_LAST());
        CHK_IF_RETURN(cudaStreamSynchronize(cuda_stream)); // publish cache entry only after generation is complete

        std::lock_guard<std::mutex> lock(Domain<S>::device_domain_mutex);
        auto insert_result = domain.arbitrary_coset_cache.emplace(coset_gen, d_coset);
        if (insert_result.second) {
          coset = d_coset;
        } else {
          CHK_IF_RETURN(cudaFreeAsync(d_coset, cuda_stream));
          coset = insert_result.first->second;
          cache_hit = true;
        }
      }
      arbitrary_coset_cache_hit = cache_hit;

      if (profile_arbitrary_coset) {
        const double total_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - profile_start).count();
        std::lock_guard<std::mutex> lock(ntt_profile_log_mutex);
        std::cerr << "[ICICLE][NTT_PROFILE_ARBITRARY_COSET]"
                  << " cache=" << (cache_hit ? "hit" : "miss") << " size=" << size << " domain_size=" << domain.max_size
                  << " total_ms=" << total_ms << std::endl;
      }
    }
    if (profile_ntt_full) {
      coset_prepare_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - coset_start).count();
    }

    const bool is_inverse = dir == NTTDir::kInverse;
    const auto ntt_start = profile_ntt_full ? profile_clock::now() : profile_clock::time_point{};

    if constexpr (IS_ECNTT) {
      used_radix2_algorithm = true;
      CHK_IF_RETURN(ntt::radix2_ntt(
        d_input, d_output, domain.twiddles, size, domain.max_size, batch_size, is_inverse, config.ordering, coset,
        coset_index, cuda_stream));
    } else {
      used_radix2_algorithm = is_choosing_radix2_algorithm(logn, batch_size, config);
      if (used_radix2_algorithm) {
        CHK_IF_RETURN(ntt::radix2_ntt(
          d_input, d_output, domain.twiddles, size, domain.max_size, batch_size, is_inverse, config.ordering, coset,
          coset_index, cuda_stream));
      } else {
        const bool is_fast_twiddles_enabled = (domain.fast_external_twiddles != nullptr);
        used_fast_twiddles = is_fast_twiddles_enabled;
        S* twiddles = is_fast_twiddles_enabled
                        ? (is_inverse ? domain.fast_external_twiddles_inv : domain.fast_external_twiddles)
                        : domain.twiddles;
        S* internal_twiddles = is_fast_twiddles_enabled
                                 ? (is_inverse ? domain.fast_internal_twiddles_inv : domain.fast_internal_twiddles)
                                 : domain.internal_twiddles;
        S* basic_twiddles = is_fast_twiddles_enabled
                              ? (is_inverse ? domain.fast_basic_twiddles_inv : domain.fast_basic_twiddles)
                              : domain.basic_twiddles;
        S* linear_twiddles = domain.twiddles; // twiddles organized as [1,w,w^2,...]
        CHK_IF_RETURN(mxntt::mixed_radix_ntt(
          d_input, d_output, twiddles, internal_twiddles, basic_twiddles, linear_twiddles, size, domain.max_log_size,
          batch_size, config.columns_batch, is_inverse, is_fast_twiddles_enabled, config.ordering, coset, coset_index,
          cuda_stream));
      }
    }
    if (profile_ntt_full) {
      ntt_execute_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - ntt_start).count();
    }

    if (!are_outputs_on_device) {
      const auto step_start = profile_ntt_full ? profile_clock::now() : profile_clock::time_point{};
      CHK_IF_RETURN(cudaMemcpyAsync(output, d_output, input_size_bytes, cudaMemcpyDeviceToHost, cuda_stream));
      if (profile_ntt_full) {
        output_copy_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - step_start).count();
      }
    }

    const auto cleanup_start = profile_ntt_full ? profile_clock::now() : profile_clock::time_point{};
    if (d_allocated_input) CHK_IF_RETURN(cudaFreeAsync(d_allocated_input, cuda_stream));
    if (d_allocated_output) CHK_IF_RETURN(cudaFreeAsync(d_allocated_output, cuda_stream));
    if (profile_ntt_full) {
      cleanup_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - cleanup_start).count();
    }
    if (!config.is_async) {
      const auto sync_start = profile_ntt_full ? profile_clock::now() : profile_clock::time_point{};
      cudaError_t sync_err = CHK_STICKY(cudaStreamSynchronize(cuda_stream));
      if (profile_ntt_full) {
        stream_sync_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - sync_start).count();
      }
      emit_profile("sync");
      return sync_err;
    }

    emit_profile("async");

    return CHK_LAST();
  }

} // namespace ntt
