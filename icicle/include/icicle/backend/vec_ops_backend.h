#pragma once

#include <cstdint>

#include "icicle/vec_ops.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {
  /*************************** Backend registration ***************************/

  using scalarConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    uint64_t size,
    bool is_to_montgomery,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using scalarVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* scalar_a,
    const scalar_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using unaryVectorOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, uint64_t size, const VecOpsConfig& config, scalar_t* output)>;

  using mergeShardsBitReverseImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* const* shards,
    uint64_t rho,
    uint64_t n,
    uint64_t mm,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using polyEvalAtImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* coeffs,
    uint64_t n,
    const scalar_t* point,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using polyEvalAtBitReverseImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* coeffs,
    uint64_t n,
    uint64_t mm,
    const scalar_t* point,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using divideByXMinusAImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* coeffs,
    uint64_t n,
    const scalar_t* point,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using supportIdentityImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* omega,
    const scalar_t* shift,
    uint64_t size_poly,
    uint64_t nb_copies,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using gatherByIndicesImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    uint64_t input_size,
    const int64_t* indices,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_t* output)>;

  void register_vector_add(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_ADD_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_add) = []() -> bool {                                                                  \
      register_vector_add(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_sub(const std::string& deviceType, scalarVectorOpImpl impl);
#define REGISTER_VECTOR_SUB_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_sub) = []() -> bool {                                                                  \
      register_vector_sub(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_mul(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_mul) = []() -> bool {                                                                  \
      register_vector_mul(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_shift(const std::string& deviceType, unaryVectorOpImpl impl);

#define REGISTER_VECTOR_SHIFT_BACKEND(DEVICE_TYPE, FUNC)                                                               \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_shift) = []() -> bool {                                                                \
      register_vector_shift(DEVICE_TYPE, FUNC);                                                                        \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_batch_inverse(const std::string& deviceType, unaryVectorOpImpl impl);

#define REGISTER_VECTOR_BATCH_INVERSE_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_batch_inverse) = []() -> bool {                                                        \
      register_vector_batch_inverse(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_convert_montgomery(const std::string& deviceType, scalarConvertMontgomeryImpl);

#define REGISTER_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                                         \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont) = []() -> bool {                                                      \
      register_scalar_convert_montgomery(DEVICE_TYPE, FUNC);                                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_mul_vec(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_SCALAR_MUL_VEC_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_mul_vec) = []() -> bool {                                                           \
      register_scalar_mul_vec(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_add_vec(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_SCALAR_ADD_VEC_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_add_vec) = []() -> bool {                                                           \
      register_scalar_add_vec(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_merge_shards_bit_reverse(const std::string& deviceType, mergeShardsBitReverseImpl impl);

#define REGISTER_MERGE_SHARDS_BIT_REVERSE_BACKEND(DEVICE_TYPE, FUNC)                                                   \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_merge_shards_bit_reverse) = []() -> bool {                                                 \
      register_merge_shards_bit_reverse(DEVICE_TYPE, FUNC);                                                            \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_poly_eval_at(const std::string& deviceType, polyEvalAtImpl impl);

#define REGISTER_POLY_EVAL_AT_BACKEND(DEVICE_TYPE, FUNC)                                                               \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_poly_eval_at) = []() -> bool {                                                             \
      register_poly_eval_at(DEVICE_TYPE, FUNC);                                                                        \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_poly_eval_at_bit_reverse(const std::string& deviceType, polyEvalAtBitReverseImpl impl);

#define REGISTER_POLY_EVAL_AT_BIT_REVERSE_BACKEND(DEVICE_TYPE, FUNC)                                                   \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_poly_eval_at_bit_reverse) = []() -> bool {                                                 \
      register_poly_eval_at_bit_reverse(DEVICE_TYPE, FUNC);                                                            \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_divide_by_x_minus_a(const std::string& deviceType, divideByXMinusAImpl impl);

#define REGISTER_DIVIDE_BY_X_MINUS_A_BACKEND(DEVICE_TYPE, FUNC)                                                        \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_divide_by_x_minus_a) = []() -> bool {                                                      \
      register_divide_by_x_minus_a(DEVICE_TYPE, FUNC);                                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_support_identity(const std::string& deviceType, supportIdentityImpl impl);

#define REGISTER_SUPPORT_IDENTITY_BACKEND(DEVICE_TYPE, FUNC)                                                           \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_support_identity) = []() -> bool {                                                         \
      register_support_identity(DEVICE_TYPE, FUNC);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_gather_by_indices(const std::string& deviceType, gatherByIndicesImpl impl);

#define REGISTER_GATHER_BY_INDICES_BACKEND(DEVICE_TYPE, FUNC)                                                          \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_gather_by_indices) = []() -> bool {                                                        \
      register_gather_by_indices(DEVICE_TYPE, FUNC);                                                                   \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle
