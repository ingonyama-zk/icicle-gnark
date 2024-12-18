#pragma once

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

} // namespace icicle