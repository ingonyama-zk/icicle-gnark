#include "icicle/backend/vec_ops_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  /*********************************** ADD ***********************************/
  ICICLE_DISPATCHER_INST(VectorAddDispatcher, vector_add, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_add)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorAddDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_add(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_add)(vec_a, vec_b, size, &config, output);
  }

  /*********************************** SUB ***********************************/
  ICICLE_DISPATCHER_INST(VectorSubDispatcher, vector_sub, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_sub)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorSubDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_sub(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_sub)(vec_a, vec_b, size, &config, output);
  }

  /*********************************** MUL ***********************************/
  ICICLE_DISPATCHER_INST(VectorMulDispatcher, vector_mul, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_mul)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorMulDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_mul(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_mul)(vec_a, vec_b, size, &config, output);
  }

  /*********************************** MUL BY SCALAR ***********************************/
  ICICLE_DISPATCHER_INST(ScalarMulDispatcher, scalar_mul_vec, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, scalar_mul_vec)(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarMulDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_mul_vec(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, scalar_mul_vec)(scalar_a, vec_b, size, &config, output);
  }

  /*********************************** CONVERT MONTGOMERY ***********************************/

  ICICLE_DISPATCHER_INST(ScalarConvertMontgomeryDispatcher, scalar_convert_montgomery, scalarConvertMontgomeryImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, scalar_convert_montgomery)(
    const scalar_t* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarConvertMontgomeryDispatcher::execute(input, size, is_to_montgomery, *config, output);
  }

  template <>
  eIcicleError convert_montgomery(
    const scalar_t* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, scalar_convert_montgomery)(input, size, is_to_montgomery, &config, output);
  }

} // namespace icicle