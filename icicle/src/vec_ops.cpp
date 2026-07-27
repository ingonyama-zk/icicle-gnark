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

  /*********************************** SHIFT ***********************************/
  ICICLE_DISPATCHER_INST(VectorShiftDispatcher, vector_shift, unaryVectorOpImpl);

  extern "C" eIcicleError
  CONCAT_EXPAND(FIELD, vector_shift)(const scalar_t* input, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorShiftDispatcher::execute(input, size, *config, output);
  }

  template <>
  eIcicleError vector_shift(const scalar_t* input, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_shift)(input, size, &config, output);
  }

  /*********************************** BATCH INVERSE ***********************************/
  ICICLE_DISPATCHER_INST(VectorBatchInverseDispatcher, vector_batch_inverse, unaryVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_batch_inverse)(
    const scalar_t* input, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorBatchInverseDispatcher::execute(input, size, *config, output);
  }

  template <>
  eIcicleError vector_batch_inverse(const scalar_t* input, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_batch_inverse)(input, size, &config, output);
  }

  /*********************************** MERGE SHARDS BIT-REVERSE ***********************************/
  ICICLE_DISPATCHER_INST(MergeShardsBitReverseDispatcher, merge_shards_bit_reverse, mergeShardsBitReverseImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, merge_shards_bit_reverse)(
    scalar_t** shards, uint64_t rho, uint64_t n, uint64_t mm, const VecOpsConfig* config, scalar_t* output)
  {
    return MergeShardsBitReverseDispatcher::execute((const scalar_t* const*)shards, rho, n, mm, *config, output);
  }

  /*********************************** POLY EVAL AT ***********************************/
  ICICLE_DISPATCHER_INST(PolyEvalAtDispatcher, poly_eval_at, polyEvalAtImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, poly_eval_at)(
    const scalar_t* coeffs, uint64_t n, const scalar_t* point, const VecOpsConfig* config, scalar_t* output)
  {
    return PolyEvalAtDispatcher::execute(coeffs, n, point, *config, output);
  }

  ICICLE_DISPATCHER_INST(PolyEvalAtBitReverseDispatcher, poly_eval_at_bit_reverse, polyEvalAtBitReverseImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, poly_eval_at_bit_reverse)(
    const scalar_t* coeffs,
    uint64_t n,
    uint64_t mm,
    const scalar_t* point,
    const VecOpsConfig* config,
    scalar_t* output)
  {
    return PolyEvalAtBitReverseDispatcher::execute(coeffs, n, mm, point, *config, output);
  }

  /*********************************** DIVIDE BY (X-A) ***********************************/
  ICICLE_DISPATCHER_INST(DivideByXMinusADispatcher, divide_by_x_minus_a, divideByXMinusAImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, divide_by_x_minus_a)(
    const scalar_t* coeffs, uint64_t n, const scalar_t* point, const VecOpsConfig* config, scalar_t* output)
  {
    return DivideByXMinusADispatcher::execute(coeffs, n, point, *config, output);
  }

  /*********************************** SUPPORT IDENTITY ***********************************/
  ICICLE_DISPATCHER_INST(SupportIdentityDispatcher, support_identity, supportIdentityImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, support_identity)(
    const scalar_t* omega,
    const scalar_t* shift,
    uint64_t size_poly,
    uint64_t nb_copies,
    const VecOpsConfig* config,
    scalar_t* output)
  {
    return SupportIdentityDispatcher::execute(omega, shift, size_poly, nb_copies, *config, output);
  }

  template <>
  eIcicleError support_identity(
    const scalar_t* omega,
    const scalar_t* shift,
    uint64_t size_poly,
    uint64_t nb_copies,
    const VecOpsConfig& config,
    scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, support_identity)(omega, shift, size_poly, nb_copies, &config, output);
  }

  /*********************************** GATHER BY INDICES ***********************************/
  ICICLE_DISPATCHER_INST(GatherByIndicesDispatcher, gather_by_indices, gatherByIndicesImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, gather_by_indices)(
    const scalar_t* input,
    uint64_t input_size,
    const int64_t* indices,
    uint64_t size,
    const VecOpsConfig* config,
    scalar_t* output)
  {
    return GatherByIndicesDispatcher::execute(input, input_size, indices, size, *config, output);
  }

  template <>
  eIcicleError gather_by_indices(
    const scalar_t* input,
    uint64_t input_size,
    const int64_t* indices,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, gather_by_indices)(input, input_size, indices, size, &config, output);
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

  /*********************************** ADD BY SCALAR ***********************************/
  ICICLE_DISPATCHER_INST(ScalarAddDispatcher, scalar_add_vec, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, scalar_add_vec)(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarAddDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_add_vec(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, scalar_add_vec)(scalar_a, vec_b, size, &config, output);
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