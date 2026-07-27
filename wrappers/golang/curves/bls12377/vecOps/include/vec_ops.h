#include <stdbool.h>
#include <stdint.h>

#ifndef _BLS12_377_VEC_OPS_H
#define _BLS12_377_VEC_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;
typedef struct DeviceContext DeviceContext;

int bls12_377_vector_mul(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_vector_add(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_vector_sub(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_vector_shift(
  scalar_t* vec_in,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_vector_batch_inverse(
  scalar_t* vec_in,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_scalar_mul_vec(
  scalar_t* scalar_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_scalar_add_vec(
  scalar_t* scalar_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_merge_shards_bit_reverse(
  scalar_t** shards,
  int rho,
  int n,
  unsigned long long mm,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_poly_eval_at(
  scalar_t* coeffs,
  int n,
  scalar_t* point,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_poly_eval_at_bit_reverse(
  scalar_t* coeffs,
  int n,
  unsigned long long mm,
  scalar_t* point,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_divide_by_x_minus_a(
  scalar_t* coeffs,
  int n,
  scalar_t* point,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_support_identity(
  scalar_t* omega,
  scalar_t* shift,
  int size_poly,
  int nb_copies,
  VecOpsConfig* config,
  scalar_t* result
);

int bls12_377_gather_by_indices(
  scalar_t* input,
  int input_size,
  int64_t* indices,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

#ifdef __cplusplus
}
#endif

#endif
