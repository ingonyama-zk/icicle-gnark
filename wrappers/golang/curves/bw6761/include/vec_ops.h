#include <cuda_runtime.h>
#include "../../include/types.h"

#ifndef _BW6_761_VEC_OPS_H
#define _BW6_761_VEC_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bw6_761MulCuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t bw6_761AddCuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t bw6_761SubCuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

#ifdef __cplusplus
}
#endif

#endif
