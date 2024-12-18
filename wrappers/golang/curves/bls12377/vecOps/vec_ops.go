package vecOps

// #cgo CFLAGS: -I./include/
// #include "vec_ops.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/runtime"
)

func VecOp(a, b, out core.HostOrDeviceSlice, config core.VecOpsConfig, op core.VecOps) (ret runtime.EIcicleError) {
	aPointer, bPointer, outPointer, cfgPointer, size := core.VecOpCheck(a, b, out, &config)

	cA := (*C.scalar_t)(aPointer)
	cB := (*C.scalar_t)(bPointer)
	cOut := (*C.scalar_t)(outPointer)
	cConfig := (*C.VecOpsConfig)(cfgPointer)
	cSize := (C.int)(size)

	switch op {
	case core.Sub:
		ret = (runtime.EIcicleError)(C.bls12_377_vector_sub(cA, cB, cSize, cConfig, cOut))
	case core.Add:
		ret = (runtime.EIcicleError)(C.bls12_377_vector_add(cA, cB, cSize, cConfig, cOut))
	case core.Mul:
		ret = (runtime.EIcicleError)(C.bls12_377_vector_mul(cA, cB, cSize, cConfig, cOut))
	}

	return ret
}
