package vecOps

// #cgo CFLAGS: -I./include/
// #include "vec_ops.h"
import "C"

import (
	"unsafe"

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

// We will add more functions here as we add more vector operations to the library.

// ScalarMulVec multiplies each element of a vector by a scalar.
// scalar must be a single scalar (length 1).
// vec and out must have the same length.
func ScalarMulVec(scalar, vec, out core.HostOrDeviceSlice, config core.VecOpsConfig) (ret runtime.EIcicleError) {
	scalarPointer, vecPointer, outPointer, cfgPointer, size := core.ScalarVecOpCheck(scalar, vec, out, &config)

	cScalar := (*C.scalar_t)(scalarPointer)
	cVec := (*C.scalar_t)(vecPointer)
	cOut := (*C.scalar_t)(outPointer)
	cConfig := (*C.VecOpsConfig)(cfgPointer)
	cSize := (C.int)(size)

	ret = (runtime.EIcicleError)(C.bls12_377_scalar_mul_vec(cScalar, cVec, cSize, cConfig, cOut))

	return ret
}

// ScalarAddVec adds a scalar to each element of a vector.
// scalar must be a single scalar (length 1).
// vec and out must have the same length.
func ScalarAddVec(scalar, vec, out core.HostOrDeviceSlice, config core.VecOpsConfig) (ret runtime.EIcicleError) {
	scalarPointer, vecPointer, outPointer, cfgPointer, size := core.ScalarVecOpCheck(scalar, vec, out, &config)

	cScalar := (*C.scalar_t)(scalarPointer)
	cVec := (*C.scalar_t)(vecPointer)
	cOut := (*C.scalar_t)(outPointer)
	cConfig := (*C.VecOpsConfig)(cfgPointer)
	cSize := (C.int)(size)

	ret = (runtime.EIcicleError)(C.bls12_377_scalar_add_vec(cScalar, cVec, cSize, cConfig, cOut))

	return ret
}

// ShiftVec performs a cyclic left shift by one position: out[i] = in[(i+1)%n].
func ShiftVec(in, out core.HostOrDeviceSlice, config core.VecOpsConfig) (ret runtime.EIcicleError) {
	inPointer, outPointer, cfgPointer, size := core.UnaryVecOpCheck(in, out, &config)

	cIn := (*C.scalar_t)(inPointer)
	cOut := (*C.scalar_t)(outPointer)
	cConfig := (*C.VecOpsConfig)(cfgPointer)
	cSize := (C.int)(size)

	ret = (runtime.EIcicleError)(C.bls12_377_vector_shift(cIn, cSize, cConfig, cOut))
	return ret
}

// BatchInverseVec computes multiplicative inverses element-wise.
func BatchInverseVec(in, out core.HostOrDeviceSlice, config core.VecOpsConfig) (ret runtime.EIcicleError) {
	inPointer, outPointer, cfgPointer, size := core.UnaryVecOpCheck(in, out, &config)

	cIn := (*C.scalar_t)(inPointer)
	cOut := (*C.scalar_t)(outPointer)
	cConfig := (*C.VecOpsConfig)(cfgPointer)
	cSize := (C.int)(size)

	ret = (runtime.EIcicleError)(C.bls12_377_vector_batch_inverse(cIn, cSize, cConfig, cOut))
	return ret
}

// MergeShardsBitReverse merges rho device shards (each length n) into one device vector
// of length n*rho, writing values at bit-reversed indices:
// result[bitReverse(rho*j+i)] = shards[i][j].
func MergeShardsBitReverse(shards []core.DeviceSlice, n int, mm uint64, out core.DeviceSlice, config core.VecOpsConfig) (ret runtime.EIcicleError) {
	if len(shards) == 0 {
		return runtime.Success
	}
	if n <= 0 {
		panic("MergeShardsBitReverse: n must be > 0")
	}
	if out.Len() != n*len(shards) {
		panic("MergeShardsBitReverse: output length mismatch")
	}

	for i := range shards {
		if shards[i].Len() != n {
			panic("MergeShardsBitReverse: shard length mismatch")
		}
		shards[i].CheckDevice()
	}
	out.CheckDevice()

	// We require device pointers for all inputs and output.
	cShards := make([]*C.scalar_t, len(shards))
	for i := range shards {
		cShards[i] = (*C.scalar_t)(shards[i].AsUnsafePointer())
	}

	cShardPtrs := (**C.scalar_t)(unsafe.Pointer(&cShards[0]))
	cRho := (C.int)(len(shards))
	cN := (C.int)(n)
	cMM := (C.ulonglong)(mm)
	cConfig := (*C.VecOpsConfig)(unsafe.Pointer(&config))
	cOut := (*C.scalar_t)(out.AsUnsafePointer())

	ret = (runtime.EIcicleError)(C.bls12_377_merge_shards_bit_reverse(cShardPtrs, cRho, cN, cMM, cConfig, cOut))
	return ret
}

// PolyEvalAt evaluates a canonical polynomial at a single point.
// coeffs and point must be on device; point and out must have length 1.
func PolyEvalAt(coeffs, point, out core.DeviceSlice, config core.VecOpsConfig) (ret runtime.EIcicleError) {
	if coeffs.IsEmpty() || coeffs.Len() <= 0 {
		panic("PolyEvalAt: coeffs must be non-empty")
	}
	if point.Len() != 1 {
		panic("PolyEvalAt: point must have length 1")
	}
	if out.Len() != 1 {
		panic("PolyEvalAt: out must have length 1")
	}
	coeffs.CheckDevice()
	point.CheckDevice()
	out.CheckDevice()

	cCoeffs := (*C.scalar_t)(coeffs.AsUnsafePointer())
	cPoint := (*C.scalar_t)(point.AsUnsafePointer())
	cOut := (*C.scalar_t)(out.AsUnsafePointer())
	cConfig := (*C.VecOpsConfig)(unsafe.Pointer(&config))
	cN := (C.int)(coeffs.Len())

	ret = (runtime.EIcicleError)(C.bls12_377_poly_eval_at(cCoeffs, cN, cPoint, cConfig, cOut))
	return ret
}

// PolyEvalAtBitReverse evaluates a canonical polynomial at a single point when
// coefficients are stored in bit-reversed order.
func PolyEvalAtBitReverse(coeffs, point core.DeviceSlice, mm uint64, out core.DeviceSlice, config core.VecOpsConfig) (ret runtime.EIcicleError) {
	if coeffs.IsEmpty() || coeffs.Len() <= 0 {
		panic("PolyEvalAtBitReverse: coeffs must be non-empty")
	}
	if point.Len() != 1 {
		panic("PolyEvalAtBitReverse: point must have length 1")
	}
	if out.Len() != 1 {
		panic("PolyEvalAtBitReverse: out must have length 1")
	}
	coeffs.CheckDevice()
	point.CheckDevice()
	out.CheckDevice()

	cCoeffs := (*C.scalar_t)(coeffs.AsUnsafePointer())
	cPoint := (*C.scalar_t)(point.AsUnsafePointer())
	cOut := (*C.scalar_t)(out.AsUnsafePointer())
	cConfig := (*C.VecOpsConfig)(unsafe.Pointer(&config))
	cN := (C.int)(coeffs.Len())
	cMM := (C.ulonglong)(mm)

	ret = (runtime.EIcicleError)(C.bls12_377_poly_eval_at_bit_reverse(cCoeffs, cN, cMM, cPoint, cConfig, cOut))
	return ret
}

// DivideByXMinusA computes q(x) = (f(x)-f(a))/(x-a) for canonical regular coefficients.
// coeffs, point and out must be on device; point length is 1, out length is len(coeffs)-1.
func DivideByXMinusA(coeffs, point, out core.DeviceSlice, config core.VecOpsConfig) (ret runtime.EIcicleError) {
	if coeffs.IsEmpty() || coeffs.Len() <= 1 {
		panic("DivideByXMinusA: coeffs length must be >= 2")
	}
	if point.Len() != 1 {
		panic("DivideByXMinusA: point must have length 1")
	}
	if out.Len() != coeffs.Len()-1 {
		panic("DivideByXMinusA: out length must be len(coeffs)-1")
	}
	coeffs.CheckDevice()
	point.CheckDevice()
	out.CheckDevice()

	cCoeffs := (*C.scalar_t)(coeffs.AsUnsafePointer())
	cPoint := (*C.scalar_t)(point.AsUnsafePointer())
	cOut := (*C.scalar_t)(out.AsUnsafePointer())
	cConfig := (*C.VecOpsConfig)(unsafe.Pointer(&config))
	cN := (C.int)(coeffs.Len())

	ret = (runtime.EIcicleError)(C.bls12_377_divide_by_x_minus_a(cCoeffs, cN, cPoint, cConfig, cOut))
	return ret
}

// SupportIdentity builds flattened identity-permutation support on device:
// out[copy*sizePoly+i] = omega^i * shift^copy.
// omega and shift must be single-element device slices.
func SupportIdentity(omega, shift core.DeviceSlice, sizePoly, nbCopies int, out core.DeviceSlice, config core.VecOpsConfig) (ret runtime.EIcicleError) {
	if omega.Len() != 1 {
		panic("SupportIdentity: omega must have length 1")
	}
	if shift.Len() != 1 {
		panic("SupportIdentity: shift must have length 1")
	}
	if sizePoly < 0 || nbCopies < 0 {
		panic("SupportIdentity: sizePoly and nbCopies must be non-negative")
	}
	if out.Len() != sizePoly*nbCopies {
		panic("SupportIdentity: output length mismatch")
	}
	omega.CheckDevice()
	shift.CheckDevice()
	out.CheckDevice()

	cOmega := (*C.scalar_t)(omega.AsUnsafePointer())
	cShift := (*C.scalar_t)(shift.AsUnsafePointer())
	cSizePoly := (C.int)(sizePoly)
	cNbCopies := (C.int)(nbCopies)
	cConfig := (*C.VecOpsConfig)(unsafe.Pointer(&config))
	cOut := (*C.scalar_t)(out.AsUnsafePointer())

	ret = (runtime.EIcicleError)(C.bls12_377_support_identity(cOmega, cShift, cSizePoly, cNbCopies, cConfig, cOut))
	return ret
}

// GatherByIndices gathers values from input using int64 indices on device.
// input, indices and out must be on device and out.Len() == indices.Len().
func GatherByIndices(input, indices, out core.DeviceSlice, config core.VecOpsConfig) (ret runtime.EIcicleError) {
	if out.Len() != indices.Len() {
		panic("GatherByIndices: output length must match indices length")
	}
	input.CheckDevice()
	indices.CheckDevice()
	out.CheckDevice()

	cInput := (*C.scalar_t)(input.AsUnsafePointer())
	cInputSize := (C.int)(input.Len())
	cIndices := (*C.int64_t)(indices.AsUnsafePointer())
	cN := (C.int)(indices.Len())
	cConfig := (*C.VecOpsConfig)(unsafe.Pointer(&config))
	cOut := (*C.scalar_t)(out.AsUnsafePointer())

	ret = (runtime.EIcicleError)(C.bls12_377_gather_by_indices(cInput, cInputSize, cIndices, cN, cConfig, cOut))
	return ret
}
