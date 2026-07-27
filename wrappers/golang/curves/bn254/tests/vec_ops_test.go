package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/core"
	bn254 "github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/curves/bn254"
	"github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/curves/bn254/vecOps"
	"github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testBn254VecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := bn254.GenerateScalars(testSize)
	b := bn254.GenerateScalars(testSize)
	var scalar bn254.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bn254.ScalarField], testSize)
	out2 := make(core.HostSlice[bn254.ScalarField], testSize)
	out3 := make(core.HostSlice[bn254.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testBn254BatchInverseVecInPlaceOnDevice(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()

	testSize := 1 << 14
	in := make(core.HostSlice[bn254.ScalarField], testSize)
	expected := make(core.HostSlice[bn254.ScalarField], testSize)
	for i := range in {
		in[i].FromUint32(uint32(i + 1))
		expected[i] = in[i].Inv()
	}

	var deviceInOut core.DeviceSlice
	in.CopyToDevice(&deviceInOut, true)
	defer deviceInOut.Free()

	cfg := core.DefaultVecOpsConfig()
	err := vecOps.BatchInverseVec(deviceInOut, deviceInOut, cfg)
	suite.Equal(runtime.Success, err, "BatchInverseVec in-place failed")

	actual := make(core.HostSlice[bn254.ScalarField], testSize)
	actual.CopyFromDevice(&deviceInOut)
	suite.Equal(expected, actual)
}

type Bn254VecOpsTestSuite struct {
	suite.Suite
}

func (s *Bn254VecOpsTestSuite) TestBn254VecOps() {
	s.Run("TestBn254VecOps", testWrapper(&s.Suite, testBn254VecOps))
	s.Run("TestBn254BatchInverseVecInPlaceOnDevice", testWrapper(&s.Suite, testBn254BatchInverseVecInPlaceOnDevice))
}

func TestSuiteBn254VecOps(t *testing.T) {
	suite.Run(t, new(Bn254VecOpsTestSuite))
}
