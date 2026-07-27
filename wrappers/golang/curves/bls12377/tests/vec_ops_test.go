package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/core"
	bls12_377 "github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/curves/bls12377"
	"github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/curves/bls12377/vecOps"
	"github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle-gnark/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testBls12_377VecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := bls12_377.GenerateScalars(testSize)
	b := bls12_377.GenerateScalars(testSize)
	var scalar bls12_377.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bls12_377.ScalarField], testSize)
	out2 := make(core.HostSlice[bls12_377.ScalarField], testSize)
	out3 := make(core.HostSlice[bls12_377.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testBls12_377BatchInverseVecInPlaceOnDevice(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()

	testSize := 1 << 14
	in := make(core.HostSlice[bls12_377.ScalarField], testSize)
	expected := make(core.HostSlice[bls12_377.ScalarField], testSize)
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

	actual := make(core.HostSlice[bls12_377.ScalarField], testSize)
	actual.CopyFromDevice(&deviceInOut)
	suite.Equal(expected, actual)
}

type Bls12_377VecOpsTestSuite struct {
	suite.Suite
}

func (s *Bls12_377VecOpsTestSuite) TestBls12_377VecOps() {
	s.Run("TestBls12_377VecOps", testWrapper(&s.Suite, testBls12_377VecOps))
	s.Run("TestBls12_377BatchInverseVecInPlaceOnDevice", testWrapper(&s.Suite, testBls12_377BatchInverseVecInPlaceOnDevice))
}

func TestSuiteBls12_377VecOps(t *testing.T) {
	suite.Run(t, new(Bls12_377VecOpsTestSuite))
}
