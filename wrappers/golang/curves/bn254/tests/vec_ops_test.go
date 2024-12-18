package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bn254 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/vecOps"
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

type Bn254VecOpsTestSuite struct {
	suite.Suite
}

func (s *Bn254VecOpsTestSuite) TestBn254VecOps() {
	s.Run("TestBn254VecOps", testWrapper(&s.Suite, testBn254VecOps))
}

func TestSuiteBn254VecOps(t *testing.T) {
	suite.Run(t, new(Bn254VecOpsTestSuite))
}
