package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bls12_381 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381/vecOps"
	"github.com/stretchr/testify/suite"
)

func testBls12_381VecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := bls12_381.GenerateScalars(testSize)
	b := bls12_381.GenerateScalars(testSize)
	var scalar bls12_381.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bls12_381.ScalarField], testSize)
	out2 := make(core.HostSlice[bls12_381.ScalarField], testSize)
	out3 := make(core.HostSlice[bls12_381.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

type Bls12_381VecOpsTestSuite struct {
	suite.Suite
}

func (s *Bls12_381VecOpsTestSuite) TestBls12_381VecOps() {
	s.Run("TestBls12_381VecOps", testWrapper(&s.Suite, testBls12_381VecOps))
}

func TestSuiteBls12_381VecOps(t *testing.T) {
	suite.Run(t, new(Bls12_381VecOpsTestSuite))
}
