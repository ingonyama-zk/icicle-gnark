package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bw6_761 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761/vecOps"
	"github.com/stretchr/testify/suite"
)

func testBw6_761VecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := bw6_761.GenerateScalars(testSize)
	b := bw6_761.GenerateScalars(testSize)
	var scalar bw6_761.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bw6_761.ScalarField], testSize)
	out2 := make(core.HostSlice[bw6_761.ScalarField], testSize)
	out3 := make(core.HostSlice[bw6_761.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

type Bw6_761VecOpsTestSuite struct {
	suite.Suite
}

func (s *Bw6_761VecOpsTestSuite) TestBw6_761VecOps() {
	s.Run("TestBw6_761VecOps", testWrapper(&s.Suite, testBw6_761VecOps))
}

func TestSuiteBw6_761VecOps(t *testing.T) {
	suite.Run(t, new(Bw6_761VecOpsTestSuite))
}
