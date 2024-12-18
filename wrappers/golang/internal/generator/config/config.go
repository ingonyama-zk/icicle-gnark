package config

type CurveData struct {
	PackageName         string
	Curve               string
	GnarkImport         string
	SupportsNTT         bool
	SupportsG2          bool
	ScalarFieldNumLimbs int
	BaseFieldNumLimbs   int
	G2FieldNumLimbs     int
}

var Curves []CurveData

func addCurve(curve CurveData) {
	Curves = append(Curves, curve)
}
