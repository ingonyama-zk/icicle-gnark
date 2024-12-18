package main

import (
	"fmt"
	"os"
	"os/exec"
	"path"

	config "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/config"
	curves "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/curves"
	fields "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/fields"
	lib_linker "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/lib_linker"
	mock "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/mock"
	msm "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/msm"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/tests"
	vecops "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/vecOps"
)

func generateFiles() {
	fmt.Println("Generating files")

	for _, curve := range config.Curves {
		curveDir := path.Join("curves", curve.PackageName)
		scalarFieldPrefix := "Scalar"
		fields.Generate(curveDir, curve.PackageName, curve.Curve, scalarFieldPrefix, true, curve.ScalarFieldNumLimbs)
		fields.Generate(curveDir, curve.PackageName, curve.Curve, "Base", false, curve.BaseFieldNumLimbs)
		curves.Generate(curveDir, curve.PackageName, curve.Curve, "")
		vecops.Generate(curveDir, curve.Curve, scalarFieldPrefix)

		lib_linker.Generate(curveDir, curve.PackageName, curve.Curve, lib_linker.CURVE, 0)

		if curve.SupportsNTT {
			ntt.Generate(curveDir, "", curve.Curve, scalarFieldPrefix, curve.GnarkImport, 0, true, "", "")
		}

		msm.Generate(curveDir, "msm", curve.Curve, "", curve.GnarkImport)
		if curve.SupportsG2 {
			g2BaseDir := path.Join(curveDir, "g2")
			packageName := "g2"
			fields.Generate(g2BaseDir, packageName, curve.Curve, "G2Base", false, curve.G2FieldNumLimbs)
			curves.Generate(g2BaseDir, packageName, curve.Curve, "G2")
			msm.Generate(curveDir, "g2", curve.Curve, "G2", curve.GnarkImport)
		}
		tests.Generate(curveDir, curve.Curve, scalarFieldPrefix, curve.GnarkImport, 0, curve.SupportsNTT)
	}

	// Mock field and curve files for core
	mock.Generate("core/internal", "internal", "Mock", "MockBase", false, 4)
}

//go:generate go run main.go
func main() {
	generateFiles()

	cmd := exec.Command("gofmt", "-w", "../../")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		fmt.Printf("\n%s\n", err.Error())
		os.Exit(-1)
	}
}
