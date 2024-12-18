#!/bin/bash

MSM_DEFINED=ON
NTT_DEFINED=ON
G2_DEFINED=ON

CUDA_COMPILER_PATH=/usr/local/cuda/bin/nvcc

DEVMODE=OFF
BUILD_CURVES=( )

SUPPORTED_CURVES=("bn254" "bls12_377" "bls12_381" "bw6_761")

BUILD_DIR="${ICICLE_BUILD_DIR:-$(realpath "$PWD/../../icicle/build")}"

if [[ $1 == "-help" ]]; then
  echo "Build script for building ICICLE cpp libraries"
  echo ""
  echo "If more than one curve is supplied, the last one supplied will be built"
  echo ""
  echo "USAGE: ./build.sh [OPTION...]"
  echo ""
  echo "OPTIONS:"
  echo "  -curve=<curve_name>       Specifies the curve to be built. If \"all\" is supplied,"
  echo "                            all curves will be built with any additional curve options."
  echo ""
  echo "  -skip_msm                 Builds the curve library with MSM (multi-scalar multiplication) disabled."
  echo ""
  echo "  -skip_ntt                 Builds the curve/field library with NTT (number theoretic transform) disabled."
  echo ""
  echo "  -skip_g2                  Builds the curve library with G2 (a secondary group) disabled."
  echo ""
  echo "  -cuda_version=<version>   Specifies the version of CUDA to use for compilation."
  echo ""
  exit 0
fi

for arg in "$@"
do
    arg_lower=$(echo "$arg" | tr '[:upper:]' '[:lower:]')
    case "$arg_lower" in
        -cuda_version=*)
            cuda_version=$(echo "$arg" | cut -d'=' -f2)
            CUDA_COMPILER_PATH=/usr/local/cuda-$cuda_version/bin/nvcc
            ;;
        -skip_msm)
            MSM_DEFINED=OFF
            ;;
        -skip_ntt)
            NTT_DEFINED=OFF
            ;;
        -skip_g2)
            G2_DEFINED=OFF
            ;;
        -curve=*)
            curve=$(echo "$arg_lower" | cut -d'=' -f2)
            if [[ $curve == "all" ]]
            then
              BUILD_CURVES=("${SUPPORTED_CURVES[@]}")
            else
              BUILD_CURVES=( $curve )
            fi
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

cd ../../icicle
mkdir -p build
rm -f "$BUILD_DIR/CMakeCache.txt"

for CURVE in "${BUILD_CURVES[@]}"
do
  echo "CURVE=${CURVE}" > build_config.txt
  echo "MSM=${MSM_DEFINED}" >> build_config.txt
  echo "NTT=${NTT_DEFINED}" >> build_config.txt
  echo "G2=${G2_DEFINED}" >> build_config.txt
  echo "DEVMODE=${DEVMODE}" >> build_config.txt
  cmake -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER_PATH \
        -DCURVE=$CURVE \
        -DMSM=$MSM_DEFINED \
        -DNTT=$NTT_DEFINED \
        -DG2=$G2_DEFINED \
        -DCMAKE_BUILD_TYPE=Release \
        -S . -B build
  cmake --build build --target install -j8 && rm build_config.txt
  rm -f "$BUILD_DIR/CMakeCache.txt"
done
