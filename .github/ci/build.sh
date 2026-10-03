#!/usr/bin/env bash
set -euo pipefail

if [[ -n ${CI_ENV_SCRIPT:-} ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$CI_ENV_SCRIPT"
  set -u
fi

read -r -a model_flags <<< "$CI_CMAKE_ARGS"
if [[ $CI_MODEL == ocl ]]; then
  model_flags+=("-DOpenCL_LIBRARY=/usr/lib/$($CXX -print-multiarch)/libOpenCL.so")
fi
if [[ $CI_MODEL == thrust && -d ${CI_CUDA_SDK_DIR:-} ]]; then
  model_flags+=("-DSDK_DIR=$CI_CUDA_SDK_DIR")
fi
native_flags=()
if [[ -n ${ACT:-} ]]; then
  native_flags=("-DCXX_EXTRA_FLAGS=$CI_NATIVE_CXX_FLAGS")
  if [[ -n ${CI_NATIVE_CUDA_FLAGS:-} ]]; then
    native_flags+=("-DCUDA_EXTRA_FLAGS=$CI_NATIVE_CUDA_FLAGS")
  fi
fi

cmake -S . -B "$CI_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release "-DBUILD_TESTING=$BUILD_TESTING" -DUSAGE=OFF "-DCMAKE_CXX_COMPILER=$CXX" \
  "-DMODEL=$CI_MODEL" "${native_flags[@]}" "${model_flags[@]}"
cmake --build "$CI_BUILD_DIR"
cmake --install "$CI_BUILD_DIR" --prefix "$CI_BUILD_DIR/install"

if [[ $BUILD_TESTING == ON ]]; then
  ctest --test-dir "$CI_BUILD_DIR" --output-on-failure --no-tests=error -L solution
fi
