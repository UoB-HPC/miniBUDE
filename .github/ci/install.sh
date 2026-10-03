#!/usr/bin/env bash
set -euo pipefail

apt_options=(-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
curl_options=(--fail --silent --show-error --location --retry 5 --retry-all-errors --retry-delay 30 --connect-timeout 30)

apt-get "${apt_options[@]}" update
apt-get "${apt_options[@]}" install -y --no-install-recommends cmake ninja-build g++ git curl ca-certificates gpg

if [[ -n ${CI_APT_SOURCE:-} ]]; then
  printf 'Downloading repository key: %s\n' "$CI_APT_KEY"
  curl "${curl_options[@]}" "$CI_APT_KEY" |
    gpg --dearmor --yes -o /usr/share/keyrings/minibude-toolchain.gpg
  printf 'Adding package repository: %s\n' "$CI_APT_SOURCE"
  printf 'deb [signed-by=/usr/share/keyrings/minibude-toolchain.gpg] %s\n' \
    "$CI_APT_SOURCE" > /etc/apt/sources.list.d/minibude-toolchain.list
  repo_host=${CI_APT_SOURCE#*://}
  repo_host=${repo_host%%/*}
  printf 'Package: *\nPin: origin %s\nPin-Priority: 600\n' "$repo_host" \
    > /etc/apt/preferences.d/minibude-toolchain
  apt-get "${apt_options[@]}" update
fi

read -r -a packages <<< "$CI_PACKAGES"
if ((${#packages[@]})); then
  apt-get "${apt_options[@]}" install -y --no-install-recommends "${packages[@]}"
fi

if [[ -n ${CI_KOKKOS_REF:-} ]]; then
  printf 'Cloning Kokkos %s\n' "$CI_KOKKOS_REF"
  git clone --depth 1 --branch "$CI_KOKKOS_REF" https://github.com/kokkos/kokkos.git /opt/kokkos
fi

if [[ -n ${CI_RAJA_REF:-} ]]; then
  printf 'Cloning RAJA %s\n' "$CI_RAJA_REF"
  git clone --depth 1 --branch "$CI_RAJA_REF" --recurse-submodules --shallow-submodules \
    https://github.com/LLNL/RAJA.git /opt/raja
fi

if [[ -n ${CI_ADAPTIVECPP_SOURCE_URL:-} ]]; then
  acpp_build=$(mktemp -d)
  printf 'Downloading AdaptiveCpp: %s\n' "$CI_ADAPTIVECPP_SOURCE_URL"
  curl "${curl_options[@]}" "$CI_ADAPTIVECPP_SOURCE_URL" -o "$acpp_build/source.tar.gz"
  echo "$CI_ADAPTIVECPP_SHA256" " $acpp_build/source.tar.gz" | sha256sum --check
  tar -xzf "$acpp_build/source.tar.gz" -C "$acpp_build" --strip-components=1
  cmake -S "$acpp_build" -B "$acpp_build/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/adaptivecpp \
    "-DCMAKE_C_COMPILER=$CC" "-DCMAKE_CXX_COMPILER=$CXX" \
    -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-18 \
    -DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=OFF \
    -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF
  cmake --build "$acpp_build/build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
  cmake --install "$acpp_build/build"
  rm -rf "$acpp_build"
fi

"$CC" --version
"$CXX" --version
cmake --version
