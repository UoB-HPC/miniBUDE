#!/usr/bin/env bash

set -eu

#MODEL=? A=B

BUILD_DIR="./build"

rm -rf "$BUILD_DIR"

cmake -B "$BUILD_DIR" -H. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMODEL="$model" $flags &>>"$log"

cmake --build "$BUILD_DIR" -j "$(nproc)"