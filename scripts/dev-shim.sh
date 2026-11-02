#!/bin/sh
# Rebuilds only the occt_capi shim against an existing OCCT build. For development.
#   OCCT_PREFIX=<dir with lib/cmake/opencascade> scripts/dev-shim.sh <output dir>
# Then: export OCCT_CAPI_LIB_DIR=<output dir>/lib LD_LIBRARY_PATH=<output dir>/lib
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
out=$1
prefix=${OCCT_PREFIX:-$root/.cache/native/occt}
cmake -S "$root/crates/occt-capi-sys/capi" -B "$out/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON "-DCMAKE_INSTALL_PREFIX=$out" \
  "-DOpenCASCADE_DIR=$prefix/lib/cmake/opencascade" >/dev/null
cmake --build "$out/build" --parallel "${OCCT_BUILD_JOBS:-2}"
cmake --install "$out/build" >/dev/null
