#!/usr/bin/env bash
# Build libodl_ar2.so (the OdinLink decode all-reduce of the cross-host TP backend) with the
# project's ROCm SDK. Run on BOTH boxes (the library links the box-local libodl_tb5).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
ODL_SRC="${EXL3_ODL_SRC:-$HOME/.cache/odinlink}"
ODL_BUILD="${EXL3_ODL_BUILD:-$ODL_SRC/build}"
source "$ROOT/env.sh" build
hipcc -O3 --offload-arch=gfx1151 -shared -fPIC -o "$HERE/libodl_ar2.so" "$HERE/odl_ar2.hip" \
    -I"$ODL_SRC/lib/include" -L"$ODL_BUILD/lib" -Wl,-rpath,"$ODL_BUILD/lib" -lodl_tb5 -lpthread
echo "built $HERE/libodl_ar2.so"
