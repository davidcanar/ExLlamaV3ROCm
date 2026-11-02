#!/usr/bin/env bash
# Build libodl_ar3.so (the OdinLink mesh decode all-reduce of the cross-host TP backend at world
# size >= 3) with the project's ROCm SDK. Run on EVERY box (the library links the box-local
# libodl_tb5). The library is built next to the old one and renamed over it: rewriting a .so in
# place can crash ranks that have it mapped (a running server keeps using the old inode).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
ODL_SRC="${EXL3_ODL_SRC:-$HOME/.cache/odinlink}"
ODL_BUILD="${EXL3_ODL_BUILD:-$ODL_SRC/build}"
source "$ROOT/env.sh" build
hipcc -O3 --offload-arch=gfx1151 -shared -fPIC -o "$HERE/libodl_ar3.so.new" "$HERE/odl_ar3.hip" \
    -I"$ODL_SRC/lib/include" -L"$ODL_BUILD/lib" -Wl,-rpath,"$ODL_BUILD/lib" -lodl_tb5 -lpthread
mv -f "$HERE/libodl_ar3.so.new" "$HERE/libodl_ar3.so"
echo "built $HERE/libodl_ar3.so"
