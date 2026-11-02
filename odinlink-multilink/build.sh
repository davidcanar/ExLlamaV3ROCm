#!/usr/bin/env bash
# build.sh — build the multi-cable OdinLink fabric for one kernel, from THIS
# repository's patch series (the single source of truth for the deployment):
#
#   patches/linux/0001-*                       thunderbolt core, source-XDomain
#                                              protocol handlers (ABI-preserving)
#   patches/odinlink/0001..0005                OdinLink-Five driver + RCCL plugin
#                                              (0001-0003 local series, 0004-0005
#                                              multi-link demux + retry bound)
#
# Outputs under ~/.cache/odl-multi/<KVER>/out:
#   thunderbolt.ko  odl_tb5.ko  lib/libodl_tb5.so
#
# Usage: odinlink-multilink/build.sh [KVER]        (default: running kernel)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
KVER=${1:-$(uname -r)}
CACHE=${ODL_MULTI_CACHE:-$HOME/.cache/odl-multi}
WORK=$CACHE/$KVER
OUT=$WORK/out
ODL_REPO=${ODL_REPO:-https://github.com/Geramy/OdinLink-Five}
ODL_PIN=${ODL_PIN:-4534f58}

KDEV=""
for d in "/usr/src/kernels/$KVER" "/lib/modules/$KVER/build"; do
  [ -f "$d/Makefile" ] && KDEV=$(readlink -f "$d") && break
done
[ -n "$KDEV" ] || { echo "!! no kernel-devel for $KVER (dnf install kernel-devel-$KVER)"; exit 1; }
echo "== target kernel: $KVER  (devel: $KDEV) =="

# ── kernel sources: only drivers/thunderbolt + the thunderbolt header ──
if [ ! -f "$WORK/src/thunderbolt.h.orig" ]; then
  rm -rf "$WORK/src"; mkdir -p "$WORK/src"
  V=${KVER%%-*}; REL=${KVER#*-}; REL=${REL%.*}
  SRPM=$CACHE/kernel-$KVER.src.rpm
  URL="https://kojipkgs.fedoraproject.org/packages/kernel/$V/$REL/src/kernel-$V-$REL.src.rpm"
  echo "== kernel source: $URL =="
  if [ ! -f "$SRPM" ] && ! timeout 600 curl -sfL -o "$SRPM" "$URL"; then
    echo "   (kojipkgs unavailable, using kernel.org tarball)"
    MAJ=${V%%.*}; SRPM=""
    timeout 900 curl -sfL -o "$WORK/src/linux-$V.tar.xz" \
      "https://cdn.kernel.org/pub/linux/kernel/v$MAJ.x/linux-$V.tar.xz"
    TARBALL=$WORK/src/linux-$V.tar.xz
  else
    ( cd "$WORK/src" && rpm2cpio "$SRPM" | cpio -idm --quiet "./linux-$V.tar.xz" "linux-$V.tar.xz" 2>/dev/null || true )
    TARBALL=$WORK/src/linux-$V.tar.xz
  fi
  tar -xf "$TARBALL" -C "$WORK/src" "linux-$V/drivers/thunderbolt" "linux-$V/include/thunderbolt.h" 2>/dev/null || \
    tar -xf "$TARBALL" -C "$WORK/src" "linux-$V/drivers/thunderbolt" "linux-$V/include"
  mv "$WORK/src/linux-$V/drivers/thunderbolt" "$WORK/src/drivers-thunderbolt"
  mv "$WORK/src/linux-$V/include/linux/thunderbolt.h" "$WORK/src/thunderbolt.h.orig"
  rm -rf "$WORK/src/linux-$V"
fi

# ── 1/3 patched thunderbolt core ──────────────────────────────────────
rm -rf "$WORK/tb-src"; mkdir -p "$WORK/tb-src/include/linux"
cp -a "$WORK/src/drivers-thunderbolt" "$WORK/tb-src/thunderbolt"
cp "$WORK/src/thunderbolt.h.orig" "$WORK/tb-src/include/linux/thunderbolt.h"
( cd "$WORK/tb-src" && patch -p1 --fuzz=0 \
    < "$REPO/patches/linux/0001-thunderbolt-source-xdomain-protocol-handlers.patch" )
grep -q tb_protocol_handler_xd "$WORK/tb-src/include/linux/thunderbolt.h" || {
  echo "!! core patch did not take"; exit 1; }

# The stock struct MUST stay byte-identical: thunderbolt_net embeds it in its
# private data, so any layout change corrupts the prebuilt module.
"$HERE/check-tb-handler-abi.sh" "$WORK/tb-src/include/linux/thunderbolt.h" --require-xd

if [ ! -d "$WORK/kdir/Makefile" ]; then
  rm -rf "$WORK/kdir"; mkdir -p "$WORK/kdir"; cp -as "$KDEV/." "$WORK/kdir/"
fi
rm -f "$WORK/kdir/include/linux/thunderbolt.h"
cp "$WORK/tb-src/include/linux/thunderbolt.h" "$WORK/kdir/include/linux/thunderbolt.h"
real=$(readlink -f "$WORK/kdir/include/config/auto.conf" || true)
if [ -n "$real" ] && [ -f "$real" ] && [ ! -L "$WORK/kdir/include/config/auto.conf" ]; then :; else
  [ -n "$real" ] && { rm -f "$WORK/kdir/include/config/auto.conf"; cp "$real" "$WORK/kdir/include/config/auto.conf"; }
fi

echo "== 1/3 thunderbolt core =="
mkdir -p "$OUT"
nice -n 19 make -j"$(nproc)" -C "$WORK/kdir" M="$WORK/tb-src/thunderbolt" 2>&1 | tail -2
cp -f "$WORK/tb-src/thunderbolt/thunderbolt.ko" "$OUT/thunderbolt.ko"
echo "   core: $(modinfo -F vermagic "$OUT/thunderbolt.ko")"

# ── 2/3 OdinLink driver: pin + this repo's odinlink series ─────────────
if [ ! -d "$CACHE/odinlink/.git" ]; then
  git clone "$ODL_REPO" "$CACHE/odinlink"
fi
( cd "$CACHE/odinlink"
  git fetch -q origin
  git checkout -q "$ODL_PIN"; git reset --hard -q "$ODL_PIN"; git clean -qfdx
  for p in "$REPO"/patches/odinlink/[0-9]*.patch; do
      git apply --check "$p" || { echo "!! $p does not apply at $ODL_PIN"; exit 1; }
      git apply "$p"
      echo "   applied $(basename "$p")"
  done )

echo "== 2/3 odl_tb5 driver =="
( cd "$CACHE/odinlink/driver"
  nice -n 19 make -j"$(nproc)" -C "$WORK/kdir" M="$PWD" \
    KBUILD_EXTRA_SYMBOLS="$WORK/tb-src/thunderbolt/Module.symvers" 2>&1 \
    | grep -E "LD|error" | tail -3 )
KO=$CACHE/odinlink/driver/odl_tb5.ko
modinfo -F parm "$KO" | grep -q '^rx_asm_max' || { echo "!! local series not in binary"; exit 1; }
readelf -sW "$KO" | grep -qE 'UND[[:space:]]+tb_register_protocol_handler_xd' || {
  echo "!! 0004 (xd demux) not in binary"; exit 1; }
grep -q connect_retries "$CACHE/odinlink/driver/odl_tb5_proto.c" || {
  echo "!! 0005 (bounded retries) not in tree"; exit 1; }
cp -f "$KO" "$OUT/odl_tb5.ko"
cp -f "$HERE/support/71-odl-tb5.rules" "$CACHE/odinlink/driver/71-odl-tb5.rules"
echo "   driver: $(modinfo -F vermagic "$KO")"

# ── 3/3 userspace lib ─────────────────────────────────────────────────
echo "== 3/3 libodl_tb5 =="
( cd "$CACHE/odinlink"; mkdir -p build/lib
  gcc -O2 -fPIC -shared -Wl,-soname,libodl_tb5.so.0 -I lib/include -I driver/uapi \
    lib/src/odl_tb5_dev.c lib/src/odl_tb5_xfer.c lib/src/odl_tb5_peer.c \
    lib/src/odl_tb5_completion.c lib/src/odl_tb5_stream.c \
    -o build/lib/libodl_tb5.so.0.1.0
  ln -sf libodl_tb5.so.0.1.0 build/lib/libodl_tb5.so.0
  ln -sf libodl_tb5.so.0 build/lib/libodl_tb5.so )
mkdir -p "$OUT/lib"; cp -a "$CACHE/odinlink/build/lib/"libodl_tb5.so* "$OUT/lib/"

echo
echo "built for $KVER in $OUT"
echo "install: sudo $HERE/install.sh $KVER     (see docs/RUNBOOK.md)"
