#!/usr/bin/env bash
# rebuild-plugin.sh — build libodl_tb5 + the RCCL v7 net plugin from THIS
# repository's patch series, into ~/.cache/odinlink/build (where the launchers
# look for it; override with EXL3_ODL_SRC / EXL3_ODL_BUILD).
#
# The plugin is a host build, not an image: exl3ds4.sh points RCCL at
# $HOME/.cache/odinlink/build/rccl/librccl_net_odl_tb5.so.
#
# The plugin and the kernel driver talk the same stream framing, so this must
# be built from the SAME series as the loaded odl_tb5.ko — use
# odinlink-multilink/build.sh for the driver and this script for userspace.
#
# Idempotent: if the tree already carries the series it just rebuilds.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
SRC=${EXL3_ODL_SRC:-$HOME/.cache/odinlink}
BUILD=${EXL3_ODL_BUILD:-$SRC/build}
ODL_REPO=${ODL_REPO:-https://github.com/Geramy/OdinLink-Five}
ODL_PIN=${ODL_PIN:-4534f58}

if [ ! -d "$SRC/.git" ]; then
    if [ -e "$SRC" ]; then
        # ~/.cache/odinlink doubles as the manual-load cache (driver/, build/)
        # on boxes that only ever ran install.sh. Keep it, but a real clone is
        # needed for the plugin sources.
        aside="$SRC.pre-plugin.$(date +%s)"
        mv "$SRC" "$aside"
        echo "!! $SRC was not a git clone; moved it to $aside"
        echo "   re-run odinlink-multilink/install.sh afterwards to restore"
        echo "   ~/.cache/odinlink/driver/odl_tb5.ko for manual loads"
    fi
    git clone "$ODL_REPO" "$SRC"
fi

# Re-apply the series whenever it changes. A marker file records the hash of
# the patch set that produced the tree: checking for one feature string is not
# enough, because a NEWER patch would be skipped whenever an older one had
# already left its own marker behind.
SERIES_SHA=$(cat "$REPO"/patches/odinlink/[0-9]*.patch | sha256sum | cut -d' ' -f1)
MARKER=$SRC/.odl-series-sha

if [ "$(cat "$MARKER" 2>/dev/null)" != "$SERIES_SHA" ]; then
    echo "== resetting $SRC to $ODL_PIN and applying patches/odinlink =="
    git -C "$SRC" fetch -q origin
    git -C "$SRC" checkout -q "$ODL_PIN"
    git -C "$SRC" reset --hard -q "$ODL_PIN"
    git -C "$SRC" clean -qfdx
    for p in "$REPO"/patches/odinlink/[0-9]*.patch; do
        git -C "$SRC" apply --check "$p" || { echo "!! $(basename "$p") does not apply"; exit 1; }
        git -C "$SRC" apply "$p"
        echo "   applied $(basename "$p")"
    done
    printf '%s\n' "$SERIES_SHA" > "$MARKER"
else
    echo "== $SRC already carries this series ($SERIES_SHA) =="
fi

echo "== cmake build (lib + RCCL v7 plugin) =="
cmake -S "$SRC" -B "$BUILD" >/dev/null
cmake --build "$BUILD" -j"$(nproc)" 2>&1 | tail -3

PLUGIN=$BUILD/rccl/librccl_net_odl_tb5.so
[ -f "$PLUGIN" ] || { echo "!! no $PLUGIN"; exit 1; }

# Proof the new plugin logic is really in the binary, and the symbol RCCL
# dlsym()s is exported.
strings "$PLUGIN" | grep -q ODL_RCCL_PEERS || { echo "!! plugin lacks ODL_RCCL_PEERS"; exit 1; }
nm -D "$PLUGIN" | grep -q 'ncclNetPlugin_v7' || { echo "!! plugin does not export ncclNetPlugin_v7"; exit 1; }

echo "plugin: $PLUGIN"
echo "  $(date -r "$PLUGIN" '+%F %T')  $(stat -c %s "$PLUGIN") bytes"
