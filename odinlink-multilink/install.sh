#!/usr/bin/env bash
# install-multilink.sh — stage the multi-cable OdinLink fabric built by
# build-multilink.sh:
#
#   /lib/modules/<KVER>/updates/thunderbolt/thunderbolt.ko
#       the callback_xd-patched core. depmod prefers extra/ over the stock
#       kernel/ copy, so "modprobe thunderbolt" (and boot-time PCI probe)
#       binds the patched core. Remove the file + depmod to go back to stock.
#   /usr/local/lib/odinlink/odl_tb5.ko (+lib, udev, odl-state, service)
#       the xd-demux driver, through the same layout install-odinlink.sh
#       uses, so odinlink.service / odl-swap.sh work unchanged.
#   ~/.cache/odinlink/driver/odl_tb5.ko
#       refreshed too, since manual odl-swap.sh runs load from there.
#
# The driver references tb_protocol_handler_abi_xd (exported only by the
# patched core), so a mismatched pair fails loudly at insmod instead of
# silently registering a handler the core would never call.
#
# Usage: sudo install-multilink.sh [KVER]     # default: running kernel
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)
KVER=${1:-$(uname -r)}
CACHE=${ODL_MULTI_CACHE:-$(eval echo "~${SUDO_USER:-$USER}")/.cache/odl-multi}
WORK=$CACHE/$KVER
OUT=$WORK/out
HOME_DIR=$(eval echo "~${SUDO_USER:-$USER}")

[ -f "$OUT/thunderbolt.ko" ] || { echo "missing $OUT/thunderbolt.ko (build first)"; exit 1; }
[ -f "$OUT/odl_tb5.ko" ]     || { echo "missing $OUT/odl_tb5.ko (build first)"; exit 1; }
[ -f "$OUT/lib/libodl_tb5.so" ] || { echo "missing libodl_tb5.so (build first)"; exit 1; }
[ -f "$CACHE/odinlink/lib/include/odl_tb5/odl_tb5.h" ] || { echo "missing odinlink clone headers"; exit 1; }

echo "== patched thunderbolt core → /lib/modules/$KVER/updates =="
install -d -m 0755 "/lib/modules/$KVER/updates/thunderbolt"
install -m 0644 "$OUT/thunderbolt.ko" "/lib/modules/$KVER/updates/thunderbolt/thunderbolt.ko"
depmod -a "$KVER"
resolved=$(modinfo -k "$KVER" -F filename thunderbolt 2>/dev/null || true)
echo "   modprobe thunderbolt → ${resolved:-<unresolved>}"
case "$resolved" in
  /lib/modules/$KVER/updates/thunderbolt/thunderbolt.ko) ;;
  *) echo "!! depmod did not pick the extra/ core — check for duplicates"; exit 1;;
esac

echo "== odinlink driver + service =="
install -d -m 0755 /usr/local/lib/odinlink /usr/local/share/odinlink /usr/local/sbin
install -m 0644 "$OUT/odl_tb5.ko" /usr/local/lib/odinlink/odl_tb5.ko
cp -a "$OUT/lib/"libodl_tb5.so* /usr/local/lib/odinlink/
install -m 0644 "$HERE/support/71-odl-tb5.rules" /usr/local/share/odinlink/71-odl-tb5.rules
install -m 0644 "$HERE/support/71-odl-tb5.rules" /etc/udev/rules.d/71-odl-tb5.rules
udevadm control --reload 2>/dev/null || true

gcc -O2 -I "$CACHE/odinlink/lib/include" \
    "$HERE/support/odl_state.c" \
    -L /usr/local/lib/odinlink -Wl,-rpath,/usr/local/lib/odinlink \
    -lodl_tb5 -o /usr/local/bin/odl-state

# odl-swap.sh + unit: our loader sets bind_any=0 (this fabric is all
# OdinLink; the Mac/bind_any sink path is what produced the -EINVAL retry
# storm) and waits a bounded 45 s for READY so a missing peer cannot stall
# boot. Set ODL_BIND_ANY=1 in the unit if a Mac sink is ever needed.
install -m 0755 "$HERE/support/odl-swap.sh" /usr/local/sbin/odl-swap.sh
install -m 0644 "$HERE/support/odinlink.service" /etc/systemd/system/odinlink.service
systemctl daemon-reload
systemctl enable odinlink.service >/dev/null

echo "== refresh the manual-load path cache =="
install -d -m 0755 "$HOME_DIR/.cache/odinlink/driver" "$HOME_DIR/.cache/odinlink/build/lib"
install -m 0644 "$OUT/odl_tb5.ko" "$HOME_DIR/.cache/odinlink/driver/odl_tb5.ko"
cp -a "$OUT/lib/"libodl_tb5.so* "$HOME_DIR/.cache/odinlink/build/lib/"

chown -R "$SUDO_USER:$SUDO_USER" "$HOME_DIR/.cache/odinlink" 2>/dev/null || true

cat <<EOF

installed for $KVER:
  thunderbolt core : patched (callback_xd) — takes effect after the core is
                     reloaded; simplest is a REBOOT, or run:
                       sudo systemctl stop odinlink
                       sudo $HERE/reload-multilink.sh
  odl_tb5 driver   : xd demux + local fixes
  odinlink.service : enabled (loads /usr/local/lib/odinlink/odl_tb5.ko at boot)

Do this on EVERY node of the fabric before bringing the links up.
EOF
