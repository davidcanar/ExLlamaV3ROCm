#!/usr/bin/env bash
# check-tb-handler-abi.sh — fail the build if struct tb_protocol_handler's
# layout differs from mainline.
#
# Why this exists: thunderbolt_net EMBEDS struct tb_protocol_handler inside its
# private struct tbnet (net->handler) and the distro ships it prebuilt. Adding
# fields in the middle of that struct makes a patched core read tbnet's fields
# at the wrong offsets; the corruption lands on net->napi.state and shows up as
#
#   kernel BUG at net/core/dev.c:7671!   (BUG_ON(!test_bit(NAPI_STATE_SCHED)))
#     napi_enable -> tbnet_open -> __dev_open
#
# under rtnl_lock, i.e. plugging a cable hangs the whole machine.
#
# The multi-link patch therefore leaves this struct alone and adds a separate
# struct tb_protocol_handler_xd + registration API. This script is the guard
# that keeps it that way.
#
# Usage: check-tb-handler-abi.sh [path/to/thunderbolt.h] [--require-xd]
#   default: assert the stock struct layout is unchanged (works on any header)
#   --require-xd: also require the source-XDomain API (patched header only)
set -euo pipefail

HDR=${1:-/lib/modules/$(uname -r)/build/include/linux/thunderbolt.h}
REQUIRE_XD=0
for a in "$@"; do [ "$a" = "--require-xd" ] && REQUIRE_XD=1; done
[ -f "$HDR" ] || { echo "!! no $HDR"; exit 1; }

extract() {  # print the struct body, whitespace-normalised
    awk '/^struct tb_protocol_handler \{/{f=1} f{print} f&&/^\};/{exit}' "$1" \
    | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//' | sed '/^$/d'
}

expected='struct tb_protocol_handler {
const uuid_t *uuid;
int (*callback)(const void *buf, size_t size, void *data);
void *data;
struct list_head list;
};'

actual=$(extract "$HDR")
if [ "$actual" != "$expected" ]; then
    echo "!! struct tb_protocol_handler is NOT the mainline layout in:" >&2
    echo "   $HDR" >&2
    echo "--- expected ---" >&2
    printf '%s\n' "$expected" >&2
    echo "--- found ---" >&2
    printf '%s\n' "$actual" >&2
    echo >&2
    echo "thunderbolt_net embeds this struct; a layout change corrupts it at" >&2
    echo "runtime (napi_enable BUG under rtnl_lock -> machine hangs). Use" >&2
    echo "struct tb_protocol_handler_xd instead of adding fields here." >&2
    exit 1
fi

if [ "$REQUIRE_XD" = 0 ]; then
    echo "ABI ok: tb_protocol_handler in $HDR matches mainline"
    exit 0
fi

# The xd API must be present, otherwise the driver silently falls back to the
# single-cable route demux and multi-link hosts only ever complete one login.
for sym in tb_protocol_handler_xd tb_register_protocol_handler_xd \
           tb_unregister_protocol_handler_xd; do
    grep -q "$sym" "$HDR" || { echo "!! $HDR lacks $sym"; exit 1; }
done

echo "ABI ok: tb_protocol_handler matches mainline; source-XDomain API present"
