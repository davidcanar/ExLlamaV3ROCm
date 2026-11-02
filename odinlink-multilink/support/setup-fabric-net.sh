#!/usr/bin/env bash
# setup-fabric-net.sh <node>   (node = box1|box2|box3)
#
# Point-to-point addressing for the 3-node OdinLink mesh, bound to the
# Thunderbolt interface MAC rather than to thunderbolt0/thunderbolt1: those
# netdev names are assigned in XDomain-appearance order and can swap between
# boots, which would otherwise move addresses onto the wrong cable.
#
# All links share 10.0.2.0/24, so each interface carries only a /32 route to
# its DIRECT peer. That keeps a reply from leaving on the wrong interface --
# the ambiguity that silently broke pings when two interfaces had equal /24
# connected routes.
#
#                Box1                 Box2                 Box3
#   Box1  --    tb0 10.0.2.1   <->   tb0 10.0.2.2
#         \--   tb1 10.0.2.4   <->                          tb1 10.0.2.3
#   Box2  --                                       tb1 10.0.2.5 <-> tb0 10.0.2.6
#
#   MACs:
#   Box1 tb0 02:c5:07:47:92:4b   Box1 tb1 02:1f:73:95:1f:2c
#   Box2 tb0 02:51:e8:af:b0:a3   Box2 tb1 02:1d:1a:e4:83:64
#   Box3 tb0 02:01:76:3b:c1:dd   Box3 tb1 02:9b:d3:47:4f:49
set -euo pipefail
NODE=${1:?usage: $0 box1|box2|box3}

# Drop every old/auto profile that claimed these interfaces by name.
for c in "usb4-link" "usb4-link-b1" "usb4-link-b2" "Wired connection 2" "Wired connection 3"; do
    nmcli con delete "$c" >/dev/null 2>&1 || true
done

add() {  # add <conname> <mac> <addr/len> <peer /32>
    local name=$1 mac=$2 addr=$3 peer=$4
    nmcli con delete "$name" >/dev/null 2>&1 || true
    nmcli con add type ethernet con-name "$name" ifname "*" \
        802-3-ethernet.mac-address "$mac" \
        ipv4.method manual ipv4.addresses "$addr" ipv4.routes "$peer" \
        ipv4.never-default yes ipv6.method ignore >/dev/null
    echo "  $name: $mac -> $addr (peer $peer)"
}

case "$NODE" in
box1)
    add odl-to-box2 02:c5:07:47:92:4b 10.0.2.1/24 10.0.2.2/32
    add odl-to-box3 02:1f:73:95:1f:2c 10.0.2.4/24 10.0.2.3/32
    ;;
box2)
    add odl-to-box1 02:51:e8:af:b0:a3 10.0.2.2/24 10.0.2.1/32
    add odl-to-box3 02:1d:1a:e4:83:64 10.0.2.5/24 10.0.2.6/32
    ;;
box3)
    add odl-to-box1 02:9b:d3:47:4f:49 10.0.2.3/24 10.0.2.4/32
    add odl-to-box2 02:01:76:3b:c1:dd 10.0.2.6/24 10.0.2.5/32
    ;;
*) echo "unknown node $NODE"; exit 2;;
esac

# Bring them up (MAC-bound: NM applies each profile to whichever netdev has
# that MAC, no matter what it is named this boot).
for c in $(nmcli -t -f NAME,NAME con show | cut -d: -f1 | grep '^odl-to-'); do
    nmcli con up "$c" >/dev/null 2>&1 || echo "  (could not activate $c yet)"
done

echo "  --- applied:"
ip -br addr show | grep thunderbolt || true
ip route show | grep 10.0.2 || true
