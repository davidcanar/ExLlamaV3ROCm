#!/usr/bin/env bash
# rccl3_run.sh — run the 3-rank RCCL all-reduce over OdinLink.
#
# One rank per box. Bootstrap/rendezvous rides the LAN (192.168.0.x) because the
# 10.0.2.x mesh is point-to-point: 10.0.2.1 is reachable from Box2 only, 10.0.2.3
# from Box1 only. The data path is OdinLink either way (NCCL_NET=ODL_TB5).
#
# !!! The per-box ODL_RCCL_PEERS map below is device->peer for the CURRENT
# !!! boot. Device indices follow XDomain probe order and DO change across
# !!! reloads/reboots; re-derive before running if odl_tb5 was reloaded
# !!! (see docs/RUNBOOK.md, "Device<->peer mapping is NOT stable").
set -uo pipefail

ROOT=${EXL3_ROOT:-/home/davidcanar/exl3ds4}
ODL_BUILD=${EXL3_ODL_BUILD:-$HOME/.cache/odinlink/build}
PORT=${EXL3_RCCL3_PORT:-$((29660 + RANDOM % 200))}
MASTER=${EXL3_RCCL3_MASTER:-192.168.0.102}
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$HERE/rccl3_allreduce.py
PY=$ROOT/.venv/bin/python

B2=${EXL3_BOX2:-192.168.0.103}
B3=${EXL3_BOX3:-192.168.0.105}

# Bootstrap (not data) rides the LAN, so each box needs ITS interface name --
# they are not uniform: Box3's comes up as enp198s0f0u2u4 (USB NIC), and NCCL
# fails with "Bootstrap : no socket interface found" if this names a device
# that does not exist. Override with EXL3_IFACE1/2/3 if the wiring changes.
IF1=${EXL3_IFACE1:-$(ip -br addr show | awk '$3 ~ /^192\.168\.0\./ {print $1; exit}')}
IF2=${EXL3_IFACE2:-$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$B2" "ip -br addr show | awk '\$3 ~ /^192\.168\.0\./ {print \$1; exit}'" 2>/dev/null)}
IF3=${EXL3_IFACE3:-$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$B3" "ip -br addr show | awk '\$3 ~ /^192\.168\.0\./ {print \$1; exit}'" 2>/dev/null)}
[ -n "$IF1" ] && [ -n "$IF2" ] && [ -n "$IF3" ] || {
    echo "!! could not determine LAN interfaces (1='$IF1' 2='$IF2' 3='$IF3')"; exit 1; }
echo "bootstrap interfaces: box1=$IF1 box2=$IF2 box3=$IF3"

# Per-rank fabric identity: canonical box address advertised to peers, and the
# peer->local-device map (current boot).
P1_LOCAL=10.0.2.1; P1_PEERS="10.0.2.2=1,10.0.2.3=0"     # Box1: dev1=Box2, dev0=Box3
P2_LOCAL=10.0.2.2; P2_PEERS="10.0.2.1=0,10.0.2.3=1"     # Box2: dev0=Box1, dev1=Box3
P3_LOCAL=10.0.2.3; P3_PEERS="10.0.2.1=0,10.0.2.2=1"     # Box3: dev0=Box1, dev1=Box2

common_env() {   # <rank> <local-ip> <peer-map> <iface>
    local rank=$1 localip=$2 peers=$3 iface=$4
    cat <<EOF
RANK=$rank WORLD_SIZE=3 MASTER_ADDR=$MASTER MASTER_PORT=$PORT
NCCL_SOCKET_IFNAME=$iface
NCCL_NET=ODL_TB5
NCCL_IB_DISABLE=1
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
ODL_DEBUG=1
NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
ODL_RCCL_LOCAL_IP=$localip
ODL_RCCL_PEERS=$peers
LD_LIBRARY_PATH=$ODL_BUILD/rccl:$ODL_BUILD/lib
EOF
}

run_local() {   # <rank> <local-ip> <peers> <iface>
    ( eval "export $(common_env "$1" "$2" "$3" "$4" | tr '\n' ' ')"
      cd "$ROOT/src/exllamav3-dsv41f" 2>/dev/null || cd "$HOME"
      source "$ROOT/env.sh" >/dev/null 2>&1 || true
      exec "$PY" "$SCRIPT" )
}

run_remote() {  # <host> <rank> <local-ip> <peers> <iface>
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
        "export $(common_env "$2" "$3" "$4" "$5" | tr '\n' ' '); \
         cd '$ROOT/src/exllamav3-dsv41f' 2>/dev/null || cd \$HOME; \
         source '$ROOT/env.sh' >/dev/null 2>&1 || true; \
         exec '$PY' '$SCRIPT'"
}

# Leftover ranks from a previous interrupted run would race this one for the
# bootstrap port and for odl streams; the ssh transport dies with the wrapper
# but the remote python does not. Always start clean.
cleanup_ranks() {   # <host|-> 
    if [ "$1" = "-" ]; then
        pkill -f "rccl3_[a]llreduce" 2>/dev/null || true
    else
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
            'pkill -f "rccl3_[a]llreduce" 2>/dev/null || true' 2>/dev/null || true
    fi
}
cleanup_ranks -; cleanup_ranks "$B2"; cleanup_ranks "$B3"
trap 'cleanup_ranks -; cleanup_ranks "$B2"; cleanup_ranks "$B3"' EXIT INT TERM

echo "== launching 3 ranks (master $MASTER:$PORT) =="
run_remote "$B2" 1 "$P2_LOCAL" "$P2_PEERS" "$IF2" > /tmp/rccl3-r1.log 2>&1 &
R1=$!
run_remote "$B3" 2 "$P3_LOCAL" "$P3_PEERS" "$IF3" > /tmp/rccl3-r2.log 2>&1 &
R2=$!
sleep 2
run_local 0 "$P1_LOCAL" "$P1_PEERS" "$IF1" > /tmp/rccl3-r0.log 2>&1
RC0=$?
wait $R1; RC1=$?
wait $R2; RC2=$?

echo "== results =="
for r in 0 1 2; do
    echo "--- rank $r:"
    tail -5 "/tmp/rccl3-r$r.log"
done
echo "== rc: rank0=$RC0 rank1=$RC1 rank2=$RC2"
[ "$RC0" = 0 ] && [ "$RC1" = 0 ] && [ "$RC2" = 0 ] && echo "ALL RANKS OK"
