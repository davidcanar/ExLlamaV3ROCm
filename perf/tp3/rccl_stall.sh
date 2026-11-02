#!/usr/bin/env bash
# Repro loop for the intermittent TP=3 RCCL stall at the first prefill-sized all-reduce.
# Each attempt starts 3 bare ranks (perf/tp3/rccl_stall.py, no model) with the serving env.
# usage: rccl_stall.sh [attempts]      (servers must be stopped)
#   ODL3_MODE=alt|0|1   odl_ar3 setup per attempt (default alt: even attempts 1, odd 0)
#   ODL_DEBUG=1         plugin trace per rank (/tmp/odl_plugin.<pid>.log, copied for stalled attempts)
#   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET   (defaults) channel/connection setup in the rank logs
set -uo pipefail
N=${1:-10}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ODL_BUILD="${EXL3_ODL_BUILD:-$HOME/.cache/odinlink/build}"
B2=${EXL3_BOX2:-192.168.0.103}
B3=${EXL3_BOX3:-192.168.0.105}
OUT=$ROOT/perf/tp3/stall/$(date +%m%d-%H%M%S)
PROBE=$ROOT/tools/ds41/odl_peer_probe.py
mkdir -p "$OUT"
for h in "$B2" "$B3"; do ssh -o BatchMode=yes "$h" "mkdir -p $OUT"; done

python3 "$PROBE" --name box1 --ip 10.0.2.1 > "$OUT/probe0" 2>&1 &
ssh -o BatchMode=yes "$B2" "python3 $PROBE --name box2 --ip 10.0.2.2" > "$OUT/probe1" 2>&1 &
ssh -o BatchMode=yes "$B3" "python3 $PROBE --name box3 --ip 10.0.2.3" > "$OUT/probe2" 2>&1 &
wait
P0=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$OUT/probe0"); P1=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$OUT/probe1"); P2=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$OUT/probe2")
echo "[stall] cable map: $P0 | $P1 | $P2  -> $OUT"

COMMON="NCCL_IB_DISABLE=1 NCCL_NET=ODL_TB5 NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET} \
NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4} ODL_DEBUG=${ODL_DEBUG:-1} STEP_TIMEOUT=${STEP_TIMEOUT:-25} \
LD_LIBRARY_PATH=$ODL_BUILD/rccl:$ODL_BUILD/lib \
LD_PRELOAD=$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1 WORLD=3 MASTER=192.168.0.102"

clean() {
    pkill -KILL -f "tp3/[r]ccl_stall.py" 2>/dev/null
    for h in "$B2" "$B3"; do ssh -o BatchMode=yes "$h" 'pkill -KILL -f "tp3/[r]ccl_stall.py"; true'; done
}
clean
ok=0; stall=0
for i in $(seq 1 "$N"); do
    case "${ODL3_MODE:-alt}" in alt) m=$(( (i + 1) % 2 ));; *) m=$ODL3_MODE;; esac
    port=$((29700 + RANDOM % 250))
    A="$COMMON ODL3_MODE=$m MPORT=$port"
    t0=$SECONDS
    ssh -n -o BatchMode=yes "$B2" "cd $ROOT; env $A RANK=1 NCCL_SOCKET_IFNAME=eno1 ODL_RCCL_LOCAL_IP=10.0.2.2 ODL_RCCL_PEERS=$P1 \
        timeout 150 $ROOT/.venv/bin/python $ROOT/perf/tp3/rccl_stall.py > $OUT/a$i-rank1.log 2>&1" &
    ssh -n -o BatchMode=yes "$B3" "cd $ROOT; env $A RANK=2 NCCL_SOCKET_IFNAME=enp198s0f0u2u4 ODL_RCCL_LOCAL_IP=10.0.2.3 ODL_RCCL_PEERS=$P2 \
        timeout 150 $ROOT/.venv/bin/python $ROOT/perf/tp3/rccl_stall.py > $OUT/a$i-rank2.log 2>&1" &
    ( cd "$ROOT"; env $A RANK=0 NCCL_SOCKET_IFNAME=eno1 ODL_RCCL_LOCAL_IP=10.0.2.1 ODL_RCCL_PEERS=$P0 \
        timeout 150 "$ROOT/.venv/bin/python" "$ROOT/perf/tp3/rccl_stall.py" > "$OUT/a$i-rank0.log" 2>&1 )
    rc0=$?
    wait
    dt=$((SECONDS - t0))
    r0=$(grep -E "^RESULT" "$OUT/a$i-rank0.log" | head -1 | cut -d' ' -f2-)
    r1=$(ssh -o BatchMode=yes "$B2" "grep -E '^RESULT' $OUT/a$i-rank1.log | head -1 | cut -d' ' -f2-")
    r2=$(ssh -o BatchMode=yes "$B3" "grep -E '^RESULT' $OUT/a$i-rank2.log | head -1 | cut -d' ' -f2-")
    echo "[stall] attempt $i odl_ar3=$m port $port ${dt}s: r0='${r0:-rc=$rc0}' r1='${r1:-?}' r2='${r2:-?}'"
    if [ "$r0" = OK ] && [ "$r1" = OK ] && [ "$r2" = OK ]; then
        ok=$((ok + 1))
    else
        stall=$((stall + 1))
        # keep this attempt's plugin traces (newest odl_plugin log on each box)
        cp "$(ls -t /tmp/odl_plugin.*.log | head -1)" "$OUT/a$i-plugin0.log" 2>/dev/null
        for spec in "$B2:1" "$B3:2"; do
            h=${spec%:*}; r=${spec#*:}
            ssh -o BatchMode=yes "$h" "cp \$(ls -t /tmp/odl_plugin.*.log | head -1) $OUT/a$i-plugin$r.log 2>/dev/null; true"
        done
    fi
    clean
    sleep 2
done
echo "[stall] done: $ok ok, $stall failed of $N -> $OUT"
