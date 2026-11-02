#!/usr/bin/env bash
# 3-rank RCCL vs odl_ar3 all-reduce microbenchmark (perf/tp3/ar3_bench.py) across Box1/2/3.
# Servers must be stopped. usage: ar3_bench.sh   (ITERS, EXL3_TP_ODL_AR3_LIB, ODL3_* pass through)
set -uo pipefail
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ODL_BUILD="${EXL3_ODL_BUILD:-$HOME/.cache/odinlink/build}"
B2=${EXL3_BOX2:-192.168.0.103}
B3=${EXL3_BOX3:-192.168.0.105}
OUT=$ROOT/perf/tp3
PROBE=$ROOT/tools/ds41/odl_peer_probe.py
PORT=$((29700 + RANDOM % 200))

python3 "$PROBE" --name box1 --ip 10.0.2.1 > "$OUT/.probe0" 2>&1 &
ssh -o BatchMode=yes "$B2" "python3 $PROBE --name box2 --ip 10.0.2.2" > "$OUT/.probe1" 2>&1 &
ssh -o BatchMode=yes "$B3" "python3 $PROBE --name box3 --ip 10.0.2.3" > "$OUT/.probe2" 2>&1 &
wait
P0=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$OUT/.probe0"); P1=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$OUT/.probe1"); P2=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$OUT/.probe2")
echo "[ar3_bench] cable map: $P0 | $P1 | $P2  port $PORT"

COMMON="NCCL_IB_DISABLE=1 NCCL_NET=ODL_TB5 NCCL_DEBUG=WARN NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4} \
LD_LIBRARY_PATH=$ODL_BUILD/rccl:$ODL_BUILD/lib \
LD_PRELOAD=$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1 \
WORLD=3 MASTER=192.168.0.102 MPORT=$PORT ITERS=${ITERS:-400}"
for v in $(compgen -e | grep -E '^(ODL3_|EXL3_TP_ODL_AR3)'); do COMMON+=" $v=${!v}"; done

ssh -n -o BatchMode=yes "$B2" "cd $ROOT; env $COMMON RANK=1 NCCL_SOCKET_IFNAME=eno1 ODL_RCCL_LOCAL_IP=10.0.2.2 ODL_RCCL_PEERS=$P1 \
    timeout 300 $ROOT/.venv/bin/python $OUT/ar3_bench.py > $OUT/ar3-rank1.log 2>&1 &"
ssh -n -o BatchMode=yes "$B3" "cd $ROOT; env $COMMON RANK=2 NCCL_SOCKET_IFNAME=enp198s0f0u2u4 ODL_RCCL_LOCAL_IP=10.0.2.3 ODL_RCCL_PEERS=$P2 \
    timeout 300 $ROOT/.venv/bin/python $OUT/ar3_bench.py > $OUT/ar3-rank2.log 2>&1 &"
cd "$ROOT"
env $COMMON RANK=0 NCCL_SOCKET_IFNAME=eno1 ODL_RCCL_LOCAL_IP=10.0.2.1 ODL_RCCL_PEERS=$P0 \
    timeout 300 "$ROOT/.venv/bin/python" "$OUT/ar3_bench.py" 2>&1 | tee "$OUT/ar3-rank0.log"
sleep 2
for h in "$B2" "$B3"; do ssh -o BatchMode=yes "$h" "tail -3 $OUT/ar3-rank*.log"; done
