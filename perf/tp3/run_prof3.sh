#!/usr/bin/env bash
# Three-node profiling run: ranks 1/2 = stock tools/ds41/tp3_worker.sh on Box2/Box3, rank 0 =
# the driver perf/tp3/$PROF_DRIVER (default ts_ab.py) here with the environment of
# tools/ds41/tp3_serve.sh. tp3_prof.py's own main() predates the interleaved engine (it calls the
# removed TPEngine._run/_run_spec); ts_ab.py runs the same measurements through TPEngine.generate.
# The tp3 server must be stopped first (tools/ds41/tp3_start.sh stop).
# usage: run_prof3.sh [tag]
#   PROF_* configure the driver (see ts_ab.py / tp3_prof.py); EXL3_*, NCCL_*, ODL2_*, HSA_*, ... are
#   forwarded to the workers too, except the per-box ones (rank, local IP, peer map, socket ifname).
#   A/B: run twice with a different tag and setting (e.g. EXL3_TP_MOE_TENSOR_SPLIT=0 / 1, the second
#   with PROF_LOGITS_REF=<first tag>), then python3 perf/tp3/ab_summary.py perf/tp3/ab.jsonl A B
set -uo pipefail

TAG=${1:-prof3}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ODL_BUILD="${EXL3_ODL_BUILD:-$HOME/.cache/odinlink/build}"
B2=${EXL3_BOX2:-192.168.0.103}
B3=${EXL3_BOX3:-192.168.0.105}
MASTER_PORT=${EXL3_MASTER_PORT:-29650}
CONTROL_PORT=${EXL3_TP_CONTROL_PORT:-29651}
OUT=$ROOT/perf/tp3
mkdir -p "$OUT"
LOG0=$OUT/$TAG-rank0.log

export EXL3_TP_MODEL_DIR=${EXL3_TP_MODEL_DIR:-$ROOT/model-3.5bpw}
export EXL3_TP_DSPARK=${EXL3_TP_DSPARK:-1}
export EXL3_TP_SAFE_CTX=${EXL3_TP_SAFE_CTX:-524288}
export EXL3_TP_CACHE_TOKENS=${EXL3_TP_CACHE_TOKENS:-524288}
export EXL3_TP_BUDGETS=${EXL3_TP_BUDGETS:-100,100,100}
export EXL3_TP_SESSIONS=${EXL3_TP_SESSIONS:-1}
export EXL3_TP_VISION=${EXL3_TP_VISION:-0}
export EXL3_TP_DISK_CACHE=${EXL3_TP_DISK_CACHE:-0}
# the server's default (tp3_serve.sh / tp3_worker.sh): unset here, rank 0 would set odl_ar3 up
# off while the workers' launcher turns it on, and every rank falls back to RCCL decode
export EXL3_TP_ODL_AR3=${EXL3_TP_ODL_AR3:-1}

PASS=""
for v in $(compgen -e | grep -E '^(EXL3_|NCCL_|ODL2_|ODL_TB5|HSA_|HIP_|AMD_|GPU_|PYTORCH_|TORCH_)'); do
    case "$v" in
        EXL3_TP_NODE_RANK|EXL3_TP_LOCAL_DEVICE|EXL3_TP_WORLD_SIZE|NCCL_SOCKET_IFNAME|EXL3_BOX2|EXL3_BOX3) continue ;;
    esac
    PASS+="$(printf '%q ' "$v=${!v}")"
done

cleanup_workers() {
    for h in "$B2" "$B3"; do
        ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'pkill -INT -f "tp3_[w]orker\.py" 2>/dev/null; sleep 3; pkill -KILL -f "tp3_[w]orker\.py" 2>/dev/null; true' || true
    done
}
cleanup_workers

# Per-boot cable map: probe all three boxes at once (device indices follow XDomain probe order and
# change across odl_tb5 reloads; a stale map cross-wires RCCL connections and hangs the warmup).
PROBE=$ROOT/tools/ds41/odl_peer_probe.py
probe_box() {  # <host|-> <name> <ip>
    if [ "$1" = "-" ]; then python3 "$PROBE" --name "$2" --ip "$3"
    else ssh -o BatchMode=yes "$1" "python3 $PROBE --name $2 --ip $3"; fi
}
if [ -z "${ODL_PEERS_NO_PROBE:-}" ]; then
    probe_box - box1 10.0.2.1 > "$OUT/.probe0" 2>&1 &
    probe_box "$B2" box2 10.0.2.2 > "$OUT/.probe1" 2>&1 &
    probe_box "$B3" box3 10.0.2.3 > "$OUT/.probe2" 2>&1 &
    wait
    P0=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$OUT/.probe0"); P1=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$OUT/.probe1")
    P2=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$OUT/.probe2")
    for p in "$P0" "$P1" "$P2"; do
        if [ "$(printf '%s' "$p" | tr ',' '\n' | grep -c '=')" -ne 2 ]; then
            echo "[run_prof3] cable probe incomplete:"; cat "$OUT"/.probe*; exit 1
        fi
    done
    export ODL_RCCL_PEERS=$P0
    echo "[run_prof3] cable map: rank0 $P0 | rank1 $P1 | rank2 $P2"
else
    P1=${ODL_RCCL_PEERS_1:-}; P2=${ODL_RCCL_PEERS_2:-}
fi
for spec in "$B2:1:$P1" "$B3:2:$P2"; do
    h=${spec%%:*}; rest=${spec#*:}; r=${rest%%:*}; peers=${rest#*:}
    penv=""; [ -n "$peers" ] && penv="ODL_RCCL_PEERS=$peers"
    echo "[run_prof3] rank $r -> $OUT/$TAG-rank$r.log on $h"
    ssh -n -o BatchMode=yes "$h" "mkdir -p $OUT; cd $ROOT/src/exllamav3-dsv41f; nohup env $PASS $penv $ROOT/tools/ds41/tp3_worker.sh $r $MASTER_PORT $CONTROL_PORT > $OUT/$TAG-rank$r.log 2>&1 < /dev/null &" || exit 1
done
# same gate as tp3_start.sh: rank 0 joins once both workers are inside their TP load
for spec in "$B2:1" "$B3:2"; do
    h=${spec%:*}; r=${spec#*:}; i=0
    until ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "grep -q 'cache .* tokens' $OUT/$TAG-rank$r.log 2>/dev/null"; do
        sleep 3; i=$((i + 3))
        if [ $i -ge 600 ]; then echo "[run_prof3] rank $r did not start"; cleanup_workers; exit 1; fi
    done
    echo "[run_prof3] rank $r loading (${i}s)"
done

ulimit -n 65536
export EXL3_TP_NODE_RANK=0 EXL3_TP_WORLD_SIZE=3 EXL3_TP_LOCAL_DEVICE=0 EXL3_TP_DISPATCH_TIMEOUT=1800
export EXL3_TP_CONTROL_PORT=$CONTROL_PORT
export EXLLAMA_MASTER_ADDR=${EXLLAMA_MASTER_ADDR:-192.168.0.102}
export EXLLAMA_MASTER_PORT=$MASTER_PORT
export NCCL_SOCKET_IFNAME=${EXL3_IFACE:-eno1}
export NCCL_IB_DISABLE=1 NCCL_NET=ODL_TB5 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export ODL_RCCL_LOCAL_IP=${ODL_RCCL_LOCAL_IP:-10.0.2.1}
export ODL_RCCL_PEERS=${ODL_RCCL_PEERS:-10.0.2.2=1,10.0.2.3=0}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
export EXL3_TP_NCCL_SYNC=${EXL3_TP_NCCL_SYNC:-stream}
export EXL3_TP_ODL_AR2=${EXL3_TP_ODL_AR2:-0}
export LD_LIBRARY_PATH="$ODL_BUILD/rccl:$ODL_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/src/exllamav3-dsv41f${PYTHONPATH:+:$PYTHONPATH}"
export PROF_TAG=${PROF_TAG:-$TAG}
cd "$ROOT/src/exllamav3-dsv41f"
echo "[run_prof3] rank 0 -> $LOG0"
"$ROOT/.venv/bin/python" "$ROOT/perf/tp3/${PROF_DRIVER:-ts_ab.py}" > "$LOG0" 2>&1
rc=$?
echo "[run_prof3] rank 0 exited $rc"
sleep 5
cleanup_workers
exit $rc
