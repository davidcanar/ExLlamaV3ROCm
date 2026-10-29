#!/usr/bin/env bash
# Two-node profiling run: rank 1 = stock tools/ds41/tp_rank1.sh on Box2, rank 0 = perf/tp_prof.py here.
# Same environment as exl3ds4.sh / tp_serve_rank0.sh. The exl3ds4 server must be stopped first.
# usage: run_prof.sh [tag]   (PROF_* and EXL3_* variables pass through; EXL3_* also go to rank 1)
set -uo pipefail

TAG=${1:-prof}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${EXL3DS4_REMOTE:-davidcanar@10.0.2.2}"
ODL_BUILD="${EXL3_ODL_BUILD:-$HOME/.cache/odinlink/build}"
MASTER_PORT=${EXL3_MASTER_PORT:-29650}
CONTROL_PORT=${EXL3_TP_CONTROL_PORT:-29651}
LOG0=$ROOT/perf/$TAG-rank0.log
LOG1=$ROOT/perf/$TAG-rank1.log

export EXL3_TP_DSPARK=${EXL3_TP_DSPARK:-1}
export EXL3_TP_SAFE_CTX=${EXL3_TP_SAFE_CTX:-524288}
export EXL3_TP_CACHE_TOKENS=${EXL3_TP_CACHE_TOKENS:-524288}
# mirror the rank launchers' defaults so both ranks agree (rank 1's tp_rank1.sh applies the same)
export EXL3_TP_ODL_AR2=${EXL3_TP_ODL_AR2:-1}
export EXL3_TP_ODL_AR2_PEER=${EXL3_TP_ODL_AR2_PEER:-10.0.2.2}
export EXL3_TP_ODL_AR2_PORT=${EXL3_TP_ODL_AR2_PORT:-29652}
export EXL3_TP_ODL_AR2_LIB=${EXL3_TP_ODL_AR2_LIB:-$ROOT/odl_ar2/libodl_ar2.so}
export ODL_RCCL_DEVICES=${ODL_RCCL_DEVICES:-0}

PASS_ENV=()
for v in $(compgen -e | grep -E '^(EXL3_|NCCL_|ODL_|HSA_|HIP_|AMD_|GPU_|PYTORCH_|TORCH_)'); do
    case "$v" in EXL3_TP_NODE_RANK|EXL3_TP_LOCAL_DEVICE) continue ;; esac
    PASS_ENV+=("$v=${!v}")
done
REMOTE_ENV=""
for spec in "${PASS_ENV[@]}"; do REMOTE_ENV+="$(printf '%q ' "$spec")"; done

ssh -o BatchMode=yes "$REMOTE" "pkill -f 'exl3ds4/tools/ds41/[t]p_rank1.py'" 2>/dev/null || true
echo "[run_prof] rank 1 -> $LOG1 (on Box2)"
ssh -n -o BatchMode=yes "$REMOTE" "mkdir -p $ROOT/perf; nohup env $REMOTE_ENV $ROOT/tools/ds41/tp_rank1.sh $MASTER_PORT $CONTROL_PORT > $LOG1 2>&1 < /dev/null &" || exit 1

ulimit -n 65536
export EXL3_TP_NODE_RANK=0 EXL3_TP_WORLD_SIZE=2 EXL3_TP_LOCAL_DEVICE=0 EXL3_TP_DISPATCH_TIMEOUT=1800
export EXL3_TP_CONTROL_PORT=$CONTROL_PORT EXLLAMA_MASTER_ADDR="${EXL3_MASTER_ADDR:-10.0.2.1}" EXLLAMA_MASTER_PORT=$MASTER_PORT
export NCCL_SOCKET_IFNAME=thunderbolt0 NCCL_IB_DISABLE=1 NCCL_NET=ODL_TB5 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export ODL_RCCL_DEVICES=${ODL_RCCL_DEVICES:-0}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
export EXL3_TP_NCCL_SYNC=${EXL3_TP_NCCL_SYNC:-stream}
export LD_LIBRARY_PATH="$ODL_BUILD/rccl:$ODL_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/src/exllamav3-dsv41f${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/src/exllamav3-dsv41f"
echo "[run_prof] rank 0 -> $LOG0"
"$ROOT/.venv/bin/python" "$ROOT/perf/tp_prof.py" > "$LOG0" 2>&1
rc=$?
echo "[run_prof] rank 0 exited $rc"
sleep 5
ssh -o BatchMode=yes "$REMOTE" "pkill -f 'exl3ds4/tools/ds41/[t]p_rank1.py'" 2>/dev/null || true
exit $rc
