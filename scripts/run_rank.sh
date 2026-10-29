#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <rank> <script-name> <master-port> <control-port>" >&2
    exit 2
fi

RANK=$1
SCRIPT_NAME=$2
MASTER_PORT=$3
CONTROL_PORT=$4
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ODL_BUILD="${EXL3_ODL_BUILD:-$HOME/.cache/odinlink/build}"

ulimit -n 65536
export EXL3_TP_NODE_RANK=$RANK
export EXL3_TP_WORLD_SIZE=2
export EXL3_TP_LOCAL_DEVICE=0
export EXL3_TP_DISPATCH_TIMEOUT=1800
export EXL3_TP_CONTROL_PORT=$CONTROL_PORT
export EXLLAMA_MASTER_ADDR="${EXL3_MASTER_ADDR:-10.0.2.1}"
export EXLLAMA_MASTER_PORT=$MASTER_PORT
export NCCL_SOCKET_IFNAME=thunderbolt0
export NCCL_IB_DISABLE=1
export NCCL_NET=ODL_TB5
export ODL_RCCL_DEVICES=${ODL_RCCL_DEVICES:-0}
export NCCL_DEBUG=WARN
# RCCL shards each collective across its channels and every shard becomes a
# stream message on the ODL_TB5 plugin, so the ~26 default channels multiply
# per-message overhead with no bandwidth win at 2-rank decode sizes. 4 is the
# value the vLLM stack measured on this same transport (ds4-cluster-env.odl.sh).
# Override per run (e.g. NCCL_MAX_NCHANNELS=2) or comment out for RCCL defaults.
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
# Cross-host collective host-sync mode for the EXL3 TP backend: "stream" syncs
# only the launching stream so MoE expert-prefetch / staging side streams keep
# running; "1" = full device sync; "0" = no host sync (stream-ordered
# consumers only). Override per run, e.g. EXL3_TP_NCCL_SYNC=1.
export EXL3_TP_NCCL_SYNC=${EXL3_TP_NCCL_SYNC:-stream}
export LD_LIBRARY_PATH="$ODL_BUILD/rccl:$ODL_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/src/exllamav3-dsv41f${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/src/exllamav3-dsv41f"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/$SCRIPT_NAME"
