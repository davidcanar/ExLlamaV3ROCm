#!/usr/bin/env bash
# Rank-1 worker environment for the two-node DS4.1 TP server.
# Same environment as scripts/run_rank.sh (left untouched), pointing at the persistent rank-1
# worker of the OpenAI-compatible server instead of a one-shot script.
# usage: tp_rank1.sh [master-port] [control-port]
set -euo pipefail

MASTER_PORT=${1:-29650}
CONTROL_PORT=${2:-29651}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ODL_BUILD="${EXL3_ODL_BUILD:-$HOME/.cache/odinlink/build}"

ulimit -n 65536
export EXL3_TP_NODE_RANK=1
export EXL3_TP_WORLD_SIZE=2
export EXL3_TP_LOCAL_DEVICE=0
export EXL3_TP_DISPATCH_TIMEOUT=1800
export EXL3_TP_CONTROL_PORT=$CONTROL_PORT
export EXLLAMA_MASTER_ADDR="${EXL3_MASTER_ADDR:-10.0.2.1}"
export EXLLAMA_MASTER_PORT=$MASTER_PORT
export NCCL_SOCKET_IFNAME=thunderbolt0
export NCCL_IB_DISABLE=1
export NCCL_NET=ODL_TB5
export NCCL_DEBUG=WARN
# Transport defaults of scripts/run_rank.sh (multi-frame stream messages, 4 RCCL channels).
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
export EXL3_TP_NCCL_SYNC=${EXL3_TP_NCCL_SYNC:-stream}
export LD_LIBRARY_PATH="$ODL_BUILD/rccl:$ODL_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/src/exllamav3-dsv41f${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/src/exllamav3-dsv41f"
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ds41/tp_rank1.py"
