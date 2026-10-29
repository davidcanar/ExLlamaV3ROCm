#!/usr/bin/env bash
# Rank-0 (Box1) environment for the two-node DS4.1 TP server: the HTTP head node.
# Same environment as scripts/run_rank.sh (left untouched), pointing at tp_serve.py.
# usage: tp_serve_rank0.sh [http-port] [master-port] [control-port]
set -euo pipefail

HTTP_PORT=${1:-8080}
MASTER_PORT=${2:-29650}
CONTROL_PORT=${3:-29651}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ODL_BUILD="${EXL3_ODL_BUILD:-$HOME/.cache/odinlink/build}"

ulimit -n 65536
export EXL3_TP_NODE_RANK=0
export EXL3_TP_WORLD_SIZE=2
export EXL3_TP_LOCAL_DEVICE=0
export EXL3_TP_DISPATCH_TIMEOUT=1800
export EXL3_TP_CONTROL_PORT=$CONTROL_PORT
export EXLLAMA_MASTER_ADDR="${EXL3_MASTER_ADDR:-10.0.2.1}"
export EXLLAMA_MASTER_PORT=$MASTER_PORT
export NCCL_SOCKET_IFNAME=thunderbolt0
export NCCL_IB_DISABLE=1
export NCCL_NET=ODL_TB5
# Only the box1<->box2 link: another Thunderbolt host on a second USB4 port shows up as
# /dev/odl_tb5_1 and must not be offered to RCCL (patched plugin, ODL_RCCL_DEVICES)
export ODL_RCCL_DEVICES=${ODL_RCCL_DEVICES:-0}
export NCCL_DEBUG=WARN
# Transport defaults of scripts/run_rank.sh (multi-frame stream messages, 4 RCCL channels).
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
export EXL3_TP_NCCL_SYNC=${EXL3_TP_NCCL_SYNC:-stream}
# Decode-sized all-reduces over the OdinLink stream API (odl_ar2, stream-ordered, ~35 us vs ~90 us
# for RCCL at 10 KB); prefill-sized collectives stay on RCCL. Build: ~/exl3ds4/odl_ar2/build.sh.
# EXL3_TP_ODL_AR2=0 restores RCCL for everything. Rank 1 listens on the rendezvous port.
export EXL3_TP_ODL_AR2=${EXL3_TP_ODL_AR2:-1}
export EXL3_TP_ODL_AR2_PEER=${EXL3_TP_ODL_AR2_PEER:-10.0.2.2}
export EXL3_TP_ODL_AR2_PORT=${EXL3_TP_ODL_AR2_PORT:-29652}
export EXL3_TP_ODL_AR2_LIB=${EXL3_TP_ODL_AR2_LIB:-$ROOT/odl_ar2/libodl_ar2.so}
export LD_LIBRARY_PATH="$ODL_BUILD/rccl:$ODL_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/src/exllamav3-dsv41f${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/src/exllamav3-dsv41f"
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ds41/tp_serve.py" \
    --host 0.0.0.0 --port "$HTTP_PORT" --control-port "$CONTROL_PORT"
