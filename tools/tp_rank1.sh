#!/usr/bin/env bash
# Rank-1 worker environment for the two-node DS4.1 TP server.
# Same environment as scripts/run_rank.sh (left untouched), pointing at the persistent rank-1
# worker of the OpenAI-compatible server instead of a one-shot script.
# usage: tp_rank1.sh [master-port] [control-port]
set -euo pipefail

MASTER_PORT=${1:-29650}
CONTROL_PORT=${2:-29651}
ROOT=/home/davidcanar/exl3ds4
ODL_BUILD=/home/davidcanar/.cache/odinlink/build

ulimit -n 65536
export EXL3_TP_NODE_RANK=1
export EXL3_TP_WORLD_SIZE=2
export EXL3_TP_LOCAL_DEVICE=0
export EXL3_TP_DISPATCH_TIMEOUT=1800
export EXL3_TP_CONTROL_PORT=$CONTROL_PORT
export EXLLAMA_MASTER_ADDR=10.0.2.1
export EXLLAMA_MASTER_PORT=$MASTER_PORT
export NCCL_SOCKET_IFNAME=thunderbolt0
export NCCL_IB_DISABLE=1
export NCCL_NET=ODL_TB5
# Per-peer routing (negotiated net plugin, odinlink 0009): both cables stay visible
# and ODL_RCCL_PEERS maps each peer to the local cable that reaches it. The old
# ODL_RCCL_DEVICES=0 pin broke whenever a driver reload swapped device indices
# (Box1 dev0 is the Box3 cable on some boots). Map matches the CURRENT boot;
# re-derive after any odl_tb5 reload (odinlink-multilink docs/RUNBOOK.md).
export ODL_RCCL_LOCAL_IP=${ODL_RCCL_LOCAL_IP:-10.0.2.2}
export ODL_RCCL_PEERS=${ODL_RCCL_PEERS:-10.0.2.1=0,10.0.2.3=1}
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
# The ar2 wire rides ONE cable; pick the device that reaches the peer from the
# same per-boot map as RCCL instead of hardcoding 0 (Box1 dev0 is not always
# the Box2 cable after a driver reload).
_ar2_peer=${EXL3_TP_ODL_AR2_PEER:-10.0.2.1}
_ar2_dev=$(printf '%s' "$ODL_RCCL_PEERS" | tr ',' '\n' | awk -F= -v p="$_ar2_peer" '$1==p {print $2; exit}')
export EXL3_TP_ODL_AR2_DEV=${EXL3_TP_ODL_AR2_DEV:-${_ar2_dev:-0}}
export EXL3_TP_ODL_AR2_LIB=${EXL3_TP_ODL_AR2_LIB:-$ROOT/odl_ar2/libodl_ar2.so}
export LD_LIBRARY_PATH="$ODL_BUILD/rccl:$ODL_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/src/exllamav3-dsv41f${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/src/exllamav3-dsv41f"
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ds41/tp_rank1.py"
