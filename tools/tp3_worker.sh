#!/usr/bin/env bash
# Worker (rank 1..N-1) environment for the THREE-node DS4.1 TP server.
# usage: tp3_worker.sh <node-rank> [master-port] [control-port]
#
# Differences from the two-node tp_rank1.sh:
#   - EXL3_TP_WORLD_SIZE=3, rank from the argument (Box2=1, Box3=2)
#   - bootstrap/rendezvous + control channel ride the LAN (one master address
#     reachable from every box; the 10.0.2.x mesh is point-to-point)
#   - both OdinLink cables visible to RCCL (no ODL_RCCL_DEVICES pinning) with
#     per-peer routing via the negotiated net plugin (odinlink 0009)
#   - EXL3_TP_ODL_AR2=0: the odl_ar2 decode path is 2-party; RCCL handles all
#     collectives at world size 3
#
# The ODL_RCCL_PEERS map matches the current driver load. Device indices follow
# XDomain probe order and DO change across reloads -- re-derive after any
# odl_tb5 reload (odinlink-multilink docs/RUNBOOK.md, "mapping is NOT stable").
set -euo pipefail

NODE_RANK=${1:?usage: tp3_worker.sh <node-rank 1|2> [master-port] [control-port]}
MASTER_PORT=${2:-29650}
CONTROL_PORT=${3:-29651}
ROOT=/home/davidcanar/exl3ds4
ODL_BUILD=/home/davidcanar/.cache/odinlink/build

case "$NODE_RANK" in
1)  # Box2: dev0=Box1, dev1=Box3 (this boot)
    LOCAL_IP=${EXL3_TP_LOCAL_IP_OVERRIDE:-10.0.2.2}
    PEERS=${ODL_RCCL_PEERS:-10.0.2.1=0,10.0.2.3=1}
    IFACE=${EXL3_IFACE:-eno1}
    ;;
2)  # Box3: dev0=Box1, dev1=Box2 (this boot)
    LOCAL_IP=${EXL3_TP_LOCAL_IP_OVERRIDE:-10.0.2.3}
    PEERS=${ODL_RCCL_PEERS:-10.0.2.1=0,10.0.2.2=1}
    IFACE=${EXL3_IFACE:-enp198s0f0u2u4}
    ;;
*)  echo "tp3_worker.sh: rank must be 1 or 2" >&2; exit 2;;
esac

ulimit -n 65536
export EXL3_TP_NODE_RANK=$NODE_RANK
export EXL3_TP_WORLD_SIZE=3
export EXL3_TP_LOCAL_DEVICE=0
export EXL3_TP_DISPATCH_TIMEOUT=1800
export EXL3_TP_CONTROL_PORT=$CONTROL_PORT
export EXLLAMA_MASTER_ADDR=${EXLLAMA_MASTER_ADDR:-192.168.0.102}
export EXLLAMA_MASTER_PORT=$MASTER_PORT
export NCCL_SOCKET_IFNAME=$IFACE
export NCCL_IB_DISABLE=1
export NCCL_NET=ODL_TB5
export ODL_RCCL_LOCAL_IP=$LOCAL_IP
export ODL_RCCL_PEERS=$PEERS
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
export EXL3_TP_NCCL_SYNC=${EXL3_TP_NCCL_SYNC:-stream}
export EXL3_TP_ODL_AR2=0
# First-light simplifications: single session, no vision tower, no NVMe tier.
export EXL3_TP_SESSIONS=${EXL3_TP_SESSIONS:-1}
export EXL3_TP_VISION=${EXL3_TP_VISION:-0}
export EXL3_TP_DISK_CACHE=${EXL3_TP_DISK_CACHE:-0}
export EXL3_TP_BUDGETS=${EXL3_TP_BUDGETS:-95,100,100}
export LD_LIBRARY_PATH="$ODL_BUILD/rccl:$ODL_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/src/exllamav3-dsv41f${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/src/exllamav3-dsv41f"
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ds41/tp3_worker.py"
