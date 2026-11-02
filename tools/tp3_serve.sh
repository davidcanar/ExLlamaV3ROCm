#!/usr/bin/env bash
# Rank-0 (Box1) environment for the THREE-node DS4.1 TP server: HTTP head node.
# usage: tp3_serve.sh [http-port] [master-port] [control-port]
# Workers: tools/ds41/tp3_worker.sh 1 on Box2 and 2 on Box3 (tp3_start.sh
# orchestrates all three).
set -euo pipefail

HTTP_PORT=${1:-1234}
MASTER_PORT=${2:-29650}
CONTROL_PORT=${3:-29651}
ROOT=/home/davidcanar/exl3ds4
ODL_BUILD=/home/davidcanar/.cache/odinlink/build

ulimit -n 65536
export EXL3_TP_NODE_RANK=0
export EXL3_TP_WORLD_SIZE=3
export EXL3_TP_LOCAL_DEVICE=0
export EXL3_TP_DISPATCH_TIMEOUT=1800
export EXL3_TP_CONTROL_PORT=$CONTROL_PORT
export EXLLAMA_MASTER_ADDR=${EXLLAMA_MASTER_ADDR:-192.168.0.102}
export EXLLAMA_MASTER_PORT=$MASTER_PORT
export NCCL_SOCKET_IFNAME=${EXL3_IFACE:-eno1}
export NCCL_IB_DISABLE=1
export NCCL_NET=ODL_TB5
# Cable map: tp3_start.sh probes it on every start (tools/ds41/odl_peer_probe.py) and passes
# ODL_RCCL_PEERS in; the default below is only the fallback for a direct launch (measured
# 2026-11-02 after the 10:42 odl_tb5 reload: Box1 dev0=Box3, dev1=Box2)
export ODL_RCCL_LOCAL_IP=${ODL_RCCL_LOCAL_IP:-10.0.2.1}
export ODL_RCCL_PEERS=${ODL_RCCL_PEERS:-10.0.2.2=1,10.0.2.3=0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
export EXL3_TP_NCCL_SYNC=${EXL3_TP_NCCL_SYNC:-stream}
export EXL3_TP_ODL_AR2=0
# Decode-sized all-reduces over the OdinLink mesh (odl_ar3: one hop to each peer over its own
# cable, stream-ordered, no host sync; ~40 us vs ~170 us for 3-rank RCCL at 10 KB). Prefill-sized
# collectives stay on RCCL. Every rank must agree (the setup is collective and falls back to RCCL
# on all ranks if any rank has it off). Build: ~/exl3ds4/odl_ar3/build.sh on every box.
export EXL3_TP_ODL_AR3=${EXL3_TP_ODL_AR3:-1}
export EXL3_TP_SESSIONS=${EXL3_TP_SESSIONS:-1}
export EXL3_TP_VISION=${EXL3_TP_VISION:-0}
export EXL3_TP_DISK_CACHE=${EXL3_TP_DISK_CACHE:-0}
export EXL3_TP_BUDGETS=${EXL3_TP_BUDGETS:-100,100,100}
# Model selection: default is the 2.9bpw quant; export EXL3_TP_MODEL_DIR to
# serve another checkpoint (e.g. model-3.5bpw). Paths must mirror on every box.
export EXL3_TP_MODEL_DIR=${EXL3_TP_MODEL_DIR:-$ROOT/model}
export EXL3_TP_SAFE_CTX=${EXL3_TP_SAFE_CTX:-524288}
export EXL3_TP_CACHE_TOKENS=${EXL3_TP_CACHE_TOKENS:-524288}
# DSpark (MTP speculative decoding) across all three ranks; opt out with
# EXL3_TP_DSPARK=0. Validated after the pow2 split landed.
export EXL3_TP_DSPARK=${EXL3_TP_DSPARK:-1}
export LD_LIBRARY_PATH="$ODL_BUILD/rccl:$ODL_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/src/exllamav3-dsv41f${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/src/exllamav3-dsv41f"
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ds41/tp3_serve.py" \
    --host 0.0.0.0 --port "$HTTP_PORT" --control-port "$CONTROL_PORT" \
    --control-host "$EXLLAMA_MASTER_ADDR"
