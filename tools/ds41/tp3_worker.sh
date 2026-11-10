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
# The ODL_RCCL_PEERS defaults below are only a fallback: tp3_start.sh probes the live cable
# map on every start (tools/ds41/odl_peer_probe.py) and passes each rank its own map. Device
# indices follow XDomain probe order and DO change across reloads (odinlink-multilink
# docs/RUNBOOK.md, "mapping is NOT stable"); "./tp3_start.sh probe" prints the current one.
set -euo pipefail

NODE_RANK=${1:?usage: tp3_worker.sh <node-rank 1|2> [master-port] [control-port]}
MASTER_PORT=${2:-29650}
CONTROL_PORT=${3:-29651}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ODL_BUILD="${EXL3_ODL_BUILD:-$HOME/.cache/odinlink/build}"

case "$NODE_RANK" in
1)  # Box2: dev0=Box3, dev1=Box1 (measured 2026-11-02 after the 10:42 reload; the old
    # 10.0.2.1=0,10.0.2.3=1 default was stale and cross-wired RCCL -> 3-rank warmup hangs)
    LOCAL_IP=${EXL3_TP_LOCAL_IP_OVERRIDE:-10.0.2.2}
    PEERS=${ODL_RCCL_PEERS:-10.0.2.1=1,10.0.2.3=0}
    IFACE=${EXL3_IFACE:-eno1}
    ;;
2)  # Box3: dev0=Box1, dev1=Box2 (measured 2026-11-02)
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
# odl_ar3 mesh decode all-reduces (see tp3_serve.sh); must match on every rank
export EXL3_TP_ODL_AR3=${EXL3_TP_ODL_AR3:-1}
# First-light simplifications: single session, no vision tower, no NVMe tier.
# Conversation cache as on TP=2 (exl3ds4.sh): 4 slots share the one 512K pool, a returning
# conversation is matched by its prompt prefix, and when all slots are hot the least-recently-
# used one spills to a per-rank NVMe image. The session count, the pool geometry and the disk
# knobs must match on every rank (tp3_start.sh forwards overrides; the handshake refuses a
# mismatch). EXL3_TP_SESSIONS=1 is the single-conversation engine.
export EXL3_TP_SESSIONS=${EXL3_TP_SESSIONS:-4}
# Interleaved requests (proto 6): up to EXL3_TP_CONCURRENCY requests in flight at once, each in
# its own session with its own live cache slot, taking turns one model step at a time (fair
# share by GPU time). Default 4 (run on the rig 2026-11-10), or EXL3_TP_SESSIONS if that is
# smaller; 1 = one request at a time, the proto-5 engine (the rollback). Must match on every
# rank (it sizes each rank's cache slots; the handshake refuses a mismatch; tp3_serve.sh has
# the same default) and may not exceed EXL3_TP_SESSIONS. EXL3_TP_INTERLEAVE_CHUNK (rank 0,
# default 512) = prefill rows per step while other requests run.
export EXL3_TP_CONCURRENCY=${EXL3_TP_CONCURRENCY:-$(( EXL3_TP_SESSIONS < 4 ? EXL3_TP_SESSIONS : 4 ))}
# Vision runs on rank 0 only (this rank just receives image rows); the per-request image cap
# must match rank 0's.
export EXL3_TP_VISION=${EXL3_TP_VISION:-1}
export EXL3_TP_MAX_IMAGES=${EXL3_TP_MAX_IMAGES:-20}
export EXL3_TP_DISK_CACHE=${EXL3_TP_DISK_CACHE:-1}
export EXL3_TP_DISK_CACHE_DIR=${EXL3_TP_DISK_CACHE_DIR:-$ROOT/diskcache}
# 100 GiB per box: these boxes serve nothing else, and a spilled conversation costs ~68 MB +
# ~3.3 KB per token (a full 1M-token conversation ~3.3 GB), so ~25 full-context conversations
# or a few hundred 100K ones survive on NVMe. Images last until the server restarts.
export EXL3_TP_DISK_CACHE_GIB=${EXL3_TP_DISK_CACHE_GIB:-100}
export EXL3_TP_DISK_TIMEOUT=${EXL3_TP_DISK_TIMEOUT:-300}
export EXL3_TP_BUDGETS=${EXL3_TP_BUDGETS:-100,100,100}
# Model selection: the 3.5 bpw quant when this box has it (TP=3 has the room, ~80 GiB of
# weights per rank), else the 2.9 bpw one in $ROOT/model; export EXL3_TP_MODEL_DIR to serve
# another checkpoint. Paths must mirror on every box: tp3_start.sh resolves the folder once
# and forwards it to both workers, so the three ranks always load the same one.
if [ -z "${EXL3_TP_MODEL_DIR:-}" ]; then
    if [ -f "$ROOT/model-3.5bpw/config.json" ]; then
        EXL3_TP_MODEL_DIR=$ROOT/model-3.5bpw
    else
        EXL3_TP_MODEL_DIR=$ROOT/model
    fi
fi
export EXL3_TP_MODEL_DIR
# 1M context (the model's own limit: max_position_embeddings 1048576, YaRN x16). The DSA KV
# pools cost ~1.6 GiB per rank per 512K tokens; must match on every rank (pool geometry).
export EXL3_TP_SAFE_CTX=${EXL3_TP_SAFE_CTX:-1048576}
export EXL3_TP_CACHE_TOKENS=${EXL3_TP_CACHE_TOKENS:-1048576}
# DSpark (MTP speculative decoding) across all three ranks; opt out with
# EXL3_TP_DSPARK=0. Validated after the pow2 split landed.
export EXL3_TP_DSPARK=${EXL3_TP_DSPARK:-1}
export LD_LIBRARY_PATH="$ODL_BUILD/rccl:$ODL_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ROOT/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/src/exllamav3-dsv41f${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/src/exllamav3-dsv41f"
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ds41/tp3_worker.py"
