#!/usr/bin/env bash
# Rank-0 (Box1) environment for the THREE-node DS4.1 TP server: HTTP head node.
# usage: tp3_serve.sh [http-port] [master-port] [control-port]
# Workers: tools/ds41/tp3_worker.sh 1 on Box2 and 2 on Box3 (tp3_start.sh
# orchestrates all three).
set -euo pipefail

HTTP_PORT=${1:-1234}
MASTER_PORT=${2:-29650}
CONTROL_PORT=${3:-29651}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ODL_BUILD="${EXL3_ODL_BUILD:-$HOME/.cache/odinlink/build}"

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
# Conversation cache as on TP=2 (exl3ds4.sh): 4 slots share the one 512K pool, a returning
# conversation is matched by its prompt prefix, and when all slots are hot the least-recently-
# used one spills to a per-rank NVMe image. The session count, the pool geometry and the disk
# knobs must match on every rank (tp3_start.sh forwards overrides; the handshake refuses a
# mismatch). EXL3_TP_SESSIONS=1 is the single-conversation engine.
export EXL3_TP_SESSIONS=${EXL3_TP_SESSIONS:-4}
# Interleaved requests (proto 6): up to EXL3_TP_CONCURRENCY requests in flight at once, each in
# its own session with its own live cache slot, taking turns one model step at a time (fair
# share by GPU time). 1 (default) = one request at a time, the proto-5 engine. Must match on
# every rank (it sizes each rank's cache slots; the handshake refuses a mismatch) and may not
# exceed EXL3_TP_SESSIONS. EXL3_TP_INTERLEAVE_CHUNK (rank 0, default 512) = prefill rows per
# step while other requests run.
export EXL3_TP_CONCURRENCY=${EXL3_TP_CONCURRENCY:-1}
# Image input: rank 0 loads the vision tower + aligner (~0.7 GiB) beside its shard and ships
# each image's rows to the workers per request (they never load it). Same defaults as TP=2.
export EXL3_TP_VISION=${EXL3_TP_VISION:-1}
export EXL3_TP_MAX_IMAGES=${EXL3_TP_MAX_IMAGES:-16}
export EXL3_TP_DISK_CACHE=${EXL3_TP_DISK_CACHE:-1}
export EXL3_TP_DISK_CACHE_DIR=${EXL3_TP_DISK_CACHE_DIR:-$ROOT/diskcache}
# 100 GiB per box: these boxes serve nothing else, and a spilled conversation costs ~68 MB +
# ~3.3 KB per token (a full 1M-token conversation ~3.3 GB), so ~25 full-context conversations
# or a few hundred 100K ones survive on NVMe. Images last until the server restarts.
export EXL3_TP_DISK_CACHE_GIB=${EXL3_TP_DISK_CACHE_GIB:-100}
export EXL3_TP_DISK_TIMEOUT=${EXL3_TP_DISK_TIMEOUT:-300}
# SSE keep-alive: a long prefill (100K tokens is minutes) sends no data, and clients with a read
# timeout drop the stream (which cancels the job). Rank 0 only: it is the HTTP layer.
export KYOJIN_SSE_PING_S=${KYOJIN_SSE_PING_S:-30}
export EXL3_TP_BUDGETS=${EXL3_TP_BUDGETS:-100,100,100}
# Model selection: default is the 2.9bpw quant; export EXL3_TP_MODEL_DIR to
# serve another checkpoint (e.g. model-3.5bpw). Paths must mirror on every box.
export EXL3_TP_MODEL_DIR=${EXL3_TP_MODEL_DIR:-$ROOT/model}
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
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ds41/tp3_serve.py" \
    --host 0.0.0.0 --port "$HTTP_PORT" --control-port "$CONTROL_PORT" \
    --control-host "$EXLLAMA_MASTER_ADDR"
