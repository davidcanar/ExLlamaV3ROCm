#!/usr/bin/env bash
# Launch the two-node OpenAI-compatible DS4.1 server over Odinlink.
#
#   Box2 (rank 1): tools/ds41/tp_rank1.sh  -- persistent mirror worker, started over SSH
#   Box1 (rank 0): tools/ds41/tp_serve_rank0.sh  -- HTTP server on 0.0.0.0:<http-port>, foreground
#
# usage: tp_serve_odl.sh [http-port] [master-port] [control-port]
#        defaults: 8080 29650 29651
#
# DSpark (MTP speculative decoding), opt-in on BOTH ranks:
#   EXL3_TP_DSPARK=1 tp_serve_odl.sh          (trunk cache max_history=8, budgets 95/104 GiB,
#                                              drafter on rank 0; greedy requests draft by default)
#   optional: EXL3_TP_BUDGETS=95,104  EXL3_TP_DSPARK_CONF=0.5 (draft confidence threshold)
# The EXL3_TP_DSPARK / EXL3_TP_BUDGETS / EXL3_TP_DSPARK_CONF / EXL3_DS41_CONF / EXL3_TP_SAFE_CTX
# / EXL3_TP_CACHE_TOKENS / EXL3_TP_SPAN / EXL3_TP_USE_PER_DEVICE / EXL3_TP_MEM_FLOOR variables
# are forwarded to Box2 (ssh forwards nothing by default; SPAN and the cache MUST match on both
# ranks or the prefill chunk sequences diverge and the collectives wedge).
#
# While the model loads, GET http://<box1>:<http-port>/health answers 503 with the stage
# ("model", "rank 1", "warm-up"); it turns 200 when serving. Logs:
#   ~/exl3ds4/logs/tpserve-rank{0,1}.log   (rank 0 also echoes to the terminal)
# Stop with Ctrl-C (rank 1 is shut down and unloaded cleanly), or pkill tp_serve.py/tp_rank1.py.
set -euo pipefail

HTTP_PORT=${1:-8080}
MASTER_PORT=${2:-29650}
CONTROL_PORT=${3:-29651}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REMOTE="${EXL3DS4_REMOTE:-davidcanar@10.0.2.2}"

mkdir -p "$ROOT/logs"
WORKER_LOG="$ROOT/logs/tpserve-rank1.log"
RANK0_LOG="$ROOT/logs/tpserve-rank0.log"

# variables both ranks must see identically; ssh forwards nothing, so they travel in the command
PASS_ENV=()
for v in EXL3_TP_DSPARK EXL3_TP_BUDGETS EXL3_TP_DSPARK_CONF EXL3_DS41_CONF EXL3_TP_SAFE_CTX \
         EXL3_TP_CACHE_TOKENS EXL3_TP_SPAN EXL3_TP_USE_PER_DEVICE EXL3_TP_MEM_FLOOR \
         EXL3_TP_ODL_AR2 EXL3_TP_ODL_AR2_PEER EXL3_TP_ODL_AR2_PORT EXL3_TP_ODL_AR2_MAX \
         EXL3_TP_SHARED_EXPERT_DEVICE EXL3_DSV4_CAND_ROWS EXL3_TP_NCCL_SYNC EXL3_TP_NCCL_FP32; do
    [[ -n "${!v:-}" ]] && PASS_ENV+=("$v=${!v}")
done
REMOTE_ENV=""
for spec in "${PASS_ENV[@]}"; do
    REMOTE_ENV+="$(printf '%q ' "$spec")"
done

cleanup() {
    echo "[tp_serve_odl] stopping rank 1 on Box2" >&2
    ssh "$REMOTE" "pkill -f 'exl3ds4/tools/ds41/tp_rank1.py'" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[tp_serve_odl] starting rank 1 on Box2; log: $WORKER_LOG"
ssh -n "$REMOTE" "$REMOTE_ENV$ROOT/tools/ds41/tp_rank1.sh $MASTER_PORT $CONTROL_PORT" \
    > "$WORKER_LOG" 2>&1 &
WORKER_JOB=$!

echo "[tp_serve_odl] starting rank 0 on Box1; HTTP on 0.0.0.0:$HTTP_PORT; log: $RANK0_LOG"
set +e
"$ROOT/tools/ds41/tp_serve_rank0.sh" "$HTTP_PORT" "$MASTER_PORT" "$CONTROL_PORT" 2>&1 | tee "$RANK0_LOG"
STATUS=$?
set -e

if [[ $STATUS -ne 0 ]]; then
    echo "[tp_serve_odl] rank 0 exited with status $STATUS; see $RANK0_LOG and $WORKER_LOG" >&2
fi
exit "$STATUS"
