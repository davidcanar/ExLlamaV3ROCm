#!/usr/bin/env bash
# Two-node DSpark (MTP) run over Odinlink: rank 1 on Box2, rank 0 on Box1.
# Usage: [SCRIPT_NAME=tp_ds41_dspark.py] [EXL3_DS41_CONF=0.5] [EXL3_DS41_PROMPT=...] \
#        [EXL3_DS41_TOKENS=32] [EXL3_DS41_EXPECTED=id,id,...] run_dspark_odl.sh [master-port] [control-port]
#
# EXL3_DS41_* variables are forwarded to BOTH ranks (both must prefill the same
# prompt); EXL3_DS41_CONF* go to rank 0 only (the drafter lives there).

set -euo pipefail

MASTER_PORT=${1:-29640}
CONTROL_PORT=${2:-29641}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${EXL3DS4_REMOTE:-davidcanar@10.0.2.2}"
SCRIPT_NAME=${SCRIPT_NAME:-tp_ds41_dspark.py}

mkdir -p "$ROOT/logs"
TAG=${SCRIPT_NAME%.py}
WORKER_LOG="$ROOT/logs/${TAG}-odl-rank1.log"
RANK0_LOG="$ROOT/logs/${TAG}-odl-rank0.log"

PASS_ENV=()
for v in EXL3_DS41_PROMPT EXL3_DS41_TOKENS EXL3_DS41_EXPECTED EXL3_DS41_MAX_ROUNDS; do
    [[ -n "${!v:-}" ]] && PASS_ENV+=("$v=${!v}")
done
# ssh forwards nothing by default: the rank-1 assignments travel inside the remote command,
# shell-quoted (prompts contain spaces)
REMOTE_ENV=""
for spec in "${PASS_ENV[@]}"; do
    REMOTE_ENV+="$(printf '%q ' "$spec")"
done

CONF_ENV=()
[[ -n "${EXL3_DS41_CONF:-}" ]] && CONF_ENV+=(EXL3_DSPARK_CONF="$EXL3_DS41_CONF")

cleanup_on_error() {
    echo "[run_dspark_odl] stopping rank 1 on Box2" >&2
    ssh "$REMOTE" "pkill -f '$ROOT/scripts/$SCRIPT_NAME'" 2>/dev/null || true
}
trap cleanup_on_error ERR INT TERM

echo "[run_dspark_odl] starting rank 1 on Box2; log: $WORKER_LOG"
ssh -n "$REMOTE" "$REMOTE_ENV$ROOT/scripts/run_rank.sh 1 $SCRIPT_NAME $MASTER_PORT $CONTROL_PORT" \
    > "$WORKER_LOG" 2>&1 &
WORKER_JOB=$!

echo "[run_dspark_odl] starting rank 0 on Box1; log: $RANK0_LOG"
set +e
env "${PASS_ENV[@]}" "${CONF_ENV[@]}" EXL3_DSPARK_CONF_STATS="${EXL3_DS41_CONF_STATS:-1}" \
    "$ROOT/scripts/run_rank.sh" 0 "$SCRIPT_NAME" "$MASTER_PORT" "$CONTROL_PORT" \
    > "$RANK0_LOG" 2>&1
RANK0_STATUS=$?
set -e

if [[ $RANK0_STATUS -ne 0 ]]; then
    echo "[run_dspark_odl] rank 0 failed with status $RANK0_STATUS; see $RANK0_LOG" >&2
    tail -30 "$RANK0_LOG" >&2 || true
    cleanup_on_error
    exit "$RANK0_STATUS"
fi

wait "$WORKER_JOB"
trap - ERR INT TERM
echo "[run_dspark_odl] $SCRIPT_NAME completed successfully on both nodes"
tail -25 "$RANK0_LOG"
