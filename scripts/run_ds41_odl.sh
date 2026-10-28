#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-generate}
MASTER_PORT=${2:-29640}
CONTROL_PORT=${3:-29641}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${EXL3DS4_REMOTE:-davidcanar@10.0.2.2}"

case "$MODE" in
    generate) SCRIPT_NAME=tp_ds41_generate.py ;;
    smoke) SCRIPT_NAME=tp_ds41_smoke.py ;;
    *) echo "usage: $0 [generate|smoke] [master-port] [control-port]" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/logs"
WORKER_LOG="$ROOT/logs/${MODE}-odl-rank1.log"
RANK0_LOG="$ROOT/logs/${MODE}-odl-rank0.log"

cleanup_on_error() {
    echo "[run_ds41_odl] stopping rank 1 on Box2" >&2
    ssh "$REMOTE" "pkill -f '$ROOT/scripts/$SCRIPT_NAME'" 2>/dev/null || true
}
trap cleanup_on_error ERR INT TERM

echo "[run_ds41_odl] starting rank 1 on Box2; log: $WORKER_LOG"
ssh -n "$REMOTE" "$ROOT/scripts/run_rank.sh 1 $SCRIPT_NAME $MASTER_PORT $CONTROL_PORT" \
    > "$WORKER_LOG" 2>&1 &
WORKER_JOB=$!

echo "[run_ds41_odl] starting rank 0 on Box1; log: $RANK0_LOG"
set +e
"$ROOT/scripts/run_rank.sh" 0 "$SCRIPT_NAME" "$MASTER_PORT" "$CONTROL_PORT" \
    > "$RANK0_LOG" 2>&1
RANK0_STATUS=$?
set -e

if [[ $RANK0_STATUS -ne 0 ]]; then
    echo "[run_ds41_odl] rank 0 failed with status $RANK0_STATUS; see $RANK0_LOG" >&2
    cleanup_on_error
    exit "$RANK0_STATUS"
fi

wait "$WORKER_JOB"
trap - ERR INT TERM
echo "[run_ds41_odl] $MODE completed successfully on both nodes"
tail -12 "$RANK0_LOG"
