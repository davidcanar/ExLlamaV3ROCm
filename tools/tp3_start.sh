#!/usr/bin/env bash
# tp3_start.sh — bring up the THREE-node DS4.1 TP stack:
#   rank 1 on Box2 (10.0.2.2), rank 2 on Box3 (10.0.2.3), rank 0 (HTTP) here.
# Logs: ~/exl3ds4/logs/tp3-rank{0,1,2}.log   (rank0 local, workers remote)
# usage: tp3_start.sh [stop|status|logs]
set -uo pipefail
ROOT=/home/davidcanar/exl3ds4
LOG=$ROOT/logs
B2=${EXL3_BOX2:-192.168.0.103}
B3=${EXL3_BOX3:-192.168.0.105}
HTTP_PORT=${EXL3_HTTP_PORT:-8080}
MASTER=29650
CONTROL=29651

wait_loaded() {   # <host> <rank> <timeout_s>
    local h=$1 r=$2 t=${3:-600} i=0
    while [ $i -lt $t ]; do
        if ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
             "grep -q 'cache .* tokens' $LOG/tp3-rank$r.log 2>/dev/null"; then
            echo "  rank$r loaded (${i}s)"; return 0
        fi
        sleep 5; i=$((i+5))
    done
    echo "  !! rank$r did not finish loading in ${t}s"; return 1
}

start_worker() {   # <host> <rank>
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
        "cd $ROOT/src/exllamav3-dsv41f 2>/dev/null; \
         mkdir -p $LOG; \
         nohup $ROOT/tools/ds41/tp3_worker.sh $2 $MASTER $CONTROL >> $LOG/tp3-rank$2.log 2>&1 \
           < /dev/null & echo \$! > $LOG/tp3-rank$2.pid; \
         sleep 1; kill -0 \$(cat $LOG/tp3-rank$2.pid) 2>/dev/null && echo started || tail -3 $LOG/tp3-rank$2.log"
}

stop_all() {
    pkill -INT -f "tp3_[s]erve\.py" 2>/dev/null || true
    for h in "$B2" "$B3"; do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" 'pkill -INT -f "tp3_[w]orker\.py" 2>/dev/null || true' || true
    done
    sleep 5
    pkill -KILL -f "tp3_[s]erve\.py" 2>/dev/null || true
    for h in "$B2" "$B3"; do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" 'pkill -KILL -f "tp3_[w]orker\.py" 2>/dev/null || true' || true
    done
    echo "stopped"
}

status() {
    echo "rank0 (here): $(pgrep -cf 'tp3_[s]erve\.py' || echo 0) proc; http: \
$(curl -s -m 3 http://127.0.0.1:$HTTP_PORT/health | head -c 200 || echo down)"
    for spec in "$B2:1" "$B3:2"; do
        h=${spec%:*}; r=${spec#*:}
        echo "rank$r ($h): $(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
            "pgrep -cf 'tp3_[w]orker\.py' || echo 0") proc"
    done
}

case "${1:-start}" in
stop)   stop_all;;
status) status;;
logs)   tail -f $LOG/tp3-rank0.log;;
start)
    mkdir -p "$LOG"
    echo "== starting workers (rank1@$B2, rank2@$B3) =="
    start_worker "$B2" 1
    start_worker "$B3" 2
    # Rank 0's load performs the world-size-3 NCCL warmup, which needs every
    # worker already loading; and the workers block in their own NCCL init
    # until it runs. Gate on the model being loaded, not on a fixed sleep.
    echo "== waiting for workers to finish loading before rank 0 =="
    wait_loaded "$B2" 1 || exit 1
    wait_loaded "$B3" 2 || exit 1
    echo "== starting rank 0 (HTTP :$HTTP_PORT) =="
    nohup "$ROOT/tools/ds41/tp3_serve.sh" "$HTTP_PORT" "$MASTER" "$CONTROL" \
        >> "$LOG/tp3-rank0.log" 2>&1 < /dev/null &
    echo $! > "$LOG/tp3-rank0.pid"
    echo "rank0 pid $(cat "$LOG/tp3-rank0.pid"); watch: $0 status | tail -f $LOG/tp3-rank0.log"
    ;;
*) echo "usage: $0 [start|stop|status|logs]"; exit 2;;
esac
