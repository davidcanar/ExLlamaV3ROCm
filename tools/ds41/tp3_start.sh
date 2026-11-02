#!/usr/bin/env bash
# tp3_start.sh — bring up the THREE-node DS4.1 TP stack:
#   rank 1 on Box2 (10.0.2.2), rank 2 on Box3 (10.0.2.3), rank 0 (HTTP) here.
# Logs: ~/exl3ds4/logs/tp3-rank{0,1,2}.log   (rank0 local, workers remote)
# usage: tp3_start.sh [stop|status|logs|probe]
#
# Every start re-derives the OdinLink cable map (which /dev/odl_tb5_N reaches which box) on all
# three boxes at once (tools/ds41/odl_peer_probe.py) and hands each rank ITS OWN ODL_RCCL_PEERS.
# Device indices follow XDomain probe order and change across odl_tb5 reloads; a stale map
# cross-wires the RCCL plugin's connections (hellos land on the wrong box's control stream) and
# hangs the world-3 NCCL warmup or the first large collective. EXL3_TP3_NO_PROBE=1 skips the probe
# (each script's built-in default map is used, or ODL_RCCL_PEERS_{0,1,2} if exported).
set -uo pipefail
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG=$ROOT/logs
B2=${EXL3_BOX2:-192.168.0.103}
B3=${EXL3_BOX3:-192.168.0.105}
HTTP_PORT=${EXL3_HTTP_PORT:-1234}
MASTER=29650
CONTROL=29651
PROBE=$ROOT/tools/ds41/odl_peer_probe.py

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

# Env overrides forwarded to the workers over ssh (ssh forwards nothing). The per-box OdinLink
# settings (ODL_RCCL_LOCAL_IP / ODL_RCCL_PEERS) and the LAN interface (EXL3_IFACE) are NOT
# forwarded: they differ on every box. EXLLAMA_MASTER_ADDR (Box1's LAN IP) is the same for all.
FORWARD_ENV=""
for v in EXLLAMA_MASTER_ADDR EXL3_TP_MODEL_DIR EXL3_TP_SAFE_CTX EXL3_TP_CACHE_TOKENS EXL3_TP_DSPARK \
         EXL3_TP_BUDGETS EXL3_TP_ODL_AR3 EXL3_TP_NCCL_SYNC EXL3_TP_SPAN EXL3_TP_SPIN_RECV \
         NCCL_MAX_NCHANNELS; do
    [ -n "${!v:-}" ] && FORWARD_ENV+="$v=${!v} "
done

# Cable map, one ODL_RCCL_PEERS per rank (P0 here, P1 Box2, P2 Box3)
P0=${ODL_RCCL_PEERS_0:-}; P1=${ODL_RCCL_PEERS_1:-}; P2=${ODL_RCCL_PEERS_2:-}
probe_cables() {
    local d; d=$(mktemp -d)
    python3 "$PROBE" --name box1 --ip 10.0.2.1 > "$d/0" 2>&1 &
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$B2" "python3 $PROBE --name box2 --ip 10.0.2.2" > "$d/1" 2>&1 &
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$B3" "python3 $PROBE --name box3 --ip 10.0.2.3" > "$d/2" 2>&1 &
    wait
    P0=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$d/0"); P1=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$d/1")
    P2=$(sed -n 's/^ODL_RCCL_PEERS=//p' "$d/2")
    local ok=1 p
    for p in "$P0" "$P1" "$P2"; do
        [ "$(printf '%s' "$p" | tr ',' '\n' | grep -c '=')" -eq 2 ] || ok=0
    done
    if [ $ok -ne 1 ]; then
        echo "  !! cable probe incomplete (a link down, or a server still holds the devices?):"
        sed 's/^/     /' "$d"/0 "$d"/1 "$d"/2
        rm -rf "$d"; return 1
    fi
    rm -rf "$d"
    echo "  cable map: rank0 $P0 | rank1 $P1 | rank2 $P2"
}

start_worker() {   # <host> <rank> <peers>
    local penv=""
    [ -n "$3" ] && penv="ODL_RCCL_PEERS=$3"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
        "cd $ROOT/src/exllamav3-dsv41f 2>/dev/null; \
         mkdir -p $LOG; \
         nohup env $FORWARD_ENV $penv $ROOT/tools/ds41/tp3_worker.sh $2 $MASTER $CONTROL >> $LOG/tp3-rank$2.log 2>&1 \
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
probe)  probe_cables;;
start)
    mkdir -p "$LOG"
    if [ -z "${EXL3_TP3_NO_PROBE:-}" ]; then
        echo "== probing the OdinLink cable map on all boxes =="
        probe_cables || exit 1
    fi
    echo "== starting workers (rank1@$B2, rank2@$B3) =="
    start_worker "$B2" 1 "$P1"
    start_worker "$B3" 2 "$P2"
    # Rank 0's load performs the world-size-3 NCCL warmup, which needs every
    # worker already loading; and the workers block in their own NCCL init
    # until it runs. Gate on the model being loaded, not on a fixed sleep.
    echo "== waiting for workers to finish loading before rank 0 =="
    wait_loaded "$B2" 1 || exit 1
    wait_loaded "$B3" 2 || exit 1
    echo "== starting rank 0 (HTTP :$HTTP_PORT) =="
    P0ENV=""
    [ -n "$P0" ] && P0ENV="ODL_RCCL_PEERS=$P0"
    nohup env $P0ENV PYTHONUNBUFFERED=1 "$ROOT/tools/ds41/tp3_serve.sh" "$HTTP_PORT" "$MASTER" "$CONTROL" \
        >> "$LOG/tp3-rank0.log" 2>&1 < /dev/null &
    echo $! > "$LOG/tp3-rank0.pid"
    echo "rank0 pid $(cat "$LOG/tp3-rank0.pid"); watch: $0 status | tail -f $LOG/tp3-rank0.log"
    ;;
*) echo "usage: $0 [start|stop|status|logs|probe]"; exit 2;;
esac
