#!/usr/bin/env bash
# tp3_start.sh — bring up the THREE-node DS4.1 TP stack:
#   rank 1 on Box2 (10.0.2.2), rank 2 on Box3 (10.0.2.3), rank 0 (HTTP) here.
# Logs: ~/exl3ds4/logs/tp3-rank{0,1,2}.log   (rank0 local, workers remote)
#       ~/exl3ds4/logs/tp3-watchdog.log      (the watchdog, here)
# usage: tp3_start.sh [start|stop|restart|status|logs|probe|watch|unwatch]
#
# Every start re-derives the OdinLink cable map (which /dev/odl_tb5_N reaches which box) on all
# three boxes at once (tools/ds41/odl_peer_probe.py) and hands each rank ITS OWN ODL_RCCL_PEERS.
# Device indices follow XDomain probe order and change across odl_tb5 reloads; a stale map
# cross-wires the RCCL plugin's connections (hellos land on the wrong box's control stream) and
# hangs the world-3 NCCL warmup or the first large collective. EXL3_TP3_NO_PROBE=1 skips the probe
# (each script's built-in default map is used, or ODL_RCCL_PEERS_{0,1,2} if exported).
#
# Watchdog: `start` leaves one running here (EXL3_TP3_WATCHDOG=0: none; `watch` attaches one to a
# stack that is already up, `unwatch` removes it and leaves the stack alone). It restarts ALL
# ranks, with the settings they were started with, when a rank's process exits or a worker logs
# "failed; restart ALL ranks" (rank 0 cannot resynchronize after that and would sit in its next
# collective until the 10-minute NCCL timeout). It never judges a rank by HTTP, so a long
# prefill is never taken for a hang. Repeated failures back off (30 s doubling to 15 min, reset
# after 30 min up). `stop` ends it. Every rank runs in its own process group (setsid), so stop
# and restart also take down its spawned GPU processes, even when one is stuck in a collective.
set -uo pipefail
ACTION=${1:-start}
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG=$ROOT/logs
SELF=$(readlink -f "${BASH_SOURCE[0]}")
REMOTE_SELF=$ROOT/tools/ds41/tp3_start.sh   # this script on Box2/Box3 (same layout, same copy)
B2=${EXL3_BOX2:-192.168.0.103}
B3=${EXL3_BOX3:-192.168.0.105}
HTTP_PORT=${EXL3_HTTP_PORT:-1234}
MASTER=29650
CONTROL=29651
PROBE=$ROOT/tools/ds41/odl_peer_probe.py
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

WANT=$LOG/tp3.want                # exists while the stack should be up (start/watch create it)
WD_PID=$LOG/tp3-watchdog.pid
WD_LOG=$LOG/tp3-watchdog.log
WD_OFFS=$LOG/tp3-watchdog.offsets # worker log sizes at the watched start: the failure scan origin
WD_INTERVAL=${EXL3_TP3_WATCH_INTERVAL:-15}
WD_LOAD_TIMEOUT=${EXL3_TP3_WATCH_LOAD_TIMEOUT:-900}
FAIL_MARK="failed; restart ALL ranks"   # tp3_worker.py's last line when it cannot go on

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
FORWARD_VARS="EXLLAMA_MASTER_ADDR EXL3_TP_MODEL_DIR EXL3_TP_SAFE_CTX EXL3_TP_CACHE_TOKENS EXL3_TP_DSPARK
    EXL3_TP_BUDGETS EXL3_TP_ODL_AR3 EXL3_TP_NCCL_SYNC EXL3_TP_SPAN EXL3_TP_SPIN_RECV
    EXL3_TP_SESSIONS EXL3_TP_SESSION_MIN_MATCH EXL3_TP_PREFIX_SLOTS EXL3_TP_PREFIX_MIN
    EXL3_TP_PREFIX_END EXL3_TP_PREFIX_CACHE EXL3_TP_PREFIX_LOG
    EXL3_TP_DISK_CACHE EXL3_TP_DISK_CACHE_DIR EXL3_TP_DISK_CACHE_GIB EXL3_TP_DISK_TIMEOUT
    EXL3_TP_VISION EXL3_TP_VISION_STRICT EXL3_TP_MAX_IMAGES
    NCCL_MAX_NCHANNELS"
forward_env() {
    local v out=""
    for v in $FORWARD_VARS; do
        [ -n "${!v:-}" ] && out+="$v=${!v} "
    done
    printf '%s' "$out"
}

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

start_worker() {   # <host> <rank> <peers> <forwarded env>; 1 unless the worker is running
    local penv="" out
    [ -n "$3" ] && penv="ODL_RCCL_PEERS=$3"
    out=$("${SSH[@]}" "$1" \
        "cd $ROOT/src/exllamav3-dsv41f 2>/dev/null; \
         mkdir -p $LOG; \
         nohup setsid env PYTHONUNBUFFERED=1 $4 $penv $ROOT/tools/ds41/tp3_worker.sh $2 $MASTER $CONTROL >> $LOG/tp3-rank$2.log 2>&1 \
           < /dev/null & echo \$! > $LOG/tp3-rank$2.pid; echo \$! > $LOG/tp3-rank$2.pgid; \
         sleep 1; kill -0 \$(cat $LOG/tp3-rank$2.pid) 2>/dev/null && echo started || tail -3 $LOG/tp3-rank$2.log")
    echo "$out"
    [ "$out" = started ]
}

# ---- one rank on THIS box (also run remotely: tp3_start.sh _state | _kill | _gone) -------------

rank_re() {   # the rank's main python process OF THIS TREE, as a regex over its command line
              # (tp3_serve.sh / tp3_worker.sh exec exactly this; an editor on the file never matches)
    local s=tp3_worker
    [ "$1" = 0 ] && s=tp3_serve
    echo "^$ROOT/\.venv/bin/python[^ ]* $ROOT/tools/ds41/$s\.py( |\$)"
}

is_rank_pid() {   # <pid> <rank>: alive and that rank's python (not a reused pid)
    local c re
    [ -n "$1" ] && [ -r "/proc/$1/cmdline" ] || return 1
    c=$(tr '\0' ' ' 2>/dev/null < "/proc/$1/cmdline") || return 1   # it may exit meanwhile
    re=$(rank_re "$2")
    [[ $c =~ $re ]]
}

rank_state() {   # <rank> [<log offset>|-]: up | down | failed
    local r=$1 off=${2:--} f="$LOG/tp3-rank$1"
    case $off in ''|*[!0-9]*) off=-;; esac
    is_rank_pid "$(cat "$f.pid" 2>/dev/null)" "$r" || { echo down; return; }
    if [ "$r" != 0 ] && [ "$off" != - ] &&
            grep -qF "$FAIL_MARK" < <(tail -c +"$((off + 1))" "$f.log" 2>/dev/null); then
        echo failed; return
    fi
    echo up
}

kill_rank() {   # <signal> <rank>: its process group (if this launcher started it and the group
                # still holds our python), else its pid and that pid's children; then leftovers
    local sig=$1 r=$2 f="$LOG/tp3-rank$2" g p
    g=$(cat "$f.pgid" 2>/dev/null)
    if [ -n "$g" ] && pgrep -g "$g" -f "^$ROOT/\.venv/bin/python" > /dev/null 2>&1; then
        pkill -"$sig" -g "$g" 2>/dev/null
    fi
    p=$(cat "$f.pid" 2>/dev/null)
    if is_rank_pid "$p" "$r"; then
        pkill -"$sig" -P "$p" 2>/dev/null
        kill -"$sig" "$p" 2>/dev/null
    fi
    pkill -"$sig" -f "$(rank_re "$r")" 2>/dev/null
    return 0
}

rank_gone() {   # <rank>: yes once nothing of it is left (then its group id is forgotten)
    local r=$1 f="$LOG/tp3-rank$1" g
    g=$(cat "$f.pgid" 2>/dev/null)
    if [ -n "$g" ] && pgrep -g "$g" > /dev/null 2>&1; then echo no; return; fi
    if pgrep -f "$(rank_re "$r")" > /dev/null 2>&1; then echo no; return; fi
    rm -f "$f.pgid"
    echo yes
}

# ---- the whole stack -----------------------------------------------------------------------

box_of() { if [ "$1" = 1 ]; then echo "$B2"; else echo "$B3"; fi; }

signal_ranks() {   # <signal>: every rank's tree, here and on the workers' boxes, then (as the
                   # launcher always did) any rank process by name, whichever tree started it
    local r
    kill_rank "$1" 0
    pkill -"$1" -f "tp3_[s]erve\.py" 2>/dev/null
    for r in 1 2; do
        "${SSH[@]}" "$(box_of $r)" "$REMOTE_SELF _kill $1 $r 2>/dev/null; pkill -$1 -f 'tp3_[w]orker\.py'; true" \
            2>/dev/null || true
    done
}

ranks_gone() {
    local r
    [ "$(rank_gone 0)" = yes ] || return 1
    for r in 1 2; do
        [ "$("${SSH[@]}" "$(box_of $r)" "$REMOTE_SELF _gone $r" 2>/dev/null)" = yes ] || return 1
    done
}

stop_all() {
    signal_ranks INT
    sleep 5
    signal_ranks KILL
    local i=0
    until ranks_gone; do
        if [ $i -ge 30 ]; then
            echo "  !! rank processes outlived SIGKILL for 30 s (stuck in the driver?); the next start may find the devices busy"
            break
        fi
        sleep 2; i=$((i + 2))
    done
    echo "stopped"
}

record_offsets() {   # each worker log's size before this start: the failure scan starts there
    local r s
    : > "$WD_OFFS"
    for r in 1 2; do
        s=$("${SSH[@]}" "$(box_of $r)" "stat -c %s $LOG/tp3-rank$r.log 2>/dev/null || echo 0" 2>/dev/null)
        case $s in ''|*[!0-9]*) s=-;; esac       # unknown: that rank is judged by its process only
        echo "$r $s" >> "$WD_OFFS"
    done
}
offset_of() { local o; o=$(awk -v r="$1" '$1 == r {print $2}' "$WD_OFFS" 2>/dev/null); echo "${o:--}"; }

start_stack() {   # probe, workers, rank 0 (no watchdog); 1 on any failure it can see
    local fwd p0env=""
    mkdir -p "$LOG"
    fwd=$(forward_env)
    if [ -z "${EXL3_TP3_NO_PROBE:-}" ]; then
        echo "== probing the OdinLink cable map on all boxes =="
        probe_cables || return 1
    fi
    record_offsets
    echo "== starting workers (rank1@$B2, rank2@$B3) =="
    start_worker "$B2" 1 "$P1" "$fwd" || return 1
    start_worker "$B3" 2 "$P2" "$fwd" || return 1
    # Rank 0's load performs the world-size-3 NCCL warmup, which needs every
    # worker already loading; and the workers block in their own NCCL init
    # until it runs. Gate on the model being loaded, not on a fixed sleep.
    echo "== waiting for workers to finish loading before rank 0 =="
    wait_loaded "$B2" 1 || return 1
    wait_loaded "$B3" 2 || return 1
    echo "== starting rank 0 (HTTP :$HTTP_PORT) =="
    [ -n "$P0" ] && p0env="ODL_RCCL_PEERS=$P0"
    nohup setsid env $p0env PYTHONUNBUFFERED=1 "$ROOT/tools/ds41/tp3_serve.sh" "$HTTP_PORT" "$MASTER" "$CONTROL" \
        >> "$LOG/tp3-rank0.log" 2>&1 < /dev/null &
    echo $! > "$LOG/tp3-rank0.pid"
    echo $! > "$LOG/tp3-rank0.pgid"
    echo "rank0 pid $(cat "$LOG/tp3-rank0.pid")"
}

status() {
    local spec h r n
    echo "rank0 (here): $(pgrep -cf "$(rank_re 0)") proc; http: \
$(curl -s -m 3 "http://127.0.0.1:$HTTP_PORT/health" | head -c 200 || echo down)"
    for spec in "$B2:1" "$B3:2"; do
        h=${spec%:*}; r=${spec#*:}
        n=$("${SSH[@]}" "$h" "pgrep -cf 'tp3_[w]orker\.py'; true" 2>/dev/null)
        echo "rank$r ($h): ${n:-?} proc"
    done
    if wd_running; then
        echo "watchdog: on (pid $(cat "$WD_PID")); last: $(tail -n 1 "$WD_LOG" 2>/dev/null)"
    elif [ -f "$WANT" ]; then
        echo "watchdog: off, but the stack is marked wanted ('$0 watch' attaches one to a running stack)"
    else
        echo "watchdog: off"
    fi
}

port_busy() { ss -Hltn "sport = :$HTTP_PORT" 2>/dev/null | grep -q .; }

# Launch settings of the running rank 0, for every variable not set here: a watchdog attached by
# hand, or a restart, then brings the stack back exactly as it was. (tp3_serve.sh and
# tp3_worker.sh export the same defaults for every forwarded variable, so adopting rank 0's
# value never changes a worker's.)
ADOPT_VARS="$FORWARD_VARS EXL3_BOX2 EXL3_BOX3 EXL3_HTTP_PORT EXL3_TP3_NO_PROBE
    ODL_RCCL_PEERS_0 ODL_RCCL_PEERS_1 ODL_RCCL_PEERS_2 EXL3_TP3_WATCH_INTERVAL EXL3_TP3_WATCH_LOAD_TIMEOUT"
adopt_env() {
    local p v val
    p=$(cat "$LOG/tp3-rank0.pid" 2>/dev/null)
    if is_rank_pid "$p" 0 && [ -r "/proc/$p/environ" ]; then
        for v in $ADOPT_VARS; do
            val=$(tr '\0' '\n' 2>/dev/null < "/proc/$p/environ" | sed -n "s/^$v=//p" | head -n 1)
            [ -n "$val" ] || continue
            if [ -z "${!v:-}" ]; then
                export "$v=$val"
            elif [ "${!v}" != "$val" ]; then
                echo "  note: $v=${!v} here but $val in the running stack; restarts use ${!v}"
            fi
        done
    fi
    B2=${EXL3_BOX2:-192.168.0.103}; B3=${EXL3_BOX3:-192.168.0.105}; HTTP_PORT=${EXL3_HTTP_PORT:-1234}
    WD_INTERVAL=${EXL3_TP3_WATCH_INTERVAL:-15}; WD_LOAD_TIMEOUT=${EXL3_TP3_WATCH_LOAD_TIMEOUT:-900}
    echo "  model for (re)starts: ${EXL3_TP_MODEL_DIR:-$ROOT/model (default; EXL3_TP_MODEL_DIR unset)}"
}

# ---- the watchdog --------------------------------------------------------------------------

wlog() { echo "$(date '+%F %T') watchdog: $*"; }

wd_running() {
    local p
    p=$(cat "$WD_PID" 2>/dev/null) && [ -n "$p" ] && [ -r "/proc/$p/cmdline" ] || return 1
    [[ $(tr '\0' ' ' 2>/dev/null < "/proc/$p/cmdline") == *"tp3_start.sh _watchdog"* ]]
}

launch_watchdog() {
    touch "$WANT"
    nohup setsid "$SELF" _watchdog >> "$WD_LOG" 2>&1 < /dev/null &
    echo $! > "$WD_PID"
    echo "watchdog pid $(cat "$WD_PID"): restarts all ranks if the stack dies (log: $WD_LOG)"
}

stop_watchdog() {
    local p i=0
    if wd_running; then
        p=$(cat "$WD_PID")
        pkill -TERM -g "$p" 2>/dev/null
        while wd_running && [ $i -lt 10 ]; do sleep 1; i=$((i + 1)); done
        wd_running && pkill -KILL -g "$p" 2>/dev/null
        echo "watchdog stopped"
    fi
    rm -f "$WD_PID"
}

stack_down() {   # prints why and returns 0 when the stack is dead; an unreachable box proves nothing
    local r s
    [ "$(rank_state 0)" = up ] || { echo "rank 0 exited"; return 0; }
    for r in 1 2; do
        s=$("${SSH[@]}" "$(box_of $r)" "$REMOTE_SELF _state $r $(offset_of $r)" 2>/dev/null)
        case $s in
            down)   echo "rank $r exited"; return 0;;
            failed) echo "rank $r failed (its log says \"$FAIL_MARK\")"; return 0;;
        esac
    done
    return 1
}

nap() {   # <seconds>: sleep in steps; 1 as soon as the stack is no longer wanted
    local i=0
    while [ $i -lt "$1" ]; do
        [ -f "$WANT" ] || return 1
        sleep 5; i=$((i + 5))
    done
}

DOWN_WHY=""
wait_up() {   # <timeout_s>: 0 once /health answers; 1 (DOWN_WHY says why) if the stack died, the
              # time ran out, or it is no longer wanted. The first rank checks wait 15 s, until
              # every launcher has exec'd its python.
    local t=$1 i=0
    DOWN_WHY=""
    while [ $i -lt "$t" ]; do
        [ -f "$WANT" ] || { DOWN_WHY="no longer wanted"; return 1; }
        curl -sf -m 5 "http://127.0.0.1:$HTTP_PORT/health" > /dev/null 2>&1 && return 0
        if [ $i -ge 15 ] && [ $((i % 15)) -eq 0 ] && DOWN_WHY=$(stack_down); then return 1; fi
        sleep 5; i=$((i + 5))
    done
    DOWN_WHY="not serving ${t}s after the start (EXL3_TP3_WATCH_LOAD_TIMEOUT)"
    return 1
}

watchdog() {
    local fails=0 up_at why="" k delay
    trap 'wlog "stopped (signal)"; exit 0' TERM INT
    wlog "started (pid $$): checks every ${WD_INTERVAL}s; restarts serve ${EXL3_TP_MODEL_DIR:-$ROOT/model}"
    up_at=$(date +%s)
    if wait_up "$WD_LOAD_TIMEOUT"; then wlog "stack up"; else why=$DOWN_WHY; fi
    while [ -f "$WANT" ]; do
        if [ -z "$why" ]; then
            sleep "$WD_INTERVAL"
            [ -f "$WANT" ] || break
            if ! why=$(stack_down); then
                why=""
                if [ $fails -gt 0 ] && [ $(( $(date +%s) - up_at )) -ge 1800 ]; then
                    fails=0; wlog "up for 30 min: restart backoff reset"
                fi
                continue
            fi
        fi
        [ -f "$WANT" ] || break
        fails=$((fails + 1))
        wlog "stack down: $why -- restarting all ranks (restart $fails in a row)"
        stop_all
        if [ $fails -gt 1 ]; then
            k=$((fails - 2)); [ $k -gt 5 ] && k=5
            delay=$((30 << k)); [ $delay -gt 900 ] && delay=900
            wlog "waiting ${delay}s before restarting again"
            nap "$delay" || break
        fi
        why=""
        DOWN_WHY=""
        if start_stack && wait_up "$WD_LOAD_TIMEOUT"; then
            up_at=$(date +%s); wlog "stack up again"
        else
            why=${DOWN_WHY:-the start failed (see above)}
        fi
    done
    wlog "exiting: the stack is no longer wanted"
}

case "$ACTION" in
stop)
    rm -f "$WANT"
    stop_watchdog
    stop_all;;
status) status;;
logs)   tail -f "$LOG/tp3-rank0.log";;
probe)  probe_cables;;
start|restart)
    if [ "$ACTION" = restart ]; then
        adopt_env
        rm -f "$WANT"
        stop_watchdog
        stop_all
    elif wd_running || is_rank_pid "$(cat "$LOG/tp3-rank0.pid" 2>/dev/null)" 0 || port_busy; then
        echo "a TP=3 stack, its watchdog or another server (:$HTTP_PORT) is already up here:"
        echo "  '$0 stop' first, or '$0 restart'"
        exit 1
    fi
    touch "$WANT"
    if ! start_stack; then
        echo "!! start failed; stopping whatever did start"
        rm -f "$WANT"
        stop_all
        exit 1
    fi
    if [ "${EXL3_TP3_WATCHDOG:-1}" = 0 ]; then
        rm -f "$WANT"
        echo "no watchdog (EXL3_TP3_WATCHDOG=0): nothing restarts the stack if it dies ('$0 watch' adds one)"
    else
        launch_watchdog
    fi
    echo "watch: $0 status | tail -f $LOG/tp3-rank0.log"
    ;;
watch)
    if ! is_rank_pid "$(cat "$LOG/tp3-rank0.pid" 2>/dev/null)" 0; then
        echo "rank 0 is not running here: use '$0 start'"; exit 1
    fi
    if wd_running; then echo "a watchdog is already running (pid $(cat "$WD_PID"))"; exit 0; fi
    adopt_env
    record_offsets
    launch_watchdog;;
unwatch)
    rm -f "$WANT"
    stop_watchdog
    echo "the stack keeps running without a watchdog ('$0 watch' re-attaches one)";;
_state)    rank_state "${2:?rank}" "${3:--}";;
_kill)     kill_rank "${2:?signal}" "${3:?rank}";;
_gone)     rank_gone "${2:?rank}";;
_watchdog) watchdog;;
*) echo "usage: $0 [start|stop|restart|status|logs|probe|watch|unwatch]"; exit 2;;
esac
