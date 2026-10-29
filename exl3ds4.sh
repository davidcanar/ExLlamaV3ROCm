#!/usr/bin/env bash
# =============================================================================
# exl3ds4 — start / stop / restart / status / logs for the two-node DS4.1 EXL3
#           OpenAI server (tools/ds41/tp_serve.py on Box1 + tp_rank1.py on Box2)
#
#   Box1 (this machine, 10.0.2.1): rank 0 — HTTP head node, binds 0.0.0.0:1234
#   Box2 (davidcanar@10.0.2.2):   rank 1 — mirror worker, driven over ssh
#
#   Link : Thunderbolt/USB4 + OdinLink (odl_tb5) — NCCL/RCCL transport; the
#          rank-0/rank-1 control frames ride plain TCP on the same rail.
#
#   Process model (modeled on ~/vllm-strix-halo.sh, minus systemd):
#     start   rank 1 first (nohup on Box2, pid file there), then rank 0 (nohup
#             here), then poll /health through the load stages until 200.
#     stop    SIGINT rank 0 — tp_serve forwards `shutdown` to rank 1 and both
#             ranks unload cleanly; TERM/KILL escalation is applied ONLY to
#             pids whose /proc/<pid>/cmdline is this stack's tp_serve.py /
#             tp_rank1.py. vllm / ds4-vllm / ray / podman jobs are never
#             touched, and a foreign listener on the port is reported, not killed.
#
#   Context knobs:
#     EXL3_TP_SAFE_CTX       serving context   (default 524288)
#     EXL3_TP_CACHE_TOKENS   cache slots/rank  (default 524288)
#     Both are exported to rank 0 and forwarded to rank 1 (ssh forwards nothing;
#     tp_serve_odl.sh forwards them too).
#
# Usage:
#   ./exl3ds4.sh                     # start (default action; idempotent)
#   ./exl3ds4.sh start               # no-op if already ready or loading
#   ./exl3ds4.sh stop                # graceful stop on BOTH boxes
#   ./exl3ds4.sh restart             # stop, then start
#   ./exl3ds4.sh status              # link/ranks/health; exit 0 ready, 3 loading, 1 down
#   ./exl3ds4.sh logs [rank0|rank1|all]   # follow logs (default rank0)
#
# Locations (all overridable, defaults = the validated rig): EXL3DS4_ROOT (this
#   checkout), EXL3DS4_REMOTE (Box2 ssh target), EXL3_MASTER_ADDR (rank-0 Thunderbolt
#   IP the launchers export as EXLLAMA_MASTER_ADDR), EXL3_ODL_BUILD (OdinLink build
#   dir), EXL3_TP_MODEL_DIR / EXL3_TP_ENGRAM_DIR (per-rank data paths).
# Other knobs: EXL3_HTTP_PORT (1234)  EXL3_MASTER_PORT (29650)
#   EXL3_TP_CONTROL_PORT (29651)  EXL3_TP_DSPARK (default 1 = on)
#   EXL3_TP_BUDGETS (default 95,104 with DSpark)  EXL3_TP_SPAN (prefill chunk, 2048)
#   EXL3_TP_MEM_FLOOR (GiB kept free per rank at startup, default 2; 0 disables)
#   EXL3_TP_VISION (default 1 = rank 0 loads the vision tower; a load or floor failure
#     degrades to text-only unless EXL3_TP_VISION_STRICT=1)  EXL3_TP_MAX_IMAGES (8)
#   EXL3_TP_DSPARK_CONF  EXL3_DS41_CONF  EXL3_TP_MODEL_DIR
#   EXL3_LOAD_TIMEOUT (2700 s)  EXL3_STOP_TIMEOUT (120 s)
#
# Logs: rank 0 -> ~/exl3ds4/logs/exl3ds4-rank0.log (Box1)
#       rank 1 -> ~/exl3ds4/logs/exl3ds4-rank1.log (Box2; 'exl3ds4.sh logs rank1')
# =============================================================================
set -uo pipefail

# ----------------------------- configuration --------------------------------
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REMOTE="${EXL3DS4_REMOTE:-davidcanar@10.0.2.2}"   # Box2 over the Thunderbolt rail (example)
ODL_DEV="/dev/odl_tb5_0"

TP_RANK0_SH="$ROOT/tools/ds41/tp_serve_rank0.sh"
TP_RANK1_SH="$ROOT/tools/ds41/tp_rank1.sh"
EXL3_TP_MODEL_DIR="${EXL3_TP_MODEL_DIR:-$ROOT/model}"   # paths must mirror on Box2
MODEL_DIR="$EXL3_TP_MODEL_DIR"

HTTP_PORT="${EXL3_HTTP_PORT:-1234}"        # OpenAI API, 0.0.0.0 (Box1)
MASTER_PORT="${EXL3_MASTER_PORT:-29650}"   # NCCL rendezvous
CONTROL_PORT="${EXL3_TP_CONTROL_PORT:-29651}"  # rank0<->rank1 control channel

EXL3_TP_SAFE_CTX="${EXL3_TP_SAFE_CTX:-524288}"
EXL3_TP_CACHE_TOKENS="${EXL3_TP_CACHE_TOKENS:-524288}"

LOAD_TIMEOUT="${EXL3_LOAD_TIMEOUT:-2700}"  # max seconds to wait for /health 200
STOP_TIMEOUT="${EXL3_STOP_TIMEOUT:-120}"   # max seconds for graceful unload per rank

RUN_DIR="$ROOT/run"
LOG_DIR="$ROOT/logs"
PID0="$RUN_DIR/exl3ds4-rank0.pid"          # Box1
R1_PID="$RUN_DIR/exl3ds4-rank1.pid"        # same path, on Box2
R1_LOG="$LOG_DIR/exl3ds4-rank1.log"        # on Box2
LOG0="$LOG_DIR/exl3ds4-rank0.log"          # on Box1
HLTH="$RUN_DIR/health.last.json"
# -----------------------------------------------------------------------------

log()  { printf '\033[1;32m[exl3ds4]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[exl3ds4]\033[0m %s\n' "$*" >&2; }
die()  { warn "$*"; exit 1; }

remote() { timeout "${2:-30}" ssh -o BatchMode=yes "$REMOTE" "$1"; }

odl_up() { [[ -e "$ODL_DEV" ]] && grep -q odl_tb5 /proc/modules; }

lan_ip() { ip -4 route get 8.8.8.8 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}'; }

# ---- knob validation and defaults ------------------------------------------
# Validate the RESOLVED values (the EXL3_* envs themselves may be unset).
for v in HTTP_PORT MASTER_PORT CONTROL_PORT \
         EXL3_TP_SAFE_CTX EXL3_TP_CACHE_TOKENS LOAD_TIMEOUT STOP_TIMEOUT; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v must be a positive integer (got '${!v}'; check the EXL3_* env of the same meaning)"
done

case "${EXL3_TP_DSPARK:-1}" in          # DSpark ON by default
    1|true|yes|on)  EXL3_TP_DSPARK=1 ;;
    0|false|no|off) EXL3_TP_DSPARK=0 ;;
    *) die "EXL3_TP_DSPARK must be 0/1/true/false/yes/no/on/off (got '${EXL3_TP_DSPARK}')" ;;
esac
export EXL3_TP_SAFE_CTX EXL3_TP_CACHE_TOKENS EXL3_TP_DSPARK EXL3_TP_MODEL_DIR

dspark_label() { [[ "$EXL3_TP_DSPARK" == 1 ]] && echo "ON (MTP speculative decoding)" || echo "OFF"; }

# Variables BOTH ranks must see identically (ssh forwards nothing, so they
# travel inside the rank-1 launch command; rank 0 inherits them via export).
# EXL3_TP_SPAN and EXL3_TP_CACHE_TOKENS MUST match on both ranks: the prefill
# chunk sequences and the DSA block-table sizing are derived from them, and a
# mismatch wedges the collectives instead of erroring.
PASS_ENV=()
for v in EXL3_TP_DSPARK EXL3_TP_BUDGETS EXL3_TP_DSPARK_CONF EXL3_DS41_CONF \
         EXL3_TP_SAFE_CTX EXL3_TP_CACHE_TOKENS EXL3_TP_MODEL_DIR \
         EXL3_TP_SPAN EXL3_TP_USE_PER_DEVICE EXL3_TP_MEM_FLOOR \
         EXL3_TP_VISION EXL3_TP_VISION_STRICT EXL3_TP_MAX_IMAGES \
         EXL3_TP_ODL_AR2 EXL3_TP_ODL_AR2_PEER EXL3_TP_ODL_AR2_PORT EXL3_TP_ODL_AR2_MAX \
         EXL3_TP_SHARED_EXPERT_DEVICE EXL3_DSV4_CAND_ROWS EXL3_TP_NCCL_SYNC EXL3_TP_NCCL_FP32 \
         EXL3_ROCM_MOE_DECODE_RANGE; do
    [[ -n "${!v:-}" ]] && PASS_ENV+=("$v=${!v}")
done
REMOTE_ENV=""
for spec in "${PASS_ENV[@]}"; do REMOTE_ENV+="$(printf '%q ' "$spec")"; done

mkdir -p "$RUN_DIR" "$LOG_DIR"

# ---- pid helpers (cmdline-verified: a pidfile alone proves nothing) ---------
# A zombie reads as dead here (empty /proc/<pid>/cmdline), and a recycled pid
# only counts if its cmdline is this stack's entry point.
rank0_state() {  # -> "pid etime" when rank 0 is alive
    local p c e
    p=$(cat "$PID0" 2>/dev/null) || return 1
    [[ -n "$p" ]] || return 1
    kill -0 "$p" 2>/dev/null || return 1
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    [[ "$c" == *tp_serve.py* ]] || return 1
    e=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
    echo "$p ${e:-?}"
}

rank1_state() {  # -> "pid etime" when rank 1 is alive on Box2
    remote "p=\$(cat '$R1_PID' 2>/dev/null) || exit 1; [ -n \"\$p\" ] || exit 1
            kill -0 \"\$p\" 2>/dev/null || exit 1
            c=\$(tr '\\0' ' ' < /proc/\"\$p\"/cmdline 2>/dev/null)
            [[ \"\$c\" == *tp_rank1* ]] || exit 1
            e=\$(ps -o etime= -p \"\$p\" 2>/dev/null | tr -d ' ')
            echo \"\$p \${e:-?}\"" 15 2>/dev/null
}

wait_pid_exit() {  # $1 pid  $2 seconds — 0 when it went away
    local pid=$1 secs=$2 i=0
    while (( i < secs )); do
        kill -0 "$pid" 2>/dev/null || return 0
        [[ -z "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" ]] && return 0  # zombie
        sleep 2; i=$((i + 2))
    done
    kill -0 "$pid" 2>/dev/null && return 1
    return 0
}

# ---- health helpers --------------------------------------------------------
health_code() {  # curl already prints 000 on a failed transfer; do not double it
    local c
    c=$(curl -s -m 5 -o "$HLTH" -w '%{http_code}' "http://127.0.0.1:$HTTP_PORT/health" 2>/dev/null || true)
    echo "${c:-000}"
}
health_message() {
    sed -n 's/.*"message": *"\([^"]*\)".*/\1/p' "$HLTH" 2>/dev/null | head -1
}

# =============================================================================
do_start() {
    trap 'warn "interrupted — cleaning up both ranks"; do_stop >/dev/null 2>&1; trap - INT TERM; exit 130' INT TERM

    # ---- idempotency: never double-load ------------------------------------
    if rank0_state >/dev/null; then
        local code; code=$(health_code)
        if [[ "$code" == 200 ]]; then
            log "DS4.1 EXL3 server already READY on :$HTTP_PORT — nothing to do ('$0 status' / '$0 logs')"
        else
            log "rank 0 already running and loading on :$HTTP_PORT — attach with '$0 logs' / '$0 status'"
        fi
        trap - INT TERM; exit 0
    fi

    # ---- preflight ---------------------------------------------------------
    [[ -x "$TP_RANK0_SH" ]] || die "rank 0 launcher missing: $TP_RANK0_SH"
    [[ -x "$ROOT/.venv/bin/python" ]] || die "venv python missing: $ROOT/.venv/bin/python"
    [[ -f "$MODEL_DIR/config.json" ]] || die "model not found on Box1: $MODEL_DIR (EXL3_TP_MODEL_DIR overrides)"
    remote "test -x '$TP_RANK1_SH' && test -x '$ROOT/.venv/bin/python' && test -f '$MODEL_DIR/config.json'" 30 >/dev/null 2>&1 \
        || die "Box2 ($REMOTE) missing $TP_RANK1_SH, the venv, or the model at $MODEL_DIR (paths must mirror Box1)."

    ip -brief addr show thunderbolt0 2>/dev/null | grep -q UP \
        || die "thunderbolt0 is down — check the USB4 cable / peer box (control channel rides it)."
    odl_up || die "OdinLink not up on Box1 ($ODL_DEV or odl_tb5 module) — NCCL_NET=ODL_TB5 has no TCP fallback here."
    remote "test -e '$ODL_DEV' && grep -q odl_tb5 /proc/modules" 15 >/dev/null 2>&1 \
        || die "OdinLink not up on Box2 ($REMOTE) — NCCL_NET=ODL_TB5 has no TCP fallback here."

    # ---- reap strays of THIS stack only (no pidfile = previous crash) ------
    if pgrep -f "exl3ds4/tools/ds41/[t]p_serve\.py" >/dev/null 2>&1; then
        warn "stale tp_serve.py found without a live pidfile — reaping (scoped to this stack only)"
        pkill -INT -f "exl3ds4/tools/ds41/[t]p_serve\.py" 2>/dev/null || true
        sleep 10
        pkill -KILL -f "exl3ds4/tools/ds41/[t]p_serve\.py" 2>/dev/null || true
    fi
    remote "pkill -TERM -f 'exl3ds4/tools/ds41/[t]p_rank1\.py' 2>/dev/null || true; rm -f '$R1_PID'" 30 >/dev/null 2>&1 || true
    rm -f "$PID0"

    # ---- port checks: report foreign holders, never kill them -------------
    local held
    held=$(ss -ltn 2>/dev/null | grep -E ":($HTTP_PORT|$MASTER_PORT|$CONTROL_PORT)\b" || true)
    if [[ -n "$held" ]]; then
        warn "port(s) already held on Box1:"
        printf '  %s\n' $held >&2
        die "if that is the ds4-vllm stack use '~/vllm-strix-halo.sh ds4 stop' — this script never touches foreign servers."
    fi

    # ---- rank 1 on Box2 (nohup; pid + log live on Box2) --------------------
    log "Starting rank 1 on $REMOTE (log on Box2: $R1_LOG)..."
    remote "{ echo; echo \"===== $(date '+%F %T') exl3ds4 rank 1 (master=$MASTER_PORT control=$CONTROL_PORT dspark=$EXL3_TP_DSPARK safe_ctx=$EXL3_TP_SAFE_CTX cache=$EXL3_TP_CACHE_TOKENS) =====\"; } >> '$R1_LOG'" 15 >/dev/null 2>&1 || true
    if ! remote "mkdir -p '$RUN_DIR'
                 nohup env $REMOTE_ENV '$TP_RANK1_SH' $MASTER_PORT $CONTROL_PORT >> '$R1_LOG' 2>&1 < /dev/null &
                 echo \$! > '$R1_PID'; sleep 1
                 p=\$(cat '$R1_PID')
                 kill -0 \"\$p\" 2>/dev/null && grep -aq tp_rank1 /proc/\"\$p\"/cmdline" 90 >/dev/null 2>&1; then
        remote "tail -n 15 '$R1_LOG' 2>/dev/null" 15 2>/dev/null | sed 's/^/  /' >&2 || true
        die "rank 1 failed to launch on $REMOTE (tail above; full log: $R1_LOG on Box2)"
    fi

    # ---- rank 0 on Box1 (nohup; pid file + log here) -----------------------
    log "Starting rank 0 on Box1 (HTTP 0.0.0.0:$HTTP_PORT; log: $LOG0)..."
    { echo; echo "===== $(date '+%F %T') exl3ds4 rank 0 (http=0.0.0.0:$HTTP_PORT master=$MASTER_PORT control=$CONTROL_PORT dspark=$EXL3_TP_DSPARK safe_ctx=$EXL3_TP_SAFE_CTX cache=$EXL3_TP_CACHE_TOKENS) ====="; } >> "$LOG0"
    nohup "$TP_RANK0_SH" "$HTTP_PORT" "$MASTER_PORT" "$CONTROL_PORT" >> "$LOG0" 2>&1 < /dev/null &
    echo $! > "$PID0"
    sleep 1
    if ! rank0_state >/dev/null; then
        tail -n 15 "$LOG0" 2>/dev/null | sed 's/^/  /' >&2 || true
        do_stop >/dev/null 2>&1 || true
        die "rank 0 failed to launch (log tail above)"
    fi

    # ---- wait for /health 200 through the load stages ----------------------
    log "Loading (stages print as /health reports them; timeout ${LOAD_TIMEOUT}s)..."
    local t0=$SECONDS i=0 last="" code msg
    while :; do
        code=$(health_code)
        [[ "$code" == 200 ]] && break
        rank0_state >/dev/null || start_fail "rank 0 exited during start-up"
        (( i % 30 == 0 )) && { [[ -n "$(rank1_state)" ]] || start_fail "rank 1 exited on Box2 during start-up"; }
        (( SECONDS - t0 >= LOAD_TIMEOUT )) && start_fail "timed out after ${LOAD_TIMEOUT}s waiting for :$HTTP_PORT (raise EXL3_LOAD_TIMEOUT)"
        msg=$(health_message)
        if [[ "$msg" != "$last" && -n "$msg" ]]; then
            log "  [$((SECONDS - t0))s] loading: $msg"
            last="$msg"
        fi
        sleep 10; i=$((i + 10))
    done

    # ---- ready -------------------------------------------------------------
    local ids tp p1 p0
    ids=$(curl -sf -m 5 "http://127.0.0.1:$HTTP_PORT/v1/models" 2>/dev/null | grep -oE '"id": *"[^"]+"' | sed 's/.*: *"//;s/"$//' | paste -sd, -)
    tp=$(curl -sf -m 5 "http://127.0.0.1:$HTTP_PORT/tp/status" 2>/dev/null | head -c 160 || true)
    read -r p1 _ < <(rank1_state)
    p0=$(cat "$PID0" 2>/dev/null)
    log "==================================================================="
    log " DS4.1 (EXL3, TP=2 over OdinLink) is ready!"
    log "   API        : http://${lan_ip:-10.0.2.1}:$HTTP_PORT   (also http://10.0.2.1:$HTTP_PORT)"
    log "   Model id   : ${ids:-?}"
    log "   Ranks      : 0 pid ${p0:-?} (Box1) | 1 pid ${p1:-?} (Box2)"
    log "   /tp/status : ${tp:-n/a}"
    log "   DSpark     : $(dspark_label)"
    log "   Context    : EXL3_TP_SAFE_CTX=$EXL3_TP_SAFE_CTX  EXL3_TP_CACHE_TOKENS=$EXL3_TP_CACHE_TOKENS"
    log "   Logs       : '$0 logs'   (rank0: $LOG0 | rank1 on Box2: $R1_LOG)"
    log " Stop with   : $0 stop"
    log "==================================================================="
    trap - INT TERM
}

start_fail() {  # $1 message — dump evidence, clean up BOTH ranks, exit 1
    warn "$1"
    warn "----- rank 0 log (last 30) -----"
    tail -n 30 "$LOG0" 2>/dev/null | sed 's/^/  /' >&2 || true
    warn "----- rank 1 log on $REMOTE (last 30) -----"
    remote "tail -n 30 '$R1_LOG' 2>/dev/null" 15 2>/dev/null | sed 's/^/  /' >&2 || true
    do_stop >/dev/null 2>&1 || true
    trap - INT TERM
    exit 1
}

# =============================================================================
do_stop() {
    log "Stopping the DS4.1 EXL3 cluster (rank 0 on Box1, rank 1 on $REMOTE)..."

    # rank 0 first: its SIGINT path forwards `shutdown` to rank 1, so both
    # ranks unload their caches cleanly (mirrors Ctrl-C on tp_serve_odl.sh).
    local p0 p
    p0=$(cat "$PID0" 2>/dev/null)
    if [[ -n "$p0" ]] && rank0_state >/dev/null; then
        log "  rank 0 (pid $p0): SIGINT (graceful; up to ${STOP_TIMEOUT}s)"
        kill -INT "$p0" 2>/dev/null || true
        if ! wait_pid_exit "$p0" "$STOP_TIMEOUT"; then
            warn "  rank 0 still alive — SIGTERM"
            kill -TERM "$p0" 2>/dev/null || true
            wait_pid_exit "$p0" 15 || { warn "  rank 0 still alive — SIGKILL"; kill -KILL "$p0" 2>/dev/null || true; }
        fi
        wait "$p0" 2>/dev/null || true
    elif pgrep -f "exl3ds4/tools/ds41/[t]p_serve\.py" >/dev/null 2>&1; then
        warn "  rank 0 pidfile stale but tp_serve.py alive — scoped reap (this stack only)"
        pkill -INT -f "exl3ds4/tools/ds41/[t]p_serve\.py" 2>/dev/null || true
        sleep 10
        pkill -KILL -f "exl3ds4/tools/ds41/[t]p_serve\.py" 2>/dev/null || true
    fi
    rm -f "$PID0"

    # rank 1 on Box2: usually already gone (rank 0 sent `shutdown`); verify,
    # never assume. Same cmdline-verified escalation as rank 0.
    p=$(remote "p=\$(cat '$R1_PID' 2>/dev/null) || exit 0
                kill -0 \"\$p\" 2>/dev/null || exit 0
                c=\$(tr '\\0' ' ' < /proc/\"\$p\"/cmdline 2>/dev/null)
                [[ \"\$c\" == *tp_rank1* ]] && echo \"\$p\"" 15 2>/dev/null)
    if [[ -n "$p" ]]; then
        log "  rank 1 (pid $p on Box2): SIGTERM (graceful)"
        remote "kill -TERM '$p' 2>/dev/null || true" 15 >/dev/null 2>&1 || true
        local i=0
        while (( i < STOP_TIMEOUT )); do
            remote "kill -0 '$p' 2>/dev/null" 10 >/dev/null 2>&1 || break
            sleep 3; i=$((i + 3))
        done
        remote "kill -0 '$p' 2>/dev/null" 10 >/dev/null 2>&1 \
            && { warn "  rank 1 still alive — SIGKILL"; remote "kill -9 '$p' 2>/dev/null || true" 15 >/dev/null 2>&1 || true; }
    fi
    # scoped fallback for a rank 1 without a pidfile; then drop the pidfile
    remote "pkill -TERM -f 'exl3ds4/tools/ds41/[t]p_rank1\.py' 2>/dev/null || true; rm -f '$R1_PID'" 30 >/dev/null 2>&1 || true

    # verify rather than assume — but never kill the foreign holder ourselves
    sleep 2
    local held
    held=$(ss -ltn 2>/dev/null | grep -E ":($HTTP_PORT|$MASTER_PORT|$CONTROL_PORT)\b" || true)
    if [[ -n "$held" ]]; then
        warn "port(s) still held on Box1 (NOT ours — left untouched):"
        printf '  %s\n' $held >&2
    else
        log "ports :$HTTP_PORT/:$MASTER_PORT/:$CONTROL_PORT free on Box1 — DS4.1 EXL3 cluster stopped."
    fi
}

# =============================================================================
do_status() {
    local rc=1 code s0 s1 p e held ids tp m
    log "DS4.1 EXL3 two-node server (controller $0)"
    log "  config : http=0.0.0.0:$HTTP_PORT  master=$MASTER_PORT  control=$CONTROL_PORT  dspark=$(dspark_label)"
    log "  context: EXL3_TP_SAFE_CTX=$EXL3_TP_SAFE_CTX  EXL3_TP_CACHE_TOKENS=$EXL3_TP_CACHE_TOKENS"

    # link + OdinLink fabric
    if ip -brief addr show thunderbolt0 2>/dev/null | grep -q UP; then
        log "  link   : thunderbolt0 UP ($(ip -brief addr show thunderbolt0 | awk '{print $3}'))"
    else
        warn "  link   : thunderbolt0 DOWN"
    fi
    if odl_up; then log "  odl    : $ODL_DEV present, odl_tb5 loaded (Box1)"
    else warn "  odl    : not up on Box1 ($ODL_DEV / odl_tb5 module)"; fi
    if remote "test -e '$ODL_DEV' && grep -q odl_tb5 /proc/modules" 15 >/dev/null 2>&1; then
        log "  odl    : $ODL_DEV present on Box2"
    else
        warn "  odl    : not up on Box2 ($REMOTE)"
    fi

    # ranks
    s0=$(rank0_state)
    if [[ -n "$s0" ]]; then
        read -r p e <<<"$s0"; log "  rank 0 : RUNNING pid $p (uptime $e) — Box1"
    elif [[ -f "$PID0" ]]; then
        warn "  rank 0 : NOT running (stale pidfile $(cat "$PID0" 2>/dev/null))"
    else
        warn "  rank 0 : not running"
    fi
    s1=$(rank1_state)
    if [[ -n "$s1" ]]; then
        read -r p e <<<"$s1"; log "  rank 1 : RUNNING pid $p (uptime $e) — Box2"
    elif remote "test -f '$R1_PID'" 10 >/dev/null 2>&1; then
        warn "  rank 1 : NOT running on Box2 (stale pidfile)"
    else
        warn "  rank 1 : not running on Box2"
    fi

    # API state: READY / LOADING(stage) / down
    code=$(health_code)
    case "$code" in
        200)
            rc=0
            ids=$(curl -sf -m 5 "http://127.0.0.1:$HTTP_PORT/v1/models" 2>/dev/null | grep -oE '"id": *"[^"]+"' | sed 's/.*: *"//;s/"$//' | paste -sd, -)
            tp=$(curl -sf -m 5 "http://127.0.0.1:$HTTP_PORT/tp/status" 2>/dev/null | head -c 160 || true)
            log "  api    : READY on http://${lan_ip:-10.0.2.1}:$HTTP_PORT  (models: ${ids:-?})"
            log "  tp     : /tp/status ${tp:-n/a}"
            ;;
        503)
            rc=3
            m=$(health_message)
            warn "  api    : LOADING — ${m:-"model load in progress"}"
            ;;
        000)
            [[ -n "$s0" ]] && warn "  api    : not answering yet on :$HTTP_PORT (rank 0 booting)" \
                                || warn "  api    : DOWN (:$HTTP_PORT not listening)"
            ;;
        *)
            warn "  api    : /health -> HTTP $code: $(head -c 160 "$HLTH" 2>/dev/null)"
            ;;
    esac

    held=$(ss -ltn 2>/dev/null | grep -oE ":($HTTP_PORT|$MASTER_PORT|$CONTROL_PORT)\b" | sort -u | paste -sd' ' -)
    [[ -n "$held" ]] && log "  listen : $held (Box1)" || warn "  listen : none of :$HTTP_PORT/:$MASTER_PORT/:$CONTROL_PORT held on Box1"

    exit "$rc"   # 0 ready, 3 loading, 1 down
}

# =============================================================================
do_logs() {
    local which="${1:-rank0}"
    case "$which" in
        rank0)
            [[ -f "$LOG0" ]] || warn "no rank 0 log yet ($LOG0) — server never started from this controller"
            exec tail -n 50 -F "$LOG0" 2>/dev/null
            ;;
        rank1)
            exec ssh -o BatchMode=yes "$REMOTE" "tail -n 50 -F '$R1_LOG' 2>/dev/null"
            ;;
        all)
            tail -n 20 -F "$LOG0" 2>/dev/null | sed -u 's/^/[rank0] /' &
            local a=$!
            ssh -o BatchMode=yes "$REMOTE" "tail -n 20 -F '$R1_LOG' 2>/dev/null" | sed -u 's/^/[rank1] /' &
            local b=$!
            trap 'kill $a $b 2>/dev/null' INT TERM
            warn "following both ranks — Ctrl-C to detach (server keeps running)"
            wait
            ;;
        *) die "logs: expected rank0, rank1 or all (got '$which')" ;;
    esac
}

# =============================================================================
ACTION="${1:-start}"; shift || true
case "$ACTION" in
    start|stop|restart|status|logs|help|-h|--help) ;;
    *) die "unknown action '$ACTION' — usage: $0 [start|stop|restart|status|logs [rank0|rank1|all]]" ;;
esac
LOG_WHICH="rank0"
if [[ "$ACTION" == logs && "${1:-}" != "" ]]; then
    case "$1" in rank0|rank1|all) LOG_WHICH="$1" ;; *) die "logs: expected rank0, rank1 or all (got '$1')" ;; esac
fi

case "$ACTION" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  do_status ;;
    logs)    do_logs "$LOG_WHICH" ;;
    help|-h|--help) sed -n '2,43p' "$0" ;;
esac
