#!/usr/bin/env bash
# wait_loading_check.sh <tp3_start.sh>: run that script's wait_loading over ssh against a FAKE log
# and pid file in a temp dir on Box2 (the live stack's logs and pid files are never touched).
set -u
SCRIPT=$1
B2=${EXL3_BOX2:-192.168.0.103}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
T=$(mktemp -d /tmp/wl-check.XXXXXX)
R=$("${SSH[@]}" "$B2" 'mktemp -d /tmp/wl-check.XXXXXX')
LOG=$R                       # wait_loading reads $LOG/tp3-rank<r>.{log,pid} on the remote box
WD_OFFS=$T/offsets           # and the offsets record_offsets wrote here
eval "$(sed -n '/^wait_loading() {/,/^}/p' "$SCRIPT")"
eval "$(sed -n '/^offset_of() /p' "$SCRIPT")"
type wait_loading > /dev/null 2>&1 && type offset_of > /dev/null 2>&1 || { echo "functions not found in $SCRIPT"; exit 1; }
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', want '$3'"; fail=1; fi; }
remote() { "${SSH[@]}" "$B2" "$1"; }
record() { echo "1 $(remote "stat -c %s $R/tp3-rank1.log")" > "$WD_OFFS"; }

# an older start's line in the appended log, and a live "worker" (a sleep) behind the pid file
remote "printf 'tp: rank 1 cache 1048576 tokens (4096 pages of 256) -- an older start\ntp3: rank 1 received signal 2, exiting\n' > $R/tp3-rank1.log; nohup sleep 300 > /dev/null 2>&1 < /dev/null & echo \$! > $R/tp3-rank1.pid"
record

echo "-- 1: only an older start's line in the log"
out=$(wait_loading "$B2" 1 10); rc=$?
check "an older start's line does not pass" "$rc" 1
check "  ... and it says so" "$(echo "$out" | grep -c 'did not begin loading in 10s')" 1

echo "-- 2: this start's line arrives after a few seconds"
( sleep 7; remote "echo 'tp: rank 1 cache 1048576 tokens (4096 pages of 256); 4 live request slots' >> $R/tp3-rank1.log" ) &
t0=$(date +%s)
out=$(wait_loading "$B2" 1 60); rc=$?
wait
check "this start's line passes" "$rc" 0
check "  ... once it is there (5-15 s)" "$(( $(date +%s) - t0 >= 5 && $(date +%s) - t0 <= 15 ))" 1
check "  ... reported as loading" "$(echo "$out" | grep -c 'rank1 loading')" 1

echo "-- 3: the worker exits before its line"
record
remote "kill \$(cat $R/tp3-rank1.pid); echo 'ImportError: boom (injected)' >> $R/tp3-rank1.log"
t0=$(date +%s)
out=$(wait_loading "$B2" 1 120); rc=$?
check "an exited worker fails" "$rc" 1
check "  ... fast (< 15 s)" "$(( $(date +%s) - t0 < 15 ))" 1
check "  ... showing the end of its log" "$(echo "$out" | grep -c 'boom (injected)')" 1

echo "-- 4: offsets unknown (record_offsets could not stat the log): the whole log counts"
echo "1 -" > "$WD_OFFS"
remote "nohup sleep 300 > /dev/null 2>&1 < /dev/null & echo \$! > $R/tp3-rank1.pid"
out=$(wait_loading "$B2" 1 10); rc=$?
check "unknown offset falls back to the whole log" "$rc" 0

remote "kill \$(cat $R/tp3-rank1.pid) 2>/dev/null; rm -rf $R"
rm -rf "$T"
[ $fail = 0 ] && echo "ALL OK" || echo "SOME CHECKS FAILED"
exit $fail
