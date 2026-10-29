#!/usr/bin/env bash
# RCCL stress across both boxes (perf/odl_stress.py): big all-reduces over the OdinLink plugin,
# every result checked. Nothing else may be using the GPUs/link (stop the server first).
# usage: run_stress.sh [master-port] [control-port]
# env: STRESS_MB, STRESS_ITERS (both ranks), STRESS_CPU (busy threads on Box1), STRESS_CPU1 (Box2)
set -uo pipefail
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${EXL3DS4_REMOTE:-davidcanar@10.0.2.2}"
MP=${1:-29662}; CP=${2:-29663}
ENVS="STRESS_MB=${STRESS_MB:-1,4,16,64} STRESS_ITERS=${STRESS_ITERS:-40}"
ssh -n -o BatchMode=yes "$REMOTE" "nohup env $ENVS STRESS_CPU=${STRESS_CPU1:-0} $ROOT/scripts/run_rank.sh 1 ../perf/odl_stress.py $MP $CP > $ROOT/perf/odl_stress-rank1.log 2>&1 < /dev/null &"
env $ENVS STRESS_CPU="${STRESS_CPU:-0}" timeout 900 "$ROOT/scripts/run_rank.sh" 0 ../perf/odl_stress.py "$MP" "$CP" 2>&1 \
    | grep -v -E "iommu|^\s*$|RCCL version|HIP version|ROCm version|Hostname|Librccl"
rc=${PIPESTATUS[0]}
sleep 2
ssh -o BatchMode=yes "$REMOTE" "grep -E 'MiB x|stress' $ROOT/perf/odl_stress-rank1.log" 2>/dev/null | sed 's/^/[box2] /'
exit "$rc"
