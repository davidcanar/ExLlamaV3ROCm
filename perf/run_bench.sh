#!/usr/bin/env bash
# odl_ar2 vs RCCL latency microbenchmark across both boxes (nothing else may be using the GPUs/link)
set -uo pipefail
ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${EXL3DS4_REMOTE:-davidcanar@10.0.2.2}"
MP=${1:-29660}; CP=${2:-29661}
ssh -n -o BatchMode=yes "$REMOTE" "nohup $ROOT/scripts/run_rank.sh 1 ../perf/odl_bench.py $MP $CP > $ROOT/perf/odl_bench-rank1.log 2>&1 < /dev/null &"
timeout 600 "$ROOT/scripts/run_rank.sh" 0 ../perf/odl_bench.py "$MP" "$CP" 2>&1 | grep -v -E "iommu|^\s*$|RCCL version|HIP version|ROCm version|Hostname|Librccl"
