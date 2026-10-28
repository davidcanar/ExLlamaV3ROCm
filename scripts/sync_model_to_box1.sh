#!/usr/bin/env bash
set -euo pipefail

ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${EXL3DS4_REMOTE:-davidcanar@10.0.2.2}"
mkdir -p "$ROOT/model" "$ROOT/engram-src-1017" "$ROOT/logs"

rsync -a --whole-file --timeout=600 \
    --exclude='.cache/' --info=progress2 \
    "$REMOTE:$ROOT/model/" "$ROOT/model/"

rsync -a --whole-file --timeout=600 \
    --exclude='.cache/' --info=progress2 \
    "$REMOTE:$ROOT/engram-src-1017/" "$ROOT/engram-src-1017/"

echo "Model and Engram synchronization to Box1 completed successfully."
