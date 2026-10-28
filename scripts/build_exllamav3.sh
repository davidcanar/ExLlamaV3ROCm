#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$ROOT/env.sh" build

uv pip install --python "$EXL3_VENV/bin/python" -r "$EXL3_ROOT/requirements.txt"
cd "$EXL3_ROOT"
"$EXL3_VENV/bin/python" setup.py build_ext --inplace
"$EXL3_VENV/bin/python" "$ROOT/scripts/smoke_torch.py" --require-ext
