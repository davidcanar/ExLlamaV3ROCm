#!/usr/bin/env bash
# =============================================================================
# setup.sh — one-shot installer for the two-node EXL3 / DS4.1 ROCm stack
#
# Run on BOTH boxes (paths and model data must mirror). For each box this
# script:
#
#   1. checks host tools (git, rsync, cmake, make, gcc, curl) and uv
#      (installs uv to ~/.local/bin via the official installer if missing)
#   2. creates the isolated Python 3.12 venv (.venv) and installs the pinned
#      ROCm 10.1 wheels (torch 2.14 + rocm 10.1 for gfx1151) and PyPI pins
#   3. clones ExLlamaV3 (anoane/exllamav3-anemone, PR #449 base d45e8961)
#      into src/exllamav3-dsv41f and applies patches/exllamav3 (git am)
#   4. clones Kyojin (Yamz-Labs/kyojin @ e2f3756) into src/kyojin as the
#      vendoring/reference source (the vendored layer is already committed
#      in tools/ds41; this is only to re-run vendor_kyojin.py / refresh)
#   5. clones OdinLink-Five (Geramy/OdinLink-Five @ 4534f58) into
#      $EXL3_ODL_SRC (default ~/.cache/odinlink), applies patches/odinlink,
#      configures cmake and builds the userspace library + RCCL v7 plugin
#      (the kernel module needs root and is NOT built here — instructions
#      are printed at the end)
#   6. builds the native ExLlamaV3 extension for gfx1151 and runs the GPU
#      smoke test (scripts/build_exllamav3.sh)
#   7. with --model (default): downloads the EXL3 quant to model/ and the
#      mandatory Engram source shards to engram-src-1017/ (about 385 GiB
#      total; hf resume is safe), and verifies the Engram SHA-256 sums
#
# Every step is idempotent: finished parts are detected and skipped.
#
# Usage:
#   ./setup.sh                  # everything (clone+build+model)
#   ./setup.sh --no-model       # skip the model/Engram download
#   ./setup.sh --no-odinlink    # skip the OdinLink clone/patch/build
#   ./setup.sh --no-kyojin      # skip the Kyojin reference clone
#   ./setup.sh --help
#
# Location overrides (all optional; defaults mirror the validated rig):
#   EXL3DS4_ROOT     install root            (default: this checkout's dir)
#   EXL3_ODL_SRC     OdinLink source clone   (default: ~/.cache/odinlink)
#   EXL3_ODL_BUILD   built plugin path       (default: ~/.cache/odinlink/build)
#   HF_TOKEN         hf token if the Engram repo needs auth on your side
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EXL3DS4_ROOT="${EXL3DS4_ROOT:-$ROOT}"
EXL3_ODL_SRC="${EXL3_ODL_SRC:-$HOME/.cache/odinlink}"
EXL3_ODL_BUILD="${EXL3_ODL_BUILD:-$EXL3_ODL_SRC/build}"

PY3=3.12
VENV="$EXL3DS4_ROOT/.venv"
VENV_PY="$VENV/bin/python"

AMD_NIGHTLY_INDEX="https://nightly.repo.amd.com/rocm/whl-next/"
AMD_STABLE_INDEX="https://stable.repo.amd.com/rocm/whl-next/"

EXL3_REPO="https://github.com/anoane/exllamav3-anemone.git"
EXL3_BASE="d45e89611738e9134f72f5c9679cc9eae113bba8"   # PR #449 / dsv41f
EXL3_DST="src/exllamav3-dsv41f"

KYOJIN_REPO="https://github.com/Yamz-Labs/kyojin.git"
KYOJIN_PIN="e2f37564bbae01a6ea8ad4176dfd97db09c253aa"   # release 1.8.2

ODL_REPO="https://github.com/Geramy/OdinLink-Five.git"
ODL_PIN="4534f58"

MODEL_REPO="LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
MODEL_DIR="$EXL3DS4_ROOT/model"
ENGRAM_REPO="deepseek-ai/DeepSeek-V4.1-Flash-1017"
ENGRAM_DIR="$EXL3DS4_ROOT/engram-src-1017"
ENGRAM_SHARDS=(
    "model-00047-of-00048.safetensors:76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5"
    "model-00048-of-00048.safetensors:9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd"
)

DO_MODEL=1; DO_ODL=1; DO_KYOJIN=1
for arg in "$@"; do
    case "$arg" in
        --no-model)    DO_MODEL=0 ;;
        --no-odinlink) DO_ODL=0 ;;
        --no-kyojin)   DO_KYOJIN=0 ;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown option '$arg' (see --help)" >&2; exit 2 ;;
    esac
done

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*" >&2; }
die()  { warn "$*"; exit 1; }

# -----------------------------------------------------------------------------
log "install root: $EXL3DS4_ROOT"

# ---- 1. host tools -----------------------------------------------------------
MISSING=()
for t in git rsync cmake make gcc curl sha256sum; do
    command -v "$t" >/dev/null 2>&1 || MISSING+=("$t")
done
((${#MISSING[@]})) && die "missing host tools: ${MISSING[*]} — install them (dnf/apt) and re-run."

if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    UV="$HOME/.local/bin/uv"
else
    log "uv not found — installing it to ~/.local/bin (official standalone installer)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
fi
log "using uv: $UV ($("$UV" --version))"

# ---- 2. venv + pinned wheels ------------------------------------------------
if [[ ! -x "$VENV_PY" ]]; then
    log "creating Python $PY3 venv at $VENV (uv downloads a managed CPython if needed)"
    "$UV" venv --python "$PY3" "$VENV"
else
    log "venv already exists: $VENV"
fi

if "$UV" pip list --python "$VENV_PY" 2>/dev/null | grep -q '^torch==2\.14'; then
    log "ROCm wheels already installed"
else
    log "installing pinned ROCm 10.1 wheels (torch[device-gfx1151] + rocm[devel,device-gfx1151])"
    log "  this downloads several GiB — the nightly snapshot pins are in requirements/rocm.txt"
    if ! "$UV" pip install --python "$VENV_PY" \
            --index-url "$AMD_NIGHTLY_INDEX" --extra-index-url https://pypi.org/simple \
            -r requirements/rocm.txt; then
        warn "nightly index install failed — retrying the stable ROCm 10.1 index"
        warn "(equivalent wheels at 2.14.0+rocm10.1.0 / 10.1.0; re-run the tests after install)"
        "$UV" pip install --python "$VENV_PY" \
            --index-url "$AMD_STABLE_INDEX" --extra-index-url https://pypi.org/simple \
            "torch[device-gfx1151]==2.14.0+rocm10.1.0" "rocm[devel,device-gfx1151]==10.1.0"
    fi
fi

log "installing pure-PyPI pins"
"$UV" pip install --python "$VENV_PY" \
    --index-url https://pypi.org/simple -r requirements/python.txt

# ---- 3. ExLlamaV3 clone + patches -------------------------------------------
if [[ -d "$EXL3_DST/.git" ]] && git -C "$EXL3_DST" rev-parse HEAD >/dev/null 2>&1; then
    HEAD_NOW=$(git -C "$EXL3_DST" rev-parse HEAD)
    WANT_HEAD=$(git -C "$EXL3_DST" rev-parse multi-node-rocm 2>/dev/null || true)
    if [[ "$HEAD_NOW" == "$WANT_HEAD" ]]; then
        log "ExLlamaV3 already cloned and patched at $EXL3_DST ($HEAD_NOW)"
    else
        warn "ExLlamaV3 at $EXL3_DST is not on the patched multi-node-rocm head"
        warn "  it is at $HEAD_NOW; re-run 'git am' manually or delete the clone to redo"
    fi
else
    log "cloning ExLlamaV3 (anoane/exllamav3-anemone) into $EXL3_DST"
    mkdir -p src
    git clone "$EXL3_REPO" "$EXL3_DST"
    git -C "$EXL3_DST" fetch origin dsv41f
    git -C "$EXL3_DST" checkout -b multi-node-rocm "$EXL3_BASE"
    log "applying patches/exllamav3 (git am — preserves authorship)"
    git -C "$EXL3_DST" am "$ROOT"/patches/exllamav3/*.patch
fi

# ---- 4. Kyojin reference clone ----------------------------------------------
if (( DO_KYOJIN )); then
    if [[ -d "src/kyojin/.git" ]]; then
        log "Kyojin reference already at src/kyojin"
    else
        log "cloning Kyojin (Yamz-Labs/kyojin @ $KYOJIN_PIN) into src/kyojin"
        git clone "$KYOJIN_REPO" src/kyojin
        git -C src/kyojin checkout "$KYOJIN_PIN"
    fi
fi

# ---- 5. OdinLink-Five clone + patch + build ---------------------------------
if (( DO_ODL )); then
    if [[ -d "$EXL3_ODL_SRC/.git" ]] && git -C "$EXL3_ODL_SRC" diff --quiet 2>/dev/null; then
        log "OdinLink already cloned and patched at $EXL3_ODL_SRC"
    elif [[ -d "$EXL3_ODL_SRC/.git" ]]; then
        log "OdinLink clone at $EXL3_ODL_SRC already carries local changes — leaving it alone"
    else
        log "cloning OdinLink-Five (Geramy/OdinLink-Five @ $ODL_PIN) into $EXL3_ODL_SRC"
        git clone "$ODL_REPO" "$EXL3_ODL_SRC"
        git -C "$EXL3_ODL_SRC" checkout "$ODL_PIN"
        log "applying patches/odinlink (5-file diff: driver multi-frame + RCCL v7 plugin)"
        git -C "$EXL3_ODL_SRC" apply "$ROOT"/patches/odinlink/*.patch
    fi

    if [[ -f "$EXL3_ODL_BUILD/rccl/librccl_net_odl_tb5.so" ]]; then
        log "OdinLink userspace + RCCL plugin already built at $EXL3_ODL_BUILD"
    else
        log "building OdinLink userspace + RCCL v7 plugin (cmake; ~minutes)"
        cmake -S "$EXL3_ODL_SRC" -B "$EXL3_ODL_BUILD"
        cmake --build "$EXL3_ODL_BUILD" -j"$(nproc)"
    fi
    cat <<EOF

  OdinLink kernel module (needs root, NOT built above):
      cmake --build "$EXL3_ODL_BUILD" --target driver
      sudo insmod  "$EXL3_ODL_BUILD/driver/odl_tb5.ko"
      sudo cp "$EXL3_ODL_SRC/driver/71-odl-tb5.rules" /etc/udev/rules.d/ && sudo udevadm control --reload-rules
    /dev/odl_tb5_0 appears only once the Thunderbolt peer connects.

EOF
fi

# ---- 6. build the ExLlamaV3 native extension --------------------------------
log "building the ExLlamaV3 native extension for gfx1151 (needs the GPU present)"
bash "$ROOT/scripts/build_exllamav3.sh"
log "GPU + extension smoke test passed"

# ---- 7. model + Engram data -------------------------------------------------
if (( DO_MODEL )); then
    HF="$VENV/bin/hf"
    [[ -x "$HF" ]] || die "hf CLI missing in the venv (huggingface-hub)"
    if [[ ! -f "$MODEL_DIR/config.json" ]]; then
        log "downloading the EXL3 quant ($MODEL_REPO) into $MODEL_DIR (43 shards, ~196.3 GiB)"
        "$HF" download "$MODEL_REPO" --local-dir "$MODEL_DIR"
    else
        log "model already present: $MODEL_DIR"
    fi
    mkdir -p "$ENGRAM_DIR"
    for spec in "${ENGRAM_SHARDS[@]}"; do
        shard="${spec%%:*}"; want_sha="${spec##*:}"
        if [[ -f "$ENGRAM_DIR/$shard" ]]; then
            log "Engram shard already present: $shard"
        else
            log "downloading Engram shard $shard ($ENGRAM_REPO) into $ENGRAM_DIR"
            "$HF" download "$ENGRAM_REPO" --include "$shard" --local-dir "$ENGRAM_DIR"
        fi
        log "verifying SHA-256 of $shard"
        got_sha=$(sha256sum "$ENGRAM_DIR/$shard" | awk '{print $1}')
        [[ "$got_sha" == "$want_sha" ]] || die "SHA-256 mismatch for $shard: got $got_sha, want $want_sha"
        log "  ok: $want_sha"
    done
    cat <<EOF

  Model data notes:
    - The Engram tables are MANDATORY and must come from
      $ENGRAM_REPO — do NOT substitute the older V4.1 release.
    - Both boxes need the model + Engram at the SAME absolute paths
      (rsync an already-downloaded copy instead of re-downloading:
       rsync -a --whole-file --exclude='.cache/' --info=progress2 \\
           <peer>:$EXL3DS4_ROOT/model/   $MODEL_DIR/
       rsync -a --whole-file --exclude='.cache/' --info=progress2 \\
           <peer>:$EXL3DS4_ROOT/engram-src-1017/ $ENGRAM_DIR/ )
EOF
fi

# ---- done ---------------------------------------------------------------------
cat <<EOF

=====================================================================
 Setup complete on this box ($EXL3DS4_ROOT).

 Next steps:
   1. Repeat this script on the second box (identical paths).
   2. Kernel cmdline for Strix Halo (both boxes, GRUB):
        amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856
   3. Load the OdinLink kernel module on both boxes (see note above);
      the stack needs /dev/odl_tb5_0 and the odl_tb5 module loaded.
   4. Sanity-check this box:
        source $EXL3DS4_ROOT/env.sh
        python $EXL3DS4_ROOT/scripts/smoke_torch.py --require-ext
   5. Start the server from the rank-0 box:
        $EXL3DS4_ROOT/exl3ds4.sh start
      and point clients at http://<rank0-lan-ip>:1234/v1/chat/completions
=====================================================================
EOF
