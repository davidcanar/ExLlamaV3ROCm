#!/usr/bin/env bash
# Shared ROCm 10.1 environment for the EXL3 DS4.1 project.
# Usage: source ~/exl3ds4/env.sh         (runtime)
#        source ~/exl3ds4/env.sh build   (native extension build)

EXL3DS4_ROOT="${EXL3DS4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EXL3_VENV="${EXL3_VENV:-$EXL3DS4_ROOT/.venv}"
EXL3_ROOT="${EXL3_ROOT:-$EXL3DS4_ROOT/src/exllamav3-dsv41f}"
EXL3_ROCM_SDK="${EXL3_ROCM_SDK:-$EXL3_VENV/lib/python3.12/site-packages/_rocm_sdk_devel}"
export EXL3DS4_ROOT EXL3_VENV EXL3_ROOT EXL3_ROCM_SDK

if [[ ! -x "$EXL3_VENV/bin/python" ]]; then
    echo "[exl3ds4] Missing Python environment: $EXL3_VENV" >&2
    return 1 2>/dev/null || exit 1
fi

# gfx1151 is native in this stack. Overriding it selects incorrect kernels.
unset HSA_OVERRIDE_GFX_VERSION

# The HSA runtime bundled with torch can crash on gfx1151. Preload the
# matching ROCm SDK runtime instead, as recommended by Kyojin.
_EXL3_HSA="$EXL3_ROCM_SDK/lib/libhsa-runtime64.so.1"
if [[ ! -r "$_EXL3_HSA" ]]; then
    echo "[exl3ds4] Missing HSA runtime: $_EXL3_HSA" >&2
    return 1 2>/dev/null || exit 1
fi
case ":${LD_PRELOAD:-}:" in
    *":$_EXL3_HSA:"*) ;;
    *) export LD_PRELOAD="$_EXL3_HSA${LD_PRELOAD:+:$LD_PRELOAD}" ;;
esac

export PATH="$EXL3_VENV/bin:$PATH"
export PYTHONPATH="$EXL3_ROOT:$EXL3DS4_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == "build" || "${EXL3_BUILD:-0}" == "1" ]]; then
    # SDK compiler paths are build-only. Do not export them for inference.
    export ROCM_HOME="$EXL3_ROCM_SDK"
    export ROCM_PATH="$EXL3_ROCM_SDK"
    export HIP_PATH="$EXL3_ROCM_SDK"
    export HIPCXX="$EXL3_ROCM_SDK/bin/hipcc"
    export PATH="$EXL3_ROCM_SDK/bin:$EXL3_ROCM_SDK/llvm/bin:$EXL3_ROCM_SDK/lib/llvm/bin:$PATH"

    if [[ -d "$EXL3_ROCM_SDK/lib/llvm/amdgcn/bitcode" ]]; then
        export HIP_DEVICE_LIB_PATH="$EXL3_ROCM_SDK/lib/llvm/amdgcn/bitcode"
    else
        export HIP_DEVICE_LIB_PATH="$EXL3_ROCM_SDK/amdgcn/bitcode"
    fi

    _EXL3_GCC_DIR="${EXL3_GCC_INSTALL_DIR:-$(dirname "$(gcc -print-libgcc-file-name)")}"
    # Fedora installs ROCm 7.1 headers in /usr/include/hip. SDK clang otherwise
    # searches those before its own 10.1 headers during the HIP device pass.
    export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=$_EXL3_GCC_DIR --rocm-device-lib-path=$HIP_DEVICE_LIB_PATH --rocm-path=$EXL3_ROCM_SDK -isystem $EXL3_ROCM_SDK/include"
    export PYTORCH_ROCM_ARCH=gfx1151
    export MAX_JOBS="${MAX_JOBS:-6}"
fi
