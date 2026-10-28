#!/usr/bin/env python3
"""Small real-GPU smoke test; no model files required."""

import argparse
import importlib
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--require-ext", action="store_true")
args = parser.parse_args()

assert torch.cuda.is_available(), "ROCm GPU is not available"
props = torch.cuda.get_device_properties(0)
print(f"torch: {torch.__version__}")
print(f"HIP: {torch.version.hip}")
print(f"GPU: {props.name} ({props.gcnArchName})")

x = torch.randn(512, 512, dtype=torch.float16, device="cuda")
result = (x @ x).sum().item()
torch.cuda.synchronize()
assert result == result, "GPU result is NaN"
print(f"GPU matmul: OK ({result})")

try:
    ext = importlib.import_module("exllamav3_ext")
except ImportError as exc:
    if args.require_ext:
        raise
    print(f"exllamav3_ext: not built ({exc})")
else:
    print("exllamav3_ext: import OK")
    if hasattr(ext, "exl3_gemv_supported"):
        print(f"exl3_gemv_supported(0): {ext.exl3_gemv_supported(0)}")
