#!/usr/bin/env python3
"""Single-GPU microbenchmark of one DS4.1 MoE layer (BlockSparseMLP + shared expert).

Loads only layers.<L>.ffn from the EXL3 checkpoint and times forward() for decode-sized row counts.
An expert range ("--range a:b") rebuilds the module as a TP rank would hold it (experts a..b-1
local, out-of-range picks contribute zero), so the TP kernel path can be measured on one box.

  python perf/moe_bench.py --layer 10 --rows 1,6 [--range 0:192] [--no-shared]
env: EXL3_ROCM_MOE_DECODE=0 forces the cooperative kernels instead of the RDNA decode path.
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "exllamav3-dsv41f"))

from exllamav3 import Config, Model  # noqa: E402
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP  # noqa: E402


def shard(full: BlockSparseMLP, first: int, last: int, keep_shared: bool) -> BlockSparseMLP:
    """The module a TP rank holding experts [first, last) runs (replicated routing gate)."""
    # The kwargs tp_export sends (the constructor folds interm_div into the routing scale again)
    m = BlockSparseMLP(
        config = None,
        key = full.key,
        hidden_size = full.hidden_size,
        latent_size = full.latent_size,
        intermediate_size = full.intermediate_size,
        activation_fn = full.activation_fn,
        num_experts = full.num_experts,
        num_experts_per_tok = full.num_experts_per_tok,
        interm_dtype = full.interm_dtype,
        interm_div = full.interm_div,
        router_type = full.router_type,
        routed_scaling_factor = full.routed_scaling_factor / full.interm_div
            if (full.interm_div != 1.0 and full.router_type != "std") else full.routed_scaling_factor,
        n_group = full.n_group,
        topk_group = full.topk_group,
        act_limit = full.act_limit,
        alt_residual_channel = full.alt_residual_channel,
        num_local_experts = last - first,
        gates = full.gates[first:last],
        ups = full.ups[first:last],
        downs = full.downs[first:last],
        shared_experts = full.shared_experts if keep_shared else None,
        routing_gate = full.routing_gate,
        routing_first = first,
        routing_last = last,
    )
    m.device = full.device
    m.e_score_correction_bias = full.e_score_correction_bias
    m.e_score_bias_vl = getattr(full, "e_score_bias_vl", None)
    m.per_expert_scale = getattr(full, "per_expert_scale", None)
    m.load_local()
    m.load_routing()
    return m


def check(args):
    dev = torch.device("cuda:0")
    config = Config.from_directory(args.model)
    model = Model.from_config(config, component = "text")
    blk = next(m for m in model.modules if getattr(m, "key", "") == f"layers.{args.layer}")
    full = blk.mlp
    ok = True
    with torch.inference_mode():
        full.load(dev)
        half_e = full.num_experts // 2
        # every row count through every path: full (RDNA, all local), the two halves on the RDNA
        # range path, the first half again on the cooperative kernels
        os.environ["EXL3_ROCM_MOE_DECODE_RANGE"] = "1"
        a = shard(full, 0, half_e, True)
        b = shard(full, half_e, full.num_experts, False)
        os.environ["EXL3_ROCM_MOE_DECODE_RANGE"] = "0"
        a_coop = shard(full, 0, half_e, True)
        os.environ["EXL3_ROCM_MOE_DECODE_RANGE"] = "1"
        for rows in [int(r) for r in args.rows.split(",")]:
            torch.manual_seed(rows)
            x = torch.randn((1, rows, full.hidden_size), device = dev, dtype = torch.half)
            # forward() returns a view of a static buffer every instance shares: copy out
            y_full = full.forward(x, {}).float().clone()
            y_a = a.forward(x, {}).float().clone()
            y_b = b.forward(x, {}).float().clone()
            y_ac = a_coop.forward(x, {}).float().clone()
            scale = y_full.abs().max().item()
            e_sum = (y_a + y_b - y_full).abs().max().item() / scale
            e_coop = (y_a - y_ac).abs().max().item() / max(y_ac.abs().max().item(), 1e-9)
            good = e_sum < 2e-3 and e_coop < 2e-3 and torch.isfinite(y_a).all().item()
            ok &= good
            print(f"  rows {rows}: |halves - full| / max {e_sum:.2e}   |rdna range - coop| / max {e_coop:.2e}  "
                  f"{'OK' if good else 'MISMATCH'}", flush = True)
    print("check:", "PASS" if ok else "FAIL", flush = True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("EXL3_TP_MODEL_DIR", os.path.join(ROOT, "model")))
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--rows", default="1,6")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--range", default="")
    ap.add_argument("--no-shared", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="correctness: shard sums vs the full module, RDNA range path vs cooperative kernels")
    args = ap.parse_args()
    if args.check:
        return check(args)

    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    config = Config.from_directory(args.model)
    model = Model.from_config(config, component = "text")
    blk = next(m for m in model.modules if getattr(m, "key", "") == f"layers.{args.layer}")
    mlp = blk.mlp
    with torch.inference_mode():
        mlp.load(dev)
        if args.range:
            a, b = (int(v) for v in args.range.split(":"))
            mlp = shard(mlp, a, b, not args.no_shared)
        elif args.no_shared:
            mlp.shared_experts = None
        rd = os.environ.get("EXL3_ROCM_MOE_DECODE", "1")
        print(f"layer {args.layer}: {mlp.num_local_experts or mlp.num_experts} local experts "
              f"(range {args.range or 'all'}), shared {'yes' if mlp.shared_experts is not None else 'no'}, "
              f"EXL3_ROCM_MOE_DECODE={rd}", flush = True)
        for rows in [int(r) for r in args.rows.split(",")]:
            # unit-RMS hidden states, as the ffn_norm hands them over
            x = torch.randn((1, rows, mlp.hidden_size), device = dev, dtype = torch.half)
            for _ in range(20):
                y = mlp.forward(x, {})
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                y = mlp.forward(x, {})
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / args.iters
            e0 = torch.cuda.Event(enable_timing = True)
            e1 = torch.cuda.Event(enable_timing = True)
            e0.record()
            for _ in range(args.iters):
                y = mlp.forward(x, {})
            e1.record()
            torch.cuda.synchronize()
            print(f"  rows {rows}: {dt * 1e6:8.1f} us/forward (host loop), "
                  f"{e0.elapsed_time(e1) / args.iters * 1e3:8.1f} us (events)  out {tuple(y.shape)} "
                  f"|y| {float(y.float().norm()):.3f}", flush = True)


if __name__ == "__main__":
    main()
