#!/usr/bin/env python3
"""3-rank decode all-reduce microbenchmark: RCCL (the backend's path: fp32 -> bf16 wire, all_reduce,
back to fp32, current-stream sync) vs odl_ar3 (OdinLink mesh, stream-ordered, no host sync).

Launched by ar3_bench.sh on all three boxes (RANK / WORLD / MASTER env). For each size:
  - correctness: per-rank random fp32 data; the odl_ar3 result must be bit-identical on every rank
    and within bf16 rounding of the RCCL result
  - latency: N back-to-back ops (pipelined, one sync at the end) and N ops each followed by a host
    sync (the RCCL path's real usage: EXL3_TP_NCCL_SYNC=stream)
"""
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "src", "exllamav3-dsv41f"))


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ.get("WORLD", "3"))
    master = os.environ.get("MASTER", "192.168.0.102")
    port = int(os.environ.get("MPORT", "29660"))
    iters = int(os.environ.get("ITERS", "400"))
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dist.init_process_group("nccl", rank=rank, world_size=world, init_method=f"tcp://{master}:{port}")
    w = torch.ones(6, device=dev)
    dist.all_reduce(w)
    torch.cuda.synchronize()

    from exllamav3.model.model_tp_odl import OdlAllReduceMesh
    mesh = OdlAllReduceMesh.create(rank, world, 0)
    if mesh is None:
        print(f"rank {rank}: odl_ar3 unavailable", flush=True)

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    for n in (5120, 10240, 30720):
        g = torch.Generator(device=dev)
        g.manual_seed(1234 + rank)
        a = torch.randn(n, device=dev, generator=g) * (rank + 1)
        ref = a.to(torch.bfloat16)
        dist.all_reduce(ref)
        ref = ref.float()
        torch.cuda.synchronize()
        res = {}
        if mesh is not None:
            b = a.clone()
            assert mesh.try_all_reduce_(b)
            torch.cuda.synchronize()
            gl = [torch.empty_like(b) for _ in range(world)]
            dist.all_gather(gl, b)
            same = all(torch.equal(gl[0], t) for t in gl)
            # exact canonical-order reference computed on every rank from everybody's inputs
            ins = [torch.empty_like(a) for _ in range(world)]
            dist.all_gather(ins, a)
            acc = torch.zeros_like(a)
            for t in ins:
                acc = acc + t.to(torch.bfloat16).float()
            exact = acc.to(torch.bfloat16).float()
            res["bit_identical_across_ranks"] = same
            res["equals_canonical_sum"] = bool(torch.equal(b, exact))
            res["max_abs_diff_vs_rccl"] = float((b - ref).abs().max())
            res["max_rel_diff_vs_rccl"] = float(((b - ref).abs() / ref.abs().clamp_min(1e-3)).max())

        # latency
        def rccl_op(t):
            tmp = t.to(torch.bfloat16)
            dist.all_reduce(tmp)
            t.copy_(tmp.float())

        x = a.clone()
        for _ in range(20):
            rccl_op(x)
            torch.cuda.current_stream().synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            rccl_op(x)
            torch.cuda.current_stream().synchronize()
        res["rccl_synced_us"] = (time.perf_counter() - t0) / iters * 1e6
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            rccl_op(x)
        torch.cuda.synchronize()
        res["rccl_pipelined_us"] = (time.perf_counter() - t0) / iters * 1e6

        if mesh is not None:
            y = a.clone()
            for _ in range(20):
                mesh.try_all_reduce_(y)
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(iters):
                mesh.try_all_reduce_(y)
                torch.cuda.current_stream().synchronize()
            res["ar3_synced_us"] = (time.perf_counter() - t0) / iters * 1e6
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(iters):
                mesh.try_all_reduce_(y)
            torch.cuda.synchronize()
            res["ar3_pipelined_us"] = (time.perf_counter() - t0) / iters * 1e6
            if mesh.lib.odl3_last_error():
                res["error_latched"] = True
        log(f"n={n} ({n * 2} wire bytes): " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                         for k, v in res.items()))
    # prefill-sized RCCL all-reduces (bf16 wire, as the backend sends a 2048-row fp32 chunk)
    for rows in (() if os.environ.get("ODL3_SKIP_RCCL_BIG") else (512, 2048)):
        big = torch.randn(rows, 5120, device=dev).to(torch.bfloat16)
        for _ in range(3):
            dist.all_reduce(big)
        torch.cuda.synchronize()
        dist.barrier()
        reps = 10
        t0 = time.perf_counter()
        for _ in range(reps):
            dist.all_reduce(big)
            torch.cuda.current_stream().synchronize()
        dt = (time.perf_counter() - t0) / reps
        nbytes = big.numel() * 2
        log(f"rccl prefill all_reduce {rows}x5120 bf16 ({nbytes / 2**20:.1f} MiB): {dt * 1e3:.2f} ms, "
            f"algbw {nbytes / dt / 1e9:.2f} GB/s, busbw {nbytes / dt / 1e9 * 2 * (world - 1) / world:.2f} GB/s")
        # the backend's real RCCL path for an fp32 prefill tensor: cast to bf16, reduce, cast back, sync
        x32 = torch.randn(rows, 5120, device=dev)
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(reps):
            tmp = x32.to(torch.bfloat16)
            dist.all_reduce(tmp)
            x32.copy_(tmp.float())
            torch.cuda.current_stream().synchronize()
        dt32 = (time.perf_counter() - t0) / reps
        log(f"rccl backend path {rows}x5120 fp32 (bf16 wire): {dt32 * 1e3:.2f} ms")
        if mesh is not None and getattr(mesh, "big", False):
            g = torch.Generator(device=dev)
            g.manual_seed(77 + rank)
            a = torch.randn(rows, 5120, device=dev, generator=g) * (rank + 1)
            b = a.clone()
            assert mesh.try_all_reduce_(b)
            torch.cuda.synchronize()
            gl = [torch.empty_like(b) for _ in range(world)]
            dist.all_gather(gl, b)
            same = all(torch.equal(gl[0], t) for t in gl)
            ins = [torch.empty_like(a) for _ in range(world)]
            dist.all_gather(ins, a)
            acc = torch.zeros_like(a)
            for t in ins:
                acc = acc + t.to(torch.bfloat16).float()
            exact = acc.to(torch.bfloat16).float()
            ref = a.to(torch.bfloat16)
            dist.all_reduce(ref)
            maxdiff = float((b - ref.float()).abs().max())
            y = a.clone()
            for _ in range(3):
                mesh.try_all_reduce_(y)
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(reps):
                mesh.try_all_reduce_(y)
                torch.cuda.current_stream().synchronize()
            dtb = (time.perf_counter() - t0) / reps
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(reps):
                mesh.try_all_reduce_(y)
            torch.cuda.synchronize()
            dtp = (time.perf_counter() - t0) / reps
            log(f"odl_ar3 big {rows}x5120 fp32: bit_identical={same} equals_canonical={bool(torch.equal(b, exact))} "
                f"max_abs_diff_vs_rccl={maxdiff:.3f} synced {dtb * 1e3:.2f} ms, pipelined {dtp * 1e3:.2f} ms "
                f"(algbw {rows * 5120 * 2 / dtb / 1e9:.2f} GB/s on the bf16 wire), err={mesh.lib.odl3_last_error()}")
    if mesh is not None:
        torch.cuda.synchronize()
        dist.barrier()
        mesh.close()
    dist.barrier()
    dist.destroy_process_group()
    log("done")


if __name__ == "__main__":
    main()
