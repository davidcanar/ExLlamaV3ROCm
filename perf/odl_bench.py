#!/usr/bin/env python3
"""Cross-box all-reduce latency: RCCL (torch.distributed over the ODL_TB5 net plugin) vs odl_ar2.

Run on both boxes through scripts/run_rank.sh (it sets the NCCL / plugin environment):
  scripts/run_rank.sh <rank> ../perf/odl_bench.py <master-port> <control-port>
"""
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

ROOT = os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "exllamav3-dsv41f"))
from exllamav3.model.model_tp_odl import OdlAllReduce  # noqa: E402

rank = int(os.environ["EXL3_TP_NODE_RANK"])
torch.cuda.set_device(0)
dist.init_process_group("nccl", rank=rank, world_size=2,
                        init_method=f"tcp://{os.environ['EXLLAMA_MASTER_ADDR']}:{os.environ['EXLLAMA_MASTER_PORT']}")
x = torch.ones(6, device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
ar = OdlAllReduce(rank)


def lat_loop(fn, t, n=400):
    lat = []
    for _ in range(n):
        t.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(t)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e6)
    lat.sort()
    return statistics.median(lat), lat[int(0.9 * len(lat))]


def pipe_loop(fn, t, n=800):
    t.zero_()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(t)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def rccl(t):
    # the backend's RCCL path: fp32 -> bf16 wire -> fp32
    tmp = t.to(torch.bfloat16)
    dist.all_reduce(tmp)
    t.copy_(tmp.to(torch.float32))


def odl(t):
    assert ar.try_all_reduce_(t)


for nel in (5120, 10240, 30720):
    t = torch.full((nel,), float(rank + 1), device="cuda")
    odl(t)
    torch.cuda.synchronize()
    assert float(t[0]) == 3.0 and float(t[-1]) == 3.0, f"odl sum wrong: {t[0]} {t[-1]}"
    t.fill_(float(rank + 1))
    rccl(t)
    torch.cuda.synchronize()
    assert float(t[0]) == 3.0, f"rccl sum wrong: {t[0]}"
    for name, fn in (("rccl", rccl), ("odl", odl)):
        for _ in range(50):
            fn(t)
        torch.cuda.synchronize()
        med, p90 = lat_loop(fn, t)
        pipe = pipe_loop(fn, t)
        if rank == 0:
            print(f"{name:5s} {nel:6d} fp32 ({nel * 2} B wire): sync'd median {med:7.1f} us  p90 {p90:7.1f} us   "
                  f"back-to-back {pipe:7.1f} us/op", flush=True)

torch.cuda.synchronize()
dist.barrier()
ar.close()
dist.destroy_process_group()
print(f"rank {rank}: done", flush=True)
