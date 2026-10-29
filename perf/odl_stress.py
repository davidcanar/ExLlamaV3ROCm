#!/usr/bin/env python3
"""Cross-box RCCL stress over the OdinLink plugin: big all-reduces, every result checked exactly.

Prefill-sized collectives (MiB-scale, split into 1 MiB multi-frame stream messages per RCCL
channel) are what drain the odl_tb5 RX assembly pool. Loading the driver with rx_asm_pool=0
sends every message down the MSG_END copy path, so this exercises that path under load; a lost
message shows up as a hang (RCCL waits forever) and a corrupted one as a wrong sum.

  scripts/run_rank.sh <rank> ../perf/odl_stress.py <master-port> <control-port>   (both boxes)
env: STRESS_MB=1,4,16,64 (tensor sizes)  STRESS_ITERS=40  STRESS_CPU=0 (busy threads on this rank,
     starving the RCCL proxy the way a CPU-heavy job did when the original failure hit)
"""
import os
import threading
import time

import torch
import torch.distributed as dist

rank = int(os.environ["EXL3_TP_NODE_RANK"])
sizes = [int(x) for x in os.environ.get("STRESS_MB", "1,4,16,64").split(",")]
iters = int(os.environ.get("STRESS_ITERS", "40"))
cpu = int(os.environ.get("STRESS_CPU", "0"))

torch.cuda.set_device(0)
dist.init_process_group("nccl", rank=rank, world_size=2,
                        init_method=f"tcp://{os.environ['EXLLAMA_MASTER_ADDR']}:{os.environ['EXLLAMA_MASTER_PORT']}")

stop = threading.Event()


def burn():
    x = 0
    while not stop.is_set():
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF


for _ in range(cpu):
    threading.Thread(target=burn, daemon=True).start()

ok = True
for mb in sizes:
    n = mb * 1024 * 1024 // 2                                  # bf16 elements
    base = torch.arange(n, device="cuda", dtype=torch.int64) % 251
    t0 = time.perf_counter()
    for i in range(iters):
        # small integers are exact in bf16: rank r contributes (base + r + i) % 7 + 1
        x = ((base + rank + i) % 7 + 1).to(torch.bfloat16)
        dist.all_reduce(x)
        want = ((base + i) % 7 + 1) + ((base + 1 + i) % 7 + 1)
        if not torch.equal(x.to(torch.int64), want):
            bad = int((x.to(torch.int64) != want).sum())
            print(f"rank {rank}: {mb} MiB iter {i}: {bad} wrong elements", flush=True)
            ok = False
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"rank {rank}: {mb:3d} MiB x {iters}: {dt / iters * 1e3:8.2f} ms/all-reduce, "
          f"{mb * iters / dt / 1024:6.2f} GiB/s algbw", flush=True)
stop.set()
dist.barrier()
print(f"rank {rank}: stress {'PASS' if ok else 'FAIL'}", flush=True)
dist.destroy_process_group()
