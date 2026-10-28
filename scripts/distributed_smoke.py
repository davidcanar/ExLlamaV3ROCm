#!/usr/bin/env python3
"""Two-node GPU RCCL smoke test using env:// rendezvous."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 2, "This smoke test currently expects exactly two ranks"

torch.cuda.set_device(0)
dist.init_process_group(
    "nccl",
    init_method="env://",
    rank=rank,
    world_size=world_size,
    timeout=timedelta(seconds=180),
)

try:
    for size in (1, 16_384, 1_048_576):
        value = torch.full(
            (size,), rank + 1, dtype=torch.float32, device="cuda"
        )
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) // 2
        torch.cuda.synchronize()
        assert torch.all(value == expected).item(), (
            f"all_reduce mismatch on rank {rank}, size {size}"
        )
        print(f"rank {rank}: all_reduce size={size} OK", flush=True)

    source = torch.tensor([41.0 if rank == 0 else -1.0], device="cuda")
    dist.broadcast(source, src=0)
    torch.cuda.synchronize()
    assert source.item() == 41.0
    print(f"rank {rank}: broadcast OK", flush=True)

    local = torch.tensor([rank + 10], dtype=torch.int64, device="cuda")
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    torch.cuda.synchronize()
    assert [item.item() for item in gathered] == [10, 11]
    print(f"rank {rank}: all_gather OK", flush=True)

    dist.barrier()
    if rank == 0:
        print("Two-node RCCL/GPU smoke test: PASS", flush=True)
finally:
    dist.destroy_process_group()
