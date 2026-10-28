#!/usr/bin/env python3
"""Exercise ExLlamaV3's actual cross-host TPBackendNCCL on two GPUs."""

import os
import torch
from exllamav3.model.model_tp_backend import TPBackendNCCL

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 2

backend = TPBackendNCCL(
    device=0,
    active_devices=[0],
    output_device=0,
    init_method="env://",
    master=rank == 0,
    uuid="unused-cross-host",
    rank=rank,
    world_size=world_size,
    cross_host=True,
)

try:
    for step in range(12):
        value = torch.full((1, 315, 256), rank + 1, dtype=torch.float32, device="cuda")
        backend.all_reduce(value)
        assert torch.all(value == 3).item()
        print(f"rank {rank}: backend all_reduce step={step} OK", flush=True)

    value = torch.tensor([41.0 if rank == 0 else -1.0], device="cuda")
    backend.broadcast(value, 0)
    assert value.item() == 41.0
    print(f"rank {rank}: backend broadcast OK", flush=True)

    for name, gather in (("gather", backend.gather), ("gather_small", backend.gather_small)):
        local = torch.full((1, 315, 256), rank + 1, dtype=torch.float16, device="cuda")
        output = (
            torch.empty((1, 315, 512), dtype=torch.float16, device="cuda")
            if rank == 0
            else None
        )
        gather(local, output, [0, 1], 0, [256, 256])
        if rank == 0:
            assert torch.all(output[..., :256] == 1).item()
            assert torch.all(output[..., 256:] == 2).item()
        backend.fwd_barrier()
        print(f"rank {rank}: backend {name} OK", flush=True)

    local_width = 128 if rank == 0 else 384
    local = torch.full((1, 315, local_width), rank + 1, dtype=torch.float16, device="cuda")
    output = (
        torch.empty((1, 315, 512), dtype=torch.float16, device="cuda")
        if rank == 0
        else None
    )
    backend.gather(local, output, [0, 1], 0, [128, 384])
    if rank == 0:
        assert torch.all(output[..., :128] == 1).item()
        assert torch.all(output[..., 128:] == 2).item()
    backend.fwd_barrier()
    print(f"rank {rank}: backend uneven gather OK", flush=True)

    if rank == 0:
        print("Cross-host TPBackendNCCL smoke test: PASS", flush=True)
finally:
    backend.close()
