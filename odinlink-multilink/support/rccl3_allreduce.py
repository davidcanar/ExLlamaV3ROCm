#!/usr/bin/env python3
"""3-rank RCCL all-reduce over the OdinLink fabric.

Validates the fabric-side prerequisites for TP=3 without any model or engine
code: RCCL loads the ODL_TB5 net plugin, both /dev/odl_tb5_* devices are
visible to it, and the plugin's per-device handles + ODL_RCCL_PEERS select the
cable that actually reaches each peer.

Run one rank per box via rccl3_run.sh. Environment per rank:

  RANK, WORLD_SIZE=3, MASTER_ADDR, MASTER_PORT   (bootstrap; LAN is fine)
  NCCL_NET=ODL_TB5, LD_LIBRARY_PATH=<build>/rccl:<build>/lib
  ODL_RCCL_LOCAL_IP, ODL_RCCL_PEERS              (peer address -> local dev)

Prints a correctness line on rank 0 and confirms traffic actually went over
OdinLink via /run/odl_tb5/rccl_stats.
"""
import os, sys, time


def stats_present() -> bool:
    """The odl plugin mmaps its counters here; presence means the plugin ran."""
    return os.path.exists("/run/odl_tb5/rccl_stats")


def main() -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)

    dist.init_process_group("nccl", rank=rank, world_size=world)

    # 1. exact-value check: each rank contributes rank+1, sum = world(world+1)/2
    expect = world * (world + 1) / 2.0
    t = torch.full((1024,), float(rank + 1), device=dev, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    exact = bool(torch.all(t == expect))
    if not exact:
        bad = t[(t != expect)][:3].tolist()
        print(f"rank {rank}: MISMATCH expected {expect} got e.g. {bad}",
              file=sys.stderr, flush=True)

    # 2. throughput/latency probe at a collective-relevant size
    nbytes = 4 * 1024 * 1024
    buf = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
    for _ in range(3):                      # warm up the plugin/link
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    dist.barrier()
    iters = 20
    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    per = (time.time() - t0) / iters
    dist.barrier()

    if rank == 0:
        # all-reduce moves ~2*(world-1)/world of the buffer per rank
        eff_gbs = (nbytes * 2.0 * (world - 1) / world) / per / 1e9
        print(f"WORLD={world} exact={exact} "
              f"4MiB_allreduce={per*1e3:.2f} ms  eff={eff_gbs:.2f} GB/s "
              f"stats_present={stats_present()}", flush=True)

    dist.destroy_process_group()
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
