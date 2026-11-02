#!/usr/bin/env python3
"""One attempt of the TP=3 start-up collective sequence, with a watchdog per step.

Replays what a TP=3 rank does before and at its first prefill (no model): NCCL init + the 6-float
warmup (mp_warmup_nccl), optionally the odl_ar3 setup (ODL3_MODE=1: 4 int32 agreement all-reduces),
the all_gather of VRAM budgets, then the warm-up prefill's first collective (a 31 x 5120 fp32 tensor
sent as bf16 = 158720 elements), a few 2048-row ones, a barrier and destroy_process_group.

If a step does not finish within STEP_TIMEOUT seconds the watchdog prints "STALL <step>" and exits
the process with code 3 (the launcher then collects the plugin traces of that attempt). Prints
"OK" when every step finished. Env: RANK, WORLD, MASTER, MPORT (+ the usual NCCL/ODL env).
"""
import faulthandler
import os
import sys
import threading
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "src", "exllamav3-dsv41f"))

STEP_TIMEOUT = float(os.environ.get("STEP_TIMEOUT", "25"))
state = {"step": "start", "t": time.monotonic()}


def watchdog():
    while True:
        time.sleep(0.5)
        if time.monotonic() - state["t"] > STEP_TIMEOUT:
            print(f"\nRESULT STALL {state['step']} after {STEP_TIMEOUT:.0f}s", flush=True)
            faulthandler.dump_traceback(all_threads=False)
            os._exit(3)


def step(name):
    state["step"] = name
    state["t"] = time.monotonic()


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ.get("WORLD", "3"))
    threading.Thread(target=watchdog, daemon=True).start()
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    step("init_process_group")
    dist.init_process_group("nccl", rank=rank, world_size=world,
                            init_method=f"tcp://{os.environ.get('MASTER', '192.168.0.102')}:{os.environ['MPORT']}")
    step("warmup_6f")
    x = torch.ones((6,), device=dev)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    mesh = None
    if os.environ.get("ODL3_MODE", "0") == "1":
        step("odl_ar3_setup")
        from exllamav3.model.model_tp_odl import OdlAllReduceMesh
        mesh = OdlAllReduceMesh.create(rank, world, 0)
        torch.cuda.synchronize()
    step("allgather_vram")
    v = torch.tensor([123, 456], dtype=torch.int64, device=dev)
    g = [torch.empty_like(v) for _ in range(world)]
    dist.all_gather(g, v)
    torch.cuda.synchronize()
    step("first_prefill_allreduce_158720")
    a = torch.randn(31, 5120, device=dev).to(torch.bfloat16)
    dist.all_reduce(a)
    torch.cuda.current_stream().synchronize()
    for i in range(3):
        step(f"prefill_allreduce_2048rows_{i}")
        b = torch.randn(2048, 5120, device=dev).to(torch.bfloat16)
        dist.all_reduce(b)
        torch.cuda.current_stream().synchronize()
    step("barrier")
    dist.barrier()
    torch.cuda.synchronize()
    if mesh is not None:
        step("odl_ar3_close")
        mesh.close()
    step("destroy_process_group")
    dist.destroy_process_group()
    print("\nRESULT OK", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
