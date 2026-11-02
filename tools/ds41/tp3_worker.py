#!/usr/bin/env python3
"""Generic TP worker of the 3-node DS4.1 TP server (ranks 1..N-1 via tools/ds41/tp3_worker.sh).

Identical to tp_rank1.py except that any worker rank >= 1 is accepted: with
world size 3, Box2 runs rank 1 and Box3 runs rank 2. The worker class itself
(tp_engine.Rank1Worker) is rank-agnostic; only this entry point and the env
(EXL3_TP_NODE_RANK / EXL3_TP_WORLD_SIZE) decide the rank.
"""
from __future__ import annotations

import os
import signal
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tp_engine  # noqa: E402


def main() -> int:
    rank = tp_engine.env_int("EXL3_TP_NODE_RANK", 1)
    world = tp_engine.env_int("EXL3_TP_WORLD_SIZE", 3)
    if rank < 1:
        print("tp3_worker: EXL3_TP_NODE_RANK must be >= 1 here; rank 0 runs tp3_serve.py",
              file=sys.stderr, flush=True)
        return 2
    control_port = tp_engine.env_int("EXL3_TP_CONTROL_PORT", 29651)
    master_addr = os.environ.get("EXLLAMA_MASTER_ADDR", "192.168.0.102")

    print(f"tp3: rank {rank} starting DS4.1 cached TP load (world size {world})", flush=True)
    backend = tp_engine.TPModelBackend.load(rank=rank, world_size=world, progress=False)
    channel = tp_engine.ControlChannel.connect(
        master_addr, control_port,
        deadline_s=tp_engine.env_float("EXL3_TP_CONNECT_TIMEOUT", 1800.0))
    channel.set_recv_timeout(None)     # idle between requests: block until rank 0 speaks again
    worker = tp_engine.Rank1Worker(backend, channel)

    def stop(signum, frame):
        print(f"tp3: rank {rank} received signal {signum}, exiting", flush=True)
        worker.drop_state()
        channel.close()
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        worker.serve_forever()
    except tp_engine.ChannelClosed:
        print(f"tp3: rank {rank} done (rank 0 closed the control channel)", flush=True)
        return 0
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        print(f"tp3: rank {rank} failed; restart ALL ranks (rank 0 cannot resynchronize)",
              file=sys.stderr, flush=True)
        return 1
    try:
        backend.unload()
        print(f"tp3: rank {rank} unloaded cleanly", flush=True)
    except Exception:                                            # noqa: BLE001
        pass
    print(f"tp3: rank {rank} served {worker.requests_served} requests", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
