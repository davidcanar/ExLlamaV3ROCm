#!/usr/bin/env python3
"""Rank-1 worker of the two-node DS4.1 TP server (run on Box2 via tools/ds41/tp_rank1.sh).

Loads this box's half of the weights exactly like scripts/tp_ds41_generate.py does, connects to
rank 0's control channel, then mirrors every model call rank 0 makes, for the server's lifetime.
The HTTP layer lives on rank 0 (tp_serve.py); this process uses no sockets other than the control
channel and the NCCL/Odinlink transport. Exits 0 when rank 0 shuts down or closes the channel,
non-zero on any failure (both ranks must then be restarted -- see README.md).
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
    world = tp_engine.env_int("EXL3_TP_WORLD_SIZE", 2)
    if rank != 1:
        print("tp_rank1: EXL3_TP_NODE_RANK must be 1 here; rank 0 runs tp_serve.py", file=sys.stderr, flush=True)
        return 2
    control_port = tp_engine.env_int("EXL3_TP_CONTROL_PORT", 29651)
    master_addr = os.environ.get("EXLLAMA_MASTER_ADDR", "10.0.2.1")

    print(f"tp: rank 1 starting DS4.1 cached TP load (world size {world})", flush=True)
    backend = tp_engine.TPModelBackend.load(rank=rank, world_size=world, progress=False)
    channel = tp_engine.ControlChannel.connect(
        master_addr, control_port,
        deadline_s=tp_engine.env_float("EXL3_TP_CONNECT_TIMEOUT", 1800.0))
    channel.set_recv_timeout(None)     # idle between requests: block until rank 0 speaks again
    worker = tp_engine.Rank1Worker(backend, channel)

    def stop(signum, frame):
        print(f"tp: rank 1 received signal {signum}, exiting", flush=True)
        worker.drop_state()
        channel.close()
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        worker.serve_forever()
    except tp_engine.ChannelClosed:
        print("tp: rank 1 done (rank 0 closed the control channel)", flush=True)
        return 0
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        print("tp: rank 1 failed; restart BOTH ranks (rank 0 cannot resynchronize)", file=sys.stderr, flush=True)
        return 1
    try:
        backend.unload()
        print("tp: rank 1 unloaded cleanly", flush=True)
    except Exception:                                            # noqa: BLE001
        pass
    print(f"tp: rank 1 served {worker.requests_served} requests", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
