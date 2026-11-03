"""tp3_channel: one rank-0 control endpoint fanned out over N-1 workers.

The 2-node stack (tp_serve.py + tp_rank1.py) is built around ONE ControlChannel
between rank 0 and rank 1. TPEngine only ever uses send / recv / request /
request_with_blobs / set_recv_timeout / close on it, so a TP=3 engine needs no
changes there: this module provides an object with the same surface that

  - accepts N-1 worker connections on one listening socket,
  - sends every frame to EVERY worker before receiving any reply, and
  - verifies the workers' replies agree (per-worker fields aside, see _PER_WORKER).

The send-all-then-recv-all ordering is not an optimization choice: a worker
reply arrives only after its model call finishes, and that call contains RCCL
collectives in which every rank participates. Sending to worker 2 only after
worker 1 replied would deadlock both workers inside their first collective.

Not used for model tensor payloads; see ControlChannel for the frame format.
"""

from __future__ import annotations

import socket
import threading

import tp_engine


class WorkerDiverged(tp_engine.ProtocolError):
    """Workers replied differently to the same frame: a lockstep violation."""


# Reply fields that legitimately differ between workers; everything else (the op, session
# and request ids, counts) is lockstep state and must match exactly.
#   rank             each worker's own id
#   bytes            spill_ok: the size of THIS worker's disk image. The image trailer holds a
#                    wall-clock timestamp whose JSON text varies in length, so two workers'
#                    images of one session often differ by a byte. Comparing it turned good
#                    spills into "divergences" (2026-11-02: 839100 vs 839101 bytes): rank 0
#                    evicted a session both workers had already spilled, both died on "evict
#                    of unknown session", and rank 0 sat in its next collective until the
#                    10-minute NCCL watchdog took the whole stack down.
#   reason, message  the free-text detail of spill_error / load_error / error replies (paths,
#                    errno text); the op already says what happened
_PER_WORKER = frozenset({"rank", "bytes", "reason", "message"})


def _comparable(reply: dict) -> dict:
    """Reply content minus the per-worker fields (_PER_WORKER)."""
    return {k: v for k, v in reply.items() if k not in _PER_WORKER}


def _agree(replies: list[dict]) -> dict:
    """The first worker's reply, once every other worker's reply says the same."""
    first = replies[0]
    want = _comparable(first)
    for i, rep in enumerate(replies[1:], start=1):
        if _comparable(rep) != want:
            raise WorkerDiverged(f"worker replies diverged: worker1={first} worker{i + 1}={rep}")
    return first


class MultiControlChannel:
    """Same call surface as ControlChannel, fanned out over every worker."""

    def __init__(self, channels: list[tp_engine.ControlChannel]):
        self.channels = list(channels)
        self.lock = threading.RLock()
        self.peer = ",".join(getattr(ch, "peer", "?") for ch in self.channels)

    @classmethod
    def listen_many(cls, host: str, port: int, count: int, *, accept_timeout: float | None = None,
                    timeout: float = 600.0, log=None) -> "MultiControlChannel":
        """Rank-0 side: bind once and block until `count` workers connected."""
        log = log or print
        if accept_timeout is None:
            accept_timeout = tp_engine.env_float("EXL3_TP_ACCEPT_TIMEOUT", 1800.0)
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(accept_timeout)
        server.bind((host, port))
        server.listen(count)
        channels = []
        for i in range(count):
            log(f"tp3: rank 0 waiting for worker {i + 1}/{count} on {host}:{port}")
            conn, addr = server.accept()
            conn.settimeout(timeout)
            log(f"tp3: worker connected from {addr[0]}:{addr[1]}")
            channels.append(tp_engine.ControlChannel(conn, timeout=timeout, log=log))
        server.close()
        return cls(channels)

    # ---- ControlChannel-compatible surface -----------------------------------

    def set_recv_timeout(self, seconds: float | None) -> None:
        for ch in self.channels:
            ch.set_recv_timeout(seconds)

    def send(self, msg: dict) -> None:
        with self.lock:
            for ch in self.channels:
                ch.send(msg)

    def send_blob(self, data: bytes) -> None:
        with self.lock:
            for ch in self.channels:
                ch.send_blob(data)

    def send_with_blobs(self, msg: dict, blobs: list[bytes]) -> None:
        with self.lock:
            for ch in self.channels:
                ch.send_with_blobs(msg, blobs)

    def recv(self) -> dict:
        return _agree([ch.recv() for ch in self.channels])

    def recv_blob(self) -> bytes:
        # No current op ships blobs worker -> rank 0; fail loudly if one appears.
        raise NotImplementedError("tp3: worker->rank0 blobs are not part of the protocol")

    def request(self, msg: dict, timeout: float = 30.0) -> dict:
        return self._exchange(lambda ch: ch.send(msg), timeout)

    def request_with_blobs(self, msg: dict, blobs: list[bytes], timeout: float = 120.0) -> dict:
        return self._exchange(lambda ch: ch.send_with_blobs(msg, blobs), timeout)

    def _exchange(self, send, timeout: float) -> dict:
        """Send one frame to every worker, then read every reply under `timeout`. Every
        socket gets its own previous timeout back afterwards, success or not (as
        ControlChannel.request does), so a later recv -- the drain of a late disk reply --
        runs under the channel's normal timeout rather than this call's."""
        with self.lock:
            olds = [ch.conn.gettimeout() for ch in self.channels]
            try:
                for ch in self.channels:
                    ch.conn.settimeout(timeout)
                    send(ch)
                replies = [ch.recv() for ch in self.channels]
            finally:
                for ch, old in zip(self.channels, olds):
                    ch.conn.settimeout(old)
        return _agree(replies)

    def close(self) -> None:
        for ch in self.channels:
            ch.close()
