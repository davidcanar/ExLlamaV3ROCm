"""tp3_channel: one rank-0 control endpoint fanned out over N-1 workers.

The 2-node stack (tp_serve.py + tp_rank1.py) is built around ONE ControlChannel
between rank 0 and rank 1. TPEngine only ever uses send / recv / request /
request_with_blobs / set_recv_timeout / close on it, so a TP=3 engine needs no
changes there: this module provides an object with the same surface that

  - accepts N-1 worker connections on one listening socket,
  - sends every frame to EVERY worker before receiving any reply, and
  - verifies the workers' replies are identical (modulo their rank field).

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


def _comparable(reply: dict) -> dict:
    """Reply content minus rank-identifying fields (workers legitimately differ there)."""
    return {k: v for k, v in reply.items() if k != "rank"}


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
        replies = [ch.recv() for ch in self.channels]
        first = replies[0]
        want = _comparable(first)
        for i, rep in enumerate(replies[1:], start=1):
            if _comparable(rep) != want:
                raise WorkerDiverged(
                    f"worker replies diverged: worker1={first} worker{i + 1}={rep}")
        return first

    def recv_blob(self) -> bytes:
        # No current op ships blobs worker -> rank 0; fail loudly if one appears.
        raise NotImplementedError("tp3: worker->rank0 blobs are not part of the protocol")

    def request(self, msg: dict, timeout: float = 30.0) -> dict:
        with self.lock:
            for ch in self.channels:
                old = ch.conn.gettimeout()
                ch.conn.settimeout(timeout)
                ch.send(msg)
            replies = []
            for ch in self.channels:
                try:
                    replies.append(ch.recv())
                finally:
                    ch.conn.settimeout(ch.conn.gettimeout())
        first = replies[0]
        want = _comparable(first)
        for i, rep in enumerate(replies[1:], start=1):
            if _comparable(rep) != want:
                raise WorkerDiverged(
                    f"worker replies diverged: worker1={first} worker{i + 1}={rep}")
        return first

    def request_with_blobs(self, msg: dict, blobs: list[bytes], timeout: float = 120.0) -> dict:
        with self.lock:
            for ch in self.channels:
                ch.conn.settimeout(timeout)
                ch.send_with_blobs(msg, blobs)
            replies = [ch.recv() for ch in self.channels]
            for ch in self.channels:
                ch.conn.settimeout(self.channels[0].timeout)
        first = replies[0]
        want = _comparable(first)
        for i, rep in enumerate(replies[1:], start=1):
            if _comparable(rep) != want:
                raise WorkerDiverged(
                    f"worker replies diverged: worker1={first} worker{i + 1}={rep}")
        return first

    def close(self) -> None:
        for ch in self.channels:
            ch.close()
