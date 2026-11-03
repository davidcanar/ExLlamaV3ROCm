"""TP=3 control fan-out (tp3_channel.MultiControlChannel) over loopback.

The 2-node suites drive rank 0's TPEngine against ONE real Rank1Worker; tp3_serve.py runs
the same engine against TWO through MultiControlChannel, which checks that the workers'
replies agree. These tests run that shape -- the real engine, two real workers, the real
fan-out -- plus the channel's reply comparison and its socket timeouts. CPU only.
"""
from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp3_channel  # noqa: E402
import tp_disk  # noqa: E402
import tp_engine  # noqa: E402
from test_tp_control import free_port  # noqa: E402
from test_tp_disk import DiskBackend, DiskHarness, DiskSpecBackend, _spill_scenario  # noqa: E402
from test_tp_prefix_cache import MIRRORED, Tok, ids_of, prompt  # noqa: E402
from test_tp_sessions import PAGE, VOCAB  # noqa: E402


class Tp3DiskHarness:
    """Rank-0 TPEngine + two real Rank1Workers behind a MultiControlChannel, every rank
    with its own disk-tier directory: the shape tp3_serve.py runs on Box1 + Box2 + Box3."""

    def __init__(self, *, spec: bool = False, sessions: int = 2, pool_pages: int = 12,
                 disk_root: Path, workers: int = 2):
        cls = DiskSpecBackend if spec else DiskBackend
        self.be0 = cls(pool_pages, PAGE)
        self.bes = [cls(pool_pages, PAGE) for _ in range(workers)]
        self.errors: list[BaseException] = []
        self.workers: list = [None] * workers
        self.disk_dirs = [disk_root / f"rank{r}" for r in range(workers + 1)]
        for d in self.disk_dirs:
            d.mkdir(parents=True, exist_ok=True)
        port = free_port()

        def worker(i):
            try:
                chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=30.0)
                chan.set_recv_timeout(None)
                self.workers[i] = tp_engine.Rank1Worker(
                    self.bes[i], chan, sessions=sessions,
                    disk_opts={"root": str(self.disk_dirs[i + 1]), "cap_gib": None,
                               "enabled": True},
                    log=lambda *_: None)
                self.workers[i].serve_forever()
            except BaseException as exc:                                  # noqa: BLE001
                self.errors.append(exc)

        self.threads = [threading.Thread(target=worker, args=(i,), name=f"tp3-worker{i + 1}",
                                         daemon=True) for i in range(workers)]
        for t in self.threads:
            t.start()
        self.chan = tp3_channel.MultiControlChannel.listen_many(
            "127.0.0.1", port, workers, accept_timeout=30.0, timeout=30.0, log=lambda *_: None)
        self.chan.send({"op": "hello", "proto": tp_engine.PROTO_SESSIONS, "vocab_size": VOCAB,
                        "dspark": spec, "sessions": sessions, "pool_pages": pool_pages,
                        "page_size": PAGE})
        for ch in self.chan.channels:                 # per worker, as tp3_serve.py does
            ok = ch.recv()
            assert ok.get("op") == "hello_ok" and ok.get("sessions") == sessions, ok
        self.engine = tp_engine.TPEngine(
            self.be0, self.chan, Tok(), ctx=pool_pages * PAGE - 2, eos_id=None,
            sessions=sessions, session_opts={"prefix_slots": 2, "min_tokens": 8},
            disk_opts={"root": str(self.disk_dirs[0]), "cap_gib": None, "enabled": True},
            log=lambda *_: None)
        assert self.engine.tier is not None and self.engine.tier.enabled

    def run(self, ids, max_tokens=4):
        stats: dict = {}

        async def go():
            out = []
            async for piece in self.engine.generate(
                    "", prompt_ids=list(ids), max_tokens=max_tokens, temperature=0.0,
                    top_p=1.0, stop=[], stats=stats):
                out.append(piece)
            return "".join(out)
        text = asyncio.run(go())
        self.sync()
        return text, stats

    @staticmethod
    def mirrored(be):
        return [c for c in be.calls if c[0] in MIRRORED]

    def sync(self, timeout=5.0):
        deadline = time.time() + timeout
        n0 = len(self.mirrored(self.be0))
        while time.time() < deadline:
            if all(be.released >= self.be0.released and len(self.mirrored(be)) == n0
                   for be in self.bes):
                return
            time.sleep(0.005)
        raise AssertionError(f"a worker did not catch up: {self.errors}")

    def check(self):
        """Both workers mirror rank 0's model calls and hold exactly its hot sessions,
        page for page and checkpoint for checkpoint."""
        mgr = self.engine.mgr
        for w, be in zip(self.workers, self.bes):
            assert self.mirrored(be) == self.mirrored(self.be0), "a worker's model calls diverged"
            assert set(mgr.sessions) == set(w.sess)
            for sid, s in mgr.sessions.items():
                assert s.pages == w.sess[sid]["pages"], (sid, s.pages, w.sess[sid]["pages"])
                assert sorted(c.key for c in s.px.ckpts) == sorted(w.sess[sid]["stashes"])
        assert not self.errors, self.errors

    def status(self):
        return self.engine.session_status()

    def stop(self):
        self.engine.close()
        for t in self.threads:
            t.join(timeout=10.0)
        assert not any(t.is_alive() for t in self.threads) and not self.errors, self.errors


class _SkewedClock:
    """tp_disk's clock, one character longer in worker 2's thread: its image trailers come
    out one byte longer than worker 1's -- how Box2 and Box3 wrote 839100- and 839101-byte
    images of the same session in production."""

    @staticmethod
    def time():
        return 1793635883.25 if threading.current_thread().name == "tp3-worker2" \
            else 1793635883.5


@pytest.mark.parametrize("spec", [False, True], ids=["plain", "dspark"])
def test_spill_with_unequal_worker_image_sizes_is_honored(tmp_path, monkeypatch, spec):
    """The two workers' images of one session differ by a byte; the spill still counts as
    agreed: the session goes cold on all three ranks, reloads on its return, and every
    reply matches the 2-node stack's. (It used to be a WorkerDiverged: rank 0 evicted a
    session both workers had already spilled, and the workers died.)"""
    monkeypatch.setattr(tp_disk, "time", _SkewedClock)
    a, b, c = prompt(1, 40), prompt(2, 40), prompt(3, 40)
    h3 = Tp3DiskHarness(spec=spec, disk_root=tmp_path / "tp3")
    h2 = DiskHarness(spec=spec, sessions=2, pool_pages=12, disk_root=tmp_path / "tp2")
    try:
        got3 = _spill_scenario(h3, a=a, b=b, c=c)          # c spills the LRU (a)
        h3.check()
        sizes = [(d / "rank1-i1.img").stat().st_size for d in h3.disk_dirs[1:]]
        assert sizes[1] - sizes[0] == 1, sizes           # the case that used to "diverge"
        assert h3.engine.tier.counters["spills"] == 1
        assert h3.engine.tier.counters["spill_errors"] == 0
        assert [cs["session"] for cs in h3.status()["cold_sessions"]] == [1]
        got2 = _spill_scenario(h2, a=a, b=b, c=c)
        got3.append(h3.run(a + ids_of(got3[0][0]) + prompt(4, 12)))
        got2.append(h2.run(a + ids_of(got2[0][0]) + prompt(4, 12), max_tokens=4))
        h3.check()
        assert h3.engine.mgr.counters["cold_loads"] == 1
        assert got3[3][1]["cached_tokens"] == got2[3][1]["cached_tokens"] > 0   # reloaded
        assert [t for t, _ in got3] == [t for t, _ in got2]
        text, _ = h3.run(prompt(5, 30))                   # and the stack keeps serving
        h3.check()
    finally:
        h3.stop()
        h2.stop()


class _Canned:
    """A ControlChannel stand-in whose every recv returns one canned reply."""
    peer = "canned"

    def __init__(self, reply: dict):
        self.reply = reply

    def recv(self) -> dict:
        return dict(self.reply)


def _pair(a: dict, b: dict) -> tp3_channel.MultiControlChannel:
    return tp3_channel.MultiControlChannel([_Canned(a), _Canned(b)])


def test_reply_comparison_ignores_only_per_worker_fields():
    ok = {"op": "spill_ok", "session": 1, "req": 227}
    assert _pair({**ok, "bytes": 839100}, {**ok, "bytes": 839101}).recv() == {**ok, "bytes": 839100}
    err = {"op": "load_error", "session": 2, "req": 9}
    assert _pair({**err, "reason": "box2: EIO"}, {**err, "reason": "box3: EIO"}).recv()["op"] \
        == "load_error"
    assert _pair({"op": "hello_ok", "rank": 1}, {"op": "hello_ok", "rank": 2}).recv()["op"] \
        == "hello_ok"
    for a, b in (({**ok, "bytes": 1}, {"op": "spill_error", "session": 1, "req": 227,
                                       "reason": "disk full"}),          # mixed outcome
                 ({**ok, "bytes": 1}, {**ok, "session": 2, "bytes": 1}),  # other session
                 ({**ok, "bytes": 1}, {**ok, "req": 228, "bytes": 1}),    # other request
                 ({"op": "imgmeta_ok", "req": 5, "images": 2},
                  {"op": "imgmeta_ok", "req": 5, "images": 3})):
        with pytest.raises(tp3_channel.WorkerDiverged):
            _pair(a, b).recv()


def test_request_restores_every_socket_timeout():
    """request() puts its short timeout on every worker socket and gives each socket its
    own previous timeout back afterwards, on success and on a timeout alike (as
    ControlChannel.request does; it used to leave the short one in place)."""
    port = free_port()
    errors: list[BaseException] = []
    late = threading.Event()

    def worker():
        try:
            ch = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=30.0)
            ch.set_recv_timeout(None)
            idx = ch.recv()["i"]
            while True:
                msg = ch.recv()
                if idx == 1 and msg["req"] == 2:
                    late.wait(5.0)                # answer only after rank 0 gave up
                ch.send({"op": "pong", "req": msg["req"]})
        except (tp_engine.ChannelClosed, ConnectionError):
            pass
        except BaseException as exc:                                      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    multi = tp3_channel.MultiControlChannel.listen_many(
        "127.0.0.1", port, 2, accept_timeout=30.0, timeout=77.0, log=lambda *_: None)
    try:
        for i, ch in enumerate(multi.channels):
            ch.send({"op": "id", "i": i})
        multi.channels[1].conn.settimeout(88.0)
        assert multi.request({"op": "ping", "req": 1}, timeout=5.0) == {"op": "pong", "req": 1}
        assert [ch.conn.gettimeout() for ch in multi.channels] == [77.0, 88.0]
        with pytest.raises((socket.timeout, TimeoutError)):
            multi.request({"op": "ping", "req": 2}, timeout=0.2)
        assert [ch.conn.gettimeout() for ch in multi.channels] == [77.0, 88.0]
        late.set()
        assert multi.channels[1].recv() == {"op": "pong", "req": 2}   # the late one
    finally:
        multi.close()
        for t in threads:
            t.join(timeout=5.0)
    assert not errors, errors
