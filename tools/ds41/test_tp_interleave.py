"""Interleaved requests (proto 6, EXL3_TP_CONCURRENCY >= 2) over a loopback control channel.

The stand-ins keep EVERY piece of per-request state inside the state object -- its slot, hash
chain, page mapping, restore marker -- and the spec flavor keeps one drafter ring per slot,
selected by bind(), as the real backend does. A request that reads another request's rows,
restores onto another session, drafts from another request's ring or primes a ring nobody bound
therefore changes the output or trips an assertion. Every scenario is checked against the same
requests run one at a time (proto 5, the serialized engine): the replies must be identical, the
rank-0 and rank-1 model call sequences must be identical (request by request), and the pool
invariants of test_tp_sessions must hold. CPU only; no GPU, no model files.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_engine  # noqa: E402
from test_tp_control import free_port  # noqa: E402
from test_tp_prefix_cache import Tok, ids_of, prompt  # noqa: E402
from test_tp_sessions import PAGE, VOCAB, SessionRanks, _RecordingChannel  # noqa: E402

MIRRORED = ("new_state", "restore", "prefill", "forward", "verify", "rewind", "stash", "drop_stash",
            "release_state")


class IlvBackend:
    """Plain interleave-safe stand-in over a paged pool (see the module docstring). The model is
    test_tp_sessions.SessionBackend's: the greedy token after position p hashes the request's own
    chain AND every row below p read through ITS mapping, so replies are a pure function of the
    request's token history -- the serialized engine's replies are the reference."""

    dspark = False
    vocab_size = VOCAB
    span = 64

    def __init__(self, pool_pages: int = 64, page_size: int = PAGE, slots: int = 4, delay: float = 0.0):
        self.pool_pages = int(pool_pages)
        self.page_size = int(page_size)
        self.cache_tokens = self.pool_pages * self.page_size
        self.slots = int(slots)
        self.delay = float(delay)                     # per model call on rank 0 (overlap timing)
        self._free = deque(range(self.slots))
        self.rows: dict[int, dict[int, int]] = {}     # page -> {offset: token}
        self.owner: dict[int, int] = {}               # page -> sid
        self.calls: list[tuple] = []
        self.live: dict[int, dict] = {}               # live checkpoints (id(stash) -> stash)
        self.states = self.released = 0
        self.max_live = 0
        self.bound = None

    # -- state and checkpoints
    def _state(self, pos: int, hist: list, restored=None) -> dict:
        assert self._free, "more live requests than the backend's slots"
        self.states += 1
        st = {"slot": self._free.popleft(), "pos": int(pos), "hist": hist, "alloc": None,
              "restored": restored, "serial": self.states}
        self.max_live = max(self.max_live, self.slots - len(self._free))
        return st

    def new_state(self):
        st = self._state(0, [0])
        self.calls.append(("new_state", st["serial"]))
        return st

    def restore(self, stash, position):
        assert self.live.get(id(stash)) is stash, "restore of a dropped checkpoint"
        assert stash["pos"] == position
        st = self._state(position, list(stash["hist"]), restored=stash)
        self.calls.append(("restore", st["serial"], int(position)))
        return st

    def stash(self, state):
        self.calls.append(("stash", state["serial"], state["pos"]))
        alloc = state["alloc"]
        s = {"pos": state["pos"], "hist": list(state["hist"][: state["pos"] + 1]),
             "sid": alloc.sid if alloc is not None else None,
             "pages": list(alloc.pages[: -(-state["pos"] // self.page_size)]) if alloc is not None else []}
        self.live[id(s)] = s
        return s

    def drop_stash(self, stash):
        self.calls.append(("drop_stash", stash["pos"]))
        assert self.live.pop(id(stash), None) is stash, "drop of an unknown checkpoint"

    def release_state(self, state):
        self.released += 1
        self.calls.append(("release_state", state["serial"]))
        self._free.appendleft(state["slot"])          # (the real Cache's free-list order)

    def bind(self, state):
        self.bound = state

    # -- the model
    def _claim(self, state, alloc):
        assert alloc is not None, "interleaved requests always carry their page mapping"
        state["alloc"] = alloc
        for p in alloc.pages:
            self.owner[int(p)] = alloc.sid
        if state["restored"] is not None:
            was, state["restored"] = state["restored"], None
            if was["sid"] is not None and was["sid"] != alloc.sid:
                raise AssertionError(f"restore onto another session's rows: checkpoint of session "
                                     f"{was['sid']} continued in session {alloc.sid}")
            if was["pages"] and list(alloc.pages[: len(was["pages"])]) != was["pages"]:
                raise AssertionError("restored prefix pages changed under the session")

    def _row_at(self, alloc, i: int) -> int:
        page = int(alloc.pages[i // self.page_size])
        if self.owner.get(page) != alloc.sid:
            raise AssertionError(f"read of page {page} (position {i}) owned by {self.owner.get(page)}, "
                                 f"not session {alloc.sid}")
        return self.rows[page][i % self.page_size]

    def _write(self, alloc, i: int, token: int) -> None:
        page = int(alloc.pages[i // self.page_size])
        if self.owner.get(page) != alloc.sid:
            raise AssertionError(f"write of page {page} (position {i}) owned by {self.owner.get(page)}, "
                                 f"not session {alloc.sid}")
        self.rows.setdefault(page, {})[i % self.page_size] = int(token)

    def _feed(self, state, token):
        p = state["pos"]
        self._write(state["alloc"], p, token)
        del state["hist"][p + 1:]
        state["hist"].append((state["hist"][p] * 1000003 + int(token) + 1) % (1 << 61))
        state["pos"] = p + 1

    def _next(self, state) -> int:
        p = state["pos"]
        acc = state["hist"][p]
        for i in range(p):
            acc = (acc * 31 + self._row_at(state["alloc"], i)) % (1 << 61)
        return 32 + acc % 95

    def _row(self, tok):
        row = torch.zeros(self.vocab_size)
        row[tok] = 8.0
        return row

    def _wait(self):
        if self.delay:
            time.sleep(self.delay)

    def prefill(self, ids, state, spec=False, start=0, embeddings=None, alloc=None):
        self.calls.append(("prefill", state["serial"], len(ids), bool(spec), int(start)))
        self._claim(state, alloc)
        assert state["pos"] == start, (state["pos"], start)
        assert len(ids) - 1 - start <= self.span, "a prefill call wider than the backend's span"
        for t in ids[start: len(ids) - 1]:
            self._feed(state, t)
        self._wait()
        return {}

    def forward(self, token, position, state, embeddings=None, alloc=None):
        self.calls.append(("forward", state["serial"], int(token), int(position)))
        self._claim(state, alloc)
        assert state["pos"] == position
        self._feed(state, token)
        self._wait()
        return self._row(self._next(state))


class IlvSpecBackend(IlvBackend):
    """Spec flavor: verify writes every window row, and the drafter keeps one position-indexed
    ring PER SLOT, chosen by bind() -- the real backend's per-slot drafter rings. A draft or a
    refresh while another request's ring is bound, or a spec prefill that primes an unbound
    ring, is a bug the stale-draft counter and the asserts catch."""

    dspark = True
    dspark_block = 5
    export_layers = frozenset({36, 37, 38})
    draft_window_rows = 8
    RING = 40

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rings: dict[int, dict[int, tuple]] = {}
        self.rounds = 0
        self.stale_drafts = 0

    def _ring(self, st):
        return self.rings.setdefault(st["slot"], {})

    def _dwrite(self, st, p):
        self._ring(st)[p % self.RING] = (p, st["hist"][p + 1])

    def _window(self, position):
        return range(max(0, position - self.draft_window_rows), position)

    def prefill(self, ids, state, spec=False, start=0, embeddings=None, alloc=None):
        out = super().prefill(ids, state, spec, start, embeddings, alloc)
        if spec:
            assert self.bound is state, "spec prefill primed the drafter ring of another request"
            for p in range(start, len(ids) - 1):
                self._dwrite(state, p)
        return out

    def draft(self, anchor, position, k_cap):
        st = self.bound
        assert st is not None and st["pos"] == position, "draft for a request that is not bound"
        self.calls.append(("draft", st["serial"], int(anchor), int(position), int(k_cap)))
        ring = self._ring(st)
        if any(ring.get(q % self.RING) != (q, st["hist"][q + 1]) for q in self._window(position)):
            self.stale_drafts += 1
        k = max(0, min(self.dspark_block, int(k_cap)))
        good = self.rounds % (k + 1) if k else 0
        self.rounds += 1
        alloc = st["alloc"]
        saved = {i: self.rows.get(int(alloc.pages[i // self.page_size]), {}).get(i % self.page_size)
                 for i in range(position, position + k + 1)}
        sim = {"pos": position, "hist": list(st["hist"][: position + 1]), "alloc": alloc}
        drafts, tok = [], int(anchor)
        for i in range(k):
            self._feed(sim, tok)
            tok = self._next(sim)
            drafts.append(tok if i < good else 32 + (tok - 31) % 95)
            tok = drafts[-1]
        for i, v in saved.items():                        # drafting leaves the trunk rows alone
            if v is not None:
                page = int(alloc.pages[i // self.page_size])
                self.rows.setdefault(page, {})[i % self.page_size] = v
        return k, drafts

    def verify(self, window, position, state, alloc=None):
        self.calls.append(("verify", state["serial"], tuple(int(t) for t in window), int(position)))
        self._claim(state, alloc)
        assert state["pos"] == position
        rows = []
        for t in window:
            self._feed(state, t)
            rows.append(self._row(self._next(state)))
        self._wait()
        return torch.stack(rows), {"export_states": ["taps"]}

    def rewind(self, state, steps):
        self.calls.append(("rewind", state["serial"], int(steps)))
        state["pos"] -= int(steps)
        del state["hist"][state["pos"] + 1:]

    def draft_prefill_kv(self, taps, position=0):
        raise AssertionError("the stand-in primes inside prefill, as the real backend does")

    def refresh_draft_kv(self, taps, accepted, position):
        st = self.bound
        assert st is not None and st["pos"] == position, "refresh for a request that is not bound"
        self.calls.append(("refresh", st["serial"], int(accepted), int(position)))
        for p in range(position - accepted, position):
            self._dwrite(st, p)

    def draft_window(self, position):
        st = self.bound
        if not self._window(position):
            return None
        self.calls.append(("draft_window", st["serial"], int(position)))
        ring = self._ring(st)
        return [ring.get(q % self.RING) for q in self._window(position)]

    def set_draft_window(self, saved, position):
        st = self.bound
        self.calls.append(("set_draft_window", st["serial"], int(position)))
        ring = self._ring(st)
        for q, row in zip(self._window(position), saved):
            if row is None:
                ring.pop(q % self.RING, None)
            else:
                ring[q % self.RING] = row


def _hello(chan, *, sessions, pool_pages, page_size=PAGE, dspark=False, concurrency=2, proto=None):
    p = tp_engine.PROTO_INTERLEAVE if proto is None else proto
    hello = {"op": "hello", "proto": p, "vocab_size": VOCAB, "dspark": dspark,
             "sessions": sessions, "pool_pages": pool_pages, "page_size": page_size}
    if concurrency is not None:
        hello["concurrency"] = concurrency
    chan.send(hello)
    return chan.recv()


class IlvRanks:
    """Rank-0 engine at concurrency C + the real Rank1Worker (proto 6) over a loopback channel."""

    def __init__(self, *, spec: bool = False, sessions: int = 4, concurrency: int = 2,
                 pool_pages: int = 64, page_size: int = PAGE, session_opts: dict | None = None,
                 span: int | None = None, chunk: int | None = None, delay: float = 0.002,
                 ctx: int | None = None, slots: int | None = None, backend_cls=None,
                 tokenizer=None, image_skip: bool = True):
        self.fatal: list[BaseException] = []
        cls = backend_cls or (IlvSpecBackend if spec else IlvBackend)
        slots = concurrency if slots is None else slots
        self.be0 = cls(pool_pages, page_size, slots=slots, delay=delay)
        self.be1 = cls(pool_pages, page_size, slots=slots)
        if span is not None:
            self.be0.span = self.be1.span = span
        self.errors: list[BaseException] = []
        self.frames: list[dict] = []
        self.done_order: list[str] = []
        port = free_port()

        def rank1():
            try:
                chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=30.0)
                chan.set_recv_timeout(None)
                self.worker = tp_engine.Rank1Worker(self.be1, chan, sessions=sessions,
                                                    concurrency=concurrency, log=lambda *_: None)
                self.worker.serve_forever()
            except BaseException as exc:                                  # noqa: BLE001
                self.errors.append(exc)

        self.thread = threading.Thread(target=rank1, daemon=True)
        self.thread.start()
        chan = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=30.0)
        want = tp_engine.PROTO_INTERLEAVE if concurrency > 1 else tp_engine.PROTO_SESSIONS
        ok = _hello(chan, sessions=sessions, pool_pages=pool_pages, page_size=page_size,
                    dspark=spec, concurrency=concurrency if concurrency > 1 else None, proto=want)
        assert ok.get("op") == "hello_ok" and ok.get("proto") == want, ok
        assert concurrency == 1 or ok.get("concurrency") == concurrency, ok
        self.chan = _RecordingChannel(chan, self.frames)
        opts = {"prefix_slots": 2, "min_tokens": 8}
        opts.update(session_opts or {})
        self.hello_ok = ok
        self.engine = tp_engine.TPEngine(
            self.be0, self.chan, tokenizer if tokenizer is not None else Tok(),
            ctx=pool_pages * page_size - 2 if ctx is None else ctx,
            eos_id=None, sessions=sessions, session_opts=opts, concurrency=concurrency,
            log=lambda *_: None, on_fatal=self.fatal.append, image_skip=image_skip)
        assert self.engine.proto == want
        if chunk is not None:
            self.engine.interleave_chunk = chunk

    async def one(self, name, ids, max_tokens=12, *, after: asyncio.Event | None = None,
                  started: asyncio.Event | None = None, started_after: int = 1, **kw):
        if after is not None:
            await after.wait()
        stats: dict = {}
        out = []
        async for piece in self.engine.generate("", prompt_ids=list(ids), max_tokens=max_tokens,
                                                temperature=0.0, top_p=1.0, stop=kw.pop("stop", []),
                                                stats=stats, **kw):
            out.append(piece)
            if started is not None and len(out) >= started_after:
                started.set()
        self.done_order.append(name)
        return "".join(out), stats

    def run(self, ids, max_tokens=12, **kw):
        res = asyncio.run(self.one("solo", ids, max_tokens, **kw))
        self.sync()
        return res

    def gather(self, *coros):
        async def go():
            return await asyncio.gather(*coros)
        res = asyncio.run(go())
        self.sync()
        return res

    def sync(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.be1.released >= self.be0.released \
                    and len(self.mirrored(self.be1)) == len(self.mirrored(self.be0)):
                return
            time.sleep(0.005)
        raise AssertionError(f"rank 1 did not catch up: {self.errors}")

    @staticmethod
    def mirrored(be):
        return [c for c in be.calls if c[0] in MIRRORED]

    def check(self):
        """Rank 0 and rank 1 ran the same model calls for the same requests in the same order,
        and the session/pool records agree page for page (test_tp_sessions.SessionRanks.check)."""
        assert self.mirrored(self.be0) == self.mirrored(self.be1), "rank-0/rank-1 model calls diverged"
        mgr = self.engine.mgr
        assert set(mgr.sessions) == set(self.worker.sess), (sorted(mgr.sessions), sorted(self.worker.sess))
        for sid, s in mgr.sessions.items():
            rec = self.worker.sess[sid]
            assert s.pages == rec["pages"], (sid, s.pages, rec["pages"])
            assert sorted(c.key for c in s.px.ckpts) == sorted(rec["stashes"]), (sid, "checkpoint sets differ")
            assert not s.active, f"session {sid} still active after every request ended"
        claimed = [p for rec in self.worker.sess.values() for p in rec["pages"]]
        assert len(claimed) == len(set(claimed)), "a page is claimed by two sessions on rank 1"
        assert len(claimed) == mgr.pool.used, (len(claimed), mgr.pool.used)
        n_ck = sum(len(s.px.ckpts) for s in mgr.sessions.values())
        assert len(self.be0.live) == len(self.be1.live) == n_ck
        assert not self.worker.jobs and not self.engine._active and not self.engine._pending
        assert self.be0.released == self.be0.states and self.be1.released == self.be1.states
        assert not self.errors, self.errors

    def ops(self, op):
        return [f for f in self.frames if f.get("op") == op]

    def stop(self):
        self.engine.close()
        self.thread.join(timeout=10.0)
        assert not self.thread.is_alive() and not self.errors, self.errors


def serial_replies(reqs, *, spec=False, sessions=4, pool_pages=64):
    """The same requests through the serialized engine (proto 5), one after the other."""
    ref = SessionRanks(spec=spec, sessions=sessions, pool_pages=pool_pages)
    try:
        out = [ref.run(ids, max_tokens=n)[0] for ids, n in reqs]
        ref.check()
    finally:
        ref.stop()
    return out


def interleaved(frames, a_req, b_req):
    """True if a model-call frame of b_req lies between two of a_req's (the requests alternated)."""
    seq = [f.get("req") for f in frames if f.get("op") in ("step", "round", "chunk")]
    first, last = seq.index(a_req), len(seq) - 1 - seq[::-1].index(a_req)
    return b_req in seq[first:last]


# ---- the scenarios ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [False, True])
def test_two_conversations_interleave_and_reply_exactly_as_serialized(spec):
    """Two unrelated conversations started together run step for step (their frames alternate,
    neither waits for the other to finish) and each gets exactly the reply it gets alone."""
    a, b = prompt(301, 40), prompt(302, 36)
    want = serial_replies([(a, 24), (b, 24)], spec=spec)
    r = IlvRanks(spec=spec, concurrency=2)
    try:
        (ta, sa), (tb, sb) = r.gather(r.one("a", a, 24), r.one("b", b, 24))
        r.check()
        assert [ta, tb] == want
        reqs = {f["req"] for f in r.ops("begin")}
        assert len(reqs) == 2
        ra, rb = sorted(reqs)
        assert interleaved(r.frames, ra, rb) and interleaved(r.frames, rb, ra)
        assert sa["session"] != sb["session"]
        assert r.engine.sched_counters["max_active"] == 2
        if spec:
            assert r.be0.stale_drafts == 0, "a draft read another request's drafter ring"
    finally:
        r.stop()


@pytest.mark.parametrize("spec", [False, True])
def test_four_conversations_at_concurrency_four(spec):
    ps = [prompt(310 + i, 30 + 5 * i) for i in range(4)]
    want = serial_replies([(p, 16) for p in ps], spec=spec, pool_pages=96)
    r = IlvRanks(spec=spec, concurrency=4, sessions=4, pool_pages=96)
    try:
        got = r.gather(*(r.one(str(i), p, 16) for i, p in enumerate(ps)))
        r.check()
        assert [t for t, _ in got] == want
        assert r.engine.sched_counters["max_active"] == 4 and r.be1.max_live == 4
        if spec:
            assert r.be0.stale_drafts == 0
    finally:
        r.stop()


@pytest.mark.parametrize("spec", [False, True])
def test_a_fifth_conversation_waits_while_four_run_in_four_sessions(spec):
    """The rig's setting, concurrency = sessions = 4: a fifth conversation arriving while four run
    can neither start nor take a session; it waits in the queue, starts once one of the four has
    ended (in a session whose conversation then leaves), and every reply matches the serialized run."""
    ps = [prompt(340 + i, 30 + 4 * i) for i in range(5)]
    want = serial_replies([(p, 16) for p in ps], spec=spec, pool_pages=96)
    r = IlvRanks(spec=spec, concurrency=4, sessions=4, pool_pages=96, delay=0.003)
    try:
        got = r.gather(*(r.one(str(i), p, 16) for i, p in enumerate(ps)))
        r.check()
        assert [t for t, _ in got] == want
        assert r.engine.sched_counters["max_active"] == 4 and r.be1.max_live == 4
        marks = [f["op"] for f in r.frames if f.get("op") in ("begin", "end")]
        assert marks.count("begin") == 5 and marks.index("end") == 4, marks   # 4 begins, an end, the 5th
        if spec:
            assert r.be0.stale_drafts == 0
    finally:
        r.stop()


@pytest.mark.parametrize("spec", [False, True])
def test_multi_turn_reuse_survives_interleaving(spec):
    """Turn 2 of conversation A runs while conversation B generates: A still resumes from its
    own checkpoint (cached_tokens > 0, same session) and both replies match the serialized run."""
    a1, b = prompt(320, 40), prompt(321, 30)

    def turns(run):
        t1 = run(a1, 12)
        a2 = a1[:-1] + [7] + ids_of(t1) + prompt(322, 20)
        return t1, a2

    ref = SessionRanks(spec=spec, sessions=4, pool_pages=64)
    try:
        t1_ref, a2 = turns(lambda ids, n: ref.run(ids, max_tokens=n)[0])
        want_a2, want_b = ref.run(a2, max_tokens=12)[0], ref.run(b, max_tokens=30)[0]
    finally:
        ref.stop()
    r = IlvRanks(spec=spec, concurrency=2)
    try:
        t1, _ = r.run(a1, 12)
        assert t1 == t1_ref
        started = None

        async def both():
            nonlocal started
            started = asyncio.Event()
            return await asyncio.gather(r.one("b", b, 30, started=started),
                                        r.one("a2", a2, 12, after=started))
        (tb, sb), (ta2, sa2) = asyncio.run(both())
        r.sync()
        r.check()
        assert (ta2, tb) == (want_a2, want_b)
        assert sa2["cached_tokens"] > 0 and sa2["session"] == 1 and sb["session"] != 1
        if spec:
            assert r.be0.stale_drafts == 0
    finally:
        r.stop()


def test_a_short_request_finishes_while_a_long_one_generates():
    """Head-of-line blocking is gone: a 4-token request submitted while a 60-token reply is
    being generated finishes first, after waiting only for its admission."""
    long_p, short_p = prompt(330, 30), prompt(331, 20)
    r = IlvRanks(concurrency=2, delay=0.003)
    try:
        started = None

        async def go():
            nonlocal started
            started = asyncio.Event()
            return await asyncio.gather(r.one("long", long_p, 60, started=started, started_after=3),
                                        r.one("short", short_p, 4, after=started))
        (tl, sl), (ts, ss) = asyncio.run(go())
        r.sync()
        r.check()
        assert r.done_order == ["short", "long"]
        assert ss["time_queued"] < sl["time_generate"]
        assert len(ts) == 4 and len(tl) == 60
    finally:
        r.stop()


def test_follow_up_of_a_running_conversation_waits_for_it_and_reuses_the_cache():
    """A prompt continuing a conversation whose request is still running must not take a second
    session (a full re-prefill of the whole conversation): it waits for the running request,
    then resumes from that conversation's checkpoint -- what the serialized engine did."""
    a1 = prompt(340, 40)
    r = IlvRanks(concurrency=2, delay=0.003)
    try:
        t1, _ = r.run(a1, 8)
        a2 = a1[:-1] + [7] + ids_of(t1) + prompt(341, 20)
        new_before = r.engine.mgr.counters["new_sessions"]
        started = None

        async def go():
            nonlocal started
            started = asyncio.Event()
            return await asyncio.gather(r.one("turn2", a2, 30, started=started, started_after=3),
                                        r.one("again", a2, 8, after=started))
        (_, s_turn2), (t_again, s_again) = asyncio.run(go())
        r.sync()
        r.check()
        assert r.done_order[-2:] == ["turn2", "again"]
        assert r.engine.sched_counters["deferred_conversation"] >= 1
        assert r.engine.mgr.counters["new_sessions"] == new_before, "the follow-up took a fresh session"
        assert s_turn2["session"] == s_again["session"] == 1
        assert s_again["cached_tokens"] == len(a2) - 1
        assert t_again == serial_replies([(a2, 8)])[0]
    finally:
        r.stop()


def test_cancel_and_resend_of_a_turn_reuses_the_conversation():
    """The client gives up on a turn and sends it again at once: whether the resend reaches the
    scheduler before or after the cancelled copy settles, it lands in the same session and
    resumes from its checkpoint."""
    a1 = prompt(342, 40)
    r = IlvRanks(concurrency=2, delay=0.003)
    try:
        t1, _ = r.run(a1, 8)
        a2 = a1[:-1] + [7] + ids_of(t1) + prompt(343, 20)
        new_before = r.engine.mgr.counters["new_sessions"]

        async def go():
            started, cancel = asyncio.Event(), asyncio.Event()

            async def first():
                out = []
                async for piece in r.engine.generate("", prompt_ids=a2, max_tokens=80, temperature=0.0,
                                                     top_p=1.0, stop=[], cancel=cancel):
                    out.append(piece)
                    if len(out) == 3:
                        started.set()
                return "".join(out)

            async def resend():
                await started.wait()
                cancel.set()
                return await r.one("resend", a2, 8)
            return await asyncio.gather(first(), resend())
        _, (t2, s2) = asyncio.run(go())
        r.sync()
        r.check()
        assert r.engine.mgr.counters["new_sessions"] == new_before, "the resend took a fresh session"
        assert s2["session"] == 1 and s2["cached_tokens"] >= len(a1)
        assert t2 == serial_replies([(a2, 8)])[0]
    finally:
        r.stop()


def test_pool_pressure_defers_until_a_running_request_ends():
    """A request whose pages are held by a running request (no inactive session to evict)
    waits instead of failing, and runs once the pages come back."""
    a, b = prompt(350, 40), prompt(351, 40)
    # 16 pages of 16 rows: each request needs ~40 + 80 + 2 rows = 8 pages -> the two do not fit
    r = IlvRanks(concurrency=2, pool_pages=12, delay=0.002)
    try:
        started = None

        async def go():
            nonlocal started
            started = asyncio.Event()
            return await asyncio.gather(r.one("a", a, 80, started=started),
                                        r.one("b", b, 80, after=started))
        (ta, _), (tb, sb) = asyncio.run(go())
        r.sync()
        r.check()
        assert r.engine.sched_counters["deferred_pool"] >= 1
        assert r.done_order == ["a", "b"]
        begins = r.ops("begin")
        ends = r.ops("end")
        assert begins[1]["req"] == ends[0]["req"] + 1 and r.frames.index(begins[1]) > r.frames.index(ends[0])
        assert [ta, tb] == serial_replies([(a, 80), (b, 80)], pool_pages=12)
    finally:
        r.stop()


def test_a_request_too_large_for_the_pool_fails_at_once_even_while_others_run():
    """Waiting cannot help a request bigger than the whole pool: it is refused immediately (as
    the serialized engine refuses it), not parked behind the running requests."""
    a = prompt(352, 30)
    r = IlvRanks(concurrency=2, pool_pages=12, ctx=1000, delay=0.003)
    try:
        with pytest.raises(Exception, match="whole pool"):
            r.run(prompt(353, 40), 400)
        started = None

        async def go():
            nonlocal started
            started = asyncio.Event()

            async def too_big():
                await started.wait()
                t0 = time.perf_counter()
                with pytest.raises(Exception, match="whole pool"):
                    await r.one("big", prompt(354, 40), 400)
                return time.perf_counter() - t0
            return await asyncio.gather(r.one("a", a, 60, started=started), too_big())
        (_, sa), waited = asyncio.run(go())
        r.sync()
        assert waited < sa["time_generate"] and r.engine.sched_counters["deferred_pool"] == 0
        r.check()
    finally:
        r.stop()


def test_cancelled_pending_request_never_reaches_the_wire():
    a, b, c = prompt(360, 30), prompt(361, 30), prompt(362, 30)
    r = IlvRanks(concurrency=2, delay=0.003)
    try:
        async def go():
            started, cancel = asyncio.Event(), asyncio.Event()
            run_a = asyncio.ensure_future(r.one("a", a, 40, started=started))
            run_b = asyncio.ensure_future(r.one("b", b, 40, after=started))
            await started.wait()
            await asyncio.sleep(0.02)                # b admitted: both slots busy

            async def waiting_c():
                with pytest.raises(tp_engine.RequestCancelled):
                    async for _ in r.engine.generate("", prompt_ids=c, max_tokens=8, temperature=0.0,
                                                     top_p=1.0, stop=[], cancel=cancel):
                        pass
            run_c = asyncio.ensure_future(waiting_c())
            await asyncio.sleep(0.02)
            cancel.set()                             # gave up while still queued
            return await run_a, await run_b, await run_c
        asyncio.run(go())
        r.sync()
        r.check()
        assert len(r.ops("begin")) == 2 and r.engine.sched_counters["cancelled_pending"] == 1
    finally:
        r.stop()


def test_cancel_of_a_running_request_leaves_the_other_running():
    a, b = prompt(370, 30), prompt(371, 30)
    want_b = serial_replies([(b, 40)])[0]
    r = IlvRanks(concurrency=2, delay=0.002)
    try:
        async def go():
            started, cancel = asyncio.Event(), asyncio.Event()

            async def a_run():
                out = []
                async for piece in r.engine.generate("", prompt_ids=a, max_tokens=80, temperature=0.0,
                                                     top_p=1.0, stop=[], cancel=cancel):
                    out.append(piece)
                    if len(out) == 2:
                        started.set()
                    if len(out) == 6:
                        cancel.set()
                return "".join(out)
            return await asyncio.gather(a_run(), r.one("b", b, 40, after=started))
        ta, (tb, _) = asyncio.run(go())
        r.sync()
        r.check()
        assert 6 <= len(ta) < 80 and tb == want_b
    finally:
        r.stop()


def test_long_prefill_is_chunked_while_another_request_decodes():
    """A long prompt arriving while another request decodes prefills in interleave-chunk rows,
    with the other request's decode steps in between; alone, it uses the full span."""
    dec, long_p = prompt(380, 20), prompt(381, 150)
    r = IlvRanks(concurrency=2, span=64, chunk=16, delay=0.002, pool_pages=96)
    try:
        started = None

        async def go():
            nonlocal started
            started = asyncio.Event()
            return await asyncio.gather(r.one("dec", dec, 60, started=started),
                                        r.one("long", long_p, 4, after=started))
        (td, _), (tl, _) = asyncio.run(go())
        r.sync()
        r.check()
        long_req = max(f["req"] for f in r.ops("begin"))
        chunks = [f for f in r.ops("chunk") if f["req"] == long_req]
        assert sum(c["b"] - c["a"] for c in chunks) == len(long_p) - 1
        assert all(c["b"] - c["a"] <= 16 for c in chunks[:-1]) and len(chunks) >= 8
        dec_req = min(f["req"] for f in r.ops("begin"))
        assert interleaved(r.frames, long_req, dec_req)
        assert [td, tl] == serial_replies([(dec, 60), (long_p, 4)], pool_pages=96)
        # alone, the same prompt prefills in full-span chunks
        r.frames.clear()
        r.run(prompt(382, 150), 2)
        solo = r.ops("chunk")
        assert [c["b"] - c["a"] for c in solo] == [64, 64, 21]
    finally:
        r.stop()


def test_fair_share_alternates_equal_decoders():
    a, b = prompt(390, 20), prompt(391, 20)
    r = IlvRanks(concurrency=2, delay=0.002)
    try:
        r.gather(r.one("a", a, 30), r.one("b", b, 30))
        r.check()
        steps = [f["req"] for f in r.ops("step")]
        ra, rb = sorted(set(steps))
        mid = [s for s in steps if s in (ra, rb)]
        # once both run, neither gets more than a couple of steps in a row
        first_b = mid.index(rb)
        runs, cur, n = [], None, 0
        for s in mid[first_b:mid.index(ra, first_b) + 40 if ra in mid[first_b:] else None]:
            if s == cur:
                n += 1
            else:
                if cur is not None:
                    runs.append(n)
                cur, n = s, 1
        assert runs and max(runs[:-1] or [1]) <= 3, runs
    finally:
        r.stop()


def test_status_reports_the_scheduler_while_requests_run():
    a, b = prompt(400, 30), prompt(401, 30)
    r = IlvRanks(concurrency=2, delay=0.004)
    try:
        snap = {}

        async def go():
            started = asyncio.Event()

            async def probe():
                await started.wait()
                await asyncio.sleep(0.01)
                snap.update(await asyncio.get_running_loop().run_in_executor(None, r.engine.session_status))
                assert r.engine.rank1_ping() == "busy"      # never pings between running frames
            return await asyncio.gather(r.one("a", a, 40, started=started), r.one("b", b, 40, after=started),
                                        probe())
        asyncio.run(go())
        r.sync()
        sch = snap["scheduler"]
        assert sch["proto"] == tp_engine.PROTO_INTERLEAVE and sch["concurrency"] == 2
        assert len(sch["active"]) == 2 and {a_["session"] for a_ in sch["active"]} == {1, 2}
        assert any(s["active"] for s in snap["sessions"])
        st = r.engine.session_status()
        assert st["scheduler"]["active"] == [] and r.engine.rank1_ping() == "ok"
        r.check()
    finally:
        r.stop()


def test_warmup_and_reset_at_concurrency_two():
    r = IlvRanks(concurrency=2)
    try:
        r.engine.warmup(prompt_tokens=16, decode_tokens=3)
        assert r.engine.mgr.sessions == {}                     # warm-up slots are ephemeral
        r.run(prompt(410, 30), 6)
        asyncio.run(r.engine.reset_prefix_cache())
        t, s = r.run(prompt(410, 30), 6)
        assert s["cached_tokens"] == 0
        assert r.ops("begin")[-1].get("drop_all") is True
        r.check()
    finally:
        r.stop()


def test_explicit_session_of_a_running_request_waits():
    a, b = prompt(420, 30), prompt(421, 30)
    r = IlvRanks(concurrency=2, delay=0.003)
    try:
        started = None

        async def go():
            nonlocal started
            started = asyncio.Event()
            return await asyncio.gather(r.one("a", a, 30, started=started, session=2),
                                        r.one("b", b, 6, after=started, session=2))
        (_, sa), (_, sb) = asyncio.run(go())
        r.sync()
        r.check()
        assert sa["session"] == sb["session"] == 2
        assert r.done_order == ["a", "b"] and r.engine.sched_counters["deferred_conversation"] >= 1
    finally:
        r.stop()


def test_rank1_failure_fails_every_request_in_flight():
    a, b = prompt(430, 30), prompt(431, 30)
    r = IlvRanks(concurrency=2, delay=0.003)
    try:
        async def go():
            started = asyncio.Event()

            async def killer():
                await started.wait()
                r.worker.channel.close()             # rank 1 dies mid-stream
            res = await asyncio.gather(r.one("a", a, 200, started=started), r.one("b", b, 200),
                                       killer(), return_exceptions=True)
            return res
        res = asyncio.run(go())
        assert isinstance(res[0], Exception) and isinstance(res[1], Exception)
        assert r.engine._fatal is not None
        with pytest.raises(RuntimeError, match="restart ALL ranks"):
            r.run(prompt(432, 20), 4)
        assert r.engine.rank1_ping() != "ok"
    finally:
        r.engine.close()
        r.thread.join(timeout=10.0)


def test_first_turn_resend_while_the_first_copy_prefills_reuses_its_session():
    """Review finding: a conversation's FIRST turn has nothing committed while it runs, so the
    running session alone cannot tell a resend belongs to it. The resend must still wait for the
    running copy and resume from its checkpoint -- one session, one prefill -- as serialized."""
    a = prompt(500, 120)
    r = IlvRanks(concurrency=2, delay=0.004, pool_pages=96, span=16)
    try:
        async def go():
            cancel = asyncio.Event()

            async def first():
                out = []
                async for piece in r.engine.generate("", prompt_ids=a, max_tokens=60, temperature=0.0,
                                                     top_p=1.0, stop=[], cancel=cancel):
                    out.append(piece)
                return "".join(out)

            async def resend():
                await asyncio.sleep(0.012)           # the first copy is still prefilling (8 chunks)
                cancel.set()
                return await r.one("resend", a, 8)
            return await asyncio.gather(first(), resend())
        _, (t2, s2) = asyncio.run(go())
        r.sync()
        r.check()
        assert r.engine.mgr.counters["new_sessions"] == 1
        assert s2["session"] == 1 and s2["cached_tokens"] == len(a) - 1
        assert r.engine.sched_counters["deferred_conversation"] >= 1
        chunks = {}
        for f in r.ops("chunk"):
            chunks.setdefault(f["req"], []).append(f)
        assert len(chunks) == 1, "the prompt was prefilled twice"
        assert t2 == serial_replies([(a, 8)], pool_pages=96)[0]
    finally:
        r.stop()


def test_a_failure_between_verify_and_accept_fails_only_that_request():
    """Review finding: rank 1 waits for the accept of a round it verified. A rank-0 error after
    the verify (here: no logits) must settle the round (accept 0) before the end frame, so rank 1
    survives and the other request finishes untouched -- not kill rank 1 mid-stream."""
    a, b = prompt(510, 30), prompt(511, 30)
    want_b = serial_replies([(b, 40)], spec=True)[0]
    r = IlvRanks(spec=True, concurrency=2, delay=0.003)
    orig = r.be0.verify
    calls = {"n": 0}

    def verify(window, position, state, alloc=None):
        out = orig(window, position, state, alloc)
        calls["n"] += 1
        if alloc.sid == 1 and calls["n"] > 6:        # request A only, a few rounds in
            return None, out[1]
        return out
    r.be0.verify = verify
    try:
        async def go():
            return await asyncio.gather(r.one("a", a, 40), r.one("b", b, 40), return_exceptions=True)
        ra, rb = asyncio.run(go())
        r.sync()
        assert isinstance(ra, RuntimeError) and "no logits" in str(ra)
        assert not isinstance(rb, BaseException) and rb[0] == want_b
        assert r.engine._fatal is None and not r.fatal and not r.errors
        r.check()                                    # rank 1 alive, both ranks' calls equal
        t, _ = r.run(prompt(512, 20), 4)             # and the pair keeps serving
        assert len(t) == 4
    finally:
        r.stop()


def test_a_failed_model_call_is_fatal_when_interleaving():
    """A rank-0 model call that fails after its frame left leaves the peers inside a collective
    rank 0 never joined: every request in flight fails at once (instead of the next one hanging),
    the on_fatal hook fires (the servers exit rank 0 there so the watchdog restarts the stack),
    and new requests are refused."""
    a, b = prompt(520, 30), prompt(521, 30)
    r = IlvRanks(concurrency=2, delay=0.003)
    orig = r.be0.forward
    calls = {"n": 0}

    def forward(token, position, state, embeddings=None, alloc=None):
        calls["n"] += 1
        if alloc.sid == 1 and calls["n"] > 8:
            raise RuntimeError("injected device fault")
        return orig(token, position, state, embeddings, alloc)
    r.be0.forward = forward
    try:
        async def go():
            return await asyncio.gather(r.one("a", a, 60), r.one("b", b, 60), return_exceptions=True)
        ra, rb = asyncio.run(go())
        assert isinstance(ra, tp_engine.RankDesync) and "injected device fault" in str(ra)
        assert isinstance(rb, RuntimeError) and "restart ALL ranks" in str(rb)
        assert len(r.fatal) == 1 and isinstance(r.fatal[0], tp_engine.RankDesync)
        with pytest.raises(RuntimeError, match="restart ALL ranks"):
            r.run(prompt(522, 20), 4)
    finally:
        r.engine.close()
        r.thread.join(timeout=10.0)


def test_shutdown_stops_a_long_prefill_between_chunks():
    """close() during a long interleaved prefill: the request stops at the next chunk boundary
    (its end frame settles it), so shutdown does not wait out the whole prefill, and only one
    thread ever writes to the channel."""
    long_p = prompt(530, 300)
    r = IlvRanks(concurrency=2, span=16, delay=0.02, pool_pages=96)
    try:
        async def go():
            async def run():
                with pytest.raises(tp_engine.EngineShutdown):
                    await r.one("long", long_p, 4)

            async def closer():
                await asyncio.sleep(0.1)             # a few of the ~19 chunks are in
                t0 = time.perf_counter()
                await asyncio.get_running_loop().run_in_executor(None, r.engine.close)
                return time.perf_counter() - t0
            _, took = await asyncio.gather(run(), closer())
            return took
        took = asyncio.run(go())
        assert took < 1.0, took                      # the full prefill would take ~19 x 20 ms + more
        r.thread.join(timeout=10.0)
        assert not r.thread.is_alive() and not r.errors, r.errors
        assert len(r.ops("chunk")) < len(long_p) // 16
        assert r.ops("end") and r.ops("end")[-1]["req"] == r.ops("begin")[-1]["req"]
    finally:
        r.engine.close()


def test_control_tasks_never_hang_on_a_stopped_engine():
    r = IlvRanks(concurrency=2)
    r.stop()
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError):
        asyncio.run(r.engine.reset_prefix_cache())
    assert time.perf_counter() - t0 < 2.0
    st = r.engine.session_status()                    # idle snapshot still works, no hang
    assert st["slots"] == 4


def test_request_ids_at_concurrency_one_count_planned_attempts_like_the_serialized_engine():
    """C=1 keeps the old numbering: a request that fails planning (here a bad explicit session
    id) still takes an id, so the next begin frame carries the id it always did."""
    ranks = SessionRanks(sessions=2, pool_pages=32, record=True)
    try:
        with pytest.raises(ValueError, match="session id"):
            ranks.run(prompt(540, 20), max_tokens=4, session=7)
        ranks.run(prompt(541, 20), max_tokens=4)
        begins = [f for f in ranks.frames if f.get("op") == "begin"]
        assert [f["req"] for f in begins] == [2]
        assert ranks.engine.requests == 2
        ranks.check()
    finally:
        ranks.stop()


@pytest.mark.parametrize("where", ["prefill", "verify"])
def test_a_rank0_local_failure_after_the_collective_fails_only_its_request(where):
    """Review finding (round 2): drafter priming after a prefill chunk, or the logits copy after a
    verify, runs on rank 0 only AFTER the collective completed on every rank (the real backend
    raises RankLocalError there). That must fail just the one request -- the spec round is
    settled first -- not latch the engine and restart the stack."""
    a, b = prompt(560, 40), prompt(561, 30)
    want_b = serial_replies([(b, 30)], spec=True)[0]
    r = IlvRanks(spec=True, concurrency=2, delay=0.003, span=16)
    calls = {"n": 0}
    if where == "prefill":
        orig = r.be0.prefill

        def prefill(ids, state, spec=False, start=0, embeddings=None, alloc=None):
            out = orig(ids, state, spec, start, embeddings, alloc)
            if alloc.sid == 1 and start >= 16:
                raise tp_engine.RankLocalError("drafter priming ran out of memory (injected)")
            return out
        r.be0.prefill = prefill
    else:
        orig = r.be0.verify

        def verify(window, position, state, alloc=None):
            out = orig(window, position, state, alloc)
            calls["n"] += 1
            if alloc.sid == 1 and calls["n"] > 4:
                raise tp_engine.RankLocalError("logits copy ran out of memory (injected)")
            return out
        r.be0.verify = verify
    try:
        async def go():
            return await asyncio.gather(r.one("a", a, 30), r.one("b", b, 30), return_exceptions=True)
        ra, rb = asyncio.run(go())
        r.sync()
        assert isinstance(ra, tp_engine.RankLocalError), ra
        assert not isinstance(rb, BaseException) and rb[0] == want_b
        assert r.engine._fatal is None and not r.fatal and not r.errors
        r.check()
        if where == "verify":
            assert any(f.get("op") == "accept" and f.get("accepted") == 0 for f in r.frames)
    finally:
        r.stop()


def test_prompt_continues_needs_a_prompt_long_enough_for_a_checkpoint():
    import tp_sessions
    from array import array
    mgr = tp_sessions.SessionManager(slots=2, pool_pages=8, page_size=PAGE, min_tokens=8)
    run_short, run_long = array("i", prompt(570, 5)), array("i", prompt(571, 30))
    assert not mgr.prompt_continues(run_short, run_short)          # no checkpoint to wait for
    assert mgr.prompt_continues(run_long, run_long)                # the same turn again
    assert mgr.prompt_continues(run_long, array("i", list(run_long[:-1]) + [7, 8, 9]))  # header rewrite
    assert not mgr.prompt_continues(run_long, array("i", list(run_long[:10]) + prompt(572, 20)))


def test_a_short_running_prompt_does_not_hold_up_an_unrelated_request():
    a = prompt(575, 3)                                            # below min_tokens (8)
    b = a[:1] + prompt(576, 20)                                   # shares one token only
    r = IlvRanks(concurrency=2, delay=0.003)
    try:
        started = None

        async def go():
            nonlocal started
            started = asyncio.Event()
            return await asyncio.gather(r.one("a", a, 40, started=started),
                                        r.one("b", b, 4, after=started))
        asyncio.run(go())
        r.sync()
        r.check()
        assert r.done_order == ["b", "a"] and r.engine.sched_counters["deferred_conversation"] == 0
    finally:
        r.stop()


def test_exit_on_fatal_hook_defaults(monkeypatch):
    monkeypatch.delenv("EXL3_TP_EXIT_ON_FATAL", raising=False)
    assert callable(tp_engine.exit_on_fatal_hook(lambda *_: None))                     # TP=3 default
    assert tp_engine.exit_on_fatal_hook(lambda *_: None, default_on=False) is None      # TP=2 default
    monkeypatch.setenv("EXL3_TP_EXIT_ON_FATAL", "1")
    assert callable(tp_engine.exit_on_fatal_hook(lambda *_: None, default_on=False))
    monkeypatch.setenv("EXL3_TP_EXIT_ON_FATAL", "0")
    assert tp_engine.exit_on_fatal_hook(lambda *_: None) is None


def test_concurrency_one_numbers_a_skipped_cancel_like_the_serialized_engine():
    """At C=1 a request cancelled while queued is skipped (nothing reaches the wire) but still
    takes the request id the serialized engine gave it, so later frames keep their numbers."""
    a, b, c = prompt(580, 30), prompt(581, 30), prompt(582, 30)
    r = IlvRanks(concurrency=1, delay=0.003)
    try:
        async def go():
            started, cancel = asyncio.Event(), asyncio.Event()
            run_a = asyncio.ensure_future(r.one("a", a, 40, started=started))
            await started.wait()

            async def queued_b():
                with pytest.raises(tp_engine.RequestCancelled):
                    async for _ in r.engine.generate("", prompt_ids=b, max_tokens=8, temperature=0.0,
                                                     top_p=1.0, stop=[], cancel=cancel):
                        pass
            run_b = asyncio.ensure_future(queued_b())
            await asyncio.sleep(0.02)
            cancel.set()
            await run_a
            await run_b
            return await r.one("c", c, 4)
        asyncio.run(go())
        r.sync()
        r.check()
        assert [f["req"] for f in r.ops("begin")] == [1, 3]
        assert r.engine.proto == tp_engine.PROTO_SESSIONS and not r.ops("chunk")
    finally:
        r.stop()


# ---- the HTTP layer -----------------------------------------------------------------------------------------


def test_http_layer_streams_two_requests_concurrently():
    """Through the vendored server (sessions=4 -> 4 server workers): a second chat request
    streams its first token while the first is still generating, and /tp/status reports
    proto 6 with the scheduler."""
    from aiohttp.test_utils import TestClient, TestServer
    import tp_serve

    r = IlvRanks(concurrency=2, pool_pages=96, delay=0.004)

    async def go():
        client = TestClient(TestServer(tp_serve.create_app(r.engine, "tp-ilv")))
        await client.start_server()
        events = []
        try:
            async def chat(name, text, n):
                body = {"model": "tp-ilv", "messages": [{"role": "user", "content": text}],
                        "max_tokens": n, "temperature": 0.0, "stream": True}
                resp = await client.post("/v1/chat/completions", json=body)
                assert resp.status == 200, await resp.text()
                async for line in resp.content:
                    if line.startswith(b"data: ") and b"[DONE]" not in line:
                        events.append((name, json.loads(line[6:])))
                events.append((name, "done"))

            first = asyncio.ensure_future(chat("long", "Tell me a long story " * 3, 60))
            while not any(n == "long" for n, _ in events):
                await asyncio.sleep(0.005)
            await chat("short", "Quick question " * 2, 4)
            await first
            st = await (await client.get("/tp/status")).json()
            return events, st
        finally:
            await client.close()

    try:
        events, st = asyncio.run(go())
        r.sync()
        order = [n for n, e in events if e == "done"]
        assert order == ["short", "long"], order
        assert st["proto"] == tp_engine.PROTO_INTERLEAVE and st["pool"]["scheduler"]["concurrency"] == 2
        r.check()
    finally:
        r.stop()


# ---- handshake and rank-1 guards (raw frames) ---------------------------------------------------------------


def _worker(backend, port, errors, *, sessions=4, concurrency=2):
    def run():
        try:
            chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
            chan.set_recv_timeout(None)
            w = tp_engine.Rank1Worker(backend, chan, sessions=sessions, concurrency=concurrency,
                                      log=lambda *_: None)
            run.worker = w
            w.serve_forever()
        except BaseException as exc:                                     # noqa: BLE001
            errors.append(exc)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, run


@pytest.mark.parametrize("worker_conc, hello_kw, needle", [
    (2, dict(concurrency=4), "EXL3_TP_CONCURRENCY"),
    (2, dict(concurrency=None), "concurrency >= 2"),
    (2, dict(proto=tp_engine.PROTO_SESSIONS, concurrency=None), "same value"),
    (4, dict(concurrency=4), "live request slots"),
])
def test_proto6_handshake_validates_concurrency(worker_conc, hello_kw, needle):
    port = free_port()
    errors = []
    backend = IlvBackend(8, slots=2)
    t, _ = _worker(backend, port, errors, concurrency=worker_conc)
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    try:
        cli.set_recv_timeout(5.0)
        with pytest.raises(Exception):
            _hello(cli, sessions=4, pool_pages=8, **hello_kw)
    finally:
        cli.close()
        t.join(5.0)
    assert not t.is_alive() and errors and needle in str(errors[0]), (needle, errors)


def test_engine_refuses_concurrency_it_cannot_serve():
    chan = object()
    with pytest.raises(ValueError, match="multi-session"):
        tp_engine.TPEngine(IlvBackend(8), chan, Tok(), ctx=64, sessions=1, concurrency=2)
    with pytest.raises(ValueError, match="EXL3_TP_SESSIONS"):
        tp_engine.TPEngine(IlvBackend(8), chan, Tok(), ctx=64, sessions=2, concurrency=3,
                           session_opts={})
    with pytest.raises(ValueError, match="live request slot"):
        tp_engine.TPEngine(IlvBackend(8, slots=1), chan, Tok(), ctx=64, sessions=4, concurrency=2,
                           session_opts={})


def _begin_frame(req, sid, pages, ids, **kw):
    f = {"op": "begin", "req": req, "ids": ids, "session": sid, "pages": pages}
    f.update(kw)
    return f


def test_rank1_proto6_guards():
    """Frames a correct rank 0 never sends are refused (the worker reports and dies): a second
    request in a running session, a step for a request that is not running, a chunk that does
    not continue the prefill, a step before the prefill completed."""
    ids = prompt(440, 20)
    cases = [
        ("second request in a running session",
         [_begin_frame(1, 1, [0, 1], ids), _begin_frame(2, 1, [0, 1], ids)], "runs in it"),
        ("unknown request", [{"op": "step", "req": 9, "token": 65}], "not running"),
        ("chunk out of order", [_begin_frame(1, 1, [0, 1], ids), {"op": "chunk", "req": 1, "a": 4, "b": 8}],
         "does not continue"),
        ("step before prefill", [_begin_frame(1, 1, [0, 1], ids), {"op": "step", "req": 1, "token": ids[-1]}],
         "before its prefill"),
        ("too many requests", [_begin_frame(1, 1, [0, 1], ids), _begin_frame(2, 2, [2, 3], ids),
                               _begin_frame(3, 3, [4, 5], ids)], "beyond the 2"),
        ("evict of a running session", [_begin_frame(1, 1, [0, 1], ids),
                                        _begin_frame(2, 2, [2, 3], ids, evict=[1])], "while request 1 runs"),
    ]
    for name, frames, needle in cases:
        port = free_port()
        errors = []
        t, _ = _worker(IlvBackend(8), port, errors)
        cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
        try:
            cli.set_recv_timeout(5.0)
            ok = _hello(cli, sessions=4, pool_pages=8)
            assert ok["op"] == "hello_ok" and ok["concurrency"] == 2
            for f in frames:
                cli.send(f)
            reply = cli.recv()
            assert reply["op"] == "error" and needle in reply["message"], (name, reply)
        finally:
            cli.close()
            t.join(5.0)
        assert not t.is_alive(), name


def test_rank1_proto6_runs_two_requests_by_id():
    """The raw proto-6 sequence: two requests set up, prefilled and stepped in alternation by
    request id, each with its own state and pages, then ended -- rank 1's records match."""
    a, b = prompt(450, 20), prompt(451, 20)
    port = free_port()
    errors = []
    be = IlvBackend(8)
    t, holder = _worker(be, port, errors)
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    try:
        cli.set_recv_timeout(5.0)
        _hello(cli, sessions=4, pool_pages=8)
        cli.send(_begin_frame(1, 1, [0, 1, 2], a))
        cli.send(_begin_frame(2, 2, [3, 4, 5], b))
        cli.send({"op": "chunk", "req": 1, "a": 0, "b": 19})
        cli.send({"op": "chunk", "req": 2, "a": 0, "b": 10})
        cli.send({"op": "step", "req": 1, "token": a[-1]})
        cli.send({"op": "chunk", "req": 2, "a": 10, "b": 19})
        cli.send({"op": "step", "req": 2, "token": b[-1]})
        cli.send({"op": "step", "req": 1, "token": 70})
        cli.send({"op": "end", "req": 2, "session": 2})
        cli.send({"op": "end", "req": 1, "session": 1})
        cli.send({"op": "ping"})
        assert cli.recv()["op"] == "pong"
        w = holder.worker
        assert not w.jobs and set(w.sess) == {1, 2} and w.requests_served == 2
        fw = [c for c in be.calls if c[0] == "forward"]
        assert [(c[1], c[3]) for c in fw] == [(1, 19), (2, 19), (1, 20)]
    finally:
        cli.close()
        t.join(5.0)
    assert not errors, errors
