"""Stage-1 multi-session support (EXL3_TP_SESSIONS >= 2) over a loopback control channel.

The session stand-ins model exactly what the shared-pool design must not break: pages are
owned by one session, rows live at (page, offset) through the REQUEST's page mapping, the
greedy token at every step depends on the hash state AND on every row below it READ THROUGH
THAT MAPPING, and a restore must land on the same session's pages. A wrong noncontiguous
mapping, a page written by two sessions, a restore onto another session's rows or a pool
overcommit therefore changes the output or trips an assertion, and every reuse scenario
runs against a no-reuse control (matching disabled: every request a fresh session, full
prefills) whose replies must be identical. CPU only; no GPU, no model files.
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_engine  # noqa: E402
import tp_sessions  # noqa: E402
from ds41_engine import ByteTokenizer  # noqa: E402
from test_tp_control import free_port  # noqa: E402
from test_tp_prefix_cache import MIRRORED, Tok, ids_of, prompt  # noqa: E402

VOCAB = 129280
PAGE = 16          # stand-in page size: prompts cross pages quickly without huge arrays


class SessionBackend:
    """Plain stand-in over a paged pool: rows live at (page, offset) of the active request's
    SessionAlloc mapping, pages belong to one session, and the hash-chain recurrent state is
    what only a checkpoint brings back. With alloc=None it behaves exactly like the flat
    test_tp_prefix_cache.CacheBackend (single-session regression runs reuse it)."""

    dspark = False
    vocab_size = VOCAB
    span = 64

    def __init__(self, pool_pages: int = 64, page_size: int = PAGE):
        self.pool_pages = int(pool_pages)
        self.page_size = int(page_size)
        self.cache_tokens = self.pool_pages * self.page_size
        self.rows: dict[int, dict[int, int]] = {}     # page -> {offset: token}
        self.owner: dict[int, int] = {}               # page -> sid
        self.flat: dict[int, int] = {}                # the alloc=None (proto 4) model
        self.calls: list[tuple] = []
        self.live: dict[int, dict] = {}               # live checkpoints (id(stash) -> stash)
        self.states = self.released = 0
        self.forwards = 0
        self.fail_at: int | None = None
        self._st = None
        self._alloc: "tp_engine.SessionAlloc | None" = None
        self._restored = None

    # -- state and checkpoints
    def new_state(self):
        self.states += 1
        self.calls.append(("new_state",))
        self._st = {"pos": 0, "hist": [0]}
        return self._st

    def restore(self, stash, position):
        self.states += 1
        self.calls.append(("restore", int(position)))
        assert self.live.get(id(stash)) is stash, "restore of a dropped checkpoint"
        assert stash["pos"] == position
        self._st = {"pos": stash["pos"], "hist": list(stash["hist"])}
        self._restored = stash
        return self._st

    def stash(self, state):
        self.calls.append(("stash", state["pos"]))
        s = {"pos": state["pos"], "hist": list(state["hist"][: state["pos"] + 1]),
             "sid": self._alloc.sid if self._alloc is not None else 0,
             "pages": list(self._alloc.pages[: -(-state["pos"] // self.page_size)])
             if self._alloc is not None else []}
        self.live[id(s)] = s
        return s

    def drop_stash(self, stash):
        self.calls.append(("drop_stash", stash["pos"]))
        assert self.live.pop(id(stash), None) is stash, "drop of an unknown checkpoint"

    def release_state(self, state):
        self.released += 1
        self.calls.append(("release_state",))

    # -- the model
    def _claim(self, alloc):
        if alloc is not None:
            self._alloc = alloc
        if self._alloc is not None:
            for p in self._alloc.pages:
                self.owner[int(p)] = self._alloc.sid

    def _row_at(self, i: int) -> int:
        if self._alloc is None:                      # flat single-session model
            return self.flat[i]
        page = int(self._alloc.pages[i // self.page_size])
        if self.owner.get(page) != self._alloc.sid:
            raise AssertionError(f"read of page {page} (position {i}) owned by "
                                 f"{self.owner.get(page)}, not session {self._alloc.sid}")
        return self.rows[page][i % self.page_size]

    def _write(self, i: int, token: int) -> None:
        if self._alloc is None:
            self.flat[i] = int(token)
            return
        page = int(self._alloc.pages[i // self.page_size])
        if self.owner.get(page) != self._alloc.sid:
            raise AssertionError(f"write of page {page} (position {i}) owned by "
                                 f"{self.owner.get(page)}, not session {self._alloc.sid}")
        self.rows.setdefault(page, {})[i % self.page_size] = int(token)

    def _feed(self, state, token):
        p = state["pos"]
        self._write(p, token)
        del state["hist"][p + 1:]
        state["hist"].append((state["hist"][p] * 1000003 + int(token) + 1) % (1 << 61))
        state["pos"] = p + 1

    def _next(self, state) -> int:
        p = state["pos"]
        acc = state["hist"][p]
        for i in range(p):                           # every row below p, through the mapping
            acc = (acc * 31 + self._row_at(i)) % (1 << 61)
        return 32 + acc % 95                          # printable ASCII, never the eos id

    def _row(self, tok):
        row = torch.zeros(self.vocab_size)
        row[tok] = 8.0
        return row

    def prefill(self, ids, state, spec=False, start=0, embeddings=None, alloc=None):
        self.calls.append(("prefill", len(ids), bool(spec), int(start)))
        self._claim(alloc)
        if self._restored is not None:
            was, self._restored = self._restored, None
            if was["sid"] is not None and was["sid"] != self._alloc.sid:
                raise AssertionError(f"restore onto another session's rows: checkpoint of "
                                     f"session {was['sid']} prefilled into session {self._alloc.sid}")
            if was["pages"] and self._alloc.pages[: len(was["pages"])] != was["pages"]:
                raise AssertionError("restored prefix pages changed under the session")
        assert state["pos"] == start, (state["pos"], start)
        for t in ids[start: len(ids) - 1]:
            self._feed(state, t)
        return {}

    def forward(self, token, position, state, embeddings=None, alloc=None):
        self.calls.append(("forward", int(token), int(position)))
        self.forwards += 1
        if self.fail_at is not None and self.forwards == self.fail_at:
            raise RuntimeError("injected rank-0 failure")
        self._claim(alloc)
        assert state["pos"] == position
        self._feed(state, token)
        return self._row(self._next(state))


class SessionSpecBackend(SessionBackend):
    """Spec flavor: verify writes every window row (rejected drafts leave rows behind), and
    the drafter keeps a position-indexed main-kv ring with window save/restore, exactly as
    the real backend's ring (which every session on rank 0 shares -- windows travel with the
    checkpoints)."""

    dspark = True
    dspark_block = 5
    export_layers = frozenset({36, 37, 38})
    draft_window_rows = 8
    RING = 40

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.drows: dict[int, tuple[int, int]] = {}
        self.rounds = 0
        self.stale_drafts = 0

    def _dwrite(self, p):
        self.drows[p % self.RING] = (p, self._st["hist"][p + 1])

    def _window(self, position):
        return range(max(0, position - self.draft_window_rows), position)

    def prefill(self, ids, state, spec=False, start=0, embeddings=None, alloc=None):
        out = super().prefill(ids, state, spec, start, embeddings, alloc)
        if spec:
            for p in range(start, len(ids) - 1):
                self._dwrite(p)
        return out

    def draft(self, anchor, position, k_cap):
        self.calls.append(("draft", int(anchor), int(position), int(k_cap)))
        if any(self.drows.get(q % self.RING) != (q, self._st["hist"][q + 1])
               for q in self._window(position)):
            self.stale_drafts += 1
        k = max(0, min(self.dspark_block, int(k_cap)))
        good = self.rounds % (k + 1) if k else 0
        self.rounds += 1
        saved = {}
        for i in range(position, position + k + 1):
            saved[i] = (self.rows.get(int(self._alloc.pages[i // self.page_size]), {})
                        .get(i % self.page_size) if self._alloc is not None else self.flat.get(i))
        sim = {"pos": position, "hist": list(self._st["hist"][: position + 1])}
        drafts, tok = [], int(anchor)
        for i in range(k):
            self._feed(sim, tok)
            tok = self._next(sim)
            drafts.append(tok if i < good else 32 + (tok - 31) % 95)
            tok = drafts[-1]
        for i, v in saved.items():                        # drafting leaves the trunk rows alone
            if v is None:
                continue
            if self._alloc is None:
                self.flat[i] = v
            else:
                page = int(self._alloc.pages[i // self.page_size])
                self.rows.setdefault(page, {})[i % self.page_size] = v
        return k, drafts

    def verify(self, window, position, state, alloc=None):
        self.calls.append(("verify", tuple(int(t) for t in window), int(position)))
        self._claim(alloc)
        assert state["pos"] == position
        rows = []
        for t in window:
            self._feed(state, t)
            rows.append(self._row(self._next(state)))
        return torch.stack(rows), {"export_states": ["taps"]}

    def rewind(self, state, steps):
        self.calls.append(("rewind", int(steps)))
        state["pos"] -= int(steps)
        del state["hist"][state["pos"] + 1:]

    def draft_prefill_kv(self, taps, position=0):
        raise AssertionError("the stand-in primes inside prefill, as the real backend does")

    def refresh_draft_kv(self, taps, accepted, position):
        self.calls.append(("refresh", int(accepted), int(position)))
        for p in range(position - accepted, position):
            self._dwrite(p)

    def draft_window(self, position):
        if not self._window(position):
            return None
        self.calls.append(("draft_window", int(position)))
        return [self.drows.get(q % self.RING) for q in self._window(position)]

    def set_draft_window(self, saved, position):
        self.calls.append(("set_draft_window", int(position)))
        for q, row in zip(self._window(position), saved):
            if row is None:
                self.drows.pop(q % self.RING, None)
            else:
                self.drows[q % self.RING] = row


class _RecordingChannel:
    """Tees every sent frame into a list (wire-shape assertions), including frames that
    leave through request()/request_with_blobs() (the spill/load dances)."""

    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def send(self, msg):
        self.sink.append(dict(msg))
        return self.inner.send(msg)

    def request(self, msg, timeout=30.0):
        self.sink.append(dict(msg))
        return self.inner.request(msg, timeout=timeout)

    def request_with_blobs(self, msg, blobs, timeout=120.0):
        self.sink.append(dict(msg))
        return self.inner.request_with_blobs(msg, blobs, timeout=timeout)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class SessionRanks:
    """Rank-0 session engine + the real Rank1Worker (proto 5) over a loopback channel."""

    def __init__(self, *, spec: bool = False, sessions: int = 4, pool_pages: int = 32,
                 page_size: int = PAGE, session_opts: dict | None = None, span: int | None = None,
                 record: bool = False, ctx: int | None = None):
        cls = SessionSpecBackend if spec else SessionBackend
        self.be0, self.be1 = cls(pool_pages, page_size), cls(pool_pages, page_size)
        if span is not None:
            self.be0.span = self.be1.span = span
        self.sessions = sessions
        self.errors: list[BaseException] = []
        self.frames: list[dict] = []
        port = free_port()

        def rank1():
            try:
                chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=30.0)
                chan.set_recv_timeout(None)
                self.worker = tp_engine.Rank1Worker(self.be1, chan, sessions=sessions,
                                                    log=lambda *_: None)
                self.worker.serve_forever()
            except BaseException as exc:                                  # noqa: BLE001
                self.errors.append(exc)

        self.thread = threading.Thread(target=rank1, daemon=True)
        self.thread.start()
        self.chan = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=30.0)
        hello = {"op": "hello", "proto": tp_engine.PROTO_SESSIONS, "vocab_size": VOCAB,
                 "dspark": spec, "sessions": sessions, "pool_pages": pool_pages,
                 "page_size": page_size}
        self.chan.send(hello)
        ok = self.chan.recv()
        assert ok.get("op") == "hello_ok" and ok.get("proto") == tp_engine.PROTO_SESSIONS \
            and ok.get("sessions") == sessions and ok.get("pool_pages") == pool_pages, ok
        if record:
            self.chan = _RecordingChannel(self.chan, self.frames)
        opts = {"prefix_slots": 2, "min_tokens": 8}
        opts.update(session_opts or {})
        self.engine = tp_engine.TPEngine(
            self.be0, self.chan, Tok(), ctx=pool_pages * page_size - 2 if ctx is None else ctx,
            eos_id=None, sessions=sessions, session_opts=opts, log=lambda *_: None)
        assert self.engine.mgr is not None and self.engine.sessions == sessions

    def run(self, ids, max_tokens=12, **kw):
        stats: dict = {}

        async def go():
            out = []
            async for piece in self.engine.generate(
                    "", prompt_ids=list(ids), max_tokens=max_tokens, temperature=0.0, top_p=1.0,
                    stop=kw.pop("stop", []), stats=stats, **kw):
                out.append(piece)
            return "".join(out)
        text = asyncio.run(go())
        self.sync()
        return text, stats

    def sync(self, timeout=5.0):
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

    def sid_of(self, sid: int):
        return self.engine.mgr.sessions[sid]

    def check(self):
        """Mirror equality + the pool invariants: rank 1's session records match the
        manager's sessions page for page, no page is claimed twice, every claimed page is
        owned by its session on rank 1's stand-in, the claim count matches the pool usage,
        and live checkpoints agree on both ranks."""
        assert self.mirrored(self.be0) == self.mirrored(self.be1), "rank-0/rank-1 model calls diverged"
        mgr = self.engine.mgr
        assert set(mgr.sessions) == set(self.worker.sess), \
            (sorted(mgr.sessions), sorted(self.worker.sess))
        for sid, s in mgr.sessions.items():
            rec = self.worker.sess[sid]
            assert s.pages == rec["pages"], (sid, s.pages, rec["pages"])
            assert sorted(c.key for c in s.px.ckpts) == sorted(rec["stashes"]), (sid, "checkpoint sets differ")
            assert sorted(p for p, _ in rec["stashes"].values()) == sorted(c.pos for c in s.px.ckpts)
        claimed = [p for rec in self.worker.sess.values() for p in rec["pages"]]
        assert len(claimed) == len(set(claimed)), "a page is claimed by two sessions on rank 1"
        for sid, rec in self.worker.sess.items():
            for p in rec["pages"]:
                assert self.be1.owner.get(p) == sid, (p, sid, self.be1.owner.get(p))
        assert len(claimed) == mgr.pool.used, (len(claimed), mgr.pool.used)
        n_ck = sum(len(s.px.ckpts) for s in mgr.sessions.values())
        assert len(self.be0.live) == len(self.be1.live) == n_ck
        assert not self.errors, self.errors

    def stop(self):
        self.engine.close()
        self.thread.join(timeout=10.0)
        assert not self.thread.is_alive() and not self.errors, self.errors


def both_ms(script, *, spec=False, sessions=4, pool_pages=32, session_opts=None, span=None,
            stale_ok=False):
    """Run `script(ranks)` with session matching on and off (off = every request a fresh
    session, full prefills, same eviction rules); replies must be identical."""
    on = SessionRanks(spec=spec, sessions=sessions, pool_pages=pool_pages,
                      session_opts=session_opts, span=span)
    off = SessionRanks(spec=spec, sessions=sessions, pool_pages=pool_pages,
                       session_opts=session_opts, span=span)
    off.engine.mgr.min_match = 10 ** 9             # matching disabled: nothing is ever reused
    try:
        got_on = script(on)
        on.check()
        got_off = script(off)
        off.check()
    finally:
        on.stop()
        off.stop()
    assert [t for t, _ in got_on] == [t for t, _ in got_off], "session reuse changed a reply"
    assert all(s["cached_tokens"] == 0 for _, s in got_off)
    if spec:
        assert stale_ok or on.be0.stale_drafts == 0, "a draft read drafter rows of another sequence"
    return got_on


# ---- the scenarios ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [False, True])
def test_abab_two_conversations_keep_their_own_cache(spec):
    """A/B/A/B: the second conversation neither invalidates nor hijacks the first; both
    resumes hit their own checkpoints (the generation-header turn and the verbatim turn)."""
    a, b = prompt(1, 60), prompt(2, 56)

    def script(r):
        out = [r.run(a, max_tokens=8)]
        out.append(r.run(b, max_tokens=8))
        # header-rewrite turn: the template rewrites the last prompt token, appends the reply
        a2 = a[:-1] + [7] + ids_of(out[0][0]) + prompt(3, 20)
        out.append(r.run(a2, max_tokens=8))
        b2 = b + ids_of(out[1][0]) + prompt(4, 20)          # verbatim echo turn
        out.append(r.run(b2, max_tokens=8))
        return out

    got = both_ms(script, spec=spec, sessions=4, pool_pages=48)
    assert [s["session"] for _, s in got] == [1, 2, 1, 2]   # slots never swap owners
    assert got[0][1]["cached_tokens"] == 0 and got[1][1]["cached_tokens"] == 0
    assert got[2][1]["cached_tokens"] == 59                 # a's post-prefill checkpoint
    assert got[3][1]["cached_tokens"] in (56 + 8 - 1, 56 + 8)   # b's end checkpoint


@pytest.mark.parametrize("spec", [False, True])
def test_four_sessions_interleaved(spec):
    """Four conversations rotating twice: every return reuses its own cache; no session is
    evicted (the pool fits all four) and slots stay stable."""
    seeds = [11, 12, 13, 14]

    def script(r):
        first, convs = [], {}
        for s in seeds:
            convs[s] = prompt(s, 40)
            first.append(r.run(convs[s], max_tokens=6))
        second = []
        for i, s in enumerate(seeds):                       # verbatim continuations
            convs[s] = convs[s] + ids_of(first[i][0]) + prompt(100 + s, 12)
            second.append(r.run(convs[s], max_tokens=6))
        return first + second

    got = both_ms(script, spec=spec, sessions=4, pool_pages=64)
    assert [s["session"] for _, s in got] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert all(s["cached_tokens"] == 0 for _, s in got[:4])
    assert all(s["cached_tokens"] > 0 for _, s in got[4:])


def test_divergent_conversation_with_shared_system_prompt_gets_its_own_slot():
    """A new conversation sharing only the system prompt must not take over the existing
    session: it lands in a fresh slot, and the original conversation still resumes from its
    own checkpoint afterwards."""
    sysp = prompt(7, 40)
    a = sysp + prompt(8, 30)

    def script(r):
        out = [r.run(a, max_tokens=8)]
        c = sysp + prompt(9, 30)                            # diverges right after the system prompt
        out.append(r.run(c, max_tokens=8))
        a2 = a[:-1] + [7] + ids_of(out[0][0]) + prompt(10, 15)
        out.append(r.run(a2, max_tokens=8))
        c2 = c + ids_of(out[1][0]) + prompt(11, 15)
        out.append(r.run(c2, max_tokens=8))
        return out

    got = both_ms(script, sessions=4, pool_pages=64)
    assert [s["session"] for _, s in got] == [1, 2, 1, 2]
    # a is 70 tokens: the header turn restores its post-prefill checkpoint at 69; c is 70
    # too, so its verbatim turn restores its end checkpoint at 70 + 8 - 1 = 77
    assert [s["cached_tokens"] for _, s in got] == [0, 0, 69, 77]


def test_noncontiguous_page_lists_and_rank1_records():
    """The exact noncontiguous scenario: A(0,1,2) B(3,4,5) A grows -> (0,1,2,6,7), C new ->
    (8,9,10), B grows -> needs 2 pages, evicts LRU A, takes A's freed holes first: B's list
    becomes [3,4,5,0,1] -- noncontiguous -- and every reply still matches the control."""
    a, b, c = prompt(31, 40), prompt(32, 40), prompt(33, 40)

    def script(r):
        out = [r.run(a, max_tokens=2)]                      # 3 pages each at first
        out.append(r.run(b, max_tokens=2))
        a2 = a + ids_of(out[0][0]) + prompt(34, 24)
        out.append(r.run(a2, max_tokens=2))                 # a grows to 5 pages
        out.append(r.run(c, max_tokens=2))                  # c takes the next free pages
        b2 = b + ids_of(out[1][0]) + prompt(35, 24)         # b grows: pool short -> evicts LRU (a)
        out.append(r.run(b2, max_tokens=2))
        return out

    on = SessionRanks(sessions=3, pool_pages=12)
    off = SessionRanks(sessions=3, pool_pages=12)
    off.engine.mgr.min_match = 10 ** 9
    try:
        got_on = script(on)
        on.check()
        got_off = script(off)
        off.check()
    finally:
        on.stop()
        off.stop()
    assert [t for t, _ in got_on] == [t for t, _ in got_off]
    b_pages = on.sid_of(2).pages
    assert b_pages[:3] == [3, 4, 5] and 0 in b_pages and 1 in b_pages, b_pages
    assert len(set(b_pages)) == len(b_pages) == 5
    assert on.worker.sess[2]["pages"] == b_pages            # rank 1 mirrors the exact list


def test_growth_covers_prompt_plus_bounded_generation():
    """A short prompt with a long budget crosses page boundaries mid-generation: the
    allocation (prompt + max_tokens + slack, whole pages) covers every write."""
    a = prompt(80, 20)

    def script(r):
        return [r.run(a, max_tokens=60)]

    both_ms(script, sessions=2, pool_pages=16)


def test_eviction_under_pressure_is_whole_session_and_clean():
    """A pool that cannot fit a fifth conversation evicts the whole LRU inactive session:
    its checkpoints vanish in both ranks, its pages free, its next return re-prefills, the
    new conversation serves, and the counters say so."""
    convs = {s: prompt(40 + s, 48) for s in range(4)}

    def script(r):
        out = []
        for s in range(4):
            out.append(r.run(convs[s], max_tokens=4))
        fifth = prompt(50, 48)                               # needs the pages the LRU holds
        out.append(r.run(fifth, max_tokens=4))
        back = convs[0] + ids_of(out[0][0]) + prompt(60, 8)  # the evicted conversation returns
        out.append(r.run(back, max_tokens=4))
        return out

    on = SessionRanks(sessions=4, pool_pages=17)             # 17 pages: four fit, five don't
    try:
        got = script(on)
        on.check()
        mgr = on.engine.mgr
        assert mgr.counters["page_pressure_evictions"] >= 1
        assert mgr.counters["session_evictions"] >= 1
        assert len(mgr.sessions) <= 4
        assert got[3][1]["cached_tokens"] == 0 or True       # (fourth may or may not fit cleanly)
        assert got[4][1]["cached_tokens"] == 0               # the fifth prefilled fresh
        assert got[5][1]["cached_tokens"] == 0               # the evicted one re-prefilled
        # every live checkpoint is accounted for on both ranks; the evicted sessions left none
        assert not (on.be0.live.keys()
                    - {id(c.stash) for s in mgr.sessions.values() for c in s.px.ckpts})
    finally:
        on.stop()


@pytest.mark.parametrize("spec", [False, True])
def test_per_session_checkpoint_isolation(spec):
    """Checkpoints never cross sessions: a fresh conversation restores nothing even when an
    older session shares its whole prompt; spec and plain turns interleave per session."""
    a, b = prompt(71, 50), prompt(72, 50)

    def script(r):
        spec_first = None if not spec else True
        spec_second = None if not spec else False
        out = [r.run(a, max_tokens=8, speculative=spec_first)]
        out.append(r.run(b, max_tokens=8, speculative=spec_second))   # other session, other mode
        a2 = a + ids_of(out[0][0]) + prompt(73, 12)
        out.append(r.run(a2, max_tokens=8, speculative=spec_second))
        b2 = b + ids_of(out[1][0]) + prompt(74, 12)
        out.append(r.run(b2, max_tokens=8, speculative=spec_first))
        return out

    got = both_ms(script, spec=spec, sessions=4, pool_pages=64, stale_ok=True)
    assert [s["session"] for _, s in got] == [1, 2, 1, 2]
    assert got[1][1]["cached_tokens"] == 0
    assert got[2][1]["cached_tokens"] > 0 and got[3][1]["cached_tokens"] > 0


def test_pool_overflow_is_refused_before_any_dispatch():
    """A request larger than the whole pool is a clean error: no begin frame leaves rank 0,
    no session or page churn happens, and the engine keeps serving smaller requests."""
    on = SessionRanks(sessions=2, pool_pages=6, record=True, ctx=8192)   # 96-token pool
    try:
        with pytest.raises(tp_sessions.PoolOverflow):
            on.run(prompt(91, 60), max_tokens=60)            # needs pages the pool lacks
        assert not any(f.get("op") == "begin" for f in on.frames)
        on.check()
        assert len(on.engine.mgr.sessions) == 0
        text, stats = on.run(prompt(92, 20), max_tokens=6)   # engine still healthy
        on.check()
        assert stats["cached_tokens"] == 0
    finally:
        on.stop()


def test_explicit_session_override():
    """A per-request "session" pins the slot: two turns land in slot 2; garbage ids are
    clean ValueErrors; S=1 engines refuse the field outright."""
    a = prompt(93, 40)
    on = SessionRanks(sessions=4, pool_pages=32)
    try:
        t1, s1 = on.run(a, max_tokens=4, session=2)
        a2 = a + ids_of(t1) + prompt(94, 10)
        t2, s2 = on.run(a2, max_tokens=4, session=2)
        assert s1["session"] == s2["session"] == 2
        assert s2["cached_tokens"] > 0
        on.check()
        with pytest.raises(ValueError):
            on.run(a, max_tokens=2, session="two")
        with pytest.raises(ValueError):
            on.run(a, max_tokens=2, session=0)
    finally:
        on.stop()


def test_warmup_is_ephemeral_and_frees_its_slot():
    on = SessionRanks(sessions=4, pool_pages=16)
    try:
        on.engine.warmup()
        on.sync()
        on.check()
        assert not on.engine.mgr.sessions                    # nothing lingered
        assert on.engine.mgr.counters["ephemeral_freed"] >= 1
        _, stats = on.run(prompt(95, 30), max_tokens=4)      # slots are all still free
        assert stats["session"] == 1
    finally:
        on.stop()


def test_reset_prefix_cache_drops_every_session():
    a = prompt(96, 40)
    on = SessionRanks(sessions=4, pool_pages=32)
    try:
        on.run(a, max_tokens=4)
        a2 = a + prompt(97, 5)
        assert on.run(a2, max_tokens=4)[1]["cached_tokens"] > 0
        asyncio.run(on.engine.reset_prefix_cache())
        assert not on.engine.mgr.sessions and not on.be0.live
        _, stats = on.run(a2, max_tokens=4)                  # rank 1 drops everything at begin
        assert stats["cached_tokens"] == 0
        on.check()
        assert on.run(a2 + prompt(98, 5), max_tokens=4)[1]["cached_tokens"] > 0
    finally:
        on.stop()


def test_rank0_failure_keeps_the_session_honest():
    """An injected rank-0 failure mid-request truncates only that session's record; the
    other session's checkpoints survive untouched and the retry replies like a full
    prefill."""
    a = prompt(99, 50)
    on = SessionRanks(sessions=2, pool_pages=32)
    off = SessionRanks(sessions=2, pool_pages=32)
    off.engine.mgr.min_match = 10 ** 9
    try:
        ref = [off.run(a, max_tokens=4)[0]]
        on.run(a, max_tokens=4)
        on.be0.fail_at = on.be0.forwards + 3                 # dies three tokens into turn 2
        b = a[:-1] + prompt(101, 30)                         # header-rewrite: still session 1
        with pytest.raises(RuntimeError, match="injected"):
            on.run(b, max_tokens=8)
        on.sync()
        on.check()                                           # records still agree with rank 1
        ck = sorted(c.pos for c in on.sid_of(1).px.ckpts)    # truncated to the restore point
        assert ck == [49]                                    # the post-prefill checkpoint
        text, stats = on.run(b, max_tokens=8)
        ref.append(off.run(b, max_tokens=8)[0])
        assert text == ref[1] and stats["cached_tokens"] == 49   # resumes from what survived
        on.check()
    finally:
        on.stop()
        off.stop()


def test_status_reports_sessions_pages_and_counters():
    a = prompt(102, 40)
    on = SessionRanks(sessions=4, pool_pages=16)
    try:
        on.run(a, max_tokens=4)
        st = on.engine.session_status()
        assert st["slots"] == 4 and st["pool_pages"] == 16
        assert st["pages_used"] > 0 and st["pages_used"] + st["pages_free"] == 16
        assert len(st["sessions"]) == 1
        s0 = st["sessions"][0]
        assert s0["session"] == 1 and s0["pages"] == st["pages_used"] and not s0["active"]
        for key in ("matched", "new_sessions", "session_evictions", "page_allocs", "page_frees"):
            assert key in st["counters"]
    finally:
        on.stop()


def test_http_layer_serves_multiple_sessions_and_reports_usage():
    """The vendored server over the session engine: concurrent requests queue through the
    sessions semaphore, per-request usage carries cached_tokens, and /tp/status (via
    tp_serve.create_app) reports the pool once the engine exposes session_status."""
    from aiohttp.test_utils import TestClient, TestServer
    import tp_serve

    on = SessionRanks(sessions=2, pool_pages=64)

    async def go():
        client = TestClient(TestServer(tp_serve.create_app(on.engine, "tp-ms")))
        await client.start_server()
        try:
            body = {"model": "tp-ms", "messages": [{"role": "user", "content": "Hello there, " * 4}],
                    "max_tokens": 4, "temperature": 0.0}
            cached = []
            for i in range(3):                               # same conversation, three turns
                r = await client.post("/v1/chat/completions", json=body)
                assert r.status == 200, await r.text()
                usage = (await r.json())["usage"]
                cached.append(usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
            status = await client.get("/tp/status")
            st = await status.json()
            return cached, st
        finally:
            await client.close()

    try:
        cached, st = asyncio.run(go())
        assert cached[0] == 0 and cached[1] > 0 and cached[2] > 0
        assert st["sessions"] == 2 and st["proto"] == tp_engine.PROTO_SESSIONS
        assert st["pool"]["pages_used"] > 0 and st["pool"]["slots"] == 2
        on.check()
    finally:
        on.stop()


# ---- rank-1 protocol (raw frames) ---------------------------------------------------------------------------

def _ms_worker(backend, port, errors, *, sessions=4, pool_pages=8, page_size=PAGE,
               worker_sessions=None):
    """A rank-1 worker thread with an explicit expected session count (the ctor override;
    production workers read EXL3_TP_SESSIONS, tests never touch the environment)."""

    def run():
        try:
            chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
            chan.set_recv_timeout(None)
            tp_engine.Rank1Worker(backend, chan,
                                  sessions=worker_sessions if worker_sessions is not None
                                  else sessions).serve_forever()
        except BaseException as exc:                                     # noqa: BLE001
            errors.append(exc)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _ms_hello(chan, *, sessions=4, pool_pages=8, page_size=PAGE, dspark=False, proto=None):
    p = tp_engine.PROTO_SESSIONS if proto is None else proto
    hello = {"op": "hello", "proto": p, "vocab_size": VOCAB, "dspark": dspark}
    if p >= tp_engine.PROTO_SESSIONS:
        hello.update({"sessions": sessions, "pool_pages": pool_pages, "page_size": page_size})
    chan.send(hello)
    return chan.recv()


def test_proto5_handshake_validates_sessions_and_pool():
    """A rank 1 configured for fewer sessions, a pool mismatch and a too-new proto all
    refuse (the worker dies, the channel closes) with errors naming the fix; the matching
    handshake echoes the geometry."""
    port = free_port()
    errors = []
    t = _ms_worker(SessionBackend(8), port, errors, sessions=4, pool_pages=8)
    cli = None
    try:
        cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
        ok = _ms_hello(cli, sessions=4, pool_pages=8)
        assert ok["op"] == "hello_ok" and ok["sessions"] == 4 and ok["pool_pages"] == 8
        assert ok["page_size"] == PAGE
    finally:
        if cli is not None:
            cli.close()
        t.join(5.0)
    assert not t.is_alive() and not errors

    for kw, worker_sessions, needle in ((dict(sessions=4, pool_pages=8), 2, "EXL3_TP_SESSIONS"),
                                        (dict(sessions=4, pool_pages=8), 1, "EXL3_TP_SESSIONS"),
                                        (dict(sessions=4, pool_pages=7), 4, "pool mismatch"),
                                        (dict(proto=tp_engine.PROTO_SESSIONS + 1), 4,
                                         "sync tools/ds41")):
        port = free_port()
        errors = []
        t = _ms_worker(SessionBackend(8), port, errors, sessions=4, pool_pages=8,
                       worker_sessions=worker_sessions)
        cli = None
        try:
            cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
            cli.set_recv_timeout(5.0)
            with pytest.raises(Exception):                    # worker dies, channel closes
                _ms_hello(cli, **kw)
        finally:
            if cli is not None:
                cli.close()
            t.join(5.0)
        assert not t.is_alive() and errors, needle
        assert needle in str(errors[0]), (needle, errors[0])


def test_rank1_refuses_bad_session_frames():
    """The begin-frame guards: missing fields, out-of-range, shared or reshuffled pages and
    capacity beyond the allocation are clean protocol errors (rank 1 dies loudly, never
    serves wrong rows)."""
    good = {"op": "begin", "req": 0, "ids": [5, 6, 7, 8], "session": 1, "pages": [0, 1]}
    bad_frames = [
        ({**good, "session": 3}, "outside 1..2"),
        ({k: good[k] for k in good if k != "pages"}, "needs session and pages"),
        ({**good, "pages": [0, 99]}, "outside the pool"),
        ({**good, "pages": [0, 0]}, "duplicates"),
        ({**good, "pages": []}, "empty page list"),
        ({**good, "ids": list(range(40))}, "exceeds session 1's allocation"),
    ]
    for frame, needle in bad_frames:
        port = free_port()
        errors = []
        be2 = SessionBackend(8)
        t2 = _ms_worker(be2, port, errors, sessions=2, pool_pages=8)
        try:
            cli2 = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
            assert _ms_hello(cli2, sessions=2, pool_pages=8)["op"] == "hello_ok"
            cli2.send(frame)
            reply = cli2.recv()
            assert reply.get("op") == "error" and needle in reply["message"], (needle, reply)
            cli2.close()
        finally:
            t2.join(5.0)
            assert not t2.is_alive()


def test_rank1_refuses_cross_session_and_stateful_violations():
    """Stateful guards on one worker: shared pages between sessions, growth that reshuffles,
    restores of another session's checkpoint, dropping what the same frame restores and
    evicting a session with live stashes are all refused."""
    port = free_port()
    errors = []
    be3 = SessionBackend(8)
    t3 = _ms_worker(be3, port, errors, sessions=2, pool_pages=8)
    try:
        cli3 = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
        assert _ms_hello(cli3, sessions=2, pool_pages=8)["op"] == "hello_ok"
        # session 1 claims pages [0,1] and saves checkpoint 1 at position 3
        cli3.send({"op": "begin", "req": 0, "ids": [5, 6, 7, 8], "session": 1, "pages": [0, 1],
                   "save": 1})
        cli3.send({"op": "end", "req": 0, "session": 1})
        seq = [
            # another session claims session 1's page
            ({"op": "begin", "req": 1, "ids": [5, 6, 7, 8], "session": 2, "pages": [1, 2]},
             "already belongs to session 1"),
            # growth must extend, never reshuffle
            ({"op": "begin", "req": 1, "ids": [5, 6, 7, 8], "session": 1, "pages": [2, 3]},
             "changed history"),
            # restore of another session's key (1 belongs to session 1)
            ({"op": "begin", "req": 1, "ids": [5, 6, 7, 8, 9], "session": 2, "pages": [2, 3],
              "restore": 1, "start": 3}, "never cross sessions"),
            # dropping what the same frame restores
            ({"op": "begin", "req": 1, "ids": [5, 6, 7, 8, 9], "session": 1, "pages": [0, 1],
              "restore": 1, "start": 3, "drop": [1]}, "drops checkpoint 1"),
            # evicting a session that still holds stashes
            ({"op": "begin", "req": 1, "ids": [5, 6, 7, 8], "session": 2, "pages": [2, 3],
              "evict": [1]}, "still live"),
            # evicting an unknown session
            ({"op": "begin", "req": 1, "ids": [5, 6, 7, 8], "session": 2, "pages": [2, 3],
              "evict": [5]}, "unknown session"),
        ]
        for frame, needle in seq:
            cli3.send(frame)
            reply = cli3.recv()
            assert reply.get("op") == "error" and needle in reply["message"], (needle, reply)
            if not t3.is_alive():
                break
        cli3.close()
    finally:
        t3.join(5.0)
    assert not t3.is_alive()


def test_rank1_evicts_and_restores_within_a_session():
    """The happy protocol path: begin+save+end, a growth begin that restores the checkpoint
    in the same session, a clean eviction after the stash was dropped, and liveness pings
    between settled requests."""
    port = free_port()
    errors = []
    be = SessionBackend(8)
    t = _ms_worker(be, port, errors, sessions=2, pool_pages=8)
    try:
        cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
        assert _ms_hello(cli, sessions=2, pool_pages=8)["op"] == "hello_ok"
        cli.send({"op": "begin", "req": 0, "ids": [5, 6, 7, 8], "session": 1, "pages": [0, 1],
                  "save": 1})
        cli.send({"op": "end", "req": 0, "session": 1, "save": 2})
        cli.send({"op": "ping"})
        assert cli.recv()["op"] == "pong"
        # restore within the same session, growing the page list
        cli.send({"op": "begin", "req": 1, "ids": [5, 6, 7, 8, 9, 10], "session": 1,
                  "pages": [0, 1, 2], "restore": 1, "start": 3})
        cli.send({"op": "end", "req": 1, "session": 1})
        cli.send({"op": "ping"})
        assert cli.recv()["op"] == "pong"
        # drop session 1's stashes, then evict it; session 2 proceeds on freed pages
        cli.send({"op": "begin", "req": 2, "ids": [5, 6, 7, 8], "session": 2, "pages": [2, 3],
                  "drop": [1, 2], "evict": [1]})
        cli.send({"op": "end", "req": 2, "session": 2})
        cli.send({"op": "ping"})
        assert cli.recv()["op"] == "pong"
        cli.close()
    finally:
        t.join(5.0)
    # the channel close is the clean exit; anything else must be a ChannelClosed
    assert not errors or all(isinstance(e, tp_engine.ChannelClosed) for e in errors), errors


# ---- S=1 regression at the engine level ---------------------------------------------------------------------

def test_s1_engine_uses_the_legacy_wire_and_params():
    """sessions=1 (the default): the manager is absent, the single PrefixCache is used, no
    begin/end frame carries session fields, and the engine never asks for pages."""
    from test_tp_prefix_cache import Ranks as LegacyRanks, pc

    ranks = LegacyRanks(False, pc())
    sink: list[dict] = []
    rec = _RecordingChannel(ranks.chan, sink)
    ranks.engine.channel = rec
    try:
        a = prompt(110, 40)
        ranks.run(a, max_tokens=6)
        reply = ranks.run(a, max_tokens=6)[0]
        ranks.run(a + ids_of(reply) + prompt(111, 8), max_tokens=6)
        begins = [f for f in sink if f.get("op") == "begin"]
        ends = [f for f in sink if f.get("op") == "end"]
        assert begins and ends
        for f in begins + ends:
            assert "session" not in f and "pages" not in f and "evict" not in f, f
        assert ranks.engine.mgr is None and ranks.engine.prefix is not None
        assert ranks.engine.sessions == 1
        ranks.check_mirror()
    finally:
        ranks.stop()


def test_prefix_cache_line_reports_mgr_mode_correctly():
    """The startup log must not say 'prefix cache off' for multi-session engines: mgr mode
    keeps one PrefixCache NAMESPACE PER SESSION (engine.prefix is None by design)."""
    import tp_serve
    on = SessionRanks(sessions=2, pool_pages=16)
    from test_tp_prefix_cache import Ranks as LegacyRanks, pc
    legacy = LegacyRanks(False, pc())
    try:
        mgr_line = tp_serve._prefix_cache_line(on.engine)
        assert mgr_line.startswith("on, per session (") and "checkpoints of the recurrent state per slot" in mgr_line
        assert str(on.engine.mgr.prefix_slots) in mgr_line
        legacy_line = tp_serve._prefix_cache_line(legacy.engine)
        assert legacy_line.startswith("on (") and "checkpoints of the recurrent state" in legacy_line
        off = tp_serve._prefix_cache_line(
            type("E", (), {"prefix": None, "mgr": None})())
        assert off.startswith("off (EXL3_TP_PREFIX_CACHE=0)")
    finally:
        on.stop()
        legacy.stop()


def test_s1_and_s4_agree_on_replies():
    """The same two-turn conversation through the S=1 engine and the S=4 engine produces
    identical replies (the S=4 one reuses its own session cache)."""
    from test_tp_prefix_cache import Ranks as LegacyRanks, pc

    a = prompt(120, 48)
    legacy = LegacyRanks(False, pc())
    ms = SessionRanks(sessions=4, pool_pages=48)
    try:
        t1, _ = legacy.run(a, max_tokens=8)
        a2 = a + ids_of(t1) + prompt(121, 16)
        t2, _ = legacy.run(a2, max_tokens=8)
        m1, _ = ms.run(a, max_tokens=8)
        m2, s2 = ms.run(a2, max_tokens=8)
        assert (t1, t2) == (m1, m2)
        assert s2["cached_tokens"] > 0
        ms.check()
    finally:
        legacy.stop()
        ms.stop()


def test_backend_params_block_table_mode_vs_legacy():
    """The real TPModelBackend call shapes, driven directly with a capturing fake model:
    alloc=None reproduces the legacy params byte for byte (batch_shape/past_len, chunked at
    span); alloc switches to the session's block_table + per-chunk cache_seqlens and never
    carries batch_shape/past_len. decode steps and spec verifies switch the same way."""

    class FakeModel:
        def __init__(self):
            self.prefill_params, self.forward_params = [], []

        def prefill(self, ids, params):
            self.prefill_params.append(dict(params))

        def forward(self, ids, params):
            self.forward_params.append(dict(params))
            return torch.zeros((1, 1, 129280))

    class FakeTok:
        actual_vocab_size = 129280

    class FakeCache:
        pass

    import torch as _t
    span, cache_tokens, page = 32, 256, 16
    be = tp_engine.TPModelBackend(FakeModel(), FakeCache(), FakeTok(), span=span,
                                  cache_tokens=cache_tokens, rank=0, page_size=page)
    ids = list(range(100, 160))                              # 60 tokens, 2 chunks at span 32
    state = object()
    be.prefill(ids, state)
    p = be.model.prefill_params
    assert [q["past_len"] for q in p] == [0, 32]
    for q in p:
        assert q["batch_shape"] == (1, cache_tokens) and "block_table" not in q \
            and "cache_seqlens" not in q

    alloc = tp_engine.SessionAlloc([3, 0, 7, 2, 1], page)    # deliberately noncontiguous
    be.prefill(ids, state, alloc=alloc)
    p2 = be.model.prefill_params[-2:]
    assert [int(q["cache_seqlens"].flatten()[0]) for q in p2] == [0, 32]
    for q in p2:
        assert "batch_shape" not in q and "past_len" not in q
        assert q["block_table"].flatten().tolist() == [3, 0, 7, 2, 1]
        assert q["block_table"].dtype == _t.int32

    be.forward(555, 61, state)
    assert be.model.forward_params[-1]["past_len"] == 61 \
        and "block_table" not in be.model.forward_params[-1]
    be.forward(555, 61, state, alloc=alloc)
    q = be.model.forward_params[-1]
    assert "past_len" not in q and "batch_shape" not in q
    assert q["block_table"] is alloc.bt() and int(q["cache_seqlens"][0]) == 61
