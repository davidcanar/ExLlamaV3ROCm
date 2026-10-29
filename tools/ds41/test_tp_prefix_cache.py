"""Prompt reuse across requests (TPEngine's prefix cache) over a loopback control channel.

The stand-in backends model the cache semantics the reuse depends on: `kv` maps position -> token
and survives requests (the position-indexed DSA pools), the state carries a running hash of the
tokens it consumed (the recurrent part only a checkpoint brings back), verify passes leave rows of
rejected drafts behind, and the greedy token at every step depends on the hash AND on every kv row
below it. A restore with the wrong state, at the wrong position or over overwritten rows changes
the output, so each scenario runs twice -- prefix cache on, and off (full prefill every request)
-- and the replies must be identical. The spec stand-in also refuses to draft over drafter rows
that do not match the trunk (the engine's `draft_ids` bookkeeping must rule that out). CPU only.
"""
from __future__ import annotations

import asyncio
import random
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_engine  # noqa: E402
from ds41_engine import ByteTokenizer  # noqa: E402
from test_tp_control import free_port  # noqa: E402

VOCAB = 129280
MIRRORED = ("new_state", "restore", "prefill", "forward", "verify", "rewind", "stash", "drop_stash",
            "release_state")


class Tok(ByteTokenizer):
    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return bytes(i if i < 256 else 63 for i in ids).decode("utf-8", "replace")


class CacheBackend:
    """Plain stand-in with position-indexed kv and a hash-chain recurrent state."""

    dspark = False
    vocab_size = VOCAB
    span = 64

    def __init__(self):
        self.kv: dict[int, int] = {}
        self.calls: list[tuple] = []
        self.live: dict[int, dict] = {}
        self.states = self.released = 0
        self.forwards = 0
        self.fail_at: int | None = None          # raise in the forward with this ordinal
        self._st = None

    # -- state and checkpoints
    def new_state(self):
        self.states += 1
        self.calls.append(("new_state",))
        self._st = {"pos": 0, "hist": [0]}       # hist[p]: hash of the tokens at positions [0, p)
        return self._st

    def restore(self, stash, position):
        self.states += 1
        self.calls.append(("restore", int(position)))
        assert self.live.get(id(stash)) is stash, "restore of a dropped checkpoint"
        assert stash["pos"] == position
        self._st = {"pos": stash["pos"], "hist": list(stash["hist"])}
        return self._st

    def stash(self, state):
        self.calls.append(("stash", state["pos"]))
        s = {"pos": state["pos"], "hist": list(state["hist"][: state["pos"] + 1])}
        self.live[id(s)] = s
        return s

    def drop_stash(self, stash):
        self.calls.append(("drop_stash", stash["pos"]))
        assert self.live.pop(id(stash), None) is stash, "drop of an unknown checkpoint"

    def release_state(self, state):
        self.released += 1
        self.calls.append(("release_state",))

    # -- the model
    def _feed(self, state, token):
        p = state["pos"]
        self.kv[p] = int(token)
        del state["hist"][p + 1:]
        state["hist"].append((state["hist"][p] * 1000003 + int(token) + 1) % (1 << 61))
        state["pos"] = p + 1

    def _next(self, state) -> int:
        p = state["pos"]
        acc = state["hist"][p]
        for i in range(p):                       # every kv row below p (KeyError: never written)
            acc = (acc * 31 + self.kv[i]) % (1 << 61)
        return 32 + acc % 95                     # printable ASCII, never the eos id

    def _row(self, tok):
        row = torch.zeros(self.vocab_size)
        row[tok] = 8.0
        return row

    def prefill(self, ids, state, spec=False, start=0):
        self.calls.append(("prefill", len(ids), bool(spec), int(start)))
        assert state["pos"] == start, (state["pos"], start)
        for t in ids[start: len(ids) - 1]:
            self._feed(state, t)
        return {}

    def forward(self, token, position, state):
        self.calls.append(("forward", int(token), int(position)))
        self.forwards += 1
        if self.fail_at is not None and self.forwards == self.fail_at:
            raise RuntimeError("injected rank-0 failure")
        assert state["pos"] == position
        self._feed(state, token)
        return self._row(self._next(state))


class CacheSpecBackend(CacheBackend):
    """Spec stand-in: verify writes every window row (rejected drafts leave garbage behind), the
    drafter has its own position-indexed rows, primed by spec prefills and refreshed per round."""

    dspark = True
    dspark_block = 5
    export_layers = frozenset({36, 37, 38})

    def __init__(self):
        super().__init__()
        self.drows: dict[int, int] = {}
        self.rounds = 0

    def prefill(self, ids, state, spec=False, start=0):
        out = super().prefill(ids, state, spec, start)
        if spec:
            for p in range(start, len(ids) - 1):
                self.drows[p] = self.kv[p]
        return out

    def draft(self, anchor, position, k_cap):
        self.calls.append(("draft", int(anchor), int(position), int(k_cap)))
        stale = [i for i in range(position) if self.drows.get(i) != self.kv.get(i)]
        assert not stale, f"draft at {position} over stale drafter rows {stale[:5]}"
        k = max(0, min(self.dspark_block, int(k_cap)))
        good = self.rounds % (k + 1) if k else 0          # the first `good` drafts are right
        self.rounds += 1
        saved = {i: self.kv.get(i) for i in range(position, position + k + 1)}
        sim = {"pos": position, "hist": list(self._st["hist"][: position + 1])}
        drafts, tok = [], int(anchor)
        for i in range(k):
            self._feed(sim, tok)
            tok = self._next(sim)
            drafts.append(tok if i < good else 32 + (tok - 31) % 95)
            tok = drafts[-1]
        for i, v in saved.items():                        # drafting leaves the trunk rows alone
            if v is None:
                self.kv.pop(i, None)
            else:
                self.kv[i] = v
        return k, drafts

    def verify(self, window, position, state):
        self.calls.append(("verify", tuple(int(t) for t in window), int(position)))
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
            self.drows[p] = self.kv[p]


class Ranks:
    """Rank-0 engine + the real Rank1Worker over a loopback channel."""

    def __init__(self, spec: bool, prefix):
        cls = CacheSpecBackend if spec else CacheBackend
        self.be0, self.be1 = cls(), cls()
        self.errors: list[BaseException] = []
        port = free_port()

        def rank1():
            try:
                chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=30.0)
                chan.set_recv_timeout(None)
                self.worker = tp_engine.Rank1Worker(self.be1, chan, log=lambda *_: None)
                self.worker.serve_forever()
            except BaseException as exc:                                  # noqa: BLE001
                self.errors.append(exc)

        self.thread = threading.Thread(target=rank1, daemon=True)
        self.thread.start()
        self.chan = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=30.0)
        self.chan.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": VOCAB, "dspark": spec})
        assert self.chan.recv()["op"] == "hello_ok"
        self.engine = tp_engine.TPEngine(self.be0, self.chan, Tok(), ctx=8192, eos_id=None,
                                         prefix=prefix, log=lambda *_: None)

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
            if self.be1.released >= self.be0.released and len(self.mirrored(self.be1)) == len(self.mirrored(self.be0)):
                return
            time.sleep(0.005)
        raise AssertionError(f"rank 1 did not catch up: {self.errors}")

    @staticmethod
    def mirrored(be):
        return [c for c in be.calls if c[0] in MIRRORED]

    def check_mirror(self):
        assert self.mirrored(self.be0) == self.mirrored(self.be1)
        px = self.engine.prefix
        n = len(px.ckpts) if px is not None else 0
        assert len(self.be0.live) == len(self.be1.live) == len(self.worker.stashes) == n
        if px is not None:
            assert sorted(self.worker.stashes) == sorted(c.key for c in px.ckpts)
            assert sorted(p for p, _ in self.worker.stashes.values()) == sorted(c.pos for c in px.ckpts)

    def stop(self):
        self.engine.close()
        self.thread.join(timeout=10.0)
        assert not self.thread.is_alive() and not self.errors, self.errors


def prompt(seed: int, n: int) -> list[int]:
    rnd = random.Random(seed)
    return [rnd.randrange(32, 127) for _ in range(n)]


def pc(**kw):
    base = {"slots": 4, "min_tokens": 16, "save_end": True}
    base.update(kw)
    return tp_engine.PrefixCache(**base)


def both(spec, script, prefix_kw=None):
    """Run `script(ranks)` -> list of (text, stats) with the prefix cache on and off."""
    on = Ranks(spec, pc(**(prefix_kw or {})))
    off = Ranks(spec, False)
    try:
        got_on = script(on)
        on.check_mirror()
        got_off = script(off)
        off.check_mirror()
    finally:
        on.stop()
        off.stop()
    assert [t for t, _ in got_on] == [t for t, _ in got_off], "prefix reuse changed a reply"
    assert all(s["cached_tokens"] == 0 for _, s in got_off)
    return got_on


def ids_of(text: str) -> list[int]:
    return list(text.encode("utf-8"))


@pytest.mark.parametrize("spec", [False, True])
def test_multi_turn_chat_reuses_the_prompt_and_matches_full_prefill(spec):
    a = prompt(1, 60)

    def script(r):
        out = [r.run(a)]
        # next turn: the template rewrites the generation header (the last prompt token) and
        # appends the reply and a new user turn
        b = a[:-1] + [7] + ids_of(out[0][0]) + prompt(2, 30)      # 7: never a prompt() token
        out.append(r.run(b))
        out.append(r.run(b))                              # a retry of the same turn
        return out

    got = both(spec, script)
    assert got[0][1]["cached_tokens"] == 0
    assert got[1][1]["cached_tokens"] == 59              # everything up to the rewritten header
    assert got[2][1]["cached_tokens"] == 60 - 1 + 1 + 0 + len(got[0][0]) + 30 - 1   # all but the last token


@pytest.mark.parametrize("spec", [False, True])
def test_verbatim_continuation_reuses_the_reply_through_the_end_checkpoint(spec):
    a = prompt(3, 50)

    def script(r):
        out = [r.run(a, max_tokens=14)]
        # tool-style turn: the reply comes back verbatim, then new tokens
        nxt = a + ids_of(out[0][0]) + prompt(4, 25)
        out.append(r.run(nxt))
        return out

    got = both(spec, script)
    reply = len(got[0][0])
    # plain: the last emitted token was never fed (max_tokens cut), so the end checkpoint covers
    # the prompt and all but one reply token; spec rewinds to the emitted boundary, which is the
    # same position (or the full reply when the final bonus token was the cut)
    assert got[1][1]["cached_tokens"] in (50 + reply - 1, 50 + reply)


@pytest.mark.parametrize("spec", [False, True])
def test_divergent_request_invalidates_and_reuse_resumes(spec):
    a, c = prompt(5, 70), prompt(6, 40)

    def script(r):
        out = [r.run(a)]
        out.append(r.run(c))                              # an unrelated request overwrites the cache
        out.append(r.run(a))                              # back to the first conversation
        out.append(r.run(a))
        return out

    got = both(spec, script)
    assert got[1][1]["cached_tokens"] == 0
    assert got[2][1]["cached_tokens"] == 0                # a's checkpoints died with c's writes
    assert got[3][1]["cached_tokens"] == 69


def test_shared_prefix_then_divergence_keeps_the_shorter_checkpoint_valid():
    sys_prompt = prompt(7, 40)
    a = sys_prompt + prompt(8, 30)
    b = sys_prompt + prompt(9, 30)

    def script(r):
        out = [r.run(sys_prompt + [ord("!")])]           # checkpoint at 40: the shared system prompt
        out.append(r.run(a))                               # reuses 40
        out.append(r.run(b))                               # still reuses 40 (a's tail diverges)
        out.append(r.run(a))
        return out

    got = both(False, script)
    assert [s["cached_tokens"] for _, s in got] == [0, 40, 40, 40]


def test_spec_request_after_plain_turns_only_reuses_drafter_valid_rows():
    """A plain request writes trunk rows but not drafter rows: a later spec request may reuse
    the trunk only as far as the drafter's rows still match (CacheSpecBackend asserts it)."""
    a = prompt(10, 60)
    c_len = []

    def script(r):
        out = [r.run(a, speculative=True)]
        b = a + ids_of(out[0][0]) + prompt(11, 20)
        out.append(r.run(b, speculative=False))           # plain: reuses the spec request's rows
        c = b + ids_of(out[1][0]) + prompt(12, 20)
        c_len.append(len(c))
        out.append(r.run(c, speculative=True))            # drafter rows only cover a's sequence
        out.append(r.run(c, speculative=False))           # the trunk could go further
        return out

    got = both(True, script)
    assert got[1][1]["cached_tokens"] >= 60
    spec_reuse, plain_reuse = got[2][1]["cached_tokens"], got[3][1]["cached_tokens"]
    assert spec_reuse <= 60 + len(got[0][0])              # never past the spec request's rows
    assert plain_reuse == c_len[0] - 1


def test_cache_prompt_false_skips_reuse_but_keeps_the_record_honest():
    a, c = prompt(13, 50), prompt(14, 50)

    def script(r):
        out = [r.run(a)]
        out.append(r.run(a, cache_prompt=False))          # no restore, no save
        out.append(r.run(c, cache_prompt=False))          # overwrites a's rows: must invalidate
        out.append(r.run(a))
        return out

    got = both(False, script)
    assert [s["cached_tokens"] for _, s in got] == [0, 0, 0, 0]


def test_slot_eviction_is_lru_and_mirrored():
    lens = []

    def script(r):
        out, ids = [], prompt(15, 30)
        lens.clear()
        for i in range(5):                                 # a growing conversation, verbatim replies
            lens.append(len(ids))
            text, st = r.run(ids, max_tokens=4)
            out.append((text, st))
            if r.engine.prefix is not None:
                assert len(r.engine.prefix.ckpts) <= 2
                assert len(r.worker.stashes) <= 2
            ids = ids + ids_of(text) + prompt(100 + i, 10)
        return out

    got = both(False, script, {"slots": 2})
    # each turn resumes from the previous end checkpoint: prompt + reply minus the unfed last token
    assert [s["cached_tokens"] for _, s in got] == [0] + [n + 4 - 1 for n in lens[:-1]]


def test_short_prompts_take_no_checkpoint():
    def script(r):
        out = [r.run(prompt(16, 10), max_tokens=3), r.run(prompt(16, 10), max_tokens=3)]
        assert r.engine.prefix is None or not r.engine.prefix.ckpts
        return out

    got = both(False, script)
    assert [s["cached_tokens"] for _, s in got] == [0, 0]


def test_rank0_failure_mid_request_forgets_what_it_may_have_written():
    a = prompt(17, 60)
    on = Ranks(False, pc())
    off = Ranks(False, False)
    try:
        ref = [off.run(a)[0]]
        on.run(a)
        on.be0.fail_at = on.be0.forwards + 3              # dies three tokens into the next request
        b = a[:-1] + prompt(18, 40)
        with pytest.raises(RuntimeError, match="injected"):
            on.run(b)
        on.sync()
        assert all(c.pos <= 59 for c in on.engine.prefix.ckpts)
        on.check_mirror()
        text, stats = on.run(b)
        ref.append(off.run(b)[0])
        assert text == ref[1] and stats["cached_tokens"] == 59
        on.check_mirror()
    finally:
        on.stop()
        off.stop()


def test_rank1_refuses_bad_prefix_frames():
    from test_tp_control import _hello, _worker_thread

    for frame, needle in (
            ({"op": "begin", "req": 0, "ids": [5, 6, 7], "restore": 9, "start": 1}, "unknown checkpoint 9"),
            ({"op": "begin", "req": 0, "ids": [5, 6, 7], "start": 1}, "no checkpoint to restore")):
        port = free_port()
        errors = []
        t = _worker_thread(CacheBackend(), port, errors)
        cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
        assert _hello(cli)["op"] == "hello_ok"
        cli.send(frame)
        assert needle in cli.recv()["message"]
        t.join(10.0)
        assert not t.is_alive()
        cli.close()


def test_reset_prefix_cache_forgets_everything_on_both_ranks():
    a = prompt(19, 60)
    on = Ranks(False, pc())
    try:
        on.run(a)
        assert on.run(a)[1]["cached_tokens"] == 59
        asyncio.run(on.engine.reset_prefix_cache())
        assert not on.engine.prefix.ckpts and not on.be0.live
        _, stats = on.run(a)                              # rank 1 drops its copies at this begin
        assert stats["cached_tokens"] == 0
        on.check_mirror()
        assert on.run(a)[1]["cached_tokens"] == 59        # and reuse works again afterwards
        on.check_mirror()
    finally:
        on.stop()


def test_cache_salt_over_http_isolates_namespaces():
    """The vendored server takes cache_salt once the engine can reset its prefix cache: the
    same salt reuses, a new salt starts from nothing."""
    from aiohttp.test_utils import TestClient, TestServer
    import ds41_testkit as kit
    serve = kit.serve_module()
    on = Ranks(False, pc())

    async def go():
        client = TestClient(TestServer(serve.create_app(on.engine, "tp-test")))
        await client.start_server()
        try:
            body = {"model": "tp-test", "messages": [{"role": "user", "content": "Hello there, " * 4}],
                    "max_tokens": 4, "temperature": 0.0}
            cached = []
            for salt in ("u1", "u1", "u2", "u2"):
                r = await client.post("/v1/chat/completions", json={**body, "cache_salt": salt})
                assert r.status == 200, await r.text()
                cached.append((await r.json())["usage"]["prompt_tokens_details"]["cached_tokens"])
            return cached
        finally:
            await client.close()

    try:
        cached = asyncio.run(go())
        assert cached[0] == 0 and cached[1] > 0 and cached[2] == 0 and cached[3] == cached[1]
        on.sync()
        on.check_mirror()
    finally:
        on.stop()


def test_common_prefix_helper():
    from array import array
    a = array("i", range(10000))
    assert tp_engine._common_prefix(a, a) == 10000
    b = array("i", range(10000))
    b[5000] = -1
    assert tp_engine._common_prefix(a, b) == 5000
    assert tp_engine._common_prefix(a, array("i", range(17))) == 17
    assert tp_engine._common_prefix(array("i"), a) == 0
