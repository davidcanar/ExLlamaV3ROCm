"""Two-node TP engine and control protocol over a loopback channel: framing, lockstep, per-request
state reset, max_tokens / stop / cancel at token boundaries, and the TP engine behind the vendored
HTTP layer. CPU only: a scripted backend stands in for the model, but the control channel is a real
TCP socket and the rank-1 worker is the real one Box2 runs."""
from __future__ import annotations

import asyncio
import json
import socket as sock_mod
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds41_testkit as kit  # noqa: E402
import tp_engine  # noqa: E402
from ds41_engine import ByteTokenizer  # noqa: E402

serve = kit.serve_module()
MODEL = "tp-test"


class DetokByteTokenizer(ByteTokenizer):
    """ByteTokenizer (one id per UTF-8 byte, specials at 256+) that can also decode, for _Detok."""

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return bytes(i if i < 256 else 63 for i in ids).decode("utf-8", "replace")


class ScriptedBackend:
    """CPU stand-in for tp_engine.TPModelBackend: greedy logits from a token script, every call recorded."""

    dspark = False                        # spec ops are refused on plain backends, as on the real one

    def __init__(self, script, vocab_size=129280):
        self.script = [int(t) for t in script]
        self.vocab_size = int(vocab_size)
        self.calls = []
        self.states = 0
        self.released = 0
        self._k = 0

    def new_state(self):
        self.states += 1
        self.calls.append(("new_state",))
        self._k = 0                     # the script is the reply of EACH request
        return f"state-{self.states}"

    def release_state(self, state):
        self.released += 1
        self.calls.append(("release_state",))

    def prefill(self, ids, state, spec=False):
        self.calls.append(("prefill", tuple(int(t) for t in ids), bool(spec)))
        return {}

    def forward(self, token, position, state):
        self.calls.append(("forward", int(token), int(position)))
        tok = self.script[self._k] if self._k < len(self.script) else self.script[-1]
        self._k += 1
        row = torch.zeros(self.vocab_size)
        row[tok % self.vocab_size] = 8.0
        return row


class ScriptedSpecBackend(ScriptedBackend):
    """Spec-mode stand-in. The base script is the target's greedy continuation t_1..t_n; drafts
    follow it except where draft_wrong says so ((round_index, draft_index), both 0-based). The
    verify rows are one-hot at the true continuation, so the engine's greedy acceptance computes
    a = the first wrong draft (or k), and the emitted stream equals the plain greedy script no
    matter how wrong the drafts are -- the property the real DSpark path guarantees."""

    dspark = True
    export_layers = frozenset({36, 37, 38})
    dspark_block = 5

    def __init__(self, script, draft_wrong=(), conf_len=5, vocab_size=129280):
        super().__init__(script, vocab_size)
        self.draft_wrong = {(int(r), int(i)) for r, i in draft_wrong}
        self.conf_len = int(conf_len)
        self.round_log = []              # (k, a) per round, as the engine's acceptance computes it
        self._round = 0
        self._next = 0                   # index of the next greedy token to be emitted
        self._drafts = []                # set by draft(); stays empty on the rank-1 side

    def new_state(self):
        out = super().new_state()
        self._round = 0
        self._next = 0
        self.round_log = []
        self._drafts = []
        return out

    def prefill(self, ids, state, spec=False):
        self.calls.append(("prefill", tuple(int(t) for t in ids), bool(spec)))
        return {"export_states": ["tap0", "tap1", "tap2"]} if spec else {}

    def draft(self, anchor, position, k_cap):
        self.calls.append(("draft", int(anchor), int(position), int(k_cap)))
        k = max(0, min(self.conf_len, self.dspark_block, int(k_cap)))
        drafts = []
        for i in range(k):
            correct = self.script[(self._next + i) % len(self.script)]
            drafts.append((correct + 1) % self.vocab_size if (self._round, i) in self.draft_wrong else correct)
        self._drafts = drafts
        return k, list(drafts)

    def verify(self, window, position, state):
        self.calls.append(("verify", tuple(int(t) for t in window), int(position)))
        rows = torch.zeros(len(window), self.vocab_size)
        for i in range(len(window)):
            rows[i][self.script[(self._next + i) % len(self.script)]] = 8.0
        a = 0
        while a < len(self._drafts) and self._drafts[a] == int(rows[a].argmax()):
            a += 1
        self.round_log.append((len(window) - 1, a))
        self._round += 1
        self._next += a + 1
        return rows, {"export_states": ["tap0", "tap1", "tap2"]}

    def rewind(self, state, steps):
        self.calls.append(("rewind", int(steps)))

    def draft_prefill_kv(self, taps):
        self.calls.append(("draft_prefill_kv",))

    def refresh_draft_kv(self, taps, accepted, position):
        self.calls.append(("refresh", int(accepted), int(position)))


def free_port() -> int:
    s = sock_mod.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def ids_of(text: str) -> list[int]:
    return list(text.encode("utf-8"))


class Pair:
    """Rank-0 engine + rank-1 worker over a real loopback TCP control channel."""

    def __init__(self, script, *, eos_id=1, ctx=512, spec=False, draft_wrong=(), conf_len=5):
        self.spec = spec
        if spec:
            self.be0 = ScriptedSpecBackend(script, draft_wrong=draft_wrong, conf_len=conf_len)
            self.be1 = ScriptedSpecBackend(script, draft_wrong=draft_wrong, conf_len=conf_len)
        else:
            self.be0 = ScriptedBackend(script)
            self.be1 = ScriptedBackend(script)
        self.port = free_port()
        self.error: list[BaseException] = []
        self.thread = None
        self.chan0 = None
        self.engine = None
        self.eos_id = eos_id
        self.ctx = ctx

    def start(self):
        def rank1():
            try:
                chan1 = tp_engine.ControlChannel.connect("127.0.0.1", self.port, deadline_s=30.0)
                chan1.set_recv_timeout(None)
                tp_engine.Rank1Worker(self.be1, chan1).serve_forever()
            except BaseException as exc:                              # noqa: BLE001
                self.error.append(exc)

        self.thread = threading.Thread(target=rank1, daemon=True)
        self.thread.start()
        self.chan0 = tp_engine.ControlChannel.listen("127.0.0.1", self.port, accept_timeout=30.0)
        self.chan0.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": self.be0.vocab_size,
                         "dspark": bool(self.be0.dspark)})
        ok = self.chan0.recv()
        assert ok.get("op") == "hello_ok" and ok.get("rank") == 1 and ok.get("vocab_size") == self.be0.vocab_size
        assert ok.get("dspark") == bool(self.be0.dspark)
        self.engine = tp_engine.TPEngine(self.be0, self.chan0, DetokByteTokenizer(),
                                         ctx=self.ctx, eos_id=self.eos_id)

    def generate(self, ids, **kw):
        async def go():
            out = []
            async for piece in self.engine.generate(
                    "", prompt_ids=list(ids), max_tokens=kw.pop("max_tokens", 64),
                    temperature=kw.pop("temperature", 0.0), top_p=1.0,
                    stop=kw.pop("stop", []),
                    speculative=kw.pop("speculative", None), spec_k=kw.pop("spec_k", None), **kw):
                out.append(piece)
            return out
        out = asyncio.run(go())
        self.sync_rank1()
        return out

    def sync_rank1(self, timeout: float = 5.0):
        """Wait until rank 1 has processed the trailing step/round/end frames (it mirrors asynchronously)."""
        def mirrors(b):
            return sum(1 for c in b.calls if c[0] in ("forward", "verify"))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if mirrors(self.be1) == mirrors(self.be0) and self.be1.released >= self.be0.released:
                return
            time.sleep(0.005)
        raise AssertionError(f"rank 1 did not catch up with rank 0: {self.error}")

    def stop(self):
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:                                         # noqa: BLE001
                pass
        if self.thread is not None:
            self.thread.join(timeout=10.0)
            assert not self.thread.is_alive(), f"rank-1 thread stuck: {self.error}"


def mirrored(b):
    """The calls that must be identical on both ranks (drafting and kv refresh are rank-0 only)."""
    return [c for c in b.calls if c[0] in ("new_state", "prefill", "forward", "verify", "rewind", "release_state")]


# ---- the framed channel -----------------------------------------------------------------------------------


def test_framing_roundtrip_unicode_and_large_payload():
    port = free_port()
    got = {}

    def server_side():
        chan = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
        got["msg"] = chan.recv()
        chan.send({"op": "hello_ok", "unicode": "｜DSML｜ </think> <｜end▁of▁sentence｜>",
                   "ids": list(range(300))})
        chan.close()

    t = threading.Thread(target=server_side, daemon=True)
    t.start()
    cli = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
    big = {"op": "hello", "x": "é" * 20000}
    cli.send(big)
    reply = cli.recv()
    t.join(10.0)
    assert got["msg"] == big                                   # byte-exact JSON round trip
    assert reply["op"] == "hello_ok" and reply["unicode"] == "｜DSML｜ </think> <｜end▁of▁sentence｜>"
    assert reply["ids"][:5] == [0, 1, 2, 3, 4] and len(reply["ids"]) == 300
    cli.close()


def test_ping_pong_over_the_channel():
    port = free_port()
    seen = {}

    def server_side():
        chan = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
        while True:
            msg = chan.recv()
            seen[msg["op"]] = True
            if msg["op"] == "shutdown":
                chan.close()
                return
            if msg["op"] == "ping":
                chan.send({"op": "pong"})

    t = threading.Thread(target=server_side, daemon=True)
    t.start()
    cli = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
    assert cli.request({"op": "ping"}, timeout=5.0) == {"op": "pong"}
    cli.send({"op": "shutdown"})
    t.join(10.0)
    assert seen == {"ping": True, "shutdown": True}
    cli.close()


# ---- lockstep: rank 1 mirrors rank 0 exactly ---------------------------------------------------------------


def test_lockstep_rank1_mirrors_rank0_exactly():
    script = ids_of("Hello from the two-node engine!")
    prompt = ids_of("prompt tokens here")
    pair = Pair(script)
    try:
        pair.start()
        pieces = pair.generate(prompt, max_tokens=len(script))
        assert "".join(pieces) == "Hello from the two-node engine!"
        stats = pair.engine.last_stats
        assert stats["new_tokens"] == len(script) and stats["eos_reason"] == "max_new_tokens"
        assert stats["prompt_tokens"] == len(prompt) and stats["eos"] is False
        # THE lockstep guarantee: identical model calls, same tokens at the same positions
        assert [c for c in pair.be0.calls if c[0] == "forward"] == [c for c in pair.be1.calls if c[0] == "forward"]
        assert [c for c in pair.be0.calls if c[0] == "prefill"] == [c for c in pair.be1.calls if c[0] == "prefill"]
        f0 = [c for c in pair.be0.calls if c[0] == "forward"]
        assert f0[0] == ("forward", prompt[-1], len(prompt) - 1)      # first decode feeds the prompt tail
        assert pair.be0.states == pair.be1.states == 1
        assert pair.be0.released == pair.be1.released == 1
    finally:
        pair.stop()


def test_per_request_state_reset_over_multiple_requests():
    script = ids_of("abcde")
    pair = Pair(script)
    try:
        pair.start()
        prompt = ids_of("user turn")
        for _ in range(3):
            assert "".join(pair.generate(prompt, max_tokens=3)) == "abc"
        f0 = [c for c in pair.be0.calls if c[0] == "forward"]
        f1 = [c for c in pair.be1.calls if c[0] == "forward"]
        assert f0 == f1 and len(f0) == 9
        assert f0[0] == f0[3] == f0[6]                               # every request restarts at the prompt tail
        assert pair.be0.states == pair.be0.released == pair.be1.states == pair.be1.released == 3
    finally:
        pair.stop()


# ---- stop conditions at token boundaries -------------------------------------------------------------------


def test_eos_stops_and_is_never_emitted():
    script = ids_of("one two") + [1]                                 # eos id after the text
    pair = Pair(script, eos_id=1)
    try:
        pair.start()
        assert "".join(pair.generate(ids_of("p!"))) == "one two"
        st = pair.engine.last_stats
        assert st["eos"] is True and st["eos_reason"] == "eos"
        assert st["new_tokens"] == len("one two")
        # both ranks ran the eos-sampling forward, neither fed the eos token afterwards
        assert [c for c in pair.be0.calls if c[0] == "forward"] == [c for c in pair.be1.calls if c[0] == "forward"]
        assert len([c for c in pair.be0.calls if c[0] == "forward"]) == len(script)
    finally:
        pair.stop()


def test_max_tokens_is_exact_on_both_ranks():
    script = ids_of("0123456789")
    pair = Pair(script)
    try:
        pair.start()
        assert "".join(pair.generate(ids_of("p!"), max_tokens=4)) == "0123"
        st = pair.engine.last_stats
        assert st["new_tokens"] == 4 and st["eos_reason"] == "max_new_tokens"
        assert len([c for c in pair.be0.calls if c[0] == "forward"]) == 4
        assert len([c for c in pair.be1.calls if c[0] == "forward"]) == 4
    finally:
        pair.stop()


def test_stop_string_ends_generation_at_the_token_boundary():
    script = ids_of("alpha beta STOP gamma")
    pair = Pair(script, eos_id=1)
    try:
        pair.start()
        text = "".join(pair.generate(ids_of("p!"), max_tokens=64, stop=["STOP"]))
        assert text.startswith("alpha beta ") and text.endswith("STOP") and "gamma" not in text
        st = pair.engine.last_stats
        assert st["eos_reason"] == "stop_word" and st["new_tokens"] == len("alpha beta STOP")
        assert [c for c in pair.be0.calls if c[0] == "forward"] == [c for c in pair.be1.calls if c[0] == "forward"]
    finally:
        pair.stop()


def test_cancel_stops_at_a_token_boundary_and_rank1_stays_usable():
    script = ids_of("0123456789" * 10)
    pair = Pair(script)
    try:
        pair.start()

        async def go():
            out, cancel = [], asyncio.Event()
            gen = pair.engine.generate("", prompt_ids=ids_of("p!"), max_tokens=100,
                                       temperature=0.0, top_p=1.0, stop=[], cancel=cancel)
            async for piece in gen:
                out.append(piece)
                if len(out) >= 3:
                    cancel.set()
            return out

        out = asyncio.run(go())
        assert 3 <= len(out) < 100                                   # stopped at a boundary, not at 100
        assert pair.engine.last_stats["eos_reason"] == "cancelled"
        assert pair.be0.released == pair.be0.states == 1             # state was cleaned up
        # the pair still serves the next request
        assert "".join(pair.generate(ids_of("q!"), max_tokens=5)) == "01234"
        assert [c for c in pair.be0.calls if c[0] == "forward"] == [c for c in pair.be1.calls if c[0] == "forward"]
        assert pair.be1.released == pair.be1.states == 2
    finally:
        pair.stop()


# ---- liveness ----------------------------------------------------------------------------------------------


def test_rank1_ping_ok_and_busy():
    pair = Pair(ids_of("xy"))
    try:
        pair.start()
        assert pair.engine.rank1_ping() == "ok"
        with pair.engine._busy:
            assert pair.engine.rank1_ping() == "busy"                # never pings mid-request
        assert pair.engine.rank1_ping() == "ok"
    finally:
        pair.stop()


def test_rank1_rejects_bad_frames():
    port = free_port()
    failures = []

    def rank1():
        try:
            chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
            tp_engine.Rank1Worker(ScriptedBackend([1, 2, 3]), chan).serve_forever()
        except tp_engine.ProtocolError as exc:
            failures.append(str(exc))

    t = threading.Thread(target=rank1, daemon=True)
    t.start()
    chan = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    chan.send({"op": "hello", "proto": tp_engine.PROTO})
    assert chan.recv()["op"] == "hello_ok"
    chan.send({"op": "nonsense"})                                    # unknown op -> error frame, worker dies
    reply = chan.recv()
    assert reply["op"] == "error" and "unknown op" in reply["message"]
    t.join(10.0)
    assert not t.is_alive()
    chan.close()


# ---- regression: cross-host TP returns no logits on non-output ranks ---------------------------------------


class _FakeCoreModel:
    """Stand-in for the exllamav3 model: records the params, returns a canned result."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def forward(self, ids, params=None):
        self.calls.append(params or {})
        return self.result


class _FakeCache:
    pass


class _FakeTokenizer:
    actual_vocab_size = 32


def test_tp_backend_forward_slices_on_output_rank_and_returns_none_on_rank1():
    """The warm-up crash: cross-host model.forward returns None on rank 1 by design; the backend
    must not touch it there (TypeError -> dead worker -> RCCL timeout on rank 0)."""
    import torch
    logits = torch.arange(64, dtype=torch.float32).view(1, 1, 64)     # padded 64, real vocab 32
    core0 = _FakeCoreModel(logits)
    r0 = tp_engine.TPModelBackend(core0, _FakeCache(), _FakeTokenizer(),
                                  span=64, cache_tokens=128, rank=0)
    row = r0.forward(7, 3, "state")
    assert row.tolist() == [float(i) for i in range(32)]              # rank 0 crops to the real vocab
    assert core0.calls[0]["past_len"] == 3 and core0.calls[0]["last_tokens_only"] == 1

    core1 = _FakeCoreModel(None)                                      # what rank 1's forward actually gets
    r1 = tp_engine.TPModelBackend(core1, _FakeCache(), _FakeTokenizer(),
                                  span=64, cache_tokens=128, rank=1)
    assert r1.forward(7, 3, "state") is None                          # used to raise TypeError
    assert core1.calls[0]["past_len"] == 3                            # the model call itself still ran


def test_rank1_worker_with_none_returning_backend():
    """End to end over the real control channel: the mirror never reads the forward result, so a
    backend that returns None on every step keeps serving in lockstep with rank 0."""
    script = ids_of("mirror ok")
    pair = Pair(script)
    inner = pair.be1.forward
    pair.be1.forward = lambda token, pos, st: (inner(token, pos, st), None)[1]
    try:
        pair.start()
        assert "".join(pair.generate(ids_of("hi!"), max_tokens=9)) == "mirror ok"
        assert [c for c in pair.be0.calls if c[0] == "forward"] == [c for c in pair.be1.calls if c[0] == "forward"]
        assert pair.be1.states == pair.be1.released == 1
    finally:
        pair.stop()


# ---- DSpark speculative rounds ------------------------------------------------------------------------------


SPEC_TEXT = "the quick speculative fox jumps over"


def expected_rounds(wrong, k_max, n_tokens):
    """Replay the acceptance schedule the fake backends will produce: [(k, a)] per round."""
    out, emitted, rnd = [], 0, 0
    while emitted < n_tokens:
        budget = n_tokens - emitted
        k = min(k_max, budget - 1)
        a = next((i for i in range(k) if (rnd, i) in wrong), k)
        out.append((k, a))
        emitted += a + 1
        rnd += 1
    return out


@pytest.mark.parametrize("wrong", [set(), {(0, 0)}, {(0, 1)}, {(1, 0)}, {(0, 0), (1, 2), (2, 4)}],
                         ids=["all-accepted", "a0", "partial", "wrong-second-round", "mixed"])
def test_spec_accept_counts_and_token_equivalence(wrong):
    """Every acceptance pattern (a=k, a=0, partial) emits exactly the plain greedy stream."""
    n = 12
    pair = Pair(ids_of(SPEC_TEXT), spec=True, draft_wrong=wrong)
    plain = Pair(ids_of(SPEC_TEXT))
    try:
        pair.start()
        spec_out = "".join(pair.generate(ids_of("prompt!"), max_tokens=n))
        st = pair.engine.last_stats
        assert spec_out == SPEC_TEXT[:n]                       # spec == plain greedy, token for token
        assert st["spec_used"] is True and st["new_tokens"] == n
        assert st["eos_reason"] == "max_new_tokens"
        rounds = [(k, a) for k, a in pair.be0.round_log]
        assert rounds == expected_rounds(wrong, 5, n)          # every accept count landed as scheduled
        assert st["spec_rounds"] == len(rounds)
        assert st["accepted_draft_tokens"] == sum(a for _, a in rounds)
        assert st["rejected_draft_tokens"] == sum(k - a for k, a in rounds)
        assert mirrored(pair.be0) == mirrored(pair.be1)        # rank 1 mirrored every trunk call
        assert pair.be1.released == pair.be1.states == 1
        # token equivalence against the plain engine over its own pair
        plain.start()
        plain_out = "".join(plain.generate(ids_of("prompt!"), max_tokens=n))
        assert plain_out == spec_out and plain.engine.last_stats["new_tokens"] == st["new_tokens"]
    finally:
        pair.stop()
        plain.stop()


def test_spec_k0_rounds_when_confidence_is_zero():
    """k=0 rounds (confidence below threshold / budget of one): window is just the anchor, a=0,
    one bonus token per round, rewind(0) still mirrored."""
    pair = Pair(ids_of("abcde"), spec=True, conf_len=0)
    try:
        pair.start()
        assert "".join(pair.generate(ids_of("hi!"), max_tokens=3)) == "abc"
        rounds = pair.be0.round_log
        assert rounds == [(0, 0)] * 3
        rewinds = [c for c in pair.be0.calls if c[0] == "rewind"]
        assert [r[1] for r in rewinds] == [0, 0, 0]
        assert mirrored(pair.be0) == mirrored(pair.be1)
    finally:
        pair.stop()


def test_spec_position_and_rewind_arithmetic():
    """Verify positions, rewind counts and draft-kv refreshes follow the proven schedule:
    pos(r) = len(prompt)-1 + sum(a_j+1) over the earlier rounds."""
    wrong = {(0, 1), (2, 0)}
    n = 14
    prompt = ids_of("prompt!")
    pair = Pair(ids_of(SPEC_TEXT), spec=True, draft_wrong=wrong)
    try:
        pair.start()
        assert "".join(pair.generate(prompt, max_tokens=n)) == SPEC_TEXT[:n]
        rounds = pair.be0.round_log
        assert rounds == expected_rounds(wrong, 5, n)
        pos = len(prompt) - 1
        ver = iter(c for c in pair.be0.calls if c[0] == "verify")
        rew = iter(c for c in pair.be0.calls if c[0] == "rewind")
        ref = iter(c for c in pair.be0.calls if c[0] == "refresh")
        for k, a in rounds:
            v = next(ver)
            assert v[2] == pos and len(v[1]) == k + 1          # window [anchor] + drafts at past_len
            assert next(rew)[1] == k - a                       # one rewind per round: the rejected tail
            pos += a + 1
            assert next(ref)[1:] == (a + 1, pos)               # refresh from the a+1 accepted rows
        assert next(ver, None) is None and next(rew, None) is None and next(ref, None) is None
        assert mirrored(pair.be0) == mirrored(pair.be1)
        # the same arithmetic ran on rank 1 (positions inside its own verify records)
        v1 = [c[2] for c in pair.be1.calls if c[0] == "verify"]
        assert v1 == [c[2] for c in pair.be0.calls if c[0] == "verify"]
    finally:
        pair.stop()


def test_spec_max_tokens_exact_and_k_capped_by_budget():
    """Budget capping keeps n_new exact: k <= budget-1, the last round is k=0 when one token is
    left, and a spec_k cap is honored on every round."""
    pair = Pair(ids_of("0123456789abcdefghij"), spec=True)
    try:
        pair.start()
        assert "".join(pair.generate(ids_of("p!"), max_tokens=7)) == "0123456"
        rounds = pair.be0.round_log
        assert rounds == expected_rounds(set(), 5, 7)          # [(5,5), (0,0)]: 6 + 1 tokens
        assert pair.engine.last_stats["eos_reason"] == "max_new_tokens"
        assert pair.engine.last_stats["new_tokens"] == 7

        assert "".join(pair.generate(ids_of("q!"), max_tokens=7, spec_k=2)) == "0123456"
        assert pair.be0.round_log == expected_rounds(set(), 2, 7)
        drafts = [c for c in pair.be0.calls if c[0] == "draft"]
        assert all(d[3] <= 2 for d in drafts[-len(pair.be0.round_log):])   # this request's k_cap honored
        assert mirrored(pair.be0) == mirrored(pair.be1)
    finally:
        pair.stop()


def test_spec_stop_string_mid_round_and_eos_boundaries():
    """A stop string that lands inside a round's accepted tokens ends the request there; both
    ranks are settled (rewind done) and the next request still runs."""
    script = ids_of("alpha beta STOP gamma delta")
    pair = Pair(script, spec=True, eos_id=1)
    try:
        pair.start()
        text = "".join(pair.generate(ids_of("p!"), max_tokens=64, stop=["STOP"]))
        assert text.startswith("alpha beta ") and text.endswith("STOP") and "gamma" not in text
        st = pair.engine.last_stats
        assert st["eos_reason"] == "stop_word" and st["spec_used"] is True

        script2 = ids_of("near end") + [1]                     # eos inside the second round
        pair.be0.script = script2
        pair.be1.script = script2
        assert "".join(pair.generate(ids_of("q!"), max_tokens=16)) == "near end"
        st = pair.engine.last_stats
        assert st["eos"] is True and st["eos_reason"] == "eos" and st["new_tokens"] == len("near end")
        assert pair.be1.released == pair.be1.states == 2
    finally:
        pair.stop()


def test_spec_cancel_at_round_boundary_keeps_pair_usable():
    pair = Pair(ids_of("0123456789" * 4), spec=True)
    pair.start()

    async def go():
        out, cancel = [], asyncio.Event()
        gen = pair.engine.generate("", prompt_ids=ids_of("p!"), max_tokens=100, temperature=0.0,
                                   top_p=1.0, stop=[], cancel=cancel, speculative=True)
        async for piece in gen:
            out.append(piece)
            if len(out) >= 2:
                cancel.set()
        return out

    out = asyncio.run(go())
    pair.sync_rank1()
    assert 2 <= len(out) < 100
    assert pair.engine.last_stats["eos_reason"] == "cancelled"
    assert pair.be0.released == pair.be0.states == 1
    assert "".join(pair.generate(ids_of("q!"), max_tokens=5)) == "01234"     # pair still healthy
    assert mirrored(pair.be0) == mirrored(pair.be1)
    pair.stop()


def test_spec_mode_switching_at_settled_boundaries():
    """Plain and spec requests alternate on one pair; the mode is fixed per request and rank 1
    follows each begin's flag. Default (null speculative) follows the server default."""
    pair = Pair(ids_of("mode switching!"), spec=True)
    try:
        pair.start()
        p = ids_of("p!")
        assert "".join(pair.generate(p, max_tokens=4, speculative=False)) == "mode"       # plain
        assert any(c[0] == "forward" for c in pair.be1.calls)
        assert "".join(pair.generate(p, max_tokens=4)) == "mode"                          # spec default
        assert any(c[0] == "draft_prefill_kv" for c in pair.be0.calls)
        assert "".join(pair.generate(p, max_tokens=4, speculative=True)) == "mode"        # spec explicit
        assert "".join(pair.generate(p, max_tokens=4, speculative=False)) == "mode"       # plain again
        specs = [c for c in pair.be1.calls if c[0] == "prefill"]
        assert [s[2] for s in specs] == [False, True, True, False]                        # begin flags mirrored
        assert pair.be1.released == pair.be1.states == 4
        assert mirrored(pair.be0) == mirrored(pair.be1)
    finally:
        pair.stop()


def test_spec_temperature_falls_back_to_plain_and_not_enabled_is_refused():
    pair = Pair(ids_of("sampling falls back"), spec=True)
    try:
        pair.start()
        # temperature > 0: silent plain fallback (stochastic output is fine -- the mode is the point)
        out = "".join(pair.generate(ids_of("p!"), max_tokens=6, temperature=1.0, speculative=True))
        st = pair.engine.last_stats
        assert len(out) >= 4 and st["spec_used"] is False and st["new_tokens"] == 6
        assert not any(c[0] == "draft" for c in pair.be0.calls)                           # plain steps ran
        assert sum(1 for c in pair.be0.calls if c[0] == "forward") == 6
    finally:
        pair.stop()

    plain = Pair(ids_of("no spec here"))
    try:
        plain.start()
        with pytest.raises(ValueError, match="not enabled"):
            _run_to_end(plain.engine, prompt_ids=ids_of("p!"), max_tokens=4,
                        temperature=0.0, top_p=1.0, stop=[], speculative=True)
        assert plain.be0.calls == [] and plain.be1.calls == []                            # nothing reached rank 1
    finally:
        plain.stop()


def test_spec_warmup_exercises_both_paths():
    pair = Pair(ids_of("warm me up"), spec=True)
    try:
        pair.start()
        pair.engine.warmup(prompt_tokens=8, decode_tokens=2)
        pair.sync_rank1()
        pre = [c for c in pair.be0.calls if c[0] == "prefill"]
        assert [s[2] for s in pre] == [True, False]                       # spec warm-up, then plain
        assert any(c[0] == "draft_prefill_kv" for c in pair.be0.calls)
        assert any(c[0] == "forward" for c in pair.be0.calls)
        assert mirrored(pair.be0) == mirrored(pair.be1)
        assert pair.be1.released == pair.be1.states == 2
    finally:
        pair.stop()


# ---- protocol guards for the spec ops ------------------------------------------------------------------------


def _worker_thread(backend, port, errors):
    def run():
        try:
            chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
            chan.set_recv_timeout(None)
            tp_engine.Rank1Worker(backend, chan).serve_forever()
        except BaseException as exc:                                     # noqa: BLE001
            errors.append(exc)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _hello(chan, dspark=False):
    chan.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": 129280, "dspark": dspark})
    return chan.recv()


def test_spec_round_and_accept_framing_roundtrip():
    """round/accept frames (window = list of ints) round-trip byte-exact through the channel."""
    port = free_port()
    got = {}
    errors = []

    def rank1():
        chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
        chan.set_recv_timeout(None)
        assert chan.recv()["op"] == "hello"                       # answer rank 0's handshake
        chan.send({"op": "hello_ok", "proto": tp_engine.PROTO, "rank": 1,
                   "vocab_size": 129280, "dspark": True})
        got["begin"] = chan.recv()
        got["round"] = chan.recv()
        got["accept"] = chan.recv()
        assert chan.recv()["op"] == "ping"                        # answer the liveness probe in line
        chan.send({"op": "pong"})
        got["end"] = chan.recv()
        chan.close()

    t = threading.Thread(target=rank1, daemon=True)
    t.start()
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    window = [123456, 7, 8, 9, 10, 11]
    cli.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": 129280, "dspark": True})
    assert cli.recv()["op"] == "hello_ok"
    cli.send({"op": "begin", "req": 0, "ids": [1, 2, 3], "spec": True})
    cli.send({"op": "round", "req": 0, "k": 5, "window": window})
    cli.send({"op": "accept", "req": 0, "accepted": 3})
    assert cli.request({"op": "ping"}, timeout=5.0) == {"op": "pong"}
    cli.send({"op": "end", "req": 0})
    t.join(10.0)
    assert not t.is_alive() and not errors
    assert got["begin"] == {"op": "begin", "req": 0, "ids": [1, 2, 3], "spec": True}
    assert got["round"] == {"op": "round", "req": 0, "k": 5, "window": window}
    assert got["accept"] == {"op": "accept", "req": 0, "accepted": 3}
    assert got["end"] == {"op": "end", "req": 0}
    cli.close()


def test_spec_protocol_guards():
    """step during spec, round during plain, accept without round, and spec begin on a plain
    worker all fail loudly with an error frame (and the worker exits: restart both ranks)."""
    # 1. plain worker: round without a spec request
    port = free_port()
    errors = []
    t = _worker_thread(ScriptedBackend([1, 2, 3]), port, errors)
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    assert _hello(cli)["op"] == "hello_ok"
    cli.send({"op": "begin", "req": 0, "ids": [5, 6], "spec": False})
    cli.send({"op": "round", "req": 0, "k": 1, "window": [5, 7]})
    assert "round without a spec request" in cli.recv()["message"]
    t.join(10.0)
    assert not t.is_alive()
    cli.close()

    # 2. spec worker: step during a spec request
    port = free_port()
    errors = []
    t = _worker_thread(ScriptedSpecBackend([1, 2, 3]), port, errors)
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    assert _hello(cli, True)["op"] == "hello_ok"
    cli.send({"op": "begin", "req": 0, "ids": [5, 6], "spec": True})
    cli.send({"op": "step", "req": 0, "token": 7})
    assert "step during a spec request" in cli.recv()["message"]
    t.join(10.0)
    assert not t.is_alive()
    cli.close()

    # 3. spec worker: accept without round
    port = free_port()
    errors = []
    t = _worker_thread(ScriptedSpecBackend([1, 2, 3]), port, errors)
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    assert _hello(cli, True)["op"] == "hello_ok"
    cli.send({"op": "begin", "req": 0, "ids": [5, 6], "spec": True})
    cli.send({"op": "accept", "req": 0, "accepted": 0})
    assert "accept without round" in cli.recv()["message"]
    t.join(10.0)
    assert not t.is_alive()
    cli.close()

    # 4. plain worker: spec begin refused (DSpark not loaded)
    port = free_port()
    errors = []
    t = _worker_thread(ScriptedBackend([1, 2, 3]), port, errors)
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    assert _hello(cli)["op"] == "hello_ok"
    cli.send({"op": "begin", "req": 0, "ids": [5, 6], "spec": True})
    assert "did not load DSpark" in cli.recv()["message"]
    t.join(10.0)
    assert not t.is_alive()
    cli.close()


def test_spec_protocol_mismatch_is_fatal():
    """The DSpark flag must match across the handshake (both ranks opt in together);
    hello_ok reports the worker's own state so rank 0 can refuse the mismatch."""
    port = free_port()
    errors = []
    t = _worker_thread(ScriptedSpecBackend([1, 2, 3]), port, errors)      # rank 1 has DSpark
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    reply = _hello(cli, dspark=False)                                     # rank 0 says plain
    assert reply["op"] == "hello_ok" and reply["dspark"] is True           # hello_ok reports the truth
    cli.close()                                                           # rank 0 refuses and goes away
    t.join(10.0)
    assert not t.is_alive() and not errors
    # tp_serve.build_tp_engine treats this mismatch as a fatal startup error (checked on the reply)


# ---- the vendored HTTP layer over the TP engine ------------------------------------------------------------


def _sse_events(text: str) -> list:
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append("[DONE]" if line[6:] == "[DONE]" else json.loads(line[6:]))
    return out


def test_vendored_server_over_tp_engine_non_stream_and_stream():
    script = ids_of("let me think</think>Hello from both ranks!")
    pair = Pair(script, ctx=2048)
    pair.start()

    async def go():
        client = TestClient(TestServer(serve.create_app(pair.engine, MODEL)))
        await client.start_server()
        try:
            body = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": len(script), "temperature": 0.0}
            r = await client.post("/v1/chat/completions", json=body)
            assert r.status == 200
            whole = await r.json()
            msg = whole["choices"][0]["message"]
            assert msg["content"] == "Hello from both ranks!" and msg["reasoning_content"] == "let me think"
            assert whole["choices"][0]["finish_reason"] == "length"
            assert whole["usage"]["completion_tokens"] == len(script)
            assert whole["usage"]["prompt_tokens"] == len(pair.engine.prompt_ids(
                __import__("ds41_chat").render_chat(body["messages"])))

            r = await client.post("/v1/chat/completions", json={**body, "stream": True})
            assert r.headers["Content-Type"].startswith("text/event-stream")
            events = _sse_events(await r.text())
            assert events[-1] == "[DONE]"
            deltas = [e for e in events if e != "[DONE]" and e.get("choices")]
            reasoning = "".join(d.get("reasoning_content") or "" for e in deltas[:-1] for d in [e["choices"][0]["delta"]])
            content = "".join(d.get("content") or "" for e in deltas[:-1] for d in [e["choices"][0]["delta"]])
            assert reasoning == "let me think" and content == "Hello from both ranks!"
            assert deltas[-1]["choices"][0]["finish_reason"] == "length"

            s = await (await client.get("/tp/status")).json()
            assert s["rank1"] == "ok" and s["status"] == "ok" and s["requests"] == 2
            h = await (await client.get("/health")).json()
            assert h["status"] == "ok" and h["model"] == MODEL
        finally:
            await client.close()

    asyncio.run(go())
    pair.stop()


def test_vendored_server_spec_round_trip_and_equivalence():
    """The per-request fields over HTTP: speculative:true drafts (stats say so), speculative:false
    and temperature>0 fall back to plain, both produce the same tokens, bad fields are 400."""
    script = ids_of("quick check</think>spec and plain agree!")
    pair = Pair(script, ctx=2048, spec=True)
    pair.start()

    async def go():
        client = TestClient(TestServer(serve.create_app(pair.engine, MODEL)))
        await client.start_server()
        try:
            base = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": len(script), "temperature": 0.0}
            r = await client.post("/v1/chat/completions", json={**base, "speculative": True})
            assert r.status == 200
            spec_body = await r.json()
            assert spec_body["choices"][0]["message"]["content"] == "spec and plain agree!"
            assert pair.engine.last_stats["spec_used"] is True
            assert spec_body["timings"]["accepted_draft_tokens"] >= 0        # spec stats reach timings

            r = await client.post("/v1/chat/completions", json={**base, "speculative": False})
            assert r.status == 200
            plain_body = await r.json()
            assert pair.engine.last_stats["spec_used"] is False
            assert plain_body["choices"][0]["message"]["content"] == spec_body["choices"][0]["message"]["content"]
            assert plain_body["usage"]["completion_tokens"] == spec_body["usage"]["completion_tokens"]

            # spec_k honored: every draft call of this request capped at 2
            r = await client.post("/v1/chat/completions", json={**base, "speculative": True, "spec_k": 2})
            assert r.status == 200
            drafts = [c for c in pair.be0.calls if c[0] == "draft"]
            assert all(d[3] <= 2 for d in drafts[-len(pair.be0.round_log):])

            # temperature > 0 with speculative:true: silent plain fallback (sampled text, so the
            # mode and the token count are the observable contract)
            r = await client.post("/v1/chat/completions", json={**base, "speculative": True,
                                                              "temperature": 1.0, "max_tokens": 8})
            assert r.status == 200
            assert pair.engine.last_stats["spec_used"] is False
            assert (await r.json())["usage"]["completion_tokens"] == 8

            for bad in ({"spec_k": 0}, {"spec_k": "two"}, {"speculative": "yes"}):
                r = await client.post("/v1/chat/completions", json={**base, **bad})
                assert r.status == 400, await r.text()

            s = await (await client.get("/tp/status")).json()
            assert s["dspark"] is True and s["rank1"] == "ok"
        finally:
            await client.close()

    asyncio.run(go())
    pair.stop()


def test_vendored_server_speculative_true_without_dspark_is_400():
    pair = Pair(ids_of("no dspark loaded"), ctx=2048)                    # plain pair
    pair.start()

    async def go():
        client = TestClient(TestServer(serve.create_app(pair.engine, MODEL)))
        await client.start_server()
        try:
            r = await client.post("/v1/chat/completions", json={
                "model": MODEL, "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4, "temperature": 0.0, "speculative": True})
            assert r.status == 400 and "not enabled" in (await r.json())["error"]["message"]
            s = await (await client.get("/tp/status")).json()
            assert s["dspark"] is False
        finally:
            await client.close()

    asyncio.run(go())
    pair.stop()


def test_tp_engine_refuses_oversized_prompts_and_single_token_prompts():
    pair = Pair(ids_of("abc"), ctx=8)
    try:
        pair.start()
        with pytest.raises(ValueError, match="does not fit the context"):
            _run_to_end(pair.engine, prompt_ids=list(range(8)), max_tokens=2,
                        temperature=0.0, top_p=1.0, stop=[])
        with pytest.raises(ValueError, match="at least 2"):
            _run_to_end(pair.engine, prompt_ids=[7], max_tokens=2,
                        temperature=0.0, top_p=1.0, stop=[])
        # nothing was sent to rank 1: no request ever started
        assert pair.be0.calls == [] and pair.be1.calls == []
    finally:
        pair.stop()


def _run_to_end(engine, **kw):
    async def go():
        async for _ in engine.generate("", **kw):
            pass
    asyncio.run(go())
