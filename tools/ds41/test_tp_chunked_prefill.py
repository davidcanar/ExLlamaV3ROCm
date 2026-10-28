"""Regression tests for the 2026-10-28 long-prompt failure of the DS4.1 TP server.

The server (cache 524288, DSpark on) died on a ~7785-token prompt with
``DSA pool overflow: entry 3892 beyond block table (8 pages)``: TPModelBackend passed
``batch_shape = (1, span)`` for every trunk call, and the library sizes the DSA pool's
block table from batch_shape (attn.prepare_flash_attn -> _get_block_table), so span 2048
bought an 8-page table. The assert fired inside a collective, rank 1 died, and rank 0 hung
on NCCL until its 600 s watchdog killed it. The same one-shot prefill transiently ate
~20 GiB of unified VRAM and dropped free memory below 100 MiB.

The fixes pinned here (CPU only, fake core models -- no GPU, no exllamav3 import):
  - every trunk call (prefill chunks, decode steps, spec verifies) carries
    ``batch_shape = (1, cache_tokens)`` so the block table spans the whole cache;
  - prefill runs in span-row chunks with ``past_len`` at each chunk's absolute offset,
    everything but the last prompt token fed exactly once, in order;
  - a chunked spec prefill exports the taps on EVERY chunk (both ranks) and rank 0 primes
    the drafter's kv rows at each chunk's offset, consuming the export so the engine does
    not re-prime;
  - prompts past the cache bound are refused on rank 0 BEFORE the begin frame and on
    rank 1 with a clean error frame (no mid-collective fault, no NCCL wedge);
  - the startup memory floor logs free VRAM and refuses to serve below it.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_engine  # noqa: E402
import test_tp_control as ctrl  # noqa: E402  (the scripted pairs and helpers of the main suite)


class _TOK:
    actual_vocab_size = 32


class FakeCore:
    """Stand-in for the exllamav3 model: records every prefill/forward call's params snapshot
    (taken at call time, before the backend pops keys it consumed) and, when asked to export
    taps, fabricates one (1, rows, 4) fp16 tensor per export layer."""

    def __init__(self, export: bool = False, n_taps: int = 3):
        self.export = export
        self.n_taps = n_taps
        self.prefill_calls: list[dict] = []
        self.forward_calls: list[dict] = []

    def _snap(self, ids, params) -> dict:
        snap = {
            "ids": [int(t) for t in ids.view(-1)],
            "batch_shape": params.get("batch_shape"),
            "past_len": params.get("past_len"),
            "export_asked": "export_state_layers" in params,
            "recurrent_states": len(params.get("recurrent_states") or []),
        }
        return snap

    def prefill(self, ids, params):
        self.prefill_calls.append(self._snap(ids, params))
        if self.export and "export_state_layers" in params:
            params["export_states"] = [
                torch.full((1, ids.shape[1], 4), float(i), dtype=torch.float16)
                for i in range(self.n_taps)
            ]
        return None

    def forward(self, ids, params):
        self.forward_calls.append(self._snap(ids, params))
        if self.export and "export_state_layers" in params:
            params["export_states"] = [
                torch.full((1, ids.shape[1], 4), float(i), dtype=torch.float16)
                for i in range(self.n_taps)
            ]
        return torch.zeros(1, ids.shape[1], 40)          # padded vocab; the backend slices [:32]


class TapRecorderDraft:
    """Stand-in for the mtp drafter: records (rows, absolute offset) of every kv priming."""

    def __init__(self):
        self.updates: list[tuple[int, int]] = []

    def update_kv_from_target(self, target_hidden, cache, params, lengths=None):
        rows = int(target_hidden[0].shape[1])
        offset = int(params["cache_seqlens"].reshape(-1)[0].item())
        if lengths is not None:
            rows = min(rows, int(max(lengths)))
        self.updates.append((rows, offset))


def make_backend(span, cache_tokens, *, rank=0, spec=False, core=None, draft=None, page_size=None):
    kwargs = {}
    if spec:
        kwargs.update(dspark=True, export_layers=frozenset({36, 37, 38}),
                      draft_cache=object(),
                      draft_bt=torch.arange(cache_tokens // (page_size or 256),
                                            dtype=torch.int32).view(1, -1))
        if draft is not None:
            kwargs["draft_model"] = draft
    return tp_engine.TPModelBackend(core if core is not None else FakeCore(export=spec),
                                    object(), _TOK(), span=span, cache_tokens=cache_tokens,
                                    rank=rank, **kwargs)


# ---- block tables sized to the cache, prefill chunked by span -----------------------------------------------


def test_prefill_chunks_by_span_and_sizes_block_tables_to_the_cache():
    be = make_backend(span=8, cache_tokens=64, page_size=16)
    ids = list(range(31))                                  # 30 prefill rows -> 8+8+8+6
    out = be.prefill(ids, "state")
    calls = be.model.prefill_calls
    assert [c["ids"] for c in calls] == [ids[0:8], ids[8:16], ids[16:24], ids[24:30]]
    assert [c["past_len"] for c in calls] == [0, 8, 16, 24]          # absolute offsets
    assert all(c["batch_shape"] == (1, 64) for c in calls)           # table covers the WHOLE cache
    assert all(c["recurrent_states"] == 1 for c in calls)
    fed = [t for c in calls for t in c["ids"]]
    assert fed == ids[:-1]                                           # each row exactly once, in order
    assert sorted(fed) == sorted(set(fed))
    assert not out.get("export_states")                              # plain prefill exports nothing


def test_short_prefill_is_one_chunk_like_the_proven_script():
    be = make_backend(span=64, cache_tokens=64, page_size=16)
    ids = list(range(20))
    be.prefill(ids, "state")
    calls = be.model.prefill_calls
    assert len(calls) == 1 and calls[0]["ids"] == ids[:-1]
    assert calls[0]["past_len"] == 0 and calls[0]["batch_shape"] == (1, 64)


def test_forward_and_verify_size_block_tables_to_the_cache():
    be = make_backend(span=8, cache_tokens=64, page_size=16, spec=True)
    row = be.forward(5, 3, "state")
    assert row.shape == (32,)                                        # rank 0 crops to the real vocab
    c = be.model.forward_calls[0]
    assert c["batch_shape"] == (1, 64) and c["past_len"] == 3 and c["ids"] == [5]

    logits, params = be.verify([1, 2, 3, 4], 40, "state")
    assert logits.shape == (4, 32)
    v = be.model.forward_calls[1]
    assert v["batch_shape"] == (1, 64)                               # NOT (1, span): high positions fit
    assert v["past_len"] == 40 and v["ids"] == [1, 2, 3, 4]
    assert v["export_asked"] is True                                 # taps still exported for the refresh


def test_chunking_is_pure_in_span_cache_and_len():
    """Both ranks derive the chunk plan from (len, span) with identical env -> identical plans."""
    for span, cache, n in ((8, 64, 31), (2048, 524288, 7785), (3, 28, 22), (256, 512, 257)):
        be = make_backend(span=span, cache_tokens=cache)
        be.prefill(list(range(n + 1)), "s")
        cuts = [(c["past_len"], len(c["ids"])) for c in be.model.prefill_calls]
        assert sum(l for _, l in cuts) == n
        assert all(l <= span for _, l in cuts)
        assert [a for a, _ in cuts] == sorted({a for a, _ in cuts})   # strictly increasing offsets
        assert all(c["batch_shape"] == (1, cache) for c in be.model.prefill_calls)


# ---- DSpark on long prompts: taps per chunk, drafter primed per chunk ---------------------------------------


def test_spec_prefill_primes_the_draft_kv_per_chunk_at_chunk_offsets():
    draft = TapRecorderDraft()
    be = make_backend(span=8, cache_tokens=64, page_size=16, spec=True, draft=draft)
    ids = list(range(31))                                             # 30 rows -> chunks 8,8,8,6
    out = be.prefill(ids, "state", spec=True)
    assert draft.updates == [(8, 0), (8, 8), (8, 16), (6, 24)]        # rows at absolute offsets
    assert all(c["export_asked"] for c in be.model.prefill_calls)     # taps exported on EVERY chunk
    assert not (out.get("export_states"))                             # consumed: engine must not re-prime


def test_short_spec_prefill_matches_the_proven_single_shot_priming():
    draft = TapRecorderDraft()
    be = make_backend(span=64, cache_tokens=64, page_size=16, spec=True, draft=draft)
    be.prefill(list(range(20)), "state", spec=True)
    assert draft.updates == [(19, 0)]                                 # exactly the proven script's call


def test_spec_prefill_without_a_tap_export_fails_loudly():
    be = make_backend(span=8, cache_tokens=64, page_size=16, spec=True,
                      core=FakeCore(export=False), draft=TapRecorderDraft())
    with pytest.raises(RuntimeError, match="tap export missing"):
        be.prefill(list(range(10)), "state", spec=True)


def test_rank1_backend_without_a_drafter_primes_nothing_but_still_exports():
    """Rank 1 runs the identical chunked spec prefill (taps exported, same dispatches) but has
    no drafter to prime -- the asymmetry that keeps the collective sequences identical."""
    be = make_backend(span=8, cache_tokens=64, page_size=16, spec=True, rank=1, draft=None)
    out = be.prefill(list(range(21)), "state", spec=True)
    assert len(be.model.prefill_calls) == 3
    assert all(c["export_asked"] for c in be.model.prefill_calls)
    assert out.get("export_states") is not None                       # not consumed on rank 1


# ---- bounds checked before any dispatch (no NCCL wedge) ------------------------------------------------------


def test_engine_rejects_prompts_beyond_the_cache_before_any_dispatch():
    pair = ctrl.Pair(ctrl.ids_of("abc"), ctx=512)
    pair.be0.cache_tokens = 16
    pair.be1.cache_tokens = 16
    try:
        pair.start()
        with pytest.raises(ValueError, match="cache bound"):
            ctrl._run_to_end(pair.engine, prompt_ids=list(range(40)), max_tokens=2,
                             temperature=0.0, top_p=1.0, stop=[])
        # nothing reached either backend: no begin, no prefill, no collectives
        assert pair.be0.calls == [] and pair.be1.calls == []
        # a prompt that fits still serves afterwards
        assert "".join(pair.generate(ctrl.ids_of("ok!"), max_tokens=2)) == "ab"
    finally:
        pair.stop()


def test_rank1_refuses_prompts_beyond_its_cache_with_a_clean_error():
    port = ctrl.free_port()
    errors = []
    t = ctrl._worker_thread(ctrl.ScriptedBackend([1, 2, 3]), port, errors)
    try:
        cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
        ctrl._hello(cli)
        cli.send({"op": "begin", "req": 0, "ids": [5, 6], "spec": False})   # fine: no bound set
        cli.send({"op": "end", "req": 0})
        cli.close()
        t.join(10.0)
        assert not t.is_alive() and not errors
    finally:
        cli.close()

    class Capped(ctrl.ScriptedBackend):
        cache_tokens = 8

    port = ctrl.free_port()
    errors = []
    t = ctrl._worker_thread(Capped([1, 2, 3]), port, errors)
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    try:
        ctrl._hello(cli)
        cli.send({"op": "begin", "req": 0, "ids": list(range(11)), "spec": False})
        reply = cli.recv()
        assert reply["op"] == "error" and "exceeds the cache bound" in reply["message"]
        t.join(10.0)
        assert not t.is_alive()                                        # worker died cleanly, no hang
    finally:
        cli.close()


# ---- the startup memory floor --------------------------------------------------------------------------------


def test_memory_floor_guard_raises_below_the_floor(monkeypatch):
    monkeypatch.setattr(tp_engine, "_vram_report", lambda dev=None: (0.09, 124.0))
    with pytest.raises(RuntimeError, match="EXL3_TP_MEM_FLOOR"):
        tp_engine.check_memory_floor("after the TP load", rank=0, floor_gib=2.0)


def test_memory_floor_guard_passes_and_logs_free_memory(monkeypatch):
    monkeypatch.setattr(tp_engine, "_vram_report", lambda dev=None: (9.5, 124.0))
    seen = []
    tp_engine.check_memory_floor("after the DSpark drafter load", rank=0, floor_gib=2.0,
                                 log=seen.append)
    assert len(seen) == 1 and "9.5 GiB free" in seen[0] and "floor 2 GiB" in seen[0]


def test_memory_floor_guard_zero_disables_and_missing_info_skips(monkeypatch):
    monkeypatch.setattr(tp_engine, "_vram_report", lambda dev=None: (0.05, 124.0))
    tp_engine.check_memory_floor("after the TP load", rank=1, floor_gib=0.0)      # disabled
    monkeypatch.setattr(tp_engine, "_vram_report", lambda dev=None: None)
    seen = []
    tp_engine.check_memory_floor("after the TP load", rank=1, floor_gib=2.0, log=seen.append)
    assert seen and "unavailable" in seen[0]                                      # CPU boxes skip


# ---- long requests return the reserved transient pool to the system ------------------------------------------


class TrimRecorder(ctrl.ScriptedBackend):
    span = 4

    def __init__(self, script):
        super().__init__(script)
        self.trims = 0

    def trim_memory(self):
        self.trims += 1


def test_long_requests_trim_the_reserved_pool_short_ones_do_not():
    """A chunked prompt grows the caching allocator's reservation (fragmentation); the engine
    returns it after the request. Short requests keep their segments warm (no trim latency)."""
    pair = ctrl.Pair(ctrl.ids_of("trim"))
    pair.be0 = TrimRecorder(ctrl.ids_of("trim"))
    try:
        pair.start()
        assert "".join(pair.generate(list(range(10)), max_tokens=2)) == "tr"    # 10 ids > span 4
        assert pair.be0.trims == 1
        assert "".join(pair.generate(ctrl.ids_of("hi!"), max_tokens=2)) == "tr"  # 3 ids < span 4
        assert pair.be0.trims == 1                                             # short: no trim
        assert pair.be1.released == pair.be1.states == 2                       # pair still healthy
    finally:
        pair.stop()


# ---- engine-level: both ranks run the identical chunked prefill over the wire --------------------------------


class ChunkedRealBackend:
    """TPEngine-facing backend whose prefill is the REAL TPModelBackend chunking over a fake
    core; decode rows come from a greedy script (the plain path under test is the chunking)."""

    dspark = False

    def __init__(self, script, span=8, cache_tokens=64):
        self.script = [int(t) for t in script]
        self.vocab_size = 129280
        self.cache_tokens = int(cache_tokens)
        self.core = FakeCore()
        self.inner = tp_engine.TPModelBackend(self.core, object(), _TOK(), span=span,
                                              cache_tokens=cache_tokens, rank=0)
        self.states = self.released = 0
        self._k = 0

    def new_state(self):
        self.states += 1
        self._k = 0
        return f"s{self.states}"

    def release_state(self, state):
        self.released += 1

    def prefill(self, ids, state, spec=False):
        self.inner.prefill([int(t) for t in ids], state, spec=spec)
        return {}

    def forward(self, token, position, state):
        tok = self.script[self._k] if self._k < len(self.script) else self.script[-1]
        self._k += 1
        row = torch.zeros(self.vocab_size)
        row[tok % self.vocab_size] = 8.0
        return row


def test_both_ranks_run_the_identical_chunked_prefill_over_the_channel():
    """The lockstep guarantee with real chunking: 30 prefill rows -> the same 4 chunks, same
    past_len offsets, same block-table shape, on rank 0 and rank 1."""
    script = ctrl.ids_of("chunked lockstep holds!")
    prompt = list(range(31))                                           # 30 rows -> 8+8+8+6 at span 8
    be0 = ChunkedRealBackend(script, span=8, cache_tokens=64)
    be1 = ChunkedRealBackend(script, span=8, cache_tokens=64)
    port = ctrl.free_port()
    errors: list[BaseException] = []

    def rank1():
        try:
            chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
            chan.set_recv_timeout(None)
            tp_engine.Rank1Worker(be1, chan).serve_forever()
        except tp_engine.ChannelClosed:
            pass
        except BaseException as exc:                                   # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=rank1, daemon=True)
    t.start()
    chan0 = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    chan0.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": be0.vocab_size,
                "dspark": False})
    assert chan0.recv()["op"] == "hello_ok"
    engine = tp_engine.TPEngine(be0, chan0, ctrl.DetokByteTokenizer(), ctx=512, eos_id=1)
    try:
        async def go():
            out = []
            async for piece in engine.generate("", prompt_ids=prompt, max_tokens=len(script),
                                               temperature=0.0, top_p=1.0, stop=[]):
                out.append(piece)
            return out

        out = "".join(asyncio.run(go()))
        ctrl_pair_wait(be1, be0)
        assert out == "chunked lockstep holds!"
        assert be0.core.prefill_calls == be1.core.prefill_calls        # THE chunk-for-chunk mirror
        assert [c["past_len"] for c in be0.core.prefill_calls] == [0, 8, 16, 24]
        assert all(c["batch_shape"] == (1, 64) for c in be0.core.prefill_calls)
        assert be0.states == be0.released == be1.states == be1.released == 1
        assert not errors
    finally:
        engine.close()
        t.join(10.0)
        assert not t.is_alive()


def ctrl_pair_wait(be1, be0, timeout: float = 5.0):
    """Rank 1 mirrors asynchronously; wait for its end frame to land (as Pair.sync_rank1)."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(be1.core.prefill_calls) == len(be0.core.prefill_calls) and be1.released >= be0.released:
            return
        time.sleep(0.005)
    raise AssertionError("rank 1 did not catch up")
