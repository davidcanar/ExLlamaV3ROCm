#!/usr/bin/env python3
"""Two-node tensor-parallel engine for the DS4.1 server: rank 0 drives, rank 1 mirrors.

Attribution / provenance:
- The HTTP-facing Engine contract, StubEngine and HFTokenizer are vendored in `ds41_engine.py`
  (Kyojin, MIT; see LICENSE.kyojin).
- `TopPSampler` and `_Detok` below are adapted from Kyojin tools/ds41/ds41_model.py (MIT).
- `stop_text` is adapted from Kyojin tools/ds41/serve.py (MIT).
- The model calls of `TPModelBackend` mirror scripts/tp_ds41_generate.py of this repository: the
  Engram-augmented config load, cache-attach-before-TP-load, `model.load(tensor_p=True, ...)`,
  the cached prefill params, the decode-step params and the rank-0 argmax / rank-1 token-mirror
  flow over a small TCP control channel. That script is the validated working path; this module
  keeps its call pattern byte for byte and only makes it persistent and request-driven.

Rank 0 (Box1, `tp_serve.py`) accepts HTTP jobs, renders the prompt (vendored ds41_chat.py),
loads its half of the weights and per request:
  1. sends `begin` with the prompt token ids      -> both ranks: fresh state, prefill, forward #1
  2. forwards and samples one token at a time     -> rank 0 picks the token (greedy or top-p)
  3. sends `step` with the chosen token           -> both ranks forward it on the next iteration
  4. sends `end` when the reply is done           -> both ranks release the request state
Rank 1 (Box2, `tp_rank1.py`) performs exactly the same model calls in the same order, driven only
by control frames; it never reads a forward's return value (cross-host TP produces logits on the
output rank, rank 0, only). Model collectives (RCCL over Odinlink) stay symmetric; the control
channel only sequences the loop, exactly like the TokenChannel of the manual script.

Optional DSpark (MTP speculative decoding), opt-in at load time with EXL3_TP_DSPARK=1 on BOTH
ranks, follows scripts/tp_ds41_dspark.py (the proven two-node path) op for op:
  - both ranks build the trunk cache with max_history=8 and load with per-rank budgets
    95,104 GiB (rank 0 leaves room for the drafter); rank 0 additionally loads the mtp.*
    drafter layer-split, attaches it (private CPU embed + private GPU head: the draft loop issues
    no collectives) and keeps its paged main-kv cache;
  - a spec request prefill exports the trunk taps at the DSpark target layers on both ranks;
    rank 0 projects them into the drafter's kv;
  - each round rank 0 drafts locally, sends `round` (the verify window), both ranks run one
    multi-row verify forward (recurrent_history=True), rank 0 accepts (greedy prefix match at
    temperature 0, exact speculative sampling otherwise: `_accept_sampled`), sends `accept`,
    and both ranks rewind the state identically (`rewind(k - a)` is a mirrored TP dispatch);
  - plain requests (`speculative: false`, sampled requests when EXL3_TP_SPEC_SAMPLED=0, or
    servers started without EXL3_TP_DSPARK) use the unchanged plain `step` path; a request
    picks its mode once, at a settled boundary (between requests, when both ranks are idle) --
    never mid-round.
Per-request `speculative` / `spec_k` (1..5) reach the engine through two marked LOCAL additions in
the vendored ds41_serve.py.

Lockstep rule: every stop decision (eos token, stop string, max_tokens, cancel) happens on rank 0
AFTER sampling a token and BEFORE sending its `step` frame. Once a `step` has been sent, the next
forward is committed on both ranks, so the two ranks always issue the same number of collectives.

Control protocol (framed JSON over TCP: 4-byte little-endian length prefix, then UTF-8 bytes):
  rank 0 -> rank 1: {"op": "hello", "proto": 3, "vocab_size": V, "dspark": B}   handshake after both loads
                    {"op": "begin", "req": N, "ids": [int, ...], "spec": B}     new request (fresh state)
                      + "restore": K, "start": S   prefix cache: restore checkpoint K (position S)
                                                   and prefill only ids[S:]
                      + "drop": [K, ...]           drop these checkpoints (after the restore)
                      + "save": K                  checkpoint the state after the prefill as K
                      + "drop_all": true           first drop every checkpoint (cache_salt reset)
                    {"op": "step", "req": N, "token": T}                        plain decode: feed T next
                    {"op": "round", "req": N, "k": K, "window": [int, ...]}     spec: verify [anchor, drafts]
                    {"op": "accept", "req": N, "accepted": A}                   spec: settle, rewind(k - A)
                    {"op": "end", "req": N}                                    request over, release state
                      + "drop": [K, ...], "save": K   (drop, then checkpoint the final state, then release;
                        "rewind": R first rolls a spec request back to the end of its emitted reply;
                        "drop_all": true clears every checkpoint after a rank-0 bookkeeping failure)
                    {"op": "ping"}                                             liveness probe (idle only)
                    {"op": "shutdown"}                                         clean exit
  rank 1 -> rank 0: {"op": "hello_ok", "proto": 3, "rank": 1, "vocab_size": V, "dspark": B}
                    {"op": "pong"}
                    {"op": "error", "message": "..."}                          rank 1 failed; restart both

Prefix cache (2026-10-28, `PrefixCache`; EXL3_TP_PREFIX_CACHE=0 turns it off on rank 0):
- the trunk cache is one linear sequence (identity block table over the whole cache), so after a
  request the DSA pools still hold the kv of every token it committed; a prompt sharing a prefix
  with them needs only the recurrent state at the restart point, a checkpoint both ranks stash
  (DSV4State.stash: SWA ring, compressor windows, Engram id ring, host memory) after each prefill
  and at the end of each request. Rank 0 picks the longest valid checkpoint, and both ranks
  restore it and prefill the rest. Rank 0 decides everything; rank 1 only obeys the frames.
- checkpoints are local copies on each rank (no collectives); spec checkpoints also carry the
  drafter's main-kv window on rank 0 (its main-kv is a ring, EXL3_TP_DRAFT_RING)

Long prompts (2026-10-28 fix, test_tp_chunked_prefill.py):
- every trunk call (prefill chunks, decode steps, spec verifies) carries batch_shape =
  (1, cache_tokens): the library sizes the DSA pool's block table from batch_shape, and the
  old (1, span) table (8 pages at span 2048) overflowed on prompts past 2048 tokens
  ("DSA pool overflow: entry 3892 beyond block table (8 pages)") inside a collective, which
  wedged the peer rank's NCCL watchdog for 600 s
- prefill runs in span-row chunks (EXL3_TP_SPAN, default 2048) with past_len at each chunk's
  offset: one 7.8K-row pass transiently ate ~20 GiB of (unified) VRAM and dropped free memory
  below 100 MiB; chunking bounds the activation transient to the loader's measured
  max_chunk_size. DSpark survives long prompts: each chunk exports the trunk taps on both
  ranks and rank 0 primes the drafter's kv rows at the chunk's absolute offset (per-position
  math, identical to the proven single-shot priming)
- prompt/context bounds are validated on rank 0 BEFORE the begin frame and again on rank 1
  (a request past the cache would fault inside a collective and hang the peer instead of
  erroring); DSpark loads default to budgets 95,104 GiB and each rank guards a startup
  memory floor (EXL3_TP_MEM_FLOOR, default 2 GiB) so a starved rank refuses to serve

First-version limitations (documented in README.md):
- one in-flight request; prompt reuse covers one conversation at a time (a request that diverges
  early, say a client's side request for a title, overwrites the cached sequence past the shared
  prefix, and the next turn of the main conversation re-prefills from there)
- serving context defaults to ~750 tokens (EXL3_TP_SAFE_CTX, tp_serve.py); the deployment
  (exl3ds4.sh) raises it to the full 512K cache, and prompts/chunks of any size up to
  cache_tokens - 2 are accepted (validated at ~8K-token prompts with DSpark)
- no rank-failure recovery: if either rank dies mid-request, both must be restarted
- DSpark: greedy requests use the script's greedy rule; sampled requests (temperature > 0) use
  exact speculative sampling against the greedy drafts (EXL3_TP_SPEC_SAMPLED=0 sends them down
  the plain path instead); `spec_k` caps the draft window at the DSpark block size (5); the
  bonus token is always emitted, EOS/stop/max cuts land on emitted-token boundaries even mid-round
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import socket
import struct
import threading
import time
from array import array
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

PROTO = 3                    # 3: prefix-cache fields on begin/end (a proto-2 rank 1 would desync)
MAX_FRAME = 1 << 26          # 64 MiB frame guard
MAX_HISTORY = 8              # recurrent rollback capacity of the spec path (as the proven script)

_ROOT = os.environ.get(
    "EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DEFAULT_MODEL_DIR = os.path.join(_ROOT, "model")
DEFAULT_ENGRAM_DIR = os.path.join(_ROOT, "engram-src-1017")
ENGRAM_SHARDS = (
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


class ChannelClosed(ConnectionError):
    """The control channel was closed by the peer (or never established)."""


class ProtocolError(RuntimeError):
    """The peer sent a frame this side does not understand."""


def _torch_generator(seed: int | None):
    if seed is None:
        return None
    import torch
    return torch.Generator().manual_seed(seed)


# ---- sampling and detokenization (adapted from Kyojin tools/ds41/ds41_model.py, MIT) -----------------------


class TopPSampler:
    """temperature + nucleus sampling, same draw rule as the official `sample` (Gumbel-max over the kept
    probabilities: one exponential draw per token). temperature <= 0 is argmax."""

    def __init__(self, temperature: float, top_p: float = 1.0, generator: Any = None):
        self.temperature, self.top_p, self.generator = temperature, top_p, generator
        self.stochastic = temperature > 0

    def probs(self, logits_row: Any) -> Any:
        import torch
        p = torch.softmax(logits_row.reshape(-1).float() / max(self.temperature, 1e-5), dim=-1)
        if self.top_p < 1.0:
            sp, si = p.sort(descending=True)
            cum = sp.cumsum(0)
            drop = (cum - sp) >= self.top_p              # keep the smallest prefix whose mass reaches top_p
            sp = sp.masked_fill(drop, 0.0)
            p = torch.zeros_like(p).scatter_(0, si, sp)
            p = p / p.sum()
        return p

    def sample(self, logits_row: Any) -> int:
        import torch
        if not self.stochastic:
            return int(logits_row.argmax())
        p = self.probs(logits_row)
        noise = torch.empty_like(p).exponential_(1, generator=self.generator)
        return int(p.div_(noise).argmax())

    def probs_rows(self, logits: Any) -> Any:
        """probs() for every row of a (R, V) logits block at once."""
        import torch
        p = torch.softmax(logits.float() / max(self.temperature, 1e-5), dim=-1)
        if self.top_p < 1.0:
            sp, si = p.sort(dim=-1, descending=True)
            cum = sp.cumsum(-1)
            sp = sp.masked_fill((cum - sp) >= self.top_p, 0.0)
            p = torch.zeros_like(p).scatter_(-1, si, sp)
            p = p / p.sum(dim=-1, keepdim=True)
        return p


# Sampled (temperature > 0) requests also draft with DSpark: exact speculative sampling
SPEC_SAMPLED = env_truthy("EXL3_TP_SPEC_SAMPLED", True)


def _accept_sampled(logits: Any, drafts: list[int], k: int, sampler: TopPSampler) -> tuple[int, int]:
    """Speculative sampling against deterministic (greedy) drafts: the round's output follows the
    target's temperature/top-p distribution exactly, as with plain sampled decoding.

    Draft i is accepted with probability p_i(d_i) (the proposal is a point mass, so min(1, p/q)
    is just p). At the first rejection the token is drawn from p_i with d_i removed (the residual
    max(0, p - q), renormalized); when every draft is accepted the bonus token is drawn from
    p_k. The residual draw of every row is made up front from independent noise, so the whole
    decision costs a single host sync. Returns (accepted, bonus)."""
    import torch
    p = sampler.probs_rows(logits[: k + 1])                              # (k + 1, V)
    gen = sampler.generator if (sampler.generator is not None and
                                getattr(sampler.generator, "device", None) == p.device) else None
    q = p.clone()
    if k:
        d = torch.tensor(drafts[:k], dtype=torch.long).to(p.device, non_blocking=True)
        rows = torch.arange(k, device=p.device)
        pd = p[rows, d]
        q[rows, d] = 0.0
        u = torch.rand(k, device=p.device, generator=gen)
        a = torch.cumprod((u < pd).to(torch.int32), dim=0).sum()
    else:
        a = torch.zeros((), dtype=torch.long, device=p.device)
    noise = torch.empty_like(q).exponential_(1, generator=gen)
    cand = q.div_(noise).argmax(dim=-1)                                  # residual draw per row
    a_host, bonus = torch.stack((a.to(torch.long), cand[a])).tolist()
    return int(a_host), int(bonus)


class _Detok:
    """Incremental detokenizer: text of the new tokens, nothing while the tail is an incomplete UTF-8 sequence."""

    def __init__(self, tk):
        self.tk, self.ids, self.prefix, self.read = tk, [], 0, 0

    def push(self, new_ids) -> str:
        self.ids.extend(new_ids)
        old = self.tk.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        new = self.tk.decode(self.ids[self.prefix:], skip_special_tokens=False)
        if len(new) > len(old) and not new.endswith("\ufffd"):
            out = new[len(old):]
            self.prefix, self.read = self.read, len(self.ids)
            return out
        return ""

    def flush(self) -> str:
        old = self.tk.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        new = self.tk.decode(self.ids[self.prefix:], skip_special_tokens=False)
        self.prefix = self.read = len(self.ids)
        return new[len(old):]


def stop_text(text: str, stop: list[str]) -> tuple[str, str | None]:
    """Cut at the earliest stop string and return content and matched stop.

    (Adapted from Kyojin tools/ds41/serve.py, MIT; the engine uses it to end generation at the
    token boundary where a stop string first appears. The server runs the same function again on
    the stream it forwards, so the two never disagree about where the cut is.)"""
    positions = [(text.find(s), s) for s in stop if s and text.find(s) >= 0]
    if not positions:
        return text, None
    pos, match = min(positions)
    return text[:pos], match


# ---- framed control channel ---------------------------------------------------------------------------------


class ControlChannel:
    """Length-prefixed JSON frames over TCP between rank 0 and rank 1.

    The accept/connect flow, retry window and socket timeouts follow the TokenChannel of
    scripts/tp_ds41_generate.py: rank 0 listens on the Thunderbolt address, rank 1 connects
    with retries until its (multi-minute) deadline. Not used for model tensor payloads.
    """

    def __init__(self, conn: socket.socket, *, timeout: float = 600.0, log: Callable[[str], None] | None = None):
        self.conn = conn
        self.timeout = float(timeout)
        self.log = log or (lambda message: None)
        self.lock = threading.RLock()
        self._closed = False
        try:
            self.peer = f"{conn.getpeername()[0]}:{conn.getpeername()[1]}"
        except OSError:
            self.peer = "?"

    @classmethod
    def listen(cls, host: str, port: int, *, accept_timeout: float | None = None, timeout: float = 600.0,
               log: Callable[[str], None] | None = None) -> "ControlChannel":
        """Rank-0 side: bind and block until rank 1 connects (it retries while rank 0 loads)."""
        if accept_timeout is None:
            accept_timeout = env_float("EXL3_TP_ACCEPT_TIMEOUT", 1800.0)
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(accept_timeout)
        server.bind((host, port))
        server.listen(1)
        (log or print)(f"tp: rank 0 waiting for rank 1 on {host}:{port}", flush=True)
        conn, addr = server.accept()
        server.close()
        conn.settimeout(timeout)
        (log or print)(f"tp: rank 1 connected from {addr[0]}:{addr[1]}", flush=True)
        return cls(conn, timeout=timeout, log=log)

    @classmethod
    def connect(cls, host: str, port: int, *, deadline_s: float = 900.0, timeout: float = 600.0,
                idle_timeout: float | None = None, log: Callable[[str], None] | None = None) -> "ControlChannel":
        """Rank-1 side: retry the connection until rank 0 starts listening, then stay connected."""
        deadline = time.time() + deadline_s
        while True:
            try:
                conn = socket.create_connection((host, port), timeout=10)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(1.0)
        conn.settimeout(idle_timeout if idle_timeout is not None else timeout)
        return cls(conn, timeout=timeout, log=log)

    def set_recv_timeout(self, seconds: float | None) -> None:
        self.conn.settimeout(seconds)

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                raise ChannelClosed("control channel closed by peer")
            buf += chunk
        return bytes(buf)

    def send(self, msg: dict) -> None:
        payload = json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(payload) > MAX_FRAME:
            raise ValueError(f"control frame too large: {len(payload)} bytes")
        with self.lock:
            self.conn.sendall(struct.pack("<I", len(payload)) + payload)

    def recv(self) -> dict:
        (length,) = struct.unpack("<I", self._read_exact(4))
        if not 0 < length <= MAX_FRAME:
            raise ChannelClosed(f"bad control frame length {length}")
        msg = json.loads(self._read_exact(length).decode("utf-8"))
        if not isinstance(msg, dict):
            raise ChannelClosed("control frame is not a JSON object")
        return msg

    def request(self, msg: dict, timeout: float = 30.0) -> dict:
        """Send one frame and wait for one reply with a shorter socket timeout (liveness probes)."""
        with self.lock:
            old = self.conn.gettimeout()
            self.conn.settimeout(timeout)
            try:
                self.send(msg)
                return self.recv()
            finally:
                self.conn.settimeout(old)

    def close(self) -> None:
        with self.lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.conn.close()
            except OSError:
                pass


# ---- startup memory floor -----------------------------------------------------------------------------------


def _vram_report(device: int | None = None) -> tuple[float, float] | None:
    """(free GiB, total GiB) of the torch-visible GPU, or None when unavailable (CPU tests).

    On the Strix Halo deployment this is unified GTT memory, i.e. what a forward pass can
    actually allocate -- exactly the pool the long-prefill incident exhausted."""
    try:
        import torch
        free_b, total_b = torch.cuda.mem_get_info(0 if device is None else int(device))
        return free_b / 2**30, total_b / 2**30
    except Exception:                                                       # noqa: BLE001  (informational)
        return None


def check_memory_floor(label: str, *, rank: int, floor_gib: float, log: Callable[[str], None] = print) -> None:
    """Log free GPU memory after a load stage and refuse to serve below the floor.

    A rank that comes up with (almost) no headroom dies at the first long prefill -- inside a
    collective, which wedges the peer rank until its 600 s NCCL watchdog kills it. The floor
    is deliberately crude (several GiB of slack for activations, not an exact model);
    EXL3_TP_MEM_FLOOR=0 disables the guard, negative floors are treated as disabled too."""
    got = _vram_report(env_int("EXL3_TP_LOCAL_DEVICE", 0))
    if got is None:
        log(f"tp: rank {rank} {label}: GPU memory info unavailable; skipping the "
            f"{floor_gib:g} GiB floor check")
        return
    free_gib, total_gib = got
    log(f"tp: rank {rank} {label}: {free_gib:.1f} GiB free of {total_gib:.1f} GiB "
        f"(floor {floor_gib:g} GiB)")
    if floor_gib > 0 and free_gib < floor_gib:
        raise RuntimeError(
            f"rank {rank} has only {free_gib:.1f} GiB free {label}, below the {floor_gib:g} GiB "
            f"floor (EXL3_TP_MEM_FLOOR): the first long prefill would run out mid-collective and "
            f"wedge the peer rank. Lower EXL3_TP_BUDGETS / EXL3_TP_USE_PER_DEVICE or "
            f"EXL3_TP_CACHE_TOKENS, then restart BOTH ranks.")


# ---- the model half of one rank (the real calls, mirroring scripts/tp_ds41_generate.py) --------------------


class TPModelBackend:
    """The exllamav3 model + cache of one rank, with the exact call pattern of the working manual scripts.

    Plain mode mirrors scripts/tp_ds41_generate.py; with `dspark=True` (EXL3_TP_DSPARK=1, both
    ranks) the trunk cache gains max_history=8, the load uses per-rank budgets 95,104 GiB and
    rank 0 additionally loads the mtp.* drafter exactly like scripts/tp_ds41_dspark.py
    (layer-split, attach_to the TP trunk, private embed on CPU + head on GPU so drafting issues
    no collectives, and a paged main-kv cache refreshed from the trunk's exported taps).

    Every trunk call carries `batch_shape = (1, cache_tokens)` so the library's DSA pool block
    table spans the whole cache (the manual scripts' `(1, SPAN)` invariant held only because
    their SPAN equaled their cache size), and prefill is chunked to `span` rows per forward to
    bound the activation transient of long prompts.

    The backend knows its TP rank: in cross-host TP, model.forward returns the logits only on the
    output rank (rank 0 in this validated topology; non-default output-rank topologies are not
    validated in this branch). forward()/verify() return the vocabulary-sliced rows on the output
    rank and None on every other rank -- the manual scripts read logits under `if rank == 0:` the
    same way. Rank 1's worker never looks at the return value; rank 0's engine samples from it.
    """

    OUTPUT_RANK = 0

    def __init__(self, model, cache, tokenizer, *, span: int, cache_tokens: int, rank: int = 0,
                 dspark: bool = False, draft_model=None, draft_cache=None, draft_bt=None,
                 export_layers=frozenset(), dspark_block: int = 5, page_size: int = 256,
                 draft_window_rows: int = 128):
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.span = int(span)
        self.cache_tokens = int(cache_tokens)
        self.rank = int(rank)
        self.dspark = bool(dspark)
        self.draft_model = draft_model            # rank 0 only, dspark
        self.draft_cache = draft_cache
        self.draft_bt = draft_bt
        self.export_layers = frozenset(export_layers)
        self.dspark_block = int(dspark_block)
        self.page_size = int(page_size)
        self.draft_window_rows = int(draft_window_rows)
        self.vocab_size = int(tokenizer.actual_vocab_size)

    @classmethod
    def load(cls, *, rank: int, world_size: int = 2, model_dir: str | None = None,
             engram_dir: str | None = None, cache_tokens: int | None = None, span: int | None = None,
             mem_per_device=None, budgets=None, dspark: bool | None = None, progress: bool | None = None,
             log: Callable[[str], None] = print) -> "TPModelBackend":
        """Two-node TP load, exactly like scripts/tp_ds41_generate.py (Engram tables included).

        With EXL3_TP_DSPARK=1 (or `dspark=True`), the trunk cache gets max_history=8, the default
        per-rank budgets become 95,104 GiB (rank 0 leaves VRAM for the drafter and the longer
        transients it adds; the same list is passed on both ranks, as in
        scripts/tp_ds41_dspark.py -- which used 101,110 before the memory-headroom fix) and
        rank 0 loads the drafter. After each load stage the rank logs its free GPU memory and
        refuses to serve below EXL3_TP_MEM_FLOOR (default 2 GiB): a starved rank would die at
        the first long prefill, inside a collective, and wedge the peer.
        The TP environment (EXL3_TP_NODE_RANK, EXLLAMA_MASTER_ADDR/PORT, NCCL_*, the Odinlink RCCL
        plugin paths) must already be set by the launcher wrappers tp_rank1.sh / tp_serve_rank0.sh.
        """
        from exllamav3 import Cache, Config, Model, Tokenizer          # heavy: only when really loading
        from exllamav3.constants import PAGE_SIZE

        model_dir = model_dir or os.environ.get("EXL3_TP_MODEL_DIR", DEFAULT_MODEL_DIR)
        engram_dir = engram_dir or os.environ.get("EXL3_TP_ENGRAM_DIR", DEFAULT_ENGRAM_DIR)
        cache_tokens = cache_tokens if cache_tokens is not None else env_int("EXL3_TP_CACHE_TOKENS", 2048)
        span = span if span is not None else env_int("EXL3_TP_SPAN", 2048)
        if dspark is None:
            dspark = env_truthy("EXL3_TP_DSPARK", False)
        if progress is None:
            progress = rank == 0

        # Engine bounds, validated BEFORE anything is dispatched (a request past these would
        # fault inside a collective and hang the peer rank's NCCL watchdog instead of erroring)
        if cache_tokens <= 0:
            raise ValueError(f"EXL3_TP_CACHE_TOKENS/cache_tokens must be positive, got {cache_tokens}")
        if cache_tokens % PAGE_SIZE:
            raise ValueError(f"EXL3_TP_CACHE_TOKENS/cache_tokens ({cache_tokens}) must be a multiple "
                             f"of the {PAGE_SIZE}-token page size (the paged DSA/draft kv pools "
                             f"allocate whole pages and the block tables assume it)")
        if span <= 0:
            raise ValueError(f"EXL3_TP_SPAN/span must be positive, got {span}")
        if span > cache_tokens:
            raise ValueError(f"EXL3_TP_SPAN/span ({span}) must not exceed the cache "
                             f"(EXL3_TP_CACHE_TOKENS={cache_tokens})")
        floor_gib = env_float("EXL3_TP_MEM_FLOOR", 2.0)

        # per-rank memory budgets (same list on both ranks; each rank's loader picks its entry)
        if budgets is not None or mem_per_device is not None:
            given = budgets if budgets is not None else mem_per_device
            use = [float(x) for x in given] if isinstance(given, (list, tuple)) else [float(given)] * world_size
        elif os.environ.get("EXL3_TP_BUDGETS"):
            use = [float(x) for x in os.environ["EXL3_TP_BUDGETS"].split(",")]
        else:
            use = [95.0, 104.0] if dspark else [env_float("EXL3_TP_USE_PER_DEVICE", 110.0)] * world_size
        if len(use) != world_size:
            raise ValueError(f"per-rank budgets must list {world_size} entries, got {use}")

        # draft-confidence threshold (read by the drafter at construction; 0.5 upstream default).
        # EXL3_TP_DSPARK_CONF / EXL3_DS41_CONF map onto the engine's EXL3_DSPARK_CONF, as the
        # script's EXL3_DS41_CONF does.
        for src in ("EXL3_TP_DSPARK_CONF", "EXL3_DS41_CONF"):
            if os.environ.get(src) and not os.environ.get("EXL3_DSPARK_CONF"):
                os.environ["EXL3_DSPARK_CONF"] = os.environ[src]

        config = Config.from_directory(model_dir)
        for filename in ENGRAM_SHARDS:
            config.stc.add_tensor_files(os.path.join(engram_dir, filename))
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as handle:
            weight_map = json.load(handle)["weight_map"]
        for layer in (1, 14):
            for name in ("q_weight", "k_weight"):
                key = f"layers.{layer}.engram.{name}"
                config.stc.tensor_file_map[key] = os.path.join(model_dir, weight_map[key])

        export_layers = frozenset(t - 1 for t in config.dspark_target_layer_ids)   # taps at {36, 37, 38}
        block = int(config.dspark_block_size)                                      # 5

        model = Model.from_config(config, component="text")
        # The cache must be attached before TP loading so cache layers are exported (as in the
        # scripts); dspark adds the recurrent rollback history the verify passes need.
        cache_kwargs = {"max_history": MAX_HISTORY} if dspark else {}
        cache = Cache(model, max_num_tokens=cache_tokens, max_batch_size=1, **cache_kwargs)
        log(f"tp: rank {rank} cache {cache_tokens} tokens ({cache_tokens // PAGE_SIZE} pages of "
            f"{PAGE_SIZE}); prefill chunks at {span} rows; block tables sized to the cache")
        started = time.time()
        model.load(
            tensor_p=True,
            tp_backend="nccl",
            use_per_device=use,
            max_chunk_size=span,
            progressbar=progress,
        )
        log(f"tp: rank {rank} cached TP load completed in {time.time() - started:.1f} s "
            f"(budgets {use} GiB, dspark {'on' if dspark else 'off'})")
        check_memory_floor("after the TP load", rank=rank, floor_gib=floor_gib, log=log)

        # ---- rank 0 only: the DSpark drafter, loaded beside the TP trunk (as the script) ----
        # (layer-split, no TP dispatches: rank 1's dispatch sequence is unaffected)
        draft_model = draft_cache = draft_bt = None
        if dspark and rank == 0:
            import torch
            report = _vram_report(env_int("EXL3_TP_LOCAL_DEVICE", 0))
            if report is not None:
                log(f"tp: rank 0 VRAM before draft load: {report[0]:.1f} GiB free of {report[1]:.1f} GiB")
            started = time.time()
            # The drafter only ever reads the main-kv rows of the last `sliding_window`
            # positions, so its cache is a ring of `ring` pages: logical page i lives in
            # physical page i % ring (reads and writes both go through draft_bt). The ring
            # must hold one prefill chunk plus the window (a chunk's rows must not alias
            # each other, and the window before a draft must survive the chunk written
            # just before it). EXL3_TP_DRAFT_RING=0 allocates the whole context again.
            window = int(config.sliding_window)
            cache_pages = cache_tokens // PAGE_SIZE
            ring = env_int("EXL3_TP_DRAFT_RING", span // PAGE_SIZE + 2)
            if ring <= 0 or ring >= cache_pages:
                ring = cache_pages
            elif ring * PAGE_SIZE < span + window:
                raise ValueError(f"EXL3_TP_DRAFT_RING={ring} pages cannot hold a {span}-row prefill "
                                 f"chunk plus the drafter's {window}-row window")
            draft_model = Model.from_config(config, component="mtp")
            draft_cache = Cache(draft_model, max_num_tokens=ring * PAGE_SIZE, max_batch_size=1)
            draft_model.load(progressbar=False)
            draft_model.attach_to(model)     # private embedding (CPU) + private head (GPU)
            assert draft_model.own_head is not None, "drafter did not get its private head (attach order bug)"
            draft_bt = (torch.arange(cache_pages, dtype=torch.int32) % ring).view(1, -1)
            ring_mib = sum(l.storage_size() for l in draft_cache.layers.values()) / 2**20
            log(f"tp: rank 0 DSpark drafter loaded in {time.time() - started:.1f} s "
                f"(block {block}, taps at layers {sorted(export_layers)}, max_history {MAX_HISTORY}; "
                f"main-kv {'ring of ' + str(ring) + ' pages' if ring < cache_pages else 'full context'}, "
                f"{ring_mib:.1f} MiB)")
            check_memory_floor("after the DSpark drafter load", rank=rank, floor_gib=floor_gib, log=log)

        tokenizer = Tokenizer.from_config(config)
        return cls(model, cache, tokenizer, span=span, cache_tokens=cache_tokens, rank=rank,
                   dspark=dspark, draft_model=draft_model, draft_cache=draft_cache, draft_bt=draft_bt,
                   export_layers=export_layers, dspark_block=block, page_size=PAGE_SIZE,
                   draft_window_rows=int(config.sliding_window))

    def new_state(self):
        import torch
        with torch.inference_mode():
            return self.cache.get_new_state()

    def release_state(self, state) -> None:
        self.cache.release_state(state)

    # ---- prefix cache: recurrent-state checkpoints (local to this rank, no collectives) ----
    #
    # The paged DSA pools are position-indexed, so after a request the pages for positions below
    # the state's end still hold that request's tokens. What a later request cannot rebuild from
    # the pages is the per-slot recurrent state (SWA ring, compressor partial windows, Engram id
    # ring), which DSV4State.stash() copies to host memory on every TP worker. Restoring a stash
    # at position c and prefilling only ids[c:] reproduces the state of a full prefill, provided no
    # request wrote positions below c since (TPEngine tracks that).

    def stash(self, state):
        import torch
        with torch.inference_mode():
            return state.stash()

    def restore(self, stashed, position: int):
        import torch
        with torch.inference_mode():
            return self.cache.new_from_stashed(stashed, int(position))

    def drop_stash(self, stashed) -> None:
        handle = stashed.get("tp_handle") if isinstance(stashed, dict) else None
        if handle is None or not getattr(self.model, "loaded_tp", False):
            return
        from exllamav3.cache.recurrent import mp_cache_recurrent_del
        self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self.cache), handle))

    def trim_memory(self) -> None:
        """Return the caching allocator's unused reserved segments to the GTT pool (both ranks).

        A long prompt's transient activations grow the reserved pool past the load-time
        reservation (fragmentation: differently-shaped blocks do not fit the freed holes), and
        the growth STAYS after the request -- on the 512K-cache deployment that cost ~1.2 GiB
        of system-available memory per box until the next reboot. Local-only, no collectives,
        so rank 0 (request end) and rank 1 (end frame) can each run it independently."""
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:                                                   # noqa: BLE001  (CPU tests, no GPU)
            pass

    def prefill(self, ids: list[int], state, spec: bool = False, start: int = 0):
        """Cached prefill of all but the last prompt token (the decode loop feeds that one),
        in `span`-row chunks, from position `start` (a restored prefix-cache checkpoint; the
        state must be at that position, 0 for a fresh state).

        Every chunk carries `batch_shape = (1, cache_tokens)`: the library sizes the DSA pool's
        block table from batch_shape (attn.prepare_flash_attn), and the old `(1, span)` table
        of span/256 pages overflowed on prompts past the span ("DSA pool overflow: entry ...
        beyond block table"). `past_len` is each chunk's absolute offset and must equal the
        recurrent state's position (the library asserts the match); one pass per chunk also
        bounds the activation transient to the loader's measured max_chunk_size instead of the
        full prompt length.

        With `spec=True` every chunk exports the DSpark taps at the target layers (identical
        dispatch on both ranks); rank 0 then projects that chunk's taps into the drafter's kv
        rows at the chunk offset immediately (per-position math, identical to the proven
        single-shot priming, but never holding more than `span` rows of taps). The tap states
        are only readable on the output rank (rank 0's TP worker runs in-process), so the
        export is requested identically on both ranks but validated/consumed on rank 0 only.
        The returned params carry `export_states` only when this backend did NOT consume them
        (the CPU test stand-ins), so the engine primes from the return value exactly as before."""
        import torch
        n = len(ids) - 1
        if n < 1:
            raise ValueError("prefill needs at least 2 token ids (all but the last are fed here)")
        if not 0 <= int(start) <= n:
            raise ValueError(f"prefill start {start} outside 0..{n}")
        last_params: dict | None = None
        for a in range(int(start), n, self.span):
            b = min(a + self.span, n)
            params = {
                "attn_mode": "flash_attn",
                "cache": self.cache,
                "batch_shape": (1, self.cache_tokens),
                "past_len": int(a),
                "recurrent_states": [state],
            }
            if spec:
                params["export_state_layers"] = self.export_layers
            self.model.prefill(torch.tensor(ids[a:b], dtype=torch.long).view(1, -1), params)
            last_params = params
            if spec:
                taps = params.get("export_states")
                if self.draft_model is not None:
                    # rank 0 only: the tap states are visible on the output rank (its TP worker
                    # runs in this process); rank 1's workers export into their own copies, and
                    # the export dispatch itself is what must match -- never the visibility
                    if not taps:
                        raise RuntimeError(
                            f"target tap export missing after spec prefill chunk [{a}:{b})")
                    self.draft_prefill_kv(taps, position=a)   # prime the drafter per chunk
                    params.pop("export_states", None)         # consumed; the engine must not re-prime
        return last_params or {}

    def verify(self, window: list[int], position: int, state):
        """One speculative verify forward over [anchor, drafts...] (multi-row, recurrent history
        on, taps exported) -- both ranks run it identically. The block table spans the whole
        cache, so verifies at any position stay inside the DSA pool. Returns (logits rows
        cropped to the real vocabulary, params) on the output rank, (None, params) elsewhere."""
        import torch
        verify_ids = torch.tensor([list(window)], dtype=torch.long)
        params = {
            "attn_mode": "flash_attn",
            "cache": self.cache,
            "batch_shape": (1, self.cache_tokens),
            "past_len": int(position),
            "recurrent_states": [state],
            "recurrent_history": True,
            "export_state_layers": self.export_layers,
        }
        logits = self.model.forward(verify_ids, params)
        if self.rank != self.OUTPUT_RANK or logits is None:
            return None, params
        return logits[0, :, : self.vocab_size].float(), params

    def rewind(self, state, steps: int) -> None:
        """Roll the recurrent state back `steps` tokens after a verify pass. A mirrored TP
        dispatch (mp_cache_recurrent_rewind): both ranks must call it with the same count."""
        state.rewind(int(steps))

    # ---- rank-0-only drafter ops (scripts/tp_ds41_dspark.py; no collectives) ----

    def draft(self, anchor: int, position: int, k_cap: int):
        """Forward the anchor through the drafter, sample the draft window and trim it to the
        drafter's confidence length and the caller's cap. Returns (k, drafts)."""
        import torch
        dparams = {
            "attn_mode": "flash_attn",
            "cache": self.draft_cache,
            "block_table": self.draft_bt,
            "cache_seqlens": torch.tensor([int(position)], dtype=torch.int32),
        }
        dstate = self.draft_model.forward(torch.tensor([int(anchor)]).view(1, 1), dparams)
        ids = self.draft_model.sample_from_state(dstate, dparams)
        conf_len = int(dparams.get("draft_confidence_len", self.dspark_block))
        k = max(0, min(self.dspark_block, conf_len, int(k_cap)))
        return k, [int(x) for x in ids[0, 1: 1 + k]]

    def draft_prefill_kv(self, taps, position: int = 0) -> None:
        """Project the trunk's prefill taps into the drafter's paged main-kv rows (rank 0).

        `position` is the absolute offset of the taps' rows; a chunked spec prefill primes each
        chunk at its own offset (write_rows places rows at cache_seqlens .. +s, and the tap
        projection is per-position, so this is identical to one big call at position 0)."""
        import torch
        self.draft_model.update_kv_from_target(
            target_hidden=taps,
            cache=self.draft_cache,
            params={
                "block_table": self.draft_bt,
                "cache_seqlens": torch.tensor([int(position)], dtype=torch.int32),
            },
        )

    def _draft_window_rows(self, position: int):
        """Physical main-kv rows of the drafter's window before `position` (None if empty)."""
        import torch
        lo = max(0, int(position) - self.draft_window_rows)
        if int(position) <= lo:
            return None
        pos = torch.arange(lo, int(position))
        pages = self.draft_bt[0].long()[pos // self.page_size]
        return pages * self.page_size + pos % self.page_size

    def draft_window(self, position: int):
        """Host copy of the drafter main-kv rows a draft at `position` reads (its sliding
        window), one tensor per drafter layer (rank 0). The main-kv is a ring that reuses a
        row once the sequence has moved a ring's length on, so a prefix-cache checkpoint
        carries this window and set_draft_window() puts it back at restore."""
        import torch
        rows = self._draft_window_rows(position)
        if rows is None or self.draft_cache is None:
            return None
        with torch.inference_mode():
            return [l.kv.view(-1, l.width)[rows.to(l.kv.device)].cpu() for l in self.draft_cache.layers.values()]

    def set_draft_window(self, saved, position: int) -> None:
        import torch
        rows = self._draft_window_rows(position)
        if rows is None or saved is None:
            return
        with torch.inference_mode():
            for l, t in zip(self.draft_cache.layers.values(), saved):
                l.kv.view(-1, l.width)[rows.to(l.kv.device)] = t.to(l.kv.device)

    def refresh_draft_kv(self, taps, accepted: int, position: int) -> None:
        """Refresh the drafter's main-kv from the accepted rows of one verify pass (rank 0)."""
        import torch
        self.draft_model.update_kv_from_target(
            target_hidden=taps,
            cache=self.draft_cache,
            lengths=[int(accepted)],
            params={
                "block_table": self.draft_bt,
                "cache_seqlens": torch.tensor([int(position) - int(accepted)], dtype=torch.int32),
            },
        )

    def unload(self) -> None:
        if self.draft_model is not None:
            self.draft_model.unload()
        self.model.unload()

    def forward(self, token: int, position: int, state):
        """One plain decode step: feed `token` with `past_len=position`. The block table spans
        the whole cache, so decode steps at any position stay inside the DSA pool. Returns the
        logits row cropped to the real vocabulary on the output rank, None on every other rank
        (cross-host TP returns no logits there by design -- reading them raised on rank 1 and
        wedged the collectives)."""
        import torch
        logits = self.model.forward(
            torch.tensor(int(token), dtype=torch.long).view(1, 1),
            params={
                "attn_mode": "flash_attn",
                "cache": self.cache,
                "batch_shape": (1, self.cache_tokens),
                "past_len": int(position),
                "recurrent_states": [state],
                "last_tokens_only": 1,
            },
        )
        if self.rank != self.OUTPUT_RANK:
            return None
        return logits[0, -1, : self.vocab_size]


def _prefill(backend, ids: list[int], state, spec: bool, start: int):
    """backend.prefill from `start` (the CPU stand-ins without prefix-cache support never get one)."""
    if start:
        return backend.prefill(ids, state, spec=spec, start=start)
    return backend.prefill(ids, state, spec=spec)


# ---- rank 1: the persistent mirror -------------------------------------------------------------------------


class Rank1Worker:
    """Runs on Box2: performs exactly the model calls rank 0 makes, driven by control frames.

    Per request (`begin`): fresh state -> prefill (with the DSpark tap export when `spec` is set)
    -> in plain mode forward #1 of the last prompt token, in spec mode nothing further (the
    round window carries the anchor). Plain requests then take one `step` per token; spec
    requests take `round` (one multi-row verify) + `accept` (one mirrored rewind) per round.
    `end` releases the state. The mode is fixed per request: a request picks plain or spec at
    its settled start boundary and never switches mid-flight, so the collective sequences of
    the two ranks stay identical (see the module docstring for the full lockstep rules).
    """

    def __init__(self, backend, channel, *, log: Callable[[str], None] = print):
        self.backend = backend
        self.channel = channel
        self.log = log
        self.state = None
        self.position = -1
        self.request_id = None
        self.requests_served = 0
        self.spec = False          # the active request's mode (plain steps vs spec rounds)
        self.round_k = None        # k of the round frame awaiting its accept
        self.prompt_len = 0        # prefilled rows of the active request (memory-trim gating at end)
        self.stashes: dict[int, tuple[int, Any]] = {}   # prefix-cache checkpoints: key -> (position, stash)

    def handshake(self) -> dict:
        hello = self.channel.recv()
        if hello.get("op") != "hello":
            raise ProtocolError(f"rank 1: expected hello, got {hello.get('op')!r}")
        if int(hello.get("proto", -1)) != PROTO:
            raise ProtocolError(
                f"rank 1: control protocol mismatch (rank 0 sent {hello.get('proto')!r}, this worker speaks {PROTO})")
        self.channel.send({"op": "hello_ok", "proto": PROTO, "rank": 1,
                           "vocab_size": int(self.backend.vocab_size),
                           "dspark": bool(getattr(self.backend, "dspark", False))})
        self.log("tp: rank 1 handshake complete")
        return hello

    def serve_forever(self) -> None:
        self.handshake()
        while True:
            try:
                msg = self.channel.recv()
            except ChannelClosed:
                self.log("tp: rank 1 control channel closed by rank 0")
                self.drop_state()
                return
            op = msg.get("op")
            try:
                if op == "begin":
                    self._begin(msg)
                elif op == "step":
                    self._step(msg)
                elif op == "round":
                    self._round(msg)
                elif op == "accept":
                    self._accept(msg)
                elif op == "end":
                    self._end(msg)
                elif op == "ping":
                    self.channel.send({"op": "pong"})
                elif op == "shutdown":
                    self.log("tp: rank 1 shutting down on rank 0's request")
                    self.drop_state()
                    self.channel.close()
                    return
                else:
                    raise ProtocolError(f"rank 1: unknown op {op!r}")
            except Exception as exc:                                   # noqa: BLE001  (reported to rank 0, then fatal)
                try:
                    self.channel.send({"op": "error", "message": f"{type(exc).__name__}: {exc}",
                                       "req": self.request_id})
                except OSError:
                    pass
                raise

    def _begin(self, msg: dict) -> None:
        if self.state is not None:
            raise ProtocolError("rank 1: begin while a request is still active")
        ids = [int(t) for t in msg["ids"]]
        if len(ids) < 2:
            raise ValueError("rank 1: prompt must hold at least 2 token ids")
        bad = [t for t in ids if not 0 <= t < self.backend.vocab_size]
        if bad:
            raise ValueError(f"rank 1: prompt token id {bad[0]} outside the vocabulary")
        # Cache bound, checked BEFORE any model call: past it the DSA pool faults inside a
        # collective and the peer rank hangs on NCCL until its watchdog kills it
        cap = int(getattr(self.backend, "cache_tokens", 0) or 0)
        if cap and len(ids) + 2 > cap:
            raise ValueError(f"rank 1: prompt of {len(ids)} tokens exceeds the cache bound "
                             f"({cap - 2} usable of EXL3_TP_CACHE_TOKENS={cap})")
        spec = bool(msg.get("spec"))
        if spec and not getattr(self.backend, "dspark", False):
            raise ProtocolError("rank 1: spec request but this rank did not load DSpark (EXL3_TP_DSPARK)")
        if msg.get("drop_all"):
            self._drop(list(self.stashes))           # rank 0 reset its prefix cache (cache_salt change)
        start = int(msg.get("start") or 0)
        restore = msg.get("restore")
        if restore is not None:
            entry = self.stashes.get(int(restore))
            if entry is None:
                raise ProtocolError(f"rank 1: restore of unknown checkpoint {restore}")
            if entry[0] != start or not 0 < start <= len(ids) - 1:
                raise ProtocolError(f"rank 1: checkpoint {restore} is at position {entry[0]}, "
                                    f"begin says start {start} (prompt of {len(ids)})")
        elif start:
            raise ProtocolError("rank 1: begin with a start position but no checkpoint to restore")
        self.spec = spec
        self.request_id = msg.get("req")
        self.prompt_len = len(ids) - start
        # the same order as rank 0: restore (or a fresh state), drops, prefill, save
        self.state = self.backend.restore(entry[1], start) if restore is not None else self.backend.new_state()
        self._drop(msg.get("drop"))
        _prefill(self.backend, ids, self.state, spec, start)
        self.position = len(ids) - 1
        if msg.get("save") is not None:
            self._save(int(msg["save"]))
        if not spec:
            self.backend.forward(ids[-1], self.position, self.state)
            self.position += 1
        # spec mode: no first forward -- the round window carries the anchor at this position

    def _step(self, msg: dict) -> None:
        if self.state is None:
            raise ProtocolError("rank 1: step without begin")
        if self.spec:
            raise ProtocolError("rank 1: step during a spec request (mode is fixed per request)")
        self.backend.forward(int(msg["token"]), self.position, self.state)
        self.position += 1

    def _round(self, msg: dict) -> None:
        if self.state is None or not self.spec:
            raise ProtocolError("rank 1: round without a spec request")
        if self.round_k is not None:
            raise ProtocolError("rank 1: round before the previous round's accept")
        k = int(msg["k"])
        window = [int(t) for t in msg["window"]]
        if not 0 <= k <= getattr(self.backend, "dspark_block", 5) or len(window) != k + 1:
            raise ProtocolError(f"rank 1: malformed round frame (k={k}, window {len(window)})")
        self.round_k = k
        self.backend.verify(window, self.position, self.state)

    def _accept(self, msg: dict) -> None:
        if self.state is None or not self.spec or self.round_k is None:
            raise ProtocolError("rank 1: accept without round")
        a = int(msg["accepted"])
        if not 0 <= a <= self.round_k:
            raise ProtocolError(f"rank 1: accepted {a} outside 0..{self.round_k}")
        self.backend.rewind(self.state, self.round_k - a)
        self.position += a + 1
        self.round_k = None

    def _save(self, key: int) -> None:
        if key in self.stashes:
            raise ProtocolError(f"rank 1: checkpoint {key} already exists")
        self.stashes[key] = (self.position, self.backend.stash(self.state))

    def _drop(self, keys) -> None:
        # unknown keys are fine: rank 0 also drops what a failed request may or may not have saved here
        for key in keys or ():
            entry = self.stashes.pop(int(key), None)
            if entry is not None:
                self.backend.drop_stash(entry[1])

    def _end(self, msg: dict | None = None) -> None:
        if self.state is None:
            raise ProtocolError("rank 1: end without begin")
        if self.round_k is not None:
            raise ProtocolError("rank 1: end with a round still awaiting its accept")
        msg = msg or {}
        self._drop(list(self.stashes) if msg.get("drop_all") else msg.get("drop"))
        if msg.get("save") is not None:
            back = int(msg.get("rewind") or 0)
            limit = min(self.position, int(getattr(self.backend, "dspark_block", 5)))
            if not 0 <= back <= limit:
                raise ProtocolError(f"rank 1: end rewind {back} outside 0..{limit}")
            if back:
                self.backend.rewind(self.state, back)      # to the end of the emitted reply (spec)
                self.position -= back
            self._save(int(msg["save"]))
        self.backend.release_state(self.state)
        self.state = None
        self.position = -1
        self.requests_served += 1
        self.request_id = None
        self.spec = False
        self.round_k = None
        trim = getattr(self.backend, "trim_memory", None)    # rank 0 trims on long requests;
        if trim is not None and self.prompt_len > getattr(self.backend, "span", 0):
            trim()                                           # rank 1 mirrors at end (local-only op)
        self.prompt_len = 0

    def drop_state(self) -> None:
        """Release the request state (end of request, shutdown, or a dead channel)."""
        if self.state is not None:
            try:
                self.backend.release_state(self.state)
            except Exception:                                       # noqa: BLE001
                pass
            self.state = None
            self.position = -1
        self.spec = False
        self.round_k = None
        self.prompt_len = 0


# ---- rank 0: prompt reuse across requests ------------------------------------------------------------------


def _common_prefix(a: array, b: array) -> int:
    """Length of the common prefix of two int arrays (slice compares run at C speed)."""
    n = min(len(a), len(b))
    i = 0
    while i < n:
        j = min(i + 4096, n)
        if a[i:j] != b[i:j]:
            while a[i] == b[i]:
                i += 1
            return i
        i = j
    return n


class Checkpoint:
    """One stashed recurrent state: the trunk state after its first `pos` tokens (`ids`), plus
    (spec requests, rank 0) the drafter's main-kv window before `pos`, or None if the drafter
    had not seen those positions."""

    __slots__ = ("key", "pos", "ids", "stash", "draft", "used")

    def __init__(self, key: int, pos: int, ids: array, stash: Any, used: int, draft: Any = None):
        self.key = key
        self.pos = pos
        self.ids = ids
        self.stash = stash
        self.draft = draft
        self.used = used


class PrefixCache:
    """Rank 0's record of what the trunk cache holds, for prompt reuse across requests.

    The cache is one linear sequence: every trunk call uses the identity block table over the
    whole cache, so the paged DSA pools are position-indexed and after a request positions
    [0, E) still hold the kv of the E tokens it committed (`kv_ids`). A new prompt that shares a
    prefix with them only needs the recurrent state at the restart point (SWA ring, compressor
    windows and overlaps, Engram id ring) to skip that prefix: a checkpoint, stashed by both
    ranks after each prefill (position len(ids) - 1: through `<｜Assistant｜>`, the part of the
    prompt the next turn's template renders identically) and at the end of each request (the
    reply too, for templates that echo it verbatim, e.g. tool conversations).

    The drafter (rank 0) reads only the main-kv rows of the last `sliding_window` positions,
    and its main-kv is a ring that reuses those rows once the sequence moves on, so a spec
    checkpoint also carries that window (`Checkpoint.draft`) and a restore puts it back. A
    checkpoint taken where the drafter had not seen the positions (a plain request) has no
    window: restoring it for a spec request is still exact, only the first drafts are poor.

    Invariant: the token prefix of every retained checkpoint is a prefix of `kv_ids`, so a
    checkpoint at pos <= common_prefix(kv_ids, prompt) is valid for the prompt. `finish` drops
    the checkpoints a request's writes invalidated; `reserve` evicts the least recently used
    one when all `slots` are taken."""

    def __init__(self, *, slots: int = 2, min_tokens: int = 256, save_end: bool = True):
        self.slots = max(1, int(slots))
        self.min_tokens = max(1, int(min_tokens))
        self.save_end = bool(save_end)
        self.kv_ids = array("i")
        self.ckpts: list[Checkpoint] = []
        self._key = 0
        self._tick = 0

    @classmethod
    def from_env(cls) -> "PrefixCache | None":
        if not env_truthy("EXL3_TP_PREFIX_CACHE", True):
            return None
        # 2 slots: the last prompt and the last reply end, which is what a chat or tool loop
        # resumes from. Each checkpoint is 32 MiB of host RAM per rank, and both boxes run with
        # under 2 GiB available (the unified-memory GPU allocations take the rest)
        return cls(slots=env_int("EXL3_TP_PREFIX_SLOTS", 2), min_tokens=env_int("EXL3_TP_PREFIX_MIN", 256),
                   save_end=env_truthy("EXL3_TP_PREFIX_END", True))

    def _touch(self, c: Checkpoint) -> None:
        self._tick += 1
        c.used = self._tick

    def plan(self, ids: array) -> Checkpoint | None:
        """The longest checkpoint valid for this prompt's prefill (ids[:-1]), or None."""
        limit = min(_common_prefix(self.kv_ids, ids), len(ids) - 1)
        best = None
        for c in self.ckpts:
            if c.pos <= limit and (best is None or c.pos > best.pos):
                best = c
        if best is not None:
            self._touch(best)
        return best

    def reserve(self, protect: Checkpoint | None = None) -> tuple[int, list[Checkpoint]]:
        """A key for one new checkpoint, and the checkpoints evicted (LRU) to make room."""
        evicted = []
        while len(self.ckpts) >= self.slots:
            pool = [c for c in self.ckpts if c is not protect]
            if not pool:
                break
            victim = min(pool, key=lambda c: c.used)
            self.ckpts.remove(victim)
            evicted.append(victim)
        self._key += 1
        return self._key, evicted

    def add(self, key: int, ids: array, stash: Any, draft: Any = None) -> Checkpoint:
        c = Checkpoint(key, len(ids), ids, stash, 0, draft)
        self._touch(c)
        self.ckpts.append(c)
        return c

    def finish(self, committed: array) -> list[Checkpoint]:
        """A request committed `committed` (positions [0, E)); returns the checkpoints it invalidated."""
        self.kv_ids = committed
        dead = [c for c in self.ckpts if c.pos > len(committed) or committed[:c.pos] != c.ids]
        for c in dead:
            self.ckpts.remove(c)
        return dead

    def fail(self, start: int) -> list[Checkpoint]:
        """A request failed after writing from `start`: only positions below it are still known."""
        self.kv_ids = self.kv_ids[:start]
        dead = [c for c in self.ckpts if c.pos > start]
        for c in dead:
            self.ckpts.remove(c)
        return dead


PREFIX_LOG = env_truthy("EXL3_TP_PREFIX_LOG", False)


# ---- rank 0: the engine the vendored server talks to -------------------------------------------------------


class TPEngine:
    """`ds41_engine.Engine` over the two-node model: rank 0 renders, prefills, samples and streams.

    `generate` yields the decoded text of the reply with special tokens included (the vendored
    parser splits `</think>` and the DSML tool markup out of that text), exactly like the engine
    contract demands. Stop conditions all land on token boundaries; the stats keys match the
    vendored server (`cached_tokens` counts the prompt tokens a prefix-cache checkpoint skipped;
    `cache_prompt: false` on a request turns reuse off for that request only).

    When the backend loaded DSpark (`EXL3_TP_DSPARK=1`), requests default to speculative rounds
    (per-request `speculative: false` opts out, `spec_k` caps the draft window): greedy requests
    accept by prefix match, sampled ones by exact speculative sampling (EXL3_TP_SPEC_SAMPLED=0
    keeps them on the plain path). `speculative: true` on a server without DSpark is refused
    before the job starts.
    """

    sessions = 1

    def __init__(self, backend, channel, tokenizer, *, ctx: int, eos_id: int | None = None,
                 seed: int | None = None, log: Callable[[str], None] = print,
                 prefix: "PrefixCache | bool | None" = None):
        self.backend = backend
        self.channel = channel
        self.tk = tokenizer
        self.log = log
        self.ctx = int(ctx)
        self.vocab_size = int(backend.vocab_size)
        self.eos_id = eos_id
        self.dspark = bool(getattr(backend, "dspark", False))
        self.requests = 0
        self.last_stats: dict[str, Any] = {}
        self._gen = _torch_generator(seed)
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="tp-model")
        self._busy = threading.Lock()     # held while a request drives rank 1; pings wait it out
        # prompt reuse needs the backend's checkpoint ops (the CPU stand-ins mostly have none)
        if prefix is None or prefix is True:
            prefix = PrefixCache.from_env()
        self.prefix = prefix or None
        if self.prefix is not None and not all(hasattr(backend, op) for op in ("stash", "restore", "drop_stash")):
            self.prefix = None
        self._drop_all_next = False       # reset_prefix_cache ran: rank 1 drops its copies at the next begin

    # -- tokenization (Kyojin HFTokenizer semantics: specials parsed from the prompt text, no extra BOS)
    def count_tokens(self, text: str) -> int:
        return len(self.tk.encode(text))

    def prompt_ids(self, prompt: str) -> list[int]:
        ids = [int(t) for t in self.tk.encode(prompt)]
        if not ids:
            raise ValueError("empty prompt")
        if max(ids) >= self.vocab_size:
            raise ValueError(f"prompt token id {max(ids)} outside the vocabulary (0..{self.vocab_size - 1})")
        return ids

    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(t) for t in self.tk.encode(text, add_bos=add_special_tokens)]

    # -- liveness (the vendored /tp/status route of tp_serve.py calls this)
    def rank1_ping(self, timeout: float = 30.0) -> str:
        if not self._busy.acquire(blocking=False):
            return "busy"
        try:
            reply = self.channel.request({"op": "ping"}, timeout=timeout)
            return "ok" if reply.get("op") == "pong" else f"unexpected:{reply.get('op')}"
        except ChannelClosed:
            return "disconnected"
        except (OSError, socket.timeout) as exc:
            return f"error:{type(exc).__name__}"
        finally:
            self._busy.release()

    def warmup(self, prompt_tokens: int = 32, decode_tokens: int = 4) -> float:
        """One tiny request end to end (both ranks) before serving, in the server's default mode;
        with DSpark on, a second one exercises the plain fallback path too."""
        n = max(8, min(int(prompt_tokens), self.ctx // 4))
        lo = min(1000, self.vocab_size - 1)
        ids = [(lo + i * 7919) % self.vocab_size or 1 for i in range(n)]
        t0 = time.perf_counter()

        async def go(speculative):
            async for _ in self.generate("", max_tokens=int(decode_tokens), temperature=0.0, top_p=1.0,
                                         stop=[], prompt_ids=ids, ignore_eos=True, speculative=speculative):
                pass

        asyncio.run(go(None))
        if self.dspark:
            asyncio.run(go(False))
        return time.perf_counter() - t0

    # -- prefix cache (rank 0 decides; rank 1 mirrors the begin/end frame fields in the same order:
    #    restore or fresh state, drops, prefill, save ... end: drops, save, release)
    def _prefix_begin(self, ids: list[int], spec: bool, use_cache: bool, frame: dict):
        """Plan the request's prompt reuse and add it to the begin frame. Returns (checkpoint to
        restore or None, start position, key to save the post-prefill state under or None,
        checkpoints evicted to make room for it)."""
        px = self.prefix
        if self._drop_all_next:
            frame["drop_all"] = True
            self._drop_all_next = False
        if px is None or not use_cache:
            return None, 0, None, []
        ck = px.plan(array("i", ids))
        start = ck.pos if ck is not None else 0
        save, evicted = None, []
        if len(ids) - 1 >= px.min_tokens and len(ids) - 1 > start:
            save, evicted = px.reserve(protect=ck)
        if ck is not None:
            frame["restore"], frame["start"] = ck.key, start
        if evicted:
            frame["drop"] = [c.key for c in evicted]
        if save is not None:
            frame["save"] = save
        return ck, start, save, evicted

    def _prefix_state(self, ck: Checkpoint | None, start: int, evicted: list[Checkpoint],
                      spec: bool = False):
        """The request's state (restored from `ck` or fresh), and for spec requests the first
        position whose drafter main-kv row is known to belong to this sequence: 0 for a fresh
        state (the spec prefill writes every row), the start of the restored window for a
        checkpoint that carries one (put back here), `start` for one that does not."""
        state = self.backend.restore(ck.stash, start) if ck is not None else self.backend.new_state()
        for c in evicted:
            self.backend.drop_stash(c.stash)
        if not spec:
            return state, None
        if ck is None:
            return state, 0
        if ck.draft is not None and hasattr(self.backend, "set_draft_window"):
            self.backend.set_draft_window(ck.draft, start)
            return state, max(0, start - int(getattr(self.backend, "draft_window_rows", 0)))
        return state, start

    def _draft_window_at(self, pos: int, draft_from: int | None):
        """The drafter window to keep with a checkpoint at `pos`, if every row of it is known."""
        if draft_from is None or not hasattr(self.backend, "draft_window"):
            return None
        if draft_from > max(0, pos - int(getattr(self.backend, "draft_window_rows", 0))):
            return None
        return self.backend.draft_window(pos)

    def _prefix_save(self, key: int | None, state, ids: list[int], draft_from: int | None = None) -> None:
        if key is not None:
            n = len(ids) - 1
            self.prefix.add(key, array("i", ids[:n]), self.backend.stash(state), self._draft_window_at(n, draft_from))

    def _prefix_end(self, state, frame: dict, start: int, spec: bool, committed: list[int] | None,
                    pending: int | None, use_cache: bool, end_pos: int | None = None,
                    draft_from: int | None = None) -> None:
        """Settle the record after a request, before its state is released: `committed` is the
        token sequence now at positions [0, E) (None: the request failed). Drops the checkpoints
        its writes invalidated, saves the final state, and adds both to the end frame. Runs for
        every request, cache_prompt or not: any request overwrites the cached sequence.

        `end_pos` (spec requests) is where the emitted reply ends: a round cut short by eos /
        stop / max_tokens committed up to `dspark_block` accepted drafts past it, which the
        client never sees and so never sends back. The state rewinds there before the save
        (both ranks: the end frame's "rewind"), so the checkpoint matches the next turn."""
        px = self.prefix
        if px is None:
            return
        if committed is None:
            dead = px.fail(start)
            keys = [c.key for c in dead] + ([pending] if pending is not None else [])
        else:
            seq = array("i", committed)
            dead = px.finish(seq)
            keys = [c.key for c in dead]
        for c in dead:
            self.backend.drop_stash(c.stash)
        save_at = None
        if committed is not None:
            save_at = len(committed) if end_pos is None else min(int(end_pos), len(committed))
        if (save_at is not None and state is not None and use_cache and px.save_end
                and save_at >= px.min_tokens and all(c.pos != save_at for c in px.ckpts)):
            key, evicted = px.reserve()
            for c in evicted:
                self.backend.drop_stash(c.stash)
            keys += [c.key for c in evicted]
            back = len(seq) - save_at
            if back:
                self.backend.rewind(state, back)
                frame["rewind"] = back
            px.add(key, seq[:save_at], self.backend.stash(state), self._draft_window_at(save_at, draft_from))
            frame["save"] = key
        if keys:
            frame["drop"] = keys
        if PREFIX_LOG:
            self.log(f"tp: prefix cache: request {self.requests} reused {start} of {len(committed or ())} "
                     f"committed tokens; checkpoints at {sorted(c.pos for c in px.ckpts)}"
                     f"{f', dropped {keys}' if keys else ''}")

    # -- one request (the model thread; the op sequence Rank1Worker mirrors)
    def _run(self, ids: list[int], max_tokens: int, sampler: TopPSampler, eos_id: int | None,
             stops: list[str], cancel: asyncio.Event | None, emit: Callable[[str], None],
             use_cache: bool = True) -> dict[str, Any]:
        begin = {"op": "begin", "req": self.requests, "ids": [int(t) for t in ids]}
        ck, start, pending, evicted = self._prefix_begin(ids, False, use_cache, begin)
        self.channel.send(begin)
        end = {"op": "end", "req": self.requests}
        state = None
        committed = None
        n_new = 0
        eos = False
        reason = "max_new_tokens"
        t0 = time.perf_counter()
        t_first: float | None = None
        detok = _Detok(getattr(self.tk, "_tk", self.tk))
        text = ""
        try:
            state, _ = self._prefix_state(ck, start, evicted)
            _prefill(self.backend, ids, state, False, start)
            self._prefix_save(pending, state, ids)
            pending = None
            cur, pos = int(ids[-1]), len(ids) - 1
            fed = list(ids)              # the tokens at positions [0, pos) once `cur` is forwarded
            # Lockstep rule: every stop decision happens after sampling, before the step frame.
            while True:
                row = self.backend.forward(cur, pos, state)
                pos += 1
                if t_first is None:
                    t_first = time.perf_counter()
                token = int(sampler.sample(row))
                if eos_id is not None and token == eos_id:
                    eos, reason = True, "eos"
                    break
                piece = detok.push([token])
                text += piece
                if piece:
                    emit(piece)
                n_new += 1
                if stops and stop_text(text, stops)[1] is not None:
                    reason = "stop_word"
                    break
                if n_new >= max_tokens:
                    reason = "max_new_tokens"
                    break
                if cancel is not None and cancel.is_set():
                    reason = "cancelled"
                    break
                self.channel.send({"op": "step", "req": self.requests, "token": token})
                cur = token
                fed.append(token)
            committed = fed[:pos]
            tail = detok.flush()
            if tail:
                text += tail
                emit(tail)
        finally:
            self._finish_request(state, end, start, False, committed, pending, use_cache, len(ids))
        t1 = time.perf_counter()
        t_first = t_first if t_first is not None else t1
        return {"cached_tokens": start, "new_tokens": n_new, "prompt_tokens": len(ids),
                "time_prefill": t_first - t0, "time_generate": t1 - t_first,
                "eos": eos, "eos_reason": reason, "spec_used": False}

    def _finish_request(self, state, end: dict, start: int, spec: bool, committed: list[int] | None,
                        pending: int | None, use_cache: bool, n_ids: int, end_pos: int | None = None,
                        draft_from: int | None = None) -> None:
        """Request teardown on rank 0 (both paths): prefix record and end-of-request checkpoint,
        state release, memory trim, then the end frame that makes rank 1 do the same."""
        try:
            self._prefix_end(state, end, start, spec, committed, pending, use_cache, end_pos, draft_from)
        except Exception as exc:                                        # noqa: BLE001
            # the record no longer matches what the ranks hold: forget every checkpoint
            self.log(f"tp: prefix cache bookkeeping failed ({type(exc).__name__}: {exc}); clearing it")
            self._prefix_reset(end)
        if state is not None:
            try:
                self.backend.release_state(state)
            except Exception as exc:                                    # noqa: BLE001
                self.log(f"tp: releasing request state failed: {type(exc).__name__}: {exc}")
        if n_ids - start > getattr(self.backend, "span", 0) and hasattr(self.backend, "trim_memory"):
            self.backend.trim_memory()          # a chunked prompt grew the reserved pool; return it
        try:
            self.channel.send(end)
        except OSError as exc:                                          # noqa: BLE001  (the request is failing anyway)
            self.log(f"tp: sending end to rank 1 failed: {type(exc).__name__}: {exc}")

    async def reset_prefix_cache(self) -> None:
        """cache_salt namespace change (the vendored server calls this between requests, with
        nothing in flight): forget every checkpoint and the cached sequence, so no request of
        the new namespace reuses anything of the old one. Rank 1 drops its copies at the next
        begin ("drop_all")."""
        if self.prefix is not None:
            await asyncio.get_running_loop().run_in_executor(self._pool, self._prefix_clear)

    def _prefix_clear(self) -> None:
        with self._busy:
            px = self.prefix
            for c in px.ckpts:
                self.backend.drop_stash(c.stash)
            px.ckpts.clear()
            px.kv_ids = array("i")
            self._drop_all_next = True

    def _prefix_reset(self, end: dict) -> None:
        px = self.prefix
        for c in px.ckpts:
            try:
                self.backend.drop_stash(c.stash)
            except Exception:                                           # noqa: BLE001
                pass
        px.ckpts.clear()
        px.kv_ids = array("i")
        end.pop("save", None)
        end.pop("rewind", None)
        end.pop("drop", None)
        end["drop_all"] = True          # rank 1 may hold keys this request reserved: drop everything

    # -- one spec request (the model thread; mirrors the round/accept sequence of Rank1Worker)
    def _run_spec(self, ids: list[int], max_tokens: int, stops: list[str], cancel: asyncio.Event | None,
                  emit: Callable[[str], None], spec_k: int | None, eos_id: int | None,
                  sampler: TopPSampler | None = None, use_cache: bool = True) -> dict[str, Any]:
        import torch                                                    # the whole request runs under inference_mode (as the script)
        begin = {"op": "begin", "req": self.requests, "ids": [int(t) for t in ids], "spec": True}
        ck, start, pending, evicted = self._prefix_begin(ids, True, use_cache, begin)
        self.channel.send(begin)
        end = {"op": "end", "req": self.requests}
        state = None
        committed = end_pos = draft_from = None
        n_new = 0
        eos = False
        reason = "max_new_tokens"
        rounds = drafted_total = accepted_total = 0
        t0 = time.perf_counter()
        t_first: float | None = None
        detok = _Detok(getattr(self.tk, "_tk", self.tk))
        text = ""
        block = int(self.backend.dspark_block)
        try:
            with torch.inference_mode():
                state, draft_from = self._prefix_state(ck, start, evicted, spec=True)
                pparams = _prefill(self.backend, ids, state, True, start)
                taps = (pparams or {}).get("export_states")
                if taps:
                    # Stand-ins (and any backend that does not prime its own drafter): prime
                    # once from the returned taps, as before. The real backend validates each
                    # chunk's export and primes per chunk inside prefill().
                    self.backend.draft_prefill_kv(taps)
                self._prefix_save(pending, state, ids, draft_from)
                pending = None
                t_first = time.perf_counter()
                anchor, position = int(ids[-1]), len(ids) - 1
                seq = list(ids)          # seq[:position] is committed, seq[position] the anchor
                # All stop decisions happen after a round has settled on both ranks (post rewind):
                # the next frame rank 1 sees is either another round or end, never a half-round.
                while True:
                    if cancel is not None and cancel.is_set():
                        reason = "cancelled"
                        break
                    budget = max_tokens - n_new
                    if budget <= 0:
                        reason = "max_new_tokens"
                        break
                    k_cap = min(int(spec_k) if spec_k else block, budget - 1)
                    k, drafts = self.backend.draft(anchor, position, k_cap)
                    self.channel.send({"op": "round", "req": self.requests, "k": int(k),
                                       "window": [int(anchor)] + [int(t) for t in drafts]})
                    logits, vparams = self.backend.verify([int(anchor)] + [int(t) for t in drafts],
                                                          position, state)
                    rounds += 1
                    drafted_total += k
                    if logits is None:
                        raise RuntimeError("spec verify returned no logits on the output rank")
                    if sampler is not None and sampler.stochastic:
                        a, bonus = _accept_sampled(logits, drafts, k, sampler)
                    else:
                        greedy = logits.argmax(dim=-1).tolist()
                        a = 0
                        while a < k and int(drafts[a]) == greedy[a]:
                            a += 1
                        bonus = int(greedy[a])
                    self.channel.send({"op": "accept", "req": self.requests, "accepted": int(a)})
                    self.backend.rewind(state, k - a)
                    position += a + 1
                    seq += [int(t) for t in drafts[:a]] + [int(bonus)]
                    accepted_total += a
                    self.backend.refresh_draft_kv((vparams or {}).get("export_states"), a + 1, position)
                    # Emit the round's tokens; eos/stop/max cut inside the round are safe: both
                    # ranks already settled (rewind done), the state is discarded at end anyway.
                    done = False
                    for tok in [int(t) for t in drafts[:a]] + [bonus]:
                        if eos_id is not None and tok == eos_id:
                            eos, reason, done = True, "eos", True
                            break
                        piece = detok.push([tok])
                        text += piece
                        if piece:
                            emit(piece)
                        n_new += 1
                        if stops and stop_text(text, stops)[1] is not None:
                            reason, done = "stop_word", True
                            break
                        if n_new >= max_tokens:
                            reason, done = "max_new_tokens", True
                            break
                    if done:
                        break
                    anchor = bonus
                committed = seq[:position]
                end_pos = min(position, len(ids) + n_new)   # where the emitted reply ends
                tail = detok.flush()
                if tail:
                    text += tail
                    emit(tail)
        finally:
            self._finish_request(state, end, start, True, committed, pending, use_cache, len(ids), end_pos,
                                 draft_from)
        t1 = time.perf_counter()
        t_first = t_first if t_first is not None else t1
        return {"cached_tokens": start, "new_tokens": n_new, "prompt_tokens": len(ids),
                "time_prefill": t_first - t0, "time_generate": t1 - t_first,
                "eos": eos, "eos_reason": reason, "spec_used": True,
                "spec_rounds": rounds, "spec_k_last": k if rounds else None,
                "accepted_draft_tokens": accepted_total, "rejected_draft_tokens": drafted_total - accepted_total,
                "draft_steps": rounds,
                "accepted_per_step": (accepted_total / rounds) if rounds else 0.0}

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float,
                       stop: list[str], cancel: asyncio.Event | None = None, stats: dict | None = None,
                       prompt_ids: list[int] | None = None, ignore_eos: bool = False, cache_prompt: bool = True,
                       images: list | None = None, speculative: bool | None = None,
                       spec_k: int | None = None, **_: Any) -> AsyncIterator[str]:
        if images:
            raise ValueError("this engine has no vision support")
        ids = [int(t) for t in prompt_ids] if prompt_ids is not None else self.prompt_ids(prompt)
        if len(ids) < 2:
            raise ValueError("prompt must hold at least 2 token ids (prefill feeds all but the last)")
        if len(ids) + 1 > self.ctx:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit the context of {self.ctx}")
        # Engine/library bounds, checked BEFORE the begin frame reaches rank 1: past the cache
        # bound the DSA pool faults inside a collective and rank 1's NCCL peer (rank 0) hangs
        # for its full watchdog timeout instead of seeing an error
        cache_cap = int(getattr(self.backend, "cache_tokens", 0) or 0)
        if cache_cap and len(ids) + 2 > cache_cap:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds the engine cache bound of "
                             f"{cache_cap - 2} usable tokens (EXL3_TP_CACHE_TOKENS={cache_cap}); "
                             f"refused before any rank dispatch")
        max_tokens = max(1, min(int(max_tokens), self.ctx - len(ids) - 1))
        # Spec mode is chosen here, once per request, at the settled start boundary. Sampled
        # requests draft too (exact speculative sampling, see _accept_sampled) unless
        # EXL3_TP_SPEC_SAMPLED=0 restores the greedy-only rule; an explicit opt-out always runs
        # the unchanged plain path whatever the server default is.
        spec = self.dspark if speculative is None else bool(speculative)
        if spec and not self.dspark:
            raise ValueError("speculative decoding is not enabled on this server (start both ranks with EXL3_TP_DSPARK=1)")
        if spec and float(temperature) > 0.0 and not SPEC_SAMPLED:
            spec = False
        if spec and spec_k is not None:
            if not isinstance(spec_k, int) or isinstance(spec_k, bool) or spec_k < 1:
                raise ValueError("spec_k must be a positive integer")
            spec_k = min(int(spec_k), int(self.backend.dspark_block))
        sampler = TopPSampler(float(temperature), float(1.0 if top_p is None else top_p), self._gen)
        eos_id = None if (ignore_eos or self.eos_id is None) else int(self.eos_id)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        self.last_stats = {}

        def work():
            try:
                self.requests += 1
                with self._busy:
                    if spec:
                        done = self._run_spec(ids, max_tokens, [s for s in (stop or []) if s],
                                              cancel, lambda piece: loop.call_soon_threadsafe(q.put_nowait, piece),
                                              spec_k, eos_id, sampler, use_cache=bool(cache_prompt))
                    else:
                        done = self._run(ids, max_tokens, sampler, eos_id, [s for s in (stop or []) if s],
                                         cancel, lambda piece: loop.call_soon_threadsafe(q.put_nowait, piece),
                                         use_cache=bool(cache_prompt))
                loop.call_soon_threadsafe(q.put_nowait, done)
            except BaseException as exc:                                # noqa: BLE001  (handed to the awaiting side)
                loop.call_soon_threadsafe(q.put_nowait, exc)

        fut = loop.run_in_executor(self._pool, work)
        done: dict[str, Any] | None = None
        try:
            while True:
                item = await q.get()
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, dict):
                    done = item
                    break
                yield item
        finally:
            if done is None and cancel is not None:
                cancel.set()            # the consumer went away: stop the worker at the next token boundary
            await asyncio.shield(fut)
        self.last_stats = done
        if stats is not None:
            stats.update(done)

    def close(self, *, shutdown_rank1: bool = True) -> None:
        """Server shutdown: tell rank 1 to exit, drain the model thread, close the channel."""
        if shutdown_rank1:
            try:
                self.channel.send({"op": "shutdown"})
            except OSError:
                pass
        self._pool.shutdown(wait=True)
        self.channel.close()


# ---- rank-0 helpers -----------------------------------------------------------------------------------------


def eos_id_of(model_dir: str | Path, tokenizer) -> int | None:
    """Id of <｜end▁of▁sentence｜>: the tokenizer's own entry wins, config.json is the fallback.

    (Adapted from Kyojin tools/ds41/ds41_model.py `_hf_eos`, MIT; handles the list form too.)"""
    try:
        tok = getattr(tokenizer, "_tk", tokenizer)
        tid = tok.token_to_id("<｜end▁of▁sentence｜>")
        if tid is not None:
            return int(tid)
    except Exception:                                                   # noqa: BLE001
        pass
    try:
        v = json.loads((Path(model_dir) / "config.json").read_text()).get("eos_token_id")
        if isinstance(v, list):
            v = v[0] if v else None
        if isinstance(v, int):
            return v
    except Exception:                                                   # noqa: BLE001
        pass
    return None
