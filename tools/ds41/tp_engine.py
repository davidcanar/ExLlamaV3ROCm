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
    multi-row verify forward (recurrent_history=True), rank 0 greedy-accepts, sends `accept`,
    and both ranks rewind the state identically (`rewind(k - a)` is a mirrored TP dispatch);
  - plain requests (`speculative: false`, sampled requests, or servers started without
    EXL3_TP_DSPARK) use the unchanged plain `step` path; a request picks its mode once, at a
    settled boundary (between requests, when both ranks are idle) -- never mid-round.
Spec is greedy-only (temperature 0); per-request `speculative` / `spec_k` (1..5) reach the engine
through two marked LOCAL additions in the vendored ds41_serve.py.

Lockstep rule: every stop decision (eos token, stop string, max_tokens, cancel) happens on rank 0
AFTER sampling a token and BEFORE sending its `step` frame. Once a `step` has been sent, the next
forward is committed on both ranks, so the two ranks always issue the same number of collectives.

Control protocol (framed JSON over TCP: 4-byte little-endian length prefix, then UTF-8 bytes):
  rank 0 -> rank 1: {"op": "hello", "proto": 2, "vocab_size": V, "dspark": B}   handshake after both loads
                    {"op": "begin", "req": N, "ids": [int, ...], "spec": B}     new request (fresh state)
                    {"op": "step", "req": N, "token": T}                        plain decode: feed T next
                    {"op": "round", "req": N, "k": K, "window": [int, ...]}     spec: verify [anchor, drafts]
                    {"op": "accept", "req": N, "accepted": A}                   spec: settle, rewind(k - A)
                    {"op": "end", "req": N}                                    request over, release state
                    {"op": "ping"}                                             liveness probe (idle only)
                    {"op": "shutdown"}                                         clean exit
  rank 1 -> rank 0: {"op": "hello_ok", "proto": 2, "rank": 1, "vocab_size": V, "dspark": B}
                    {"op": "pong"}
                    {"op": "error", "message": "..."}                          rank 1 failed; restart both

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
- one in-flight request; no prefix/prompt cache reuse between requests (each request re-prefills)
- serving context defaults to ~750 tokens (EXL3_TP_SAFE_CTX, tp_serve.py); the deployment
  (exl3ds4.sh) raises it to the full 512K cache, and prompts/chunks of any size up to
  cache_tokens - 2 are accepted (validated at ~8K-token prompts with DSpark)
- no rank-failure recovery: if either rank dies mid-request, both must be restarted
- DSpark: greedy requests only (sampled requests silently use the plain path); `spec_k` caps the
  draft window at the DSpark block size (5); acceptance is the script's greedy rule (bonus token
  always emitted, EOS/stop/max cuts land on emitted-token boundaries even mid-round)
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
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

PROTO = 2
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
                 export_layers=frozenset(), dspark_block: int = 5, page_size: int = 256):
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
            draft_model = Model.from_config(config, component="mtp")
            draft_cache = Cache(draft_model, max_num_tokens=cache_tokens, max_batch_size=1)
            draft_model.load(progressbar=False)
            draft_model.attach_to(model)     # private embedding (CPU) + private head (GPU)
            assert draft_model.own_head is not None, "drafter did not get its private head (attach order bug)"
            draft_bt = torch.arange(cache_tokens // PAGE_SIZE, dtype=torch.int32).view(1, -1)
            log(f"tp: rank 0 DSpark drafter loaded in {time.time() - started:.1f} s "
                f"(block {block}, taps at layers {sorted(export_layers)}, max_history {MAX_HISTORY})")
            check_memory_floor("after the DSpark drafter load", rank=rank, floor_gib=floor_gib, log=log)

        tokenizer = Tokenizer.from_config(config)
        return cls(model, cache, tokenizer, span=span, cache_tokens=cache_tokens, rank=rank,
                   dspark=dspark, draft_model=draft_model, draft_cache=draft_cache, draft_bt=draft_bt,
                   export_layers=export_layers, dspark_block=block, page_size=PAGE_SIZE)

    def new_state(self):
        import torch
        with torch.inference_mode():
            return self.cache.get_new_state()

    def release_state(self, state) -> None:
        self.cache.release_state(state)

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

    def prefill(self, ids: list[int], state, spec: bool = False):
        """Cached prefill of all but the last prompt token (the decode loop feeds that one),
        in `span`-row chunks.

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
        last_params: dict | None = None
        for a in range(0, n, self.span):
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
        self.prompt_len = 0        # ids of the active request (memory-trim gating at end)

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
                    self._end()
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
        self.spec = spec
        self.request_id = msg.get("req")
        self.prompt_len = len(ids)
        self.state = self.backend.new_state()
        self.backend.prefill(ids, self.state, spec=spec)
        self.position = len(ids) - 1
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

    def _end(self) -> None:
        if self.state is None:
            raise ProtocolError("rank 1: end without begin")
        if self.round_k is not None:
            raise ProtocolError("rank 1: end with a round still awaiting its accept")
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


# ---- rank 0: the engine the vendored server talks to -------------------------------------------------------


class TPEngine:
    """`ds41_engine.Engine` over the two-node model: rank 0 renders, prefills, samples and streams.

    `generate` yields the decoded text of the reply with special tokens included (the vendored
    parser splits `</think>` and the DSML tool markup out of that text), exactly like the engine
    contract demands. Stop conditions all land on token boundaries; the stats keys match the
    vendored server (`cached_tokens` is always 0: the first version re-prefills every request).

    When the backend loaded DSpark (`EXL3_TP_DSPARK=1`), greedy requests (temperature 0) default
    to speculative rounds (per-request `speculative: false` opts out, `spec_k` caps the draft
    window); sampled requests always use the plain path. `speculative: true` on a server without
    DSpark is refused before the job starts.
    """

    sessions = 1

    def __init__(self, backend, channel, tokenizer, *, ctx: int, eos_id: int | None = None,
                 seed: int | None = None, log: Callable[[str], None] = print):
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

    # -- one request (the model thread; the op sequence Rank1Worker mirrors)
    def _run(self, ids: list[int], max_tokens: int, sampler: TopPSampler, eos_id: int | None,
             stops: list[str], cancel: asyncio.Event | None, emit: Callable[[str], None]) -> dict[str, Any]:
        self.channel.send({"op": "begin", "req": self.requests, "ids": [int(t) for t in ids]})
        state = None
        n_new = 0
        eos = False
        reason = "max_new_tokens"
        t0 = time.perf_counter()
        t_first: float | None = None
        detok = _Detok(getattr(self.tk, "_tk", self.tk))
        text = ""
        try:
            state = self.backend.new_state()
            self.backend.prefill(ids, state)
            cur, pos = int(ids[-1]), len(ids) - 1
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
            tail = detok.flush()
            if tail:
                text += tail
                emit(tail)
        finally:
            if state is not None:
                try:
                    self.backend.release_state(state)
                except Exception as exc:                                # noqa: BLE001
                    self.log(f"tp: releasing request state failed: {type(exc).__name__}: {exc}")
            if len(ids) > getattr(self.backend, "span", 0) and hasattr(self.backend, "trim_memory"):
                self.backend.trim_memory()      # a chunked prompt grew the reserved pool; return it
            try:
                self.channel.send({"op": "end", "req": self.requests})
            except OSError as exc:                                      # noqa: BLE001  (the request is failing anyway)
                self.log(f"tp: sending end to rank 1 failed: {type(exc).__name__}: {exc}")
        t1 = time.perf_counter()
        t_first = t_first if t_first is not None else t1
        return {"cached_tokens": 0, "new_tokens": n_new, "prompt_tokens": len(ids),
                "time_prefill": t_first - t0, "time_generate": t1 - t_first,
                "eos": eos, "eos_reason": reason, "spec_used": False}

    # -- one spec request (the model thread; mirrors the round/accept sequence of Rank1Worker)
    def _run_spec(self, ids: list[int], max_tokens: int, stops: list[str], cancel: asyncio.Event | None,
                  emit: Callable[[str], None], spec_k: int | None, eos_id: int | None) -> dict[str, Any]:
        import torch                                                    # the whole request runs under inference_mode (as the script)
        self.channel.send({"op": "begin", "req": self.requests, "ids": [int(t) for t in ids], "spec": True})
        state = None
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
                state = self.backend.new_state()
                pparams = self.backend.prefill(ids, state, spec=True)
                taps = (pparams or {}).get("export_states")
                if taps:
                    # Stand-ins (and any backend that does not prime its own drafter): prime
                    # once from the returned taps, as before. The real backend validates each
                    # chunk's export and primes per chunk inside prefill().
                    self.backend.draft_prefill_kv(taps)
                t_first = time.perf_counter()
                anchor, position = int(ids[-1]), len(ids) - 1
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
                    greedy = [int(x) for x in logits.argmax(dim=-1)]
                    a = 0
                    while a < k and int(drafts[a]) == greedy[a]:
                        a += 1
                    bonus = greedy[a]
                    self.channel.send({"op": "accept", "req": self.requests, "accepted": int(a)})
                    self.backend.rewind(state, k - a)
                    position += a + 1
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
                tail = detok.flush()
                if tail:
                    text += tail
                    emit(tail)
        finally:
            if state is not None:
                try:
                    self.backend.release_state(state)
                except Exception as exc:                                # noqa: BLE001
                    self.log(f"tp: releasing request state failed: {type(exc).__name__}: {exc}")
            if len(ids) > getattr(self.backend, "span", 0) and hasattr(self.backend, "trim_memory"):
                self.backend.trim_memory()      # a chunked prompt grew the reserved pool; return it
            try:
                self.channel.send({"op": "end", "req": self.requests})
            except OSError as exc:                                      # noqa: BLE001  (the request is failing anyway)
                self.log(f"tp: sending end to rank 1 failed: {type(exc).__name__}: {exc}")
        t1 = time.perf_counter()
        t_first = t_first if t_first is not None else t1
        return {"cached_tokens": 0, "new_tokens": n_new, "prompt_tokens": len(ids),
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
        # Spec mode is chosen here, once per request, at the settled start boundary. Greedy only:
        # a sampled request (or an explicit opt-out) runs the unchanged plain path whatever the
        # server default is.
        spec = self.dspark if speculative is None else bool(speculative)
        if spec and not self.dspark:
            raise ValueError("speculative decoding is not enabled on this server (start both ranks with EXL3_TP_DSPARK=1)")
        if spec and float(temperature) > 0.0:
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
                                              spec_k, eos_id)
                    else:
                        done = self._run(ids, max_tokens, sampler, eos_id, [s for s in (stop or []) if s],
                                         cancel, lambda piece: loop.call_soon_threadsafe(q.put_nowait, piece))
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
