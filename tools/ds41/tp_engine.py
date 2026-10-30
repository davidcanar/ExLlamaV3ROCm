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

Vision (2026-10-29, `tp_vision.py`; rank 0 loads the tower + aligner, `component = "vision"`,
EXL3_TP_VISION=0 disables, EXL3_TP_VISION_STRICT=1 makes a load/floor failure fatal instead of
degrading to text-only):
- image requests carry ids >= 1000000000 (the indexed-embeddings dynamic base, PR #449): the
  trunk is natively multimodal through `params["indexed_embeddings"]` (MoE bias_vl routing,
  Engram n-gram block, PLE mapping all key off those ids), so rank 1 only needs the tower's
  OUTPUT ROWS, not the tower. Rank 0 computes them once per image, ships them as one imgmeta
  frame + raw binary frames (sha256-validated), and both ranks attach the same list to every
  prefill chunk; decode steps never see image ids (a prompt must end on a text token)
- the vendored chat layer renders one `<｜deepseek_image｜>` per image; `prepare_images`
  (CPU, off the event loop) decodes safely (data:/base64/http(s), byte + pixel caps) and plans
  the exact token span from the tower's own resize math; `prompt_ids(prompt, images)` splices
  each occurrence into its unique dynamic-id range, so lengths account for the expansion
- image requests bypass the prefix cache entirely (no restore, no save): the dynamic ids are
  unique per allocation, so cross-request identity would need pixel hashing -- deliberately
  not attempted in this first version. Text requests keep the cache, including against the
  text prefix an image request left in the cache (identical ids, still valid)
- DSpark composes with vision (the drafter never sees image ids: it drafts from the anchor
  and verifies text rows) -- and a first vision run that does not fit VRAM beside the drafter
  is practical per launch with `EXL3_TP_DSPARK=0 ./exl3ds4.sh restart` (both ranks read it;
  no file edits, budgets fall back to the no-drafter defaults)

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

Control protocol (framed JSON over TCP: 4-byte little-endian length prefix, then UTF-8 bytes;
binary payloads -- vision rows -- use the same length prefix with a 0x01 tag byte ahead of the
raw bytes, which no JSON frame can start with):
  rank 0 -> rank 1: {"op": "hello", "proto": 4, "vocab_size": V, "dspark": B, "vision": B}   handshake
                    {"op": "begin", "req": N, "ids": [int, ...], "spec": B}     new request (fresh state)
                      ids may hold image rows as dynamic ids >= 1000000000 when an
                      imgmeta frame with matching ranges preceded the begin
                      + "restore": K, "start": S   prefix cache: restore checkpoint K (position S)
                                                   and prefill only ids[S:]
                      + "drop": [K, ...]           drop these checkpoints (after the restore)
                      + "save": K                  checkpoint the state after the prefill as K
                      + "drop_all": true           first drop every checkpoint (cache_salt reset)
                    {"op": "imgmeta", "n": K, "images": [spec...]}   vision: the tower's rows follow
                      as K binary frames (alias, id range, rows/dim/dtype, sha256 per image);
                      rank 1 rebuilds its indexed_embeddings and answers imgmeta_ok BEFORE
                      rank 0 sends begin (a corrupt shipment never reaches a model call)
                    {"op": "step", "req": N, "token": T}                        plain decode: feed T next
                    {"op": "round", "req": N, "k": K, "window": [int, ...]}     spec: verify [anchor, drafts]
                    {"op": "accept", "req": N, "accepted": A}                   spec: settle, rewind(k - A)
                    {"op": "end", "req": N}                                    request over, release state
                      + "drop": [K, ...], "save": K   (drop, then checkpoint the final state, then release;
                        "rewind": R first rolls a spec request back to the end of its emitted reply;
                        "drop_all": true clears every checkpoint after a rank-0 bookkeeping failure)
                    {"op": "ping"}                                             liveness probe (idle only)
                    {"op": "shutdown"}                                         clean exit
  rank 1 -> rank 0: {"op": "hello_ok", "proto": 4, "rank": 1, "vocab_size": V, "dspark": B,
                     "vision_capable": B}   (proto-4 workers speak imgmeta + binary frames)
                    {"op": "imgmeta_ok", "req": N, "images": K, "rows": R}      vision rows landed intact
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

Multiple cached sessions (Stage 1, 2026-10-30, `tp_sessions.py`; EXL3_TP_SESSIONS, default 1):
- EXL3_TP_SESSIONS=1 keeps the engine, the control protocol (proto 4) and every frame byte-
  compatible with the deployment above. S >= 2 (opt-in) switches rank 0 to a SessionManager and
  the control protocol to proto 5: S conversation slots share the ONE 512K pool through dynamic,
  noncontiguous page allocation (per-session block_table rows; the library's flash_attn
  block_table mode + cache_seqlens, shipped to the TP workers by prepare_inputs_for_tp -- no
  library change). A session grows while the pool fits; a request that needs a slot or pages
  evicts the whole least-recently-used inactive session (checkpoints, stashes, pages) with
  counters in /tp/status. GPU work stays serialized (one request at a time; the vendored server
  queues the rest), and allocation always covers prompt + max_tokens + slack before any dispatch.
- returning sessions are recognized WITHOUT a session id (OpenCode sends none): the request
  matches the session whose entire committed sequence is a prefix of the new prompt, modulo the
  generation-header 1-token rewrite (see tp_sessions). A divergent conversation sharing only the
  system prompt matches nothing and gets its own slot. An explicit per-request "session" id
  overrides the search; image and warm-up requests are ephemeral slots that free themselves.
- rank 0 owns every allocation decision and sends the session's full page list in each begin
  frame; rank 1 validates it against its own records (growth-only, no page shared between
  sessions, in range) and mirrors the same block table, so the two ranks can never quietly
  disagree about who owns which rows. Restores are session-scoped: rank 1 refuses a restore
  whose checkpoint belongs to another session. hello/hello_ok validate proto, sessions and
  pool geometry with clear mismatch errors (a stale rank 1 refuses rather than desyncing).

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

import tp_vision

PROTO = 4                    # 4: vision frames (imgmeta + binary blobs, hello.vision) and the
                             #    dynamic-id prompt rows they license; a proto-3 rank 1 would
                             #    misread an imgmeta frame as JSON and desync
PROTO_SESSIONS = 5           # 5: per-session page allocation (hello.sessions/pool_pages; begin
                             #    carries session/pages/evict). Only sent with EXL3_TP_SESSIONS
                             #    >= 2; sessions=1 traffic stays proto 4, byte-compatible
SESSIONS_DEFAULT = 1         # EXL3_TP_SESSIONS: 1 (default) = single-session engine above
MAX_FRAME = 1 << 26          # 64 MiB frame guard (JSON and binary frames alike)
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


class SessionAlloc:
    """One request's page mapping for multi-session mode (EXL3_TP_SESSIONS >= 2): the
    session's ordered physical pages (logical page i -> pages[i]) both ranks build their
    block-table row from. Rank 0 derives it from the SessionManager's decisions, rank 1
    from the begin frame's "pages" list; identical lists mean identical noncontiguous
    mappings. Torch tensors are built lazily so the CPU tests can drive the same class
    with plain lists, and the block-table tensor is marked _static_dev_cache so the
    library keeps persistent per-device copies instead of re-uploading it per call."""

    __slots__ = ("pages", "page_size", "sid", "_bt")

    def __init__(self, pages, page_size: int, sid: int = 0):
        self.pages = [int(p) for p in pages]
        self.page_size = int(page_size)
        self.sid = int(sid)           # the session this mapping belongs to (logging, stand-ins)
        self._bt = None

    @property
    def tokens(self) -> int:
        return len(self.pages) * self.page_size

    def bt(self):
        """The (1, pages) int32 block-table row for this session (cached; static per device)."""
        import torch
        if self._bt is None:
            t = torch.tensor(self.pages, dtype=torch.int32).view(1, len(self.pages))
            t._static_dev_cache = True
            self._bt = t
        return self._bt

    def seqlens(self, position: int):
        """A fresh (1,) int32 cache_seqlens row at `position` (changes every step)."""
        import torch
        return torch.tensor([int(position)], dtype=torch.int32)

    def __repr__(self):                                                     # pragma: no cover (debug)
        return f"SessionAlloc(pages={self.pages[:8]}{'...' if len(self.pages) > 8 else ''}, page_size={self.page_size})"


class ProtocolError(RuntimeError):
    """The peer sent a frame this side does not understand."""


_BLOB_TAG = b"\x01"          # binary-frame tag: no JSON frame can start with this byte


def _mm_embedding_mod():
    """exllamav3.tokenizer.mm_embedding, imported lazily (CPU tests run without a model)."""
    global FIRST_MM_EMBEDDING_INDEX
    from exllamav3.tokenizer import mm_embedding
    if FIRST_MM_EMBEDDING_INDEX != mm_embedding.FIRST_MM_EMBEDDING_INDEX:
        raise RuntimeError(
            f"exllamav3 dynamic-id base {mm_embedding.FIRST_MM_EMBEDDING_INDEX} differs from this "
            f"engine's {FIRST_MM_EMBEDDING_INDEX}; the shipped prompt rows would address the wrong ids")
    return mm_embedding


FIRST_MM_EMBEDDING_INDEX = tp_vision.FIRST_MM_EMBEDDING_INDEX


def make_mm_embedding(first_index: int, rows, *, alias: str = "", metadata: dict | None = None):
    """An MMEmbedding bound to an explicit dynamic-id range (the engine assigns ranges at
    prepare time so prompt_ids and generate agree without rerunning the allocator).

    Built through the same dict form recv_embeddings uses on TP workers, so rank 0 (fresh
    tower rows) and rank 1 (rows off the wire) produce identical objects."""
    mod = _mm_embedding_mod()
    n = int(rows.shape[0])
    return mod.MMEmbedding(imp={
        "metadata": dict(metadata or {}),
        "full_length": n, "mm_length": n,
        "first_index": int(first_index), "last_index": int(first_index) + n,
        "text_alias": alias or None, "grid_thw": None, "mrope_merge_size": None,
        "embeddings": rows, "deepstack_embeddings": None,
        "align": 0, "align_phase": 0, "align_lead": 0,
    })


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

    def _send_frame(self, payload: bytes) -> None:
        if len(payload) > MAX_FRAME:
            raise ValueError(f"control frame too large: {len(payload)} bytes")
        self.conn.sendall(struct.pack("<I", len(payload)) + payload)

    def send(self, msg: dict) -> None:
        payload = json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        with self.lock:
            self._send_frame(payload)

    def send_blob(self, data: bytes) -> None:
        """One binary frame (vision rows). Same length-prefixed envelope as JSON frames; a
        leading 0x01 tag byte distinguishes the two, and no JSON payload can begin with it
        (JSON frames always start with '{')."""
        with self.lock:
            self._send_frame(_BLOB_TAG + data)

    def send_with_blobs(self, msg: dict, blobs: list[bytes]) -> None:
        """One JSON frame followed by its binary frames, atomically under the channel lock:
        nothing (not even a ping) can interleave, and the reader knows to take exactly
        len(blobs) binary frames after the JSON frame that announced them."""
        payload = json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        with self.lock:
            self._send_frame(payload)
            for blob in blobs:
                self._send_frame(_BLOB_TAG + blob)

    def recv(self) -> dict:
        (length,) = struct.unpack("<I", self._read_exact(4))
        if not 0 < length <= MAX_FRAME:
            raise ChannelClosed(f"bad control frame length {length}")
        payload = self._read_exact(length)
        if payload[:1] != b"{":
            if payload[:1] == _BLOB_TAG:
                raise ProtocolError("expected a JSON control frame but a binary frame arrived (desync)")
            raise ChannelClosed("control frame is not JSON")
        msg = json.loads(payload.decode("utf-8"))
        if not isinstance(msg, dict):
            raise ChannelClosed("control frame is not a JSON object")
        return msg

    def recv_blob(self) -> bytes:
        """The raw bytes of one binary frame (rank 1, after an imgmeta JSON frame)."""
        (length,) = struct.unpack("<I", self._read_exact(4))
        if not 0 < length <= MAX_FRAME:
            raise ChannelClosed(f"bad control frame length {length}")
        payload = self._read_exact(length)
        if payload[:1] != _BLOB_TAG:
            raise ProtocolError("expected a binary frame after imgmeta; a JSON frame arrived instead")
        return payload[1:]

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

    def request_with_blobs(self, msg: dict, blobs: list[bytes], timeout: float = 120.0) -> dict:
        """Send one JSON frame plus its binary frames and wait for one JSON reply (the
        imgmeta -> imgmeta_ok handshake; a shipment rank 1 could not land intact never
        reaches a model call)."""
        with self.lock:
            old = self.conn.gettimeout()
            self.conn.settimeout(timeout)
            try:
                self.send_with_blobs(msg, blobs)
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
                 draft_window_rows: int = 128, vision=None, vision_cfg=None):
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
        self.vision = vision                      # the component="vision" tower+aligner Model,
        self.vision_cfg = vision_cfg              # rank 0 only; None everywhere else. vision_cfg is
        # the config.vision namespace (patch_size, downsample_ratio, max_n_token, min_pixels,
        # max_wh_ratio) the CPU-side geometry plan of tp_vision needs -- present whenever the
        # config describes a tower, even when the tower itself is not loaded.

    @classmethod
    def load(cls, *, rank: int, world_size: int = 2, model_dir: str | None = None,
             engram_dir: str | None = None, cache_tokens: int | None = None, span: int | None = None,
             mem_per_device=None, budgets=None, dspark: bool | None = None, progress: bool | None = None,
             vision: bool | None = None, log: Callable[[str], None] = print) -> "TPModelBackend":
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

        # ---- rank 0 only: the vision tower + aligner, beside the TP trunk ---------------------
        # (a separate non-TP Model: its forward issues no collectives, so rank 1's dispatch
        # sequence is unaffected; rank 1 never loads it -- it only receives the tower's output
        # rows per request, over the control channel, as indexed_embeddings)
        vision_model = None
        want_vision = env_truthy("EXL3_TP_VISION", True) if vision is None else bool(vision)
        tower_present = getattr(config, "vision", None) is not None and "vision" in config.model_classes
        if want_vision and rank == 0 and tower_present:
            strict = env_truthy("EXL3_TP_VISION_STRICT", False)
            try:
                started = time.time()
                vision_model = Model.from_config(config, component="vision")
                vision_model.load(progressbar=False)
                check_memory_floor("after the vision tower load", rank=rank, floor_gib=floor_gib, log=log)
                log(f"tp: rank {rank} vision tower + aligner loaded in {time.time() - started:.1f} s "
                    f"(image input on; a request pays one tower pass per image)")
            except Exception as exc:                                        # noqa: BLE001  (degrade, or rethrow)
                if vision_model is not None:
                    try:
                        vision_model.unload()
                    except Exception:                                       # noqa: BLE001
                        pass
                    vision_model = None
                if strict:
                    raise
                # Degrade to text-only instead of wedging the box: the server comes up, refuses
                # image requests with a clean 400, and the log says exactly what to change.
                log(f"tp: rank {rank} WARNING: vision unavailable ({type(exc).__name__}: {exc}); "
                    f"serving text-only. Room for the tower: restart with EXL3_TP_DSPARK=0 (frees "
                    f"the drafter budget; both ranks), lower EXL3_TP_BUDGETS, or accept text-only "
                    f"with EXL3_TP_VISION=0. EXL3_TP_VISION_STRICT=1 makes this fatal.")
        elif want_vision and rank == 0 and not tower_present:
            if vision is not None or os.environ.get("EXL3_TP_VISION", ""):
                raise RuntimeError("EXL3_TP_VISION=1 but this model folder has no vision tower tensors")
            log(f"tp: rank {rank} config has no vision tower tensors; serving text-only")

        tokenizer = Tokenizer.from_config(config)
        return cls(model, cache, tokenizer, span=span, cache_tokens=cache_tokens, rank=rank,
                   dspark=dspark, draft_model=draft_model, draft_cache=draft_cache, draft_bt=draft_bt,
                   export_layers=export_layers, dspark_block=block, page_size=PAGE_SIZE,
                   draft_window_rows=int(config.sliding_window), vision=vision_model,
                   vision_cfg=getattr(config, "vision", None))

    def new_state(self):
        import torch
        with torch.inference_mode():
            return self.cache.get_new_state()

    def _check_alloc(self, alloc: "SessionAlloc", position: int, rows: int, state=None) -> None:
        """Engine-side pre-dispatch guards for block_table calls (multi-session): every
        physical page in range, the request's rows covered by the session's allocation,
        and the recurrent state exactly at the chunk's absolute offset. A violation would
        otherwise fault inside a collective and wedge the peer rank's NCCL watchdog."""
        cap = self.cache_tokens // self.page_size
        for p in alloc.pages:
            assert 0 <= int(p) < cap, f"session page {p} outside the pool (0..{cap - 1})"
        assert position + rows <= alloc.tokens, \
            f"session allocation of {alloc.tokens} rows does not cover position {position} +{rows}"
        if state is not None:
            pos = getattr(state, "position", None)
            if pos is None and isinstance(state, dict):
                pos = state.get("pos")
            if pos is not None:
                assert int(pos) == int(position), \
                    f"recurrent state at position {pos}, the call says {position} (chunk offset drift)"

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

    # ---- stage-2 disk hooks (tp_disk): per-page blobs, stash packing; rank-local ----
    #
    # The tier stores one opaque blob per LOGICAL page and a picklable payload per
    # checkpoint; the real backend serializes a page's cache-layer tensor slices (page-major
    # by construction of the paged caches; disk_fingerprint refuses otherwise) through the
    # recurrent HostPool with chunked device<->host copies, and pack/unpubed the TP stashes
    # whose tensors live in the worker processes (mp_disk_pack/unpack ride the existing
    # dispatch-with-results machinery -- no library change, no collectives, host only).

    def _tp_dev_args(self, local_parts: list) -> list:
        """Build dev_args for a cross-host TP dispatch. The library's dispatch localizes
        the device list (only this rank's own workers run) but indexes the suffix by the
        FULL active_devices position: dev_args[active_devices.index(device)]. So the list
        needs one placeholder entry per ACTIVE device, with this rank's parts at exactly
        its localized positions (the placeholders are never read). local_parts arrive in
        localized order from tp_worker_dispatch_wait_multi."""
        active = self.model.active_devices
        out = [()] * len(active)
        for part, device in zip(local_parts, self.model.tp_localize_devices(active)):
            out[active.index(device)] = (part,)
        return out

    def disk_fingerprint(self) -> dict:
        """Identity of this rank's cache layout: an image written by any other model, cache
        size or layer layout refuses to load (clean fallback, never wrong output).

        In TP mode the driver's cache.layers are unallocated shells -- the paged tensors
        live in the TP workers -- so the geometry comes from a results-returning dispatch
        (mp_disk_geometry, the same lookup the library's mp_cache_page_copy uses); the
        digest covers every worker's shard in device order. Non-TP backends read their own
        in-process layers."""
        import hashlib
        parts = []
        num_pages = None
        if getattr(self.model, "loaded_tp", False):
            specs = self.model.tp_worker_dispatch_wait_multi(
                self.model.active_devices, mp_disk_geometry, (id(self.cache),))
            for dev, spec in zip(self.model.active_devices, specs):
                for shape, dtype in spec:
                    if len(shape) == 0:
                        raise ValueError("cache tensor without the page dimension")
                    if num_pages is None:
                        num_pages = int(shape[0])
                    if int(shape[0]) != num_pages:
                        raise ValueError(f"cache tensor pages {shape[0]} != {num_pages}: "
                                         "layout is not page-major, disk tier unusable")
                    parts.append(f"{dev}|{shape}:{dtype}")
            if num_pages is None:
                raise ValueError("no allocated cache tensors in any TP worker "
                                 "(disk tier cannot run before the model loads)")
        else:
            for key in sorted(self.cache.layers, key=repr):
                tensors = self.cache.layers[key].get_tensors()
                if not tensors:
                    continue
                shapes = []
                for t in tensors:
                    if t.dim() == 0:
                        raise ValueError("cache tensor without the page dimension")
                    if num_pages is None:
                        num_pages = int(t.shape[0])
                    if int(t.shape[0]) != num_pages:
                        raise ValueError(f"cache tensor pages {t.shape[0]} != {num_pages}: "
                                         "layout is not page-major, disk tier unusable")
                    shapes.append(f"{tuple(t.shape)}:{t.dtype}")
                parts.append(f"{key}|{';'.join(shapes)}")
        if num_pages != self.cache_tokens // self.page_size:
            raise ValueError(f"cache tensors span {num_pages} pages, the pool has "
                             f"{self.cache_tokens // self.page_size}")
        return {"layout": hashlib.sha256("|".join(parts).encode()).hexdigest()[:16],
                "cache_tokens": self.cache_tokens, "page_size": self.page_size,
                "num_pages": num_pages}

    def disk_page_blob(self, page: int) -> bytes:
        """One physical page as an opaque blob of PLAIN BYTE RECORDS (torch.save per
        tensor -- no torch tensor ever crosses a dispatch pipe, so torch's
        torch_shm_manager daemon is never involved). TP mode: every worker exports its
        shard's page slices (mp_disk_page_out), the blob holds each worker's records in
        device order and the load routes them back to exactly the worker they came from.
        Non-TP: the page's slice of every in-process cache-layer tensor. Either way one
        page is the staging unit -- no whole-session transient."""
        import pickle
        page = int(page)
        if getattr(self.model, "loaded_tp", False):
            parts = self.model.tp_worker_dispatch_wait_multi(
                self.model.active_devices, mp_disk_page_out, (id(self.cache), page))
            return pickle.dumps(parts)
        out: dict[str, list] = {}
        for key in sorted(self.cache.layers, key=repr):
            out[repr(key)] = [_tensor_to_record(t[page])
                              for t in self.cache.layers[key].get_tensors()]
        return pickle.dumps(out)

    def disk_page_store(self, page: int, blob: bytes) -> None:
        """Write one page's blob back into this rank's pool (the load path). TP mode: each
        worker writes back its own records (mp_disk_page_in, dev_args routing padded to
        full active_devices order); non-TP: the in-process layers."""
        import pickle
        data = pickle.loads(blob)                                       # noqa: S301  (our own checksummed file)
        page = int(page)
        if getattr(self.model, "loaded_tp", False):
            # dev_args must be padded to FULL active_devices order (cross-host rank 1
            # indexes position 1 of a two-host list -- a one-entry list raised the
            # third GPU attempt's "IndexError: list index out of range")
            self.model.tp_worker_dispatch_wait_multi(
                self.model.active_devices, mp_disk_page_in, (id(self.cache), page),
                dev_args=self._tp_dev_args(data))
            return
        import torch
        with torch.inference_mode():
            for key in sorted(self.cache.layers, key=repr):
                tensors = self.cache.layers[key].get_tensors()
                for t, rec in zip(tensors, data[repr(key)]):
                    t[page].copy_(_record_to_tensor(rec))

    def disk_stash_pack(self, stash) -> Any:
        """A picklable payload for one checkpoint: the driver-side fields plus, in TP mode,
        the worker-held tensors fetched with a results-returning dispatch (a copy; the
        worker copies stay until the normal drop path frees them)."""
        payload = {k: v for k, v in stash.items() if k != "tp_handle"}
        if isinstance(stash, dict) and stash.get("tp_handle") is not None \
                and getattr(self.model, "loaded_tp", False):
            parts = self.model.tp_worker_dispatch_wait_multi(
                self.model.active_devices, mp_disk_pack, (id(self.cache), stash["tp_handle"]))
            payload["_worker_parts"] = parts
        return payload

    def disk_stash_unpack(self, payload) -> Any:
        """Rebuild a stashed-state dict from a payload: in TP mode the worker parts go back
        into every worker's recurrent cache under a fresh checkpoint handle."""
        try:
            from exllamav3.cache.recurrent import new_checkpoint_handle
        except ModuleNotFoundError:                          # CPU tests without the lib tree;
            new_checkpoint_handle = _local_checkpoint_handle  # same monotonic-handle idea
        out = {k: v for k, v in payload.items() if k != "_worker_parts"}
        if isinstance(payload, dict) and payload.get("_worker_parts") is not None \
                and getattr(self.model, "loaded_tp", False):
            handle = new_checkpoint_handle()
            # dev_args padded to full active_devices order (see _tp_dev_args): cross-host
            # rank 1's suffix sits at active position 1, not 0
            self.model.tp_worker_dispatch_wait_multi(
                self.model.active_devices, mp_disk_unpack, (id(self.cache), handle),
                dev_args=self._tp_dev_args(payload["_worker_parts"]))
            out["tp_handle"] = handle
        return out

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

    def prefill(self, ids: list[int], state, spec: bool = False, start: int = 0, embeddings=None,
                alloc: "SessionAlloc | None" = None):
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

        `alloc` (multi-session mode) replaces batch_shape/past_len with the session's explicit
        block-table row + cache_seqlens = the chunk's absolute offset: the same library path
        (prepare_flash_attn's block_table branch; positions derive from cache_seqlens exactly
        as they do from past_len) but addressing only the pages the session owns, wherever in
        the pool they lie. Both ranks pass identical allocs (rank 1's from the begin frame).

        `embeddings` (image requests) is attached to EVERY chunk as indexed_embeddings: the
        ids of an image span are dynamic ids >= 1e9, and whichever chunk holds them must carry
        the rows they address. The library ships the list to this rank's workers itself
        (prepare_inputs_for_tp -> send_embeddings), so both ranks attach their own identical
        list and nothing image-shaped rides the collectives. Text requests pass None and get
        byte-for-byte the params they always had.

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
            if alloc is not None:
                # multi-session: the session's own pages instead of the identity table
                params.pop("batch_shape")
                params.pop("past_len")
                params["block_table"] = alloc.bt()
                params["cache_seqlens"] = alloc.seqlens(a)
                self._check_alloc(alloc, a, b - a, state)
            if embeddings is not None:
                params["indexed_embeddings"] = list(embeddings)
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

    def verify(self, window: list[int], position: int, state, alloc: "SessionAlloc | None" = None):
        """One speculative verify forward over [anchor, drafts...] (multi-row, recurrent history
        on, taps exported) -- both ranks run it identically. The block table spans the whole
        cache (or, with `alloc`, exactly the session's pages), so verifies at any position stay
        inside the DSA pool. Returns (logits rows cropped to the real vocabulary, params) on
        the output rank, (None, params) elsewhere."""
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
        if alloc is not None:
            params.pop("batch_shape")
            params.pop("past_len")
            params["block_table"] = alloc.bt()
            params["cache_seqlens"] = alloc.seqlens(position)
            self._check_alloc(alloc, position, len(window), state)
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
        if self.vision is not None:
            self.vision.unload()
        if self.draft_model is not None:
            self.draft_model.unload()
        self.model.unload()

    def forward(self, token: int, position: int, state, embeddings=None, alloc: "SessionAlloc | None" = None):
        """One plain decode step: feed `token` with `past_len=position`. The block table spans
        the whole cache (or, with `alloc`, exactly the session's pages), so decode steps at any
        position stay inside the DSA pool. Returns the logits row cropped to the real
        vocabulary on the output rank, None on every other rank (cross-host TP returns no
        logits there by design -- reading them raised on rank 1 and wedged the collectives).

        `embeddings` (image requests) is attached only when the fed id is itself a dynamic
        image id (>= 1e9), which happens only for the first forward of a prompt that ends on
        an image span -- the engine refuses those prompts today, so in practice every decode
        step carries no image payload at all. Both ranks make the same decision from the same
        id, so the collective sequences stay identical."""
        import torch
        params = {
            "attn_mode": "flash_attn",
            "cache": self.cache,
            "batch_shape": (1, self.cache_tokens),
            "past_len": int(position),
            "recurrent_states": [state],
            "last_tokens_only": 1,
        }
        if alloc is not None:
            params.pop("batch_shape")
            params.pop("past_len")
            params["block_table"] = alloc.bt()
            params["cache_seqlens"] = alloc.seqlens(position)
            self._check_alloc(alloc, position, 1, state)
        if embeddings is not None and int(token) >= FIRST_MM_EMBEDDING_INDEX:
            params["indexed_embeddings"] = list(embeddings)
        logits = self.model.forward(torch.tensor(int(token), dtype=torch.long).view(1, 1), params)
        if self.rank != self.OUTPUT_RANK:
            return None
        return logits[0, -1, : self.vocab_size]


def _prefill(backend, ids: list[int], state, spec: bool, start: int, embeddings=None,
             alloc: "SessionAlloc | None" = None):
    """backend.prefill from `start` (the CPU stand-ins without prefix-cache support never get one).
    Image requests add `embeddings`; multi-session requests add `alloc`; the scripted test
    backends without the parameters are called exactly as before (image tests use backends
    that accept it)."""
    kw = {}
    if embeddings is not None:
        kw["embeddings"] = embeddings
    if alloc is not None:
        kw["alloc"] = alloc
    if start:
        return backend.prefill(ids, state, spec=spec, start=start, **kw)
    return backend.prefill(ids, state, spec=spec, **kw)


def _forward(backend, token: int, position: int, state, embeddings=None,
             alloc: "SessionAlloc | None" = None):
    """backend.forward with only the optional kwargs the backend actually supports (the
    proto-4 scripted stand-ins keep their three-argument signature; multi-session backends
    take alloc)."""
    kw = {}
    if embeddings is not None:
        kw["embeddings"] = embeddings
    if alloc is not None:
        kw["alloc"] = alloc
    return backend.forward(token, position, state, **kw)


def _verify(backend, window: list[int], position: int, state, alloc: "SessionAlloc | None" = None):
    """backend.verify, alloc only in multi-session mode (same pattern as _forward)."""
    if alloc is not None:
        return backend.verify(window, position, state, alloc=alloc)
    return backend.verify(window, position, state)


def _tensor_to_record(t) -> dict:
    """One tensor as plain picklable data (shape, dtype, torch.save bytes) -- NO torch
    shared-memory involvement. Every tensor that crosses a TP dispatch pipe goes through
    this: pickling tensors over multiprocessing connections routes them through torch's
    torch_shm_manager daemon, whose "no response" failure took down the rank-1 reload (the
    library's own prepare_inputs_for_tp comment warns about the same fragility). One page
    slice or stash tensor is the staging unit, so the copy is bounded."""
    import io
    import torch
    buf = io.BytesIO()
    torch.save(t.detach().cpu().contiguous(), buf)
    return {"shape": tuple(t.shape), "dtype": str(t.dtype), "data": buf.getvalue()}


def _record_to_tensor(rec):
    import io
    import torch
    try:
        t = torch.load(io.BytesIO(rec["data"]), weights_only=True)
    except TypeError:                                                   # older torch signature
        t = torch.load(io.BytesIO(rec["data"]))
    return t.view(rec["shape"]) if rec.get("shape") else t.reshape(())


def _pack_host(obj):
    """Recursively convert a stashed structure (tensors inside lists/tuples/dicts) to
    plain data for the pipe or the image file."""
    import torch
    if torch.is_tensor(obj):
        return _tensor_to_record(obj)
    if isinstance(obj, tuple):
        return {"__tuple__": [_pack_host(o) for o in obj]}
    if isinstance(obj, list):
        return [_pack_host(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _pack_host(v) for k, v in obj.items()}
    return obj


def _unpack_host(obj):
    """The inverse of _pack_host: plain data back to tensors (worker side, before the
    stashes go into the worker's recurrent cache)."""
    if isinstance(obj, dict):
        if "shape" in obj and "dtype" in obj and "data" in obj:
            return _record_to_tensor(obj)
        if set(obj) == {"__tuple__"}:
            return tuple(_unpack_host(o) for o in obj["__tuple__"])
        return {k: _unpack_host(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack_host(o) for o in obj]
    return obj


def _worker_cache_tensors(local_context: dict, cache_id: int) -> list:
    """(worker side) the ordered, allocated cache tensors of THIS worker's shard of
    `cache_id`: every kv module's layer for that cache. In TP mode the tensors exist only
    in the workers -- the driver's parent-side cache.layers are unallocated shells -- so
    geometry, export and import all run here. Mirrors the library's own
    mp_cache_page_copy / mp_cpu_cache_tensors lookup (no library changes)."""
    out = []
    for module in local_context.get("kv_modules", ()):
        layer = getattr(module, "tp_cache_lookup", {}).get(cache_id)
        if layer is None:
            continue
        out += [t for t in layer.get_tensors() if t is not None]
    return out


def mp_disk_geometry(local_context: dict, cache_id: int):
    """(worker side) this worker's page geometry for the cache: (shape, dtype) per tensor
    in the deterministic order mp_disk_page_out/in use. The driver digests every worker's
    answer into the image fingerprint."""
    return [(tuple(t.shape), str(t.dtype)) for t in _worker_cache_tensors(local_context, cache_id)]


def mp_disk_page_out(local_context: dict, cache_id: int, page: int):
    """(worker side) one page of this worker's shard as PLAIN BYTE records (torch.save per
    tensor): no torch tensor ever crosses the dispatch pipe, so torch's torch_shm_manager
    daemon is never involved (its "no response" failure took down the rank-1 reloads of
    earlier GPU attempts; the library's own prepare_inputs_for_tp comment warns about the
    same fragility). One page slice is the staging unit; no collectives."""
    return [_tensor_to_record(t[int(page)])
            for t in _worker_cache_tensors(local_context, cache_id)]


def mp_disk_page_in(local_context: dict, cache_id: int, page: int, rows):
    """(worker side) write one page's byte records back into this worker's shard (the
    path). The driver routes each worker exactly the records it exported."""
    import torch
    tensors = _worker_cache_tensors(local_context, cache_id)
    assert len(tensors) == len(rows), \
        f"page layout changed between spill and load ({len(tensors)} tensors, {len(rows)} rows)"
    with torch.inference_mode():
        for t, rec in zip(tensors, rows):
            t[int(page)].copy_(_record_to_tensor(rec))


_next_local_checkpoint_handle = 0


def _local_checkpoint_handle() -> int:
    """Monotonic checkpoint handles when the exllamav3 tree is not importable (CPU tests);
    production always uses the library's own counter (handles must not collide with the
    ones DSV4State.stash allocates driver-side)."""
    global _next_local_checkpoint_handle
    _next_local_checkpoint_handle += 1
    return _next_local_checkpoint_handle


def mp_disk_pack(local_context: dict, cache_id: int, cp_handle: int):
    """(disk tier, worker side) Copy this checkpoint's stashed tensors back to the driver
    as PLAIN DATA (tensors converted to torch.save byte records inside the stashed
    structure): host bytes over the worker pipe, no torch shared memory, no collectives,
    no shared-memory lifetime games."""
    return _pack_host(local_context["recurrent_cache"][cp_handle])


def mp_disk_unpack(local_context: dict, cache_id: int, cp_handle: int, part):
    """(disk tier, worker side) Park a driver-supplied stash payload under a fresh handle,
    exactly where mp_cache_recurrent_unstash will look for it. `part` is plain data; the
    tensors are rebuilt here, worker-side."""
    local_context["recurrent_cache"][cp_handle] = _unpack_host(part)


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

    def __init__(self, backend, channel, *, log: Callable[[str], None] = print,
                 sessions: int | None = None, disk_opts: dict | None = None):
        self.backend = backend
        self.channel = channel
        self.log = log
        self.sessions_env = sessions     # tests pin the expected session count; production
        #                                  validates rank 0's hello against EXL3_TP_SESSIONS
        self.disk_opts = disk_opts       # tests pin this rank's tier dir/cap; production env
        self.state = None
        self.position = -1
        self.request_id = None
        self.requests_served = 0
        self.spec = False          # the active request's mode (plain steps vs spec rounds)
        self.round_k = None        # k of the round frame awaiting its accept
        self.prompt_len = 0        # prefilled rows of the active request (memory-trim gating at end)
        self.stashes: dict[int, tuple[int, Any]] = {}   # proto 4: key -> (position, stash)
        self.embeddings = None     # image rows of the request being armed (imgmeta -> begin)
        # proto 5 (multi-session): per-session records {sid: {"pages": [...], "stashes": {}}},
        # plus the page -> sid claim map every begin validates against (no page may belong to
        # two sessions; a session's list only ever grows) and the active request's page mapping
        self.proto = 4
        self.sessions_max = 1
        self.pool_pages = 0
        self.page_size = 256
        self.sess: dict[int, dict] = {}
        self.claimed: dict[int, int] = {}
        self.alloc: "SessionAlloc | None" = None
        self.sid = 0               # the active request's session (proto 5)
        self.tier = None           # stage 2: this rank's NVMe tier (built at the proto-5 handshake)

    def handshake(self) -> dict:
        hello = self.channel.recv()
        if hello.get("op") != "hello":
            raise ProtocolError(f"rank 1: expected hello, got {hello.get('op')!r}")
        proto = int(hello.get("proto", -1))
        if proto not in (PROTO, PROTO_SESSIONS):
            if proto > PROTO_SESSIONS:
                raise ProtocolError(
                    f"rank 1: rank 0 speaks control protocol {proto}, this worker speaks at most "
                    f"{PROTO_SESSIONS}; sync tools/ds41 on both boxes and restart both ranks")
            raise ProtocolError(
                f"rank 1: control protocol mismatch (rank 0 sent {proto!r}, this worker speaks "
                f"{PROTO}/{PROTO_SESSIONS})")
        ok = {"op": "hello_ok", "proto": proto, "rank": 1,
              "vocab_size": int(self.backend.vocab_size),
              "dspark": bool(getattr(self.backend, "dspark", False)),
              "vision_capable": True}   # proto 4: imgmeta + binary frames understood
        if proto >= PROTO_SESSIONS:
            want_sessions = int(hello.get("sessions", 0) or 0)
            self.page_size = int(getattr(self.backend, "page_size", 256) or 256)
            self.pool_pages = int(getattr(self.backend, "cache_tokens", 0) or 0) // self.page_size
            here_sessions = self.sessions_env if self.sessions_env is not None \
                else env_int("EXL3_TP_SESSIONS", 1)
            if want_sessions < 2:
                raise ProtocolError(f"rank 1: proto {proto} hello without sessions >= 2")
            if want_sessions != here_sessions:
                raise ProtocolError(
                    f"rank 1: rank 0 serves {want_sessions} sessions but this rank started with "
                    f"EXL3_TP_SESSIONS={here_sessions}; forward EXL3_TP_SESSIONS to this box "
                    f"and restart both ranks")
            got_pages = int(hello.get("pool_pages", 0) or 0)
            if not self.pool_pages or got_pages != self.pool_pages:
                raise ProtocolError(
                    f"rank 1: pool mismatch (rank 0 says {got_pages} pages, this rank's cache "
                    f"holds {self.pool_pages}); EXL3_TP_CACHE_TOKENS must match on both ranks")
            self.proto = proto
            self.sessions_max = want_sessions
            ok.update({"sessions": want_sessions, "pool_pages": self.pool_pages,
                       "page_size": self.page_size})
            # stage 2: this rank's own NVMe tier (per-rank local images; disabled when the
            # backend lacks the hooks -- spills then come back as spill_error and rank 0
            # falls back to eviction, never wrong output)
            self.tier = None
            if env_truthy("EXL3_TP_DISK_CACHE", True) and \
                    all(hasattr(self.backend, h) for h in ("disk_fingerprint", "disk_page_blob",
                                                           "disk_page_store", "disk_stash_pack",
                                                           "disk_stash_unpack")):
                try:
                    import tp_disk
                    self.tier = tp_disk.tier_from_env(
                        rank=1, fingerprint=self.backend.disk_fingerprint(),
                        log=lambda line: self.log(f"tp: rank 1 {line}"), default_on=True,
                        **(self.disk_opts or {}))
                except Exception as exc:                                # noqa: BLE001  (degrade)
                    self.log(f"tp: rank 1 disk tier unavailable ({type(exc).__name__}: {exc})")
        self.channel.send(ok)
        self.log(f"tp: rank 1 handshake complete (proto {proto}"
                 + (f", {self.sessions_max} sessions over {self.pool_pages} pages" if proto >= PROTO_SESSIONS else "")
                 + ")")
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
                elif op == "imgmeta":
                    self._imgmeta(msg)
                elif op == "step":
                    self._step(msg)
                elif op == "round":
                    self._round(msg)
                elif op == "accept":
                    self._accept(msg)
                elif op == "end":
                    self._end(msg)
                elif op == "spill":
                    self._spill(msg)
                elif op == "load":
                    self._load(msg)
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

    def _imgmeta(self, msg: dict) -> None:
        """Rank 0's tower output for the request being armed: one JSON frame plus n binary
        frames. Validated (blob sizes, sha256, dynamic-id ranges) and rebuilt as this rank's
        indexed_embeddings, then acknowledged BEFORE rank 0 sends begin -- a shipment that did
        not land byte-exact never reaches a model call, where a bad row would fault inside a
        collective and wedge rank 0's NCCL peer instead of erroring."""
        if self.state is not None:
            raise ProtocolError("rank 1: imgmeta while a request is still active")
        specs = msg.get("images") or []
        blobs = [self.channel.recv_blob() for _ in range(int(msg.get("n", len(specs))))]
        try:
            described = tp_vision.decode_imgmeta(msg, blobs, FIRST_MM_EMBEDDING_INDEX)
        except ValueError as exc:
            raise ProtocolError(f"rank 1: bad imgmeta shipment: {exc}") from exc
        self.embeddings = [make_mm_embedding(d["first_index"], d["tensor"], alias=d["alias"],
                                             metadata={"source": "rank0-wire"})
                           for d in described]
        self.channel.send({"op": "imgmeta_ok", "req": msg.get("req"), "images": len(self.embeddings),
                           "rows": sum(int(d["rows"]) for d in described)})

    # ---- proto 5 (multi-session) helpers ----------------------------------------------------

    def _sess_record(self, sid: int, *, create: bool = False) -> dict | None:
        rec = self.sess.get(int(sid))
        if rec is None and create:
            rec = self.sess[int(sid)] = {"pages": [], "stashes": {}}
        return rec

    def _stash_ns(self) -> dict[int, tuple[int, Any]]:
        """The active request's stash namespace: proto 4 keeps one flat dict (the tests read
        it), proto 5 keeps one per session so a session's checkpoints die with it."""
        if self.proto < PROTO_SESSIONS:
            return self.stashes
        rec = self.sess.get(self.sid)
        if rec is None:
            raise ProtocolError(f"rank 1: session {self.sid} has no record (begin without pages?)")
        return rec["stashes"]

    def _drop_sessions(self, sids) -> None:
        """Free whole sessions (rank 0 evicted them for slots/pages): their stashes must
        already be gone (the same frame's drop list carried every key), then the records
        and their page claims go. No model calls -- rank 0 freed its side the same way."""
        for sid in sids or ():
            sid = int(sid)
            rec = self.sess.get(sid)
            if rec is None:
                raise ProtocolError(f"rank 1: evict of unknown session {sid}")
            if rec["stashes"]:
                raise ProtocolError(
                    f"rank 1: evict of session {sid} with checkpoints {sorted(rec['stashes'])} "
                    "still live (rank 0 must drop them in the same frame)")
            for p in rec["pages"]:
                if self.claimed.get(p) != sid:
                    raise ProtocolError(f"rank 1: page {p} claim mismatch while evicting session {sid}")
                del self.claimed[p]
            del self.sess[sid]

    def _drop_all_sessions(self) -> None:
        """drop_all (cache_salt reset / rank-0 bookkeeping failure): forget every stash,
        session record, page claim and disk image on this rank."""
        for rec in self.sess.values():
            for _, (_, stash) in list(rec["stashes"].items()):
                try:
                    self.backend.drop_stash(stash)
                except Exception:                                       # noqa: BLE001  (best effort)
                    pass
            rec["stashes"].clear()
            rec["pages"].clear()
        self.sess.clear()
        self.claimed.clear()
        if self.tier is not None:                  # cold sessions die with the reset too
            for sid in list(self.tier._sizes):
                self.tier.drop(sid)

    def _apply_pages(self, sid: int, pages: list[int]) -> None:
        """Validate the begin frame's page list against this rank's records and adopt it:
        in range, no duplicates, never another session's page, and growth-only for a known
        session (the allocator never shruffles; rank 0's list replaces ours verbatim)."""
        if len(set(pages)) != len(pages):
            raise ProtocolError(f"rank 1: session {sid} pages hold duplicates")
        for p in pages:
            if not 0 <= int(p) < self.pool_pages:
                raise ProtocolError(f"rank 1: page {p} outside the pool (0..{self.pool_pages - 1})")
            owner = self.claimed.get(int(p))
            if owner is not None and owner != sid:
                raise ProtocolError(
                    f"rank 1: page {p} of session {sid} already belongs to session {owner}; "
                    "sessions must never share pages (rank 0 allocator drift?)")
        rec = self._sess_record(sid, create=True)
        if rec["pages"] and pages[: len(rec["pages"])] != rec["pages"]:
            raise ProtocolError(
                f"rank 1: session {sid} pages changed history (had {rec['pages']}, got a list "
                "that does not extend it); restart both ranks (rank 0 allocator drift?)")
        for p in pages[len(rec["pages"]):]:
            self.claimed[int(p)] = sid
        rec["pages"] = list(pages)

    # ---- proto 5 stage 2: the disk tier (rank 0 drives; both ranks write their own image) ----

    def _spill(self, msg: dict) -> None:
        """Rank 0 parked a session on NVMe: stream this rank's image (one page blob at a
        time, cap enforced while writing), then drop the stashes, free the page claims and
        forget the record. A failure keeps the state untouched and answers spill_error --
        rank 0 falls back to eviction."""
        import tp_disk
        if self.state is not None:
            raise ProtocolError("rank 1: spill while a request is still active")
        if self.proto < PROTO_SESSIONS:
            raise ProtocolError("rank 1: spill frame in proto 4")
        sid = int(msg["session"])
        img = int(msg["image"])
        keys = [int(k) for k in msg.get("keys") or ()]
        rec = self.sess.get(sid)
        if rec is None:
            raise ProtocolError(f"rank 1: spill of unknown session {sid}")
        if self.tier is None or not self.tier.enabled:
            self.channel.send({"op": "spill_error", "session": sid, "req": msg.get("req"),
                               "reason": "rank 1 disk tier off (EXL3_TP_DISK_CACHE)"})
            return
        if sorted(keys) != sorted(rec["stashes"]):
            raise ProtocolError(f"rank 1: spill of session {sid} lists checkpoints {keys} but "
                                f"this rank holds {sorted(rec['stashes'])}")
        try:
            meta = {"positions": {str(k): rec["stashes"][k][0] for k in keys}}
            size = self.tier.write_image(
                img, meta=meta, n_pages=len(rec["pages"]),
                page_blob_fn=lambda i: self.backend.disk_page_blob(rec["pages"][i]),
                stash_payload={k: (rec["stashes"][k][0],
                                   self.backend.disk_stash_pack(rec["stashes"][k][1]))
                               for k in keys})
        except Exception as exc:                                        # noqa: BLE001  (fall back)
            self.channel.send({"op": "spill_error", "session": sid, "req": msg.get("req"),
                               "reason": f"{type(exc).__name__}: {exc}"})
            return
        for k in keys:                        # the image is safe: free this rank's copy
            pos, st = rec["stashes"].pop(k)
            self.backend.drop_stash(st)
        for p in rec["pages"]:
            del self.claimed[p]
        rec["pages"] = []
        del self.sess[sid]
        self.log(f"tp: rank 1 spilled session {sid} into image {img} "
                 f"({size} bytes, {len(keys)} checkpoints)")
        self.channel.send({"op": "spill_ok", "session": sid, "bytes": size, "req": msg.get("req")})

    def _load(self, msg: dict) -> None:
        """Rank 0 is reloading a spilled session: claim the fresh pages it picked, restore
        this rank's image into them and rebuild the record. Any failure cleans this rank's
        half up and answers load_error -- rank 0 discards its side and the request starts a
        fresh session (never wrong output)."""
        import tp_disk
        if self.state is not None:
            raise ProtocolError("rank 1: load while a request is still active")
        if self.proto < PROTO_SESSIONS:
            raise ProtocolError("rank 1: load frame in proto 4")
        sid = int(msg["session"])
        image_sid = int(msg.get("image") or sid)
        pages = [int(p) for p in msg["pages"]]
        if self.tier is None or not self.tier.enabled:
            self.channel.send({"op": "load_error", "session": sid, "req": msg.get("req"),
                               "reason": "rank 1 disk tier off (EXL3_TP_DISK_CACHE)"})
            return
        if sid in self.sess:
            self.channel.send({"op": "load_error", "session": sid, "req": msg.get("req"),
                               "reason": f"session {sid} is already hot"})
            return
        for p in pages:
            if not 0 <= p < self.pool_pages:
                self.channel.send({"op": "load_error", "session": sid, "req": msg.get("req"),
                                   "reason": f"page {p} outside the pool"})
                return
            if self.claimed.get(p) not in (None, sid):
                self.channel.send({"op": "load_error", "session": sid, "req": msg.get("req"),
                                   "reason": f"page {p} already belongs to session "
                                             f"{self.claimed.get(p)}"})
                return
        # the evictions rank 0 made for this load (same semantics as a begin frame)
        self._drop(msg.get("drop"))
        self._drop_sessions(msg.get("evict"))
        rec = self.sess[sid] = {"pages": pages, "stashes": {}}
        for p in pages:
            self.claimed[p] = sid
        try:
            def blobs_into(i, blob):
                self.backend.disk_page_store(pages[i], blob)

            def stash_into(payload):
                for k, (pos, packed) in payload.items():
                    rec["stashes"][int(k)] = (pos, self.backend.disk_stash_unpack(packed))

            self.tier.read_image(image_sid, blobs_into=blobs_into, stash_into=stash_into,
                                 expect=self.tier.fingerprint)
            want = sorted(int(k) for k in (msg.get("keys") or ()))
            if want and sorted(rec["stashes"]) != want:
                raise tp_disk.DiskCorrupt(f"image holds checkpoints {sorted(rec['stashes'])}, "
                                          f"rank 0 said {want}")
        except Exception as exc:                                        # noqa: BLE001  (fall back)
            # nothing was unpacked (page writes and payload validation precede the unpack),
            # so the cleanup drops no stashes -- exactly rank 0's state at this point
            for p in pages:
                self.claimed.pop(p, None)
            del self.sess[sid]
            self.tier.drop(image_sid)
            self.log(f"tp: rank 1 reload of session {sid} failed "
                     f"({type(exc).__name__}: {exc}); rank 0 falls back to a fresh session")
            self.channel.send({"op": "load_error", "session": sid, "req": msg.get("req"),
                               "reason": f"{type(exc).__name__}: {exc}"})
            return
        self.tier.drop(image_sid)              # one-shot images (no restart persistence)
        self.log(f"tp: rank 1 reloaded session {sid} into pages {pages[:4]}"
                 f"{'...' if len(pages) > 4 else ''} ({len(rec['stashes'])} checkpoints)")
        self.channel.send({"op": "load_ok", "session": sid, "req": msg.get("req")})

    def _begin(self, msg: dict) -> None:
        if self.state is not None:
            raise ProtocolError("rank 1: begin while a request is still active")
        ids = [int(t) for t in msg["ids"]]
        if len(ids) < 2:
            raise ValueError("rank 1: prompt must hold at least 2 token ids")
        # Image rows ride as dynamic ids >= 1e9; they are legal only when an imgmeta shipment
        # with covering ranges preceded this begin (rank 1 never sees the tower itself)
        embeds = self.embeddings or None
        mm_ids = [t for t in ids if t >= FIRST_MM_EMBEDDING_INDEX]
        if mm_ids and embeds is None:
            raise ValueError(f"rank 1: prompt holds {len(mm_ids)} image row id(s) but no imgmeta "
                             "shipment preceded the begin")
        if embeds is not None and not mm_ids:
            raise ProtocolError("rank 1: imgmeta shipped but the begin ids hold no image rows")
        if embeds is not None:
            ranges = [(e.first_index, int(e.first_index) + int(e.mm_length)) for e in embeds]
            for t in mm_ids:
                if not any(a <= t < b for a, b in ranges):
                    raise ValueError(f"rank 1: image row id {t} is outside every shipped range "
                                     f"({[(a, b) for a, b in ranges]})")
        bad = [t for t in ids if FIRST_MM_EMBEDDING_INDEX > t >= self.backend.vocab_size or t < 0]
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
        # ---- proto 5: the session routing (rank 0 owns every allocation decision).
        # Order matters for the mirror: drop_all and the drop list (drop_stash ops) FIRST,
        # then whole-session evict frees (no model ops), then the page upsert -- exactly the
        # sequence rank 0's session path runs (drop the evicted stashes, restore, prefill,
        # save). drop_all must clear every record BEFORE this request's own record lands. ----
        alloc = None
        sid = 0
        dropped = False
        if self.proto >= PROTO_SESSIONS:
            if "session" not in msg or "pages" not in msg:
                raise ProtocolError("rank 1: proto 5 begin needs session and pages")
            sid = int(msg["session"])
            if not 1 <= sid <= self.sessions_max:
                raise ProtocolError(f"rank 1: session {sid} outside 1..{self.sessions_max}")
            pages = [int(p) for p in msg["pages"]]
            if not pages:
                raise ProtocolError(f"rank 1: session {sid} begin with an empty page list")
            if len(ids) + 2 > len(pages) * self.page_size:
                raise ValueError(
                    f"rank 1: prompt of {len(ids)} tokens exceeds session {sid}'s allocation "
                    f"({len(pages) * self.page_size - 2} usable of {len(pages)} pages)")
            for sid_e in msg.get("evict") or ():
                if int(sid_e) not in self.sess:
                    raise ProtocolError(f"rank 1: evict of unknown session {sid_e}")
            if msg.get("save") is not None and int(msg["save"]) in \
                    (self.sess.get(sid) or {}).get("stashes", {}):
                raise ProtocolError(f"rank 1: checkpoint {msg['save']} already exists")
            if msg.get("drop_all"):
                self._drop_all_sessions()      # rank 0 reset everything: forget every record
            self._drop(msg.get("drop"))
            dropped = True
            self._drop_sessions(msg.get("evict"))
            for gone in msg.get("drop_images") or ():   # rank 0 spent/lost these images
                if self.tier is not None:
                    self.tier.drop(int(gone))
            self._apply_pages(sid, pages)
            alloc = SessionAlloc(pages, self.page_size, sid)
        start = int(msg.get("start") or 0)
        restore = msg.get("restore")
        entry = None
        if restore is not None:
            if self.proto >= PROTO_SESSIONS:
                rec = self.sess.get(sid)               # _apply_pages created/updated it
                entry = (rec or {}).get("stashes", {}).get(int(restore))
                if entry is None:
                    raise ProtocolError(
                        f"rank 1: restore of checkpoint {restore} unknown to session {sid} "
                        "(checkpoints never cross sessions)")
                if int(restore) in [int(k) for k in (msg.get("drop") or ())]:
                    raise ProtocolError(f"rank 1: begin drops checkpoint {restore} it also restores")
            else:
                entry = self.stashes.get(int(restore))
                if entry is None:
                    raise ProtocolError(f"rank 1: restore of unknown checkpoint {restore}")
            if entry[0] != start or not 0 < start <= len(ids) - 1:
                raise ProtocolError(f"rank 1: checkpoint {restore} is at position {entry[0]}, "
                                    f"begin says start {start} (prompt of {len(ids)})")
        elif start:
            raise ProtocolError("rank 1: begin with a start position but no checkpoint to restore")
        if msg.get("drop_all") and self.proto < PROTO_SESSIONS:
            self._drop(list(self.stashes))             # rank 0 reset its prefix cache (cache_salt change)
        # (proto 5 handled drop_all above, before this request's own record was created)
        self.spec = spec
        self.sid = sid
        self.alloc = alloc
        self.request_id = msg.get("req")
        self.prompt_len = len(ids) - start
        # the same order as rank 0: restore (or a fresh state), drops, prefill, save
        self.state = self.backend.restore(entry[1], start) if restore is not None else self.backend.new_state()
        if not dropped:
            self._drop(msg.get("drop"))
        _prefill(self.backend, ids, self.state, spec, start, embeds, alloc)
        self.position = len(ids) - 1
        if msg.get("save") is not None:
            self._save(int(msg["save"]))
        if not spec:
            em = embeds if (embeds is not None and ids[-1] >= FIRST_MM_EMBEDDING_INDEX) else None
            _forward(self.backend, ids[-1], self.position, self.state, em, alloc)
            self.position += 1
        # spec mode: no first forward -- the round window carries the anchor at this position

    def _step(self, msg: dict) -> None:
        if self.state is None:
            raise ProtocolError("rank 1: step without begin")
        if self.spec:
            raise ProtocolError("rank 1: step during a spec request (mode is fixed per request)")
        token = int(msg["token"])
        # a dynamic image id is the only token that needs the rows (mirrors rank 0's gate);
        # every real decode step carries nothing. Proto 4 keeps the exact three-argument
        # call of the scripted test stand-ins; proto 5 adds the session's page mapping.
        em = self.embeddings if (self.embeddings and token >= FIRST_MM_EMBEDDING_INDEX) else None
        _forward(self.backend, token, self.position, self.state, em, self.alloc)
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
        _verify(self.backend, window, self.position, self.state, self.alloc)

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
        ns = self._stash_ns()
        if key in ns:
            raise ProtocolError(f"rank 1: checkpoint {key} already exists")
        ns[key] = (self.position, self.backend.stash(self.state))

    def _drop(self, keys) -> None:
        # unknown keys are fine: rank 0 also drops what a failed request may or may not have
        # saved here. Proto 5 keys are globally unique across sessions, so a drop list may
        # span sessions (whole-session evictions drop every key they owned in one list).
        if self.proto < PROTO_SESSIONS:
            for key in keys or ():
                entry = self.stashes.pop(int(key), None)
                if entry is not None:
                    self.backend.drop_stash(entry[1])
            return
        wanted = {int(k) for k in (keys or ())}
        if not wanted:
            return
        for rec in self.sess.values():
            for key in [k for k in rec["stashes"] if k in wanted]:
                entry = rec["stashes"].pop(key)
                self.backend.drop_stash(entry[1])
                wanted.discard(key)

    def _end(self, msg: dict | None = None) -> None:
        if self.state is None:
            raise ProtocolError("rank 1: end without begin")
        if self.round_k is not None:
            raise ProtocolError("rank 1: end with a round still awaiting its accept")
        msg = msg or {}
        if self.proto >= PROTO_SESSIONS:
            if msg.get("session") is not None and int(msg["session"]) != self.sid:
                raise ProtocolError(f"rank 1: end names session {msg.get('session')} but the "
                                    f"request runs in session {self.sid}")
        if msg.get("drop_all"):
            if self.proto >= PROTO_SESSIONS:
                self._drop_all_sessions()
            else:
                self._drop(list(self.stashes))
        else:
            self._drop(msg.get("drop"))
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
        self.embeddings = None     # the next image request ships its own rows
        if self.proto >= PROTO_SESSIONS:
            # whole sessions freed at the settled end of their request (ephemeral image and
            # warm-up slots; the drops above already emptied their stashes)
            self._drop_sessions(msg.get("evict"))
        trim = getattr(self.backend, "trim_memory", None)    # rank 0 trims on long requests;
        if trim is not None and self.prompt_len > getattr(self.backend, "span", 0):
            trim()                                           # rank 1 mirrors at end (local-only op)
        self.prompt_len = 0
        self.alloc = None
        self.sid = 0

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
        self.embeddings = None
        self.alloc = None
        self.sid = 0


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


class _ReqCtx:
    """The active request's multi-session context (model thread; one at a time -- the GPU is
    serialized and every routing decision runs under the engine's _busy lock)."""

    __slots__ = ("session", "alloc", "evicted", "ephemeral", "prompt_len")

    def __init__(self, session, alloc, evicted, *, ephemeral: bool = False, prompt_len: int = 0):
        self.session = session                 # tp_sessions.Session
        self.alloc = alloc                     # SessionAlloc for this request's pages
        self.evicted = evicted                 # [(sid, Checkpoints)] whole sessions this request evicted
        self.ephemeral = bool(ephemeral)
        self.prompt_len = int(prompt_len)


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
                 prefix: "PrefixCache | bool | None" = None, sessions: int | None = None,
                 session_opts: dict | None = None, disk_opts: dict | None = None):
        self.backend = backend
        self.channel = channel
        self.tk = tokenizer
        self.log = log
        self.ctx = int(ctx)
        self.vocab_size = int(backend.vocab_size)
        self.eos_id = eos_id
        self.dspark = bool(getattr(backend, "dspark", False))
        self.vision = getattr(backend, "vision", None)         # the tower lives on rank 0 only
        self.vision_cfg = getattr(backend, "vision_cfg", None)
        self._mm_alloc = tp_vision.MMIndexAllocator()
        self.image_token_id = self._resolve_image_token()
        self.requests = 0
        self.last_stats: dict[str, Any] = {}
        self._gen = _torch_generator(seed)
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="tp-model")
        self._busy = threading.Lock()     # held while a request drives rank 1; pings wait it out
        # ---- sessions: 1 (default) = the single-conversation prefix cache above, byte-
        # compatible frames and backend params; >= 2 (opt-in) = the SessionManager over the
        # shared pool (proto 5). The vendored server reads engine.sessions for its queue.
        self.sessions = max(1, int(sessions if sessions is not None else env_int("EXL3_TP_SESSIONS", 1)))
        self.mgr = None
        page_size = int(getattr(backend, "page_size", 256) or 256)
        cache_tokens = int(getattr(backend, "cache_tokens", 0) or 0)
        if self.sessions > 1 and cache_tokens:
            import tp_sessions
            if session_opts is not None:
                self.mgr = tp_sessions.SessionManager(slots=self.sessions,
                                                      pool_pages=cache_tokens // page_size,
                                                      page_size=page_size, **session_opts)
            else:
                self.mgr = tp_sessions.SessionManager.from_env(
                    pool_pages=cache_tokens // page_size, page_size=page_size, log=log)
                if self.mgr is None:      # env said 1 but the constructor was explicit
                    self.mgr = tp_sessions.SessionManager(slots=self.sessions,
                                                          pool_pages=cache_tokens // page_size,
                                                          page_size=page_size)
            self.log(f"tp: multi-session engine on ({self.sessions} slots; explicit per-request "
                     f'"session" ids 1..{self.sessions}; /tp/status carries the pool state)')
        # prompt reuse needs the backend's checkpoint ops (the CPU stand-ins mostly have none)
        if prefix is None or prefix is True:
            prefix = PrefixCache.from_env()
        self.prefix = None if self.mgr is not None else (prefix or None)
        if self.prefix is not None and not all(hasattr(backend, op) for op in ("stash", "restore", "drop_stash")):
            self.prefix = None
        if self.mgr is not None and not all(hasattr(backend, op) for op in ("stash", "restore", "drop_stash")):
            raise ValueError("multi-session mode needs the backend's checkpoint ops "
                             "(stash/restore/drop_stash)")
        self._drop_all_next = False       # reset_prefix_cache ran: rank 1 drops its copies at the next begin
        self._req: "_ReqCtx | None" = None   # the active request's session context (model thread)
        self._drop_images: list[int] = []    # sids whose rank-1 images must go at the next begin
        self._carry_evict: list[int] = []    # sids rank 1 must free at the next begin (a load
        self._carry_drop: list[int] = []     # failed before its frame left; keys ditto)
        # ---- stage 2: the per-rank NVMe tier (multi-session only; S=1 never builds one) ----
        self.tier = None
        if self.mgr is not None:
            import tp_disk
            hooks = ("disk_fingerprint", "disk_page_blob", "disk_page_store",
                     "disk_stash_pack", "disk_stash_unpack")
            if all(hasattr(backend, h) for h in hooks):
                try:
                    fp = backend.disk_fingerprint()
                    self.tier = tp_disk.tier_from_env(rank=0, fingerprint=fp, log=log,
                                                      default_on=True, **(disk_opts or {}))
                except Exception as exc:                                # noqa: BLE001  (degrade)
                    log(f"tp: disk tier unavailable ({type(exc).__name__}: {exc}); "
                        "eviction-only under pressure (EXL3_TP_DISK_CACHE)")
                    self.tier = None
            else:
                log("tp: disk tier off (backend lacks the disk hooks); eviction-only "
                    "under pressure")
        self._disk_timeout = env_float("EXL3_TP_DISK_TIMEOUT", 300.0)
        if self.tier is not None and self.tier.enabled:
            log("tp: cold sessions spill to " + str(self.tier.root) +
                f" (cap {self.tier.cap_bytes / (1 << 30):.0f} GiB); returning conversations "
                "reload instead of re-prefilling")

    # -- vision (rank 0 prepares and ships; rank 1 receives through the control channel)
    def _resolve_image_token(self) -> int | None:
        """Id of `<｜deepseek_image｜>`, the one placeholder the rendered prompt carries per
        image. The tokenizer's own entry wins (the HF tokenizer of the model folder); a
        `image_token_id` attribute on a test tokenizer is honored too."""
        explicit = getattr(self.tk, "image_token_id", None)
        if isinstance(explicit, int) and explicit >= 0:
            return int(explicit)
        hf = getattr(self.tk, "_tk", None)
        for getter in ("token_to_id",):
            fn = getattr(hf, getter, None)
            if fn is not None:
                try:
                    tid = fn("<｜deepseek_image｜>")
                    if tid is not None:
                        return int(tid)
                except Exception:                                       # noqa: BLE001  (not an HF tokenizer)
                    pass
        return None

    @property
    def supports_vision(self) -> bool:
        """The vendored server refuses image parts unless this is True (Engine contract)."""
        return self.vision is not None and self.image_token_id is not None

    def prepare_images(self, records: list[dict]) -> list[tp_vision.PreparedImage]:
        """Request image records -> prepared images (CPU, off the event loop: fetch, decode,
        geometry). Raises ImageError (a ValueError) on anything wrong; the server answers 400.

        The span each image will occupy is fixed here (unique dynamic-id range + exact row
        count from the tower's own resize math), so `prompt_ids` can expand placeholders and
        count the expanded length before any GPU work happens."""
        if not self.supports_vision:
            raise tp_vision.ImageError("this server has no vision support (no tower loaded: "
                                       "EXL3_TP_VISION=0, a failed load, or a text-only model)")
        if not isinstance(records, list) or not records:
            raise tp_vision.ImageError("prepare_images needs a non-empty list of image records")
        limit = tp_vision.env_int("EXL3_TP_MAX_IMAGES", tp_vision.DEFAULT_MAX_IMAGES)
        if len(records) > limit:
            raise tp_vision.ImageError(f"{len(records)} images exceed the per-request limit of "
                                       f"{limit} (EXL3_TP_MAX_IMAGES)")
        return [tp_vision.prepare_one(r, self.vision_cfg, self._mm_alloc) for r in records]

    def _vision_encode(self, images: list[tp_vision.PreparedImage]) -> list:
        """Run the tower + aligner for each image (model thread, no collectives) and bind the
        rows to the dynamic-id range assigned at prepare time. The block layout (start marker,
        image rows, per-row newline, end marker) is the tower's own `get_image_embeddings`; the
        row count is asserted against the prepare-time plan, so the CPU geometry and the tower
        can never quietly disagree."""
        import torch
        out = []
        dim = None
        with torch.inference_mode():
            for img in images:
                mme = self.vision.get_image_embeddings(None, img.pil)
                rows = mme.embeddings
                if rows.shape[0] != img.n_tokens:
                    raise RuntimeError(
                        f"vision tower produced {rows.shape[0]} rows for {img.alias} but the "
                        f"prepare-time plan promised {img.n_tokens} (grid {img.n_llm_h}x{img.n_llm_w}); "
                        "the resize math drifted -- refuse rather than mis-splice")
                if dim is None:
                    dim = int(rows.shape[1])
                elif int(rows.shape[1]) != dim:
                    raise RuntimeError(f"vision rows of inconsistent width: {rows.shape[1]} vs {dim}")
                img.rows = rows
                out.append(make_mm_embedding(img.first_index, rows, alias=img.alias,
                                             metadata={"key": img.key, "grid_llm": (img.n_llm_h, img.n_llm_w)}))
        return out

    def _ship_embeddings(self, images: list[tp_vision.PreparedImage], embeddings: list) -> None:
        """Hand rank 1 the tower rows for the request being armed: one imgmeta frame plus one
        binary frame per image, then the imgmeta_ok reply (validated: count, rows, sha256
        checked on rank 1 before it answers)."""
        meta, blobs = tp_vision.encode_imgmeta(images)
        meta["req"] = self.requests
        reply = self.channel.request_with_blobs(meta, blobs, timeout=env_float("EXL3_TP_IMGMETA_TIMEOUT", 120.0))
        if reply.get("op") != "imgmeta_ok" or int(reply.get("images", -1)) != len(embeddings) \
                or int(reply.get("rows", -1)) != sum(e.mm_length for e in embeddings):
            raise RuntimeError(f"rank 1 rejected the image shipment: {reply}")

    # -- tokenization (Kyojin HFTokenizer semantics: specials parsed from the prompt text, no extra BOS)
    def count_tokens(self, text: str) -> int:
        return len(self.tk.encode(text))

    def prompt_ids(self, prompt: str, images: list | None = None) -> list[int]:
        ids = [int(t) for t in self.tk.encode(prompt)]
        if not ids:
            raise ValueError("empty prompt")
        if images:
            # each <｜deepseek_image｜> occurrence becomes its image's dynamic-id span, in order;
            # the returned length is the EXPANDED length (what the cache bound must check)
            if not self.supports_vision:
                raise ValueError("image input is not supported on this server (no vision tower)")
            ids = tp_vision.expand_prompt_ids(ids, list(images), self.image_token_id)
            ids = [int(t) for t in ids]
        text_ceiling = max((t for t in ids if t < FIRST_MM_EMBEDDING_INDEX), default=-1)
        if text_ceiling >= self.vocab_size:
            raise ValueError(f"prompt token id {text_ceiling} outside the vocabulary (0..{self.vocab_size - 1})")
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
        with DSpark on, a second one exercises the plain fallback path too. Multi-session
        engines run both as ephemeral slots: a warm-up conversation must not occupy (or evict
        for) a real session slot."""
        n = max(8, min(int(prompt_tokens), self.ctx // 4))
        lo = min(1000, self.vocab_size - 1)
        ids = [(lo + i * 7919) % self.vocab_size or 1 for i in range(n)]
        t0 = time.perf_counter()

        async def go(speculative):
            async for _ in self.generate("", max_tokens=int(decode_tokens), temperature=0.0, top_p=1.0,
                                         stop=[], prompt_ids=ids, ignore_eos=True, speculative=speculative,
                                         ephemeral=True):
                pass

        asyncio.run(go(None))
        if self.dspark:
            asyncio.run(go(False))
        return time.perf_counter() - t0

    # -- prefix cache (rank 0 decides; rank 1 mirrors the begin/end frame fields in the same order:
    #    restore or fresh state, drops, prefill, save ... end: drops, save, release)
    #
    # Multi-session mode (mgr is not None) routes every request through _session_route first;
    # the helpers below then operate on the ACTIVE session's PrefixCache instead of the single
    # one, and checkpoint keys come from the manager (globally unique across sessions).
    def _active_px(self) -> "PrefixCache | None":
        if self.mgr is None:
            return self.prefix
        return self._req.session.px if self._req is not None else None

    def _session_plan(self, ids: list[int], max_tokens: int, *, spec: bool,
                      ephemeral: bool = False, explicit_sid: int | None = None):
        """EXL3_TP_SESSIONS >= 2, phase 1 (pure): pick the session (or the cold session to
        reload), plan whole-session evictions/spills and page growth. Mutates nothing, so
        the fallible pre-work (vision encoding, the imgmeta shipment) can still fail cleanly
        between plan and commit -- rank 0's records and rank 1's never diverge."""
        import tp_sessions
        if self.mgr is None:
            return None
        slack = tp_sessions.spill_slack(spec, getattr(self.backend, "dspark_block", 5))
        return self.mgr.plan(array("i", ids), need_tokens=len(ids) + int(max_tokens) + slack,
                             explicit_sid=explicit_sid, ephemeral=ephemeral)

    def _session_commit(self, plan, frame: dict | None = None,
                        tried: frozenset = frozenset()) -> "SessionAlloc | None":
        """Phase 2: apply the plan -- the disk dances first (cold reload, whole-session
        spills), then evictions, page growth and the begin-frame stamping. Nothing after
        this may fail before the begin frame is sent. Returns None in single-session mode.

        `tried` is the set of cold image ids already attempted for THIS request: each
        matching cold conversation is loaded at most once (a failed image is discarded,
        the replan skips it), and once the reload budget is spent the request falls back
        to a fresh session -- never a raise, never wrong output."""
        if self.mgr is None or plan is None:
            return None
        cold = plan.get("load_cold")
        if cold is not None:
            if cold.img in tried:
                # cannot happen through match(skip_cold=...): the image was discarded on
                # its failed attempt. Catch it anyway -- silently, never a 500.
                self.log(f"tp: routing bug: image {cold.img} matched twice in one request; "
                         f"falling back to a fresh session")
                self.mgr.discard_cold(cold.img)
                plan = self.mgr.plan(plan["ids"], need_tokens=plan["need_tokens"],
                                     explicit_sid=None if plan["ephemeral"]
                                     else plan.get("explicit_sid"),
                                     ephemeral=plan["ephemeral"], skip_cold=tried, cold_ok=False)
            else:
                self._disk_load(plan)   # success or clean fallback; either way, re-plan
                return self._session_replan(plan, frame, tried | {cold.img})
        evicted = self._apply_victims(plan)
        rt = self.mgr.commit(plan)
        sess = rt["session"]
        all_evicted = evicted + rt["evicted"]
        alloc = SessionAlloc(sess.pages, self.mgr.page_size, sess.sid)
        self._req = _ReqCtx(sess, alloc, all_evicted, ephemeral=sess.ephemeral,
                            prompt_len=len(plan["ids"]))
        if frame is not None:
            frame["session"] = sess.sid
            frame["pages"] = list(sess.pages)
            evict_sids = [sid for sid, _ in all_evicted]
            if self._carry_evict:
                evict_sids += list(self._carry_evict)
                self._carry_evict = []
            if evict_sids:
                frame["evict"] = evict_sids
            if self._drop_images:
                frame["drop_images"] = list(self._drop_images)
                self._drop_images = []
        if PREFIX_LOG and all_evicted:
            st = self.mgr.status()
            self.log(f"tp: sessions: request {self.requests} evicted sessions "
                     f"{[sid for sid, _ in all_evicted]} for {len(sess.pages)} pages; "
                     f"pool {st['pages_used']}/{st['pool_pages']} pages")
        return alloc

    # each matching cold conversation is reloaded at most once per request; beyond that
    # (pathological prefix-twin storms) the request serves fresh instead of thrashing
    COLD_LOAD_MAX = 8

    def _session_replan(self, plan, frame: dict | None, tried: frozenset):
        """A cold reload was attempted (it succeeded or fell back cleanly): re-plan
        against the new state with the tried images excluded. The chain is bounded --
        every round either turns a cold hot, discards a failed image, or falls back to a
        fresh session (cold_ok=False once the budget is spent), so it always settles."""
        cold_ok = len(tried) < max(self.COLD_LOAD_MAX, len(self.mgr.cold) + 1)
        again = self.mgr.plan(plan["ids"], need_tokens=plan["need_tokens"],
                              explicit_sid=None if plan["ephemeral"] else plan.get("explicit_sid"),
                              ephemeral=plan["ephemeral"], skip_cold=tried, cold_ok=cold_ok)
        return self._session_commit(again, frame, tried)

    def _apply_victims(self, plan) -> list[tuple[int, list]]:
        """Evict the plan's victims, spilling each to the disk tier first when possible
        (whole-session images; a failed spill demotes to plain eviction). Returns the
        evicted (sid, Checkpoints) pairs whose keys/sids the begin frame must carry."""
        evicted: list[tuple[int, list]] = []
        for victim in list(plan.get("victims", ())):
            if self._disk_spill(victim):
                continue                    # parked on NVMe; no drop/evict keys to carry
            ckpts = self.mgr.evict(victim, reason="pressure")
            evicted.append((victim.sid, ckpts))
        plan["victims"] = []
        return evicted

    # -- stage 2: the spill/reload dances (settled boundaries only; rank 0 drives, both
    #    ranks write their own image, nothing collectives-related rides the file IO) ------
    def _disk_request(self, frame: dict) -> dict:
        """One disk frame with a non-desyncing timeout protocol. A disk reply can be slow
        (NVMe writes); if it exceeds EXL3_TP_DISK_TIMEOUT the reply is still owed and MUST
        be drained before anything else may use the channel, or a later request would read
        it as its own. So: on timeout, block for the channel's full receive timeout and
        drain exactly that one reply -- it is honored like a normal answer (both ranks did
        the same work, they are in sync). If even the drain times out, rank 1 is wedged:
        fatal by the documented no-recovery rule, never a silent divergence."""
        try:
            return self.channel.request(frame, timeout=self._disk_timeout)
        except (socket.timeout, TimeoutError):
            self.log(f"tp: rank 1 exceeded EXL3_TP_DISK_TIMEOUT={self._disk_timeout:g}s on "
                     f"{frame.get('op')}; draining the late reply before anything else runs")
            reply = self.channel.recv()          # the one owed frame, full channel timeout
            self.log(f"tp: drained the late reply ({reply.get('op')}); honoring it")
            return reply

    def _disk_spill(self, victim) -> bool:
        """Park one victim session on the NVMe tier. Rank 0 streams its image first (one
        page blob at a time; cap enforced while writing), then rank 1 streams its own and
        acks; only then do both sides free the pages/stashes. Any failure falls back to
        plain eviction -- the session dies, never wrong output."""
        if self.tier is None or not self.tier.enabled:
            return False
        keys = [c.key for c in victim.px.ckpts]
        img = self.mgr.next_image_id()
        try:
            meta = {"positions": {str(c.key): c.pos for c in victim.px.ckpts},
                    "committed": victim.committed, "last_prompt_len": victim.last_prompt_len}
            size = self.tier.write_image(
                img, meta=meta, n_pages=len(victim.pages),
                page_blob_fn=lambda i: self.backend.disk_page_blob(victim.pages[i]),
                stash_payload={c.key: (c.pos, self.backend.disk_stash_pack(c.stash))
                               for c in victim.px.ckpts})
        except Exception as exc:                                        # noqa: BLE001  (fall back)
            self.tier.counters["spill_errors"] += 1
            self.log(f"tp: spill of session {victim.sid} failed on rank 0 "
                     f"({type(exc).__name__}: {exc}); evicting it instead")
            return False
        try:
            reply = self._disk_request({"op": "spill", "req": self.requests,
                                        "session": victim.sid, "image": img, "keys": keys})
            if reply.get("op") != "spill_ok" or int(reply.get("session", -1)) != victim.sid:
                raise RuntimeError(f"rank 1 refused the spill: {reply.get('op')}: "
                                   f"{reply.get('reason') or reply}")
        except Exception as exc:                                        # noqa: BLE001  (fall back)
            self.tier.drop(img)                 # my image is dead; rank 1 kept its state
            self.tier.counters["spill_errors"] += 1
            self.log(f"tp: spill of session {victim.sid} failed on rank 1 "
                     f"({type(exc).__name__}: {exc}); evicting it instead")
            return False
        cold, ckpts = self.mgr.spill(victim, img)
        self.mgr.spill_bytes_note(cold, size, int(reply.get("bytes", 0)))
        for c in ckpts:                         # mirrors rank 1's spill-time drops
            self.backend.drop_stash(c.stash)
        return True

    def _disk_load(self, plan) -> bool:
        """Reload the cold session this request matched: make room (spilling/evicting the
        plan's victims), allocate fresh pages, restore both ranks' images into them, and
        reheat the record. Returns True on success; on any failure both sides discard the
        image and the request falls back to a fresh session (full re-prefill)."""
        import tp_disk
        cold = plan["load_cold"]
        if self.tier is None or not self.tier.enabled:
            self.mgr.discard_cold(cold.img)
            return False
        if not self.tier.has(cold.img):
            self.log(f"tp: image {cold.img} of session {cold.sid} is gone; re-prefilling")
            self.mgr.discard_cold(cold.img)
            return False
        evicted = self._apply_victims(plan)
        session = None
        stashes: dict[int, Any] = {}
        try:
            session = self.mgr.begin_load(cold, plan["n_pages"])

            def blobs_into(i, blob):
                self.backend.disk_page_store(session.pages[i], blob)

            def stash_into(payload):
                # validated but NOT unpacked: rank 1 unpacks on its side only after both
                # images proved readable, and this side unpacks only after its ack -- a
                # failed load then leaves zero restored stashes on BOTH ranks (the mirror
                # stays equal and the fallback drops nothing asymmetrically)
                got = {int(k): (pos, packed) for k, (pos, packed) in payload.items()}
                want = sorted(k for k, _ in cold.ckpt_meta)
                if sorted(got) != want:
                    raise tp_disk.DiskCorrupt(f"image holds checkpoints {sorted(got)}, "
                                              f"the record says {want}")
                stashes.update(got)

            self.tier.read_image(cold.img, blobs_into=blobs_into, stash_into=stash_into,
                                 expect=self.tier.fingerprint)
            ckpts = self.mgr.reattach(session, cold)      # Checkpoints with stash=None until
        except Exception as exc:                                        # noqa: BLE001  (fall back)
            self.log(f"tp: reload of session {cold.sid} failed on rank 0 "
                     f"({type(exc).__name__}: {exc}); re-prefilling instead")
            self.tier.drop(cold.img)
            self._drop_images.append(cold.img)    # rank 1's image is spent too
            self.mgr.discard_cold(cold.img)
            if session is not None:
                for c in session.px.ckpts:       # rank 1 never saw a frame: nothing to mirror
                    if c.stash is not None:
                        try:
                            self.backend.drop_stash(c.stash)
                        except Exception:                               # noqa: BLE001
                            pass
                self.mgr.abort_load(session)
            # the victims' evictions stand on rank 0 but rank 1 has not heard of them (no
            # frame was sent): carry their keys/sids into the next begin frame
            for sid, cks in evicted:
                for c in cks:
                    try:
                        self.backend.drop_stash(c.stash)
                    except Exception:                                   # noqa: BLE001
                        pass
                self._carry_evict.append(sid)
                self._carry_drop.extend(c.key for c in cks)
            return False
        # rank 1 restores its half (the frame carries the evictions' keys/sids like a begin)
        frame = {"op": "load", "req": self.requests, "session": session.sid, "image": cold.img,
                 "pages": list(session.pages), "keys": [k for k, _ in cold.ckpt_meta],
                 "positions": [p for _, p in cold.ckpt_meta]}
        if evicted:
            frame["evict"] = [sid for sid, _ in evicted]
            frame["drop"] = [c.key for _, cks in evicted for c in cks]
        for sid, cks in evicted:
            for c in cks:                        # mirrors rank 1's load-frame drops
                self.backend.drop_stash(c.stash)
        try:
            reply = self._disk_request(frame)
            if reply.get("op") != "load_ok" or int(reply.get("session", -1)) != session.sid:
                raise RuntimeError(reply.get("reason") or str(reply))
        except Exception as exc:                                        # noqa: BLE001  (fall back)
            self.log(f"tp: reload of session {cold.sid} failed on rank 1 "
                     f"({type(exc).__name__}: {exc}); re-prefilling instead")
            self.tier.drop(cold.img)
            self.mgr.discard_cold(cold.img)       # already popped by begin_load: count it
            self.mgr.abort_load(session)          # no stashes were unpacked on either side
            return False
        # both images proved out: unpack this side's stashes. A failure here is a backend
        # fault past the point of clean fallback (rank 1 already holds its copies) -- fatal,
        # restart both ranks (the documented no-recovery rule), never wrong output.
        for c in session.px.ckpts:
            c.stash = self.backend.disk_stash_unpack(stashes[c.key][1])
        self.tier.drop(cold.img)                  # images are one-shot (no restart persistence)
        self.mgr.counters["cold_loads"] += 1      # only a fully-acked reload counts
        return True

    def _prefix_begin(self, ids: list[int], spec: bool, use_cache: bool, frame: dict):
        """Plan the request's prompt reuse and add it to the begin frame. Returns (checkpoint to
        restore or None, start position, key to save the post-prefill state under or None,
        checkpoints evicted to make room for it)."""
        if self.mgr is not None:
            return self._prefix_begin_ms(ids, spec, use_cache, frame)
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

    def _prefix_begin_ms(self, ids: list[int], spec: bool, use_cache: bool, frame: dict):
        """The multi-session flavor of _prefix_begin: reuse lives inside the routed session
        (ephemeral slots never restore or save), keys are globally unique, and whole evicted
        sessions' checkpoints ride the same drop list (their sids ride "evict")."""
        ctx = self._req
        sess = ctx.session
        if self._drop_all_next:
            frame["drop_all"] = True
            self._drop_all_next = False
        ck = None
        if use_cache and not ctx.ephemeral:
            ck = sess.px.plan(array("i", ids))
        start = ck.pos if ck is not None else 0
        save, evicted = None, []
        if use_cache and not ctx.ephemeral and len(ids) - 1 >= sess.px.min_tokens and len(ids) - 1 > start:
            save, evicted = self.mgr.reserve(sess, protect=ck)
        if ck is not None:
            frame["restore"], frame["start"] = ck.key, start
        # frame order == rank-0 drop order: evicted sessions' checkpoints first, then this
        # session's own prefix-slot evictions (check_mirror compares the op sequences)
        drops = [c.key for _, cks in ctx.evicted for c in cks] + [c.key for c in evicted]
        if self._carry_drop:
            drops += list(self._carry_drop)
            self._carry_drop = []
        if drops:
            frame["drop"] = drops
        if save is not None:
            frame["save"] = save
        return ck, start, save, [c for _, cks in ctx.evicted for c in cks] + list(evicted)

    def _prefix_state(self, ck: Checkpoint | None, start: int, evicted: list[Checkpoint],
                      spec: bool = False):
        """The request's state (restored from `ck` or fresh), and for spec requests the first
        position whose drafter main-kv row is known to belong to this sequence: 0 for a fresh
        state (the spec prefill writes every row), the start of the restored window for a
        checkpoint that carries one (put back here), `start` for one that does not.

        Session mode drops the evicted stashes BEFORE the restore, mirroring rank 1's
        begin-frame order (drops, evict frees, restore); the legacy single-session path keeps
        its historical restore-then-drops sequence byte for byte."""
        if self.mgr is not None:
            for c in evicted:
                self.backend.drop_stash(c.stash)
            state = self.backend.restore(ck.stash, start) if ck is not None else self.backend.new_state()
        else:
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
        px = self._active_px()
        if key is not None and px is not None:
            n = len(ids) - 1
            px.add(key, array("i", ids[:n]), self.backend.stash(state), self._draft_window_at(n, draft_from))

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
        (both ranks: the end frame's "rewind"), so the checkpoint matches the next turn.

        Multi-session mode settles the ACTIVE session's record the same way (mgr.finish), and
        frees ephemeral slots (image/warm-up) at this settled boundary: the end frame carries
        their sid in "evict" and rank 1 frees the record after the drops/save above."""
        px = self._active_px()
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
                and (self.mgr is None or not self._req.ephemeral)
                and save_at >= px.min_tokens and all(c.pos != save_at for c in px.ckpts)):
            if self.mgr is not None:
                key, evicted = self.mgr.reserve(self._req.session)
            else:
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
        if self.mgr is not None:
            ctx = self._req
            frame["session"] = ctx.session.sid
            if ctx.ephemeral:
                freed = self.mgr.release_ephemeral(ctx.session)
                for c in freed:
                    self.backend.drop_stash(c.stash)
                    keys.append(c.key)
                frame["evict"] = [ctx.session.sid]
            else:
                # a failed request leaves the header position where the record says it was:
                # its prompt never committed, and the retry must still match the session
                self.mgr.finish(ctx.session,
                                prompt_len=ctx.prompt_len if committed is not None else None)
        if keys:
            frame["drop"] = keys
        if PREFIX_LOG:
            where = f"session {self._req.session.sid}" if self.mgr is not None else "cache"
            self.log(f"tp: prefix cache ({where}): request {self.requests} reused {start} of "
                     f"{len(committed or ())} committed tokens; checkpoints at "
                     f"{sorted(c.pos for c in px.ckpts)}"
                     f"{f', dropped {keys}' if keys else ''}")

    # -- one request (the model thread; the op sequence Rank1Worker mirrors)
    def _run(self, ids: list[int], max_tokens: int, sampler: TopPSampler, eos_id: int | None,
             stops: list[str], cancel: asyncio.Event | None, emit: Callable[[str], None],
             use_cache: bool = True, images: list | None = None, session: int | None = None,
             ephemeral: bool = False) -> dict[str, Any]:
        # routing phase 1 (pure) -> everything fallible (vision encode + shipment) -> commit
        plan = self._session_plan(ids, max_tokens, spec=False,
                                  ephemeral=ephemeral or bool(images), explicit_sid=session)
        embeddings = self._vision_encode(images) if images else None
        if embeddings:
            self._ship_embeddings(images, embeddings)    # imgmeta + blobs + imgmeta_ok, before begin
        begin = {"op": "begin", "req": self.requests, "ids": [int(t) for t in ids]}
        alloc = self._session_commit(plan, begin)
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
            _prefill(self.backend, ids, state, False, start, embeddings, alloc)
            self._prefix_save(pending, state, ids)
            pending = None
            cur, pos = int(ids[-1]), len(ids) - 1
            # a prompt ending on an image span is refused in generate(); the decode loop's
            # first forward is therefore always a text token (no rows to attach)
            assert cur < FIRST_MM_EMBEDDING_INDEX, "image-span prompt tail slipped through"
            fed = list(ids)              # the tokens at positions [0, pos) once `cur` is forwarded
            # Lockstep rule: every stop decision happens after sampling, before the step frame.
            while True:
                row = _forward(self.backend, cur, pos, state, None, alloc)
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
        stats = {"cached_tokens": start, "new_tokens": n_new, "prompt_tokens": len(ids),
                 "time_prefill": t_first - t0, "time_generate": t1 - t_first,
                 "eos": eos, "eos_reason": reason, "spec_used": False}
        if self.mgr is not None and self._req is not None:
            stats["session"] = self._req.session.sid
        if images:
            stats.update(images=len(images), image_tokens=sum(1 for t in ids if t >= FIRST_MM_EMBEDDING_INDEX))
        return stats

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
        begin ("drop_all"). Multi-session mode forgets every session the same way."""
        if self.prefix is not None or self.mgr is not None:
            await asyncio.get_running_loop().run_in_executor(self._pool, self._prefix_clear)

    def _prefix_clear(self) -> None:
        with self._busy:
            if self.mgr is not None:
                for c in self.mgr.clear():
                    try:
                        self.backend.drop_stash(c.stash)
                    except Exception:                                   # noqa: BLE001
                        pass
                if self.tier is not None:         # cold sessions die with the reset too
                    for img in list(self.tier._sizes):
                        self.tier.drop(img)
                # drop_all makes rank 1 forget every record, stash and image: anything we
                # were still carrying for it (late evictions from a failed load, spent
                # image ids) must not leak into the next begin frame
                self._carry_evict.clear()
                self._carry_drop.clear()
                self._drop_images.clear()
                self._drop_all_next = True
                return
            px = self.prefix
            for c in px.ckpts:
                self.backend.drop_stash(c.stash)
            px.ckpts.clear()
            px.kv_ids = array("i")
            self._drop_all_next = True

    def _prefix_reset(self, end: dict) -> None:
        if self.mgr is not None:
            # the record no longer matches what the ranks hold: forget every session, and
            # rank 1 drops its stashes, records and images at the end frame's drop_all
            for c in self.mgr.clear():
                try:
                    self.backend.drop_stash(c.stash)
                except Exception:                                       # noqa: BLE001
                    pass
            if self.tier is not None:
                for img in list(self.tier._sizes):
                    self.tier.drop(img)
            self._carry_evict.clear()            # drop_all below supersedes any carry
            self._carry_drop.clear()
            self._drop_images.clear()
            end.pop("save", None)
            end.pop("rewind", None)
            end.pop("drop", None)
            end.pop("evict", None)      # drop_all below clears every session record too
            end["drop_all"] = True
            return
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

    def session_status(self) -> dict[str, Any]:
        """Pool/session state for /tp/status (single-session engines report the plain
        shape). The manager is mutated on the model thread, so the snapshot is taken under
        the engine lock; if a request holds it for longer than the (short) wait, the
        report says so instead of reading a half-mutated structure."""
        if self.mgr is None:
            return {"slots": 1, "pool_pages": None, "pages_used": None, "pages_free": None,
                    "sessions": [], "counters": {"requests": self.requests}}
        if not self._busy.acquire(timeout=2.0):
            return {"slots": self.mgr.slots, "pool_pages": self.mgr.pool.pool_pages,
                    "busy": True, "sessions": [], "cold_sessions": [],
                    "counters": {**self.mgr.counters, **self.mgr.pool.counters}}
        try:
            st = self.mgr.status()
        finally:
            self._busy.release()
        if self.tier is not None:
            st["disk"] = self.tier.status()
        return st

    # -- one spec request (the model thread; mirrors the round/accept sequence of Rank1Worker)
    def _run_spec(self, ids: list[int], max_tokens: int, stops: list[str], cancel: asyncio.Event | None,
                  emit: Callable[[str], None], spec_k: int | None, eos_id: int | None,
                  sampler: TopPSampler | None = None, use_cache: bool = True,
                  images: list | None = None, session: int | None = None,
                  ephemeral: bool = False) -> dict[str, Any]:
        import torch                                                    # the whole request runs under inference_mode (as the script)
        # routing phase 1 (pure) -> vision encode/shipment -> commit (nothing fails after it)
        plan = self._session_plan(ids, max_tokens, spec=True,
                                  ephemeral=ephemeral or bool(images), explicit_sid=session)
        embeddings = self._vision_encode(images) if images else None
        if embeddings:
            self._ship_embeddings(images, embeddings)    # imgmeta + blobs + imgmeta_ok, before begin
        begin = {"op": "begin", "req": self.requests, "ids": [int(t) for t in ids], "spec": True}
        alloc = self._session_commit(plan, begin)
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
                pparams = _prefill(self.backend, ids, state, True, start, embeddings, alloc)
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
                    logits, vparams = _verify(self.backend, [int(anchor)] + [int(t) for t in drafts],
                                              position, state, alloc)
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
        stats = {"cached_tokens": start, "new_tokens": n_new, "prompt_tokens": len(ids),
                 "time_prefill": t_first - t0, "time_generate": t1 - t_first,
                 "eos": eos, "eos_reason": reason, "spec_used": True,
                 "spec_rounds": rounds, "spec_k_last": k if rounds else None,
                 "accepted_draft_tokens": accepted_total, "rejected_draft_tokens": drafted_total - accepted_total,
                 "draft_steps": rounds,
                 "accepted_per_step": (accepted_total / rounds) if rounds else 0.0}
        if self.mgr is not None and self._req is not None:
            stats["session"] = self._req.session.sid
        if images:
            stats.update(images=len(images), image_tokens=sum(1 for t in ids if t >= FIRST_MM_EMBEDDING_INDEX))
        return stats

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float,
                       stop: list[str], cancel: asyncio.Event | None = None, stats: dict | None = None,
                       prompt_ids: list[int] | None = None, ignore_eos: bool = False, cache_prompt: bool = True,
                       images: list | None = None, speculative: bool | None = None,
                       spec_k: int | None = None, session: int | None = None, ephemeral: bool = False,
                       **_: Any) -> AsyncIterator[str]:
        if session is not None and self.mgr is None:
            raise ValueError('"session" needs a multi-session server (start both ranks with '
                             'EXL3_TP_SESSIONS >= 2)')
        if images:
            if not isinstance(images, list) or not all(isinstance(x, tp_vision.PreparedImage) for x in images):
                raise ValueError("images must be a list of images this engine prepared (prepare_images)")
            if not self.supports_vision:
                raise ValueError("image input is not supported on this server (no vision tower)")
        if prompt_ids is not None:
            ids = [int(t) for t in prompt_ids]
        elif images:
            ids = self.prompt_ids(prompt, images)          # expand placeholders here (direct callers)
        else:
            ids = self.prompt_ids(prompt)
        if len(ids) < 2:
            raise ValueError("prompt must hold at least 2 token ids (prefill feeds all but the last)")
        # image rows ride as dynamic ids >= 1e9; they are legal only as the expanded spans of
        # this request's images (never as bare numbers -- /v1/completions prompt ids stay text)
        mm_positions = [i for i, t in enumerate(ids) if t >= FIRST_MM_EMBEDDING_INDEX]
        if mm_positions and not images:
            raise ValueError(f"prompt holds {len(mm_positions)} image row id(s) outside the vocabulary "
                             f"(first at position {mm_positions[0]}) but no images were sent")
        if images and not mm_positions:
            raise ValueError("the prompt has no image placeholders for the images sent "
                             "(the rendered prompt needs one <｜deepseek_image｜> per image)")
        if mm_positions and ids[-1] >= FIRST_MM_EMBEDDING_INDEX:
            raise ValueError("the prompt must end with a text token, not an image span "
                             "(append text after the last image)")
        if len(ids) + 1 > self.ctx:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit the context of {self.ctx}"
                             + (" (image spans expand to their full token counts)" if images else ""))
        # Engine/library bounds, checked BEFORE the begin frame reaches rank 1: past the cache
        # bound the DSA pool faults inside a collective and rank 1's NCCL peer (rank 0) hangs
        # for its full watchdog timeout instead of seeing an error
        cache_cap = int(getattr(self.backend, "cache_tokens", 0) or 0)
        if cache_cap and len(ids) + 2 > cache_cap:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds the engine cache bound of "
                             f"{cache_cap - 2} usable tokens (EXL3_TP_CACHE_TOKENS={cache_cap}); "
                             f"refused before any rank dispatch")
        max_tokens = max(1, min(int(max_tokens), self.ctx - len(ids) - 1))
        # Image requests bypass the prefix cache for now: the dynamic ids are unique per
        # allocation, so identifying "the same image" across requests needs pixel hashing
        # (deliberately not in this first version). cache_prompt stays effective for text.
        if images:
            cache_prompt = False
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
                                              spec_k, eos_id, sampler, use_cache=bool(cache_prompt),
                                              images=images, session=session, ephemeral=ephemeral)
                    else:
                        done = self._run(ids, max_tokens, sampler, eos_id, [s for s in (stop or []) if s],
                                         cancel, lambda piece: loop.call_soon_threadsafe(q.put_nowait, piece),
                                         use_cache=bool(cache_prompt), images=images, session=session,
                                         ephemeral=ephemeral)
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
