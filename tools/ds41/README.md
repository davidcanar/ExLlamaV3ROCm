# tools/ds41 — OpenAI-compatible server for the two-node EXL3 DS4.1 rig

An OpenAI-compatible HTTP layer over the working two-node cached prefill/decode path
(`scripts/tp_ds41_generate.py`, left untouched). Rank 0 (Box1) accepts jobs and renders prompts;
rank 1 (Box2) mirrors every model call over a framed control channel. Model collectives keep
running over Odinlink/RCCL exactly as in the manual script.

```
Box1  tp_serve.py   rank 0: aiohttp server (vendored Kyojin layer) ─┐
                    TPModelBackend (EXL3 TP load, cache, prefill/decode)
                          │ control frames (Thunderbolt TCP, 4-byte LE length + JSON)
                          ▼
Box2  tp_rank1.py   rank 1: Rank1Worker (same model calls, same order)
                    TPModelBackend ── NCCL/RCCL ──▶ Odinlink TB5 ◀── NCCL/RCCL ── TPModelBackend (Box1)
```

## Attribution

The HTTP/chat/parser/test layer is vendored, unmodified, from **Kyojin**
(`Yamz-Labs/kyojin`, commit `e2f37564bbae01a6ea8ad4176dfd97db09c253aa`, release 1.8.2),
MIT License, Copyright (c) 2025 Turboderp — see `LICENSE.kyojin`. Each vendored file carries a
header naming its source path. Vendored files:

| file | source in Kyojin | role |
|---|---|---|
| `ds41_serve.py`* | `tools/ds41/serve.py` | create_app: all routes, streaming, queue/worker, stop handling |
| `ds41_chat.py` | `tools/ds41/ds41_chat.py` | official DS4.1 prompt rendering (thinking/effort/tools) |
| `ds41_parse.py` | `tools/ds41/ds41_parse.py` | thinking / DSML tool-call parsing (stream and whole) |
| `ds41_engine.py` | `tools/ds41/ds41_engine.py` | Engine contract, StubEngine, HF/byte tokenizers |
| `serve_compat.py`, `startup_health.py`, `serve_metrics.py`, `serve_ping.py`, `serve_reqlog.py` | `tools/` | request helpers, load-progress `/health`, metrics, SSE keep-alive, request log |
| `conftest.py`, `ds41_testkit.py`*, `test_ds41_chat.py`, `test_ds41_parse.py`, `test_ds41_serve.py`, `test_serve_compat.py`, `testdata/` | `tools/ds41/` | golden tests against the official reference encoding (DeepSeek-AI, MIT) |

Kyojin's `test_serve_logging.py` and `test_startup_health.py` are not vendored: they exercise the
three Kyojin servers (`tools/glm`, `tools/qwen`, `tools/strix_halo`) that are not part of this
folder; the helpers themselves are covered through the suite above.

`*` adapted: `ds41_testkit.serve_module()` loads `tp_serve.py`; `tokenizer_json()` also checks
`$EXL3_TP_MODEL_DIR`; `ds41_serve.py` carries two marked `LOCAL (exl3-tp)` blocks (applied by
`vendor_kyojin.py`) that validate and forward the per-request `speculative` / `spec_k` fields to
`engine.generate` -- everything else in it is byte-identical upstream.

New code in this folder: `tp_engine.py` (control protocol, `TPModelBackend`, `Rank1Worker`,
`TPEngine`; sampling/detok adapted from Kyojin `ds41_model.py`, model calls mirroring
`scripts/tp_ds41_generate.py`), `tp_serve.py` (rank-0 entry point + `/tp/status`), `tp_rank1.py`
(rank-1 entry point), `tp_vision.py` (CPU-side image handling for the two-node server: safe
loading, the V4.1 geometry plan, dynamic-id ranges and the rank-0→rank-1 row codec — no GPU
needed to import or test it), `test_tp_control.py` (CPU tests incl. the rank-0/rank-1 lockstep
guarantee), `test_tp_chunked_prefill.py` (CPU regression tests for the long-prompt fix:
cache-sized block tables, span-chunked prefill, pre-dispatch bounds, startup memory floor),
`test_tp_vision.py` (CPU tests of the vision path incl. both ranks over loopback), the
`tp_*.sh` launchers (environment identical to `scripts/run_rank.sh`, which stays untouched),
and `vendor_kyojin.py` (the script that copied the vendored files above, headers included —
rerunnable against `src/kyojin` to refresh the vendored layer).

## Endpoints

Inherited verbatim from the vendored layer: `GET /health`, `GET /v1/models`, `GET /metrics`,
`POST /v1/chat/completions` (stream and non-stream), `POST /v1/completions`, `POST /completion`,
`POST /tokenize`, `POST /apply-template`. Added: `GET /tp/status` (rank-1 liveness, request
count, context). `/health` answers with load progress (503 + stage) while the model loads.
Thinking mode, `reasoning_effort` (numeric and levels), tool schemas, tool-call round trips,
`cache_salt` (a new salt drops the prefix cache, so namespaces never share checkpoints), `stop` strings, `max_tokens`, client-disconnect
cancellation — all inherited from the vendored, golden-tested layer.

## DSpark (MTP speculative decoding) — optional, opt-in

Start both ranks with `EXL3_TP_DSPARK=1` (the launcher forwards it to Box2). The load then
mirrors `scripts/tp_ds41_dspark.py`: trunk cache `max_history=8`, per-rank budgets 95/104 GiB,
and on rank 0 the `mtp.*` drafter (layer-split, `attach_to` the trunk, private CPU embed + GPU
head so drafting issues no collectives) with its paged main-kv refreshed from the trunk taps.
The handshake verifies both ranks agree on DSpark; a mismatch is fatal at startup. Long prompts
keep DSpark: the chunked spec prefill exports the taps on every chunk (both ranks, identical
dispatches) and rank 0 primes the drafter's kv rows at each chunk's absolute offset — the same
per-position math as the proven single-shot priming, bounded to `EXL3_TP_SPAN` rows at a time.

Per request (chat completions): `"speculative": true|false` (null = server default: **on** when
enabled, greedy and sampled requests alike) and `"spec_k": 1..5` (cap on drafted tokens per
round). Sampled requests (temperature > 0) accept drafts by exact speculative sampling against
the greedy drafts (`tp_engine._accept_sampled`): draft d_i is kept with probability p_i(d_i) under
the request's temperature/top-p distribution, the first rejection draws from p_i with d_i
removed, so the output distribution is exactly that of plain sampling. `EXL3_TP_SPEC_SAMPLED=0`
(rank 0) restores the old rule (sampled requests on the plain path). `"speculative": true` on a
server without DSpark is a clean 400. Spec stats (`spec_rounds`, `accepted_draft_tokens`,
`rejected_draft_tokens`) land in `timings` and `/metrics` (llama.cpp spec-decode counters).

```bash
EXL3_TP_DSPARK=1 ~/exl3ds4/tools/ds41/tp_serve_odl.sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Name three primes."}],
  "max_tokens": 96, "temperature": 0, "speculative": true, "spec_k": 5}'
```

Protocol: a spec request prefill exports the DSpark taps on both ranks; each round rank 0
drafts, sends `round {window}`, both run one multi-row verify forward, rank 0 accepts (prefix
match when greedy, speculative sampling otherwise), sends `accept {a}`, and both rewind
identically (mirrored TP dispatch). Greedy output is guaranteed equal to plain greedy (the
acceptance rule emits the target's argmax as the bonus token); all stop decisions land on
settled round boundaries. Knobs: `EXL3_TP_BUDGETS=95,104`,
`EXL3_TP_DSPARK_CONF` (draft confidence threshold, stock 0.5, maps to the engine's
`EXL3_DSPARK_CONF` as `EXL3_DS41_CONF` does in the script).

## Long prompts, block tables and memory (2026-10-28 fix)

A ~7.8K-token prompt on a 512K cache (DSpark on) used to die with
`DSA pool overflow: entry 3892 beyond block table (8 pages)`: every trunk call passed
`batch_shape = (1, span)`, and the library sizes the DSA pool's block table from
`batch_shape` — 2048/256 = 8 pages. The assert fired inside a collective, rank 1 died and
rank 0 hung on NCCL until its 600 s watchdog killed it; the same one-shot prefill transiently
ate ~20 GiB of unified VRAM and dropped free memory below 100 MiB. The engine now:

- sizes **every** trunk call (prefill chunks, decode steps, spec verifies) to
  `batch_shape = (1, cache_tokens)` — the block table spans the whole pool;
- chunks prefill to `EXL3_TP_SPAN` rows (default 2048) with `past_len` at each chunk's
  absolute offset, bounding the activation transient to the loader's measured
  `max_chunk_size` (both ranks derive the identical chunk plan — the launcher forwards
  `EXL3_TP_SPAN`/`EXL3_TP_CACHE_TOKENS` to Box2; a mismatch would wedge the collectives);
- validates prompt/context against the cache **before** the `begin` frame (rank 0) and again
  at `begin` (rank 1), so an oversized request is a clean HTTP error instead of a
  mid-collective fault that hangs the peer;
- loads DSpark with per-rank budgets **95/104 GiB** (was 101/110) for headroom, and guards a
  startup memory floor: each rank logs free memory after the trunk and draft loads and
  refuses to serve below `EXL3_TP_MEM_FLOOR` (default 2 GiB, 0 disables) — a starved rank
  would otherwise die at the first long prefill and wedge the peer;
- returns the caching allocator's unused reserved segments to the system after every
  chunked request (`trim_memory`, both ranks, local-only): a long prompt's transient
  activations otherwise stay reserved (~1.2 GiB per box) after the request ends.

Measured on the 2× Strix Halo deployment (124 GiB unified memory per box, 512K cache,
DSpark on): with weights + 512K cache + drafter resident, idle system-available is
~2.1 GiB (Box1/rank 0) and ~4.3 GiB (Box2/rank 1); an ~8.4K-token spec prefill dips Box1
to ~0.7 GiB for its ~27 s duration and recovers with the trim. Budgets barely move the
footprint (weights + fixed-size 512K pools dominate: 101/110 vs 95/104 changed free memory
by ~0.5 GiB), so the real headroom knobs are `EXL3_TP_CACHE_TOKENS` and `EXL3_TP_SPAN`
(halving the span halves the per-chunk transient at roughly half the prefill rate).

Validated live (2× Strix Halo, 512K cache, DSpark on): ~8.4K-token spec prompts, short plain
and spec prompts, back-to-back requests on one server, server healthy throughout.

## Control protocol (rank 0 ⇄ rank 1)

Framed JSON over the Thunderbolt TCP control channel (`EXL3_TP_CONTROL_PORT`): 4-byte
little-endian length prefix + UTF-8 JSON. Binary payloads (vision rows) use the same prefix
with a `0x01` tag byte ahead of the raw bytes — a byte no JSON frame can start with.
`hello`/`hello_ok` (protocol + vocab + DSpark check) after both loads; then per request
`begin {ids, spec}` → fresh state, prefill (with the DSpark tap export on spec requests), and
in spec mode forward #1 of the last prompt token only on plain requests; `step {token}` per
plain token, or `round {k, window}` + `accept {a}` per spec round (one multi-row verify + one
mirrored rewind); `end` → both release the state. Image requests first send
`imgmeta {n, images[]}` followed by `n` binary frames (the tower's rows per image: alias,
dynamic-id range, rows/dim/dtype, sha256); rank 1 validates, rebuilds its
`indexed_embeddings` and answers `imgmeta_ok` **before** rank 0 sends `begin`, so a corrupt
shipment is a clean error, never a mid-collective fault. Protocol 3 added the prefix-cache
fields (`begin` may carry `restore`/`start` — restore checkpoint K at position S, prefill
only `ids[S:]` — plus `drop`, `save`, `drop_all`; `end` may carry `drop`, `rewind` and
`save`; rank 0 decides all of them and rank 1 applies them in the same order), and protocol 4
added the vision frames (`hello.vision`, `imgmeta` + binary blobs; a proto-3 rank 1 would
misread an imgmeta frame as JSON and desync, so the handshake refuses it). `ping`/`pong` for
`/tp/status`, `shutdown` for clean exit.

**Lockstep rule:** every stop decision (eos, stop string, `max_tokens`, cancel) happens on
rank 0 *after* sampling a token and *before* sending its `step`. Once a `step` is sent the next
forward is committed on both ranks, so the collective sequences never diverge
(`test_tp_control.py` asserts call-for-call equality of the two ranks' model calls).

## Prefix cache (2026-10-28)

Follow-up turns skip the part of the prompt the cache already holds. Every trunk call uses the
identity block table over the whole cache, so the paged DSA pools are position-indexed and
after a request positions `[0, E)` still hold the kv of the `E` tokens it committed. What a
new request cannot rebuild from the pools is the per-slot recurrent state (SWA ring,
compressor windows and overlaps, Engram id ring), so both ranks stash it (`DSV4State.stash`,
host memory, no collectives) after each prefill (position `len(ids) - 1`, which with the
DS4.1 template runs through `<｜Assistant｜>`: the part the next turn renders identically,
since it drops the old turn's reasoning) and at the end of each reply (spec requests rewind
the accepted drafts past the emitted text first). For a new prompt, rank 0 restores the
longest checkpoint inside the common prefix of the prompt and the committed sequence, and
both ranks prefill only the rest; `cached_tokens` / `timings.cache_n` report the reuse.

- `PrefixCache` (rank 0) keeps the invariant that every retained checkpoint's tokens are a
  prefix of the committed sequence: after each request it drops the checkpoints the request's
  writes invalidated (any request, `cache_prompt: false` included), and evicts the least
  recently used when the slots are full.
- The DSpark drafter (rank 0) only reads the main-kv rows of the last 128 positions, so its
  main-kv is a ring of `EXL3_TP_DRAFT_RING` pages (10 by default: one 2048-row prefill chunk
  plus the window; 7.5 MiB instead of 1.5 GiB at 512K). Spec checkpoints keep that window
  (384 KiB) and a restore puts it back, so drafting after a restore reads the right rows even
  when the ring has wrapped since. A checkpoint from a plain request has no window: restoring
  it is still exact, only the drafts within the first 128 positions are poorer.
- `cache_prompt: false` skips restore and checkpoints for that request; a new `cache_salt`
  drops every checkpoint (`reset_prefix_cache`).
- One linear sequence: a request that diverges early (a client's side request for a title,
  say) overwrites the cached tokens past the shared prefix, and the next turn of the main
  conversation re-prefills from there.

## Vision (image input, 2026-10-29)

Rank 0 loads the tower + aligner (`component = "vision"`, the PR #449 architecture
`deepseek_v41_vision.py`) beside the TP trunk — a separate non-TP model, so its forward runs
no collectives and rank 1's dispatch sequence is untouched. Rank 1 never loads it: image rows
travel per request as `imgmeta` + binary frames, and both ranks attach the identical
`indexed_embeddings` list to every prefill chunk (the trunk is natively multimodal through the
dynamic ids ≥ 1000000000: MoE `bias_vl` routing, the Engram n-gram block and the PLE mapping
all key off them).

- **Request path:** the vendored chat layer renders one `<｜deepseek_image｜>` per image part
  (`render_chat_vl`); `prepare_images` (CPU, off the event loop) loads bytes safely — data:
  URLs, bare base64 and http(s), capped at 32 MiB fetched / 100 Mpx decoded, EXIF-oriented,
  every failure a clean 400 — and plans each image's exact token span from the tower's own
  resize math; `prompt_ids(prompt, images)` splices each placeholder into its image's unique
  dynamic-id range, so context/cache bounds check the *expanded* length; `generate` runs the
  tower once per image on the model thread, ships the rows, prefills and decodes as usual.
- **Prefix cache:** image requests bypass it entirely (unique ids per allocation make
  cross-request identity a pixel-hash problem, deliberately not attempted here). Text requests
  keep the full cache, including against the text prefix an image request left behind.
- **DSpark composes** with vision (the drafter only ever sees the anchor and text rows). A
  first run that does not fit VRAM beside the drafter is practical per launch:
  `EXL3_TP_DSPARK=0 ./exl3ds4.sh restart` (both ranks read it, budgets fall back to the
  no-drafter defaults, no file edits). If the tower still does not fit, the rank degrades to
  text-only with a loud log instead of wedging (`EXL3_TP_VISION_STRICT=1` makes it fatal).
- **Limits:** prompts must end on a text token (the chat template always does); at most
  `EXL3_TP_MAX_IMAGES` (8) images per request; `EXL3_TP_IMAGE_HTTP=0` disables URL fetches.

Validated live (2026-10-29, the running server, `proto 4`, DSpark on, 512K cache):
`/health` and `/tp/status` report `vision: true`; an 800×600 test card reading **BLUE HERON
42** comes back verbatim with plain decoding and with DSpark (`"speculative": true`, 5 draft
tokens accepted), streamed over SSE (chunked deltas + `[DONE]`), at 336 expanded prompt
tokens and ~1.4 s per request end to end; `data:image/png;base64,AAAA` is a clean 400 that
reaches neither rank's model; image requests report `cached_tokens: 0` (cache bypassed) and a
text follow-up after image traffic still restores 1644 of its prompt tokens.

## Launch (from Box1)

```bash
# 1. one-time: replicate this folder to Box2 (same absolute paths as the model)
ssh davidcanar@10.0.2.2 'mkdir -p /home/davidcanar/exl3ds4/tools'
scp -r /home/davidcanar/exl3ds4/tools/ds41 davidcanar@10.0.2.2:/home/davidcanar/exl3ds4/tools/
ssh davidcanar@10.0.2.2 'chmod +x /home/davidcanar/exl3ds4/tools/ds41/*.sh'

# 2. start both ranks (rank 1 over SSH, rank 0 = HTTP on 0.0.0.0:8080, foreground)
/home/davidcanar/exl3ds4/tools/ds41/tp_serve_odl.sh            # [http-port master-port control-port]

# 3. in another shell / from the LAN
curl -s http://192.168.0.102:8080/health       # 503 "loading" with stage, then {"status":"ok",...}
curl -s http://192.168.0.102:8080/v1/models
curl -s http://192.168.0.102:8080/tp/status    # rank1: ok | busy | disconnected
curl -s http://192.168.0.102:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role": "user", "content": "Name three prime numbers."}],
  "max_tokens": 200}' | python3 -m json.tool
curl -N http://192.168.0.102:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role": "user", "content": "Say hi in French."}],
  "stream": true, "max_tokens": 128}'

# stop: Ctrl-C in the launcher (rank 1 gets `shutdown`, unloads, exits 0)
```

Logs: `~/exl3ds4/logs/tpserve-rank{0,1}.log`. The launchers export the same ROCm/Odinlink/NCCL
environment as `scripts/run_rank.sh` (which is not modified). Single-host CPU smoke of the HTTP
layer: `/home/davidcanar/exl3ds4/.venv/bin/python tp_serve.py --loader stub --port 8090`.

## Tests (CPU only, Box1)

```bash
cd /home/davidcanar/exl3ds4/tools/ds41
DS41_REF_DIR=/home/davidcanar/exl3ds4/model ../../.venv/bin/python -m pytest -q \
    test_ds41_chat.py test_ds41_parse.py test_ds41_serve.py test_tp_control.py \
    test_tp_chunked_prefill.py test_tp_prefix_cache.py test_serve_compat.py
PYTHONPATH=/home/davidcanar/exl3ds4/src/exllamav3-dsv41f \
    ../../.venv/bin/python -m pytest -q test_tp_vision.py     # engine-level vision tests
```

- `test_ds41_chat.py` — golden: rendered prompts byte- and token-identical to the official
  reference encoding (`testdata/reference`, DeepSeek-AI MIT); real-tokenizer test uses
  `$DS41_REF_DIR/tokenizer.json`.
- `test_ds41_parse.py` — thinking/DSML tool parsing, whole and streamed at any chunk size.
- `test_ds41_serve.py` — the vendored HTTP layer over StubEngine: streaming, tool round trips,
  tokenize, completions, defaults, errors (including the new `speculative`/`spec_k` 400s).
- `test_tp_control.py` - new: framing (incl. `round`/`accept` windows), handshake, **lockstep
  equality of both ranks' model calls**, per-request state reset, eos/`max_tokens`/stop/cancel
  at token boundaries (plain and spec), every DSpark accept count (a=0/partial/a=k/k=0) with
  token equivalence to plain greedy, position/rewind arithmetic, mode switching at settled
  boundaries, protocol guards, rank-1 liveness, and the vendored server running on real
  `TPEngine`+`Rank1Worker` pairs (plain, spec, fallback and equivalence over HTTP).
- `test_tp_chunked_prefill.py` - the long-prompt regressions: `batch_shape`/block tables sized
  to the cache in prefill/forward/verify, span-chunked prefill with correct `past_len` (pure in
  span/cache/len, so both ranks derive identical plans), per-chunk DSpark tap exports and
  drafter-kv priming at chunk offsets, cache-bound refusals on rank 0 (before dispatch) and
  rank 1 (clean error frame), the startup memory-floor guard, and the chunk-for-chunk rank-0 vs
  rank-1 mirror over a real loopback control channel.
- `test_tp_prefix_cache.py` - prompt reuse with stand-ins that model the cache semantics
  (position-indexed kv that survives requests, a hash-chain recurrent state, verify rows of
  rejected drafts left behind): every scenario (multi-turn, retry, verbatim continuation,
  divergence, plain/spec mixes, `cache_prompt: false`, LRU eviction, a rank-0 failure
  mid-request, `cache_salt`) runs with the cache on and off and the replies must be identical;
  both ranks' model calls and checkpoint tables must match. The spec stand-in keeps its drafter
  rows in a small ring and counts drafts that read rows of another sequence: none may, except
  right after restoring a plain request's checkpoint; a long reply that wraps the ring past a
  checkpoint must still draft from the right rows after the restore.
- `test_tp_vision.py` - the vision support: safe image loading (data:/base64/http(s), byte +
  pixel caps, scheme refusal), the V4.1 geometry plan against the architecture's own
  `plan_resize`, unique dynamic-id ranges/aliases and placeholder expansion, JSON+binary frame
  sharing on one channel, sha256-validated rank-1 reconstruction (corruption/overlap/truncation
  refused), `indexed_embeddings` on every prefill chunk at unchanged offsets (never on text),
  engine end-to-end pairs over loopback (tower stand-in, bit-exact rows on rank 1, cache
  bypass, broken-prompt refusals before any dispatch), and the vendored HTTP layer over the
  real `TPEngine` with images (needs the exllamav3 tree on `PYTHONPATH`; skips otherwise).

## Knobs and defaults

| env / flag | default | meaning |
|---|---|---|
| `EXL3_TP_SAFE_CTX` | 750 | serving context when `-c` is not given (the deployment's `exl3ds4.sh` raises it to the full cache) |
| `tp_serve.py -c N` | — | explicit context override (clamped to `EXL3_TP_CACHE_TOKENS - 2`) |
| `EXL3_TP_CONTROL_PORT` / `--control-port` | 29651 | rank-0 control channel port |
| `EXLLAMA_MASTER_ADDR` | 10.0.2.1 | control bind + NCCL rendezvous address |
| `EXL3_TP_CACHE_TOKENS` | 2048 | cache slots per rank (multiple of 256; block tables and the DSA pool span it) |
| `EXL3_TP_SPAN` | 2048 | prefill chunk in rows = loader `max_chunk_size`; both ranks must match |
| `EXL3_TP_USE_PER_DEVICE` | 110.0 | plain-path GiB budget per rank (replicated) |
| `EXL3_TP_MEM_FLOOR` | 2.0 | GiB kept free per rank at startup (log + guard; 0 disables) |
| `EXL3_TP_DSPARK` | off | DSpark on both ranks: max_history 8, budgets 95/104, drafter on rank 0 |
| `EXL3_TP_BUDGETS` | 95,104 when DSpark | per-rank GiB budgets (comma list, same on both ranks) |
| `EXL3_TP_DSPARK_CONF` | 0.5 (stock) | draft confidence threshold (maps to `EXL3_DSPARK_CONF`) |
| `EXL3_SERVE_WARMUP=0` | on | skip the tiny end-to-end warm-up requests (default + plain fallback when DSpark is on) |
| `EXL3_TP_ACCEPT_TIMEOUT` | 1800 s | how long rank 0 waits for rank 1 to finish loading |
| `EXL3_TP_SPEC_SAMPLED` | 1 | sampled requests draft too (exact speculative sampling); 0 = plain path for them (rank 0 only) |
| `EXL3_TP_ODL_AR2` | 1 (launchers) | decode-sized all-reduces over the OdinLink stream API (`~/exl3ds4/odl_ar2`, stream-ordered, no host sync); 0 = RCCL for everything |
| `EXL3_TP_ODL_AR2_PEER` / `_PORT` / `_MAX` | 10.0.2.2 / 29652 / 65536 | rank 1's address (it listens), rendezvous port, largest wire payload routed to odl_ar2 |
| `ODL_RCCL_DEVICES` | 0 (launchers) | `/dev/odl_tb5_N` devices offered to RCCL (patched net plugin); keeps RCCL on the box1-box2 link when another Thunderbolt host is plugged in |
| `EXL3_TP_SHARED_EXPERT_DEVICE` | unset | pin the whole-on-one-rank shared experts to TP rank N (`alt`: alternate by layer); unset = the allocator's choice (every one on the larger-budget rank) |
| `EXL3_DSV4_CAND_ROWS` | 1 | candidate-reader indexer layers score per row from fixed-width block lists (no host syncs); 0 = union path |
| `EXL3_ENGRAM_PREFETCH` | 1 | stage the Engram table rows on a worker thread at the start of each pass |
| `EXL3_ROCM_MOE_DECODE_RANGE` | 1 | TP expert-range shards use the RDNA MoE decode kernels; 0 = the cooperative kernels (patch `0012`) |
| `EXL3_TP_PREFIX_CACHE` | 1 | reuse cached prompt prefixes across requests (rank 0 only; rank 1 follows the frames) |
| `EXL3_TP_PREFIX_SLOTS` | 2 | recurrent-state checkpoints kept (LRU; 32 MiB of host RAM each, on both ranks: the last prompt and the last reply end) |
| `EXL3_TP_PREFIX_MIN` | 256 | shortest prefix worth a checkpoint, in tokens |
| `EXL3_TP_PREFIX_END` | 1 | also checkpoint the end of each reply (useful when clients echo replies verbatim, e.g. tool calls) |
| `EXL3_TP_PREFIX_LOG` | 0 | log one line per request: reused tokens and the checkpoint positions |
| `EXL3_TP_DRAFT_RING` | span/256 + 2 (10) | DSpark drafter main-kv ring in pages (rank 0); 0 = allocate the whole context (1.5 GiB at 512K) |
| `EXL3_TP_VISION` | 1 (auto) | rank 0 loads the vision tower + aligner; a load or memory-floor failure degrades to text-only (loud log) unless `EXL3_TP_VISION_STRICT=1` |
| `EXL3_TP_MAX_IMAGES` | 8 | per-request image limit (each image costs up to `max_image_tokens - 2` prompt rows) |
| `EXL3_TP_IMAGE_HTTP` | 1 | allow http(s) image URL fetches (data:/base64 always allowed); 0 refuses URLs |
| `EXL3_TP_IMAGE_MAX_BYTES` / `_MAX_PIXELS` / `_FETCH_TIMEOUT` | 32 MiB / 100 Mpx / 30 s | caps on fetched image bytes, decoded pixels (decompression bombs) and fetch timeout |
| `EXL3_TP_IMGMETA_TIMEOUT` | 120 s | how long rank 0 waits for rank 1's `imgmeta_ok` per image request |

`EXL3_TP_SPAN`, `EXL3_TP_CACHE_TOKENS`, `EXL3_TP_USE_PER_DEVICE` and `EXL3_TP_MEM_FLOOR` join
the DSpark/env variables the launchers forward to Box2 (ssh forwards nothing): the chunk plan
and the block-table sizing are derived from them on both ranks, so they must match. The same
holds for `EXL3_TP_ODL_AR2*`, `EXL3_TP_SHARED_EXPERT_DEVICE` (it changes the TP plan) and
`EXL3_DSV4_CAND_ROWS`, which `exl3ds4.sh` / `tp_serve_odl.sh` forward as well.

## First-version limitations (deliberate)

1. **One cached sequence.** The prefix cache follows one conversation at a time (see Prefix
   cache). Prompts are prefilled up to the full cache: prefill is chunked at `EXL3_TP_SPAN`,
   block tables span the cache, and prompts past `cache_tokens - 2` are a clean 400 before
   any rank dispatch.
2. **One in-flight request** (`sessions = 1`); others queue in the server.
3. **No rank-failure recovery.** If either rank fails mid-request, RCCL wedges; restart both
   (`pkill -f tp_serve.py; pkill -f tp_rank1.py`, then relaunch). `/tp/status` surfaces rank-1
   health between requests. The pre-dispatch bounds checks and the startup memory floor remove
   the two known ways a request could kill a rank mid-collective, but a genuine library fault
   still wedges the peer.
4. **Greedy by default** (`temperature 0`); `temperature/top_p > 0` sampling runs on rank 0 only
   (safe: sampling happens after the collectives). Non-DSpark MTP is not wired into the TP
   path; DSpark serves sampled requests through speculative sampling and composes with vision.
5. The HTTP default bind is `0.0.0.0:8080` (the deployment binds `0.0.0.0:1234`); no
   authentication — keep it on the LAN, or pass `--host 127.0.0.1`. Image URL fetches
   (`http(s)`) are operator-trust like the rest of the endpoint (no SSRF filtering;
   `EXL3_TP_IMAGE_HTTP=0` turns them off).
6. **Vision first-version limits:** image prompts bypass the prefix cache (no pixel-hash
   identity yet — a repeated image re-encodes and re-prefills); prompts must end on a text
   token; the tower runs only on rank 0 (a second concurrent image request would queue behind
   `sessions = 1`, as everything else).
