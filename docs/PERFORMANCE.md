# Performance notes

Measurements from the two-box rig (2 × Strix Halo, 128 GB each, one Thunderbolt
link), October 2026. HTTP numbers come from `perf/bench_ctx.py` (120 generated
tokens, one run each); per-op numbers from the rank-0 profiler (`perf/tp_prof.py`).

## Current numbers

Decode, tok/s:

| prompt tokens | DSpark greedy | sampled T=0.6 / T=1.0 | plain |
|---:|---:|---:|---:|
| 847 | 34.8 | 34.7 / 34.5 | 25.4 |
| 25,669 | 36.4 | 31.5 / 30.4 | 24.5 |
| 62,822 | 35.6 | 35.0 / 33.2 | 24.3 |

Prefill runs at ~300 tok/s. With the prefix cache, the next turn of a 19.8K-token
conversation starts generating after 0.47 s instead of 65 s. A DSpark streaming
run on the multi-session build measured **45.74 tok/s**.

Where a DSpark round goes at 60K context (synchronized profile, which stretches
the 97 ms round to 131 ms): ~45 ms MoE (six rows), ~21 ms other linears, ~16 ms
all-reduces (mostly rank 0 waiting for the slower Box2 GPU), ~8 ms attention,
~7 ms Engram, 11–12 ms draft. The draft runs on rank 0 between verifies and does
not grow with context. Plain decoding costs ~40 ms/token at 2K and at 60K.

## How it got here

| step | what changed | 62.8K plain | 62.8K DSpark greedy | 62.8K sampled T=1.0 |
|---|---|---:|---:|---:|
| start | RCCL for every collective, sampled requests on the plain path | 6.6* | 19.8 | 6.6 |
| pass 1 | `odl_ar2`, sync-free decode, DSpark for sampled requests | 13.8 | 22.4 | 25.1 |
| pass 2 | RDNA MoE kernels on the TP ranks | 24.3 | 35.6 | 33.2 |

\* a long-running server with ~10 GiB swapped out; a fresh process measured 11.5.

**Pass 1** (patches `exllamav3/0008`–`0011`, `odl_ar2/`, `odinlink/0002`):

- Decode-sized all-reduces (≤ 64 KiB on the wire) go over the OdinLink stream
  API (`odl_ar2`, adapted from the vLLM stack) instead of RCCL: 34 µs vs 88 µs at
  10 KB, stream-ordered, no host sync, bf16 wire bit-identical to the RCCL path.
- No device syncs in the module walk: Engram hashes a host copy of the ids and
  stages its rows at the start of the pass; the candidate-reader indexer layers
  score each row's own blocks in place. Bit-identical outputs.
- Sampled requests use DSpark through exact speculative sampling (they used to
  bypass it, and the server default is temperature 1.0).

**Pass 2** (patch `exllamav3/0012`): the ~31 ms/token of MoE time came from the
wrong kernels. ExLlamaV3's RDNA MoE decode path only ran when every expert is
local, a TP rank holds an expert range, so both ranks used the cooperative
kernels: 2.9× slower per layer (`perf/moe_bench.py`: 455 vs 1317 µs at 1 row,
1328 vs 4181 µs at the 6-row verify). The RDNA kernels now take the range, and a
scratch buffer that sent 6-row verifies back to the slow path is fixed.
`moe_bench.py --check`: half-range shards sum to the full module within 1.4e-7,
the range path matches the cooperative kernels within 1e-3.

Not worth doing: shorter drafts for sampled requests. At the measured acceptance
(≈0.73 per draft at T=1.0), going from 5 to 3–4 drafts loses about as many
accepted tokens as the smaller verify saves.

## Prefix cache

The paged DSA pools are position-indexed and keep a finished request's tokens,
so both ranks only checkpoint the recurrent state (SWA ring, compressor windows,
Engram ids; 32 MiB of host RAM per rank) after each prefill and at the end of
each reply. A follow-up turn restores the longest valid checkpoint and prefills
only the new tokens. With `EXL3_TP_SESSIONS=4` each conversation keeps its own
checkpoint namespace and page mapping; auto-matching chooses the slot from the
prompt prefix, so concurrent client conversations no longer overwrite one
another. Design and protocol: `tools/ds41/README.md`.

- `perf/prefix_check.py` (HTTP, 19.8K-token conversation): turn 2 reused 19,786
  of 19,811 prompt tokens and began generating after 0.47 s (65 s fresh).
- `perf/prefix_logits.py` (first-token logits, server stopped): two fresh
  prefills are bit-identical. With the checkpoint on a 2048-token chunk
  boundary, restore + partial prefill is bit-identical to a fresh prefill, in
  plain and spec mode. Normally the checkpoint sits mid-chunk and the continued
  prefill starts its chunks there, which reorders float sums: max |Δlogit| 1.2,
  KL 0.005 nats, top-5 unchanged. Greedy replies can therefore differ from a
  cache-free run where two tokens are nearly tied.
- Costs: stash 1–12 ms, restore 2–6 ms.

## Multi-session and NVMe spill (2026-10-30)

Four cached conversations share the one 512K-token pool, but GPU work stays
serialized, so the per-request decode numbers above are unchanged. Routing is by
prompt prefix (no client id): every turn that still contains its previous prompt
returns to its own checkpoint namespace.

Validated live with DSpark on and the 512K pool:

- Four interleaved conversations each resumed with `cached_tokens=1195`; a
  divergent side conversation did not overwrite them.
- A fifth conversation spilled the LRU one to one image per rank, ~71.5 MB each,
  instead of evicting it.
- Returning to the spilled conversation reloaded both images (`cold_loads=1`)
  and resumed with `cached_tokens=1206`; the reply was identical to a fresh
  `cache_prompt: false` prefill of the same messages.
- A 4,820-token prompt resumed from its checkpoint instead of re-prefilling.
- The restore counters recorded no fallbacks (`spill_restore_fallbacks=0`), and
  the vision card still read **BLUE HERON 42**.

The spill writes one page at a time (~0.8 MiB at 512K) plus the pickled stashes,
so peak host RAM stays bounded; the 71.5 MB image is the per-rank bytes on disk,
not a RAM copy. Images are one-shot and are not reused after a restart.

## Vision

Image input (protocol 4/5, 2026-10-29) loads the tower + aligner on rank 0 only:
~0.7 GiB (12.7 → 12.0 GiB free of the 124 GiB GTT with DSpark and the 512K
cache resident) and 4.2 s at startup. Rank 1 loads nothing; each image request
ships the tower's output rows over the control channel (~0.3 MiB per image at
bf16, sha256-checked) and both ranks attach them as `indexed_embeddings`.

Live numbers (running server, DSpark on, greedy, one 800×600 PNG ≈ 326 image
tokens + chat template = 336 prompt tokens): **~1.4 s end to end** over HTTP —
tower pass + ship + 336-token prefill + 6 decoded tokens. The same request with
`"speculative": true` accepted 5 draft tokens and answered identically; the SSE
stream of it delivers the same text in chunks. `perf/` has no dedicated vision
benchmark yet; these numbers came from the validation run below.

Image requests bypass the prefix cache deliberately (`cached_tokens: 0` in
usage): each image gets a fresh dynamic-id range, so cross-request identity
needs pixel hashing — **not enabled in this first version**. Text caching is
unaffected: a text follow-up sent after image traffic still restored 1644 of
its prompt tokens.

Validated live on 2026-10-29 against the running server: `/health` and
`/tp/status` report `vision: true` (proto 4/5); a test card reading **BLUE HERON
42** was transcribed verbatim with plain decoding, with DSpark, and over SSE;
`data:image/png;base64,AAAA` answered 400 (`cannot decode image`) without
reaching either rank's model; a text request before, between and after the
image requests behaved exactly as without vision (prefix cache engaged:
1644 cached tokens on the follow-up turn).

## Memory

GPU memory is system RAM on these boxes (unified memory, ~124 GiB GTT), so every
GPU allocation competes with the OS.

| item | Box1 (rank 0) | Box2 (rank 1) |
|---|---:|---:|
| weights (load budgets with DSpark) | 95 GiB | 104 GiB |
| DSA KV pools at 512K tokens (fp16) | 1.6 GiB | 1.6 GiB |
| DSpark drafter main-kv (ring of 10 pages) | 7.5 MiB | — |
| vision tower + aligner (rank 0) | ~0.7 GiB | — |
| prefix checkpoints (4 conversations × 2 × 32 MiB) | 256 MiB | 256 MiB |
| RAM left for the OS | ~2.9 GiB (with the vision tower) | ~2 GiB |

The NVMe tier adds no resident GPU memory: a spilled conversation's pages and
stashes move to its rank's local images (up to `EXL3_TP_DISK_CACHE_GIB`, 32 GiB,
per rank) and come back only when that conversation returns.

The KV cache is small: V4.1-Flash keeps four pool-owning layers (2, 8, 14, 20;
the other layers read their owner's pool), three at compression rate 2 and one
at rate 1, 1,280 bytes per entry. It has to stay in memory: every decode step
scores the indexer keys of every cached position.

The DSpark drafter only ever reads the main-kv rows of the last 128 positions,
yet its cache used to be allocated for the whole context: 512K positions × 3
drafter layers × 1 KiB = 1.5 GiB on Box1. It is now a ring of 10 pages
(`EXL3_TP_DRAFT_RING`; one prefill chunk plus the window, 7.5 MiB), and prefix
checkpoints keep the drafter's window so a restore after the ring wrapped still
drafts from the right rows. A/B on Box1, 60 s after a restart: 2,160 MiB
available with the full cache, 3,662 MiB with the ring. Greedy acceptance is
unchanged (2.33 per round at 847 tokens, 2.75 at 25.7K), and a corrupt-and-restore
check on the real drafter (`perf/draft_ring_check.py`) gives bit-identical drafts.

The rank-0 process had ~8.5 GiB swapped out right after a restart, and the box
had ~0.9 GiB free. Keep other memory-hungry or CPU-heavy jobs off the boxes
while serving.

## Reproducibility

Prefill and the first generated token are bit-reproducible run to run. Later
decode steps are not always: a 19.8K-token DSpark request sent twice without the
cache diverged after ~140 characters (a 2.5K-token one repeated exactly).
Compare logits, not text, when validating kernels or caching.

## OdinLink RX pool hang (fixed)

During large prefills the `odl_tb5` driver's per-stream spare assembly buffers
(`rx_asm_pool`, 4 × 2 MiB) run dry routinely (`rx_asm pool empty` in the kernel
log). The fallback copy was an order-8 `kmalloc(GFP_ATOMIC)`, which fails on a
fragmented host. The message was dropped and both ranks hung in RCCL until
restart (seen twice on 2026-10-28). `patches/odinlink/0003` makes the copy
`kvmalloc(GFP_KERNEL)` (the RX callback runs in a workqueue, so it may sleep),
fixes the module exit path that left the DMA tunnel registered (the next load
then failed `enable_paths` with -ENOMEM until a reboot), and clears such a stale
tunnel at load. Stress-tested with `perf/run_stress.sh`: 1–64 MiB all-reduces,
exact results, with the pool disabled and with CPU load on both boxes.

## Tools

| tool | what it measures |
|---|---|
| `perf/bench_ctx.py` | the running server over HTTP: decode/prefill speed vs context |
| `perf/run_prof.sh` + `perf/tp_prof.py` | per-op decode profile (server stopped) |
| `perf/moe_bench.py` | one MoE layer on one GPU, sharded the way a rank holds it; `--check` |
| `perf/prefix_check.py` | multi-turn prefix reuse over HTTP |
| `PROF_DRIVER=prefix_logits.py perf/run_prof.sh pfx` | first-token logits, cached vs fresh (`PFX_ALIGN=1`: bit-exact variant) |
| `perf/run_bench.sh` | cross-box all-reduce latency, RCCL vs `odl_ar2` |
| `perf/run_stress.sh` | large RCCL all-reduces with exact checks (link robustness) |
| `PROF_DRIVER=draft_ring_check.py perf/run_prof.sh ring` | the drafter ring: draft before and after a corrupt-and-restore of its window |
