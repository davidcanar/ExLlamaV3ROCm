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
conversation starts generating after 0.47 s instead of 65 s.

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
only the new tokens. Design and protocol: `tools/ds41/README.md`.

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

## Memory

GPU memory is system RAM on these boxes (unified memory, ~124 GiB GTT), so every
GPU allocation competes with the OS.

| item | Box1 (rank 0) | Box2 (rank 1) |
|---|---:|---:|
| weights (load budgets with DSpark) | 95 GiB | 104 GiB |
| DSA KV pools at 512K tokens (fp16) | 1.6 GiB | 1.6 GiB |
| DSpark drafter main-kv at 512K tokens | 1.5 GiB | — |
| prefix checkpoints (2 × 32 MiB) | 64 MiB | 64 MiB |
| RAM left for the OS | ~1.8 GiB | ~2 GiB |

The KV cache is small: V4.1-Flash keeps four pool-owning layers (2, 8, 14, 20;
the other layers read their owner's pool), three at compression rate 2 and one
at rate 1, 1,280 bytes per entry. It has to stay in memory: every decode step
scores the indexer keys of every cached position. The drafter's main-kv is
allocated for the full context, but the drafter only ever reads the last 128
positions.

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
