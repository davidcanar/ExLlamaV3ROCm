# TP=3 runtime (tp3_*) — state and the current blocker

The 3-node control plane is built and verified; the model kernels are not yet
world-size-3 capable. This note records both, so the next step is unambiguous.

## What exists (tools/ds41/, the 2-node stack untouched)

| File | Role |
|---|---|
| `tp3_channel.py` | `MultiControlChannel`: same surface as `ControlChannel` (`send`/`recv`/`request`/`request_with_blobs`/`set_recv_timeout`/`close`) fanned out over N-1 workers. **Send-to-all then recv-from-all** — a worker reply only arrives after its model call finished, and that call contains RCCL collectives, so sending to worker 2 after worker 1 replied would deadlock both. Replies are compared modulo the `rank` field; divergence raises `WorkerDiverged`. |
| `tp3_serve.py` | `tp_serve.py` with the single rank-1 channel replaced by the multi-channel plus per-worker handshake verification. HTTP layer, args and `TPEngine` are reused verbatim. |
| `tp3_worker.py` | `tp_rank1.py` accepting any rank >= 1 (Box2 = rank 1, Box3 = rank 2). `tp_engine.Rank1Worker` is already rank-agnostic. |
| `tp3_worker.sh` | Per-rank worker env: world 3, LAN bootstrap/control (the 10.0.2.x mesh is point-to-point so one master address is not reachable from every box), both OdinLink cables visible with `ODL_RCCL_PEERS` per-peer routing, `EXL3_TP_ODL_AR2=0` (the odl_ar2 decode path is 2-party). |
| `tp3_serve.sh` | Rank-0 env: HTTP head node, 2K context/first light. |
| `tp3_start.sh` | Orchestrator (start/stop/status/logs) with a **load gate**: rank 0 must not start until every worker logged its model load, because rank 0's load performs the world-3 NCCL warmup that the workers block in. |

## Verified working (fabric + control plane)

- world-3 NCCL communicator + warmup across all three boxes over OdinLink
  (`NCCL init: world_size 3` / `Finished NCCL warmup`).
- Both workers connect, handshake and are verified by rank 0
  (`tp3-serve: 2 worker(s) connected and verified`).
- Symmetric 3-way load: budgets `[95, 100, 100]` GiB, each rank ~50 GiB free
  after loading, engram overrides applied on every rank.

## RESOLVED: power-of-two per-rank shards (option 2b-uneven)

The blocker was Triton's power-of-2 requirement on per-rank head partitions
(`tl.arange(0, HP)`); at world 3 a memory-ratio split of the 8 attention
o-groups gives 2/3/3 groups = 16/24/24 heads and 24 is not a power of two.

Fix (option **2b-uneven**): `patches/exllamav3/0013-TP3-pow2-attention-group-split.patch`
adds `pow2_units` to `TPAllocation`; the allocator then snaps the attention
group split to the most-even power-of-two distribution — **4/2/2 groups =
heads 32/16/16** (which rank gets 32 follows remaining memory; experts
rebalance automatically in the same allocation pass). Zero padded heads, zero
wasted attention FLOPs, and a **no-op at world 2** (8 groups over 2 ranks snap
to 4/4, exactly the ratio split).

Verified on the 3-box mesh, 2.9bpw, 2K ctx, DSpark off:

- 3-rank load + NCCL world-3 warmup + both-worker handshake: clean
- first generation through the HTTP API: coherent; decode ~19 tok/s
- **greedy agreement with the 2-node baseline**: 413/450 chars identical on a
  two-sentence probe (divergence is final-phrase phrasing — float reduction
  order, not a math bug); identical stop point (115 tokens) on the primes
  probe
- decode speed vs TP=2: 19.3 vs 23.3 tok/s at trivial context — the gap is
  the missing `odl_ar2` fast path (2-party only; TP=3 routes decode all-reduces
  through 3-way RCCL) and should narrow with context length as expert sharding
  pays off (128 vs 192 experts per rank)

Remaining TP=3 gaps: no DSpark (spec decoding) validation, no multi-session /
disk tier / vision, 2K ctx only, and `odl_ar2` has no 3-rank story yet.

## Publishing / operational notes

- The 2-node stack (`tp_serve.py`, `tp_rank1.py`, `exl3ds4.sh`) is untouched
  and remains the supported path.
- Per-boot OdinLink device mapping must be re-derived after any `odl_tb5`
  reload (`odinlink-multilink/docs/RUNBOOK.md`).
- 3.5bpw checkpoint (~237 GiB) is downloading on Box2; Box1 was freed by moving
  the 178 GiB GLM checkpoint to Box2 (verified) and deleting the Box1 copy.
