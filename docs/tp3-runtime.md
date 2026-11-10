# TP=3 runtime (three boxes)

How the three-node stack is put together. How to install and start it is in the
[README](../README.md#tp3-box1--box2--box3); measurements and the 2026-11-02 review are in
[tp3-review-2026-11-02.md](tp3-review-2026-11-02.md).

```bash
cd ~/exl3ds4
EXL3_TP_MODEL_DIR=$HOME/exl3ds4/model-3.5bpw ./tools/ds41/tp3_start.sh start   # stop | restart | status | logs | probe | watch | unwatch
```

## Files (tools/ds41/)

| File | Role |
|---|---|
| `tp3_start.sh` | Controller (start/stop/restart/status/logs/probe/watch/unwatch), run on Box1. Probes the cable map, starts the workers over ssh, gates rank 0 on their model load, starts rank 0, and leaves the [watchdog](#watchdog) running. |
| `tp3_serve.sh`, `tp3_serve.py` | Rank 0 (Box1): `tp_serve.py` with the single rank-1 channel replaced by `MultiControlChannel` plus per-worker handshake verification. The HTTP layer, its arguments and `TPEngine` are reused verbatim. |
| `tp3_worker.sh`, `tp3_worker.py` | Ranks 1..N-1 (Box2 = 1, Box3 = 2): `tp_rank1.py` accepting any rank >= 1; `tp_engine.Rank1Worker` is rank-agnostic. |
| `tp3_channel.py` | `MultiControlChannel`: the `ControlChannel` surface (`send`/`recv`/`request`/`request_with_blobs`/`set_recv_timeout`/`close`) fanned out over N-1 workers. **Send to all, then receive from all**: a worker replies only after its model call finished, and that call contains collectives, so sending to worker 2 after worker 1 replied would deadlock both. Replies must agree except in per-worker fields (`rank`, the size of each worker's own spill image `bytes`, the error text `reason`/`message`); any other difference raises `WorkerDiverged`. |
| `odl_peer_probe.py` | Cable map: pings on every `/dev/odl_tb5_N` and prints which box answered on which device, as `ODL_RCCL_PEERS`. Both controllers (`tp3_start.sh`, `exl3ds4.sh`) run it on every start. |

## What a start does

1. **Cable probe.** `odl_peer_probe.py` runs on all three boxes at once and each rank gets
   its own `ODL_RCCL_PEERS` (peer identity -> local device). Device indices follow
   XDomain probe order and change across reboots and `odl_tb5` reloads. A stale map
   makes the RCCL plugin open connections on the cable to the *other* peer, and the
   world-3 warmup or the first large collective hangs. `tp3_start.sh probe` prints the
   map; `EXL3_TP3_NO_PROBE=1` skips it.
2. **Workers.** Ranks 1 and 2 start over ssh and load their shards, then block in NCCL
   init.
3. **Load gate.** Rank 0 starts only after both workers logged their model load: rank 0's
   own load runs the world-3 NCCL warmup, which needs every worker already inside NCCL
   init.
4. **Rank 0** loads, warms up, connects to both workers over the control channel and
   verifies their handshakes (control protocol, vocabulary — i.e. the same model —
   and DSpark state must match; with multi-session also the session count and pool
   geometry), then `/health` turns `ok`.

The NCCL bootstrap and the control channel ride the LAN (`EXLLAMA_MASTER_ADDR`, Box1's
LAN IP): every `10.0.2.x` Thunderbolt link is point-to-point, so no single address on the
mesh reaches every box. All tensor traffic rides OdinLink.

## Watchdog

`tp3_start.sh start` leaves `tp3_start.sh _watchdog` running on Box1 (pid in
`logs/tp3-watchdog.pid`, events in `logs/tp3-watchdog.log`; `EXL3_TP3_WATCHDOG=0` starts
without it, `watch` attaches one to a stack that is already up, `unwatch` removes it).
Every `EXL3_TP3_WATCH_INTERVAL` seconds (15) it asks each box about its rank
(`tp3_start.sh _state`, over ssh for the workers): is the pid in `logs/tp3-rankN.pid` still
that rank's python, and has a worker logged `failed; restart ALL ranks` since the start it
watches (`logs/tp3-watchdog.offsets` holds the log sizes at that start)? A worker that hits a
protocol error prints that line and then waits for its GPU child, which may sit in a
collective until the NCCL timeout; rank 0 cannot resynchronize after it either way. An
unreachable box proves nothing and is skipped. When a rank is down or failed the watchdog
stops every rank (SIGINT to each rank's process group, SIGKILL 5 s later) and runs the start
again with the same settings: the environment of the `start` it came from, or, for `watch`
and `restart`, the running rank 0's (read from `/proc/<pid>/environ`), so a restart cannot
silently fall back to the default model. A restart that is not serving within
`EXL3_TP3_WATCH_LOAD_TIMEOUT` seconds (900) counts as a failure; consecutive failures wait
30 s, 60 s, ... up to 15 minutes, and 30 minutes up resets that.

The watchdog never judges a rank by HTTP, so a slow answer during a long prefill can never
trigger a restart. A hang inside a collective (a frame lost by RCCL, below) ends when
NCCL's 10-minute timeout takes the ranks down; the watchdog restarts them then. It cannot
fix the fabric: if a box rebooted or the cable probe stays incomplete, it keeps retrying
with backoff and logs each attempt. It does not survive a reboot of Box1, and `stop` (which
removes `logs/tp3.want`) is the only clean way to take TP=3 down.

Every rank starts under `setsid`, so its python, the spawned GPU process and the resource
tracker share a process group (`logs/tp3-rankN.pgid`). A stop signals the group, the pid and
its children, then any rank process by name (as the launcher always did), so the GPU process
goes down too, even when it is stuck in a collective after its parent exited. The liveness
checks match only this install tree's python running the rank script, never an editor with
the file open.

## Model split at world size 3

- **Attention: power-of-two group split** (`patches/exllamav3/0013`). Triton's attention
  kernels need a power-of-two head partition per rank (`tl.arange(0, HP)`), and a memory-
  ratio split of the 8 o-groups gives 2/3/3 groups (24 heads is not a power of two). The
  allocator snaps the split to the most even power-of-two distribution, **4/2/2 groups =
  32/16/16 heads**. Which rank gets 32 follows remaining memory, and the experts
  rebalance in the same allocation pass. No padded heads, and a no-op at world 2 (4/4).
- **Experts** are split 128 per rank; the shared expert lives whole on one rank.
- Budgets default to 100/100/100 GiB (`EXL3_TP_BUDGETS`).

## Collectives

- **Decode all-reduces (up to 64 KB): `odl_ar3`** (`odl_ar3/`, `patches/exllamav3/0014`,
  `model_tp_odl.OdlAllReduceMesh`). Each rank sends its partial straight to both peers
  over their own cables and sums all three in fixed rank order: one hop, both cable
  directions busy, stream-ordered (no host sync), bit-identical on every rank. A message
  carries its round number; a gap or a timeout triggers a NACK and a resend from the send
  slot, so a lost frame costs a retransmit instead of a hang. The setup is collective:
  if any rank has `EXL3_TP_ODL_AR3=0` (or cannot load the library), every rank stays on
  RCCL. 10 KB: 28 us, against 140 us for three-rank RCCL plus a host sync.
- **Prefill and other large collectives: RCCL** over the OdinLink net plugin
  (`patches/odinlink/0006`-`0009`, `0012`), routed per peer by `ODL_RCCL_PEERS`.
- `odl_ar2`, the two-party decode path of TP=2, is off at TP=3 (`EXL3_TP_ODL_AR2=0`).

## Fabric

A triangle of USB4 cables with the multi-cable fabric on every box (patched `thunderbolt`
core + `odl_tb5`, `odinlink-multilink/`, loaded at boot by `odinlink.service`);
`/usr/local/bin/odl-state` shows two `ready` devices per box. Bring-up, kernel updates and
troubleshooting: [odinlink-multilink/docs/RUNBOOK.md](../odinlink-multilink/docs/RUNBOOK.md).

Fixed on the way to a stable start (2026-11-02):

| Symptom | Cause | Fix |
|---|---|---|
| world-3 warmup hangs after a driver reload | stale `ODL_RCCL_PEERS` cross-wires the plugin's connections | the cable probe on every start |
| about 1 start in 6 hangs in the first prefill-sized collective | the plugin's connector registry was keyed by stream id only; ids are per cable and collide | `patches/odinlink/0012` |
| on 7.2 kernels: links READY, data flows one way only, `can_send=0` in dmesg | `rx_repost()` raced with itself and drained the frame pool | `patches/odinlink/0013` |

`perf/tp3/rccl_stall.sh N` replays the start-up collective sequence on bare ranks (about
two seconds per attempt) and is the quickest check after any driver or plugin change.

## Status

- Validated: DSpark, 512K cache, 3.5 and 2.9 bpw, and the TP=2 conversation cache
  (`EXL3_TP_SESSIONS=4`, control protocol 5, per-rank NVMe spill): `perf/tp3/session_check.py`
  resumes a 10K-token conversation from cache after a different one ran, keeps a thinking-on
  conversation whose client drops the reasoning, and reloads a spilled conversation from
  NVMe on all three ranks (`cold_loads=1`, no fallbacks). Greedy plain output is
  byte-identical to the RCCL-only path. Until 2026-11-02 TP=3 ran the single-conversation
  engine, where any other conversation (a client's title request, a test) overwrote the
  cached one and its next turn re-prefilled from scratch.
- Interleaved requests (`EXL3_TP_CONCURRENCY=2..4`, control protocol 6, default 1 = off):
  up to C requests take turns step by step instead of queueing (no throughput gain; no
  head-of-line blocking). `tp3_start.sh` forwards the knob; every rank must use the same value
  (it sizes each rank's live cache slots) and the handshake refuses a mismatch. See
  `tools/ds41/README.md#interleaved-requests-2026-11-10`. CPU-validated
  (`test_tp_interleave.py`); not yet run on the rig.
- Vision (rank 0 loads the tower, the workers get each image's rows): on since
  2026-11-02; `perf/tp3/vision_check.py` reads the BLUE HERON 42 card exactly with DSpark
  and plain decoding, and a broken image is a clean 400.
- Context: 1M tokens since 2026-11-02 (4096-page pool, +1.6 GiB per rank over 512K; Box1
  keeps ~15 GiB free with vision). The morning's 1M attempt hung in the first prefill
  all-reduce, which was the plugin's connector registry (`patches/odinlink/0012`); no prompt
  longer than 512K has been run since.
- RCCL has no retransmit over OdinLink: a frame dropped by a cable (RX CRC error, rare)
  during a prefill all-reduce hangs the server until NCCL's 10-minute timeout takes the
  ranks down; the watchdog then restarts them.
- 2026-11-02 17:31: the stack went down after a spill. Both workers wrote their images, but
  the image trailer carries a timestamp whose text length varies, so their sizes differed by
  a byte (839100 / 839101). Rank 0 compared the `spill_ok` replies including that size, read
  a divergence, and evicted a session the workers had already spilled; both died on `evict
  of unknown session` and rank 0 waited in its next prefill all-reduce until the NCCL
  timeout. Fixed in `tp3_channel.py` (per-worker fields are not compared);
  `tools/ds41/test_tp3_channel.py` reproduces it with the real engine and two real workers.
- Not measured yet: decode beyond 32K context, prefill at full context.

## Tools

| Tool | Use |
|---|---|
| `perf/tp3/run_prof3.sh` (+ `tp3_prof.py`, `summarize.py`) | TP=3 profiler: decode and prefill per op; probes the cable map itself |
| `perf/tp3/ar3_bench.sh` | RCCL vs `odl_ar3` all-reduce microbenchmark |
| `perf/tp3/rccl_stall.sh` | start-up collective repro (`ODL_DEBUG=1` keeps plugin traces of failing attempts) |
| `perf/tp3/http_check.py` | one DSpark and one plain request against the running server |
| `perf/tp3/session_check.py` | conversation cache: interleaved conversations, thinking on, NVMe spill and reload |
| `perf/tp3/vision_check.py` | image input: a known card with DSpark and plain, a broken image, a text request |
