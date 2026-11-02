# TP=3 runtime (three boxes)

How the three-node stack is put together. How to install and start it is in the
[README](../README.md#tp3-box1--box2--box3); measurements and the 2026-11-02 review are in
[tp3-review-2026-11-02.md](tp3-review-2026-11-02.md).

```bash
cd ~/exl3ds4
EXL3_TP_MODEL_DIR=$HOME/exl3ds4/model-3.5bpw ./tools/ds41/tp3_start.sh start   # stop | status | logs | probe
```

## Files (tools/ds41/)

| File | Role |
|---|---|
| `tp3_start.sh` | Controller (start/stop/status/logs/probe), run on Box1. Probes the cable map, starts the workers over ssh, gates rank 0 on their model load, starts rank 0. |
| `tp3_serve.sh`, `tp3_serve.py` | Rank 0 (Box1): `tp_serve.py` with the single rank-1 channel replaced by `MultiControlChannel` plus per-worker handshake verification. The HTTP layer, its arguments and `TPEngine` are reused verbatim. |
| `tp3_worker.sh`, `tp3_worker.py` | Ranks 1..N-1 (Box2 = 1, Box3 = 2): `tp_rank1.py` accepting any rank >= 1; `tp_engine.Rank1Worker` is rank-agnostic. |
| `tp3_channel.py` | `MultiControlChannel`: the `ControlChannel` surface (`send`/`recv`/`request`/`request_with_blobs`/`set_recv_timeout`/`close`) fanned out over N-1 workers. **Send to all, then receive from all**: a worker replies only after its model call finished, and that call contains collectives, so sending to worker 2 after worker 1 replied would deadlock both. Replies are compared modulo the `rank` field; divergence raises `WorkerDiverged`. |
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
- Off and not validated at TP=3: vision (`EXL3_TP_VISION=0`).
- RCCL has no retransmit over OdinLink: a frame dropped by a cable (RX CRC error, rare)
  during a prefill all-reduce hangs the server until it is restarted.
- Not measured yet: decode beyond 32K context, prefill at full context.

## Tools

| Tool | Use |
|---|---|
| `perf/tp3/run_prof3.sh` (+ `tp3_prof.py`, `summarize.py`) | TP=3 profiler: decode and prefill per op; probes the cable map itself |
| `perf/tp3/ar3_bench.sh` | RCCL vs `odl_ar3` all-reduce microbenchmark |
| `perf/tp3/rccl_stall.sh` | start-up collective repro (`ODL_DEBUG=1` keeps plugin traces of failing attempts) |
| `perf/tp3/http_check.py` | one DSpark and one plain request against the running server |
| `perf/tp3/session_check.py` | conversation cache: interleaved conversations, thinking on, NVMe spill and reload |
