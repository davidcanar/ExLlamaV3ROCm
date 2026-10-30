# Development notes

## How the two ranks work together

- **Rank 0 (Box1)** runs the HTTP server (`tools/ds41/tp_serve.py`), renders
  prompts, samples tokens and drives the request. **Rank 1 (Box2)** runs
  `tools/ds41/tp_rank1.py` and mirrors every model call, driven by framed JSON
  over a TCP control channel on the Thunderbolt IP link (protocol 4: JSON
  frames, prefix-cache fields and the vision `imgmeta` + binary blob frames;
  protocol 5 at `EXL3_TP_SESSIONS >= 2` adds session page maps and spill/load
  frames). `tp_sessions.py` owns the four shared-pool conversation slots and
  prefix auto-matching; `tp_disk.py` owns each rank's local NVMe images.
- Model collectives use RCCL over OdinLink (`NCCL_NET=ODL_TB5`). Decode-sized
  all-reduces bypass RCCL through `odl_ar2` (stream-ordered, no host sync).
- The vision tower + aligner runs on rank 0 only (`tools/ds41/tp_vision.py` is
  its CPU-side half: safe loading, geometry, the dynamic-id allocator and the
  rank-0→rank-1 row codec). Rank 1 receives each image's tower output rows per
  request over the control channel, so image input adds no collectives.
- Lockstep rule: every stop decision happens on rank 0 after sampling and
  before the next frame, so both ranks always issue the same collectives.
  Details, the control protocol and the prefix-cache design:
  [`tools/ds41/README.md`](../tools/ds41/README.md).

## Tests (CPU only)

```bash
cd ~/exl3ds4/tools/ds41
DS41_REF_DIR=~/exl3ds4/model ../../.venv/bin/python -m pytest -q \
    test_ds41_chat.py test_ds41_parse.py test_ds41_serve.py test_tp_control.py \
    test_tp_chunked_prefill.py test_tp_prefix_cache.py test_serve_compat.py \
    test_tp_sessions.py test_tp_disk.py test_vendor_kyojin.py
PYTHONPATH=~/exl3ds4/src/exllamav3-dsv41f \
    ../../.venv/bin/python -m pytest -q test_tp_vision.py   # engine-level vision tests
# stage 1 + stage 2 in one process: the disk tier must not leak state across suites
../../.venv/bin/python -m pytest -q test_tp_sessions.py test_tp_disk.py
```

Last run: **606 passed, 6 skipped** (the skips need the real tokenizer; set
`DS41_REF_DIR`), plus **44 passed** in the dedicated combined sessions+disk run.
The TP tests run the real `Rank1Worker` over a loopback socket against stand-in
backends, and assert both ranks make identical model calls; the session tests
cover page routing, auto-matching, eviction and proto-5 guards; the disk tests
cover checksums, streaming, caps, spill/reload and every fallback; the vision
tests additionally run the real `TPEngine` pairs with a tower stand-in and (with
the exllamav3 tree on `PYTHONPATH`) against the real `MMEmbedding`/geometry code
— no GPU needed.

## GPU smoke tests (server stopped)

```bash
~/exl3ds4/scripts/run_ds41_odl.sh generate   # cached prefill + 8 greedy tokens, both nodes
~/exl3ds4/scripts/run_ds41_odl.sh smoke      # cache-less full-checkpoint pass
```

Logs go to `~/exl3ds4/logs/{generate,smoke}-odl-rank{0,1}.log`.
`scripts/tp_crosshost_smoke.py` runs a tiny synthetic DeepSeek-V4 checkpoint
across both nodes (start rank 1 on Box2, then rank 0 on Box1, with
`EXL3_TP_NODE_RANK`, `EXL3_TP_WORLD_SIZE=2`, `EXL3_TP_LOCAL_DEVICE=0` and a
shared `EXLLAMA_MASTER_ADDR/PORT`) and compares against a single-device load;
`tp_backend_smoke.py` exercises the backend alone, including uneven shards.

## Runtime knobs worth knowing

- `EXL3_TP_NODE_RANK` / `EXL3_TP_WORLD_SIZE` are mandatory for cross-host mode;
  `EXL3_TP_LOCAL_DEVICE` is the CUDA ordinal within each host.
- `EXL3_TP_NCCL_SYNC=stream` (launcher default) syncs only the launching stream
  after RCCL collectives; `1` restores full-device syncs.
- `NCCL_MAX_NCHANNELS=4` (launcher default): RCCL splits each collective per
  channel and every shard becomes an OdinLink stream message.
- `ODL_RCCL_DEVICES=0` keeps RCCL on the Box1–Box2 link when another
  Thunderbolt host is plugged into a box.
- Full-group gathers use RCCL `all_gather`, chunked at 64 MiB
  (`EXL3_TP_GATHER_CHUNK_BYTES`).
- `EXL3_TP_SESSIONS=4` (deployment default) keeps four cached conversations over
  one shared pool; `1` rolls back to the single-session engine and protocol 4.
  `EXL3_TP_SESSION_MIN_MATCH` tunes the auto-routing floor.
- `EXL3_TP_DISK_CACHE=1` spills whole LRU conversations to each rank's local
  `EXL3_TP_DISK_CACHE_DIR` (default `<checkout>/diskcache`), capped by
  `EXL3_TP_DISK_CACHE_GIB` (32) with a 300 s `EXL3_TP_DISK_TIMEOUT`. Images are
  one-shot; `0` keeps the pool eviction-only.
- Diagnostics: `EXL3_TP_TRACE_COLLECTIVES=1` (ordered collective log),
  `EXL3_TP_STREAM_HASH=N` (residual-stream hashes), `EXL3_TP_PREFIX_LOG=1`
  (one line per request with the reused tokens). `/tp/status` reports the pool,
  hot/cold conversations, counters and the local disk tier snapshot.

Settings that change the TP plan, the session geometry or the collective
sequence must match on both ranks; `exl3ds4.sh` forwards them to Box2 (ssh
forwards no environment). The proto-5 handshake refuses a rank 1 with a
different session count or pool page count.

## Rebuilding

```bash
source ~/exl3ds4/env.sh
bash ~/exl3ds4/scripts/build_exllamav3.sh     # after changing native ExLlamaV3 sources
python ~/exl3ds4/scripts/smoke_torch.py --require-ext
~/exl3ds4/odl_ar2/build.sh                    # odl_ar2, per box
```

`env.sh` handles the gfx1151 HSA preload and HIP header ordering; do not set
`HSA_OVERRIDE_GFX_VERSION`.

## Patches

- `patches/exllamav3/`: the `multi-node-rocm` branch as `git am` patches on PR
  #449 (`d45e8961`); see its README for the list.
- `patches/odinlink/`: plain diffs for `git apply` on OdinLink-Five `4534f58`
  (driver multi-frame framing, RCCL v7 plugin, device allow-list, RX copy and
  stale-tunnel fixes).

Profiling and benchmark tools are listed in [`PERFORMANCE.md`](PERFORMANCE.md#tools).
