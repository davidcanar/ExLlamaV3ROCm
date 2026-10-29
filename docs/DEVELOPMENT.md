# Development notes

## How the two ranks work together

- **Rank 0 (Box1)** runs the HTTP server (`tools/ds41/tp_serve.py`), renders
  prompts, samples tokens and drives the request. **Rank 1 (Box2)** runs
  `tools/ds41/tp_rank1.py` and mirrors every model call, driven by framed JSON
  over a TCP control channel on the Thunderbolt IP link.
- Model collectives use RCCL over OdinLink (`NCCL_NET=ODL_TB5`). Decode-sized
  all-reduces bypass RCCL through `odl_ar2` (stream-ordered, no host sync).
- Lockstep rule: every stop decision happens on rank 0 after sampling and
  before the next frame, so both ranks always issue the same collectives.
  Details, the control protocol and the prefix-cache design:
  [`tools/ds41/README.md`](../tools/ds41/README.md).

## Tests (CPU only)

```bash
cd ~/exl3ds4/tools/ds41
DS41_REF_DIR=~/exl3ds4/model ../../.venv/bin/python -m pytest -q \
    test_ds41_chat.py test_ds41_parse.py test_ds41_serve.py test_tp_control.py \
    test_tp_chunked_prefill.py test_tp_prefix_cache.py test_serve_compat.py
```

Last run: 525 passed, 6 skipped (the skips need the real tokenizer; set
`DS41_REF_DIR`). The TP tests run the real `Rank1Worker` over a loopback socket
against stand-in backends, and assert both ranks make identical model calls.

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
- Diagnostics: `EXL3_TP_TRACE_COLLECTIVES=1` (ordered collective log),
  `EXL3_TP_STREAM_HASH=N` (residual-stream hashes), `EXL3_TP_PREFIX_LOG=1`
  (one line per request with the reused tokens).

Settings that change the TP plan or the collective sequence must match on both
ranks; `exl3ds4.sh` forwards them to Box2 (ssh forwards no environment).

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
