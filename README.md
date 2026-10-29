# ExLlamaV3ROCm — two-node Strix Halo ROCm stack serving DeepSeek-V4.1-Flash (EXL3)

Goal: serve `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` with
[ExLlamaV3](https://github.com/turboderp/ExLlamaV3) tensor parallelism on two
AMD Strix Halo (`gfx1151`) machines over a Thunderbolt/USB4 link, using
[OdinLink](https://github.com/Geramy/OdinLink-Five) as the RCCL network
transport — with an OpenAI-compatible server, MTP speculative decoding
("DSpark") and a 512K-token serving context.

Everything here is the working deployment from the two-box rig (Box1 + Box2),
packaged for reinstallation: controller (`exl3ds4.sh`), launchers, smoke/bench
scripts, the OpenAI server (`tools/ds41`), the ExLlamaV3 `multi-node-rocm`
patch series, the OdinLink driver/RCCL-plugin patch, pinned dependencies
(ROCm 10.1 / PyTorch 2.14 / Python 3.12) and a one-shot `setup.sh`.

**Not included by design:** model weights, Engram tables, `.venv`, logs, run
state, caches, backups or secrets (see `.gitignore`). Model data is downloaded
by `setup.sh` / `scripts/sync_model_to_box1.sh`.

## Current status

Working now (validated on the rig):

- ROCm 10.1 / PyTorch 2.14 userspace on both nodes, isolated from the distro
  ROCm installation.
- PR #449's native ExLlamaV3 extension built for `gfx1151`; real GPU and
  extension smoke tests pass.
- A new `multi-node-rocm` ExLlamaV3 runtime with global TP ranks decoupled
  from local CUDA ordinals.
- Symmetric two-host loading: both hosts exchange memory budgets, compute the
  same plan, and read their own local shards.
- Two-node RCCL all-reduce, broadcast, uneven gather, and forced-small-chunk
  gather tests.
- A real six-layer synthetic DeepSeek-V4 checkpoint loaded and executed across
  both nodes, with correct logits and clean teardown. Rank 0 also passes a
  single-host comparison.
- 54 targeted TP/config/cache tests pass.
- The full LibertAIDAI DS4.1 checkpoint loads and executes across both nodes,
  including cached prefill and greedy decode.
- OdinLink's RCCL v7 network plugin is built on both nodes and confirmed with
  `Using network ODL_TB5`; model tensor traffic uses OdinLink rather than
  Thunderbolt TCP sockets.
- A full cached generation run over OdinLink measured 2.02 s prefill and
  11.62 tok/s over eight decode tokens.
- An OpenAI-compatible server adapted from
  [Kyojin](https://github.com/Yamz-Labs/kyojin) passes 493 CPU tests and real
  two-node `/health`, `/v1/models`, `/tp/status`, non-stream chat, and SSE
  streaming chat requests. Real requests separate `reasoning_content` from
  `content` and run at roughly 12–17 generated tokens/s in short tests.
- DSpark/MTP speculative decoding works across both nodes, including HTTP
  `speculative`/`spec_k` controls and rank-lockstep draft/verify/accept
  operations. Predictable-output tests reached 21.40 tok/s vs 12.04 plain
  (1.78×) with exact output equality; a real HTTP request reached 16.22 tok/s
  vs 13.11 plain (1.24×).
- The OdinLink plugin uses 1 MiB multi-frame messages by default
  (`patches/odinlink`). Multi-frame + stream-scoped sync + four RCCL channels
  passed the full model and DSpark regressions. Legacy framing remains
  available with `ODL_RCCL_CHUNK=4000`.
- A persistent two-node OpenAI server managed by the `exl3ds4.sh` controller
  (start/stop/restart/status/logs), listening on **`0.0.0.0:1234`**, with
  **DSpark on by default** and a configured **512K context**
  (`EXL3_TP_SAFE_CTX=524288`, `EXL3_TP_CACHE_TOKENS=524288`; `/tp/status`
  reports ctx 524286).
- Long-context serving validated: the DSV4 per-worker recurrent-`wshift`
  readback is fixed (patch `0007`, source commit `0e259659`), and
  >750-token needle/counting tests pass with and without DSpark. The 512K
  allocation leaves >10 GiB free on each node.
- Decode profiling pass (patches `0008`–`0011`, `odl_ar2/`, see
  [Decode performance](#decode-performance-2026-10-28-profiling-pass)):
  decode all-reduces over the OdinLink stream API instead of RCCL, no device
  syncs in the module walk, and DSpark for sampled requests (they used plain
  decoding before). Over HTTP at a 62.8K-token context: **22.4 tok/s greedy,
  23.1 tok/s at T=0.6, 25.1 tok/s at T=1.0** (sampled requests were ~11.5
  tok/s), plain decoding 13.8 tok/s.

Not done yet: vision support and full 512K prefill performance measurement.
GPU zero-copy is not required; see the performance notes below.

## Repository layout

```
env.sh                 shared ROCm/HIP environment (source it; 'build' for compiles)
exl3ds4.sh             controller: start|stop|restart|status|logs for both boxes
setup.sh               one-shot installer (venv, wheels, clones, patches, builds, model)
scripts/               build script, rank launcher, smoke/bench harnesses, model sync
tools/ds41/            OpenAI-compatible server + CPU test suite (Kyojin-vendored layer)
patches/exllamav3/     11-commit 'multi-node-rocm' series on PR #449 base d45e8961
patches/odinlink/      local diffs of OdinLink-Five @ 4534f58 (multi-frame + RCCL device allow-list)
odl_ar2/               OdinLink stream all-reduce for decode collectives (HIP; build.sh per box)
perf/                  profiler (tp_prof.py + prof_hooks.py), link and HTTP benchmarks
requirements/          exact dependency pins (ROCm wheel index + PyPI freeze)
LICENSE / NOTICE       MIT + third-party attribution (Kyojin, ExLlamaV3, OdinLink, …)
```

## Nodes (the validated rig)

| Role | SSH through LAN | Thunderbolt address |
|---|---|---|
| Box1 / rank 0 | `davidcanar@192.168.0.102` | `10.0.2.1` |
| Box2 / rank 1 | `davidcanar@10.0.2.2` from Box1 | `10.0.2.2` |

Both boxes run Fedora with the same userspace; all paths (checkout, `.venv`,
model, Engram) must mirror across the two machines.

## Reinstall on a new pair of machines

On **both** boxes (identical paths, e.g. `~/exl3ds4`):

```bash
git clone https://github.com/davidcanar/ExLlamaV3ROCm.git ~/exl3ds4
cd ~/exl3ds4
./setup.sh                    # venv + ROCm wheels + patched clones + extension build
                              # + model/Engram download (~385 GiB) — use --no-model to skip
```

`setup.sh` is idempotent and safe to re-run. It needs: `git rsync cmake make
gcc curl sha256sum` (dnf/apt), and `uv` (auto-installed to `~/.local/bin` if
missing). Manual steps it deliberately does NOT do with root rights:

```bash
# kernel cmdline (both boxes, GRUB): amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856
# OdinLink kernel module (both boxes):
cmake --build ~/.cache/odinlink/build --target driver
sudo insmod ~/.cache/odinlink/build/driver/odl_tb5.ko
sudo cp ~/.cache/odinlink/driver/71-odl-tb5.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules
```

`/dev/odl_tb5_0` appears only once the Thunderbolt peer connects. Then, from
the rank-0 box only:

```bash
~/exl3ds4/exl3ds4.sh start     # starts rank 1 over ssh, then rank 0; polls /health
~/exl3ds4/exl3ds4.sh status
curl http://127.0.0.1:1234/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Name three prime numbers."}],"max_tokens":48,"temperature":0}'
```

### Parameterization (other machines)

The rig's values are the defaults; override per machine through the
environment (all optional):

| Variable | Default (example rig) | Meaning |
|---|---|---|
| `EXL3DS4_ROOT` | checkout directory | install root; venv/`src/`/`model/` live under it |
| `EXL3DS4_REMOTE` | `davidcanar@10.0.2.2` | Box2 ssh target used by the controller/orchestration scripts |
| `EXL3_MASTER_ADDR` | `10.0.2.1` | rank-0 Thunderbolt IP (exported as `EXLLAMA_MASTER_ADDR`) |
| `EXL3_TP_MODEL_DIR` | `$EXL3DS4_ROOT/model` | per-rank model directory (must mirror) |
| `EXL3_TP_ENGRAM_DIR` | `$EXL3DS4_ROOT/engram-src-1017` | per-rank Engram directory (must mirror) |
| `EXL3_ODL_SRC` | `~/.cache/odinlink` | OdinLink source clone (used by `setup.sh`) |
| `EXL3_ODL_BUILD` | `~/.cache/odinlink/build` | built RCCL plugin / libs the launchers load |

## Source pins

- ExLlamaV3 base: `anoane/exllamav3-anemone`, branch `dsv41f`, commit
  `d45e89611738e9134f72f5c9679cc9eae113bba8` (PR #449) — cloned by
  `setup.sh`, never vendored here.
- Working branch: `multi-node-rocm`, head `0e259659` — reproduced by
  `git am` of `patches/exllamav3` (cross-host runtime, upstream fixes, scoped
  sync modes, and the long-context recurrent-`wshift` readback fix).
- Kyojin reference: `Yamz-Labs/kyojin`, commit
  `e2f37564bbae01a6ea8ad4176dfd97db09c253aa` (release 1.8.2) — the vendored
  layer in `tools/ds41` is committed here; the clone is only needed to refresh
  it (`tools/ds41/vendor_kyojin.py`).
- OdinLink-Five: `Geramy/OdinLink-Five`, commit `4534f58` + the 5-file local
  diff in `patches/odinlink` (driver multi-frame framing + RCCL v7 plugin;
  driver is GPL-2.0, userspace/plugin MIT — see `NOTICE`).

## ROCm and build

Use the isolated Python 3.12 environment in `.venv`, not the distro ROCm
installation:

- `torch==2.14.0+rocm10.1.0a20260822`
- `rocm[devel,device-gfx1151]==10.1.0a20260822`
- GPU architecture: `gfx1151`

Both nodes need the kernel settings: `amd_iommu=off amdgpu.gttsize=126976
ttm.pages_limit=32505856` (about 124 GiB GTT).

```bash
source ~/exl3ds4/env.sh
python ~/exl3ds4/scripts/smoke_torch.py --require-ext
# Rebuild only when native sources change:
bash ~/exl3ds4/scripts/build_exllamav3.sh
```

`env.sh` handles the gfx1151 HSA preload and distro/SDK HIP header ordering.
Do not set `HSA_OVERRIDE_GFX_VERSION`.

## Run full DS4.1 over OdinLink

From Box1:

```bash
# Cached prefill + eight greedy decode tokens on both nodes:
~/exl3ds4/scripts/run_ds41_odl.sh generate

# Cache-less full-checkpoint smoke test:
~/exl3ds4/scripts/run_ds41_odl.sh smoke

# Optional custom rendezvous/control ports:
~/exl3ds4/scripts/run_ds41_odl.sh generate 29640 29641
```

Logs are written to `~/exl3ds4/logs/generate-odl-rank{0,1}.log` (or
`smoke-odl-*`). The launcher sets the ROCm/HSA environment, `NCCL_NET=ODL_TB5`,
the RCCL plugin paths, and a 65,536 file-descriptor limit. Rendezvous and the
rank-0 token-selection control channel use Thunderbolt IP; model collectives
use OdinLink.

The generation harness uses a 2K cache and a short prompt; it predates the
long-context work and remains a quick smoke test. The per-worker DSV4
`wshift` readback is fixed at source commit `0e259659` (patch `0007`), and
>750-token needle/counting tests now pass with and without DSpark (see the
server section below). Cache-slot reuse and vision are not yet validated.

## OpenAI-compatible server

`exl3ds4.sh` is the canonical controller. Run it from Box1; it starts rank 1
on Box2 over ssh, then rank 0 locally, and polls `/health` through the load
stages until ready:

```bash
~/exl3ds4/exl3ds4.sh start      # default action; idempotent (no-op if already ready or loading)
~/exl3ds4/exl3ds4.sh status     # link/ranks/health; exit 0 ready, 3 loading, 1 down
~/exl3ds4/exl3ds4.sh logs       # follow rank 0 (also: logs rank1 / logs all)
~/exl3ds4/exl3ds4.sh restart
~/exl3ds4/exl3ds4.sh stop       # graceful stop on BOTH boxes
```

Defaults: HTTP `0.0.0.0:1234`, TP master 29650, control 29651,
DSpark/MTP speculative decoding **on** (`EXL3_TP_DSPARK=1`), serving context
`EXL3_TP_SAFE_CTX=524288` with `EXL3_TP_CACHE_TOKENS=524288` cache slots per
rank. `/tp/status` reports ctx 524286. `stop` is cmdline-verified and scoped
to this stack only: foreign jobs and foreign port holders are reported,
never touched.

With DSpark on, sampled requests (`temperature > 0`) draft too: drafts are
accepted by exact speculative sampling, so the output distribution is that of
plain sampling (`EXL3_TP_SPEC_SAMPLED=0` sends sampled requests down the
plain path again).

Context and memory: 512K tokens is the configured and supported context.
>750-token needle/counting tests pass with and without DSpark, and the 512K
allocation leaves >10 GiB free on each node. Full 512K prefill performance is
unmeasured. The server still serves one in-flight request and does not use a
prefix cache.

`tools/ds41/tp_serve_odl.sh` remains available as an optional lower-level,
foreground launcher (defaults: HTTP 8080, TP master 29650, control 29651;
DSpark off unless `EXL3_TP_DSPARK=1`; stop with Ctrl-C and restart both ranks
if either fails). The context knobs above are honored by both launchers and
forwarded to rank 1 over ssh, but `exl3ds4.sh` is the supported way to run
the stack.

Endpoints include `GET /health`, `GET /v1/models`, `GET /tp/status`, and
`POST /v1/chat/completions` with `stream=true` or `stream=false`. The server
also inherits Kyojin's thinking/effort rendering and DSML tool-call parsing.
Details of the DSpark controls (`"speculative"`, `"spec_k"`), the control
protocol, the long-prompt/block-table fix and the full knob table are in
[`tools/ds41/README.md`](tools/ds41/README.md).

Use `http://192.168.0.102:1234` from other LAN clients. Logs are
`~/exl3ds4/logs/exl3ds4-rank0.log` (Box1) and
`~/exl3ds4/logs/exl3ds4-rank1.log` (Box2; `exl3ds4.sh logs rank1` follows it
remotely). `tools/ds41` also contains the CPU test suite.

## Two-node smoke test

Start rank 1 on Box2 (from Box1):

```bash
ssh davidcanar@10.0.2.2
source ~/exl3ds4/env.sh
EXL3_TP_NODE_RANK=1 EXL3_TP_WORLD_SIZE=2 EXL3_TP_LOCAL_DEVICE=0 \
EXL3_TP_DISPATCH_TIMEOUT=600 \
EXLLAMA_MASTER_ADDR=10.0.2.1 EXLLAMA_MASTER_PORT=29620 \
NCCL_SOCKET_IFNAME=thunderbolt0 NCCL_IB_DISABLE=1 \
python ~/exl3ds4/scripts/tp_crosshost_smoke.py
```

Start rank 0 on Box1 with the same master port:

```bash
source ~/exl3ds4/env.sh
EXL3_TP_NODE_RANK=0 EXL3_TP_WORLD_SIZE=2 EXL3_TP_LOCAL_DEVICE=0 \
EXL3_TP_DISPATCH_TIMEOUT=600 \
EXLLAMA_MASTER_ADDR=10.0.2.1 EXLLAMA_MASTER_PORT=29620 \
NCCL_SOCKET_IFNAME=thunderbolt0 NCCL_IB_DISABLE=1 \
python ~/exl3ds4/scripts/tp_crosshost_smoke.py
```

The harness creates an identical tiny checkpoint in `/tmp/exl3_tp_tiny_dsv4`,
runs a cache-less forward pass, unloads cleanly, then compares against a
single-device load on rank 0. `tp_backend_smoke.py` exercises the backend
independently, including uneven shards. Set `EXL3_TP_GATHER_CHUNK_BYTES=4096`
to force the chunked-gather test path.

Implementation notes:

- `EXL3_TP_NODE_RANK` and `EXL3_TP_WORLD_SIZE` are mandatory for cross-host
  mode. `EXL3_TP_LOCAL_DEVICE` is the CUDA ordinal within each host.
- Full-group gathers use RCCL `all_gather`, padded for uneven shards and
  capped at 64 MiB per chunk by default (`EXL3_TP_GATHER_CHUNK_BYTES`).
- Launchers default to `EXL3_TP_NCCL_SYNC=stream` and
  `NCCL_MAX_NCHANNELS=4`, validated against full-model and DSpark runs.
  `EXL3_TP_NCCL_SYNC=1` restores full-device synchronization; `0` remains
  experimental.
- Decode-sized all-reduces (wire payload <= 64 KiB) bypass RCCL through
  `odl_ar2` (`EXL3_TP_ODL_AR2=1` in the launchers): stream-ordered, so they
  need no host sync. `ODL_RCCL_DEVICES=0` keeps RCCL on the Box1–Box2 link
  when another Thunderbolt host is plugged into a box.
- `EXL3_TP_TRACE_COLLECTIVES=1` enables ordered collective diagnostics;
  `EXL3_TP_STREAM_HASH=N` enables residual-stream hashes.
- Basic cached prefill/decode passes. Cross-host CPU/disk cache tiers are
  explicitly rejected for now; vision and non-default output-rank topologies
  still need validation. Long-context recurrent-state handling is fixed and
  validated at source commit `0e259659`, and MTP/DSpark is validated in
  two-node serving.

## Model data

Download destinations per box (downloaded by `setup.sh` when `--model` is
active):

- Quant: `~/exl3ds4/model` (43 shards, about 196.3 GiB) from
  `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`
- Engram source: `~/exl3ds4/engram-src-1017` (two shards, about 189.2 GiB)
  from `deepseek-ai/DeepSeek-V4.1-Flash-1017`

The Engram tables are mandatory and must come from
`deepseek-ai/DeepSeek-V4.1-Flash-1017`. Do not substitute tables from the
older V4.1 release.

Expected Engram SHA-256 values (verified by `setup.sh`):

```text
76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5  model-00047-of-00048.safetensors
9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd  model-00048-of-00048.safetensors
```

Each rank has local weight shards and local Engram shards. To replicate an
existing copy to another box instead of re-downloading, use
`scripts/sync_model_to_box1.sh` (edit `EXL3DS4_REMOTE`) or plain rsync
without stale `.cache` fragments.

## Performance and zero-copy

GPU zero-copy is **not required** for correct or fast operation. Decode
collectives are small and the link is not bandwidth-saturated; host-to-device
copies are a minor cost next to collective latency and kernel work.

Validated improvements instead:

- 1 MiB multi-frame OdinLink messages (`ODL_RCCL_CHUNK=4000` restores legacy
  framing on both ranks).
- `EXL3_TP_NCCL_SYNC=stream` instead of full-device synchronization.
- `NCCL_MAX_NCHANNELS=4`; two channels was not better in the A/B tests.
- DSpark with the stock confidence threshold (`0.5`), which avoids low-value
  drafts. Full-window drafting can be slower on unpredictable text.

Measured examples: DSpark counting/structured output reached **21.40 tok/s vs
12.04 plain (1.78×)** with 100% acceptance. A real chat HTTP request reached
**16.22 tok/s vs 13.11 plain (1.24×)**. DSpark is not universally faster;
short or unpredictable outputs can lose to plain decoding.

## Decode performance (2026-10-28 profiling pass)

Tools: `perf/tp_prof.py` + `perf/prof_hooks.py` (rank-0 synchronized per-op
timing by monkeypatching, no library edits), launched with
`perf/run_prof.sh <tag>` while the server is stopped (results in
`perf/prof_*.jsonl`); `perf/run_bench.sh` (cross-box all-reduce latency) and
`perf/bench_ctx.py` (the running server over HTTP).

Findings:

- Decode did not degrade with context in a fresh process: plain decoding cost
  ~87 ms/token at 2K and at 60K. The 6.6 tok/s once measured at 63K came from a
  long-running server with ~10 GiB swapped out (both boxes keep only ~2 GiB of
  RAM available).
- Real traffic was mostly sampled (the server default is temperature 1.0), and
  sampled requests bypassed DSpark entirely.
- Per plain token: ~31 ms MoE kernels, ~25 ms in 80 all-reduces (~313 µs each
  in the model vs ~88 µs on an idle link: mostly rank 0 waiting on rank 1),
  ~13 ms attention linears, ~5 ms Engram CPU lookups, ~6 ms attention and
  indexer launches, ~9 ms mHC and norms.
- Box2's GPU sustains ~1585 MHz vs ~1748 MHz on Box1 at the same ~78 W under the
  2100 MHz OD cap (~14% less GEMM throughput), and with the 95/104 GiB budgets
  the allocator puts all 40 shared experts and 201 of 384 routed experts on
  rank 1.

Changes:

- `odl_ar2` (`odl_ar2/`, patch `0008`): adapted from the vLLM stack's OdinLink
  stream all-reduce, bf16 wire bit-identical to the RCCL path, per-slot round
  sizes. 10 KB: 34 µs vs 88 µs (RCCL); 60 KB: 97 µs vs 131 µs.
- No device syncs in the module walk (patches `0009`, `0010`): Engram hashes a
  host copy of the ids and stages its rows at the start of the pass
  (`EXL3_ENGRAM_PREFETCH`); the candidate-reader indexer layers score each
  row's own blocks in place (`EXL3_DSV4_CAND_ROWS`). Bit-identical outputs.
- DSpark for sampled requests (`tools/ds41/tp_engine.py`,
  `EXL3_TP_SPEC_SAMPLED`).
- `EXL3_TP_SHARED_EXPERT_DEVICE` (patch `0011`) pins the shared experts;
  measured neutral (plain −2 ms, DSpark +3 ms at 60K), so it is unset.
- `ODL_RCCL_DEVICES` (patch `odinlink/0002`).

Over HTTP (`perf/bench_ctx.py`, 120 generated tokens, one run each; tok/s):

| prompt tokens | DSpark greedy | plain | sampled T=0.6 / T=1.0 |
|---:|---|---|---|
| 847 | 19.3 → **21.4** | 12.0 → **14.1** | 12.0 (plain) → **19.5 / 21.2** |
| 25,669 | 21.2 → **23.8** | 11.7 → **13.8** | 11.7 (plain) → **22.1 / 20.8** |
| 62,822 | 19.8 → **22.4** | 6.6 → **13.8** | 6.6 (plain) → **23.1 / 25.1** |

Rollback switches: `EXL3_TP_ODL_AR2=0`, `EXL3_TP_SPEC_SAMPLED=0`,
`EXL3_DSV4_CAND_ROWS=0`, `EXL3_ENGRAM_PREFETCH=0`. Largest remaining costs:
the MoE kernels and the 12 ms rank-0-only draft per DSpark round.

## Tests (CPU only, no GPU/model needed except one tokenizer read)

```bash
cd ~/exl3ds4/tools/ds41
DS41_REF_DIR=~/exl3ds4/model ../../.venv/bin/python -m pytest -q \
    test_ds41_chat.py test_ds41_parse.py test_ds41_serve.py test_tp_control.py \
    test_tp_chunked_prefill.py test_serve_compat.py
```

Last run (whole `tools/ds41` suite on Box1): 509 passed, 6 skipped (the skips
need the real model tokenizer; set `DS41_REF_DIR` to remove them).

## Remaining work

1. MoE decode kernels (~31 ms/token, well below the GPU's 182 GB/s), the
   rank-0-only DSpark draft (~12 ms/round), per-temperature draft thresholds,
   and a prefix cache (every request re-prefills at ~310 tok/s).
2. Measure full 512K prefill performance and test cache-slot reuse at large
   contexts; the DSV4 recurrent-state readback fix and long-context
   validation are done (source commit `0e259659`).
3. Profile sustained decode/prefill against the existing GGUF runtime;
   evaluate offline weight packing, fixed-shape graphs, and symmetric clocks.
4. Validate vision, non-default output-rank topologies, and uneven memory
   splits. GPU DMA-buf zero-copy remains a later bulk-transfer experiment.

PR #449 and Kyojin are each single-host as supplied. Multi-node execution
here is new integration work, not an upstream configuration option.

## Licenses and attribution

This repository is MIT (see `LICENSE`); third-party components keep their own
licenses — Kyojin (MIT, vendored in `tools/ds41` with per-file headers and
`tools/ds41/LICENSE.kyojin`), ExLlamaV3 (MIT, fetched + patched), OdinLink-Five
(kernel driver GPL-2.0, userspace/RCCL plugin MIT, fetched + patched), and the
DeepSeek-AI reference encoding used by the golden tests. Model weights are not
redistributed. Full details in [`NOTICE`](NOTICE).
