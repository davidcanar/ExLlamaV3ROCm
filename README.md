# ExLlamaV3ROCm

Serve **DeepSeek-V4.1-Flash** on **two or three AMD Strix Halo machines** joined by
Thunderbolt/USB4 cables. [ExLlamaV3](https://github.com/turboderp/ExLlamaV3) splits the
model across the boxes (tensor parallelism), [OdinLink](https://github.com/Geramy/OdinLink-Five)
carries the traffic over the cables, and an OpenAI-compatible server exposes it with MTP
speculative decoding ("DSpark"), image input, cached conversations and a 512K-token (TP=2)
or 1M-token (TP=3) context.

| | TP=2 | TP=3 |
|---|---|---|
| boxes, cables | 2, one cable | 3, a triangle of three cables |
| controller (run on Box1) | `exl3ds4.sh` | `tools/ds41/tp3_start.sh` |
| quant | [2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (196 GiB) | [3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) (240 GiB), or the 2.9 bpw |
| decode all-reduce | `odl_ar2` (two parties, one cable) | `odl_ar3` (one hop to each peer, over its own cable) |
| context | 512K tokens | 1M tokens (the model's maximum) |
| server | DSpark, four cached conversations in one shared pool, per-rank NVMe spill, image input | the same |

This repository is the working deployment, packaged for reinstallation: an installer,
the two controllers, the server, and the patches it needs on top of ExLlamaV3, OdinLink
and the Linux `thunderbolt` driver. Model weights are downloaded, not stored here.

## Results

### TP=2 (2.9 bpw)

Decode speed over HTTP (tok/s, 120 generated tokens):

| prompt tokens | DSpark, greedy | DSpark, sampled T=1.0 | plain decoding |
|---:|---:|---:|---:|
| 847 | 34.8 | 34.5 | 25.4 |
| 25,669 | 36.4 | 30.4 | 24.5 |
| 62,822 | 35.6 | 33.2 | 24.3 |

Prompt processing runs at ~300 tok/s. Follow-up turns reuse the cached prompt:
the next turn of a 19.8K-token conversation starts answering after **0.47 s**
instead of 65 s. Details, history and memory budget: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

The multi-session build was validated live with DSpark on and a 512K cache:
four interleaved conversations each resumed with `cached_tokens=1195`; a fifth
conversation spilled the LRU one to two ~71.5 MB per-rank NVMe images, and that
conversation's return reloaded them (`cold_loads=1`) with `cached_tokens=1206`
instead of re-prefilling; the restored reply matched a fresh prefill exactly.
A 4,820-token prompt resumed from its checkpoint, the vision card still read
**BLUE HERON 42**, a DSpark streaming run measured **45.74 tok/s**, and the
restore counters recorded no fallbacks. CPU validation: **606 passed, 6 skipped**,
plus **44 passed** in the dedicated combined sessions+disk run.

### TP=3 (3.5 bpw)

Measured 2026-11-02 on the three-box rig, 512K cache, greedy:

| | plain decoding | DSpark |
|---|---:|---:|
| 2K context, profiler (`perf/tp3/run_prof3.sh`) | 20.5–21.4 tok/s (46.8 ms/token) | 30.2 tok/s |
| HTTP, short prompt, 400 tokens (`perf/tp3/http_check.py`) | 22.8–24.2 tok/s | 27.0–29.4 tok/s |

Prompt processing runs at ~290–410 tok/s (2K–32K context). The decode all-reduces
run over `odl_ar3`, a one-hop all-reduce on the OdinLink mesh (28 µs per 10 KB
all-reduce, against 140 µs for three-rank RCCL), which took plain decode from 17.2
to 21 tok/s and DSpark from 23.0 to 30.2. Greedy plain output is byte-identical to
the RCCL path. The conversation cache works as on TP=2 (`perf/tp3/session_check.py`):
a 10K-token conversation came back from cache after a different conversation had run
(10,058 of 10,075 prompt tokens reused), and after it was spilled to NVMe it reloaded
in 0.8 s instead of a 29 s prefill. Details: [docs/tp3-runtime.md](docs/tp3-runtime.md)
and [docs/tp3-review-2026-11-02.md](docs/tp3-review-2026-11-02.md).

## What you need

- Two (TP=2) or three (TP=3) Strix Halo (`gfx1151`) machines with 128 GB each,
  Fedora (validated on Fedora 44 with kernel 7.2.14 on all three boxes; TP=2 also
  ran on 7.1), the same user and paths on every box.
- Cables: one Thunderbolt/USB4 cable for TP=2; for TP=3 one cable between every
  pair of boxes, so each box uses two USB4 ports.
- Disk per box: the quant (196 GiB for 2.9 bpw, 240 GiB for 3.5 bpw) plus 189 GiB of
  Engram tables, plus up to 32 GiB for the optional NVMe conversation-spill tier.
- Root once per box, for the kernel command line and the kernel modules.

The validated rig, whose addresses are the defaults:

| | LAN ssh | Thunderbolt IP | role |
|---|---|---|---|
| Box1 | `davidcanar@192.168.0.102` | `10.0.2.1` | rank 0, runs the HTTP server |
| Box2 | `davidcanar@192.168.0.103` (TP=2 reaches it as `10.0.2.2`) | `10.0.2.2` | rank 1 |
| Box3 | `davidcanar@192.168.0.105` | `10.0.2.3` | rank 2 (TP=3 only) |

All tensor traffic rides OdinLink. TP=2 reaches Box2 and runs its control channel
over IP on the Thunderbolt cable; TP=3 uses the LAN for ssh, the NCCL bootstrap and
the control channel, because each `10.0.2.x` link is point-to-point and no single
address reaches every box.

## Install

On **every** box, at the same path:

```bash
git clone https://github.com/davidcanar/ExLlamaV3ROCm.git ~/exl3ds4
cd ~/exl3ds4
./setup.sh            # venv, ROCm 10.1 + PyTorch 2.14 wheels, patched ExLlamaV3 and OdinLink,
                      # native builds (incl. odl_ar3), 2.9 bpw model + Engram download (--no-model skips it)

# TP=3 with the 3.5 bpw quant (optional; TP=3 also runs the 2.9 bpw one):
~/exl3ds4/.venv/bin/hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw \
    --local-dir ~/exl3ds4/model-3.5bpw
```

`setup.sh` is idempotent. It needs `git rsync cmake make gcc curl sha256sum`
and installs `uv` if missing. Then, as root on every box:

1. Kernel command line (GRUB), then reboot:
   `amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856`
2. The OdinLink fabric: a patched `thunderbolt` core (it tells the driver which
   peer an XDomain packet came from, so one box can serve several cables), the
   `odl_tb5` driver, and `odinlink.service`, which loads it at boot. Both modules
   are built for the running kernel (`kernel-devel` required):

   ```bash
   cd ~/exl3ds4
   odinlink-multilink/build.sh
   sudo odinlink-multilink/install.sh
   printf 'omit_drivers+=" thunderbolt "\n' | sudo tee /etc/dracut.conf.d/60-odl-multilink.conf
   sudo dracut --force --regenerate-all && sudo reboot
   ```

   After the reboot, `/usr/local/bin/odl-state` must show one `state=ready` device
   per cable on every box. If a box came up after its peers had given up waiting,
   run `sudo systemctl restart odinlink` on all boxes at about the same time.
   TP=2 also needs static IPs on the Thunderbolt network interfaces (`10.0.2.1` ↔
   `10.0.2.2`; `odinlink-multilink/support/setup-fabric-net.sh` sets up the
   validated rig's mesh by MAC). Full procedure, verification and troubleshooting:
   [odinlink-multilink/docs/RUNBOOK.md](odinlink-multilink/docs/RUNBOOK.md).
3. Check a box with `source ~/exl3ds4/env.sh && python ~/exl3ds4/scripts/smoke_torch.py --require-ext`.

**Kernel updates.** The two modules are built for one kernel. Before rebooting into a
new kernel, stage them on every box with `odinlink-multilink/build.sh <new-kver>` and
`sudo odinlink-multilink/install.sh <new-kver>`
([RUNBOOK, "Kernel updates"](odinlink-multilink/docs/RUNBOOK.md#kernel-updates)).

The Engram tables must come from `deepseek-ai/DeepSeek-V4.1-Flash-1017`
(`setup.sh` downloads and SHA-checks them). To copy the model to another box
instead of downloading it again, use `scripts/sync_model_to_box1.sh` or rsync.

## Run

Run one stack at a time: TP=2 and TP=3 use the same GPUs and ports (HTTP `1234`,
NCCL `29650`, control `29651`), so stop one before starting the other. Both
controllers first probe the OdinLink cable map (which `/dev/odl_tb5_N` reaches which
box: it changes across reboots and driver reloads) and hand each rank its own
`ODL_RCCL_PEERS`.

### TP=2 (Box1 + Box2)

From Box1:

```bash
~/exl3ds4/exl3ds4.sh start      # probes the cables, starts rank 1 on Box2 over ssh, then rank 0; ~2 min to load
~/exl3ds4/exl3ds4.sh status     # exit code 0 ready, 3 loading, 1 down
~/exl3ds4/exl3ds4.sh logs       # follow rank 0 (logs rank1 / logs all)
~/exl3ds4/exl3ds4.sh stop       # stops both boxes
~/exl3ds4/exl3ds4.sh probe      # print the current cable map (server stopped)
```

Logs: `~/exl3ds4/logs/exl3ds4-rank0.log` on Box1, `...-rank1.log` on Box2.

### TP=3 (Box1 + Box2 + Box3)

From Box1:

```bash
cd ~/exl3ds4
EXL3_TP_MODEL_DIR=$HOME/exl3ds4/model-3.5bpw ./tools/ds41/tp3_start.sh start
./tools/ds41/tp3_start.sh status    # rank 0 /health, the worker processes, the watchdog
./tools/ds41/tp3_start.sh logs      # follow rank 0
./tools/ds41/tp3_start.sh restart   # stop + start, keeping the running stack's settings
./tools/ds41/tp3_start.sh stop      # stops the watchdog and all three boxes
./tools/ds41/tp3_start.sh probe     # print the cable map of all three boxes (server stopped)
./tools/ds41/tp3_start.sh watch     # attach a watchdog to a stack that is already up (unwatch: remove it)
```

`start` probes the cables on all three boxes at once, starts the workers (rank 1 on
Box2, rank 2 on Box3) over ssh, waits until both have loaded the model, then starts
rank 0. The server is ready about three minutes later, when
`curl -s localhost:1234/health` reports `"status": "ok"`. Without
`EXL3_TP_MODEL_DIR` it serves the 2.9 bpw quant from `~/exl3ds4/model`. Logs:
`~/exl3ds4/logs/tp3-rank0.log` on Box1, `tp3-rank1.log` on Box2, `tp3-rank2.log` on Box3.

`start` also leaves a watchdog running on Box1 (`~/exl3ds4/logs/tp3-watchdog.log`). When
the stack dies it restarts all three ranks with the settings they were started with: a
rank's process exited, or a worker logged that it failed (rank 0 cannot resynchronize after
that, and would otherwise sit in its next collective until the 10-minute NCCL timeout). The
server is back about three minutes later, with an empty conversation cache. Repeated
failures back off (30 s, doubling to 15 minutes; reset after 30 minutes up). It never
judges a rank by HTTP, so a long prefill is not taken for a hang; a hang inside a
collective ends when NCCL's own timeout takes the ranks down, and the watchdog restarts
them then. Each rank runs in its own process group, so a stop or restart also kills the
GPU processes it spawned, even one stuck in a collective. `EXL3_TP3_WATCHDOG=0` starts
without a watchdog. Always stop TP=3 with `tp3_start.sh stop` (before starting TP=2, for
instance): to the watchdog, any other way of killing it is a crash.

### Requests

Both stacks serve the same API on Box1:

```bash
curl http://192.168.0.102:1234/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Name three prime numbers."}],"max_tokens":200}'
```

`perf/tp3/http_check.py [url]` sends one DSpark and one plain greedy request and
prints the decode speed from the server's own timings. The server listens on
`0.0.0.0:1234` with no authentication: keep it on the LAN.

## Using the server

Endpoints: `POST /v1/chat/completions` (streaming or not), `POST /v1/completions`,
`GET /v1/models`, `GET /health`, `GET /metrics`, `GET /tp/status` (every rank's
health). Thinking is on by default: replies carry `reasoning_content` and
`content`, and tool calls use the DeepSeek DSML format.

Image input is on by default (`/health` reports `"vision": true`): send
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
parts in a message and the model reads them. Each image costs one vision-tower
pass on Box1 plus its expanded token span in the prompt; DSpark and SSE
streaming compose with it, and a broken image is a clean 400. Design, caps and
knobs (`EXL3_TP_VISION`, `EXL3_TP_MAX_IMAGES`, ...):
[tools/ds41/README.md](tools/ds41/README.md#vision-image-input-2026-10-29).

Per-request options beyond the OpenAI ones:

| field | effect |
|---|---|
| `"speculative": false` | plain decoding for this request (DSpark is on by default, sampled requests included) |
| `"spec_k": 1..5` | cap the draft length |
| `"session": 1..4` | pin a hot conversation slot (TP=2); normally omit it and let prefix matching route |
| `"cache_prompt": false` | do not reuse or save the cached prompt |
| `"cache_salt": "..."` | prompt-cache namespace; a new salt drops every cached conversation |
| `"reasoning_effort"`, `"chat_template_kwargs": {"thinking": false}` | thinking budget / off |

The server keeps four conversation slots by default (`EXL3_TP_SESSIONS=4`) over
one shared 512K-token pool. It recognizes a returning conversation from the
prompt prefix itself, so clients need no session id: a turn that still contains
the previous prompt (modulo the template's one-token generation header) returns
to its slot. A different conversation, such as a client's title or summary
request, gets another slot instead of overwriting the first. When all slots are
hot, the least-recently-used conversation is spilled to one local NVMe image per
rank and reloaded on its next turn instead of re-prefilling; images last for the
server's lifetime only. `"session": N` can pin a hot slot, but auto-matching is
the normal path. One request runs at a time; others queue, so GPU work stays serialized.

## Configuration

### TP=2

`exl3ds4.sh` sets the validated defaults: 512K-token context
(`EXL3_TP_SAFE_CTX` / `EXL3_TP_CACHE_TOKENS`), DSpark on (`EXL3_TP_DSPARK=1`),
four cached conversations (`EXL3_TP_SESSIONS=4`) with per-session prompt reuse
(two 32 MiB checkpoints per conversation; `EXL3_TP_PREFIX_CACHE` controls the
single-session engine and multi-session keeps one namespace per conversation),
the NVMe overflow tier on (`EXL3_TP_DISK_CACHE=1`: 32 GiB per rank under
`EXL3_TP_DISK_CACHE_DIR`, default `$EXL3DS4_ROOT/diskcache`, 300 s frame
timeout), the vision tower on (`EXL3_TP_VISION=1`, up to 16 images per request;
a failed tower load degrades to text-only instead of failing startup;
`EXL3_TP_VISION_STRICT=1` makes it fatal), and a 30 s SSE keep-alive
(`KYOJIN_SSE_PING_S=30`). `EXL3_TP_SESSIONS=1` rolls back to the single-session
engine, protocol 4 and the old one-conversation cache exactly.

Every optimization can be switched off on its own if you suspect it:
`EXL3_ROCM_MOE_DECODE_RANGE=0`, `EXL3_TP_ODL_AR2=0`, `EXL3_TP_SPEC_SAMPLED=0`,
`EXL3_TP_DISK_CACHE=0`, `EXL3_TP_VISION=0`, or `EXL3_TP_SESSIONS=1` for the old
single-session engine. In multi-session mode use `"cache_prompt": false` to skip
prompt reuse for a request; `EXL3_TP_PREFIX_CACHE=0` disables it in the
single-session engine. The full list is in
[tools/ds41/README.md](tools/ds41/README.md#knobs-and-defaults).

On other machines, override the rig's addresses and paths:

| variable | default | meaning |
|---|---|---|
| `EXL3DS4_ROOT` | the checkout | install root (venv, `src/`, `model/`) |
| `EXL3DS4_REMOTE` | `davidcanar@10.0.2.2` | ssh target of Box2 |
| `EXL3_MASTER_ADDR` | `10.0.2.1` | Box1's Thunderbolt IP |
| `EXL3_ODL_IP0`, `EXL3_ODL_IP1` | `10.0.2.1`, `10.0.2.2` | the ranks' OdinLink identities in the cable map |
| `EXL3_TP2_NO_PROBE` | unset | `1` skips the cable probe (the launchers' default maps apply) |
| `EXL3_TP_MODEL_DIR`, `EXL3_TP_ENGRAM_DIR` | `$EXL3DS4_ROOT/model`, `.../engram-src-1017` | model data (same path on both boxes) |
| `EXL3_ODL_SRC`, `EXL3_ODL_BUILD` | `~/.cache/odinlink`, `.../build` | OdinLink source and build |

### TP=3

`tp3_start.sh` forwards these to every rank, and they must match on all of them:
`EXL3_TP_MODEL_DIR` (default `$EXL3DS4_ROOT/model`), `EXL3_TP_SAFE_CTX` /
`EXL3_TP_CACHE_TOKENS` (1048576), `EXL3_TP_DSPARK` (1), `EXL3_TP_BUDGETS` (GiB per rank,
`100,100,100`), `EXL3_TP_ODL_AR3` (1; `0` puts decode back on RCCL), `EXL3_TP_NCCL_SYNC`
(`stream`), `EXL3_TP_SPAN`, `NCCL_MAX_NCHANNELS` (4), the conversation cache knobs
(`EXL3_TP_SESSIONS` 4, `EXL3_TP_PREFIX_*`, `EXL3_TP_DISK_CACHE` 1, `EXL3_TP_DISK_CACHE_DIR`
`$EXL3DS4_ROOT/diskcache`, `EXL3_TP_DISK_CACHE_GIB` 100, `EXL3_TP_DISK_TIMEOUT` 300) and
the vision knobs (`EXL3_TP_VISION` 1, `EXL3_TP_VISION_STRICT`, `EXL3_TP_MAX_IMAGES` 16).
The 1M pool costs 1.6 GiB per rank more than 512K, and the vision tower ~0.7 GiB on rank 0
only; with both, Box1 keeps ~15 GiB free. Spilled conversations take ~68 MB plus ~3.3 KB
per token on each box (a full 1M-token one ~3.3 GB), so the 100 GiB tier holds a few hundred
long conversations; they live until the server restarts. `EXL3_TP_DISK_TIMEOUT` is not a
lifetime: it is how long rank 0 waits for a worker's disk write or read before treating it
as late. Rank 0 sends an SSE keep-alive comment after 30 s without output
(`KYOJIN_SSE_PING_S`), so clients do not time out during a long prefill.

| variable | default | meaning |
|---|---|---|
| `EXL3_BOX2`, `EXL3_BOX3` | `192.168.0.103`, `192.168.0.105` | ssh targets of the workers |
| `EXLLAMA_MASTER_ADDR` | `192.168.0.102` | Box1's LAN address: NCCL bootstrap and control channel (forwarded to every rank) |
| `EXL3_HTTP_PORT` | `1234` | HTTP port on Box1 |
| `EXL3_TP3_NO_PROBE` | unset | `1` skips the cable probe; `ODL_RCCL_PEERS_0/1/2` (or the scripts' defaults) apply |

The LAN interface each rank bootstraps over (`eno1` on Box1/Box2, `enp198s0f0u2u4` on
Box3) differs per box, so it is not forwarded: it is a rig default in
`tools/ds41/tp3_serve.sh` (rank 0) and `tools/ds41/tp3_worker.sh` (ranks 1-2), next to
the fallback cable maps. Edit those for another rig (`EXL3_IFACE` overrides it for a
rank started by hand).

## Known limitations

- If any rank dies mid-request, all of them have to restart: `exl3ds4.sh restart` at
  TP=2; at TP=3 the watchdog does it. The request in flight is lost either way, and so is
  the conversation cache. Nothing starts a stack at boot.
- One request runs at a time; four cached conversations share the pool but their
  GPU work is serialized. Pages are grow-only within a conversation's life;
  freedom comes from whole-session spill or eviction.
- Spilled images are one-shot: a server restart starts with an empty disk tier
  and every conversation re-prefills.
- A conversation that repeats another conversation's entire last prompt within
  the one-token generation-header tolerance can match that slot. This is the
  same matching rule as the single-session cache and never changes the output.
- Memory is tight: GPU memory is system RAM, and at TP=2 the model leaves ~3.6 GiB on
  Box1 and ~2 GiB on Box2 for the OS. Keep other heavy jobs off the boxes while
  serving.
- Greedy output is not always bit-identical between two identical requests
  (see [docs/PERFORMANCE.md](docs/PERFORMANCE.md#reproducibility)).
- Image prompts do not use the prefix cache yet: each image request re-encodes
  its images and re-prefills (identifying "the same image" across requests needs
  pixel hashing, not built in this first version; text prompts keep the cache).
  Prefill speed at the full 512K context is unmeasured.
- An OdinLink cable occasionally drops a frame (an RX CRC error, around one per
  10^7 frames under load; the kernel log says `odl_tb5: RX CRC error`). `odl_ar3`
  detects and resends lost decode messages, but RCCL has no retransmit over
  OdinLink, so a loss during a large transfer such as a prefill all-reduce hangs the
  server until it is restarted.
- TP=3 serves a 1M-token context, but no prompt longer than 512K tokens has been run on it
  yet, and decode beyond 32K and prefill beyond 100K are unmeasured. Prefill slows as the
  context grows: a full 1M-token prompt takes on the order of an hour the first time
  (cached afterwards).

## Security notes

The HTTP API has no authentication and binds `0.0.0.0:1234` by default: keep it
on a trusted LAN or bind `127.0.0.1`. Image URL fetches are operator-trust and
are not SSRF-filtered (`EXL3_TP_IMAGE_HTTP=0` turns them off; the data:/base64
path stays available). The NVMe tier is local to each rank and trusted: image
metadata is checksummed, but the stash region is a pickle, so never point
`EXL3_TP_DISK_CACHE_DIR` at storage an untrusted party can write to. It is
created 0o700 and stale files are swept at startup. `EXL3_TP_DISK_CACHE_GIB`
caps each rank's images, and the image fetch limits
(`EXL3_TP_IMAGE_MAX_BYTES`, `_MAX_PIXELS`, `_FETCH_TIMEOUT`) bound the other
untrusted input path.

## Repository layout

```
setup.sh, env.sh       installer; shared ROCm/HIP environment
exl3ds4.sh             TP=2 controller: start | stop | restart | status | logs | probe
tools/ds41/            the OpenAI server, the TP engine and the launchers (+ CPU tests)
  tp_serve.py          TP=2 rank 0 (tp_serve_rank0.sh); tp_rank1.py / tp_rank1.sh is rank 1
  tp3_start.sh         TP=3 controller + watchdog: start | stop | restart | status | logs | probe | watch
  tp3_serve.py         TP=3 rank 0 (tp3_serve.sh); tp3_worker.py / tp3_worker.sh are ranks 1-2
  tp3_channel.py       rank 0's fan-out control channel to N-1 workers
  odl_peer_probe.py    per-boot OdinLink cable map, used by both controllers
  tp_sessions.py       shared-pool conversation slots and prefix auto-matching
  tp_disk.py           per-rank NVMe image store for spilled conversations
patches/exllamav3/     14 ExLlamaV3 commits (git am) on PR #449
patches/odinlink/      12 OdinLink-Five patches (git apply): driver fixes, multi-cable demux, RCCL plugin
patches/linux/         thunderbolt core: source-XDomain protocol handlers (multi-cable fabric)
odinlink-multilink/    builds and installs the patched core + driver for a kernel; RUNBOOK
odl_ar2/               OdinLink all-reduce used for TP=2 decode
odl_ar3/               OdinLink mesh all-reduce used for TP=3 decode
perf/, perf/tp3/       benchmarks, profilers, correctness checks, TP=3 repro harnesses
scripts/               build, launchers, smoke tests, model sync
docs/                  PERFORMANCE.md, DEVELOPMENT.md, tp3-runtime.md, tp3-review-2026-11-02.md
```

Running the tests, smoke tests and the rank protocol: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Sources and licenses

- ExLlamaV3: `anoane/exllamav3-anemone` branch `dsv41f` @ `d45e8961` (PR #449),
  cloned and patched by `setup.sh`.
- OdinLink-Five: `Geramy/OdinLink-Five` @ `4534f58`, cloned and patched by `setup.sh`
  (userspace) and `odinlink-multilink/build.sh` (driver).
- Linux `drivers/thunderbolt`: the distro kernel's own source, fetched and patched by
  `odinlink-multilink/build.sh`.
- The server layer is vendored from [Kyojin](https://github.com/Yamz-Labs/kyojin)
  1.8.2 (`e2f37564`).

This repository is MIT. Third-party parts keep their licenses: Kyojin (MIT),
ExLlamaV3 (MIT), OdinLink-Five (driver GPL-2.0, userspace MIT), the Linux
`thunderbolt` patch (GPL-2.0), and the DeepSeek-AI reference encoding used by the
tests. Model weights are not redistributed. Details in [NOTICE](NOTICE).
