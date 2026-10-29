# ExLlamaV3ROCm

Serve **DeepSeek-V4.1-Flash** (the `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`
quant) on **two AMD Strix Halo machines** joined by one Thunderbolt/USB4 cable.
[ExLlamaV3](https://github.com/turboderp/ExLlamaV3) splits the model across both
boxes (tensor parallelism), [OdinLink](https://github.com/Geramy/OdinLink-Five)
carries the traffic over the cable, and an OpenAI-compatible server exposes it
with MTP speculative decoding ("DSpark"), a prompt cache and a 512K-token context.

This repository is the working deployment, packaged for reinstallation: an
installer, a controller script, the server, and the patches it needs on top of
ExLlamaV3 and OdinLink. Model weights are downloaded by the installer, not
stored here.

## Results

Decode speed over HTTP (tok/s, 120 generated tokens):

| prompt tokens | DSpark, greedy | DSpark, sampled T=1.0 | plain decoding |
|---:|---:|---:|---:|
| 847 | 34.8 | 34.5 | 25.4 |
| 25,669 | 36.4 | 30.4 | 24.5 |
| 62,822 | 35.6 | 33.2 | 24.3 |

Prompt processing runs at ~300 tok/s. Follow-up turns reuse the cached prompt:
the next turn of a 19.8K-token conversation starts answering after **0.47 s**
instead of 65 s. Details, history and memory budget: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## What you need

- Two Strix Halo (`gfx1151`) machines with 128 GB each, one Thunderbolt/USB4
  cable between them, Fedora (validated on Fedora 44, kernel 7.1), the same
  user and paths on both.
- About 400 GiB of disk per box (196 GiB quant + 189 GiB Engram tables).
- Root once per box, for the kernel command line and the OdinLink module.

The validated rig, whose addresses are the defaults:

| | LAN ssh | Thunderbolt IP |
|---|---|---|
| Box1, rank 0 (runs the server) | `davidcanar@192.168.0.102` | `10.0.2.1` |
| Box2, rank 1 | `davidcanar@10.0.2.2` (from Box1) | `10.0.2.2` |

## Install

On **both** boxes, at the same path:

```bash
git clone https://github.com/davidcanar/ExLlamaV3ROCm.git ~/exl3ds4
cd ~/exl3ds4
./setup.sh            # venv, ROCm 10.1 + PyTorch 2.14 wheels, patched ExLlamaV3 and
                      # OdinLink, native builds, model + Engram download (--no-model skips it)
```

`setup.sh` is idempotent. It needs `git rsync cmake make gcc curl sha256sum`
and installs `uv` if missing. Then, as root on both boxes:

```bash
# 1. kernel command line (GRUB), then reboot:
#      amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856
# 2. OdinLink kernel module (built from the patched source):
cmake --build ~/.cache/odinlink/build --target driver
sudo insmod ~/.cache/odinlink/build/driver/odl_tb5.ko odl_ring_size=1024
sudo cp ~/.cache/odinlink/driver/71-odl-tb5.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules
```

`/dev/odl_tb5_0` appears once both boxes have loaded the module. Check a box
with `source ~/exl3ds4/env.sh && python ~/exl3ds4/scripts/smoke_torch.py --require-ext`.

The Engram tables must come from `deepseek-ai/DeepSeek-V4.1-Flash-1017`
(`setup.sh` downloads and SHA-checks them). To copy the model to the second box
instead of downloading it again, use `scripts/sync_model_to_box1.sh` or rsync.

## Run

From Box1:

```bash
~/exl3ds4/exl3ds4.sh start      # starts rank 1 on Box2 over ssh, then rank 0; ~2 min to load
~/exl3ds4/exl3ds4.sh status     # exit code 0 ready, 3 loading, 1 down
~/exl3ds4/exl3ds4.sh logs       # follow rank 0 (logs rank1 / logs all)
~/exl3ds4/exl3ds4.sh stop       # stops both boxes
```

```bash
curl http://192.168.0.102:1234/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Name three prime numbers."}],"max_tokens":200}'
```

The server listens on `0.0.0.0:1234` with no authentication: keep it on the
LAN. Logs: `~/exl3ds4/logs/exl3ds4-rank0.log` on Box1, `...-rank1.log` on Box2.

## Using the server

Endpoints: `POST /v1/chat/completions` (streaming or not), `POST /v1/completions`,
`GET /v1/models`, `GET /health`, `GET /metrics`, `GET /tp/status` (both ranks'
health). Thinking is on by default: replies carry `reasoning_content` and
`content`, and tool calls use the DeepSeek DSML format.

Per-request options beyond the OpenAI ones:

| field | effect |
|---|---|
| `"speculative": false` | plain decoding for this request (DSpark is on by default, sampled requests included) |
| `"spec_k": 1..5` | cap the draft length |
| `"cache_prompt": false` | do not reuse or save the cached prompt |
| `"cache_salt": "..."` | prompt-cache namespace; a new salt drops the cache |
| `"reasoning_effort"`, `"chat_template_kwargs": {"thinking": false}` | thinking budget / off |

The prompt cache follows one conversation: a request that starts differently
(a client's title or summary request, say) overwrites it, and the next turn of
the main conversation prefills from where they diverge. One request runs at a
time; others queue.

## Configuration

`exl3ds4.sh` sets the validated defaults: 512K-token context
(`EXL3_TP_SAFE_CTX` / `EXL3_TP_CACHE_TOKENS`), DSpark on (`EXL3_TP_DSPARK=1`),
prompt cache on (`EXL3_TP_PREFIX_CACHE=1`, two 32 MiB checkpoints per box). Every
optimization can be switched off on its own if you suspect it:
`EXL3_ROCM_MOE_DECODE_RANGE=0`, `EXL3_TP_ODL_AR2=0`, `EXL3_TP_SPEC_SAMPLED=0`,
`EXL3_TP_PREFIX_CACHE=0`. The full list is in
[tools/ds41/README.md](tools/ds41/README.md#knobs-and-defaults).

On other machines, override the rig's addresses and paths:

| variable | default | meaning |
|---|---|---|
| `EXL3DS4_ROOT` | the checkout | install root (venv, `src/`, `model/`) |
| `EXL3DS4_REMOTE` | `davidcanar@10.0.2.2` | ssh target of Box2 |
| `EXL3_MASTER_ADDR` | `10.0.2.1` | Box1's Thunderbolt IP |
| `EXL3_TP_MODEL_DIR`, `EXL3_TP_ENGRAM_DIR` | `$EXL3DS4_ROOT/model`, `.../engram-src-1017` | model data (same path on both boxes) |
| `EXL3_ODL_SRC`, `EXL3_ODL_BUILD` | `~/.cache/odinlink`, `.../build` | OdinLink source and build |

## Known limitations

- If either rank dies mid-request, restart both (`exl3ds4.sh restart`).
- Memory is tight: GPU memory is system RAM, and the model leaves ~2 GiB per
  box for the OS. Keep other heavy jobs off the boxes while serving.
- Greedy output is not always bit-identical between two identical requests
  (see [docs/PERFORMANCE.md](docs/PERFORMANCE.md#reproducibility)).
- No vision input. Prefill speed at the full 512K context is unmeasured.

## Repository layout

```
setup.sh, env.sh       installer; shared ROCm/HIP environment
exl3ds4.sh             controller: start | stop | restart | status | logs
tools/ds41/            the OpenAI server and the two-rank engine (+ CPU tests)
patches/exllamav3/     12 ExLlamaV3 commits (git am) on PR #449
patches/odinlink/      3 OdinLink-Five diffs (git apply): multi-frame, device allow-list, RX/teardown fixes
odl_ar2/               OdinLink all-reduce used for decode
perf/                  benchmarks, profiler, correctness checks
scripts/               build, launchers, smoke tests, model sync
docs/                  PERFORMANCE.md, DEVELOPMENT.md
```

Running the tests, smoke tests and the rank protocol: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Sources and licenses

- ExLlamaV3: `anoane/exllamav3-anemone` branch `dsv41f` @ `d45e8961` (PR #449),
  cloned and patched by `setup.sh`.
- OdinLink-Five: `Geramy/OdinLink-Five` @ `4534f58`, cloned and patched by `setup.sh`.
- The server layer is vendored from [Kyojin](https://github.com/Yamz-Labs/kyojin)
  1.8.2 (`e2f37564`).

This repository is MIT. Third-party parts keep their licenses: Kyojin (MIT),
ExLlamaV3 (MIT), OdinLink-Five (driver GPL-2.0, userspace MIT), and the
DeepSeek-AI reference encoding used by the tests. Model weights are not
redistributed. Details in [NOTICE](NOTICE).
