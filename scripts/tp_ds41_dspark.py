#!/usr/bin/env python3
"""Two-node DSpark (MTP) speculative cached generation for DS4.1 EXL3.

Extends tp_ds41_generate.py with the PR449 DSpark drafter:

  - the trunk loads tensor-parallel across both hosts (rank 0 = box1, rank 1 = box2);
  - rank 0 additionally loads the mtp.* DSpark component layer-split on its own GPU,
    attaches it to the TP trunk (private embedding on CPU + private head on GPU, so
    the draft sampling loop issues no collectives), and keeps the drafter's paged
    main-kv cache (CacheLayer_dspark) that rides the trunk's contiguous block table;
  - both ranks run IDENTICAL trunk dispatch sequences: prefill, then per round one
    multi-row verify forward (recurrent_history=True) and one state rewind. Rank 0
    drafts locally, then broadcasts the verify window and the accepted count to
    rank 1 over the control channel so both drivers stay in lockstep;
  - greedy acceptance: output must equal the plain greedy continuation recorded by
    tp_ds41_generate.py (logs/generate-odl-rank0.log).
"""

import json
import multiprocessing
import os
import socket
import struct
import time

import torch

from exllamav3 import Cache, Config, Model, Tokenizer
from exllamav3.constants import PAGE_SIZE

_ROOT = os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.environ.get("EXL3_TP_MODEL_DIR", os.path.join(_ROOT, "model"))
ENGRAM_DIR = os.environ.get("EXL3_TP_ENGRAM_DIR", os.path.join(_ROOT, "engram-src-1017"))
ENGRAM_SHARDS = (
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
)
CACHE_TOKENS = 2048
SPAN = 2048
MAX_HISTORY = 8          # recurrent rollback capacity: draft window is 5 + slack
WANT_TOKENS = int(os.environ.get("EXL3_DS41_TOKENS", "8"))
MAX_ROUNDS = int(os.environ.get("EXL3_DS41_MAX_ROUNDS", "64"))
PROMPT_TEXT = os.environ.get("EXL3_DS41_PROMPT", "The quick brown fox jumps over the lazy")
# Plain greedy baseline ids (comma-separated). Empty = no baseline: record and report only.
_EXPECTED_ENV = os.environ.get("EXL3_DS41_EXPECTED", "").strip()
if _EXPECTED_ENV:
    EXPECTED_IDS = [int(x) for x in _EXPECTED_ENV.split(",")]
elif PROMPT_TEXT == "The quick brown fox jumps over the lazy" and WANT_TOKENS == 8:
    EXPECTED_IDS = [6397, 271, 5, 8077, 223, 20, 28, 455]   # logs/generate-odl-rank0.log
else:
    EXPECTED_IDS = None
CHANNEL_TIMEOUT = 1200.0
USE_PER_DEVICE = [float(x) for x in os.environ.get("EXL3_DS41_BUDGETS", "101,110").split(",")]
# Draft-confidence threshold override (read by the drafter at construction; 0.5 upstream default).
# EXL3_DS41_CONF=0.05 e.g. drafts the full 5-token window every round.
if os.environ.get("EXL3_DS41_CONF"):
    os.environ["EXL3_DSPARK_CONF"] = os.environ["EXL3_DS41_CONF"]


def load_config_with_engram():
    config = Config.from_directory(MODEL_DIR)
    for filename in ENGRAM_SHARDS:
        config.stc.add_tensor_files(os.path.join(ENGRAM_DIR, filename))
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as handle:
        weight_map = json.load(handle)["weight_map"]
    for layer in (1, 14):
        for name in ("q_weight", "k_weight"):
            key = f"layers.{layer}.engram.{name}"
            config.stc.tensor_file_map[key] = os.path.join(MODEL_DIR, weight_map[key])
    return config


class TokenChannel:
    """Small rank-0-to-rank-1 channel for draft decisions; not used for model tensors."""

    def __init__(self, rank, peer_addr, port, timeout = 600.0):
        self.conn = None
        if rank == 0:
            server = socket.socket()
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(timeout)
            server.bind((peer_addr, port))
            server.listen(1)
            self.conn, _ = server.accept()
            server.close()
        else:
            deadline = time.time() + timeout
            while True:
                try:
                    self.conn = socket.create_connection((peer_addr, port), timeout=10)
                    break
                except OSError:
                    if time.time() > deadline:
                        raise
                    time.sleep(1)
        self.conn.settimeout(timeout)

    def send_int(self, value):
        self.conn.sendall(struct.pack("<q", value))

    def recv_int(self):
        data = b""
        while len(data) < 8:
            chunk = self.conn.recv(8 - len(data))
            if not chunk:
                raise ConnectionError("token channel closed by peer")
            data += chunk
        return struct.unpack("<q", data)[0]

    def close(self):
        if self.conn is not None:
            self.conn.close()


def main():
    rank = int(os.environ["EXL3_TP_NODE_RANK"])
    world_size = int(os.environ.get("EXL3_TP_WORLD_SIZE", "2"))
    control_port = int(os.environ.get("EXL3_TP_CONTROL_PORT", "29630"))

    config = load_config_with_engram()
    export_layers = frozenset(t - 1 for t in config.dspark_target_layer_ids)   # {36, 37, 38}
    block = config.dspark_block_size                                          # 5

    model = Model.from_config(config, component="text")
    # Draft rollback history on the recurrent layers (verify passes up to 6 rows)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_batch_size=1, max_history=MAX_HISTORY)

    print(f"rank {rank}: starting DS4.1 cached TP load (budgets {USE_PER_DEVICE} GiB)", flush=True)
    started = time.time()
    model.load(
        tensor_p=True,
        tp_backend="nccl",
        use_per_device=USE_PER_DEVICE,
        max_chunk_size=SPAN,
        progressbar=rank == 0,
    )
    print(f"rank {rank}: cached TP load completed in {time.time() - started:.1f}s", flush=True)

    # ---- rank 0 only: the DSpark drafter, loaded beside the TP trunk ----
    # (layer-split, no TP dispatches: rank 1's dispatch sequence is unaffected)
    draft_model = draft_cache = draft_bt = None
    if rank == 0:
        free_b, total_b = torch.cuda.mem_get_info(0)
        print(f"rank 0: VRAM before draft load: {free_b / 2**30:.1f} GiB free of {total_b / 2**30:.1f} GiB", flush=True)
        started = time.time()
        draft_model = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=CACHE_TOKENS, max_batch_size=1)
        draft_model.load(progressbar=False)
        draft_model.attach_to(model)   # loads private embed (CPU) + head (GPU): collective-free sampling
        draft_bt = torch.arange(CACHE_TOKENS // PAGE_SIZE, dtype=torch.int32).view(1, -1)
        assert draft_model.own_head is not None, "drafter did not get its private head (attach order bug)"
        free_b, _ = torch.cuda.mem_get_info(0)
        print(f"rank 0: DSpark draft loaded in {time.time() - started:.1f}s, "
              f"{free_b / 2**30:.1f} GiB VRAM free", flush=True)

    tokenizer = Tokenizer.from_config(config)
    vocab_size = tokenizer.actual_vocab_size
    channel = TokenChannel(rank, os.environ["EXLLAMA_MASTER_ADDR"], control_port, timeout=CHANNEL_TIMEOUT)
    prompt = tokenizer.encode(PROMPT_TEXT, add_bos=True)
    prompt_len = prompt.shape[-1]

    emitted = []
    stats = []
    margins = []          # per round: top-2 logit gap of every verify row (near-tie diagnosis)
    anchor = int(prompt[0, -1])
    position = prompt_len - 1
    emitted_count = 0
    round_idx = 0

    with torch.inference_mode():
        state = cache.get_new_state()

        # ---- prefill (both ranks; taps exported at layers 36/37/38) ----
        pparams = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "batch_shape": (1, SPAN),
            "past_len": 0,
            "recurrent_states": [state],
            "export_state_layers": export_layers,
        }
        started = time.time()
        model.prefill(prompt[:, :-1], pparams)
        print(f"rank {rank}: prefill ({prompt_len - 1} tokens) in {time.time() - started:.2f}s", flush=True)

        # rank 0: project the trunk taps into the drafter's paged main-kv rows
        if rank == 0:
            taps = pparams.get("export_states")
            assert taps and len(taps) == len(export_layers), \
                f"tap export missing after prefill (got {None if not taps else len(taps)} taps)"
            assert all(torch.isfinite(t).all().item() for t in taps), "non-finite tap export"
            started = time.time()
            draft_model.update_kv_from_target(
                target_hidden=taps,
                cache=draft_cache,
                params={
                    "block_table": draft_bt,
                    "cache_seqlens": torch.tensor([0], dtype=torch.int32),
                },
            )
            print(f"rank 0: draft main-kv written for {taps[0].shape[1]} positions "
                  f"in {time.time() - started:.2f}s", flush=True)

        # ---- speculative rounds ----
        while emitted_count < WANT_TOKENS and round_idx < MAX_ROUNDS:
            round_idx += 1

            # -- rank 0 drafts locally (no collectives; rank 1 waits) --
            if rank == 0:
                t0 = time.time()
                dparams = {
                    "attn_mode": "flash_attn",
                    "cache": draft_cache,
                    "block_table": draft_bt,
                    "cache_seqlens": torch.tensor([position], dtype=torch.int32),
                }
                dstate = draft_model.forward(torch.tensor([anchor]).view(1, 1), dparams)
                ids = draft_model.sample_from_state(dstate, dparams)
                conf_len = int(dparams.get("draft_confidence_len", block))
                k = max(0, min(block, conf_len))
                drafts = [int(x) for x in ids[0, 1: 1 + k]]
                t_draft = time.time() - t0
            else:
                drafts = None
                k = None

            # -- rank 0 broadcasts the verify window [anchor, drafts...] --
            if rank == 0:
                channel.send_int(k)
                for tok in [anchor] + drafts:
                    channel.send_int(tok)
                verify_ids = torch.tensor([[anchor] + drafts], dtype=torch.long)
            else:
                k = channel.recv_int()
                window = [channel.recv_int() for _ in range(k + 1)]
                verify_ids = torch.tensor([window], dtype=torch.long)

            # -- both ranks: one verify forward over the window (history + taps) --
            vparams = {
                "attn_mode": "flash_attn",
                "cache": cache,
                "batch_shape": (1, SPAN),
                "past_len": position,
                "recurrent_states": [state],
                "recurrent_history": True,
                "export_state_layers": export_layers,
            }
            t0 = time.time()
            logits = model.forward(verify_ids, vparams)     # None on rank 1 by design
            t_verify = time.time() - t0

            # -- rank 0: greedy acceptance against each verify row --
            if rank == 0:
                lg = logits[0, :, :vocab_size].float()
                greedy = [int(x) for x in lg.argmax(dim=-1)]
                top2 = lg.topk(2, dim=-1).values
                round_margins = (top2[:, 0] - top2[:, 1]).tolist()
                a = 0
                while a < k and drafts[a] == greedy[a]:
                    a += 1
                bonus = greedy[a]
            # -- broadcast accepted count; both ranks settle the state identically --
            if rank == 0:
                channel.send_int(a)
            else:
                a = channel.recv_int()

            # TP dispatch on both hosts (mp_cache_recurrent_rewind): must be mirrored
            state.rewind(k - a)
            position += a + 1
            emitted_count += a + 1

            if rank == 0:
                # refresh the drafter's main-kv from the accepted rows of this pass
                draft_model.update_kv_from_target(
                    target_hidden=vparams.get("export_states"),
                    cache=draft_cache,
                    lengths=[a + 1],
                    params={
                        "block_table": draft_bt,
                        "cache_seqlens": torch.tensor([position - (a + 1)], dtype=torch.int32),
                    },
                )
                new_tokens = drafts[:a] + [bonus]
                emitted += new_tokens
                stats.append((k, a, t_draft, t_verify, new_tokens))
                margins.append(round_margins)
                anchor = bonus
                print(f"rank 0: round {round_idx}: drafted k={k}, accepted a={a}, "
                      f"emitted {new_tokens}  (draft {t_draft:.2f}s, verify {t_verify:.2f}s)", flush=True)

    channel.close()

    if rank == 0:
        drafted_total = sum(s[0] for s in stats)
        accepted_total = sum(s[1] for s in stats)
        t_draft_total = sum(s[2] for s in stats)
        t_verify_total = sum(s[3] for s in stats)
        elapsed = t_draft_total + t_verify_total
        print(f"\nrank 0: DSpark summary: {len(stats)} rounds, drafted {drafted_total}, "
              f"accepted {accepted_total} ({100.0 * accepted_total / max(1, drafted_total):.0f}% acceptance)", flush=True)
        print(f"rank 0: draft time {t_draft_total:.2f}s, verify time {t_verify_total:.2f}s "
              f"({elapsed:.2f}s for {len(emitted)} tokens = {len(emitted) / max(elapsed, 1e-9):.2f} tok/s)", flush=True)
        print(f"rank 0: emitted ids={emitted}", flush=True)
        text = tokenizer.decode(torch.tensor(emitted[:WANT_TOKENS]).view(1, -1))[0]
        print(f"rank 0: text={text!r}", flush=True)

        got = emitted[:WANT_TOKENS]
        assert accepted_total > 0, f"no drafted tokens were accepted in {len(stats)} rounds"
        if draft_model._conf_stats is not None:
            print(f"rank 0: draft confidence scores per round: "
                  f"{[[round(v, 3) for v in r] for r in draft_model._conf_stats]}", flush=True)
        if EXPECTED_IDS is None:
            print("rank 0: no baseline given; recorded ids above for comparison", flush=True)
        else:
            if got != EXPECTED_IDS[:len(got)]:
                # Locate the first divergence and report the near-tie margin at that row
                for i, (g, e) in enumerate(zip(got, EXPECTED_IDS)):
                    if g != e:
                        consumed = 0
                        for (k, a, _, _, toks), rm in zip(stats, margins):
                            if consumed <= i < consumed + len(toks):
                                j = i - consumed
                                print(f"rank 0: DIVERGENCE at emitted index {i}: got {g}, expected {e}; "
                                      f"round tokens {toks}, row {j} top-2 margin {rm[min(j, len(rm) - 1)]:.3f}", flush=True)
                                break
                            consumed += len(toks)
                        break
                raise AssertionError(f"DSpark output diverges from plain greedy: {got} != {EXPECTED_IDS[:len(got)]}")
            print("rank 0: DSpark output matches plain greedy token for token", flush=True)

    if rank == 0:
        started = time.time()
        draft_model.unload()
        print(f"rank 0: draft unload completed in {time.time() - started:.1f}s", flush=True)
    model.unload()
    print(f"rank {rank}: cached DS4.1 unload completed", flush=True)
    if rank == 0:
        print("DS4.1 two-node DSpark cached generation: PASS", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
