#!/usr/bin/env python3
"""Plain (no-draft) two-node greedy baseline for DS4.1 EXL3.

Copy of tp_ds41_generate.py with the prompt and decode length parametrized by
environment (EXL3_DS41_PROMPT, EXL3_DS41_STEPS) so DSpark runs can be compared
against the exact plain greedy continuation of the same prompt.
"""

import json
import multiprocessing
import os
import socket
import struct
import time

import torch

from exllamav3 import Cache, Config, Model, Tokenizer

_ROOT = os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.environ.get("EXL3_TP_MODEL_DIR", os.path.join(_ROOT, "model"))
ENGRAM_DIR = os.environ.get("EXL3_TP_ENGRAM_DIR", os.path.join(_ROOT, "engram-src-1017"))
ENGRAM_SHARDS = (
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
)
CACHE_TOKENS = 2048
SPAN = 2048
PROMPT_TEXT = os.environ.get("EXL3_DS41_PROMPT", "The quick brown fox jumps over the lazy")
DECODE_STEPS = int(os.environ.get("EXL3_DS41_TOKENS", "24"))


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
    def __init__(self, rank, peer_addr, port):
        self.conn = None
        if rank == 0:
            server = socket.socket()
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(600)
            server.bind((peer_addr, port))
            server.listen(1)
            self.conn, _ = server.accept()
            server.close()
        else:
            deadline = time.time() + 600
            while True:
                try:
                    self.conn = socket.create_connection((peer_addr, port), timeout=10)
                    break
                except OSError:
                    if time.time() > deadline:
                        raise
                    time.sleep(1)
        self.conn.settimeout(600)

    def send_int(self, value):
        self.conn.sendall(struct.pack("<q", value))

    def recv_int(self):
        data = b""
        while len(data) < 8:
            chunk = self.conn.recv(8 - len(data))
            if not chunk:
                raise ConnectionError("Token channel closed by rank 0")
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
    model = Model.from_config(config, component="text")
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_batch_size=1)

    print(f"rank {rank}: starting DS4.1 cached TP load", flush=True)
    started = time.time()
    model.load(
        tensor_p=True,
        tp_backend="nccl",
        use_per_device=[110.0] * world_size,
        max_chunk_size=SPAN,
        progressbar=rank == 0,
    )
    print(f"rank {rank}: cached TP load completed in {time.time() - started:.1f}s", flush=True)

    tokenizer = Tokenizer.from_config(config)
    vocab_size = tokenizer.actual_vocab_size
    channel = TokenChannel(rank, os.environ["EXLLAMA_MASTER_ADDR"], control_port)
    prompt = tokenizer.encode(PROMPT_TEXT, add_bos=True)
    prompt_len = prompt.shape[-1]
    with torch.inference_mode():
        state = cache.get_new_state()

    started = time.time()
    model.prefill(
        prompt[:, :-1],
        params={
            "attn_mode": "flash_attn",
            "cache": cache,
            "batch_shape": (1, SPAN),
            "past_len": 0,
            "recurrent_states": [state],
        },
    )
    print(f"rank {rank}: prefill completed in {time.time() - started:.2f}s", flush=True)

    generated = []
    token = prompt[0, -1]
    position = prompt_len - 1
    started = time.time()
    for step in range(DECODE_STEPS):
        params = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "batch_shape": (1, SPAN),
            "past_len": position,
            "recurrent_states": [state],
            "last_tokens_only": 1,
        }
        logits = model.forward(token.view(1, 1), params)
        if rank == 0:
            next_token = int(logits[0, -1, :vocab_size].argmax().item())
            channel.send_int(next_token)
        else:
            next_token = channel.recv_int()
        generated.append(next_token)
        token = torch.tensor(next_token, dtype=torch.long)
        position += 1

    elapsed = time.time() - started
    if rank == 0:
        text = tokenizer.decode(torch.tensor(generated).view(1, -1))[0]
        print(
            f"rank 0: generated {DECODE_STEPS} tokens in {elapsed:.2f}s "
            f"({DECODE_STEPS / elapsed:.2f} tok/s)",
            flush=True,
        )
        print(f"rank 0: PLAIN_IDS={generated}", flush=True)
        print(f"rank 0: text={text!r}", flush=True)

    channel.close()
    model.unload()
    print(f"rank {rank}: cached DS4.1 unload completed", flush=True)
    if rank == 0:
        print("DS4.1 two-node cached prefill/decode: PASS", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
