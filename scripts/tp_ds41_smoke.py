#!/usr/bin/env python3
"""Full DeepSeek-V4.1-Flash-1017 EXL3 two-node text-only smoke test."""

import json
import multiprocessing
import os
import time

import torch

from exllamav3 import Config, Model

_ROOT = os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.environ.get("EXL3_TP_MODEL_DIR", os.path.join(_ROOT, "model"))
ENGRAM_DIR = os.environ.get("EXL3_TP_ENGRAM_DIR", os.path.join(_ROOT, "engram-src-1017"))
ENGRAM_SHARDS = (
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
)


def load_config_with_engram():
    config = Config.from_directory(MODEL_DIR)
    for filename in ENGRAM_SHARDS:
        config.stc.add_tensor_files(os.path.join(ENGRAM_DIR, filename))

    # The base-model Engram shards also contain BF16 q/k norm weights. The quant
    # intentionally ships FP32 versions; restore those mappings after adding tables.
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as handle:
        weight_map = json.load(handle)["weight_map"]
    for layer in (1, 14):
        for name in ("q_weight", "k_weight"):
            key = f"layers.{layer}.engram.{name}"
            config.stc.tensor_file_map[key] = os.path.join(MODEL_DIR, weight_map[key])
    return config


def main():
    rank = int(os.environ["EXL3_TP_NODE_RANK"])
    world_size = int(os.environ.get("EXL3_TP_WORLD_SIZE", "2"))
    config = load_config_with_engram()
    model = Model.from_config(config, component="text")

    print(f"rank {rank}: starting full DS4.1 TP load", flush=True)
    started = time.time()
    model.load(
        tensor_p=True,
        tp_backend="nccl",
        use_per_device=[110.0] * world_size,
        max_chunk_size=2048,
        progressbar=rank == 0,
    )
    print(f"rank {rank}: DS4.1 TP load completed in {time.time() - started:.1f}s", flush=True)

    generator = torch.Generator().manual_seed(1017)
    input_ids = torch.randint(
        0, config.vocab_size, (1, 8), generator=generator, dtype=torch.long
    )
    started = time.time()
    with torch.inference_mode():
        logits = model.forward(
            input_ids,
            params={"attn_mode": "flash_attn_nc", "last_tokens_only": 1},
        )

    if rank == 0:
        assert logits is not None
        assert torch.isfinite(logits).all(), "Non-finite logits in DS4.1 forward"
        top_ids = logits[0, 0].topk(5).indices.tolist()
        print(
            f"rank 0: DS4.1 forward completed in {time.time() - started:.2f}s; "
            f"shape={tuple(logits.shape)}; top5={top_ids}",
            flush=True,
        )
    else:
        assert logits is None

    model.unload()
    print(f"rank {rank}: DS4.1 unload completed", flush=True)
    if rank == 0:
        print("Full DS4.1 two-node load/forward/unload: PASS", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
