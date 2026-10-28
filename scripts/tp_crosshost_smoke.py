#!/usr/bin/env python3
"""Symmetric two-node TP load/forward/unload test with a tiny real checkpoint."""

import hashlib
import multiprocessing
import os
import sys
import time

import torch

REPO = os.environ.get("EXL3_ROOT", os.path.expanduser("~/exl3ds4/src/exllamav3-dsv41f"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

from testlib.tiny_models import DSV4_TINY, make_dsv4_checkpoint
from testlib.dsv4 import noise_floor
from testlib.compare import confident_top1_agreement, kl_divergence
from exllamav3 import Config, Model


def main():
    rank = int(os.environ["EXL3_TP_NODE_RANK"])
    world_size = int(os.environ.get("EXL3_TP_WORLD_SIZE", "2"))
    tiny_dir = os.environ.get("EXL3_TP_TINY_DIR", "/tmp/exl3_tp_tiny_dsv4")
    sequence_length = 315
    seed = 7

    if not os.path.isfile(os.path.join(tiny_dir, "model.safetensors")):
        make_dsv4_checkpoint(tiny_dir, seed=seed)
    with open(os.path.join(tiny_dir, "model.safetensors"), "rb") as handle:
        checkpoint_sha = hashlib.sha256(handle.read()).hexdigest()
    print(f"rank {rank}: tiny checkpoint sha256 {checkpoint_sha}", flush=True)
    if "EXL3_TP_TINY_SHA" in os.environ:
        assert checkpoint_sha == os.environ["EXL3_TP_TINY_SHA"], "checkpoints differ between hosts"

    generator = torch.Generator().manual_seed(seed + 1)
    input_ids = torch.randint(
        0,
        DSV4_TINY["vocab_size"],
        (1, sequence_length),
        generator=generator,
        dtype=torch.long,
    )

    model = Model.from_config(Config.from_directory(tiny_dir))
    started = time.time()
    model.load(
        tensor_p=True,
        tp_backend="nccl",
        use_per_device=[1000.0] * world_size,
        max_chunk_size=sequence_length,
        progressbar=rank == 0,
    )
    print(f"rank {rank}: TP load completed in {time.time() - started:.1f}s", flush=True)

    started = time.time()
    with torch.inference_mode():
        logits = model.forward(input_ids, params={"attn_mode": "flash_attn_nc"})
    if rank == 0:
        tp_logits = logits[0].float().cpu()
        print(
            f"rank 0: forward completed in {time.time() - started:.2f}s; shape={tuple(tp_logits.shape)}",
            flush=True,
        )
    else:
        assert logits is None

    model.unload()
    print(f"rank {rank}: TP unload completed", flush=True)

    if rank == 0:
        reference_model = Model.from_config(Config.from_directory(tiny_dir))
        reference_model.load(torch.device("cuda:0"))
        with torch.inference_mode():
            reference_logits = reference_model.forward(
                input_ids, params={"attn_mode": "flash_attn_nc"}
            )[0].float().cpu()
            floor_kl, floor_argmax = noise_floor(reference_model, input_ids, reference_logits)
        reference_model.unload()

        kl = kl_divergence(reference_logits, tp_logits)
        agreement, count = confident_top1_agreement(reference_logits, tp_logits, margin=1.0)
        argmax_agreement = (tp_logits.argmax(-1) == reference_logits.argmax(-1)).float().mean().item()
        print(f"noise floor: KL={floor_kl:.3e}, argmax={floor_argmax:.5f}")
        print(
            f"TP vs single: KL median={kl.median():.3e} mean={kl.mean():.3e} max={kl.max():.3e}; "
            f"argmax={argmax_agreement:.5f}; confident top1={agreement:.5f} over {count}"
        )
        assert torch.isfinite(tp_logits).all(), "non-finite TP logits"
        assert kl.mean().item() < max(5e-3, 10 * floor_kl), "KL above noise-scaled bound"
        assert count == 0 or agreement >= 0.97, "confident top-1 agreement too low"
        print("Cross-host model load/forward/unload: PASS", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
