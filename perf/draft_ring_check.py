#!/usr/bin/env python3
"""Rank-0 check of the DSpark drafter's main-kv ring (rank 1 runs the stock tools/ds41/tp_rank1.py).

  PROF_DRIVER=draft_ring_check.py perf/run_prof.sh ring      # the exl3ds4 server must be stopped

After a spec request whose prefill wrapped the ring several times, the draft at the prompt's
anchor position is taken three times: as is, with its window rows overwritten by noise, and with
the window put back from a draft_window() copy. The first and last must match exactly (and the
noisy one should not), which shows draft_window()/set_draft_window() address exactly the rows the
drafter reads through the modular block table.

env: RING_CTX=6000 (prompt tokens; > 2 ring lengths at the default 10-page ring)
"""
from __future__ import annotations

import asyncio
import glob
import os
import sys
from pathlib import Path

ROOT = os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))

import ds41_chat  # noqa: E402
import ds41_engine  # noqa: E402
import tp_engine  # noqa: E402

MODEL = os.environ.get("EXL3_TP_MODEL_DIR", tp_engine.DEFAULT_MODEL_DIR)
CTX = int(os.environ.get("RING_CTX", "6000"))
CORPUS = os.path.join(ROOT, "src", "exllamav3-dsv41f", "exllamav3")


def log(*a, **k):
    print(*a, flush=True)


def main():
    import torch
    backend = tp_engine.TPModelBackend.load(rank=0, world_size=2, model_dir=MODEL, log=log)
    channel = tp_engine.ControlChannel.listen(os.environ.get("EXLLAMA_MASTER_ADDR", "10.0.2.1"),
                                              tp_engine.env_int("EXL3_TP_CONTROL_PORT", 29651),
                                              accept_timeout=1800.0, log=log)
    channel.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": backend.vocab_size,
                  "dspark": bool(backend.dspark)})
    assert channel.recv().get("op") == "hello_ok"
    tk = ds41_engine.HFTokenizer(Path(MODEL) / "tokenizer.json")
    engine = tp_engine.TPEngine(backend, channel, tk, ctx=backend.cache_tokens - 2, eos_id=None, log=log)
    ring_rows = next(iter(backend.draft_cache.layers.values())).kv.shape[0] * backend.page_size
    log(f"ring: drafter main-kv {ring_rows} rows, window {backend.draft_window_rows}, "
        f"block table {tuple(backend.draft_bt.shape)} -> pages 0..{int(backend.draft_bt.max())}")

    parts, n = [], 0
    for f in sorted(glob.glob(os.path.join(CORPUS, "**", "*.py"), recursive=True)):
        parts.append(f"### File: {os.path.relpath(f, CORPUS)}\n{open(f, encoding='utf-8', errors='replace').read()}\n")
        n += len(parts[-1])
        if n > CTX * 6:
            break
    body = tk._tk.decode(tk.encode("".join(parts))[:CTX], skip_special_tokens=False)
    ids = tk.encode(ds41_chat.render_chat([{"role": "user", "content": body + "\n\nSummarize the code above."}],
                                          thinking=False))
    stats: dict = {}

    async def go():
        async for _ in engine.generate("", prompt_ids=ids, max_tokens=24, temperature=0.0, top_p=1.0,
                                       stop=[], stats=stats, speculative=True, cache_prompt=False,
                                       ignore_eos=True):
            pass
    asyncio.run(go())
    p = len(ids) - 1
    log(f"ring: prompt {len(ids)} tokens ({len(ids) / ring_rows:.1f} ring lengths), "
        f"accepted/step {stats.get('accepted_per_step', 0):.2f}")

    # The 24 decoded tokens wrote rows p..p+23 only: the window [p - 128, p) is still the prefill's
    with torch.inference_mode():
        k1, d1 = backend.draft(ids[-1], p, 5)
        saved = backend.draft_window(p)
        backend.set_draft_window([torch.randn_like(t.float()).half() * 4 for t in saved], p)
        k2, d2 = backend.draft(ids[-1], p, 5)
        backend.set_draft_window(saved, p)
        k3, d3 = backend.draft(ids[-1], p, 5)
        back = backend.draft_window(p)
    same_rows = all(torch.equal(a, b) for a, b in zip(saved, back))
    log(f"ring: draft as is k={k1} {d1}; noisy window k={k2} {d2}; restored k={k3} {d3}")
    ok = (k1, d1) == (k3, d3) and same_rows
    log(f"ring: window rows round-trip {'exactly' if same_rows else 'NOT exactly'}; noisy window "
        f"{'changed' if (k1, d1) != (k2, d2) else 'did NOT change'} the draft; {'PASS' if ok else 'CHECK'}")
    engine.close()
    backend.unload()
    log("ring: done")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
