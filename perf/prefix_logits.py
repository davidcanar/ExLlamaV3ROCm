#!/usr/bin/env python3
"""Rank-0 logit check of the TP engine's prefix cache (rank 1 runs the stock tools/ds41/tp_rank1.py).

  PROF_DRIVER=prefix_logits.py perf/run_prof.sh pfx      # the exl3ds4 server must be stopped

Greedy replies on this stack are not bit-reproducible between runs (float reductions reorder), so
they cannot be compared as text. This compares the logits of the first generated token instead,
for a two-turn conversation in plain and spec mode:
  - the same turn-1 prompt prefilled from scratch twice: the run-to-run noise floor,
  - turn 2 through a restored checkpoint + partial prefill (PFX_ADD new tokens, so the cached
    prefill crosses chunk boundaries) vs turn 2 prefilled from scratch (twice),
  - turn 2 again through the checkpoint taken at its own prefill end (nothing left to prefill).
A correct restore lands inside the noise floor; a wrong ring / compressor / Engram state or a
stale pool row shows up as a large max-abs difference and a changed top-1.

env: PFX_CTX=12000 (turn-1 tokens)  PFX_ADD=3000  PFX_MODES=plain,spec
     PFX_OUT=$EXL3DS4_ROOT/perf/prefix_logits.jsonl
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))

import ds41_chat  # noqa: E402
import ds41_engine  # noqa: E402
import tp_engine  # noqa: E402

MODEL = os.environ.get("EXL3_TP_MODEL_DIR", tp_engine.DEFAULT_MODEL_DIR)
CTX = int(os.environ.get("PFX_CTX", "12000"))
ADD = int(os.environ.get("PFX_ADD", "3000"))
MODES = os.environ.get("PFX_MODES", "plain,spec").split(",")
ALIGN = os.environ.get("PFX_ALIGN", "0") != "0"
OUT = os.path.expanduser(os.environ.get("PFX_OUT", os.path.join(ROOT, "perf", "prefix_logits.jsonl")))
CORPUS = os.path.join(ROOT, "src", "exllamav3-dsv41f", "exllamav3")


def log(*a, **k):
    print(*a, flush=True)


def corpus_tokens(tk, skip_chars, n_tokens):
    parts, n = [], 0
    for f in sorted(glob.glob(os.path.join(CORPUS, "**", "*.py"), recursive=True)):
        try:
            txt = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        parts.append(f"### File: {os.path.relpath(f, CORPUS)}\n{txt}\n")
        n += len(parts[-1])
        if n >= skip_chars + n_tokens * 6:
            break
    text = "".join(parts)[skip_chars:]
    return tk._tk.decode(tk.encode(text)[:n_tokens], skip_special_tokens=False)


def compare(a, b) -> dict:
    import torch
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = float((la.exp() * (la - lb)).sum())
    t5a, t5b = set(a.topk(5).indices.tolist()), set(b.topk(5).indices.tolist())
    return {"max_abs": float((a - b).abs().max()), "top1_same": int(a.argmax()) == int(b.argmax()),
            "top5_overlap": len(t5a & t5b), "kl": kl}


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
    assert engine.prefix is not None, "prefix cache is off (EXL3_TP_PREFIX_CACHE)"
    log(f"pfx: warm-up {engine.warmup():.1f} s")

    cap = {"row": None, "drafts": None}
    fwd, ver, drf = backend.forward, backend.verify, backend.draft

    def forward(*a, **k):
        row = fwd(*a, **k)
        if cap["row"] is None and row is not None:
            cap["row"] = row.float().cpu().clone()
        return row

    def verify(*a, **k):
        logits, params = ver(*a, **k)
        if cap["row"] is None and logits is not None:
            cap["row"] = logits[0].float().cpu().clone()
        return logits, params

    def draft(*a, **k):
        k_, d = drf(*a, **k)
        if cap["drafts"] is None:
            cap["drafts"] = list(d)
        return k_, d

    backend.forward, backend.verify, backend.draft = forward, verify, draft

    ops: dict[str, list] = {}
    for name in ("stash", "restore", "drop_stash"):
        def timed(*a, _fn=getattr(backend, name), _name=name, **k):
            t0 = time.perf_counter()
            r = _fn(*a, **k)
            ops.setdefault(_name, []).append(round((time.perf_counter() - t0) * 1e3, 1))
            return r
        setattr(backend, name, timed)

    def first(ids, spec, cache):
        cap["row"] = cap["drafts"] = None
        stats: dict = {}

        async def go():
            async for _ in engine.generate("", prompt_ids=ids, max_tokens=6 if spec else 1, temperature=0.0,
                                           top_p=1.0, stop=[], stats=stats, speculative=spec,
                                           cache_prompt=cache, ignore_eos=True):
                pass
        t0 = time.perf_counter()
        asyncio.run(go())
        return cap["row"], cap["drafts"], stats, time.perf_counter() - t0

    context = corpus_tokens(tk, 0, CTX)
    extra = corpus_tokens(tk, 400_000, ADD)
    q1 = "Briefly explain what the code above does, file by file. Be concise."
    q2 = "Here is more code:\n\n" + extra + "\n\nHow does it relate to the first part? One sentence."
    turn1 = [{"role": "user", "content": context + "\n\n" + q1}]
    turn2 = turn1 + [{"role": "assistant", "content": "It is the exllamav3 library: model, cache and kernels."},
                     {"role": "user", "content": q2}]
    ids_a = tk.encode(ds41_chat.render_chat(turn1, thinking=True, reasoning_effort="high"))
    ids_b = tk.encode(ds41_chat.render_chat(turn2, thinking=True, reasoning_effort="high"))
    if ALIGN:
        # Put the turn-1 checkpoint (len(ids_a) - 1) on a prefill chunk boundary: the cached
        # turn-2 prefill then runs exactly the chunks a fresh prefill runs from there on, so an
        # exact restore must reproduce the fresh logits bit for bit (unaligned, the different
        # chunk boundaries reorder float sums and the logits differ by rounding alone)
        span = backend.span
        tail = 64                                        # the question + generation header, with margin
        cut = (len(ids_a) - 1) // span * span            # the new len(ids_a) - 1
        rest_b = ids_b[len(ids_a) - 1:]                  # turn 2 after the shared <｜Assistant｜>
        ids_a = ids_a[: cut + 1 - tail] + ids_a[-tail:]
        ids_b = ids_a[:-1] + rest_b
        assert (len(ids_a) - 1) % span == 0
    shared = 0
    while shared < min(len(ids_a), len(ids_b)) and ids_a[shared] == ids_b[shared]:
        shared += 1
    log(f"pfx: turn 1 {len(ids_a)} tokens, turn 2 {len(ids_b)} tokens, shared prefix {shared}")

    ok = True
    for mode in MODES:
        spec = mode == "spec"
        a1, _, _, _ = first(ids_a, spec, False)
        a2, _, _, _ = first(ids_a, spec, False)
        _, _, st_a, t_a = first(ids_a, spec, True)                    # takes the checkpoints
        bc, dc, st_bc, t_bc = first(ids_b, spec, True)                # restore + partial prefill
        b1, d1, _, t_b1 = first(ids_b, spec, False)
        b2, d2, _, _ = first(ids_b, spec, False)
        br, dr, st_br, t_br = first(ids_b, spec, True)                # restore at turn 2's own end
        rec = {"mode": mode, "turn1": len(ids_a), "turn2": len(ids_b), "shared": shared,
               "cached_turn2": st_bc.get("cached_tokens"), "cached_retry": st_br.get("cached_tokens"),
               "wall_turn2_cached_s": round(t_bc, 2), "wall_turn2_fresh_s": round(t_b1, 2),
               "wall_retry_s": round(t_br, 2), "wall_turn1_s": round(t_a, 2),
               "noise_turn1": compare(a1, a2), "noise_turn2": compare(b1, b2),
               "cached_vs_fresh": [compare(bc, b1), compare(bc, b2)],
               "retry_vs_fresh": [compare(br, b1), compare(br, b2)],
               "drafts": {"cached": dc, "fresh": [d1, d2], "retry": dr} if spec else None,
               "op_ms": dict(ops),
               "checkpoint_bytes": [c.stash.get("checkpoint_size") for c in engine.prefix.ckpts]}
        ops.clear()
        noise = max(rec["noise_turn1"]["max_abs"], rec["noise_turn2"]["max_abs"])
        worst = max(c["max_abs"] for c in rec["cached_vs_fresh"] + rec["retry_vs_fresh"])
        good = (rec["cached_turn2"] == len(ids_a) - 1 and rec["cached_retry"] == len(ids_b) - 1
                and all(c["top1_same"] for c in rec["cached_vs_fresh"] + rec["retry_vs_fresh"])
                and worst <= max(4 * noise, 0.05))
        rec["verdict"] = "OK" if good else "CHECK"
        ok &= good
        log(f"pfx: {mode}: reused {rec['cached_turn2']} / retry {rec['cached_retry']}; turn 2 "
            f"{rec['wall_turn2_cached_s']} s cached vs {rec['wall_turn2_fresh_s']} s fresh; max|dlogit| "
            f"noise {noise:.4f}, cached {worst:.4f}; {rec['verdict']}")
        log("pfx: " + json.dumps(rec))
        with open(OUT, "a") as h:
            h.write(json.dumps(rec) + "\n")
    log(f"pfx: {'ALL OK' if ok else 'CHECK THE RECORDS'}")
    engine.close()
    backend.unload()
    log("pfx: done")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
