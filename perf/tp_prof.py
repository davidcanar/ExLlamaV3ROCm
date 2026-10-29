#!/usr/bin/env python3
"""Rank-0 profiling driver for the two-node DS4.1 TP stack (rank 1 runs the stock tools/ds41/tp_rank1.py).

Loads rank 0 exactly like tp_serve.py, then for each context length N and mode, ONE request:
  - the first PROF_WARM steps/rounds run unperturbed (per-op wall times recorded),
  - the next PROF_STEPS run with synchronized nested timing (prof_hooks) for a per-op breakdown.
spec mode = DSpark rounds (draft / verify / settle), plain = one forward per token.

env: PROF_CTX="2000,24000"  PROF_WARM=24  PROF_STEPS=6  PROF_MODES="spec,plain"  PROF_TEMP=0
     PROF_OUT=$EXL3DS4_ROOT/perf/prof.jsonl  PROF_HOOKS=1 (0: no instrumentation at all)
"""
from __future__ import annotations

import glob
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("PROF_DS41", os.path.join(ROOT, "tools", "ds41")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prof_hooks  # noqa: E402
from prof_hooks import PROF  # noqa: E402

import ds41_chat  # noqa: E402
import ds41_engine  # noqa: E402
import tp_engine  # noqa: E402

MODEL = os.environ.get("EXL3_TP_MODEL_DIR", tp_engine.DEFAULT_MODEL_DIR)
CTXS = [int(x) for x in os.environ.get("PROF_CTX", "2000,24000").split(",")]
WARM = int(os.environ.get("PROF_WARM", "24"))
PSTEPS = int(os.environ.get("PROF_STEPS", "6"))
MODES = os.environ.get("PROF_MODES", "spec,plain").split(",")
TEMP = float(os.environ.get("PROF_TEMP", "0"))
HOOKS = os.environ.get("PROF_HOOKS", "1") != "0"
OUT = os.path.expanduser(os.environ.get("PROF_OUT", os.path.join(ROOT, "perf", "prof.jsonl")))
CORPUS = os.path.join(ROOT, "src", "exllamav3-dsv41f", "exllamav3")
QUESTION = "Briefly explain what the code above does, file by file. Be concise."


def log(*a, **k):
    print(*a, flush=True)


def corpus_text(chars):
    parts, n = [], 0
    for f in sorted(glob.glob(os.path.join(CORPUS, "**", "*.py"), recursive=True)):
        try:
            txt = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        parts.append(f"### File: {os.path.relpath(f, CORPUS)}\n{txt}\n")
        n += len(parts[-1])
        if n >= chars:
            break
    return "".join(parts)


def build_ids(tk, n_tokens):
    text = corpus_text(n_tokens * 6)
    body = tk.encode(text)[: max(16, n_tokens - 64)]
    code = tk._tk.decode(body, skip_special_tokens=False)
    prompt = ds41_chat.render_chat([{"role": "user", "content": code + "\n\n" + QUESTION}],
                                   thinking=True, reasoning_effort="high")
    return tk.encode(prompt)


def stats(v):
    if not v:
        return "n=0"
    vv = sorted(v)
    return f"n={len(v):3d} mean {sum(v) / len(v) * 1e3:8.2f} ms  median {vv[len(vv) // 2] * 1e3:8.2f} ms"


def main():
    import torch
    if HOOKS:
        prof_hooks.install()
    backend = tp_engine.TPModelBackend.load(rank=0, world_size=2, model_dir=MODEL, log=log)
    channel = tp_engine.ControlChannel.listen(os.environ.get("EXLLAMA_MASTER_ADDR", "10.0.2.1"),
                                              tp_engine.env_int("EXL3_TP_CONTROL_PORT", 29651),
                                              accept_timeout=1800.0, log=log)
    channel.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": backend.vocab_size,
                  "dspark": bool(backend.dspark)})
    reply = channel.recv()
    assert reply.get("op") == "hello_ok", reply
    tk = ds41_engine.HFTokenizer(Path(MODEL) / "tokenizer.json")
    engine = tp_engine.TPEngine(backend, channel, tk, ctx=backend.cache_tokens - 2, eos_id=None)
    log(f"prof: warm-up {engine.warmup():.1f} s")

    times: dict[str, list] = {}
    ctl = {"round": 0, "prof_from": None, "mode": "plain"}

    def wrap(name, counts_round):
        fn = getattr(backend, name)

        def w(*a, **k):
            if counts_round:
                ctl["round"] += 1
            on = ctl["prof_from"] is not None and ctl["round"] >= ctl["prof_from"]
            PROF.on = on and HOOKS
            t0 = time.perf_counter()
            try:
                r = fn(*a, **k)
                torch.cuda.synchronize()
            finally:
                PROF.on = False
            dt = time.perf_counter() - t0
            if not on and name != "prefill":
                times.setdefault(name, []).append(dt)
            elif name == "prefill":
                times.setdefault("prefill", []).append(dt)
            return r
        setattr(backend, name, w)

    wrap("draft", True)          # spec: one draft per round (first op of the round)
    wrap("forward", True)        # plain: one forward per token
    wrap("verify", False)
    wrap("refresh_draft_kv", False)
    wrap("rewind", False)
    wrap("prefill", False)

    for n in CTXS:
        ids = build_ids(tk, n)
        for mode_spec in MODES:
            # "spec" / "plain", optionally "spec:<temperature>" (sampled requests)
            mode, _, temp_s = mode_spec.partition(":")
            temp = float(temp_s) if temp_s else TEMP
            times.clear()
            PROF.reset()
            ctl["round"] = 0
            ctl["prof_from"] = WARM + 1
            sampler = tp_engine.TopPSampler(temp, 1.0, None)
            pieces = []
            emit = pieces.append
            if mode == "spec":
                max_tokens = 6 * (WARM + PSTEPS) + 8
                extra = (sampler,) if "sampler" in inspect.signature(engine._run_spec).parameters else ()
                t0 = time.perf_counter()
                st = engine._run_spec(ids, max_tokens, [], None, emit, None, None, *extra)
            else:
                max_tokens = WARM + PSTEPS
                t0 = time.perf_counter()
                st = engine._run(ids, max_tokens, sampler, None, [], None, emit)
            wall = time.perf_counter() - t0
            engine.requests += 1
            rounds = ctl["round"]
            psteps = max(0, rounds - WARM)
            log(f"\nprof: ctx {len(ids)} {mode} T={temp}: {st['new_tokens']} tokens, {rounds} "
                f"{'rounds' if mode == 'spec' else 'steps'}; prefill {times.get('prefill', [0])[0]:.1f} s; "
                f"generate {st['time_generate']:.2f} s")
            for k in ("draft", "verify", "refresh_draft_kv", "rewind", "forward"):
                if times.get(k):
                    log(f"   {k:<18} {stats(times[k])}")
            if mode == "spec" and times.get("verify"):
                per_round = sum(sum(times.get(k, [])) for k in ("draft", "verify", "refresh_draft_kv", "rewind"))
                nr = len(times["verify"])
                log(f"   unprofiled rounds: {per_round / max(nr, 1) * 1e3:.1f} ms/round (model ops only), "
                    f"accepted/round {st.get('accepted_per_step', 0):.2f}")
            if HOOKS and psteps:
                log(PROF.report(psteps, f"ctx {len(ids)} {mode} profiled"))
            text = "".join(pieces)
            import hashlib
            log(f"   output sha1 {hashlib.sha1(text.encode()).hexdigest()[:16]}  {text[:160]!r}")
            rec = {"ctx": len(ids), "mode": mode, "temp": temp, "wall": wall, "stats": st, "text": text,
                   "times_ms": {k: [round(x * 1e3, 3) for x in v] for k, v in times.items()},
                   "prof_steps": psteps, "prof": {k: list(v) for k, v in PROF.stats.items()},
                   "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "NCCL_", "PROF_"))}}
            with open(OUT, "a") as h:
                h.write(json.dumps(rec) + "\n")

    engine.close()
    backend.unload()
    log("prof: done")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
