#!/usr/bin/env python3
"""Rank-0 profiling driver for the THREE-node DS4.1 TP stack (ranks 1/2 run the stock tp3_worker.py).

perf/tp_prof.py adapted to world size N: the rank-1 ControlChannel becomes a
tp3_channel.MultiControlChannel over N-1 workers. Same measurements:
  - prefill wall time per context (one chunked prefill per request),
  - the first PROF_WARM steps/rounds unperturbed (per-op wall, one sync per op),
  - the next PROF_STEPS with synchronized nested timing (prof_hooks) for a per-op breakdown.

env: PROF_CTX="2000,24000"  PROF_WARM=24  PROF_STEPS=6  PROF_MODES="spec,plain"  PROF_TEMP=0
     PROF_OUT=$EXL3DS4_ROOT/perf/tp3/prof3.jsonl  PROF_HOOKS=1 (0: no instrumentation)
     PROF_TAG (label stored in the jsonl record)

main() below drives TPEngine._run/_run_spec, which the interleaved engine (proto 6) no longer
has: run ts_ab.py (run_prof3.sh's default driver), which imports the helpers of this module and
measures the same things through TPEngine.generate.
"""
from __future__ import annotations

import glob
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))
sys.path.insert(0, os.path.join(ROOT, "perf"))          # prof_hooks

import prof_hooks  # noqa: E402
from prof_hooks import PROF  # noqa: E402

import ds41_chat  # noqa: E402
import ds41_engine  # noqa: E402
import tp_engine  # noqa: E402
import tp3_channel  # noqa: E402

MODEL = os.environ.get("EXL3_TP_MODEL_DIR", tp_engine.DEFAULT_MODEL_DIR)
CTXS = [int(x) for x in os.environ.get("PROF_CTX", "2000,24000").split(",")]
WARM = int(os.environ.get("PROF_WARM", "24"))
PSTEPS = int(os.environ.get("PROF_STEPS", "6"))
MODES = os.environ.get("PROF_MODES", "spec,plain").split(",")
TEMP = float(os.environ.get("PROF_TEMP", "0"))
HOOKS = os.environ.get("PROF_HOOKS", "1") != "0"
TAG = os.environ.get("PROF_TAG", "")
OUT = os.path.expanduser(os.environ.get("PROF_OUT", os.path.join(ROOT, "perf", "tp3", "prof3.jsonl")))
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


def connect_workers(backend, world):
    host = os.environ.get("EXLLAMA_MASTER_ADDR", "192.168.0.102")
    port = tp_engine.env_int("EXL3_TP_CONTROL_PORT", 29651)
    channel = tp3_channel.MultiControlChannel.listen_many(host, port, world - 1,
                                                          accept_timeout=1800.0, log=log)
    channel.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": backend.vocab_size,
                  "dspark": bool(backend.dspark), "vision": False})
    for i in range(world - 1):
        reply = channel.channels[i].recv()
        assert reply.get("op") == "hello_ok", reply
        assert bool(reply.get("dspark")) == bool(backend.dspark), reply
        assert bool(reply.get("moe_tensor_split", False)) == bool(getattr(backend, "moe_tensor_split", False)), \
            f"MoE layout differs from rank 0's: {reply}"
    return channel


def main():
    import torch
    world = tp_engine.env_int("EXL3_TP_WORLD_SIZE", 3)
    if HOOKS:
        prof_hooks.install()
    backend = tp_engine.TPModelBackend.load(rank=0, world_size=world, model_dir=MODEL, log=log)
    channel = connect_workers(backend, world)
    tk = ds41_engine.HFTokenizer(Path(MODEL) / "tokenizer.json")
    engine = tp_engine.TPEngine(backend, channel, tk, ctx=backend.cache_tokens - 2, eos_id=None)
    log(f"prof: warm-up {engine.warmup():.1f} s")

    times: dict[str, list] = {}
    ctl = {"round": 0, "prof_from": None}

    def wrap(name, counts_round):
        fn = getattr(backend, name)

        def w(*a, **k):
            if counts_round:
                ctl["round"] += 1
            on = ctl["prof_from"] is not None and ctl["round"] >= ctl["prof_from"]
            if name == "prefill":
                on = bool(ctl.get("prof_prefill"))
            PROF.on = on and HOOKS
            t0 = time.perf_counter()
            try:
                r = fn(*a, **k)
                th = time.perf_counter()          # host returned (work may still be queued)
                torch.cuda.synchronize()
            finally:
                PROF.on = False
            dt = time.perf_counter() - t0
            if name == "prefill" or not on:
                times.setdefault(name, []).append(dt)
                times.setdefault(name + "_host", []).append(th - t0)
            return r
        setattr(backend, name, w)

    wrap("draft", True)          # spec: one draft per round (first op of the round)
    wrap("forward", True)        # plain: one forward per token
    wrap("verify", False)
    wrap("refresh_draft_kv", False)
    wrap("rewind", False)
    wrap("prefill", False)

    plan = []
    if os.environ.get("PROF_PF_CTX"):            # one profiled prefill first, at its own context
        plan.append((int(os.environ["PROF_PF_CTX"]), ["pf"]))
    plan += [(n, MODES) for n in CTXS]
    for n, modes in plan:
        ids = build_ids(tk, n)
        for mode_spec in modes:
            mode, _, temp_s = mode_spec.partition(":")
            temp = float(temp_s) if temp_s else TEMP
            times.clear()
            PROF.reset()
            ctl["round"] = 0
            ctl["prof_from"] = WARM + 1
            sampler = tp_engine.TopPSampler(temp, 1.0, None)
            pieces = []
            emit = pieces.append
            use_cache = os.environ.get("PROF_PREFIX_CACHE", "0") != "0"
            ctl["prof_prefill"] = mode == "pf"
            if mode == "pf":
                # prefill-only request, profiled per op across every chunk
                kw = {"use_cache": use_cache} if "use_cache" in inspect.signature(engine._run).parameters else {}
                t0 = time.perf_counter()
                st = engine._run(ids, 1, sampler, None, [], None, emit, **kw)
                wall = time.perf_counter() - t0
                engine.requests += 1
                pf_s = sum(times.get("prefill", [0.0]))
                chunks = max(1, -(-(len(ids) - 1) // backend.span))
                log(f"\nprof: ctx {len(ids)} prefill-profile: {chunks} chunk(s) of {backend.span}; profiled "
                    f"prefill {pf_s:.2f} s ({(len(ids) - 1) / max(pf_s, 1e-9):.1f} tok/s incl. sync overhead)")
                if HOOKS:
                    log(PROF.report(chunks, f"ctx {len(ids)} prefill per chunk"))
                rec = {"tag": TAG, "ctx": len(ids), "mode": mode, "wall": wall, "prefill_s": pf_s, "chunks": chunks,
                       "prof": {k: list(v) for k, v in PROF.stats.items()},
                       "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "NCCL_", "PROF_", "ODL"))}}
                with open(OUT, "a") as h:
                    h.write(json.dumps(rec) + "\n")
                ctl["prof_prefill"] = False
                continue
            if mode == "spec":
                max_tokens = 6 * (WARM + PSTEPS) + 8
                kw = {"use_cache": use_cache} if "use_cache" in inspect.signature(engine._run_spec).parameters else {}
                t0 = time.perf_counter()
                st = engine._run_spec(ids, max_tokens, [], None, emit, None, None, sampler, **kw)
            else:
                max_tokens = WARM + PSTEPS
                kw = {"use_cache": use_cache} if "use_cache" in inspect.signature(engine._run).parameters else {}
                t0 = time.perf_counter()
                st = engine._run(ids, max_tokens, sampler, None, [], None, emit, **kw)
            wall = time.perf_counter() - t0
            engine.requests += 1
            rounds = ctl["round"]
            psteps = max(0, rounds - WARM)
            pf = times.get("prefill", [0.0])
            pf_s = sum(pf)
            log(f"\nprof: ctx {len(ids)} {mode} T={temp}: {st['new_tokens']} tokens, {rounds} "
                f"{'rounds' if mode == 'spec' else 'steps'}; prefill {pf_s:.2f} s "
                f"({(len(ids) - 1) / max(pf_s, 1e-9):.1f} tok/s); generate {st['time_generate']:.2f} s")
            for k in ("draft", "verify", "refresh_draft_kv", "rewind", "forward", "verify_host", "forward_host", "draft_host"):
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
            log(f"   output sha1 {hashlib.sha1(text.encode()).hexdigest()[:16]}  {text[:160]!r}")
            rec = {"tag": TAG, "ctx": len(ids), "mode": mode, "temp": temp, "wall": wall, "stats": st,
                   "text": text, "prefill_s": pf_s,
                   "times_ms": {k: [round(x * 1e3, 3) for x in v] for k, v in times.items()},
                   "prof_steps": psteps, "prof": {k: list(v) for k, v in PROF.stats.items()},
                   "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "NCCL_", "PROF_", "ODL"))}}
            with open(OUT, "a") as h:
                h.write(json.dumps(rec) + "\n")

    engine.close()
    backend.unload()
    log("prof: done")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
