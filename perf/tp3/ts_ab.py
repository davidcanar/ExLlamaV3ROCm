#!/usr/bin/env python3
"""A/B driver for the three-node DS4.1 TP stack (rank 0; ranks 1/2 run the stock tp3_worker.py).

tp3_prof.py ported to the current engine API: it drove TPEngine._run/_run_spec, which the
interleave refactor (proto 6) replaced with per-job step generators. This driver goes through
TPEngine.generate, the server's own entry point, and times the backend calls the same way
(one device sync per call). Per context and mode it appends one record to PROF_OUT:
  plain: per-forward wall times; the first forward's logits (the first generated token) are
         saved to PROF_LOGITS_DIR/<tag>-ctx<N>.pt and, with PROF_LOGITS_REF=<other tag>,
         compared with that run's file (KL, top-1/top-5 agreement): the numerics half of an A/B
  spec:  per-round draft/verify wall times and every verify's row count
  both:  the steps/rounds after the first PROF_WARM run under prof_hooks (per-op breakdown)

  PROF_DRIVER=ts_ab.py PROF_TAG=<tag> perf/tp3/run_prof3.sh <tag>     (tp3 server stopped)

env: PROF_CTX="2000,24000"  PROF_WARM=24  PROF_STEPS=6  PROF_MODES="spec,plain"  PROF_TEMP=0
     PROF_OUT (default perf/tp3/ab.jsonl)  PROF_HOOKS=1  PROF_LOGITS_DIR (default
     perf/tp3/ab-logits)  PROF_LOGITS_REF (tag of the reference run, optional)
Summary: python3 perf/tp3/ab_summary.py perf/tp3/ab.jsonl [tags...]
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp3_prof  # noqa: E402  (sys.path setup, prof_hooks, build_ids, connect_workers, the PROF_* knobs)
from tp3_prof import PROF, build_ids, connect_workers, ds41_engine, log, prof_hooks, tp_engine  # noqa: E402

OUT = os.path.expanduser(os.environ.get("PROF_OUT", os.path.join(tp3_prof.ROOT, "perf", "tp3", "ab.jsonl")))
LOGITS_DIR = os.path.expanduser(os.environ.get(
    "PROF_LOGITS_DIR", os.path.join(tp3_prof.ROOT, "perf", "tp3", "ab-logits")))
LOGITS_REF = os.environ.get("PROF_LOGITS_REF", "")


def compare_logits(ref, cur) -> dict:
    import torch
    lp = torch.log_softmax(ref.double(), -1)
    lq = torch.log_softmax(cur.double(), -1)
    kl = float((lp.exp() * (lp - lq)).sum())
    t5r = set(torch.topk(ref, 5).indices.tolist())
    t5c = set(torch.topk(cur, 5).indices.tolist())
    return {"kl": kl, "top1_same": int(ref.argmax()) == int(cur.argmax()),
            "top5_overlap": len(t5r & t5c), "max_abs": float((ref - cur).abs().max())}


def main():
    import torch
    world = tp_engine.env_int("EXL3_TP_WORLD_SIZE", 3)
    if tp3_prof.HOOKS:
        prof_hooks.install()
    backend = tp_engine.TPModelBackend.load(rank=0, world_size=world, model_dir=tp3_prof.MODEL, log=log)
    channel = connect_workers(backend, world)
    tk = ds41_engine.HFTokenizer(Path(tp3_prof.MODEL) / "tokenizer.json")
    engine = tp_engine.TPEngine(backend, channel, tk, ctx=backend.cache_tokens - 2, eos_id=None, log=log)
    log(f"ab: warm-up {engine.warmup():.1f} s")
    free, total = torch.cuda.mem_get_info()
    free_gib = free / 2**30
    log(f"ab: rank 0 after warm-up: {free_gib:.2f} GiB free of {total / 2**30:.2f} GiB")
    env = {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "NCCL_", "PROF_", "ODL"))}

    times: dict[str, list] = {}
    rows: list = []
    ctl = {"round": 0, "prof_from": None, "want_logits": False, "logits": None}

    def wrap(name, counts_round):
        if not hasattr(backend, name):
            log(f"ab: backend has no {name}(), not timed")
            return
        fn = getattr(backend, name)

        def w(*a, **k):
            if counts_round:
                ctl["round"] += 1
            on = name != "prefill" and ctl["prof_from"] is not None and ctl["round"] >= ctl["prof_from"]
            PROF.on = on and tp3_prof.HOOKS
            t0 = time.perf_counter()
            try:
                r = fn(*a, **k)
                torch.cuda.synchronize()
            finally:
                PROF.on = False
            dt = time.perf_counter() - t0
            if not on:
                times.setdefault(name, []).append(dt)
                if name == "verify" and a:
                    rows.append((len(a[0]), round(dt * 1e3, 3)))
            if name == "forward" and ctl["want_logits"] and r is not None:
                ctl["logits"] = r.detach().float().cpu().clone()
                ctl["want_logits"] = False
            return r
        setattr(backend, name, w)

    wrap("draft", True)          # spec: one draft per round (first op of the round)
    wrap("forward", True)        # plain: one forward per token
    wrap("verify", False)
    wrap("refresh_draft_kv", False)
    wrap("rewind", False)
    wrap("prefill", False)

    async def run_one(ids, max_tokens, spec, temp, st):
        pieces = []
        async for piece in engine.generate("", max_tokens=max_tokens, temperature=temp, top_p=1.0, stop=[],
                                           prompt_ids=ids, ignore_eos=True, cache_prompt=False,
                                           speculative=spec, stats=st):
            pieces.append(piece)
        return "".join(pieces)

    warm, psteps = tp3_prof.WARM, tp3_prof.PSTEPS
    try:
        for n in tp3_prof.CTXS:
            ids = build_ids(tk, n)
            for mode_spec in tp3_prof.MODES:
                mode, _, temp_s = mode_spec.partition(":")
                temp = float(temp_s) if temp_s else tp3_prof.TEMP
                spec = mode == "spec"
                times.clear()
                rows.clear()
                PROF.reset()
                ctl.update(round=0, prof_from=warm + 1, want_logits=(mode == "plain"), logits=None)
                max_tokens = 4 * (warm + psteps) + 8 if spec else warm + psteps
                st: dict = {}
                t0 = time.perf_counter()
                text = asyncio.run(run_one(ids, max_tokens, spec, temp, st))
                wall = time.perf_counter() - t0
                nrounds = ctl["round"]
                prof_steps = max(0, nrounds - warm)
                pf_s = sum(times.get("prefill", [0.0]))
                log(f"\nab: ctx {len(ids)} {mode} T={temp}: {st.get('new_tokens')} tokens, {nrounds} "
                    f"{'rounds' if spec else 'steps'}; prefill {pf_s:.2f} s "
                    f"({(len(ids) - 1) / max(pf_s, 1e-9):.1f} tok/s); generate {st.get('time_generate', 0):.2f} s "
                    f"({st.get('new_tokens', 0) / max(st.get('time_generate', 0), 1e-9):.2f} tok/s)"
                    + (f", accepted/round {st.get('accepted_per_step', 0):.2f}" if spec else ""))
                for k in ("draft", "verify", "refresh_draft_kv", "rewind", "forward"):
                    if times.get(k):
                        log(f"   {k:<18} {tp3_prof.stats(times[k])}")
                if tp3_prof.HOOKS and prof_steps:
                    log(PROF.report(prof_steps, f"ctx {len(ids)} {mode} profiled"))
                log(f"   output sha1 {hashlib.sha1(text.encode()).hexdigest()[:16]}  {text[:160]!r}")
                rec = {"tag": tp3_prof.TAG, "ctx": len(ids), "mode": mode, "temp": temp, "wall": wall,
                       "stats": st, "text": text, "prefill_s": pf_s, "rounds": nrounds,
                       "times_ms": {k: [round(x * 1e3, 3) for x in v] for k, v in times.items()},
                       "verify_rows": list(rows), "prof_steps": prof_steps,
                       "prof": {k: list(v) for k, v in PROF.stats.items()},
                       "mem_free_gib_rank0": round(free_gib, 3), "env": env}
                if ctl["logits"] is not None:
                    os.makedirs(LOGITS_DIR, exist_ok=True)
                    torch.save(ctl["logits"], os.path.join(LOGITS_DIR, f"{tp3_prof.TAG}-ctx{len(ids)}.pt"))
                    ref_path = os.path.join(LOGITS_DIR, f"{LOGITS_REF}-ctx{len(ids)}.pt") if LOGITS_REF else ""
                    if ref_path and os.path.exists(ref_path):
                        cmp = compare_logits(torch.load(ref_path), ctl["logits"])
                        rec["logits_vs_ref"] = {"ref": LOGITS_REF, **cmp}
                        log(f"   first-token logits vs {LOGITS_REF}: {cmp}")
                with open(OUT, "a") as h:
                    h.write(json.dumps(rec) + "\n")
    finally:
        engine.close()
        backend.unload()
    log("ab: done")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
