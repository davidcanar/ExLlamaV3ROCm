#!/usr/bin/env python3
"""Prefix-cache check against the running exl3ds4 server: multi-turn reuse and equivalence.

Turn 1 sends a long code-context prompt; each later turn appends the previous reply and a new
question (a real multi-turn conversation, rendered by the server's chat template). Every turn runs
twice, greedy: with the prefix cache (the server default) and with "cache_prompt": false (a full
prefill), and the replies are compared. Turn 1 is also run twice without reuse first, which shows
whether greedy decoding is bit-reproducible on this setup at all (split-K reductions and the
all-reduce may reorder float sums between runs).

  python perf/prefix_check.py --ctx 20000 --turns 3 --configs spec,plain,spec-nothink
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ctx import ROOT, URL, build_prompt  # noqa: E402

QUESTIONS = [
    "Briefly explain what the code above does, file by file. Be concise.",
    "Which of these files would you change to add a new cache layer type? Answer in two sentences.",
    "Name one function that looks performance critical and say why, in one sentence.",
    "Summarize our conversation so far in one sentence.",
]


def post(messages, *, max_tokens, speculative, thinking, cache_prompt, timeout=3600):
    body = {"model": "deepseek-v4.1-flash", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False, "speculative": speculative}
    if not thinking:
        body["chat_template_kwargs"] = {"thinking": False}
    if not cache_prompt:
        body["cache_prompt"] = False
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    wall = time.perf_counter() - t0
    msg = out["choices"][0]["message"]
    t = out.get("timings", {})
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "cache_n": int(t.get("cache_n", 0)), "prompt_tokens": int(out["usage"]["prompt_tokens"]),
            "prompt_ms": float(t.get("prompt_ms", 0.0)), "decode_tps": float(t.get("predicted_per_second", 0.0)),
            "wall": wall}


def text_of(r):
    return r["reasoning"] + "\x00" + r["content"]


def first_diff(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "src", "exllamav3-dsv41f", "exllamav3"))
    ap.add_argument("--ctx", type=int, default=20000)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--configs", default="spec,plain,spec-nothink")
    ap.add_argument("--out", default=os.path.join(ROOT, "perf", "prefix_check.jsonl"))
    args = ap.parse_args()

    context = build_prompt(args.root, args.ctx)
    ok_all = True
    for cfg in args.configs.split(","):
        spec = cfg.startswith("spec")
        thinking = not cfg.endswith("nothink")
        kw = dict(max_tokens=args.max_tokens, speculative=spec, thinking=thinking)
        messages = [{"role": "user", "content": context + "\n\n" + QUESTIONS[0]}]
        base = post(messages, cache_prompt=False, **kw)       # determinism baseline (no reuse)
        for turn in range(args.turns):
            got = post(messages, cache_prompt=True, **kw)
            ref = post(messages, cache_prompt=False, **kw) if turn else base
            d = first_diff(text_of(got), text_of(ref))
            rec = {"config": cfg, "turn": turn + 1, "prompt_tokens": got["prompt_tokens"],
                   "cache_n": got["cache_n"], "prefill_ms_cached": round(got["prompt_ms"]),
                   "prefill_ms_full": round(ref["prompt_ms"]), "decode_tps": round(got["decode_tps"], 2),
                   "match": d < 0, "first_diff_char": d, "len": len(text_of(got))}
            if turn == 0:
                rec["baseline_repeatable"] = first_diff(text_of(base), text_of(got)) < 0
            print(json.dumps(rec), flush=True)
            with open(args.out, "a") as h:
                h.write(json.dumps({**rec, "got": text_of(got), "ref": text_of(ref)}) + "\n")
            ok_all &= d < 0 or turn == 0
            messages = messages + [{"role": "assistant", "content": got["content"],
                                    **({"reasoning_content": got["reasoning"]} if got["reasoning"] else {})},
                                   {"role": "user", "content": QUESTIONS[(turn + 1) % len(QUESTIONS)]}]
    print("prefix_check:", "ALL MATCH" if ok_all else "MISMATCH (see first_diff_char)", flush=True)


if __name__ == "__main__":
    main()
