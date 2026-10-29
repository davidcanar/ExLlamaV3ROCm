#!/usr/bin/env python3
"""Decode-speed vs context-length benchmark against the running exl3ds4 OpenAI server.

Builds prompts of roughly N tokens from the exllamav3 source tree (realistic code context),
asks a question about it, and records prefill / decode timings per (mode, temperature).
Modes: spec (speculative: true), plain (speculative: false), default (server decides).
"""
import argparse
import glob
import json
import os
import time
import urllib.request

URL = os.environ.get("EXL3_BENCH_URL", "http://127.0.0.1:1234/v1/chat/completions")
ROOT = os.environ.get("EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def corpus(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as h:
                yield f, h.read()
        except OSError:
            pass


def build_prompt(root, approx_tokens, chars_per_token=4.0):
    budget = int(approx_tokens * chars_per_token)
    parts, n = [], 0
    for f, txt in corpus(root):
        chunk = f"### File: {os.path.relpath(f, root)}\n{txt}\n"
        if n + len(chunk) > budget:
            chunk = chunk[: budget - n]
        parts.append(chunk)
        n += len(chunk)
        if n >= budget:
            break
    return "".join(parts)


def run(prompt, question, max_tokens, speculative, temperature, timeout=3600):
    body = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt + "\n\n" + question}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if speculative is not None:
        body["speculative"] = speculative
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    wall = time.perf_counter() - t0
    return out, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "src", "exllamav3-dsv41f", "exllamav3"))
    ap.add_argument("--ctx", default="1000,16000,32000")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--modes", default="spec,plain")
    ap.add_argument("--temps", default="0")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "bench_ctx.jsonl"))
    ap.add_argument("--question", default="Briefly explain what the code above does, file by file. Be concise.")
    args = ap.parse_args()

    for ctx in [int(x) for x in args.ctx.split(",")]:
        prompt = build_prompt(args.root, ctx)
        for temp in [float(t) for t in args.temps.split(",")]:
            for mode in args.modes.split(","):
                spec = {"spec": True, "plain": False, "default": None}[mode]
                try:
                    out, wall = run(prompt, args.question, args.max_tokens, spec, temp)
                except Exception as e:  # noqa: BLE001
                    print(f"ctx~{ctx} {mode} T={temp}: FAILED {type(e).__name__}: {e}", flush=True)
                    continue
                t = out.get("timings", {})
                u = out.get("usage", {})
                rec = {"ctx_target": ctx, "mode": mode, "temp": temp, "wall": wall, "usage": u, "timings": t}
                print(json.dumps(rec), flush=True)
                with open(args.out, "a") as h:
                    h.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
