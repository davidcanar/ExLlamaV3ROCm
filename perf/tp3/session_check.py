#!/usr/bin/env python3
"""Multi-session cache check against the running server (TP=2 or TP=3, EXL3_TP_SESSIONS >= 2).

A1 long conversation, B1 a different one in between (single-session mode would evict A),
A2 follow-up of A -> must be served from A's cache; B2 likewise; C with thinking ON and the
client dropping reasoning_content (as most clients do); D1, E1 fill and overflow the 4 slots
so the least-recently-used conversation spills to NVMe; A3 returns to it -> cold reload.
usage: session_check.py [base url]   (default http://127.0.0.1:1234)
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1234"
NO_THINK = {"thinking": False}


def chat(messages, thinking=False, max_tokens=24):
    body = {"model": "deepseek-v4.1-flash", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    if not thinking:
        body["chat_template_kwargs"] = NO_THINK
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    wall = time.perf_counter() - t0
    msg = out["choices"][0]["message"]
    u, t = out.get("usage", {}), out.get("timings", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return msg, u.get("prompt_tokens", 0), cached, t.get("prompt_ms", 0.0), wall


def turn(name, messages, expect_min_cached, **kw):
    msg, prompt, cached, prompt_ms, wall = chat(messages, **kw)
    ok = cached >= expect_min_cached
    print(f"{name:3s} prompt={prompt:6d} cached={cached:6d} (need >= {expect_min_cached:6d}) "
          f"prefill={prompt_ms / 1000:6.2f}s wall={wall:6.2f}s  {'OK' if ok else 'MISS'}", flush=True)
    reply = {"role": "assistant", "content": msg.get("content") or ""}
    return reply, prompt, ok


def doc(tag, lines):
    return "\n".join(f"{tag} line {i}: record {i * 7919 % 10007} belongs to group {i % 13} "
                     f"and was filed under code {tag}-{i:05d}." for i in range(lines))


results = []
qa = {"role": "user", "content": doc("alpha", 420) + "\n\nHow many lines does the document above have? "
      "Answer with the number only."}
A = [qa]
ra, pa1, ok = turn("A1", A, 0); results.append(ok)

B = [{"role": "user", "content": doc("bravo", 60) + "\n\nWhich group does line 5 belong to? Number only."}]
rb, pb1, ok = turn("B1", B, 0); results.append(ok)

A += [ra, {"role": "user", "content": "And which group does line 12 belong to? Number only."}]
ra, pa2, ok = turn("A2", A, pa1 - 1); results.append(ok)

B += [rb, {"role": "user", "content": "And line 7? Number only."}]
rb, pb2, ok = turn("B2", B, pb1 - 1); results.append(ok)

C = [{"role": "user", "content": doc("charlie", 80) + "\n\nWhat is the code of line 3? Reply briefly."}]
rc, pc1, ok = turn("C1", C, 0, thinking=True, max_tokens=200); results.append(ok)
C += [rc, {"role": "user", "content": "And of line 4? Reply briefly."}]
rc, pc2, ok = turn("C2", C, pc1 - 1, thinking=True, max_tokens=200); results.append(ok)

D = [{"role": "user", "content": doc("delta", 60) + "\n\nWhich group does line 2 belong to? Number only."}]
_, _, ok = turn("D1", D, 0); results.append(ok)
E = [{"role": "user", "content": doc("echo", 60) + "\n\nWhich group does line 3 belong to? Number only."}]
_, _, ok = turn("E1", E, 0); results.append(ok)      # 5th conversation: the LRU one (A) spills

A += [ra, {"role": "user", "content": "And line 20? Number only."}]
_, _, ok = turn("A3", A, pa2 - 1); results.append(ok)  # back to A: reload from NVMe, not re-prefill

st = json.loads(urllib.request.urlopen(BASE + "/tp/status", timeout=10).read())
print("tp/status:", json.dumps({k: st[k] for k in st if k not in ("source",)})[:900])
print("RESULT", "PASS" if all(results) else "FAIL")
