#!/usr/bin/env python3
"""HTTP sanity check against the running server (TP=2 or TP=3): one DSpark (default) and one plain
request, greedy, a fixed short prompt; prints decode speed from the server's own timings and the
reply head. usage: http_check.py [url]   (default http://127.0.0.1:1234/v1/chat/completions)"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1234/v1/chat/completions"
PROMPT = ("Explain in about 150 words why tensor parallelism across three machines needs fast "
          "all-reduce operations, and what limits decode speed.")


def ask(speculative):
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 400, "temperature": 0, "stream": False}
    if speculative is not None:
        body["speculative"] = speculative
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    wall = time.perf_counter() - t0
    u = out.get("usage", {})
    t = out.get("timings", {})
    msg = out["choices"][0]["message"]
    text = (msg.get("content") or "") or (msg.get("reasoning_content") or "")
    print(json.dumps({"speculative": speculative, "wall_s": round(wall, 2), "usage": u, "timings": t}))
    print("  reply:", repr(text[:240]))


ask(None)
ask(False)
