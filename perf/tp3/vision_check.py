#!/usr/bin/env python3
"""Vision check against the running server: the saved BLUE HERON 42 card with DSpark and plain,
a broken image (must be a clean 400), then a text request. usage: vision_check.py <request.json>"""
import json
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:1234/v1/chat/completions"
base = json.load(open(sys.argv[1]))


def send(body, label):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"{label}: HTTP {e.code}: {e.read()[:160]!r}")
        return
    msg = out["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    t, u = out.get("timings", {}), out.get("usage", {})
    print(f"{label}: {text!r} | prompt={u.get('prompt_tokens')} "
          f"decode={t.get('predicted_per_second', 0):.1f} tok/s wall={time.perf_counter() - t0:.1f}s")


send(dict(base, stream=False), "image, DSpark")
send(dict(base, stream=False, speculative=False), "image, plain ")
bad = json.loads(json.dumps(base))
bad["stream"] = False
bad["messages"][0]["content"][1]["image_url"]["url"] = "data:image/png;base64,AAAA"
send(bad, "broken image")
send({"model": "deepseek-v4.1-flash", "max_tokens": 40, "temperature": 0,
      "messages": [{"role": "user", "content": "Name three prime numbers."}],
      "chat_template_kwargs": {"thinking": False}}, "text after  ")
