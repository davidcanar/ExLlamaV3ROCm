#!/usr/bin/env python3
"""Interleaved-request check against the running server (EXL3_TP_CONCURRENCY >= 2, proto 6).

Run it only on a server nobody else is using: it creates (and evicts) cached conversations.

  S0/L0  solo baselines: a short reply and a long reply, each alone (their time and text)
  LS     the long reply again, and the short one sent once the long one streams: the short one
         must finish long before the long one (no head-of-line blocking), both complete, and
         the short reply should equal its solo text (greedy; see the note below)
  PF     a long uncached prompt arrives while the long reply streams: the stream's largest gap
         between two chunks shows the interleaved prefill (EXL3_TP_INTERLEAVE_CHUNK rows per
         step) instead of one stall for the whole prefill
  CR     a conversation turn is cancelled mid-stream and sent again at once: the resend must
         resume from the conversation's cache (cached_tokens), not re-prefill it
  MX     (concurrency >= 3) that many streams at once share the GPU fairly (similar chunk rates,
         together about the solo rate), and one more request waits for a free slot
  ST     /tp/status reports proto 6 and the scheduler; nothing is left running

Greedy replies can differ run to run on this rig even serially (near-ties, see
docs/PERFORMANCE.md), so a text mismatch is reported, not failed; timing and cache numbers
decide PASS. usage: interleave_check.py [base url]   (default http://127.0.0.1:1234)
"""
import http.client
import json
import sys
import threading
import time
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1234"
URL = urllib.parse.urlparse(BASE)
MODEL = "deepseek-v4.1-flash"


def doc(tag, lines):
    return "\n".join(f"{tag} line {i}: record {i * 7919 % 10007} belongs to group {i % 13} "
                     f"and was filed under code {tag}-{i:05d}." for i in range(lines))


class Stream:
    """One streaming chat request on its own thread; records chunk arrival times."""

    def __init__(self, name, messages, max_tokens, *, cancel_after=None, extra=None):
        self.name, self.messages, self.max_tokens = name, messages, max_tokens
        self.cancel_after = cancel_after          # close the connection after N chunks
        self.extra = extra or {}
        self.text, self.times, self.usage, self.error = "", [], {}, None
        self.t0 = self.t_end = None
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def join(self, timeout=3600):
        self.thread.join(timeout)
        return self

    def _run(self):
        body = {"model": MODEL, "messages": self.messages, "max_tokens": self.max_tokens,
                "temperature": 0, "stream": True,          # (the last chunk carries usage)
                "chat_template_kwargs": {"thinking": False}}
        body.update(self.extra)
        conn = http.client.HTTPConnection(URL.hostname, URL.port or 80, timeout=3600)
        self.t0 = time.perf_counter()
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps(body),
                         {"content-type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                self.error = f"HTTP {resp.status}: {resp.read()[:300]!r}"
                return
            n = 0
            while True:
                line = resp.readline()
                if not line:
                    break
                if not line.startswith(b"data: ") or b"[DONE]" in line:
                    continue
                ev = json.loads(line[6:])
                if ev.get("usage"):
                    self.usage = ev["usage"]
                for ch in ev.get("choices") or ():
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        self.text += piece
                        self.times.append(time.perf_counter())
                        n += 1
                        self.started.set()
                        if self.cancel_after is not None and n >= self.cancel_after:
                            conn.close()                 # the client walks away mid-stream
                            return
        except Exception as exc:                          # noqa: BLE001
            if self.cancel_after is None:
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.t_end = time.perf_counter()
            self.started.set()
            conn.close()

    @property
    def wall(self):
        return (self.t_end or time.perf_counter()) - self.t0

    @property
    def ttft(self):
        return (self.times[0] - self.t0) if self.times else float("nan")

    def gap(self, t_from=None, t_to=None):
        """The longest pause between two consecutive chunks that overlaps [t_from, t_to] (a
        pause that straddles either end counts: that is the stall the stream saw)."""
        lo = float("-inf") if t_from is None else t_from
        hi = float("inf") if t_to is None else t_to
        return max((b - a for a, b in zip(self.times, self.times[1:]) if b >= lo and a <= hi),
                   default=0.0)

    def cached(self):
        return (self.usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def status():
    return json.loads(urllib.request.urlopen(BASE + "/tp/status", timeout=30).read())


def same(a, b):
    if a == b:
        return "identical"
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"diverged at char {n} (near-tie; see docstring)"


results = {}
st = status()
if st.get("proto") != 6:
    print(f"server speaks proto {st.get('proto')}: start every rank with EXL3_TP_CONCURRENCY >= 2")
    sys.exit(2)
conc = st.get("pool", {}).get("scheduler", {}).get("concurrency")
print(f"server: proto {st['proto']}, concurrency {conc}, sessions {st.get('sessions')}", flush=True)

LONG = [{"role": "user", "content": "Write a detailed, numbered list of 40 distinct facts about the "
                                    "history of tide prediction machines, one per line."}]
SHORT = [{"role": "user", "content": "In one word: what color is a clear daytime sky?"}]

# ---- S0 / L0: solo baselines
s0 = Stream("S0", SHORT, 16).start().join()
l0 = Stream("L0", LONG, 400).start().join()
for s in (s0, l0):
    assert s.error is None, (s.name, s.error)
l0_rate = len(l0.times) / max(l0.times[-1] - l0.times[0], 1e-9)
print(f"S0 solo short: wall {s0.wall:6.2f}s ttft {s0.ttft:5.2f}s | L0 solo long: wall {l0.wall:6.1f}s "
      f"~{l0_rate:5.1f} chunks/s", flush=True)

# ---- LS: the short one while the long one streams
l1 = Stream("L1", LONG, 400).start()
l1.started.wait(600)
time.sleep(1.0)
t_short = time.perf_counter()
s1 = Stream("S1", SHORT, 16).start().join()
l1.join()
ok = (s1.error is None and l1.error is None and s1.t_end < l1.t_end
      and s1.wall < max(4 * s0.wall, s0.wall + 10.0))
results["LS"] = ok
print(f"LS short beside long: wall {s1.wall:6.2f}s (solo {s0.wall:5.2f}s) finished "
      f"{l1.t_end - s1.t_end:6.1f}s before the long one; long stream's worst gap while the short ran "
      f"{l1.gap(t_short, s1.t_end):5.2f}s; short reply {same(s1.text, s0.text)}  "
      f"{'OK' if ok else 'FAIL'}", flush=True)

# ---- PF: a long uncached prefill beside a decoding stream
l2 = Stream("L2", LONG, 400).start()
l2.started.wait(600)
time.sleep(1.0)
big = [{"role": "user", "content": doc(f"pf{int(time.time())}", 700) + "\n\nHow many lines? Number only."}]
t_pf = time.perf_counter()
p = Stream("PF", big, 8).start().join()
l2.join()
pf_gap = l2.gap(t_pf, p.t_end)
prefill_s = p.ttft
ok = p.error is None and l2.error is None and pf_gap < max(5.0, prefill_s / 3)
results["PF"] = ok
print(f"PF {p.usage.get('prompt_tokens', '?')}-token prefill beside a stream: first token after "
      f"{prefill_s:6.1f}s; the stream's worst gap meanwhile {pf_gap:5.2f}s (serialized it would "
      f"stall ~{prefill_s:5.1f}s)  {'OK' if ok else 'FAIL'}", flush=True)

# ---- CR: cancel + resend of a running conversation turn
conv = [{"role": "user", "content": doc("cr", 300) + "\n\nWhich group does line 9 belong to? Number only."}]
c1 = Stream("C1", conv, 8).start().join()
assert c1.error is None, c1.error
turn2 = conv + [{"role": "assistant", "content": c1.text},
                {"role": "user", "content": "Now list every line number from 1 to 200, comma separated."}]
c2 = Stream("C2", turn2, 600, cancel_after=20).start().join()
c3 = Stream("C3", turn2, 16).start().join()
ok = c3.error is None and c3.cached() >= c1.usage.get("prompt_tokens", 1) - 1
results["CR"] = ok
print(f"CR cancel + resend: resend cached {c3.cached()} of {c3.usage.get('prompt_tokens')} prompt "
      f"tokens (turn 1 prompt {c1.usage.get('prompt_tokens')}), first token after {c3.ttft:5.2f}s  "
      f"{'OK' if ok else 'MISS'}", flush=True)

# ---- MX: as many streams as the server interleaves at once, and one more, which must wait
if conc and conc >= 3:
    mx = [Stream(f"M{k}", [{"role": "user", "content": f"Count upwards from {1000 * (k + 1)}, one "
                            "number per line, nothing else."}], 160).start() for k in range(conc)]
    for m in mx:
        m.started.wait(600)
    t_all = time.perf_counter()                       # every one of them is streaming now
    extra = Stream("MX", [{"role": "user", "content": "Name one prime number. Number only."}], 8).start()
    time.sleep(0.5)
    waiting = len(status().get("pool", {}).get("scheduler", {}).get("pending", []))
    for m in mx + [extra]:
        m.join()
    t_first = min(m.times[-1] for m in mx if m.times)   # the last chunk of the first one to finish
    win = max(t_first - t_all, 1e-9)
    rates = [sum(t_all <= t <= t_first for t in m.times) / win for m in mx]
    waited = bool(extra.times) and extra.times[0] >= t_first - 0.25
    ok = (all(m.error is None for m in mx + [extra]) and waited
          and sum(rates) >= 0.6 * l0_rate and min(rates) >= 0.5 * max(rates))
    results["MX"] = ok
    print(f"MX {conc} streams at once: {min(rates):4.1f}-{max(rates):4.1f} chunks/s each, "
          f"{sum(rates):5.1f} together (solo {l0_rate:5.1f}); request {conc + 1} "
          f"{'waited for a slot' if waited else 'did NOT wait'} (pending while they ran: {waiting}), "
          f"first token {extra.ttft:5.2f}s after it was sent  {'OK' if ok else 'FAIL'}", flush=True)

# ---- ST: status
time.sleep(1.0)
st = status()
sch = st.get("pool", {}).get("scheduler", {})
ok = st.get("proto") == 6 and not sch.get("active") and not sch.get("pending") and not sch.get("fatal")
results["ST"] = ok
print("ST scheduler:", json.dumps(sch)[:600], "OK" if ok else "FAIL")
print("RESULT", "PASS" if all(results.values()) else "FAIL", results)
