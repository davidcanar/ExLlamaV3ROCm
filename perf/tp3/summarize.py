#!/usr/bin/env python3
"""One line per profiled request in perf/tp3/prof3.jsonl: tag, context, mode, decode speed, output hash."""
import hashlib
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "prof3.jsonl"
for line in open(path):
    r = json.loads(line)
    if r.get("mode") == "pf":
        print(f"{r.get('tag', ''):6s} ctx={r['ctx']:6d} prefill-profile {r['prefill_s']:.2f}s")
        continue
    st = r["stats"]
    tg = st["new_tokens"] / max(st["time_generate"], 1e-9)
    t = r["times_ms"]
    med = lambda k: sorted(t[k])[len(t[k]) // 2] if t.get(k) else 0.0
    step = med("forward") if r["mode"] == "plain" else med("verify") + med("draft")
    h = hashlib.sha1(r["text"].encode()).hexdigest()[:12]
    print(f"{r.get('tag', ''):6s} ctx={r['ctx']:6d} {r['mode']:5s} tok={st['new_tokens']:4d} "
          f"gen={st['time_generate']:5.2f}s tok/s={tg:5.1f} median_step={step:6.1f}ms "
          f"acc/rnd={st.get('accepted_per_step', 0):.2f} prefill={(r['ctx'] - 1) / max(r['prefill_s'], 1e-9):5.0f}tok/s "
          f"sha={h} ar3={r['env'].get('EXL3_TP_ODL_AR3', '0')}")
