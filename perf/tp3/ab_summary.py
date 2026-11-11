#!/usr/bin/env python3
"""Side-by-side view of perf/tp3/ab.jsonl (ts_ab.py records): decode/verify/draft medians, tok/s,
acceptance, verify time by row count, prefill speed, profiled per-op time, numerics vs the ref.

usage: ab_summary.py [ab.jsonl] [tag ...]   (tags: only these, in this order; default all)
"""
import collections
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ab.jsonl"
tags = sys.argv[2:]
recs = [json.loads(line) for line in open(path) if line.strip()]
if tags:
    recs = sorted((r for r in recs if r.get("tag") in tags), key=lambda r: tags.index(r["tag"]))


def med(v):
    return sorted(v)[len(v) // 2] if v else float("nan")


groups = collections.OrderedDict()
for r in recs:
    groups.setdefault((r["ctx"], r["mode"]), []).append(r)

for (ctx, mode), rs in groups.items():
    print(f"== ctx {ctx} {mode}")
    base = rs[0]["text"]
    for r in rs:
        st, t = r["stats"], r["times_ms"]
        tg = st.get("new_tokens", 0) / max(st.get("time_generate", 0), 1e-9)
        pf = (r["ctx"] - 1) / max(r["prefill_s"], 1e-9)
        same = next((i for i, (a, b) in enumerate(zip(base, r["text"])) if a != b), min(len(base), len(r["text"])))
        line = f"  {r['tag']:<8} {tg:6.2f} tok/s  prefill {pf:6.1f} tok/s  "
        if mode == "plain":
            line += f"forward median {med(t.get('forward', [])):6.2f} ms (n={len(t.get('forward', []))})"
        else:
            d, v = med(t.get("draft", [])), med(t.get("verify", []))
            line += (f"draft {d:5.2f} + verify {v:6.2f} ms (n={len(t.get('verify', []))}), "
                     f"accepted/round {st.get('accepted_per_step', 0):.2f}")
        line += f"  | free {r.get('mem_free_gib_rank0', float('nan')):.1f} GiB | text same for {same} chars"
        print(line)
        if r.get("verify_rows"):
            byr = collections.defaultdict(list)
            for n, ms in r["verify_rows"]:
                byr[n].append(ms)
            print("           verify ms by rows: "
                  + "  ".join(f"{n}:{med(v):.1f}(n={len(v)})" for n, v in sorted(byr.items())))
        prof, ps = r.get("prof", {}), max(r.get("prof_steps", 0), 1)
        if prof:
            ex = {k.strip(): v[2] for k, v in prof.items()}
            ar = sum(v for k, v in ex.items() if "all_reduce" in k)
            tot = sum(ex.values())
            print(f"           profiled/step ({r.get('prof_steps')}): total {tot / ps * 1e3:6.1f} ms  "
                  f"moe {ex.get('moe', 0) / ps * 1e3:6.2f}  all-reduce {ar / ps * 1e3:5.2f}  "
                  f"linear {ex.get('linear(other)', 0) / ps * 1e3:5.2f}  dsa {ex.get('attn.dsa_attn', 0) / ps * 1e3:5.2f}  "
                  f"engram {ex.get('engram', 0) / ps * 1e3:5.2f}")
        if r.get("logits_vs_ref"):
            lv = r["logits_vs_ref"]
            print(f"           first-token logits vs {lv['ref']}: KL {lv['kl']:.2e}, top-1 "
                  f"{'same' if lv['top1_same'] else 'DIFFERENT'}, top-5 overlap {lv['top5_overlap']}/5, "
                  f"max |diff| {lv['max_abs']:.3f}")
