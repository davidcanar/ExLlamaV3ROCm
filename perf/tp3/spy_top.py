#!/usr/bin/env python3
"""Aggregate py-spy `--format raw` (collapsed stacks) output: per-thread totals, and the top frames by
inclusive and self samples for the main thread (or --thread substring).
usage: spy_top.py file.txt [--thread MainThread] [--grep substring] [-n 40]"""
import argparse
import collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--thread", default="MainThread")
    ap.add_argument("--grep", default=None, help="only stacks containing this substring")
    ap.add_argument("-n", type=int, default=40)
    a = ap.parse_args()
    threads = collections.Counter()
    incl = collections.Counter()
    selfc = collections.Counter()
    total = 0
    for line in open(a.file, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line or " " not in line:
            continue
        stack, _, cnt = line.rpartition(" ")
        try:
            cnt = int(cnt)
        except ValueError:
            continue
        frames = stack.split(";")
        th = frames[0]
        threads[th] += cnt
        if a.thread not in th:
            continue
        if a.grep and a.grep not in stack:
            continue
        total += cnt
        seen = set()
        for f in frames[1:]:
            if f not in seen:
                incl[f] += cnt
                seen.add(f)
        selfc[frames[-1]] += cnt
    print("threads:")
    for th, c in threads.most_common(12):
        print(f"  {c:7d}  {th}")
    print(f"\n{a.thread}: {total} samples; top inclusive:")
    for f, c in incl.most_common(a.n):
        print(f"  {100 * c / max(total, 1):5.1f}%  {f[:150]}")
    print("\ntop self:")
    for f, c in selfc.most_common(25):
        print(f"  {100 * c / max(total, 1):5.1f}%  {f[:150]}")


if __name__ == "__main__":
    main()
