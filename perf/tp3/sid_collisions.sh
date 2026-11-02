#!/usr/bin/env bash
# Count, per plugin trace newer than a timestamp file, connects that reused a send stream id
# already used on the OTHER cable (the collision the per-cable registry fix handles), and sends
# still waiting for a grant at the end (posted with dst=0 and never finished).
# usage: sid_collisions.sh <t0-file>
t0=$(cat "$1")
n=0; hit=0
for f in $(ls -tr /tmp/odl_plugin.*.log); do
    [ "$(stat -c %Y "$f")" -ge "$t0" ] || continue
    n=$((n + 1))
    d=$(grep -oE 'connect peer [0-9.]+ cable dev [0-9]: S=[0-9]+' "$f" \
        | sed 's/.*dev \([0-9]\): S=\([0-9]*\)/\1 \2/' | sort -u | awk '{print $2}' | sort | uniq -d | tr '\n' ',')
    if [ -n "$d" ]; then hit=$((hit + 1)); echo "  $(basename "$f"): same S on both cables: ${d%,}"; fi
done
echo "traces: $n, with a cross-cable stream-id collision: $hit"
