# -*- coding: utf-8 -*-
"""快速采样：看 head 碎片化的真实形态，好写准别名表。
只读前 5000 行，统计 head 的单复数对、常见形态变体。"""
import json
import sys
from collections import Counter

heads = Counter()
n = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    for bucket in ("positive", "negative", "uncertain", "other"):
        for f in obj.get("findings_graph_compact", {}).get(bucket, []) or []:
            h = (f.get("head") or "").strip()
            if h:
                heads[h.lower()] += 1
    n += 1
    if n >= 5000:
        break

print(f"扫描 {n} 行，unique head = {len(heads)}")
print("\n## 频次 >= 20 的 head（看单复数/形态变体）")
for h, c in sorted(heads.items(), key=lambda x: -x[1]):
    if c >= 20:
        print(f"  {c:5d}  {h}")

print("\n## 以 s 结尾的高频 head（疑似复数）")
for h, c in sorted(heads.items(), key=lambda x: -x[1]):
    if c >= 10 and h.endswith("s") and not h.endswith("ss"):
        print(f"  {c:5d}  {h}")
