# -*- coding: utf-8 -*-
"""采样 location 自由文本的真实形态，用于写准 §4.0(b) 解剖词表与 laterality 规则。"""
import json
import sys
from collections import Counter

c = Counter()
n = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    o = json.loads(line)
    for b in ("positive", "negative", "uncertain", "other"):
        for f in o.get("findings_graph_compact", {}).get(b, []) or []:
            for loc in (f.get("locations") or []):
                t = loc.lower().strip() if isinstance(loc, str) else (loc.get("text", "") if isinstance(loc, dict) else "")
                if t:
                    c[t] += 1
    n += 1
    if n >= 8000:
        break

print(f"扫描 {n} 行，unique location = {len(c)}")
print("\n## 含 'left' 的 top15")
for t, k in sorted(((t, k) for t, k in c.items() if "left" in t), key=lambda x: -x[1])[:15]:
    print(f"  {k:5d}  {t}")
print("\n## 含 'bilateral' 的 top15")
for t, k in sorted(((t, k) for t, k in c.items() if "bilateral" in t), key=lambda x: -x[1])[:15]:
    print(f"  {k:5d}  {t}")
print("\n## 含 'right'（且不含 bilateral）的 top15")
for t, k in sorted(((t, k) for t, k in c.items() if "right" in t and "bilateral" not in t), key=lambda x: -x[1])[:15]:
    print(f"  {k:5d}  {t}")

pref = Counter()
for t, k in c.items():
    for tk in ("left", "right", "bilateral"):
        if t.startswith(tk):
            pref[tk] += k
            break
print("\n## 以 left/right/bilateral 开头的 location 总频次:", dict(pref))
