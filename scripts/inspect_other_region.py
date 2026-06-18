# -*- coding: utf-8 -*-
"""诊断：看归一后 region='other' 的 location 都是什么，决定要不要补 REGION_TABLE。"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

loc_path = os.path.join(ROOT, "outputs", "location_norm.json")
with open(loc_path, "r", encoding="utf-8") as fh:
    loc_map = json.load(fh)

# 需要 frequency，重新从数据扫一遍（只统计 region=other 的）
DATA = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"
other_freq = Counter()
n = 0
with open(DATA, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        for sec in ("findings_graph_compact", "impression_graph_compact"):
            for b in ("positive", "negative", "uncertain", "other"):
                for f in o.get(sec, {}).get(b, []) or []:
                    for loc in (f.get("locations") or []):
                        t = loc if isinstance(loc, str) else (loc.get("text") or loc.get("finding") or "")
                        t = (t or "").strip()
                        if t and loc_map.get(t, {}).get("region") == "other":
                            other_freq[t] += 1
        n += 1

print(f"扫描 {n} 行，region=other 的 unique location = {len(other_freq)}")
print("\n## region=other 的 top40（按频次）")
for t, c in other_freq.most_common(40):
    print(f"  {c:6d}  {t}")
