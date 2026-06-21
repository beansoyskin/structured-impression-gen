# -*- coding: utf-8 -*-
"""构建联合知识表（pair共现）+ 评测（包含三种优化）。"""
import sys, json, time
sys.path.insert(0, ".")
from src.knowledge.suggestive_table import build_pair_table

DATA = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"

def iter_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

t0 = time.time()
records = list(iter_records(DATA))
t1 = time.time()
print(f"读 {len(records)} 条: {t1-t0:.1f}s", flush=True)

t2 = time.time()
pair_table = build_pair_table(records)
t3 = time.time()
print(f"构建 pair 表: {t3-t2:.1f}s", flush=True)
print(f"  unique pairs: {len(pair_table)}", flush=True)

# 存盘
OUT = "outputs/pair_table.json"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(pair_table, f, ensure_ascii=False)
sz = len(json.dumps(pair_table, ensure_ascii=False)) / 1024 / 1024
print(f"  已存: {OUT} ({sz:.1f} MB)", flush=True)

# 展示
print("\n## Top 20 高频 pair → target:", flush=True)
for key, rows in sorted(pair_table.items(), key=lambda x: -x[1][0]["count"])[:20]:
    top = rows[0]
    print(f"  {key:40} → {top['target_head']:20} ({top['count']:5}次, conf={top['confidence']:.2f})", flush=True)
