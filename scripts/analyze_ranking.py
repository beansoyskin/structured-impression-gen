# -*- coding: utf-8 -*-
"""分析排序问题：看知识表 top1 正确 vs 错误的样本，定位置信度和频次的关系。"""
import json, sys, os, random
from collections import Counter

sys.path.insert(0, ".")
from src.norm.head_norm import normalize_head
from src.knowledge.suggestive_table import SuggestiveKnowledge

DATA = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"
KNOWLEDGE_PATH = "outputs/suggestive_of_table.json"

# 只加载表本身，不加载 query 器
with open(KNOWLEDGE_PATH) as f:
    raw_table = json.load(f)

# 对每个 finding head，看其候选的 confidence 分布
# 特别看"top1 命中"和"top1 没命中"时，top1 候选的 confidence 分别是多少
top1_correct_conf = []
top1_wrong_conf = []
top1_correct_count = []
top1_wrong_count = []

# 抽样真实数据看排序
records = []
with open(DATA, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
rng = random.Random(42)
sample = rng.sample(records, 2000)

head_occurrence = Counter()  # finding head 在数据中出现的频次（用于分析稀有度）

for rec in sample:
    fg = rec.get("findings_graph_compact", {})
    ig = rec.get("impression_graph_compact", {})

    find_heads = set()
    for b in ("positive","negative","uncertain","other"):
        for f in fg.get(b,[]):
            h = normalize_head(f.get("head",""))
            if h: find_heads.add(h); head_occurrence[h] += 1

    imp_heads = set()
    for b in ("positive","negative","uncertain","other"):
        for f in ig.get(b,[]):
            h = normalize_head(f.get("head",""))
            if h: imp_heads.add(h)
            for s in (f.get("suggestive_of") or []):
                if isinstance(s,dict):
                    sh = normalize_head(s.get("finding",""))
                    if sh: imp_heads.add(sh)

    emergent = imp_heads - find_heads
    if not emergent: continue

    for gold_h in emergent:
        for find_h in find_heads:
            rows = raw_table.get(find_h, [])
            if not rows: continue
            top1 = rows[0]
            if gold_h == top1["target_head"]:
                top1_correct_conf.append(top1["confidence"])
                top1_correct_count.append(top1["count"])
            else:
                top1_wrong_conf.append(top1["confidence"])
                top1_wrong_count.append(top1["count"])

print(f"Top1 正确样本数: {len(top1_correct_conf)}")
print(f"Top1 错误样本数: {len(top1_wrong_conf)}")
print()
print(f"{'统计量':15} {'正确(top1命中)':>15} {'错误(top1未命中)':>15}")
print("-" * 47)
import statistics
for name, vals_c, vals_w in [
    ("置信度均值", top1_correct_conf, top1_wrong_conf),
    ("置信度中位数", top1_correct_conf, top1_wrong_conf),
    ("频次均值", top1_correct_count, top1_wrong_count),
    ("频次中位数", top1_correct_count, top1_wrong_count),
]:
    if name.endswith("均值"):
        vc = statistics.mean(vals_c) if vals_c else 0
        vw = statistics.mean(vals_w) if vals_w else 0
    else:
        vc = statistics.median(vals_c) if vals_c else 0
        vw = statistics.median(vals_w) if vals_w else 0
    print(f"{name:15} {vc:>15.4f} {vw:>15.4f}")

print()
print("## Top1 正确时 top1 候选的频次分布")
buckets_c = Counter()
for c in top1_correct_count:
    if c >= 1000: buckets_c["1000+"] += 1
    elif c >= 100: buckets_c["100-999"] += 1
    elif c >= 10: buckets_c["10-99"] += 1
    else: buckets_c["1-9"] += 1
for k in ["1-9","10-99","100-999","1000+"]:
    print(f"  {k:8}: {buckets_c[k]:5} ({buckets_c[k]/len(top1_correct_count)*100:.1f}%)")

print("\n## Top1 错误时 top1 候选的频次分布")
buckets_w = Counter()
for c in top1_wrong_count:
    if c >= 1000: buckets_w["1000+"] += 1
    elif c >= 100: buckets_w["100-999"] += 1
    elif c >= 10: buckets_w["10-99"] += 1
    else: buckets_w["1-9"] += 1
for k in ["1-9","10-99","100-999","1000+"]:
    print(f"  {k:8}: {buckets_w[k]:5} ({buckets_w[k]/len(top1_wrong_count)*100:.1f}%)")
