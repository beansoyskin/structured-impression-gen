# -*- coding: utf-8 -*-
"""Profile rexgradient_radgraph_structured_v3_full.jsonl
输出决定 §4.2 检索器 / §4.5 verifier 规格的关键统计。"""
import json
import sys
from collections import Counter, defaultdict

PATH = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"

n = 0
# assertion 词表
assertion_set = set()
# finding head 词表 + 频次
finding_head = Counter()
# impression head 词表 + 频次（合法诊断词表候选）
impression_head = Counter()
# finding->impression head 涌现率：impression head 不在 finding head 集合的比例（按样本）
emergent_count = 0
emergent_heads = Counter()
# suggestive_of 使用率（impression 端）
sugg_count = 0
sugg_heads = Counter()
# locations 自由文本词表
loc_texts = Counter()
laterality_tokens = Counter()  # left/right/bilateral
# 每样本 fact 数量分布（impression）
imp_fact_n = []
finding_fact_n = []
# compact 里没有 suggestive_of 字段的（结构不一致）
no_sugg_field = 0
# head 全小写归一化后 finding 与 impression 的重叠
finding_head_lc = Counter()
impression_head_lc = Counter()
# location 同时含 left 和 right 之类的奇怪组合
ambiguous_laterality = 0

LATERALITY = ("left", "right", "bilateral", "unilateral", "bilateralism")

def collect_locations(loc_list):
    for loc in loc_list:
        if isinstance(loc, dict):
            t = loc.get("text") or loc.get("finding") or ""
        else:
            t = str(loc)
        if t:
            loc_texts[t.lower().strip()] += 1
            low = t.lower()
            for tok in LATERALITY:
                if tok in low:
                    laterality_tokens[tok] += 1

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    n += 1

    fg = obj.get("findings_graph_compact", {})
    ig = obj.get("impression_graph_compact", {})

    # finding facts
    f_heads = set()
    fcount = 0
    for bucket in ("positive", "negative", "uncertain", "other"):
        for f in fg.get(bucket, []) or []:
            fcount += 1
            h = (f.get("head") or "").lower().strip()
            if h:
                finding_head[h] += 1
                finding_head_lc[h] += 1
                f_heads.add(h)
            a = (f.get("assertion") or "").strip()
            if a:
                assertion_set.add(a)
            collect_locations(f.get("locations", []) or [])
    finding_fact_n.append(fcount)

    # impression facts
    icount = 0
    ig_has_sugg = False
    for bucket in ("positive", "negative", "uncertain", "other"):
        for f in ig.get(bucket, []) or []:
            icount += 1
            h = (f.get("head") or "").lower().strip()
            if h:
                impression_head[h] += 1
                impression_head_lc[h] += 1
                # 涌现：impression head 不在当前样本 finding head 集合里
                if h not in f_heads:
                    emergent_heads[h] += 1
            a = (f.get("assertion") or "").strip()
            if a:
                assertion_set.add(a)
            collect_locations(f.get("locations", []) or [])
            sugg = f.get("suggestive_of", None)
            if sugg is None:
                pass
            else:
                ig_has_sugg = True
                if sugg:
                    sugg_count += 1
                    for s in sugg:
                        sh = (s.get("finding") or s.get("head") or "").lower().strip()
                        if sh:
                            sugg_heads[sh] += 1
    imp_fact_n.append(icount)
    if not ig_has_sugg and any("suggestive_of" in (ig.get(b, [{}])[0] if ig.get(b) else {}) for b in ig):
        pass

# 涌现统计已在循环内累加，无需二次处理
print("=" * 60)
print(f"总样本数 N = {n}")
print()
print("## Assertion 词表（闭合集）")
for a in sorted(assertion_set):
    print(f"  - {a}")
print()
print("## Finding head 词表大小 / top20")
print(f"  unique = {len(finding_head)}")
for h, c in finding_head.most_common(20):
    print(f"  {c:6d}  {h}")
print()
print("## Impression head 词表大小 / top20（合法诊断词表候选）")
print(f"  unique = {len(impression_head)}")
for h, c in impression_head.most_common(20):
    print(f"  {c:6d}  {h}")
print()
print("## 涌现 head（impression head 出现时不在同样本 finding heads 中）事件数 / top20")
print(f"  emergent events = {sum(emergent_heads.values())}")
for h, c in emergent_heads.most_common(20):
    print(f"  {c:6d}  {h}")
print()
print("## suggestive_of 使用（impression 端）")
print(f"  suggestive_of events = {sugg_count}")
for h, c in sugg_heads.most_common(15):
    print(f"  {c:6d}  {h}")
print()
print("## Location 自由文本：unique / top30")
print(f"  unique locations = {len(loc_texts)}")
for t, c in loc_texts.most_common(30):
    print(f"  {c:6d}  {t}")
print()
print("## Laterality token 频次（埋在 location 文本里）")
for t, c in laterality_tokens.most_common():
    print(f"  {c:6d}  {t}")
print()
print("## 每样本 fact 数量分布")
def stats(name, arr):
    arr = sorted(arr)
    import statistics
    print(f"  {name}: min={arr[0]} p50={int(statistics.median(arr))} mean={statistics.mean(arr):.2f} max={arr[-1]}")
stats("finding facts/sample", finding_fact_n)
stats("impression facts/sample", imp_fact_n)
print()
print("## finding head 与 impression head 的词表重叠（归一化后）")
overlap = set(finding_head_lc) & set(impression_head_lc)
only_imp = set(impression_head_lc) - set(finding_head_lc)
print(f"  overlap = {len(overlap)}  impression-only = {len(only_imp)}  finding-only = {len(set(finding_head_lc)-set(impression_head_lc))}")
print(f"  impression-only top15（=几乎只能靠推断/抽象得到的诊断词）:")
for h in list(only_imp)[:15]:
    print(f"    {impression_head_lc[h]:6d}  {h}")
