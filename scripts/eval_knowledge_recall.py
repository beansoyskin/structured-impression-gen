# -*- coding: utf-8 -*-
"""评测 §4.3 知识表的 Recall@K：knowledge 候选是否覆盖真实 impression head。

定义：
  - Query: (finding_head, finding_assertion) 组合
  - Gold: 该样本 impression 的 head（归一后）
  - Recall@K: TopK 候选里是否包含 gold head（按 count 降序）

关键过滤：
  - 只测"涌现"情况（impression head 不在 finding head 里）——因为这类才需要知识表
  - 空 impression 跳过
  - finding 无对应的 knowledge entry 则记为 miss
"""
import json, sys, os
from collections import defaultdict, Counter

sys.path.insert(0, ".")
from src.norm.head_norm import normalize_head
from src.knowledge.suggestive_table import SuggestiveKnowledge

DATA = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"
KNOWLEDGE_PATH = "outputs/suggestive_of_table.json"
knowledge = SuggestiveKnowledge(KNOWLEDGE_PATH, min_count=1)

# 统计
n_samples = 0
n_with_gold = 0   # 有涌现 impression head 的样本数
n_query_pairs = 0 # (finding_head, gold_head) 对的数量
total_pairs = 0
correct_at_k = {1: 0, 3: 0, 5: 0, 10: 0, "all": 0}

# 按 finding head 聚合统计
by_finding_head = defaultdict(lambda: {"total": 0, "correct_top1": 0, "correct_top5": 0})

records = []
with open(DATA, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

for rec in records:
    fg = rec.get("findings_graph_compact", {})
    ig = rec.get("impression_graph_compact", {})
    if not fg or not ig:
        continue

    # 收集 finding heads（归一）
    find_heads = set()
    find_head_assertions = {}  # head -> assertion
    for b in ("positive", "negative", "uncertain", "other"):
        for f in fg.get(b, []):
            h = normalize_head(f.get("head", ""))
            a = (f.get("assertion", "") or "").lower().replace("measurement::", "")
            if h:
                find_heads.add(h)
                find_head_assertions[h] = a

    # 收集 impression heads（归一）
    imp_heads = set()
    for b in ("positive", "negative", "uncertain", "other"):
        for f in ig.get(b, []):
            h = normalize_head(f.get("head", ""))
            if h:
                imp_heads.add(h)
            # suggestive_of 的 target 也视为 impression 的诊断结论
            for s in (f.get("suggestive_of") or []):
                if isinstance(s, dict):
                    sh = normalize_head(s.get("finding", ""))
                    if sh:
                        imp_heads.add(sh)

    if not imp_heads:
        continue

    # 只测涌现：impression head 不在 finding head 里的
    emergent = imp_heads - find_heads
    if not emergent:
        continue

    n_samples += 1
    n_with_gold += 1

    for gold_h in emergent:
        # 找该样本 finding 里哪些 head 的知识表能覆盖这个 gold_h
        match_k = None  # 最小的 K 使得 gold_h 出现在候选里
        for find_h in find_heads:
            cands = knowledge.query_candidates(find_h, topk=None, finding_compact=fg)
            cand_heads = [c["target_head"] for c in cands]
            if gold_h not in cand_heads:
                continue
            # 找到此 finding head 下 gold 出现的最小 K
            found = False
            for k in [1, 3, 5, 10]:
                if gold_h in cand_heads[:k]:
                    if match_k is None or (isinstance(match_k, int) and k < match_k) or match_k == "all":
                        match_k = k
                    found = True
                    break
            if not found:
                if match_k is None:
                    match_k = "all"
                elif match_k != "all":
                    # 出现在候选里但不在 top10，统一标 all
                    match_k = "all"

        total_pairs += 1
        if match_k is not None:
            for k in [1, 3, 5, 10]:
                if match_k == "all" or match_k <= k:
                    correct_at_k[k] += 1
            if match_k == "all":
                correct_at_k["all"] += 1

        # 按 finding head 聚合
        for find_h in find_heads:
            by_finding_head[find_h]["total"] += 1
            cands = knowledge.query_candidates(find_h, topk=None, finding_compact=fg)
            cand_heads = [c["target_head"] for c in cands]
            if cand_heads and gold_h == cand_heads[0]:
                by_finding_head[find_h]["correct_top1"] += 1
            if gold_h in cand_heads[:5]:
                by_finding_head[find_h]["correct_top5"] += 1

print("=" * 60)
print("§4.3 知识表 Recall@K 评测（涌现场景）")
print("=" * 60)
print(f"样本数（有涌现impression head）: {n_samples}")
print(f"(finding_head, gold_head) 对总数: {total_pairs}")
print()
print(f"{'指标':20} {'值':>10}")
print("-" * 32)
for k in [1, 3, 5, 10, "all"]:
    rate = correct_at_k[k] / total_pairs * 100 if total_pairs else 0
    print(f"{'Recall@' + str(k):20} {rate:>8.2f}%  ({correct_at_k[k]}/{total_pairs})")
print()

# 按频次排序，看高频 finding head 的准确率
print("高频 finding head 的 Top1 正确率（只列发生 >=50 次的）:")
print(f"{'finding head':20} {'total':>6} {'top1%':>8} {'top5%':>8} {'最常见gold':>20}")
print("-" * 62)
for find_h, stats in sorted(by_finding_head.items(), key=lambda x: -x[1]["total"]):
    if stats["total"] >= 50:
        t1 = stats["correct_top1"] / stats["total"] * 100
        t5 = stats["correct_top5"] / stats["total"] * 100
        # 找最常见的 gold head
        print(f"{find_h:20} {stats['total']:6} {t1:7.1f}% {t5:7.1f}%")
