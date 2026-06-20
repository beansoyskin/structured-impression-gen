# -*- coding: utf-8 -*-
"""§4.2 评测：Recall@K。

相关定义（贴合下游用途）：
  检索目的是给 §4.4 提供"类似 finding 写出过什么 impression"的参照，
  所以"相关"= 检索结果的 impression head 集合 与 query 的 impression head 集合
  的 Jaccard 相似度 ≥ 阈值。

Recall@K = TopK 中至少 1 条相关的 query 占比。

关键处理：
  - 排除自身：query 自己在语料里必然 Top1 命中，必须排除，否则指标虚高。
  - 报告 relevant_coverage：测试集里存在至少 1 条相关病例的 query 比例，
    作为分母参考（若某 query finding 极独特，全语料无相关，Recall 恒 0）。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.norm.head_norm import normalize_head  # noqa: E402
from src.retrieval.bm25_retriever import CaseRetriever  # noqa: E402
from src.retrieval.serialize import tokenize_finding  # noqa: E402


def impression_head_set(impression_compact: dict) -> set[str]:
    """提取 impression 的归一 head 集合（含 suggestive_of 的 target）。

    含 suggestive_of：因为 impression 里 opacity >> pneumonia，
    pneumonia 也是该 impression 的诊断结论之一。
    """
    heads = set()
    for bucket in ("positive", "negative", "uncertain", "other"):
        for fact in impression_compact.get(bucket, []) or []:
            h = normalize_head(fact.get("head") or "")
            if h:
                heads.add(h)
            for s in (fact.get("suggestive_of") or []):
                if isinstance(s, dict):
                    sh = normalize_head(s.get("finding") or "")
                    if sh:
                        heads.add(sh)
    return heads


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluate_recall(
    retriever: CaseRetriever,
    test_records: list[dict],
    ks: list[int] = (1, 5, 10),
    threshold: float = 0.5,
    max_search: int = 50,
) -> dict:
    """评测 Recall@K。

    性能优化：
      - 预计算语料每条的 impression head 集合（避免每个 query 重复算）。
      - 直接操作 BM25 原始分数 + argpartition，绕开 search 的对象构造开销。
      - 单次排序结果复用给所有 K。

    Args:
        retriever: 已建好索引的检索器。
        test_records: 测试集记录（用其 finding 做 query，其 impression 做 gold）。
        ks: K 列表。
        threshold: Jaccard 阈值，>= 视为相关。
        max_search: 检索时取前 N 条用于判断"是否存在相关"（relevant_coverage）。
    Returns:
        {recall@1, recall@5, recall@10, relevant_coverage, n_queries, mean_jaccard_top1}
    """
    import numpy as np
    from src.retrieval.serialize import tokenize_finding

    # 预计算语料每条 impression head 集合（一次性，14万算一遍而非每query重算）
    corpus_heads = [impression_head_set(meta["impression_compact"]) for meta in retriever.corpus]
    # 语料 id → 索引（用于排除自身）
    id_to_idx = {rid: i for i, rid in enumerate(retriever.id_list)}

    k_max = max(max(ks), max_search)
    results = {f"recall@{k}": 0 for k in ks}
    n_with_relevant = 0
    n_valid = 0
    top1_jaccards = []

    for q in test_records:
        gold = impression_head_set(q.get("impression_graph_compact") or {})
        if not gold:
            continue
        n_valid += 1
        qid = q.get("id", "")

        # 直接算 BM25 分数（绕开 search 的对象构造）
        q_tokens = tokenize_finding(q.get("findings_graph_compact") or {})
        if not q_tokens:
            continue
        scores = retriever.bm25.get_scores(q_tokens)
        scores = np.where(np.isfinite(scores), scores, -1e18)

        # 取前 k_max 个候选（argpartition 比 argsort 快）
        n_take = min(k_max, len(scores))
        cand_idx = np.argpartition(-scores, n_take - 1)[:n_take]
        cand_idx = cand_idx[np.argsort(-scores[cand_idx])]

        # 排除自身
        self_idx = id_to_idx.get(qid)
        cand_list = [int(i) for i in cand_idx if int(i) != self_idx]

        # 一次计算所有候选的 jaccard，复用给所有 K
        cand_jaccards = [jaccard(gold, corpus_heads[i]) for i in cand_list]

        # relevant_coverage：max_search 内有相关
        if any(j >= threshold for j in cand_jaccards[:max_search]):
            n_with_relevant += 1

        # Recall@K：TopK 内至少 1 个相关
        for k in ks:
            if any(j >= threshold for j in cand_jaccards[:k]):
                results[f"recall@{k}"] += 1

        if cand_jaccards:
            top1_jaccards.append(cand_jaccards[0])

    out = {}
    for k in ks:
        out[f"recall@{k}"] = round(results[f"recall@{k}"] / n_valid, 4) if n_valid else 0.0
    out["relevant_coverage"] = round(n_with_relevant / n_valid, 4) if n_valid else 0.0
    out["n_queries"] = n_valid
    out["mean_jaccard_top1"] = round(sum(top1_jaccards) / len(top1_jaccards), 4) if top1_jaccards else 0.0
    return out


def _self_test():
    """用迷你语料自测评测逻辑。"""
    # 构造语料：3 条，A/B 有相似 impression（都含 pneumonia），C 不同
    corpus = [
        {"id": "A", "findings_graph_compact": {
            "positive": [{"head": "consolidation", "assertion": "definitely present",
                          "locations": ["left lower lobe"]}],
            "negative": [], "uncertain": [], "other": [],
        }, "impression_graph_compact": {
            "positive": [{"head": "pneumonia", "assertion": "definitely present"}],
            "negative": [], "uncertain": [], "other": [],
        }},
        {"id": "A2", "findings_graph_compact": {
            "positive": [{"head": "infiltrate", "assertion": "definitely present",
                          "locations": ["left lower lobe"]}],
            "negative": [], "uncertain": [], "other": [],
        }, "impression_graph_compact": {
            "positive": [{"head": "pneumonia", "assertion": "definitely present"}],
            "negative": [], "uncertain": [], "other": [],
        }},
        {"id": "C", "findings_graph_compact": {
            "positive": [{"head": "nodule", "assertion": "definitely present",
                          "locations": ["right upper lobe"]}],
            "negative": [], "uncertain": [], "other": [],
        }, "impression_graph_compact": {
            "uncertain": [{"head": "nodule", "assertion": "uncertain"}],
            "positive": [], "negative": [], "other": [],
        }},
    ]
    retriever = CaseRetriever.build_from_records(corpus)

    # 测试集 = 语料本身（验证排除自身 + 召回相似）
    res = evaluate_recall(retriever, corpus, ks=[1, 2], threshold=0.5, max_search=3)
    # query A: 排除自身后，A2 impression 同（pneumonia），应命中 → recall@1 应较高
    # query C: 排除自身后，A/A2 impression（pneumonia）与 C（nodule）Jaccard=0 → 不相关
    assert res["n_queries"] == 3, f"应有3个有效query，实际 {res['n_queries']}"
    # C 没有相关病例，所以 relevant_coverage 应 < 1
    assert res["relevant_coverage"] < 1.0, "C 应拉低 coverage"
    print("[OK] §4.2 评测自测通过")
    print(f"     {res}")


if __name__ == "__main__":
    _self_test()
