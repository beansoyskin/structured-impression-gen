# -*- coding: utf-8 -*-
"""§4.2.2 BM25 检索器：在 14 万结构化对上建索引，输入 finding 召回 TopK。

设计：
  - 索引键：tokenize_finding(finding_graph_compact) 的 token 列表（含 assertion）。
  - 索引值：记录 id + 对应 impression（compact + 序列化文本），供 §4.4 few-shot 用。
  - 持久化：索引构建一次（约 1-2 分钟），存 JSON，下次直接 load。
  - 查询：BM25 算分，返回 TopK。

依赖：rank-bm25（纯 Python，无 GPU）。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rank_bm25 import BM25Okapi  # noqa: E402

from src.retrieval.serialize import (  # noqa: E402
    tokenize_finding,
    serialize_finding,
    serialize_impression,
)


@dataclass
class RetrievalResult:
    """单条检索结果。"""
    rank: int
    record_id: str
    score: float
    finding_text: str        # 可读 finding（§4.4 展示用）
    impression_text: str     # 可读 impression（§4.4 展示用）
    impression_compact: dict # 结构化 impression（§4.5 验证参照用）


class CaseRetriever:
    """BM25 案例检索器。

    用法：
        # 构建索引（一次性）
        retriever = CaseRetriever.build_from_jsonl(data_path, index_path)
        # 或加载已有索引
        retriever = CaseRetriever.load(index_path)
        # 查询
        results = retriever.search(finding_graph_compact, topk=5)
    """

    def __init__(self, bm25: BM25Okapi, corpus: list[dict], id_list: list[str],
                 tokenized_corpus: list[list[str]] | None = None):
        self.bm25 = bm25
        self.corpus = corpus      # [{finding_text, impression_text, impression_compact}, ...]
        self.id_list = id_list    # 与 corpus 顺序对齐的 record id
        # tokenized_corpus 单独存（rank-bm25 不保留原始 token，持久化需要）
        self.tokenized_corpus = tokenized_corpus

    # ----- 构建 -----
    @classmethod
    def build_from_records(cls, records: list[dict]) -> "CaseRetriever":
        """从记录列表构建索引。
        records: 每条含 findings_graph_compact / impression_graph_compact / id。
        """
        tokenized_corpus = []
        corpus_meta = []
        id_list = []
        for obj in records:
            fg = obj.get("findings_graph_compact") or {}
            ig = obj.get("impression_graph_compact") or {}
            tokens = tokenize_finding(fg)
            # 跳过空 finding（无法检索，也会污染 BM25 统计）
            if not tokens:
                continue
            tokenized_corpus.append(tokens)
            corpus_meta.append({
                "finding_text": serialize_finding(fg),
                "impression_text": serialize_impression(ig),
                "impression_compact": ig,
            })
            id_list.append(obj.get("id", ""))
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(bm25, corpus_meta, id_list, tokenized_corpus)

    @classmethod
    def build_from_jsonl(cls, jsonl_path: str) -> "CaseRetriever":
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls.build_from_records(records)

    # ----- 持久化 -----
    def save(self, path: str):
        """存索引元数据。BM25 对象本身不保留原始 token（rank-bm25 无原生持久化），
        存 tokenized_corpus + corpus_meta + id_list，load 时重建 BM25Okapi。
        """
        data = {
            "id_list": self.id_list,
            "corpus_meta": self.corpus,
            "tokenized_corpus": self.tokenized_corpus,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CaseRetriever":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        bm25 = BM25Okapi(data["tokenized_corpus"])
        return cls(bm25, data["corpus_meta"], data["id_list"], data["tokenized_corpus"])

    # ----- 查询 -----
    def search(self, finding_graph_compact: dict, topk: int = 5) -> list[RetrievalResult]:
        """检索 TopK 相似病例。
        Args:
            finding_graph_compact: 查询 finding 的 compact graph。
            topk: 返回数量。
        Returns:
            RetrievalResult 列表（按 score 降序）。
        """
        query_tokens = tokenize_finding(finding_graph_compact)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        # 取 topk（argsort 降序）
        import numpy as np
        # 过滤 NaN/Inf
        scores = np.where(np.isfinite(scores), scores, -1e18)
        topk = min(topk, len(scores))
        # argpartition 取前 topk 再排序，比全排序快
        top_idx = np.argpartition(-scores, topk - 1)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results = []
        for rank, idx in enumerate(top_idx, 1):
            meta = self.corpus[int(idx)]
            results.append(RetrievalResult(
                rank=rank,
                record_id=self.id_list[int(idx)],
                score=float(scores[int(idx)]),
                finding_text=meta["finding_text"],
                impression_text=meta["impression_text"],
                impression_compact=meta["impression_compact"],
            ))
        return results


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _self_test():
    """用迷你语料自测：相同 finding 应召回自身或最相似的。"""
    mini = [
        # case A: 左下叶肺炎样表现
        {"id": "A", "findings_graph_compact": {
            "positive": [
                {"head": "consolidation", "assertion": "definitely present",
                 "locations": ["left lower lobe"]},
                {"head": "air bronchogram", "assertion": "definitely present",
                 "locations": ["left lower lobe"]},
            ],
            "negative": [{"head": "effusion", "assertion": "definitely absent"}],
            "uncertain": [], "other": [],
        }, "impression_graph_compact": {
            "positive": [{"head": "pneumonia", "assertion": "definitely present"}],
            "negative": [], "uncertain": [], "other": [],
        }},
        # case B: 正常
        {"id": "B", "findings_graph_compact": {
            "positive": [{"head": "normal", "assertion": "definitely present",
                          "locations": ["heart"]}],
            "negative": [{"head": "consolidation", "assertion": "definitely absent"},
                         {"head": "effusion", "assertion": "definitely absent"}],
            "uncertain": [], "other": [],
        }, "impression_graph_compact": {
            "negative": [{"head": "disease", "assertion": "definitely absent",
                          "locations": ["cardiopulmonary"]}],
            "positive": [], "uncertain": [], "other": [],
        }},
        # case C: 右上叶结节
        {"id": "C", "findings_graph_compact": {
            "positive": [{"head": "nodule", "assertion": "definitely present",
                          "locations": ["right upper lobe"]}],
            "negative": [], "uncertain": [], "other": [],
        }, "impression_graph_compact": {
            "uncertain": [{"head": "nodule", "assertion": "uncertain"}],
            "positive": [], "negative": [], "other": [],
        }},
    ]

    retriever = CaseRetriever.build_from_records(mini)

    # 查询1：类似 case A（左下叶实变）
    q1 = {"positive": [
        {"head": "consolidation", "assertion": "definitely present",
         "locations": ["left lower lobe"]},
    ], "negative": [], "uncertain": [], "other": []}
    res1 = retriever.search(q1, topk=3)
    assert len(res1) == 3, "应返回3条"
    assert res1[0].record_id == "A", f"最相似应为 A，实际 {res1[0].record_id} (score={res1[0].score:.2f})"
    assert "pneumonia" in res1[0].impression_text.lower(), "应召回带 pneumonia 的 impression"

    # 查询2：正常胸片
    q2 = {"positive": [{"head": "normal", "assertion": "definitely present",
                        "locations": ["heart"]}],
          "negative": [{"head": "consolidation", "assertion": "definitely absent"}],
          "uncertain": [], "other": []}
    res2 = retriever.search(q2, topk=3)
    assert res2[0].record_id == "B", f"正常胸片最相似应为 B，实际 {res2[0].record_id}"

    # 查询3：结节
    q3 = {"positive": [{"head": "nodule", "assertion": "definitely present",
                        "locations": ["right upper lobe"]}],
          "negative": [], "uncertain": [], "other": []}
    res3 = retriever.search(q3, topk=3)
    assert res3[0].record_id == "C", f"结节最相似应为 C，实际 {res3[0].record_id}"

    # 查询4：空 finding
    res4 = retriever.search({}, topk=3)
    assert res4 == [], "空 finding 应返回空"

    # 持久化测试
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        tmp = tf.name
    retriever.save(tmp)
    retriever2 = CaseRetriever.load(tmp)
    res_loaded = retriever2.search(q1, topk=1)
    assert res_loaded[0].record_id == "A", "load 后检索结果应一致"
    os.unlink(tmp)

    print("[OK] §4.2.2 BM25 检索器自测全部通过")
    print(f"     迷你语料 {len(mini)} 条")
    print("     查询1（左下叶实变）Top1:")
    print(f"       id={res1[0].record_id} score={res1[0].score:.3f}")
    print(f"       impression: {res1[0].impression_text}")


if __name__ == "__main__":
    _self_test()
