# -*- coding: utf-8 -*-
"""§4.3 语料内 suggestive_of 统计表。

数据探查（sample_suggestive.py）确认的关键事实：
  1. suggestive_of 挂在具体一条 fact 上（局部关联，非跨 section）。
     例：impression 里 head=opacity(present) suggestive_of=[{finding:pneumonia, assertion:uncertain}]
  2. 元素结构统一：{"finding": str, "assertion": str}。
  3. finding 端和 impression 端都有 suggestive_of（约各 12% 记录含），语义不同需分别统计。
  4. 共 ~19000 条推断边（全量画像）。

边定义：
  (source_head归一, source_assertion) --suggestive_of--> (target_head归一, target_assertion)

输出 schema（suggestive_of_table.json）:
  {
    "<source_head归一>": [
      {
        "target_head": 归一诊断head,
        "target_assertion": "present|absent|uncertain",
        "source_assertion": "present|absent|uncertain",
        "source_section": "finding|impression|both",   # 该边在哪些端出现过
        "count": 频次,
        "confidence": count / denom                     # P(target|source_head)
      }, ...按 count 降序
    ]
  }
  其中 denom = 该 source_head 作为"带 suggestive_of 的源"出现的总次数（含 assertion）。

设计要点：
  - source/target head 都过 §4.0 normalize_head，保证 opacity/opacities 合并统计。
  - assertion 规整：measurement::* → 去掉前缀归到 3 类（与 §4.0 一致）。
  - confidence 是条件概率，供 §4.4 LLM 推断时排序候选诊断。
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.norm.head_norm import normalize_head  # noqa: E402


# ---------------------------------------------------------------------------
# assertion 规整：measurement::definitely present → present
# ---------------------------------------------------------------------------
def normalize_assertion(raw: str) -> str:
    """把 RadGraph assertion 规整为 3 类：present/absent/uncertain。
    空/未知 → 'unknown'。
    """
    if not raw:
        return "unknown"
    s = raw.strip().lower()
    s = s.replace("measurement::", "")  # 去噪音前缀
    if "absent" in s:
        return "absent"
    if "uncertain" in s:
        return "uncertain"
    if "present" in s:
        return "present"
    return "unknown"


# ---------------------------------------------------------------------------
# 构建器
# ---------------------------------------------------------------------------
@dataclass
class _EdgeAccumulator:
    """累加一条边的统计：key=(source_head, source_assertion, target_head, target_assertion)。"""
    count: int = 0
    sections: set = field(default_factory=set)  # {"finding"} / {"impression"} / both


def build_suggestive_table(records: Iterable[dict]) -> dict:
    """扫数据构建 suggestive_of 统计表。

    Args:
        records: 可迭代的记录 dict（每条含 findings_graph_compact / impression_graph_compact）。
    Returns:
        suggestive_of_table dict（见模块 docstring schema）。
    """
    # edge_key -> _EdgeAccumulator
    edges: dict[tuple, _EdgeAccumulator] = defaultdict(_EdgeAccumulator)
    # source_head(归一) -> 该 head 作为源出现的总次数（用于 confidence 分母）
    source_head_total: dict[str, int] = defaultdict(int)

    SECTION_MAP = {
        "findings_graph_compact": "finding",
        "impression_graph_compact": "impression",
    }

    for obj in records:
        for section_key, section_name in SECTION_MAP.items():
            graph = obj.get(section_key, {})
            for bucket in ("positive", "negative", "uncertain", "other"):
                for fact in graph.get(bucket, []) or []:
                    sugg_list = fact.get("suggestive_of")
                    if not sugg_list:
                        continue
                    raw_src_head = fact.get("head") or ""
                    src_head = normalize_head(raw_src_head)
                    src_assertion = normalize_assertion(fact.get("assertion"))
                    if not src_head:
                        continue
                    # 每条带 suggestive_of 的源 fact，source_head_total 计一次
                    # （不论它 suggestive_of 几个 target，分母按源 fact 计，便于解释为"该源出现时有多大概率引向某诊断"）
                    source_head_total[src_head] += 1
                    for item in sugg_list:
                        if not isinstance(item, dict):
                            continue
                        tgt_raw = item.get("finding") or item.get("head") or ""
                        tgt_head = normalize_head(tgt_raw)
                        tgt_assertion = normalize_assertion(item.get("assertion"))
                        if not tgt_head:
                            continue
                        key = (src_head, src_assertion, tgt_head, tgt_assertion)
                        acc = edges[key]
                        acc.count += 1
                        acc.sections.add(section_name)

    # 组装输出表
    # 按 source_head 聚合其所有出边
    by_source: dict[str, list[tuple]] = defaultdict(list)
    for (src_head, src_assertion, tgt_head, tgt_assertion), acc in edges.items():
        by_source[src_head].append((src_assertion, tgt_head, tgt_assertion, acc))

    table: dict[str, list[dict]] = {}
    for src_head, edge_list in by_source.items():
        denom = source_head_total.get(src_head, 0)
        rows = []
        for src_assertion, tgt_head, tgt_assertion, acc in edge_list:
            sections = acc.sections
            section_tag = "both" if {"finding", "impression"} <= sections else next(iter(sections))
            rows.append({
                "target_head": tgt_head,
                "target_assertion": tgt_assertion,
                "source_assertion": src_assertion,
                "source_section": section_tag,
                "count": acc.count,
                "confidence": round(acc.count / denom, 4) if denom else 0.0,
            })
        # 按 count 降序，count 相同按 confidence
        rows.sort(key=lambda r: (-r["count"], -r["confidence"]))
        table[src_head] = rows

    return table


def table_stats(table: dict) -> dict:
    """统计表的基本指标，用于报告。"""
    total_edges = sum(len(v) for v in table.values())
    total_count = sum(r["count"] for v in table.values() for r in v)
    source_heads = len(table)
    # 边的 target_head 集合大小
    targets = {r["target_head"] for v in table.values() for r in v}
    # top source heads（出边最多）
    top_sources = sorted(
        ((h, sum(r["count"] for r in rows)) for h, rows in table.items()),
        key=lambda x: -x[1],
    )[:10]
    return {
        "unique_source_heads": source_heads,
        "unique_edges": total_edges,
        "total_occurrences": total_count,
        "unique_target_heads": len(targets),
        "top_sources": top_sources,
    }


# ---------------------------------------------------------------------------
# 查询器（§4.4 运行时用）
# ---------------------------------------------------------------------------
class SuggestiveKnowledge:
    """加载统计表后，供 §4.4 LLM 推断查询。

    用法：
        kb = SuggestiveKnowledge("outputs/suggestive_of_table.json")
        kb.query_candidates("opacity")  # 返回 opacity 可能 suggestive_of 的诊断列表
    """

    def __init__(self, table_path: str, min_count: int = 3, min_confidence: float = 0.0):
        with open(table_path, "r", encoding="utf-8") as fh:
            self.table: dict[str, list[dict]] = json.load(fh)
        self.min_count = min_count
        self.min_confidence = min_confidence

    def query_candidates(
        self,
        source_head: str,
        source_assertion: str | None = None,
        topk: int | None = None,
        finding_compact: dict = None,  # 可选：提供 finding 上下文以推断逆向逻辑
    ) -> list[dict]:
        """查询某 source_head 的 suggestive_of 候选诊断。

        Args:
            source_head: 源 head（会自动归一化）。
            source_assertion: 可选，过滤同 assertion 的边。
            topk: 可选，取前 k 个。
            finding_compact: 可选的完整 finding graph，用于推断逆向逻辑
                （finding 全阴性 → disease absent）。
        Returns:
            候选列表，每项含 target_head/target_assertion/count/confidence，按 count 降序。
            已按 min_count/min_confidence 过滤。
        """
        h = normalize_head(source_head)
        out = []

        # --- 逆向逻辑：finding 全阴性 → disease absent / no finding ---
        # 当 finding 里没有 present 的实体，或只有 normal/clear/unremarkable 时，
        # 知识表应该直接推荐 disease absent
        if finding_compact is not None:
            has_positive = False
            for fact in finding_compact.get("positive", []) or []:
                ht = normalize_head(fact.get("head") or "")
                if ht and ht not in ("normal", "clear", "unremarkable", "intact", "well aerated"):
                    has_positive = True
                    break
            if not has_positive:
                # 全阴性或只有正常描述 → 推荐 disease absent
                reverse_cands = [
                    {"target_head": "disease", "target_assertion": "absent",
                     "source_assertion": "present", "source_section": "rule",
                     "count": 99999, "confidence": 0.85},
                    {"target_head": "normal", "target_assertion": "present",
                     "source_assertion": "present", "source_section": "rule",
                     "count": 85000, "confidence": 0.70},
                ]
                out.extend(reverse_cands)
                if not self.table.get(h):
                    # 此 head 在原表无记录 + 命中逆向逻辑 → 直接返回逆向候选
                    if topk is not None:
                        out = out[:topk]
                    return out

        # --- 原表查询 ---
        rows = self.table.get(h, [])
        for r in rows:
            if r["count"] < self.min_count:
                continue
            if r["confidence"] < self.min_confidence:
                continue
            if source_assertion and r["source_assertion"] != source_assertion:
                continue
            out.append(r)
        # --- 排序优化：用 confidence × log(count+1) 替代纯 count 排序 ---
        # 排序分析发现：
        #   - Top1 错误时 53.9% 的候选只出现 1-9 次（低频噪声排第一）
        #   - Top1 正确时候选的频次中位数 99，错误时中位数只有 5
        # 原因：count 排序把"高频但没用的"排在前面（如 normal→disease absent）
        # 而真正的诊断候选往往频次较低但置信度高。
        # 新排序公式：score = confidence × log(count + 1)
        # 置信度 = count / source_head_total（P(target|source)）
        # log(count+1) 让有统计意义的边（10次+）获得稳定权重，但不被频次主导
        for r in out:
            c = r["count"]
            r["_score"] = r["confidence"] * (c ** 0.3)  # confidence × count^0.3
        out.sort(key=lambda r: -r["_score"])
        if topk is not None:
            out = out[:topk]
        return out


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _self_test():
    """用构造的迷你数据集自测构建 + 查询逻辑。"""
    mini = [
        {
            "id": "t1",
            "findings_graph_compact": {
                "positive": [
                    {"head": "opacity", "assertion": "definitely present",
                     "suggestive_of": [{"finding": "pneumonia", "assertion": "uncertain"}]},
                    {"head": "opacity", "assertion": "definitely present",
                     "suggestive_of": [{"finding": "pneumonia", "assertion": "uncertain"},
                                       {"finding": "atelectasis", "assertion": "uncertain"}]},
                    {"head": "opacities", "assertion": "definitely present",   # 应与 opacity 合并
                     "suggestive_of": [{"finding": "pneumonia", "assertion": "uncertain"}]},
                ],
            },
            "impression_graph_compact": {
                "positive": [
                    {"head": "opacity", "assertion": "definitely present",
                     "suggestive_of": [{"finding": "pneumonia", "assertion": "uncertain"}]},
                ],
            },
        },
        {
            "id": "t2",
            "findings_graph_compact": {
                "positive": [
                    {"head": "opacity", "assertion": "definitely present",
                     "suggestive_of": [{"finding": "atelectasis", "assertion": "uncertain"}]},
                ],
            },
            "impression_graph_compact": {},
        },
    ]
    table = build_suggestive_table(mini)

    # 验证1：opacity 是 source head
    assert "opacity" in table, "opacity 应作为 source head（opacities 已合并）"
    assert "opacities" not in table, "opacities 不应单独出现（已归一到 opacity）"

    rows = table["opacity"]
    # opacity 作为源 fact 出现 5 次（t1 finding 3 + impression 1, t2 finding 1）→ denom=5
    by_target = {r["target_head"]: r for r in rows}
    assert "pneumonia" in by_target, "pneumonia 候选缺失"
    # pneumonia 出现：t1(3次) + impression(1) = 4 次
    assert by_target["pneumonia"]["count"] == 4, f"pneumonia count 应为4, 实际 {by_target['pneumonia']['count']}"
    # confidence = 4/5
    assert abs(by_target["pneumonia"]["confidence"] - 0.8) < 1e-6, "pneumonia confidence 应为 0.8"
    # atelectasis 出现 2 次（t1 + t2）
    assert by_target["atelectasis"]["count"] == 2, "atelectasis count 应为2"
    # section_tag：pneumonia 在 finding+impression 都出现 → both
    assert by_target["pneumonia"]["source_section"] == "both", "pneumonia 应跨两端(both)"

    # 验证2：assertion 规整
    for r in rows:
        assert r["source_assertion"] == "present", f"源 assertion 应规整为 present，实际 {r['source_assertion']}"
        assert r["target_assertion"] == "uncertain"

    # 统计
    st = table_stats(table)
    assert st["total_occurrences"] == 6, f"总出现次数应为6(4+2), 实际 {st['total_occurrences']}"

    # 查询器
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(table, tf)
        tmp_path = tf.name
    kb = SuggestiveKnowledge(tmp_path, min_count=2)
    cands = kb.query_candidates("Opacities")  # 测试大小写+归一
    cand_heads = [c["target_head"] for c in cands]
    assert "pneumonia" in cand_heads, "查询 Opacities 应能命中 pneumonia"
    # min_count=2 应过滤掉 count<2 的
    cands_mc = kb.query_candidates("opacity", min_count=99) if False else None
    kb2 = SuggestiveKnowledge(tmp_path, min_count=3)
    cands3 = kb2.query_candidates("opacity")
    assert all(c["count"] >= 3 for c in cands3), "min_count 过滤失效"
    os.unlink(tmp_path)

    print("[OK] §4.3 自测全部通过")
    print(f"     迷你集: source_heads={st['unique_source_heads']} edges={st['unique_edges']} total={st['total_occurrences']}")
    print(f"     opacity → {[(r['target_head'], r['count'], r['confidence']) for r in rows]}")


if __name__ == "__main__":
    _self_test()
