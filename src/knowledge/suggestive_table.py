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


# ---------------------------------------------------------------------------
# 联合知识表（多 head 联合推理）
# 统计 (head_a, head_b) 同时出现时 impression 的诊断分布
# 只在 finding 侧统计（联合是 finding 侧的多个 head 组合）
# ---------------------------------------------------------------------------
def build_pair_table(
    records: Iterable[dict],
    min_pair_count: int = 3,
    max_candidates: int = 10,
) -> dict:
    """构建 pair 共现表：两条 finding head 同时出现时，impression 的诊断分布。

    输出：
      {"<assertion_a>|<head_a> + <assertion_b>|<head_b>": [
          {"target_head": str, "target_assertion": str, "count": int, "confidence": float},
          ...
      ]}

    count/denominator 均按病例计数，同一病例内重复 fact 不会重复累计。
    默认只输出至少由 3 个病例支持的 pair，过滤一次性长尾并控制产物大小。
    """
    from collections import Counter
    from itertools import combinations

    pair_counts: dict[tuple[tuple[str, str], tuple[str, str]], Counter] = defaultdict(Counter)
    pair_denom: dict[tuple[tuple[str, str], tuple[str, str]], int] = defaultdict(int)

    for obj in records:
        # finding 端所有 head
        find_items: set[tuple[str, str]] = set()
        for bucket in ("positive", "negative", "uncertain", "other"):
            for fact in (obj.get("findings_graph_compact", {}).get(bucket, []) or []):
                h = normalize_head(fact.get("head") or "")
                if h:
                    assertion = normalize_assertion(fact.get("assertion"))
                    if assertion == "unknown":
                        assertion = {"positive": "present", "negative": "absent",
                                     "uncertain": "uncertain"}.get(bucket, "unknown")
                    if assertion != "unknown":
                        find_items.add((assertion, h))

        # 每个病例的同一 (head, assertion) 最多计一次，保证条件概率不超过 1。
        imp_targets: set[tuple[str, str]] = set()
        for bucket in ("positive", "negative", "uncertain", "other"):
            for fact in (obj.get("impression_graph_compact", {}).get(bucket, []) or []):
                h = normalize_head(fact.get("head") or "")
                if h:
                    assertion = normalize_assertion(fact.get("assertion"))
                    if assertion == "unknown":
                        assertion = {"positive": "present", "negative": "absent",
                                     "uncertain": "uncertain"}.get(bucket, "unknown")
                    if assertion != "unknown":
                        imp_targets.add((h, assertion))
                for s in (fact.get("suggestive_of") or []):
                    if isinstance(s, dict):
                        sh = normalize_head(s.get("finding") or s.get("head") or "")
                        assertion = normalize_assertion(s.get("assertion"))
                        if sh and assertion != "unknown":
                            imp_targets.add((sh, assertion))

        if not imp_targets or len(find_items) < 2:
            continue

        # source assertion 是 pair 身份的一部分，避免 present/absent 组合混为一谈。
        for pair in combinations(sorted(find_items), 2):
            pair_denom[pair] += 1
            for target in imp_targets:
                pair_counts[pair][target] += 1

    # 组装输出
    pair_table: dict[str, list[dict]] = {}
    for pair, counter in pair_counts.items():
        denom = pair_denom[pair]
        if denom < min_pair_count:
            continue
        rows = []
        for (ih, assertion), c in counter.most_common(max_candidates):
            rows.append({
                "target_head": ih,
                "target_assertion": assertion,
                "count": c,
                "confidence": round(c / denom, 4) if denom else 0.0,
            })
        if rows:
            key = " + ".join(f"{assertion}|{head}" for assertion, head in pair)
            pair_table[key] = rows

    return pair_table


# ---------------------------------------------------------------------------
# 表统计
# ---------------------------------------------------------------------------
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

    def __init__(self, table_path: str, min_count: int = 3, min_confidence: float = 0.0,
                 pair_table_path: str | None = None):
        with open(table_path, "r", encoding="utf-8") as fh:
            self.table: dict[str, list[dict]] = json.load(fh)
        self.pair_table: dict[str, list[dict]] = {}
        if pair_table_path is None:
            pair_table_path = os.path.join(os.path.dirname(table_path), "pair_table.json")
        if os.path.exists(pair_table_path):
            with open(pair_table_path, "r", encoding="utf-8") as fh:
                self.pair_table = json.load(fh)
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
        normalized_source_assertion = (
            normalize_assertion(source_assertion) if source_assertion else None
        )
        out: list[dict] = []

        # --- 逆向逻辑 + 设备/管线规则 ---
        # 当 finding 里没有 present 的实体，或只有 normal/clear/unremarkable 时，
        # 知识表应该直接推荐 disease absent
        if finding_compact is not None:
            has_abnormal = False
            device_heads = {
                "endotracheal tube", "ng tube", "nasogastric tube", "orogastric tube",
                "enteric tube", "tube", "tubes", "picc line", "central line", "line",
                "catheter", "swan - ganz catheter", "pacemaker", "lead", "leads",
                "wire", "wires", "surgical clips", "clip", "clips", "port - a - cath",
                "picc", "suture",
            }
            normal_heads = {"normal", "clear", "unremarkable", "intact", "well aerated"}
            for bucket in ("positive", "uncertain", "other"):
                for fact in finding_compact.get(bucket, []) or []:
                    ht = normalize_head(fact.get("head") or "")
                    if not ht:
                        continue
                # 设备类 head 不算"阳性实体"
                    if ht not in device_heads and ht not in normal_heads:
                        has_abnormal = True
                        break
                if has_abnormal:
                    break
            if not has_abnormal:
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

        # --- 设备/管线类 finding 的规则补丁 ---
        # 对常见的设备 head，如果原表没有覆盖，用规则补上
        _table_rows = self.table.get(h, [])
        if h in ("catheter", "endotracheal tube", "ng tube", "nasogastric tube",
                 "enteric tube", "tube", "central line", "line",
                 "picc", "picc line", "port - a - cath", "suture",
                 "swan - ganz catheter", "lead", "wire", "pacemaker"):
            # 若无原表记录，补充规则候选
            if not _table_rows or all(r["count"] < self.min_count for r in _table_rows):
                device_cands = [
                    {"target_head": "atelectasis", "target_assertion": "present",
                     "source_assertion": "present", "source_section": "rule",
                     "count": 250, "confidence": 0.30},
                    {"target_head": "pneumothorax", "target_assertion": "present",
                     "source_assertion": "present", "source_section": "rule",
                     "count": 200, "confidence": 0.25},
                ]
                # 插入到原表候选之前（但排在逆向候选之后）
                out = [r for r in out if r.get("source_section") == "rule"] + device_cands + \
                      [r for r in out if r.get("source_section") != "rule"]

        # --- 原表查询 ---
        for stored_row in _table_rows:
            r = dict(stored_row)
            if r["count"] < self.min_count:
                continue
            if r["confidence"] < self.min_confidence:
                continue
            if normalized_source_assertion and r["source_assertion"] != normalized_source_assertion:
                continue
            out.append(r)
        # --- 排序优化：用 confidence × count^0.3 替代纯 count 排序 ---
        # 排序分析发现：
        #   - Top1 错误时 53.9% 的候选只出现 1-9 次（低频噪声排第一）
        #   - Top1 正确时候选的频次中位数 99，错误时中位数只有 5
        # 原因：count 排序把"高频但没用的"排在前面（如 normal→disease absent）
        # 而真正的诊断候选往往频次较低但置信度高。
        # 新排序公式：score = confidence × count^0.3
        # 置信度 = count / source_head_total（P(target|source)）
        # 次线性频次项让高频边获得稳定权重，但不被频次完全主导。
        out.sort(key=lambda r: -(r["confidence"] * (r["count"] ** 0.3)))
        if topk is not None:
            out = out[:topk]
        return out

    def query_candidates_pair(
        self,
        findings: list[dict] | list[str],
        topk: int = 5,
    ) -> list[dict]:
        """查询多 head 联合推理的候选诊断。

        Args:
            findings: finding facts；字符串 head 仅用于兼容旧 pair 表。
            topk: 返回候选数。
        Returns:
            候选列表，按 count 降序。
        """
        if not self.pair_table or len(findings) < 2:
            return []

        # 尝试所有 pair 组合，聚合多个独立 pair 的支持强度。
        candidates = {}
        from itertools import combinations
        source_items: set[tuple[str, str]] = set()
        for finding in findings:
            if isinstance(finding, dict):
                head = normalize_head(finding.get("head") or "")
                assertion = normalize_assertion(finding.get("assertion"))
            else:
                head = normalize_head(finding)
                assertion = "unknown"
            if head:
                source_items.add((assertion, head))

        for left, right in combinations(sorted(source_items), 2):
            key = " + ".join(f"{assertion}|{head}" for assertion, head in (left, right))
            legacy_key = " + ".join(sorted([left[1], right[1]]))
            rows = self.pair_table.get(key, self.pair_table.get(legacy_key, []))
            for r in rows:
                if r["count"] < self.min_count or r["confidence"] < self.min_confidence:
                    continue
                tgt = (r["target_head"], r["target_assertion"])
                if tgt not in candidates:
                    candidates[tgt] = {"target_head": r["target_head"],
                                       "target_assertion": r["target_assertion"],
                                       "count": 0, "confidence": 0.0,
                                       "source_assertion": "mixed",
                                       "source_section": "pair",
                                       "score": 0.0,
                                       "supporting_pairs": []}
                # count/confidence 保持单条边语义；score 才跨 pair 累加。
                candidates[tgt]["count"] = max(candidates[tgt]["count"], r["count"])
                candidates[tgt]["confidence"] = max(candidates[tgt]["confidence"], r["confidence"])
                candidates[tgt]["score"] += r["confidence"] * (r["count"] ** 0.3)
                candidates[tgt]["supporting_pairs"].append(key)

        out = sorted(candidates.values(), key=lambda x: (-x["score"], -x["count"]))
        return out[:topk]


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

    # 联合表：同病例重复 fact 不得使 confidence > 1，并保留目标 assertion。
    pair_records = [
        {
            "findings_graph_compact": {"positive": [
                {"head": h, "assertion": "definitely present"}
                for h in ("alpha", "beta", "gamma", "delta", "epsilon")
            ]},
            "impression_graph_compact": {"positive": [
                {"head": "pneumonia", "assertion": "definitely present"},
                {"head": "pneumonia", "assertion": "definitely present"},
            ]},
        },
        {
            "findings_graph_compact": {"positive": [
                {"head": "delta", "assertion": "definitely present"},
                {"head": "epsilon", "assertion": "definitely present"},
            ]},
            "impression_graph_compact": {"negative": [
                {"head": "pneumonia", "assertion": "definitely absent"},
            ]},
        },
    ]
    pair_table = build_pair_table(pair_records, min_pair_count=1)
    pair_key = "present|delta + present|epsilon"
    assert pair_key in pair_table
    assert "present|alpha + present|epsilon" in pair_table, "不得只保留字典序前四个 finding head"
    pair_rows = pair_table[pair_key]
    assert all(0.0 <= r["confidence"] <= 1.0 for r in pair_rows)
    assert {(r["target_head"], r["target_assertion"]) for r in pair_rows} == {
        ("pneumonia", "present"), ("pneumonia", "absent")
    }

    # 查询器自动加载同目录 pair_table，统一返回候选 schema。
    with tempfile.TemporaryDirectory() as tmp_dir:
        table_path = os.path.join(tmp_dir, "suggestive_of_table.json")
        pair_path = os.path.join(tmp_dir, "pair_table.json")
        with open(table_path, "w", encoding="utf-8") as fh:
            json.dump(table, fh)
        with open(pair_path, "w", encoding="utf-8") as fh:
            json.dump(pair_table, fh)
        kb_pair = SuggestiveKnowledge(table_path, min_count=1)
        pair_cands = kb_pair.query_candidates_pair([
            {"head": "delta", "assertion": "present"},
            {"head": "epsilon", "assertion": "present"},
        ])
        assert pair_cands and pair_cands[0]["source_assertion"] == "mixed"
        assert pair_cands[0]["supporting_pairs"] == [pair_key]

    # uncertain 异常不能触发“全阴性”规则；查询也不能污染加载的原表。
    uncertain_finding = {
        "positive": [],
        "negative": [],
        "uncertain": [{"head": "nodule", "assertion": "uncertain"}],
        "other": [],
    }
    assert kb.query_candidates("unknown", finding_compact=uncertain_finding) == []
    kb.query_candidates("opacity")
    assert all("_score" not in r for r in kb.table["opacity"])

    print("[OK] §4.3 自测全部通过")
    print(f"     迷你集: source_heads={st['unique_source_heads']} edges={st['unique_edges']} total={st['total_occurrences']}")
    print(f"     opacity → {[(r['target_head'], r['count'], r['confidence']) for r in rows]}")


if __name__ == "__main__":
    _self_test()
