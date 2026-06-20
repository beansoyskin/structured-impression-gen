# -*- coding: utf-8 -*-
"""§4.2.1 finding 序列化器：把 findings_graph_compact → 归一化 token 序列。

设计决策（用户确认）：
  - 含 assertion：序列化保留阴阳性，让检索能区分"有实变"vs"无实变"。
  - 复用 §4.0 归一化：head 过 normalize_head，location 过 normalize_location。

序列化策略（为 BM25 设计，BM25 基于词频，所以要把各维度拆成独立 token）：
  每个 fact → 一组 token：
    [assertion_token, head_token, loc_tokens...]
  其中：
    - assertion_token: "asp" (absent/present) + "pre"/"abs"/"unc" → 如 "asppre"、"aspabs"
      用前缀避免与 head 词冲突，且让 BM25 把"阳性集合"和"阴性集合"区分开。
    - head_token: normalize_head 后的 head，多词短语用下划线连，如 "left_lower_lobe"→
      不，head 本身是诊断词不含解剖，直接用归一 head（可能多词用下划线连）。
    - loc_tokens: 每个 location 拆成 laterality/region/lobe 三个维度 token：
      "lat_left"、"reg_lung"、"lobe_lower"，让"左肺下叶"与"右肺下叶"在 region/lobe 维度
      仍能匹配（共享 reg/lobe token），但 laterality 区分。

这样 BM25 召回语义：
  - 完全相同结构 → 高分（所有 token 命中）
  - 同部位不同侧 → 中分（reg/lobe token 命中，lat token 不同）
  - 同阴阳性不同实体 → 中分（assertion token 命中）

输出两种形式：
  - serialize_finding(record) -> str : 人类可读紧凑文本（§4.4 LLM few-shot 展示用）
  - tokenize_finding(record) -> list[str] : token 列表（BM25 索引用）
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.norm.head_norm import normalize_head  # noqa: E402
from src.norm.location_norm import normalize_location, normalize_location_list  # noqa: E402

# assertion 简写前缀（与 §4.3 一致的 3 类规整）
def _assertion_short(raw_assertion: str) -> str:
    if not raw_assertion:
        return "unc"
    s = raw_assertion.lower()
    s = s.replace("measurement::", "")
    if "absent" in s:
        return "abs"
    if "uncertain" in s:
        return "unc"
    if "present" in s:
        return "pre"
    return "unc"


def _assertion_token(raw_assertion: str) -> str:
    """BM25 token: 'asppre' / 'aspabs' / 'aspunc'。"""
    return "asp" + _assertion_short(raw_assertion)


def _loc_tokens(loc_list) -> list[str]:
    """一个 fact 的 locations → laterality/region/lobe 维度 token 列表。"""
    norms = normalize_location_list(loc_list)
    tokens = []
    for ln in norms:
        tokens.append("lat_" + ln.laterality)
        tokens.append("reg_" + ln.region)
        if ln.lobe != "none":
            tokens.append("lobe_" + ln.lobe)
    return tokens


def _head_token(raw_head: str) -> str:
    """head → 归一 token，多词用下划线连。"""
    h = normalize_head(raw_head)
    return h.replace(" ", "_") if h else ""


def tokenize_finding(graph_compact: dict) -> list[str]:
    """把 findings_graph_compact → BM25 token 列表。

    每个 fact 贡献: [assertion_token, head_token, *loc_tokens]
    顺序无关（BM25 用词频，不看顺序）。
    """
    tokens = []
    for bucket in ("positive", "negative", "uncertain", "other"):
        for fact in graph_compact.get(bucket, []) or []:
            ht = _head_token(fact.get("head") or "")
            at = _assertion_token(fact.get("assertion") or "")
            if ht:
                tokens.append(at)
                tokens.append(ht)
            tokens.extend(_loc_tokens(fact.get("locations") or []))
    return tokens


def serialize_finding(graph_compact: dict) -> str:
    """人类可读紧凑文本（§4.4 LLM few-shot 展示用，非 BM25 索引用）。

    格式：
      + opacity | left lower lobe
      - effusion |
      = nodule | right upper lobe
    其中 +/-/= 对应 present/absent/uncertain。
    """
    sym_map = {"pre": "+", "abs": "-", "unc": "="}
    lines = []
    for bucket in ("positive", "negative", "uncertain", "other"):
        for fact in graph_compact.get(bucket, []) or []:
            ht = normalize_head(fact.get("head") or "")
            if not ht:
                continue
            sym = sym_map.get(_assertion_short(fact.get("assertion") or ""), "=")
            loc_strs = []
            for ln in normalize_location_list(fact.get("locations") or []):
                parts = []
                if ln.laterality != "none":
                    parts.append(ln.laterality)
                if ln.region != "other":
                    parts.append(ln.region)
                if ln.lobe != "none":
                    parts.append(ln.lobe)
                if parts:
                    loc_strs.append(" ".join(parts))
            loc_text = "; ".join(loc_strs) if loc_strs else ""
            lines.append(f"{sym} {ht} | {loc_text}".rstrip("| ").rstrip())
    return "\n".join(lines)


def serialize_impression(graph_compact: dict) -> str:
    """impression 可读文本（含 suggestive_of）。"""
    sym_map = {"pre": "+", "abs": "-", "unc": "="}
    lines = []
    for bucket in ("positive", "negative", "uncertain", "other"):
        for fact in graph_compact.get(bucket, []) or []:
            ht = normalize_head(fact.get("head") or "")
            if not ht:
                continue
            sym = sym_map.get(_assertion_short(fact.get("assertion") or ""), "=")
            sugg = fact.get("suggestive_of") or []
            sugg_strs = []
            for s in sugg:
                if isinstance(s, dict):
                    sh = normalize_head(s.get("finding") or "")
                    if sh:
                        sugg_strs.append(sh)
            sugg_text = f"  >> {', '.join(sugg_strs)}" if sugg_strs else ""
            lines.append(f"{sym} {ht}{sugg_text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _self_test():
    sample_finding = {
        "positive": [
            {"head": "opacities", "assertion": "definitely present",
             "locations": ["left lower lobe"]},
            {"head": "consolidation", "assertion": "definitely present",
             "locations": ["left lower lobe"]},
        ],
        "negative": [
            {"head": "effusion", "assertion": "definitely absent", "locations": []},
            {"head": "pleural effusions", "assertion": "definitely absent",
             "locations": ["right"]},
        ],
        "uncertain": [
            {"head": "nodule", "assertion": "uncertain",
             "locations": ["right upper lobe"]},
        ],
        "other": [],
    }

    # token 测试
    tokens = tokenize_finding(sample_finding)
    # opacities 应归一为 opacity → head_token "opacity"
    assert "opacity" in tokens, f"opacity 归一丢失: {tokens}"
    # assertion token
    assert "asppre" in tokens, "阳性 assertion token 缺失"
    assert "aspabs" in tokens, "阴性 assertion token 缺失"
    assert "aspunc" in tokens, "不确定 assertion token 缺失"
    # location 维度 token
    assert "lat_left" in tokens and "reg_lobe" in tokens and "lobe_lower" in tokens, \
        f"left lower lobe 维度 token 缺失: {tokens}"
    assert "lat_right" in tokens, "right laterality token 缺失"
    # pleural effusions: head 应归一（effusions→effusion），但 pleural 是 location
    # 注意：fact 里 head="pleural effusions" 归一后可能是 "pleural effusion"
    assert any(t.startswith("effusion") for t in tokens), "effusion head 归一丢失"

    # 可读序列化测试
    text = serialize_finding(sample_finding)
    assert "+ opacity | left lobe lower" in text or "+ opacity" in text, f"可读文本异常:\n{text}"
    assert "- effusion" in text, "阴性行缺失"

    # impression 序列化
    sample_impression = {
        "positive": [
            {"head": "opacity", "assertion": "definitely present",
             "suggestive_of": [{"finding": "pneumonia", "assertion": "uncertain"}]},
        ],
        "negative": [],
        "uncertain": [],
        "other": [],
    }
    imp_text = serialize_impression(sample_impression)
    assert ">> pneumonia" in imp_text, f"suggestive_of 未序列化:\n{imp_text}"

    # 空输入
    assert tokenize_finding({}) == []
    assert serialize_finding({}) == ""

    print("[OK] §4.2.1 序列化器自测全部通过")
    print("     sample finding tokens:", tokens)
    print("     可读文本:")
    print("    " + serialize_finding(sample_finding).replace("\n", "\n     "))
    print("     impression 可读文本:")
    print("    " + serialize_impression(sample_impression))


if __name__ == "__main__":
    _self_test()
