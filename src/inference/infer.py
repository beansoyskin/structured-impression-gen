# -*- coding: utf-8 -*-
"""§4.4 LLM 推理模块：结构化 finding → 结构化 impression。

整合 §4.0(归一化) + §4.2(检索) + §4.3(知识表) + §4.5(验证)，
用 LLM + schema 约束输出 5-元组 fact。

LLM 调用配置：
  - API URL: 环境变量 LLM_API_URL（默认 Ollama 地址）
  - 模型名: 环境变量 LLM_MODEL（默认 qwen3.5:9b）
  - API Key: 环境变量 LLM_API_KEY（可选，Ollama 本地无需）
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    import sys
    sys.path.insert(0, _ROOT)

from src.retrieval.serialize import serialize_finding, serialize_impression  # noqa: E402
from src.knowledge.suggestive_table import SuggestiveKnowledge  # noqa: E402
from src.retrieval.bm25_retriever import CaseRetriever  # noqa: E402


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEFAULT_API_URL = "https://d8rplvkp420c73e7s2tg-11434.agent.damodel.com/v1/chat/completions"
DEFAULT_MODEL = "qwen3:8b"


def _get_config():
    return {
        "api_url": os.environ.get("LLM_API_URL", DEFAULT_API_URL),
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "api_key": os.environ.get("LLM_API_KEY", ""),
    }


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------
def call_llm(messages: list[dict], temperature: float = 0.1, max_tokens: int = 4096) -> str:
    """调用 Ollama API（OpenAI 兼容格式）。

    Args:
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        temperature: 采样温度
        max_tokens: 最大输出 token
    Returns:
        模型输出文本（已去除 thinking 标签）
    """
    cfg = _get_config()
    payload = json.dumps({
        "model": cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    req = urllib.request.Request(cfg["api_url"], data=payload, headers=headers)
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")

    # Qwen3/3.5 的 thinking：reasoning 字段或 <think> 标签
    # content 可能包含 <think>...</think> 前缀，也可能是空的（reasoning 在 reasoning 字段）
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    # 如果 content 为空但 finish_reason=length，说明 thinking 吃完了 token
    # 不做额外处理，返回空（调用者需处理）

    return content


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a radiologist generating structured impression facts from chest X-ray findings.

CRITICAL RULE: ABSTRACT findings into diagnoses. Do NOT copy findings into impression.
- Example: finding "consolidation + air bronchogram" → impression "pneumonia"
- Example: finding "congestion + effusion" → impression "congestive heart failure"
- Example: all findings normal → impression "disease absent"

RULES:
1. Output a JSON array of fact objects with: head, assertion, locations, modifiers, suggestive_of.
2. assertion must be one of: "present", "absent", "uncertain".
3. head should be the DIAGNOSIS, not a raw observation.
4. If a finding suggests a specific diagnosis, add it in suggestive_of.
5. Keep impression concise: 1-3 facts. Do NOT list raw findings.
6. Output ONLY the JSON array, no explanation."""

# 系统示例：让模型理解 abstraction
EXAMPLE_FINDING = """+ consolidation | left lower lobe
+ air bronchogram | left lower lobe
- effusion"""
EXAMPLE_IMPRESSION = """[{"head": "pneumonia", "assertion": "present", "locations": ["left lower lobe"], "modifiers": [], "suggestive_of": []}]"""


def build_prompt(
    finding_compact: dict,
    retrieved_cases: list = None,
    suggestive_candidates: list = None,
) -> list[dict]:
    """构建 §4.4 的完整 prompt（system + few-shot + query）。

    Args:
        finding_compact: findings_graph_compact
        retrieved_cases: §4.2 检索结果（用于 few-shot）
        suggestive_candidates: §4.3 知识表候选（用于引导推断）
    Returns:
        messages 列表（供 call_llm 使用）
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Few-shot：用 hardcoded 示例教 abstraction
    messages.append({"role": "user", "content": "### Findings:\n" + EXAMPLE_FINDING + "\n### Impression:"})
    messages.append({"role": "assistant", "content": EXAMPLE_IMPRESSION})

    # 额外示例：正常情况
    norm_finding = """+ clear | lung
+ normal | heart size
- effusion
- consolidation"""
    norm_impression = """[{"head": "disease", "assertion": "absent", "locations": ["cardiopulmonary"], "modifiers": [], "suggestive_of": []}]"""
    messages.append({"role": "user", "content": "### Findings:\n" + norm_finding + "\n### Impression:"})
    messages.append({"role": "assistant", "content": norm_impression})

    # Few-shot：用检索到的相似病例作示例
    if retrieved_cases:
        for i, case in enumerate(retrieved_cases[:2]):  # 最多 2 个示例
            finding_text = case.finding_text if hasattr(case, "finding_text") else serialize_finding(case.get("finding_compact", {}))
            impression_text = case.impression_text if hasattr(case, "impression_text") else serialize_impression(case.get("impression_compact", {}))
            # 把可读 impression 转成 JSON fact 数组（简化：直接用 impression_text 作为参考）
            sugg_text = "None"
            messages.append({
                "role": "user",
                "content": FEW_SHOT_TEMPLATE.format(
                    n=i+1, finding_text=finding_text,
                    sugg_text=sugg_text,
                    impression_json=impression_text,
                ),
            })
            messages.append({"role": "assistant", "content": impression_text})

    # 构建当前 query
    finding_text = serialize_finding(finding_compact)

    # suggestive_of 候选
    sugg_lines = []
    if suggestive_candidates:
        for s in suggestive_candidates[:10]:
            sugg_lines.append(f"  - {s['target_head']} (confidence: {s['confidence']:.0%}, source_assertion: {s['source_assertion']})")
    sugg_text = "\n".join(sugg_lines) if sugg_lines else "None"

    query = f"""### Findings:
{finding_text}
### Suggestive_of candidates from knowledge:
{sugg_text}
### Impression:"""

    messages.append({"role": "user", "content": query})
    return messages


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------
def parse_facts(response_text: str) -> list[dict]:
    """从 LLM 输出中解析 fact 列表。

    尝试解析 JSON 数组；若失败则尝试提取 JSON 块。
    """
    # 尝试直接解析
    text = response_text.strip()
    # 提取 JSON 数组（可能被 markdown code block 包裹）
    json_match = re.search(r'\[[\s\S]*\]', text)
    if not json_match:
        return []
    try:
        facts = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(facts, list):
        return []

    # 规范化每个 fact
    valid = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        head = f.get("head", "").strip()
        if not head:
            continue
        assertion = f.get("assertion", "uncertain").strip().lower()
        if assertion not in ("present", "absent", "uncertain"):
            assertion = "uncertain"
        fact = {
            "head": head,
            "assertion": assertion,
            "locations": f.get("locations", []),
            "modifiers": f.get("modifiers", []),
            "suggestive_of": f.get("suggestive_of", []),
        }
        valid.append(fact)
    return valid


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------
@dataclass
class InferenceResult:
    facts: list[dict]            # 生成的结构化 impression facts
    raw_response: str             # LLM 原始输出
    prompt_messages: list[dict]   # 完整 prompt（调试用）


def infer_impression(
    finding_compact: dict,
    retriever: CaseRetriever = None,
    knowledge: SuggestiveKnowledge = None,
    topk_retrieval: int = 3,
    topk_knowledge: int = 10,
) -> InferenceResult:
    """§4.4 核心：从结构化 finding 推断结构化 impression。

    流程：
      1. §4.2 检索相似病例（few-shot）
      2. §4.3 查 suggestive_of 候选（引导推断）
      3. 构建 prompt + schema 约束
      4. 调用 LLM
      5. 解析输出

    Args:
        finding_compact: findings_graph_compact
        retriever: §4.2 检索器（可选）
        knowledge: §4.3 知识表（可选）
        topk_retrieval: 检索返回数
        topk_knowledge: 知识候选数
    Returns:
        InferenceResult
    """
    # 1. 检索
    retrieved = None
    if retriever:
        try:
            retrieved = retriever.search(finding_compact, topk=topk_retrieval)
        except Exception:
            retrieved = None

    # 2. 知识查询
    sugg_candidates = None
    if knowledge:
        # 对 finding 里所有 head 查候选
        all_candidates = []
        seen_targets = set()
        # 全局 finding 上下文（用于逆向逻辑）
        for bucket in ("positive", "negative", "uncertain", "other"):
            for f in finding_compact.get(bucket, []) or []:
                h = f.get("head") or ""
                if h:
                    cands = knowledge.query_candidates(h, topk=5, finding_compact=finding_compact)
                    for c in cands:
                        key = (c["target_head"], c["target_assertion"])
                        if key not in seen_targets:
                            seen_targets.add(key)
                            all_candidates.append(c)
        sugg_candidates = all_candidates[:topk_knowledge]

    # 3. 构建 prompt
    messages = build_prompt(finding_compact, retrieved, sugg_candidates)

    # 4. 调用 LLM
    response = call_llm(messages)

    # 5. 解析
    facts = parse_facts(response)

    return InferenceResult(
        facts=facts,
        raw_response=response,
        prompt_messages=messages,
    )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _self_test():
    # 不依赖检索和知识的最小测试
    finding = {
        "positive": [
            {"head": "consolidation", "assertion": "definitely present",
             "locations": ["left lower lobe"]},
            {"head": "air bronchogram", "assertion": "definitely present",
             "locations": ["left lower lobe"]},
        ],
        "negative": [{"head": "effusion", "assertion": "definitely absent"}],
        "uncertain": [], "other": [],
    }

    print("测试 LLM 推理（无 RAG / 无知识表）...")
    result = infer_impression(finding)
    print(f"  LLM 输出 fact 数: {len(result.facts)}")
    for f in result.facts:
        print(f"  - {f['assertion']:8} {f['head']:20} loc={f['locations']}")

    # 验证输出格式
    assert len(result.facts) > 0, "应至少生成1条fact"
    for f in result.facts:
        assert "head" in f and "assertion" in f, f"fact 缺少必要字段: {f}"
        assert f["assertion"] in ("present", "absent", "uncertain"), f"assertion 不合法: {f['assertion']}"

    print("\n[OK] §4.4 推理模块自测通过")


if __name__ == "__main__":
    _self_test()
