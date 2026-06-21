# -*- coding: utf-8 -*-
"""端到端评测：§4.4 推理 + §4.5 验证，测 fact 级指标。

用真实数据做 test：输入 finding → §4.4 推理出 impression → 与真实 impression 对比。
对比方法：将生成的 fact 和真实 fact 归一化后，计算与旧方法（v3 规则）同口径的指标。

指标定义（与 v3 保持一致）：
  - head/entity_f1: head 匹配的 F1
  - head_assertion_f1: (head, assertion) 匹配的 F1
  - head_assertion_location_f1: (head, assertion, location) 匹配的 F1
  - exact_fact_f1: 5-元组完全匹配的 F1
  - location_error_rate: head+assertion 匹配但 location 错误的比例

用法：python -m scripts.eval_e2e [数据路径] [n_samples] [use_rag]
  use_rag: 0=无RAG无知识, 1=有知识表, 2=有知识表+检索（默认1）
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.norm.head_norm import normalize_head  # noqa: E402
from src.norm.location_norm import normalize_location_list  # noqa: E402
from src.inference.infer import infer_impression  # noqa: E402
from src.verify.verifier import verify_impression  # noqa: E402
from src.knowledge.suggestive_table import SuggestiveKnowledge  # noqa: E402

DATA_DEFAULT = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"
SUGGESTIVE_TABLE_PATH = os.path.join(_ROOT, "outputs", "suggestive_of_table.json")


# ---------------------------------------------------------------------------
# fact 归一化（用于对比）
# ---------------------------------------------------------------------------
def _norm_assertion(raw: str) -> str:
    if not raw:
        return "unknown"
    s = raw.lower().replace("measurement::", "")
    if "absent" in s: return "absent"
    if "uncertain" in s: return "uncertain"
    if "present" in s: return "present"
    return "unknown"


def normalize_fact_for_eval(fact: dict) -> dict:
    """把一条 fact 归一化为可对比的标准形式。"""
    head = normalize_head(fact.get("head") or "")
    assertion = _norm_assertion(fact.get("assertion") or "")
    loc_norms = normalize_location_list(fact.get("locations") or [])
    loc_set = set()
    for ln in loc_norms:
        parts = []
        if ln.laterality != "none":
            parts.append(ln.laterality)
        if ln.region != "other":
            parts.append(ln.region)
        if ln.lobe != "none":
            parts.append(ln.lobe)
        if parts:
            loc_set.add(tuple(sorted(parts)))
    mods = tuple(sorted(normalize_head(m) for m in (fact.get("modifiers") or []) if m))
    sugg = []
    for s in (fact.get("suggestive_of") or []):
        if isinstance(s, dict):
            sh = normalize_head(s.get("head") or s.get("finding") or "")
            sa = _norm_assertion(s.get("assertion") or "")
            if sh:
                sugg.append((sh, sa))
        elif isinstance(s, str):
            sh = normalize_head(s)
            if sh:
                sugg.append((sh, "uncertain"))
    sugg = tuple(sorted(sugg))
    return {
        "head": head,
        "assertion": assertion,
        "locations": frozenset(loc_set) if loc_set else frozenset(),
        "modifiers": mods,
        "suggestive_of": sugg,
    }


def extract_facts_from_graph(graph_compact: dict) -> list[dict]:
    """从 compact graph 提取所有 fact。"""
    facts = []
    for bucket in ("positive", "negative", "uncertain", "other"):
        for f in graph_compact.get(bucket, []) or []:
            facts.append(f)
    return facts


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def compute_metrics(pred_facts: list[dict], gold_facts: list[dict]) -> dict:
    """计算 fact 级指标。"""
    # 归一化
    pred_norms = [normalize_fact_for_eval(f) for f in pred_facts]
    gold_norms = [normalize_fact_for_eval(f) for f in gold_facts]

    # 各级 key
    def head_key(nf): return nf["head"]
    def ha_key(nf): return (nf["head"], nf["assertion"])
    def hal_key(nf): return (nf["head"], nf["assertion"], nf["locations"])
    def exact_key(nf): return (nf["head"], nf["assertion"], nf["locations"], nf["modifiers"], nf["suggestive_of"])

    def f1_from_keys(pred_keys, gold_keys):
        pred_c = Counter(pred_keys)
        gold_c = Counter(gold_keys)
        # true positive = min count for each matching key
        tp = sum(min(pred_c[k], gold_c[k]) for k in set(pred_c) & set(gold_c))
        fp = sum(pred_c.values()) - tp
        fn = sum(gold_c.values()) - tp
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return f1, prec, rec

    head_f1, _, _ = f1_from_keys([head_key(p) for p in pred_norms], [head_key(g) for g in gold_norms])
    ha_f1, _, _ = f1_from_keys([ha_key(p) for p in pred_norms], [ha_key(g) for g in gold_norms])
    hal_f1, _, _ = f1_from_keys([hal_key(p) for p in pred_norms], [hal_key(g) for g in gold_norms])
    exact_f1, _, _ = f1_from_keys([exact_key(p) for p in pred_norms], [exact_key(g) for g in gold_norms])

    # location_error_rate: head+assertion 匹配但 location 不匹配的比例
    ha_match = 0
    ha_loc_wrong = 0
    pred_ha = Counter([ha_key(p) for p in pred_norms])
    gold_ha = Counter([ha_key(g) for g in gold_norms])
    for k in set(pred_ha) & set(gold_ha):
        n_match = min(pred_ha[k], gold_ha[k])
        ha_match += n_match
        # 其中 location 有多少匹配
        pred_locs = [p["locations"] for p in pred_norms if ha_key(p) == k]
        gold_locs = [g["locations"] for g in gold_norms if ha_key(g) == k]
        pred_loc_c = Counter(pred_locs)
        gold_loc_c = Counter(gold_locs)
        loc_match = sum(min(pred_loc_c[lk], gold_loc_c[lk]) for lk in set(pred_loc_c) & set(gold_loc_c))
        ha_loc_wrong += (n_match - loc_match)
    loc_err = ha_loc_wrong / ha_match if ha_match else 0.0

    return {
        "head/entity_f1": round(head_f1, 5),
        "head_assertion_f1": round(ha_f1, 5),
        "head_assertion_location_f1": round(hal_f1, 5),
        "exact_fact_f1": round(exact_f1, 5),
        "location_error_rate": round(loc_err, 5),
    }


# ---------------------------------------------------------------------------
# 主评测
# ---------------------------------------------------------------------------
def main(data_path: str, n_samples: int, use_rag: int) -> int:
    rng = random.Random(42)

    # 加载知识表（use_rag >= 1）
    knowledge = None
    if use_rag >= 1 and os.path.exists(SUGGESTIVE_TABLE_PATH):
        print(f"加载知识表: {SUGGESTIVE_TABLE_PATH}")
        knowledge = SuggestiveKnowledge(SUGGESTIVE_TABLE_PATH, min_count=3)

    # 读取样本
    print(f"读取数据: {data_path}")
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    sample = rng.sample(records, min(n_samples, len(records)))

    # 评测
    rag_label = ["无RAG无知识", "有知识表", "有知识表+检索"][use_rag]
    print(f"\n{'='*60}")
    print(f"§4.4 端到端评测 | 模型={os.environ.get('LLM_MODEL', 'qwen3:8b')} | RAG={rag_label}")
    print(f"样本数={len(sample)}")
    print(f"{'='*60}")

    all_metrics = []
    n_success = 0
    n_empty = 0
    n_verify_passed = 0
    t_total = 0

    for i, rec in enumerate(sample):
        fg = rec.get("findings_graph_compact") or {}
        ig = rec.get("impression_graph_compact") or {}
        if not fg or not ig:
            continue
        if not any(ig.get(b) for b in ("positive", "negative", "uncertain", "other")):
            continue

        t0 = time.time()
        result = infer_impression(fg, knowledge=knowledge)
        t1 = time.time()
        t_total += (t1 - t0)

        if not result.facts:
            n_empty += 1
            # 空 pred vs gold
            gold_facts = extract_facts_from_graph(ig)
            m = compute_metrics([], gold_facts)
            all_metrics.append(m)
            continue

        n_success += 1

        # 验证
        # 把 pred facts 组装成 impression_graph_compact 格式给 verifier
        pred_compact = {"positive": [], "negative": [], "uncertain": [], "other": []}
        for f in result.facts:
            a = f.get("assertion", "uncertain")
            if a == "present":
                pred_compact["positive"].append(f)
            elif a == "absent":
                pred_compact["negative"].append(f)
            else:
                pred_compact["uncertain"].append(f)
        trace = verify_impression(pred_compact, fg)
        if trace.passed:
            n_verify_passed += 1

        # 计算指标
        gold_facts = extract_facts_from_graph(ig)
        m = compute_metrics(result.facts, gold_facts)
        all_metrics.append(m)

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(sample)} ({(i+1)/len(sample)*100:.0f}%)", flush=True)

    # 汇总
    n_valid = len(all_metrics)
    avg = {}
    for key in all_metrics[0] if all_metrics else []:
        avg[key] = round(sum(m[key] for m in all_metrics) / n_valid, 5) if n_valid else 0.0

    print(f"\n{'='*60}")
    print(f"结果 | 模型={os.environ.get('LLM_MODEL', 'qwen3:8b')} | RAG={rag_label}")
    print(f"{'='*60}")
    print(f"总样本     : {n_valid}")
    print(f"成功推理   : {n_success} ({n_success/n_valid*100:.1f}%)") if n_valid else None
    print(f"空输出     : {n_empty}")
    print(f"验证通过   : {n_verify_passed}/{n_success}") if n_success else None
    print(f"平均耗时   : {t_total/n_valid:.1f}s/样本") if n_valid else None
    print(f"\n指标对比:")
    print(f"  {'指标':30} {'新pipeline':>10} {'v3规则':>10}")
    print(f"  {'-'*52}")
    v3_baseline = {
        "exact_fact_f1": 0.22598,
        "head_assertion_location_f1": 0.26061,
        "head_assertion_f1": 0.39404,
        "head/entity_f1": 0.44292,
        "location_error_rate": 0.34182,
    }
    for key in ["exact_fact_f1", "head_assertion_location_f1", "head_assertion_f1", "head/entity_f1", "location_error_rate"]:
        new_val = avg.get(key, 0)
        old_val = v3_baseline.get(key, 0)
        diff = new_val - old_val
        sign = "+" if diff > 0 else ""
        print(f"  {key:30} {new_val:>10.5f} {old_val:>10.5f}  ({sign}{diff:.5f})")
    return 0


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DEFAULT
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 50  # 50 样本约 30 分钟
    use_rag = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在 -> {data_path}", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(data_path, n_samples, use_rag))
