# -*- coding: utf-8 -*-
"""§4.5 verifier 组合器：对一条 impression 逐 fact 跑所有 check，输出 trace。

设计：
  - 输入：impression_graph_compact + findings_graph_compact
  - 对每个 impression fact 跑 6 个 check
  - 输出 VerificationTrace：整体通过率 + 每个 fact 的 check 明细 + 溯源
  - 不做修正（修正留给 LLM repair），只做"判定 + 标记问题"

trace 结构（§4.6 溯源用）：
  {
    "passed": bool,            # 是否所有 fact 全过（无 error）
    "n_facts": int,
    "n_passed_facts": int,
    "fact_results": [          # 逐 fact
      {
        "fact": {head, assertion, locations, ...},
        "checks": [CheckResult...],
        "passed": bool,        # 该 fact 是否无 error
        "issues": [severity, message...]
      }
    ],
    "summary": {               # 问题汇总
      "errors": int, "warnings": int, "infos": int,
      "by_check": {check_name: {passed, failed}}
    }
  }
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.verify.checks import (  # noqa: E402
    normalize_fact, NormFact, CheckResult,
    check_head_legal, check_location_legal, check_laterality,
    check_assertion, check_conflict, check_redundancy,
)

# 单 fact 的 check 列表（顺序：先查合法性，再查一致性，再查冲突）
SINGLE_FACT_CHECKS = [
    check_head_legal,
    check_location_legal,
]
# 依赖 finding 上下文的 check
FINDING_DEPENDENT_CHECKS = [
    check_laterality,
    check_assertion,
]
# 依赖其他 fact 的 check
INTER_FACT_CHECKS = [
    check_conflict,
    check_redundancy,
]


@dataclass
class FactVerification:
    fact: dict                          # 原始 compact fact
    norm_fact: NormFact
    checks: list[CheckResult]
    passed: bool                        # 无 error
    issues: list[dict]                  # [{severity, check, message, action}]

    def to_dict(self) -> dict:
        return {
            "fact": self.fact,
            "passed": self.passed,
            "issues": self.issues,
            "checks": [
                {"check": c.check_name, "passed": c.passed, "severity": c.severity,
                 "message": c.message, "action": c.suggested_action}
                for c in self.checks
            ],
        }


@dataclass
class VerificationTrace:
    passed: bool
    n_facts: int
    n_passed_facts: int
    fact_results: list[FactVerification]
    summary: dict

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_facts": self.n_facts,
            "n_passed_facts": self.n_passed_facts,
            "fact_results": [fr.to_dict() for fr in self.fact_results],
            "summary": self.summary,
        }


def verify_impression(
    impression_compact: dict,
    finding_compact: dict | None = None,
) -> VerificationTrace:
    """对一条 impression 跑所有确定性 check。

    Args:
        impression_compact: impression_graph_compact
        finding_compact: findings_graph_compact（可选，提供后启用 laterality/assertion check）
    Returns:
        VerificationTrace
    """
    # 规范化所有 impression fact
    imp_facts_raw: list[dict] = []
    imp_norms: list[NormFact] = []
    for bucket in ("positive", "negative", "uncertain", "other"):
        for fact in impression_compact.get(bucket, []) or []:
            imp_facts_raw.append(fact)
            imp_norms.append(normalize_fact(fact))

    # 规范化 finding fact（用于 laterality/assertion check）
    find_norms: list[NormFact] = []
    if finding_compact:
        for bucket in ("positive", "negative", "uncertain", "other"):
            for fact in finding_compact.get(bucket, []) or []:
                find_norms.append(normalize_fact(fact))

    fact_results: list[FactVerification] = []
    n_errors_total = 0
    n_warnings_total = 0
    n_infos_total = 0
    by_check: dict[str, dict[int, int]] = {}  # check -> {passed, failed}

    for i, (fact_raw, nf) in enumerate(zip(imp_facts_raw, imp_norms)):
        all_checks: list[CheckResult] = []
        issues: list[dict] = []

        # 1. 单 fact 合法性 check
        for check_fn in SINGLE_FACT_CHECKS:
            r = check_fn(nf)
            all_checks.append(r)
            _tally(by_check, r)

        # 2. 依赖 finding 的 check
        for check_fn in FINDING_DEPENDENT_CHECKS:
            r = check_fn(nf, finding_facts=find_norms)
            all_checks.append(r)
            _tally(by_check, r)

        # 3. inter-fact check（其他 fact 作为上下文）
        others = [imp_norms[j] for j in range(len(imp_norms)) if j != i]
        for check_fn in INTER_FACT_CHECKS:
            r = check_fn(nf, other_facts=others)
            all_checks.append(r)
            _tally(by_check, r)

        # 汇总该 fact 的问题
        has_error = False
        for c in all_checks:
            if not c.passed:
                issues.append({
                    "severity": c.severity,
                    "check": c.check_name,
                    "message": c.message,
                    "action": c.suggested_action,
                })
                if c.severity == "error":
                    has_error = True
                    n_errors_total += 1
                elif c.severity == "warning":
                    n_warnings_total += 1
                else:
                    n_infos_total += 1

        fact_results.append(FactVerification(
            fact=fact_raw, norm_fact=nf, checks=all_checks,
            passed=not has_error, issues=issues,
        ))

    n_passed = sum(1 for fr in fact_results if fr.passed)
    summary = {
        "errors": n_errors_total,
        "warnings": n_warnings_total,
        "infos": n_infos_total,
        "by_check": {k: v for k, v in by_check.items()},
    }
    return VerificationTrace(
        passed=(n_errors_total == 0),
        n_facts=len(imp_facts_raw),
        n_passed_facts=n_passed,
        fact_results=fact_results,
        summary=summary,
    )


def _tally(by_check: dict, r: CheckResult):
    slot = by_check.setdefault(r.check_name, {"passed": 0, "failed": 0})
    if r.passed:
        slot["passed"] += 1
    else:
        slot["failed"] += 1


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _self_test():
    # 合理的 finding + impression 对
    finding = {
        "positive": [
            {"head": "consolidation", "assertion": "definitely present",
             "locations": ["left lower lobe"]},
            {"head": "opacity", "assertion": "definitely present",
             "locations": ["left lower lobe"]},
        ],
        "negative": [{"head": "effusion", "assertion": "definitely absent"}],
        "uncertain": [], "other": [],
    }
    impression = {
        "positive": [
            {"head": "pneumonia", "assertion": "definitely present",
             "locations": ["left lower lobe"]},
        ],
        "negative": [], "uncertain": [], "other": [],
    }
    trace = verify_impression(impression, finding)
    assert trace.passed, f"合理 impression 应全过，但有 error:\n{trace.to_dict()}"
    assert trace.n_facts == 1
    assert trace.summary["errors"] == 0

    # 错误 impression：左右写反 + 阴阳性错 + 幻觉 head
    bad_impression = {
        "positive": [
            # 左右反：finding 是左肺，impression 却写右肺
            {"head": "pneumonia", "assertion": "definitely present",
             "locations": ["right lower lobe"]},
            # 幻觉 head
            {"head": "zyx_fake_disease", "assertion": "definitely present"},
        ],
        "negative": [
            # 阴阳性错：finding 有 consolidation present，impression 却说 absent
            {"head": "consolidation", "assertion": "definitely absent"},
        ],
        "uncertain": [], "other": [],
    }
    trace2 = verify_impression(bad_impression, finding)
    assert not trace2.passed, "错误 impression 应被检出 error"
    assert trace2.summary["errors"] >= 2, f"应至少2个error，实际 {trace2.summary['errors']}"

    # 检查具体问题
    all_issues = []
    for fr in trace2.fact_results:
        all_issues.extend(fr.issues)
    # error 级别问题
    error_checks = [i["check"] for i in all_issues if i["severity"] == "error"]
    # warning 级别问题（head_legal 对非词表 head 降级为 warning，不误伤长尾真词）
    warning_checks = [i["check"] for i in all_issues if i["severity"] == "warning"]
    assert "laterality" in error_checks, f"laterality 错误应被抓(error): {error_checks}"
    assert "head_legal" in warning_checks, f"幻觉 head 应被抓(warning): {warning_checks}"
    assert "assertion" in error_checks, f"阴阳性错应被抓(error): {error_checks}"

    print("[OK] §4.5 verifier 组合器自测全部通过")
    print(f"     合理 impression: passed={trace.passed}, facts={trace.n_facts}")
    print(f"     错误 impression: passed={trace2.passed}, errors={trace2.summary['errors']}")
    print(f"       error 级 check: {error_checks}")
    print(f"       warning 级 check: {warning_checks}")


if __name__ == "__main__":
    _self_test()
