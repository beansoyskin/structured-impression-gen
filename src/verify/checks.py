# -*- coding: utf-8 -*-
"""§4.5 确定性 check 函数：纯规则，无 LLM。

数据探查（sample_verify_dist.py）的关键发现，直接决定 check 设计：
  - impression fact 带 location 的比例 52.8% → laterality check 对一半 fact 适用
  - assertion 3 类闭合：present 56% / absent 34% / uncertain 10%
  - 真实数据里"同 head 不同 assertion" 1331 次、"重复" 2825 次 —— 但这些大多是合理的！
    例："左肺炎(present) + 右无肺炎(absent)" 同 head 两种 assertion，因位置不同不冲突。
  ⇒ 结论：conflict/redundancy 必须结合 location 判断，不能只看 head。

fact 身份键定义（区分"是否同一个 fact"）：
  identity = (归一head, assertion, frozenset(laterality), frozenset(region), frozenset(lobe))
  位置不同 → 不是同一个 fact。

6 个 check（与方法文档 §4.5.1 对齐，evidence check 留给 LLM 判官，此处 6 个纯规则）：
  1. head_legal       : head 是否在合法诊断词表内（防幻觉）
  2. location_legal   : location 的 region 是否合法解剖区
  3. laterality       : impression fact 的 laterality 是否被 finding 支撑
  4. assertion        : impression 的 assertion 与 finding 支持的倾向是否一致
  5. conflict         : 同 head+同 location 但 assertion 矛盾（present vs absent）
  6. redundancy       : 身份键完全相同的重复 fact
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.norm.head_norm import normalize_head  # noqa: E402
from src.norm.location_norm import normalize_location, normalize_location_list  # noqa: E402


# ---------------------------------------------------------------------------
# 合法诊断词表（CheXpert 14 类 + 数据画像 top impression head）
# 用于 head_legal check。归一后的形式。
# ---------------------------------------------------------------------------
LEGAL_DIAGNOSTIC_HEADS: set[str] = {
    # CheXpert 14 类
    "atelectasis", "cardiomegaly", "consolidation", "edema", "effusion",
    "pneumonia", "pneumothorax", "pleural effusion",
    "enlarged cardiomediastinum", "fracture", "lung lesion", "lung opacity",
    "no finding", "support devices",
    # 数据画像 top impression head（归一后）
    "disease", "abnormality", "process", "infiltrate", "opacity",
    "thickening", "congestion", "congestive heart failure", "nodule",
    "mass", "emphysema", "fibrosis", "copd", "stable", "normal", "clear",
    "degenerative change", "scoliosis", "spondylosis", "tortuosity",
    "calcification", "granuloma", "tuberculosis", "aspiration", "bronchitis",
    "pneumoperitoneum", "lines tubes", "catheter", "tube", "line",
    "pacemaker", "hiatal hernia", "hernia", "collapse", "engorgement",
    "prominence", "enlargement", "hyperinflation", "coarsening",
    "scarring", "scar", "cuffing", "marking", "density", "calcified",
    "blunting", "elevation", "flattening", "deformity", "osteopenia",
    "atherosclerosis", "spondylolisthesis", "suture", "clip", "wire",
    "lead", "granuloma", "chylothorax", "hemithorax", "hemothorax",
    "infarct", "contusion", "abscess", "empyema", "bronchiectasis",
    "reaction", "reactivity", "airway", "vascularity", "vasculature",
}


def _norm_assertion(raw: str) -> str:
    if not raw:
        return "unknown"
    s = raw.lower().replace("measurement::", "")
    if "absent" in s:
        return "absent"
    if "uncertain" in s:
        return "uncertain"
    if "present" in s:
        return "present"
    return "unknown"


# ---------------------------------------------------------------------------
# fact 规范化辅助：把 compact fact 转成统一的规范化结构
# ---------------------------------------------------------------------------

@dataclass
class NormFact:
    """规范化后的 fact（归一 head + 规范 location + assertion）。"""
    head: str                                 # 归一 head
    assertion: str                            # present/absent/uncertain
    laterality: frozenset[str]                # {left,right,bilateral,none}
    region: frozenset[str]                    # {lung,pleura,...}
    lobe: frozenset[str]                      # {upper,middle,lower}
    raw_head: str = ""
    raw_locations: list = field(default_factory=list)
    suggestive_of: list = field(default_factory=list)  # [{head,assertion}]


def normalize_fact(fact: dict) -> NormFact:
    """把 compact fact 规范化为 NormFact。"""
    raw_head = fact.get("head") or ""
    head = normalize_head(raw_head)
    assertion = _norm_assertion(fact.get("assertion"))
    loc_norms = normalize_location_list(fact.get("locations") or [])
    lat = frozenset(ln.laterality for ln in loc_norms) if loc_norms else frozenset({"none"})
    reg = frozenset(ln.region for ln in loc_norms) if loc_norms else frozenset()
    lob = frozenset(ln.lobe for ln in loc_norms if ln.lobe != "none")
    sugg = []
    for s in (fact.get("suggestive_of") or []):
        if isinstance(s, dict):
            sugg.append({
                "head": normalize_head(s.get("finding") or ""),
                "assertion": _norm_assertion(s.get("assertion")),
            })
    return NormFact(
        head=head, assertion=assertion, laterality=lat, region=reg, lobe=lob,
        raw_head=raw_head, raw_locations=fact.get("locations") or [], suggestive_of=sugg,
    )


def fact_identity_key(nf: NormFact) -> tuple:
    """fact 身份键：位置不同就不是同一 fact。"""
    return (nf.head, nf.assertion, nf.laterality, nf.region, nf.lobe)


# ---------------------------------------------------------------------------
# Check 结果
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    check_name: str
    passed: bool
    severity: str = "info"        # info/warning/error
    message: str = ""
    suggested_action: str = ""    # drop/re-locate/flip-assertion/merge/none
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Check 函数
# ---------------------------------------------------------------------------

def check_head_legal(nf: NormFact, **ctx) -> CheckResult:
    """1. head 是否在合法诊断词表内（防幻觉）。
    空头 → error；不在词表 → warning（长尾诊断词可能合法，降级 warning 不直接 error）。
    """
    if not nf.head:
        return CheckResult("head_legal", False, "error", "head 为空", "drop")
    if nf.head in LEGAL_DIAGNOSTIC_HEADS:
        return CheckResult("head_legal", True)
    return CheckResult(
        "head_legal", False, "warning",
        f"head '{nf.head}' 不在合法诊断词表（可能是长尾或幻觉）",
        "review", {"head": nf.head},
    )


def check_location_legal(nf: NormFact, **ctx) -> CheckResult:
    """2. location 的 region 是否合法解剖区。"""
    for reg in nf.region:
        if reg == "other":
            # region=other 不算硬错误（§4.0 仍有 29% other），降级 info
            return CheckResult(
                "location_legal", True, "info",
                f"location 含未对齐解剖区(other)，原始: {nf.raw_locations}",
                "none",
            )
    return CheckResult("location_legal", True)


def check_laterality(nf: NormFact, finding_facts: list[NormFact] = None, **ctx) -> CheckResult:
    """3. impression fact 的 laterality 是否被 finding 支撑。

    规则：
      - impression fact 无 laterality（none/空）→ 跳过（不检查）
      - impression fact = bilateral → finding 须有 bilateral 或同时有 left+right 支撑
      - impression fact = left → finding 须有 left 或 bilateral 的支撑
      - impression fact = right → finding 须有 right 或 bilateral 的支撑
    finding_facts: finding 端规范化的 fact 列表。
    """
    if not finding_facts:
        return CheckResult("laterality", True, "info", "无 finding 上下文，跳过", "none")
    # impression 无明确侧向 → 跳过
    if not nf.laterality or nf.laterality <= {"none"}:
        return CheckResult("laterality", True, "info", "impression fact 无侧向，跳过", "none")

    # 收集 finding 端所有出现的 laterality
    find_lat = set()
    for ff in finding_facts:
        find_lat |= set(ff.laterality)
    find_lat.discard("none")

    # finding 端无任何 laterality 信息 → 无法判定，跳过（避免误报）
    if not find_lat:
        return CheckResult("laterality", True, "info", "finding 无 laterality 信息，跳过", "none")

    imp_lat = set(nf.laterality)
    imp_lat.discard("none")
    if not imp_lat:
        return CheckResult("laterality", True)

    # bilateral 需要 finding 有 bilateral 或 left+right
    if "bilateral" in imp_lat:
        if "bilateral" in find_lat or ({"left", "right"} <= find_lat):
            return CheckResult("laterality", True)
        return CheckResult(
            "laterality", False, "error",
            f"impression bilateral 但 finding 无双侧支撑 (finding laterality: {find_lat or '无'})",
            "re-locate", {"imp_lat": imp_lat, "find_lat": find_lat},
        )
    # left/right 需要对应侧或 bilateral
    for lat in imp_lat:
        if lat not in find_lat and "bilateral" not in find_lat:
            return CheckResult(
                "laterality", False, "error",
                f"impression {lat} 但 finding 无该侧支撑 (finding laterality: {find_lat or '无'})",
                "re-locate", {"imp_lat": imp_lat, "find_lat": find_lat},
            )
    return CheckResult("laterality", True)


def check_assertion(nf: NormFact, finding_facts: list[NormFact] = None, **ctx) -> CheckResult:
    """4. impression 的 assertion 与 finding 支持的倾向是否一致。

    规则（保守，只抓明显矛盾）：
      - impression = absent X，但 finding 里有 X(present) 且同 head → 可能矛盾（warning）
        例：impression "无肺炎"，但 finding "肺炎(present)" → 矛盾
      - impression = present X，但 finding 里 X 明确 absent 且无 suggestive_of 指向 X → warning
        例：impression "肺炎(present)"，但 finding 无任何肺炎征象也无 suggestive_of → 无证据
      - uncertain 一律通过（不确定是合法的）

    注意：finding 里的 head 与 impression head 可能不同名（涌现），所以同 head 判断只在
    head 相同时生效；head 不同时靠 suggestive_of 判断 evidence（留给 LLM，此处只做弱检查）。
    """
    if not finding_facts:
        return CheckResult("assertion", True, "info", "无 finding 上下文，跳过", "none")
    if nf.assertion == "uncertain":
        return CheckResult("assertion", True)

    # 找 finding 里同 head 的 fact
    same_head_finding = [ff for ff in finding_facts if ff.head == nf.head]

    if nf.assertion == "absent":
        # impression 说 absent，但 finding 同 head present → 可能合理（临床判断推翻）
        # 降级为 warning 而非 error：放射科医生有权在 impression 推翻 finding 的初步判断
        # 例：finding "fracture present" → impression "fracture absent"（判断为陈旧/伪影）
        for ff in same_head_finding:
            if ff.assertion == "present":
                return CheckResult(
                    "assertion", False, "warning",
                    f"impression absent '{nf.head}' 但 finding 同 head present（可能是临床判断推翻，需确认）",
                    "review", {"finding_assertion": "present"},
                )
        return CheckResult("assertion", True)

    if nf.assertion == "present":
        # impression present，finding 同 head absent → 可能漏诊/矛盾（warning）
        for ff in same_head_finding:
            if ff.assertion == "absent":
                return CheckResult(
                    "assertion", False, "warning",
                    f"impression present '{nf.head}' 但 finding 同 head absent（可能是涌现推断，需 evidence check）",
                    "review", {"finding_assertion": "absent"},
                )
        return CheckResult("assertion", True)

    return CheckResult("assertion", True)


def check_conflict(nf: NormFact, other_facts: list[NormFact] = None, **ctx) -> CheckResult:
    """5. 同 head + 同 location 但 assertion 矛盾（present vs absent）。

    关键设计（数据探查证实）：位置不同不算冲突。
    "左肺炎(present) + 右无肺炎(absent)" 合理，不报。
    只有同 head + 同 laterality/region/lobe 但 present&absent 才是真冲突。
    """
    if not other_facts:
        return CheckResult("conflict", True)
    for of in other_facts:
        if of.head != nf.head:
            continue
        # head 相同，看位置是否相同
        if of.laterality == nf.laterality and of.region == nf.region and of.lobe == nf.lobe:
            # 位置相同，看 assertion 是否矛盾
            if {nf.assertion, of.assertion} == {"present", "absent"}:
                return CheckResult(
                    "conflict", False, "error",
                    f"同 head '{nf.head}' 同位置 assertion 矛盾: {nf.assertion} vs {of.assertion}",
                    "drop", {"conflict_with_assertion": of.assertion},
                )
    return CheckResult("conflict", True)


def check_redundancy(nf: NormFact, other_facts: list[NormFact] = None, **ctx) -> CheckResult:
    """6. 身份键完全相同的重复 fact。

    只有 head+assertion+laterality+region+lobe 全同才算重复。
    位置不同不算重复。
    """
    if not other_facts:
        return CheckResult("redundancy", True)
    my_key = fact_identity_key(nf)
    for of in other_facts:
        if fact_identity_key(of) == my_key:
            return CheckResult(
                "redundancy", False, "warning",
                f"与另一 fact 完全重复 (head={nf.head}, assertion={nf.assertion}, loc=({nf.laterality},{nf.region},{nf.lobe}))",
                "merge", {},
            )
    return CheckResult("redundancy", True)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _self_test():
    # 构造 finding facts
    def mkfact(head, assertion, locs=None):
        f = {"head": head, "assertion": assertion, "locations": locs or [], "suggestive_of": []}
        return normalize_fact(f)

    finding = [
        mkfact("consolidation", "definitely present", ["left lower lobe"]),
        mkfact("opacity", "definitely present", ["left lower lobe"]),
        mkfact("effusion", "definitely absent"),
    ]

    # 测试1: 合理的 impression fact 应全部通过
    good_fact = normalize_fact({
        "head": "pneumonia", "assertion": "definitely present",
        "locations": ["left lower lobe"], "suggestive_of": [],
    })
    assert check_head_legal(good_fact).passed
    assert check_location_legal(good_fact).passed
    r_lat = check_laterality(good_fact, finding_facts=finding)
    assert r_lat.passed, f"合理 laterality 应通过: {r_lat.message}"
    r_assert = check_assertion(good_fact, finding_facts=finding)
    assert r_assert.passed, f"合理 assertion 应通过: {r_assert.message}"
    assert check_conflict(good_fact, other_facts=[]).passed
    assert check_redundancy(good_fact, other_facts=[]).passed

    # 测试2: laterality 矛盾（impression 左，finding 只有右）
    wrong_lat = normalize_fact({
        "head": "pneumonia", "assertion": "definitely present",
        "locations": ["right lower lobe"], "suggestive_of": [],
    })
    # finding 里只有 left，impression 却 right → 应 fail
    find_right_only = [mkfact("consolidation", "definitely present", ["left lower lobe"])]
    r = check_laterality(wrong_lat, finding_facts=find_right_only)
    assert not r.passed, f"左右矛盾应被抓: {r.message}"
    assert r.suggested_action == "re-locate"

    # 测试3: assertion 矛盾（impression absent pneumonia，但 finding present pneumonia）
    find_with_pneumonia = [mkfact("pneumonia", "definitely present")]
    absent_pneumonia = normalize_fact({
        "head": "pneumonia", "assertion": "definitely absent", "locations": [],
    })
    r = check_assertion(absent_pneumonia, finding_facts=find_with_pneumonia)
    assert not r.passed, f"阴阳性矛盾应被抓: {r.message}"
    assert r.severity == "warning"
    assert r.suggested_action == "review"

    # 测试4: 合理的非冲突（左肺炎present + 右无肺炎absent）
    left_pneu = normalize_fact({"head": "pneumonia", "assertion": "definitely present", "locations": ["left"]})
    right_no_pneu = normalize_fact({"head": "pneumonia", "assertion": "definitely absent", "locations": ["right"]})
    r = check_conflict(left_pneu, other_facts=[right_no_pneu])
    assert r.passed, f"不同位置的左右肺炎不应报冲突: {r.message}"

    # 测试5: 真冲突（同位置 present+absent）
    left_pneu2 = normalize_fact({"head": "pneumonia", "assertion": "definitely absent", "locations": ["left"]})
    r = check_conflict(left_pneu, other_facts=[left_pneu2])
    assert not r.passed, f"同位置同 head 矛盾应被抓: {r.message}"

    # 测试6: 重复（完全相同）
    r = check_redundancy(left_pneu, other_facts=[normalize_fact({"head": "pneumonia", "assertion": "definitely present", "locations": ["left"]})])
    assert not r.passed, f"完全重复应被抓: {r.message}"

    # 测试7: head 非法（幻觉）
    hallucination = normalize_fact({"head": "zyxwvut_fake_disease", "assertion": "definitely present"})
    r = check_head_legal(hallucination)
    assert not r.passed, "幻觉 head 应被抓"

    print("[OK] §4.5 check 函数自测全部通过")
    print(f"     合法诊断词表大小: {len(LEGAL_DIAGNOSTIC_HEADS)}")


if __name__ == "__main__":
    _self_test()
