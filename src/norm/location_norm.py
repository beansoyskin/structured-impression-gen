# -*- coding: utf-8 -*-
"""§4.0(b) location 规范化 —— 把自由文本 location 解析为结构化三元组。

数据画像（sample_locations.py，8000 行采样）发现的规律：
  1. laterality 几乎都是前缀（left pleural / right lower lobe / bilateral airspace），
     少数后缀（lungs bilaterally / interstitial bilaterally）。
  2. laterality 词形有限：left / right / bilateral / bilaterally / -sided 形式。
     right 1551 + left 1472 + bilateral 475（8000 行内），几乎只有 3 类。
  3. region 有明确层级：肺/胸膜/纵隔/心/骨。
  4. lobe 分叶（upper/middle/lower）是临床定位核心，单独抽取。

输出结构化三元组（lobe 作为额外维度，因为临床重要性高）：
  LocationNorm = {
    "laterality": "left" | "right" | "bilateral" | "none",
    "region":     受控解剖区（见 REGION_TABLE）,
    "lobe":       "upper" | "middle" | "lower" | "none",
    "raw":        原始文本（溯源用）
  }

职责边界：只解析自由文本 → 结构化，不做 head 归一、不改 assertion。
laterality 与 region 的组合冲突（如 "left" + "cardiac"）不在此纠正，
留给 §4.5 的 laterality check 处理。

纯规则实现，无外部依赖。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# laterality 抽取
# ---------------------------------------------------------------------------
# 顺序敏感：bilateral 必须在 left/right 之前判，否则 bilateral 含 "l" 不影响但
# "bilaterally" 含 "lateral" 会误判。用整词边界。
_BILATERAL_RE = re.compile(r"\bbilateral(?:ly)?\b", re.IGNORECASE)
_LEFT_RE = re.compile(r"\bleft\b|\bleft\s*-\s*sided\b", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\bright\b|\bright\s*-\s*sided\b", re.IGNORECASE)
# "unilateral" 仅 8 次（profile），出现时标记为需人工判定，这里归 none 并打 flag
_UNILATERAL_RE = re.compile(r"\bunilateral(?:ly)?\b", re.IGNORECASE)


def extract_laterality(text: str) -> tuple[str, bool]:
    """从 location 文本抽 laterality。

    Returns:
        (laterality, needs_review)：
        laterality ∈ {left, right, bilateral, none}；
        needs_review=True 表示出现 unilateral 这类需人工判定的词。
    """
    if not text:
        return "none", False
    needs_review = bool(_UNILATERAL_RE.search(text))
    if _BILATERAL_RE.search(text):
        return "bilateral", needs_review
    has_left = bool(_LEFT_RE.search(text))
    has_right = bool(_RIGHT_RE.search(text))
    if has_left and has_right:
        # 文本里同时提左右（罕见），视为 bilateral
        return "bilateral", True
    if has_left:
        return "left", needs_review
    if has_right:
        return "right", needs_review
    return "none", needs_review


# ---------------------------------------------------------------------------
# lobe（肺叶）抽取 —— 临床定位核心
# ---------------------------------------------------------------------------
# "upper lobe"、"lower lobe"、"middle lobe"。
# bibasilar 是连写词（bi+basilar），\bbasilar\b 匹配不到，需单独列；
# apical→upper, basilar/basal→lower。
_LOBE_RE = re.compile(
    r"\b(upper|middle|lower|apical|basilar|basal)\b|bibasilar",
    re.IGNORECASE,
)


def extract_lobe(text: str) -> str:
    """抽肺叶/肺段信息。apical/basilar 归到最接近的叶段语义。
    bibasilar 是连写词，单独匹配（无捕获组）。
    Returns: 'upper' | 'middle' | 'lower' | 'none'
    """
    if not text:
        return "none"
    m = _LOBE_RE.search(text)
    if not m:
        return "none"
    # bibasilar 整体匹配时 group(1) 为 None，归 lower
    tok = (m.group(1) or "").lower()
    if m.group(0).lower() == "bibasilar":
        return "lower"
    if tok in ("upper", "apical"):
        return "upper"
    if tok == "middle":
        return "middle"
    if tok in ("lower", "basilar", "basal"):
        return "lower"
    return "none"


# ---------------------------------------------------------------------------
# region 受控解剖词表
# 键 = 规范 region 名，值 = 匹配关键词列表（命中任一即归此 region）。
# 顺序敏感：更具体的放前面（如 hilar 在 mediastinal 之前判，避免被 mediastinum 吞）。
# 基于数据 top location（lungs/pleural/cardiopulmonary/heart size/...）。
# ---------------------------------------------------------------------------
REGION_TABLE: list[tuple[str, list[str]]] = [
    # --- 心脏/大血管 ---
    # vasculature 在 lung 前：让 "pulmonary venous / pulmonary vessel / pulmonary artery"
    # 这类"pulmonary+血管修饰"先命中 vasculature，单独 "pulmonary" 才归 lung。
    ("vasculature", [
        "vascular", "vasculature", "venous", "pulmonary vessel", "pulmonary venous",
        "pulmonary artery", "aortic", "aorta", "svc", "iva",
    ]),
    # --- 肺实质与气道 ---
    ("lobe", ["lobe", "lingula"]),            # lingula 是左肺上叶一部分，归 lobe
    ("lung", [
        "lung", "lungs", "parenchyma", "air space", "airspace", "aeration", "inflat",
        "pulmonary", "interstitial", "alveolar", "volumes", "subsegmental", "lobar",
        "base", "basilar", "bibasilar",  # 肺底/基底
    ]),
    ("bronch", ["bronch"]),                   # peribronchial / bronchial
    ("airway", ["airway", "trachea"]),
    # --- 胸膜 ---
    ("pleura", ["pleural", "pleura", "costophrenic", "hemidiaphragm", "hemithorax", "diaphragm"]),
    # --- 纵隔/肺门 ---
    ("hilar", ["hilar", "perihilar", "hila"]),  # hila 是 hilar 复数；必须在 mediastinum 前
    ("mediastinum", ["mediastin", "cardiomediastin"]),
    # --- 心脏 ---
    ("heart", ["heart", "cardiac", "silhouette", "cardiomegaly", "cardiopulmonary"]),
    # --- 胸廓/骨/软组织 ---
    # bone 在 chest 前：让 "bony thorax"/"thoracic spine" 先命中 bone（骨性结构），
    # 而单独 "thorax"/"chest" 才归 chest。
    ("bone", ["bone", "osseous", "skeletal", "rib", "spine", "thoracic spine", "spondyl", "scoliosis", "clavicle", "bony"]),
    ("chest", ["chest", "thorax", "thoracic"]),  # 胸廓整体
    ("soft_tissue", ["soft tissue", "chest wall", "soft-tissue"]),
    # --- 腹部（RadGraph 偶尔抽到，单独标记）---
    ("abdomen", ["abdomen", "stomach", "bowel", "upper abdomen"]),
    # --- 设备/管线（与解剖并列，但 RadGraph 会抽到）---
    ("device", ["tube", "catheter", "line", "lead", "wire", "picc", "pacemaker", "port"]),
]


def extract_region(text: str) -> str:
    """把 location 文本对齐到受控解剖 region。
    Returns: region 名（REGION_TABLE 的键）或 'other'（未命中）。
    多 region 命中时取第一个（按表顺序，具体的优先）。
    """
    if not text:
        return "other"
    low = text.lower()
    for region, keywords in REGION_TABLE:
        for kw in keywords:
            if kw in low:
                return region
    return "other"


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

@dataclass
class LocationNorm:
    laterality: str  # left | right | bilateral | none
    region: str      # REGION_TABLE 键 | other
    lobe: str        # upper | middle | lower | none
    raw: str         # 原始文本（溯源）
    needs_review: bool = False  # 出现 unilateral 等需人工判定词


def normalize_location(raw: str) -> LocationNorm:
    """把一条原始 location 文本解析为结构化 LocationNorm。

    Args:
        raw: 原始 location 文本，如 "left lower lobe"、"interstitial bilaterally"。
    Returns:
        LocationNorm 结构。
    """
    if not raw:
        return LocationNorm(laterality="none", region="other", lobe="none", raw="", needs_review=False)
    text = re.sub(r"\s+", " ", raw.strip())
    lat, needs_review = extract_laterality(text)
    return LocationNorm(
        laterality=lat,
        region=extract_region(text),
        lobe=extract_lobe(text),
        raw=text,
        needs_review=needs_review,
    )


def normalize_location_list(locs: list) -> list[LocationNorm]:
    """批量规范化一个 fact 的 locations 列表（兼容 str/dict 元素）。"""
    out = []
    for loc in locs or []:
        if isinstance(loc, dict):
            text = loc.get("text") or loc.get("finding") or ""
        else:
            text = str(loc)
        if text and text.strip():
            out.append(normalize_location(text))
    return out


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        # (raw, 期望 laterality, 期望 region, 期望 lobe)
        ("left pleural", "left", "pleura", "none"),
        ("left lower lobe", "left", "lobe", "lower"),
        ("right upper lobe", "right", "lobe", "upper"),
        ("bilateral airspace", "bilateral", "lung", "none"),
        ("lungs bilaterally", "bilateral", "lung", "none"),
        ("interstitial bilaterally", "bilateral", "lung", "none"),      # 新：interstitial→lung
        ("bilateral perihilar", "bilateral", "hilar", "none"),
        ("heart size", "none", "heart", "none"),
        ("mediastinal contours", "none", "mediastinum", "none"),
        ("left - sided", "left", "other", "none"),
        ("right - sided pleural", "right", "pleura", "none"),
        ("right hemidiaphragm", "right", "pleura", "none"),
        ("left costophrenic angle", "left", "pleura", "none"),
        ("osseous structures", "none", "bone", "none"),
        ("left basilar", "left", "lung", "lower"),                      # 新：basilar→lung
        ("bibasilar", "none", "lung", "lower"),                         # 新
        ("pulmonary", "none", "lung", "none"),                          # 新：单独 pulmonary→lung
        ("pulmonary venous", "none", "vasculature", "none"),            # 新：pulmonary+血管→vasculature（前置）
        ("bony thorax", "none", "bone", "none"),                        # 新：bony→bone（优先于 chest）
        ("thoracic spine", "none", "bone", "none"),                     # 新：先命中 bone
        ("chest", "none", "chest", "none"),                             # 新
        ("hila", "none", "hilar", "none"),                              # 新：hila→hilar
        ("stomach", "none", "abdomen", "none"),                         # 新
        ("", "none", "other", "none"),
    ]
    ok = True
    for raw, e_lat, e_reg, e_lobe in cases:
        r = normalize_location(raw)
        good = (r.laterality == e_lat and r.region == e_reg and r.lobe == e_lobe)
        flag = "OK" if good else "FAIL"
        if not good:
            ok = False
        print(f"  [{flag}] {raw!r:30} -> lat={r.laterality:9} reg={r.region:12} lobe={r.lobe:6}"
              f"  (expect {e_lat}/{e_reg}/{e_lobe})")
    print("\n全部通过" if ok else "\n有失败用例")
