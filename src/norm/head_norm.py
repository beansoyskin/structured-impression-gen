# -*- coding: utf-8 -*-
"""§4.0(a) head 归一化 —— 受控别名表与规则。

数据画像发现的 head 碎片化分四类（见 sample_heads.py 输出）：
  1. 纯单复数            effusion/effusions  —— lemmatizer 处理
  2. 不规则复数          pneumothorax/pneumothoraces  —— 别名表
  3. 词形/同义变体        hyperinflation/hyperinflated  —— 别名表
  4. 短语归并            normal limits / within normal limits → normal  —— 短语规则

职责边界（与 v1 审核式归一化一致）：只做形态归一，
  不合并左右侧、不改 assertion、不合并不同解剖位。

可选依赖：nltk（提升单复数质量）。未安装时用内置规则兜底，质量略降但不阻塞。
"""
from __future__ import annotations

import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# 第 2/3 类：受控别名表（手写，来自 sample_heads.py 的高频形态对）
# 键 = 小写原形，值 = 规范形。所有键已小写。
# ---------------------------------------------------------------------------
ALIAS_TABLE: dict[str, str] = {
    # --- 不规则复数 ---
    "pneumothoraces": "pneumothorax",
    "masses": "mass",
    "metastases": "metastasis",
    "metastasis": "metastasis",
    # --- 词形/同义变体：充气 ---
    "hyperinflated": "hyperinflation",
    "hyperexpanded": "hyperinflation",
    "hypoinflated": "hypoinflation",
    "hypoinflation": "hypoinflation",
    "aerated": "aeration",
    "well - aerated": "well aerated",
    "well-expanded": "well expanded",
    "well - expanded": "well expanded",
    "adequately inflated": "well expanded",
    "inflated": "well expanded",
    # --- 词形变体：钙化 ---
    "calcified": "calcification",
    "calcifications": "calcification",
    # --- 词形变体：增大 ---
    "enlarged": "enlargement",
    "engorged": "engorgement",
    "prominent": "prominence",
    "prominence": "prominence",
    # --- 词形变体：迂曲 ---
    "tortuous": "tortuosity",
    # --- 词形变体：退化 ---
    "degenerative": "degenerative change",
    "degenerative changes": "degenerative change",
    "demineralized": "osteopenia",
    "osteopenic": "osteopenia",
    # --- 词形变体：肺气肿 ---
    "emphysematous": "emphysema",
    "emphysematous changes": "emphysema",
    # --- 词形变体：充血 ---
    "congestive": "congestion",
    "congestive failure": "congestive heart failure",
    "chf": "congestive heart failure",
    # --- 词形变体：骨/血管 ---
    "atherosclerotic": "atherosclerosis",
    "sclerotic": "atherosclerosis",
    # --- 词形变体：结节/瘢痕 ---
    "scarring": "scar",
    "scars": "scar",
    "nodularity": "nodule",
    # 设计决策（v0.2）：带修饰的复合 head（acute/focal/confluent/hazy + 核心）
    # 一律保留原样，不归并到核心。原因：急性/慢性/局灶等修饰在 §4.5 的
    # assertion/evidence check 中是重要依据，归并会造成不可逆信息损失。
    # 如需聚合，下游检索用"核心 head 粗召回 + 复合 head 精排"两套并存即可。
    # 因此下面这些项【故意不写进别名表】：
    #   focal opacities / focal opacification / focal opacity
    #   focal consolidation / focal infiltrate / focal infiltrates
    #   acute infiltrate / confluent opacities / hazy consolidations
}

# ---------------------------------------------------------------------------
# 第 4 类：短语归并规则（normal 家族是最高频的碎片化来源）
# 顺序敏感：长串优先匹配。
# ---------------------------------------------------------------------------
_PHRASE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bwithin normal limits?\b"), "normal"),
    (re.compile(r"\bwithin normal\b"), "normal"),
    (re.compile(r"\bnormal limits?\b"), "normal"),
    (re.compile(r"\bin place\b"), "in place"),
]


# ---------------------------------------------------------------------------
# 内置 lemmatizer 兜底（无需 nltk）
# 处理第 1 类纯单复数：规则化去 -s/-es/-ies。
# 注：这是保守规则，宁可少归并也不要错归并（如 "gas" 不是 "ga"）。
# ---------------------------------------------------------------------------
_IRREGULAR_PLURAL = {  # 已含不规则，与别名表互补
    "atelectases": "atelectasis",
    "metastases": "metastasis",
    "metastasis": "metastasis",
}

# 这些词以 s 结尾但不是复数，避免误删
_NOT_PLURAL = {
    "atelectasis", "scoliosis", "spondylosis", "fibrosis", "bronchitis",
    "appendicitis", "arthritis", "atherosclerosis", "tuberculosis",
    "pleuritis", "encephalomalacia", "osteoporosis", "thrombosis",
    "sepsis", "acidosis", "alkalosis", "status", "axis", "process",
    "gas", "dress", "cross", "class", "illness", "disease", "ischemia",
}


def _rule_lemmatize(word: str) -> str:
    """内置规则化单数化。输入应为已小写的单词。"""
    if word in _IRREGULAR_PLURAL:
        return _IRREGULAR_PLURAL[word]
    if word in _NOT_PLURAL or not word.endswith("s"):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        # -es 仅当词干以 x/s/z/ch/sh 结尾时才去 es（boxes→box, dishes→dish）；
        # 否则按普通 -s 处理只去单 s（nodules→nodule, fractures→fracture）。
        stem = word[:-2]
        if stem.endswith(("x", "s", "z", "ch", "sh")):
            return stem
        return word[:-1]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


# ---------------------------------------------------------------------------
# 可选：接入 nltk WordNetLemmatizer 提升质量
# ---------------------------------------------------------------------------
_NLTK_LEMMA = None
_NLTK_TRIED = False


@lru_cache(maxsize=1)
def _get_nltk_lemmatizer():
    global _NLTK_LEMMA, _NLTK_TRIED
    if _NLTK_TRIED:
        return _NLTK_LEMMA
    _NLTK_TRIED = True
    try:
        from nltk.stem import WordNetLemmatizer  # type: ignore
        _NLTK_LEMMA = WordNetLemmatizer()
        # 触发一次确保词典可用
        _NLTK_LEMMA.lemmatize("tests", pos="n")
    except Exception:
        _NLTK_LEMMA = None
    return _NLTK_LEMMA


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def normalize_head(raw: str) -> str:
    """把一条原始 head 归一化为规范形。

    流程（顺序敏感）：
      1. 清洗：去首尾空格、小写、折叠多空格
      2. 短语规则（第4类）：within normal limits → normal
      3. 别名表（第2/3类）：硬查表
      4. lemmatize（第1类）：nltk 优先，否则规则兜底
      5. 单词：返回归一结果；多词短语：对末词做 lemmatize（保留前缀修饰）

    Args:
        raw: 原始 head 文本，如 "Effusions"、"within normal limits"。
    Returns:
        规范形，如 "effusion"、"normal"。
    """
    if not raw:
        return ""
    # 1. 清洗
    s = re.sub(r"\s+", " ", raw.strip().lower())

    # 2. 短语规则
    for pat, repl in _PHRASE_RULES:
        if pat.search(s):
            s = pat.sub(repl, s)
            s = re.sub(r"\s+", " ", s.strip())

    # 3. 别名表（精确匹配）
    if s in ALIAS_TABLE:
        return ALIAS_TABLE[s]

    # 4/5. lemmatize
    lemmatizer = _get_nltk_lemmatizer()
    if " " in s:
        # 多词短语：对末词归一（保留前缀修饰词），例 "acute infiltrate" → 不归并但末词处理
        parts = s.split(" ")
        parts[-1] = _lemmatize_word(parts[-1], lemmatizer)
        return " ".join(parts)
    return _lemmatize_word(s, lemmatizer)


def _lemmatize_word(word: str, lemmatizer) -> str:
    if word in _IRREGULAR_PLURAL:
        return _IRREGULAR_PLURAL[word]
    if lemmatizer is not None:
        try:
            lemma = lemmatizer.lemmatize(word, pos="n")
            if lemma:
                return lemma
        except Exception:
            pass
    return _rule_lemmatize(word)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        ("Effusions", "effusion"),
        ("effusion", "effusion"),
        ("opacities", "opacity"),
        ("pneumothoraces", "pneumothorax"),
        ("within normal limits", "normal"),
        ("normal limits", "normal"),
        ("hyperinflated", "hyperinflation"),
        ("calcified", "calcification"),
        ("atelectasis", "atelectasis"),
        ("nodules", "nodule"),
        ("fractures", "fracture"),
        ("gas", "gas"),  # 不该被误删
        ("disease", "disease"),
        ("scoliosis", "scoliosis"),
        ("boxes", "box"),  # -es + x 词干
        ("dishes", "dish"),  # -es + sh 词干
        # 复合 head 保留原样（设计决策），末词做单复数归一：
        ("focal opacities", "focal opacity"),
        ("acute infiltrate", "acute infiltrate"),
    ]
    ok = True
    for raw, expected in cases:
        got = normalize_head(raw)
        flag = "OK" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  [{flag}] {raw!r:30} -> {got!r:25} (expect {expected!r})")
    print("\n全部通过" if ok else "\n有失败用例")
