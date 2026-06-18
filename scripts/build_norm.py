# -*- coding: utf-8 -*-
"""§4.0 构建脚本：在 rexgradient_radgraph_structured_v3_full.jsonl 上跑归一化，
产出两份映射产物 + 效果统计。

产物：
  outputs/head_norm.json
      { "<原始head>": "<归一head>", ... }      全量原始→归一映射
  outputs/location_norm.json
      { "<原始location>": {laterality, region, lobe, needs_review}, ... }

用法（stdin 喂数据，避免把大路径写死）：
  python scripts/build_norm.py < <数据jsonl路径>

依赖：仅项目内 src/norm，无第三方库。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

# 让脚本能 import 项目内的 src
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from norm import normalize_head, normalize_location  # noqa: E402

DATA_DEFAULT = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"
OUT_DIR = os.path.join(ROOT, "outputs")


def iter_records(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def all_facts(obj: dict):
    """遍历一条记录里 finding + impression 的所有 fact（compact 形式）。"""
    for section in ("findings_graph_compact", "impression_graph_compact"):
        g = obj.get(section, {})
        for bucket in ("positive", "negative", "uncertain", "other"):
            for f in g.get(bucket, []) or []:
                yield section, f


def main(path: str) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    head_map: dict[str, str] = {}          # raw -> normalized
    loc_map: dict[str, dict] = {}          # raw -> {laterality, region, lobe, needs_review}

    # 统计用
    raw_head_freq = Counter()
    norm_head_freq = Counter()
    loc_freq = Counter()
    region_freq = Counter()
    laterality_freq = Counter()
    needs_review_freq = 0
    n_records = 0

    for obj in iter_records(path):
        n_records += 1
        for _section, f in all_facts(obj):
            # head
            raw_h = (f.get("head") or "").strip()
            if raw_h:
                if raw_h not in head_map:
                    head_map[raw_h] = normalize_head(raw_h)
                raw_head_freq[raw_h] += 1
                norm_head_freq[head_map[raw_h]] += 1
            # locations
            for loc in (f.get("locations") or []):
                text = loc if isinstance(loc, str) else (loc.get("text") or loc.get("finding") or "")
                text = (text or "").strip()
                if not text:
                    continue
                loc_freq[text] += 1
                if text not in loc_map:
                    ln = normalize_location(text)
                    loc_map[text] = {
                        "laterality": ln.laterality,
                        "region": ln.region,
                        "lobe": ln.lobe,
                        "needs_review": ln.needs_review,
                    }
                    region_freq[ln.region] += 1
                    laterality_freq[ln.laterality] += 1
                    if ln.needs_review:
                        needs_review_freq += 1

    # 写产物
    head_path = os.path.join(OUT_DIR, "head_norm.json")
    loc_path = os.path.join(OUT_DIR, "location_norm.json")
    with open(head_path, "w", encoding="utf-8") as fh:
        json.dump(head_map, fh, ensure_ascii=False, indent=2, sort_keys=True)
    with open(loc_path, "w", encoding="utf-8") as fh:
        json.dump(loc_map, fh, ensure_ascii=False, indent=2, sort_keys=True)

    # 效果报告
    print("=" * 60)
    print(f"扫描记录数        : {n_records}")
    print(f"原始 head 词表大小 : {len(raw_head_freq)}")
    print(f"归一 head 词表大小 : {len(norm_head_freq)}")
    comp = (1 - len(norm_head_freq) / max(len(raw_head_freq), 1)) * 100
    print(f"head 词表压缩率    : {comp:.1f}%")
    print(f"原始 location 数   : {len(loc_freq)}")
    print()
    print("归一后 region 分布（unique location 计）:")
    for r, c in region_freq.most_common():
        print(f"  {c:6d}  {r}")
    print()
    print("归一后 laterality 分布（unique location 计）:")
    for l, c in laterality_freq.most_common():
        print(f"  {c:6d}  {l}")
    print(f"\nneeds_review (unilateral 等) location 数: {needs_review_freq}")
    print()
    print("归一 head top20（按归一后频次）:")
    for h, c in norm_head_freq.most_common(20):
        print(f"  {c:6d}  {h}")
    print()
    print(f"产物已写出:\n  {head_path}\n  {loc_path}")
    return 0


if __name__ == "__main__":
    # 优先用命令行参数，否则用默认路径
    src_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DEFAULT
    if not os.path.exists(src_path):
        print(f"错误：数据文件不存在 -> {src_path}", file=sys.stderr)
        sys.exit(1)
    print(f"输入数据: {src_path}")
    sys.exit(main(src_path))
