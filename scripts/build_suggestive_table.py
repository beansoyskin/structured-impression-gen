# -*- coding: utf-8 -*-
"""§4.3 构建脚本：在全量数据上构建 suggestive_of 统计表。

产物：outputs/suggestive_of_table.json
用法：python -m scripts.build_suggestive_table [数据路径]
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.knowledge.suggestive_table import build_suggestive_table, table_stats  # noqa: E402

DATA_DEFAULT = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"
OUT_PATH = os.path.join(_ROOT, "outputs", "suggestive_of_table.json")


def iter_records(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(path: str) -> int:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    n = 0
    def gen():
        nonlocal n
        for obj in iter_records(path):
            n += 1
            yield obj

    table = build_suggestive_table(gen())
    st = table_stats(table)

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(table, fh, ensure_ascii=False, indent=2, sort_keys=True)

    print("=" * 60)
    print(f"扫描记录数          : {n}")
    print(f"unique source heads : {st['unique_source_heads']}")
    print(f"unique edges        : {st['unique_edges']}")
    print(f"total occurrences   : {st['total_occurrences']}")
    print(f"unique target heads : {st['unique_target_heads']}")
    print()
    print("## top source heads（按出边总频次）")
    for h, c in st["top_sources"]:
        print(f"  {c:6d}  {h}")
    print()

    # 抽样展示几条高价值边
    print("## 高价值边样例（opacity / coarsening / nodular opacity 等的候选）")
    for probe in ("opacity", "coarsening", "nodule", "consolidation", "infiltrate"):
        rows = table.get(probe, [])
        if rows:
            top = rows[:3]
            cands = ", ".join(f"{r['target_head']}({r['count']}/{r['confidence']:.2f})" for r in top)
            print(f"  {probe:16} → {cands}")
    print()
    print(f"产物已写出: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    src_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DEFAULT
    if not os.path.exists(src_path):
        print(f"错误：数据文件不存在 -> {src_path}", file=sys.stderr)
        sys.exit(1)
    print(f"输入数据: {src_path}")
    sys.exit(main(src_path))
