# -*- coding: utf-8 -*-
"""Build the finding-pair knowledge table from a structured JSONL corpus.

Usage: python -m scripts.build_pair_table [data_path] [output_path]
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.knowledge.suggestive_table import build_pair_table  # noqa: E402

DATA_DEFAULT = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"
OUT_DEFAULT = os.path.join(_ROOT, "outputs", "pair_table.json")


def iter_records(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main(data_path: str, output_path: str) -> int:
    started = time.time()
    pair_table = build_pair_table(iter_records(data_path))
    elapsed = time.time() - started

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(pair_table, fh, ensure_ascii=False)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Built {len(pair_table)} unique pairs in {elapsed:.1f}s", flush=True)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)", flush=True)
    print("\nTop 20 frequent pair candidates:")
    ranked = sorted(
        pair_table.items(),
        key=lambda item: -item[1][0]["count"] if item[1] else 0,
    )[:20]
    for key, rows in ranked:
        top = rows[0]
        print(
            f"  {key:40} -> {top['target_head']:20} "
            f"({top['target_assertion']}, count={top['count']}, "
            f"confidence={top['confidence']:.2f})"
        )
    return 0


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else DATA_DEFAULT
    destination = sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT
    if not os.path.exists(source):
        print(f"Data file not found: {source}", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(source, destination))
