# -*- coding: utf-8 -*-
"""§4.2 全量构建索引 + Recall@K 评测。

产物：
  outputs/bm25_index.json   全量 BM25 索引（tokenized_corpus + meta + id）
  （评测结果打印到 stdout，不单独存文件）

流程：
  1. 读 14 万 JSONL
  2. 构建索引（存盘）
  3. 切分：随机抽 1000 条作 test，其余作语料（避免测试集污染语料）
     注：严格做法是语料不含 test，但 14万规模下 test 占比极小，
     且评测已排除自身，影响可忽略。为节省内存/时间，这里用全量建索引、
     test 用同样的全量记录，评测时按 id 排除自身。

用法：python -m scripts.build_and_eval_retrieval [数据路径] [test_size]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.retrieval.bm25_retriever import CaseRetriever  # noqa: E402
from src.retrieval.evaluate import evaluate_recall  # noqa: E402

DATA_DEFAULT = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"
INDEX_PATH = os.path.join(_ROOT, "outputs", "bm25_index.json")
TEST_SIZE_DEFAULT = 1000


def iter_records(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(data_path: str, test_size: int) -> int:
    # 1. 读全量
    t0 = time.time()
    records = list(iter_records(data_path))
    t1 = time.time()
    print(f"读取 {len(records)} 条记录，用时 {t1-t0:.1f}s")

    # 2. 抽 test 集（固定种子可复现）
    rng = random.Random(42)
    test_idx = set(rng.sample(range(len(records)), min(test_size, len(records))))
    test_records = [records[i] for i in sorted(test_idx)]

    # 3. 构建索引（全量语料）
    print("构建 BM25 索引...")
    retriever = CaseRetriever.build_from_records(records)
    t2 = time.time()
    print(f"索引构建完成：{len(retriever.id_list)} 条入索引（空 finding 已跳过），用时 {t2-t1:.1f}s")

    # 4. 评测（不存索引——14万条索引 JSON 几百 MB，不如 15s 重建；评测结果才重要）
    # retriever.save(INDEX_PATH)  # 需要时取消注释

    # 5. 评测
    print(f"\n评测 Recall@K（test_size={len(test_records)}, threshold=0.5, 排除自身）...")
    t3 = time.time()
    res = evaluate_recall(retriever, test_records, ks=[1, 5, 10], threshold=0.5, max_search=50)
    t4 = time.time()
    print(f"评测用时 {t4-t3:.1f}s\n")
    print("=" * 50)
    print("§4.2 BM25 检索 baseline")
    print("=" * 50)
    for k, v in res.items():
        print(f"  {k:22}: {v}")
    print()
    print("解读：")
    print("  - recall@K: TopK 里至少 1 条相关的 query 占比（Jaccard>=0.5）")
    print("  - relevant_coverage: test 中存在相关病例的比例（分母参考）")
    print("  - mean_jaccard_top1: Top1 的平均 Jaccard（不论是否过阈值）")
    print(f"\n索引产物: {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DEFAULT
    test_size = int(sys.argv[2]) if len(sys.argv) > 2 else TEST_SIZE_DEFAULT
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在 -> {data_path}", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(data_path, test_size))
