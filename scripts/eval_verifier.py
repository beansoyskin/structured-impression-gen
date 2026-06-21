# -*- coding: utf-8 -*-
"""§4.5 sanity check：用真实 impression 数据验证 check 规则不会误报。

核心假设：真实 impression 是放射科医生写的正确结论，应大部分通过。
若真实数据大量报 error，说明 check 规则太严有 bug。

同时做错误注入：把真实 impression 篡改（左右反、阴阳性反），验证 check 能抓到。

用法：python -m scripts.eval_verifier [数据路径] [n_samples]
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.verify.verifier import verify_impression  # noqa: E402

DATA_DEFAULT = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"


def iter_records(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def flip_laterality(loc_text: str) -> str:
    """左右翻转一个 location 文本。"""
    if "left" in loc_text.lower():
        return loc_text.replace("left", "right").replace("Left", "Right")
    if "right" in loc_text.lower():
        return loc_text.replace("right", "left").replace("Right", "Left")
    return loc_text


def inject_errors(impression_compact: dict) -> dict:
    """对 impression 注入错误：左右翻转所有带 left/right 的 location。
    返回篡改后的副本。
    """
    import copy
    bad = copy.deepcopy(impression_compact)
    for bucket in ("positive", "negative", "uncertain", "other"):
        for fact in bad.get(bucket, []) or []:
            new_locs = []
            for loc in (fact.get("locations") or []):
                if isinstance(loc, str):
                    new_locs.append(flip_laterality(loc))
                elif isinstance(loc, dict):
                    nl = dict(loc)
                    if "text" in nl:
                        nl["text"] = flip_laterality(nl["text"])
                    new_locs.append(nl)
            fact["locations"] = new_locs
    return bad


def main(data_path: str, n_samples: int) -> int:
    rng = random.Random(42)
    records = list(iter_records(data_path))
    sample = rng.sample(records, min(n_samples, len(records)))

    # ---- 1. 真实数据 sanity check ----
    print("=" * 60)
    print("1. 真实数据 sanity check（应大部分通过）")
    print("=" * 60)
    n_total = 0
    n_passed = 0
    error_by_check = Counter()
    warning_by_check = Counter()
    n_facts_total = 0
    n_facts_passed = 0
    sample_errors = []

    for rec in sample:
        ig = rec.get("impression_graph_compact") or {}
        fg = rec.get("findings_graph_compact") or {}
        if not ig or not any(ig.get(b) for b in ("positive", "negative", "uncertain", "other")):
            continue
        n_total += 1
        trace = verify_impression(ig, fg)
        if trace.passed:
            n_passed += 1
        else:
            if len(sample_errors) < 5:
                sample_errors.append((rec.get("id", "")[-16:], trace))
        n_facts_total += trace.n_facts
        n_facts_passed += trace.n_passed_facts
        for chk, slot in trace.summary["by_check"].items():
            error_by_check[chk] += slot.get("failed", 0) if slot.get("failed", 0) > 0 else 0
        # warning 统计
        for fr in trace.fact_results:
            for issue in fr.issues:
                if issue["severity"] == "warning":
                    warning_by_check[issue["check"]] += 1
                elif issue["severity"] == "error":
                    error_by_check[issue["check"]] += 1

    pass_rate = n_passed / n_total * 100 if n_total else 0
    fact_pass_rate = n_facts_passed / n_facts_total * 100 if n_facts_total else 0
    print(f"样本数            : {n_total}")
    print(f"impression 通过率 : {n_passed}/{n_total} = {pass_rate:.1f}%")
    print(f"fact 通过率       : {n_facts_passed}/{n_facts_total} = {fact_pass_rate:.1f}%")
    print(f"\nerror 分布（按 check）:")
    for chk, c in error_by_check.most_common():
        print(f"  {c:5d}  {chk}")
    print(f"\nwarning 分布（按 check）:")
    for chk, c in warning_by_check.most_common():
        print(f"  {c:5d}  {chk}")

    if sample_errors:
        print(f"\n## 未通过样本示例（前 {min(3, len(sample_errors))} 个）:")
        for rid, trace in sample_errors[:3]:
            print(f"  --- id ...{rid} ---")
            for fr in trace.fact_results:
                if not fr.passed:
                    for issue in fr.issues:
                        if issue["severity"] == "error":
                            print(f"    [{issue['check']}] {issue['message']}")

    # ---- 2. 错误注入测试 ----
    print("\n" + "=" * 60)
    print("2. 错误注入测试（篡改后应被抓出更多 error）")
    print("=" * 60)
    n_inject_total = 0
    n_inject_caught = 0  # 篡改后新增 error 的样本
    lat_caught = 0
    # 只对带 laterality 的 impression 注入
    injectable = [r for r in sample
                  if any(
                      any(("left" in (str(l).lower()) or "right" in (str(l).lower()))
                          for l in (f.get("locations") or []))
                      for b in ("positive", "negative", "uncertain", "other")
                      for f in (r.get("impression_graph_compact", {}).get(b, []) or [])
                  )]
    for rec in injectable[:500]:
        ig = rec.get("impression_graph_compact") or {}
        fg = rec.get("findings_graph_compact") or {}
        good_trace = verify_impression(ig, fg)
        bad_ig = inject_errors(ig)
        bad_trace = verify_impression(bad_ig, fg)
        n_inject_total += 1
        new_errors = bad_trace.summary["errors"] - good_trace.summary["errors"]
        if new_errors > 0:
            n_inject_caught += 1
            # 检查是否抓到了 laterality
            for fr in bad_trace.fact_results:
                for issue in fr.issues:
                    if issue["severity"] == "error" and issue["check"] == "laterality":
                        lat_caught += 1
                        break
                else:
                    continue
                break

    print(f"可注入样本数      : {n_inject_total}")
    print(f"篡改后被抓出      : {n_inject_caught}/{n_inject_total} = "
          f"{n_inject_caught/n_inject_total*100:.1f}%" if n_inject_total else "无")
    print(f"其中 laterality 被抓: {lat_caught}")
    print(f"\n解读：篡改后 if check 能抓出新增 error，说明 check 有效。")
    print(f"      未被抓的可能是 finding 端也含双侧（双侧对翻转无影响）。")
    return 0


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DEFAULT
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在 -> {data_path}", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(data_path, n_samples))
