# -*- coding: utf-8 -*-
"""探查 suggestive_of 在数据中的真实结构与关联方式。

关键问题：
  1. suggestive_of 出现在 finding 端还是 impression 端？
  2. 它挂在哪条 fact 上（即"谁 suggestive_of 谁"的关联方式）？
  3. suggestive_of 元素的结构（dict? 含哪些字段?）
  4. finding 侧 head 与 impression 侧 suggestive_of 的对应关系能否建立？

只扫前若干条带 suggestive_of 的记录。
"""
import json
import sys
from collections import Counter

DATA = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"

# 统计：suggestive_of 在 finding vs impression 端的出现
find_sugg = 0
imp_sugg = 0
both_sugg = 0
n = 0
examples = []  # 完整记录样本

# suggestive_of 元素结构探测
sugg_key_shapes = Counter()  # dict 时的 key 集合

with open(DATA, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        n += 1

        f_has = False
        i_has = False
        # finding 端
        for b in ("positive", "negative", "uncertain", "other"):
            for f in o.get("findings_graph_compact", {}).get(b, []) or []:
                s = f.get("suggestive_of", None)
                if s:
                    f_has = True
                    for item in s:
                        if isinstance(item, dict):
                            sugg_key_shapes[tuple(sorted(item.keys()))] += 1
        # impression 端
        for b in ("positive", "negative", "uncertain", "other"):
            for f in o.get("impression_graph_compact", {}).get(b, []) or []:
                s = f.get("suggestive_of", None)
                if s:
                    i_has = True
                    for item in s:
                        if isinstance(item, dict):
                            sugg_key_shapes[tuple(sorted(item.keys()))] += 1

        if f_has:
            find_sugg += 1
        if i_has:
            imp_sugg += 1
        if f_has and i_has:
            both_sugg += 1

        # 收集含 suggestive_of 的完整记录（最多 5 个，且优先含 finding 端的）
        if (f_has or i_has) and len(examples) < 5:
            examples.append(o)

        if n >= 20000:
            break

print(f"扫描 {n} 行")
print(f"finding 端含 suggestive_of 的记录数 : {find_sugg}")
print(f"impression 端含 suggestive_of 的记录数: {imp_sugg}")
print(f"两端都含的记录数                     : {both_sugg}")
print()
print("## suggestive_of 元素的 key 结构分布")
for keys, c in sugg_key_shapes.most_common():
    print(f"  {c:6d}  {list(keys)}")
print()

# 打印几个完整样本（只看 compact + suggestive_of 相关）
def compact_view(o):
    """精简展示 finding/impression 的 compact + suggestive_of。"""
    out = []
    for name, key in [("FINDING", "findings_graph_compact"), ("IMPRESS", "impression_graph_compact")]:
        g = o.get(key, {})
        for b in ("positive", "negative", "uncertain", "other"):
            for f in g.get(b, []) or []:
                if f.get("suggestive_of"):
                    head = f.get("head")
                    assertn = f.get("assertion")
                    sugg = f.get("suggestive_of")
                    out.append(f"  [{name:7}] head={head!r:25} assertion={assertn:24} suggestive_of={sugg}")
    return out

for i, o in enumerate(examples[:5]):
    print(f"=== 样本 {i+1} (id 末尾: ...{o.get('id','')[-20:]}) ===")
    for line in compact_view(o):
        print(line)
    print()
