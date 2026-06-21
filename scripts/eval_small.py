# -*- coding: utf-8 -*-
"""精简版端到端评测：10 样本，快速出结果。"""
import sys, json, time, random, os
sys.path.insert(0, ".")
from src.norm.head_norm import normalize_head
from src.norm.location_norm import normalize_location_list
from src.inference.infer import infer_impression
from src.knowledge.suggestive_table import SuggestiveKnowledge
from collections import Counter

DATA = r"E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl"
knowledge = SuggestiveKnowledge("outputs/suggestive_of_table.json")

def _norm_assertion(r):
    if not r: return "unknown"
    s = r.lower().replace("measurement::","")
    if "absent" in s: return "absent"
    if "uncertain" in s: return "uncertain"
    if "present" in s: return "present"
    return "unknown"

def normalize_fact(f):
    h = normalize_head(f.get("head") or "")
    a = _norm_assertion(f.get("assertion") or "")
    ln = normalize_location_list(f.get("locations") or [])
    locs = frozenset()
    for l in ln:
        pts = []
        if l.laterality != "none": pts.append(l.laterality)
        if l.region != "other": pts.append(l.region)
        if l.lobe != "none": pts.append(l.lobe)
        if pts: locs = locs | {tuple(sorted(pts))}
    suggestion = 0
    for s_ in (f.get("suggestive_of") or []):
        if isinstance(s_, dict) and normalize_head(s_.get("finding","")): suggestion = 1
        break
    return (h, a, locs, suggestion)

def exact_key(nf): return (nf[0], nf[1], nf[2], nf[3])
def ha_key(nf): return (nf[0], nf[1])
def h_key(nf): return nf[0]

def hal_key(nf): return (nf[0], nf[1], nf[2])

def f1(preds, golds, key_fn):
    p = Counter(key_fn(p) for p in preds)
    g = Counter(key_fn(g) for g in golds)
    tp = sum(min(p[k], g[k]) for k in set(p)&set(g))
    fp = sum(p.values()) - tp
    fn = sum(g.values()) - tp
    prec = tp/(tp+fp) if tp+fp else 1
    rec = tp/(tp+fn) if tp+fn else 1
    f = 2*prec*rec/(prec+rec) if prec+rec else 0
    return f

def loc_err(preds, golds):
    ha_match = loc_wrong = 0
    p_ha = Counter(ha_key(p) for p in preds)
    g_ha = Counter(ha_key(g) for g in golds)
    for k in set(p_ha)&set(g_ha):
        n = min(p_ha[k], g_ha[k])
        ha_match += n
        pl = [p[2] for p in preds if ha_key(p)==k]
        gl = [g[2] for g in golds if ha_key(g)==k]
        pc = Counter(pl); gc = Counter(gl)
        lm = sum(min(pc[lk],gc[lk]) for lk in set(pc)&set(gc))
        loc_wrong += n - lm
    return loc_wrong/ha_match if ha_match else 0

records = []
with open(DATA, encoding="utf-8") as f:
    for line in f:
        if line.strip(): records.append(json.loads(line))

rng = random.Random(42)
sample = rng.sample(records, 10)
v3_base = {"exact_fact_f1":0.22598,"head_assertion_location_f1":0.26061,"head_assertion_f1":0.39404,"head/entity_f1":0.44292,"location_error_rate":0.34182}

all_metrics = []
for i, rec in enumerate(sample):
    fg = rec.get("findings_graph_compact",{})
    ig = rec.get("impression_graph_compact",{})
    t0 = time.time()
    result = infer_impression(fg, knowledge=knowledge)
    t1 = time.time()

    golds = []
    for b in ("positive","negative","uncertain","other"):
        for f_ in ig.get(b,[]): golds.append(normalize_fact(f_))

    preds = [normalize_fact(f_) for f_ in result.facts]
    m = {
        "exact_fact_f1": f1(preds,golds,exact_key),
        "head_assertion_location_f1": f1(preds,golds,hal_key),
        "head_assertion_f1": f1(preds,golds,ha_key),
        "head/entity_f1": f1(preds,golds,h_key),
        "location_error_rate": loc_err(preds,golds),
        "time": t1-t0,
        "n_facts": len(result.facts),
    }
    all_metrics.append(m)
    print(f"[{i+1}/10] {t1-t0:.0f}s | facts={len(result.facts)} | exact_f1={m['exact_fact_f1']:.3f}")

avg = {k: sum(m[k] for m in all_metrics)/len(all_metrics) for k in all_metrics[0] if k not in ("time","n_facts")}
avg["avg_time"] = sum(m["time"] for m in all_metrics)/len(all_metrics)
avg["avg_facts"] = sum(m["n_facts"] for m in all_metrics)/len(all_metrics)

print("\n"+"="*55)
print(f"{'指标':34} {'新pipeline':>10} {'v3规则':>10}")
print("-"*55)
v3_map = {"exact_fact_f1":"exact_fact_f1","head_assertion_location_f1":"head_assertion_location_f1","head_assertion_f1":"head_assertion_f1","head/entity_f1":"head/entity_f1","location_error_rate":"location_error_rate"}
for k in ["exact_fact_f1","head_assertion_location_f1","head_assertion_f1","head/entity_f1","location_error_rate"]:
    nv = avg[k]
    ov = v3_base[k]
    d = nv - ov
    s = "+" if d>0 else ""
    print(f"{k:34} {nv:>10.5f} {ov:>10.5f}  ({s}{d:.5f})")
print(f"\n平均推理: {avg['avg_time']:.0f}s/样本")
print(f"平均facts: {avg['avg_facts']:.1f}")
print(f"(真实 median=1，说明 LLM 倾向于多输出 fact)")
