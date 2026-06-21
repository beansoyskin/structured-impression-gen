# 结构化中间表示驱动的可解释 CXR Impression 生成

> 研究：用结构化 5-元组 fact 作为显式中间表示，替代端到端黑盒生成 impression。
> 配套方法文档：`C:\Users\小天\Desktop\方法草稿_结构化Impression生成.md`

## 当前进度

| 阶段 | 状态 | 说明 |
|---|---|---|
| §4.0 数据归一化 | ✅ 完成 | head 归一 + location 规范化，14万样本已跑出产物 |
| §4.3 Knowledge Lookup | ✅ 完成 | 语料内 suggestive_of 统计表（1585源/10245边/50852次） |
| §4.2 Case Retrieval | ✅ BM25 baseline 完成 | recall@5=50.7%, recall@10=59.5% (Jaccard≥0.5) |
| §4.5 Verification | ✅ 确定性 check 完成 | 6个纯规则check，真实数据98.1%通过 |
| §4.4 Abstraction+Inference | ✅ v1 完成（需迭代） | Qwen3:14b + 知识RAG，exact_fact_f1≈0.25 |
| §4.6 Verbalize | ✅ deterministic v1 | 保真组装 assertion/location/modifier/suggestive_of |

## 目录结构

```
ZCodeProject/
├── src/
│   ├── norm/                    # §4.0 归一化模块（零第三方依赖）
│   │   ├── head_norm.py         # head 归一化：lemmatizer + 别名表 + 短语归并
│   │   └── location_norm.py     # location 规范化：laterality + region + lobe
│   ├── knowledge/               # §4.3 知识源
│   │   └── suggestive_table.py  # suggestive_of 统计表（构建器+查询器）
│   ├── retrieval/               # §4.2 案例检索
│   │   ├── serialize.py         # finding 序列化器（BM25 token + 可读文本）
│   │   ├── bm25_retriever.py    # BM25 检索器（建索引+查询+持久化）
│   │   └── evaluate.py          # Recall@K 评测
│   └── verbalize/               # §4.6 结构化 fact 保真组装为文本
│       └── verbalizer.py        # 确定性 verbalizer + source fact trace
├── scripts/
│   ├── build_norm.py            # §4.0 全量构建 head/location 映射
│   ├── build_suggestive_table.py# §4.3 全量构建 suggestive_of 统计表
│   ├── build_and_eval_retrieval.py # §4.2 构建索引+评测
│   └── inspect_other_region.py  # 诊断 region=other 的 location
├── outputs/                     # 产物（入库，可重建）
│   ├── head_norm.json           # §4.0: raw head → 归一 head
│   ├── location_norm.json       # §4.0: raw location → {laterality, region, lobe}
│   └── suggestive_of_table.json # §4.3: source_head → 候选诊断(频次/置信度)
├── docs/                        # 研究方法文档
├── profile_data.py              # 全量数据画像（附录A来源）
├── sample_*.py                  # 采样诊断脚本（heads/locations/suggestive）
├── requirements.txt             # 可选依赖（核心代码零依赖）
└── .gitignore
```

## 数据基础

- `E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl`
- N = 140000 条 finding/impression 结构化对（RadGraph v3）

## §4.0 用法

```cmd
# 构建/重建两份归一化产物（约 1-2 分钟，纯 CPU）
python scripts\build_norm.py

# 跑归一化模块自测
python src\norm\head_norm.py
python src\norm\location_norm.py
```

代码零第三方依赖。可选装 `nltk` 提升 head 单复数归一质量：
```cmd
pip install nltk
python -c "import nltk; nltk.download('wordnet')"
```

## §4.0 关键设计决策（防止上下文丢失）

1. **head 归一化三层**：纯单复数(lemmatizer) + 不规则/同义变体(别名表) + 短语归并(规则)。
   - 别名表见 `head_norm.py: ALIAS_TABLE`，基于采样数据手写。
   - 全量结果：head 词表 17872 → 14215，压缩率 20.5%。

2. **带修饰的复合 head 一律保留**（如 `acute infiltrate` 不归并到 `infiltrate`）。
   - 理由：急性/慢性修饰在 §4.5 的 assertion/evidence check 中是依据，归并造成不可逆信息损失。

3. **location 输出四元组**：`(laterality, region, lobe, needs_review)`。
   - laterality: left/right/bilateral/none（数据确认几乎只有 3 类 + unilateral 408 条标 review）
   - region: 受控解剖词表 13 类（lung/pleura/hilar/mediastinum/heart/vasculature/chest/bone/soft_tissue/abdomen/airway/bronch/device/other）
   - lobe: upper/middle/lower/none（叶段是临床定位核心，单独抽）
   - 全量结果：region=other 从 41% 降到 29%，剩余长尾留给 §4.5 verifier 兜底。

4. **词表顺序敏感**：vasculature 在 lung 前（让 pulmonary venous 归血管）；bone 在 chest 前（让 bony thorax 归骨）。
