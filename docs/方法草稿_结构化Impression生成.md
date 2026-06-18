# 结构化中间表示驱动的可解释 CXR Impression 生成方法

> **状态**：方法 section 草稿 v0.2，目标逐步演化为实验开发文档
> **与 `技术路线.md` 的关系**：互补。`技术路线.md` 记录"从结构化对中挖掘规则/权重"路线及其瓶颈（exact_fact_f1 ≈ 0.226）；本文件记录**新路线**——结构化中间表示 + 知识引导 LLM 推断 + 自反思验证，用以突破规则挖掘的稀疏天花板。
> **数据基础**：`E:\程序\医疗影像项目\数据\rexgradient\rexgradient_radgraph_structured_v3_full.jsonl`（N=140000）
> **标注约定**：`[TODO]` = 待设计；`[待验证]` = 待实验确定；`[复用v3]` = 直接复用技术路线 v3 已有实现；`[改]` = 本版相对 v0.1 的修订。

---

## 1. 问题定义

### 1.1 结构化事实表示（5-元组）

沿用 RadGraph 的结构化抽取结果，一条 impression fact 表示为 5-元组：

```
g = ( assertion, head, locations, modifiers, suggestive_of )
```

- `assertion` ∈ {**present**, **absent**, **uncertain**}：决定**阴阳性**。数据画像确认这是闭合 3 类（RadGraph 的 `definitely present / definitely absent / uncertain`，`measurement::*` 视为噪音合并）。
- `head`：核心临床实体（disease / finding），如 pneumonia、opacity。
- `locations`：解剖位置集合，元素带**左右侧/双侧**，如 {left lower lobe}。
- `modifiers`：程度/属性修饰，如 mild、chronic、stable。
- `suggestive_of`：提示性指向（impression 专属），如 opacity suggestive_of pneumonia。

> **数据对齐**：JSONL 里 `impression_graph_compact` 已是 `{finding, head, assertion, locations, modifiers, suggestive_of}` 结构，与本 5-元组同构，**无需重新构造**。`generation_pair.{input,target}` 已配好对，直接可用。

### 1.2 任务

**结构化 finding → 结构化 impression**：给定 F（`findings_graph_compact`），产出 Ĝ，使得 Ĝ 在 fact 级结构化指标上逼近 G（`impression_graph_compact`）。再由小 LLM 将 Ĝ 组装成自然语言 impression。**结构化正确性在 Ĝ 层评估，与 verbalize 解耦。**

### 1.3 评估指标（沿用现有实现）

| 指标 | 含义 | 当前(v3) |
|---|---|---|
| `exact_fact_f1` | 5-元组完全匹配 | 0.22598 |
| `head_assertion_location_f1` | (head, assertion, location) 匹配 | 0.26061 |
| `head_assertion_f1` | (head, assertion) 匹配 | 0.39404 |
| `head/entity_f1` | head 匹配 | 0.44292 |
| `location_error_rate` | 位置错误率 | 0.34182 |

> **观察**：head/entity(0.443) > head_assertion(0.394) > HAL(0.261) > exact(0.226)。越往细粒度分数越低，说明规则方法只能学好"实体存在"这一层——这是规则挖掘稀疏天花板的直接证据。

---

## 2. 动机

### 2.1 规则挖掘的天花板根因（数据画像证实）

数据画像量化了"涌现实体"问题：**152930 次涌现事件**（impression head 不在同样本 finding heads 中）。top 涌现 head 是 `disease`(30897)、`pneumonia`(7828)、`process`(6111)、`edema`(5101)。这些 head 在 finding 图中**无对应节点**，任何 copy/transform/位置继承都无法生成——这是规则方法的结构性上限，靠优化规则挖掘无法突破。

### 2.2 把任务分解为三个性质不同的操作

| 操作 | 性质 | 例子 | 现有规则覆盖? | 新方案由谁负责 |
|---|---|---|---|---|
| **Selection 选择** | 确定性 | 哪些 finding entity 进入 impression | 部分 | 规则/检索 |
| **Abstraction 抽象** | 半确定，本体可约束 | 多个观察 → inflammatory process | 否 | 知识引导 LLM |
| **Inference 推断** | 涌现性，需先验 | 实变 → pneumonia | 否 | 知识引导 LLM |

### 2.3 数据画像的关键利好

- **impression 极简**：median=1 条 fact（mean 1.95，max 23）。impression 不是长文本生成，而是**从 finding 提炼 1-2 个诊断结论**——这意味着 LLM 推断输出规模小（1-3 fact），schema 约束 + 验证环的代价完全可承受。
- **suggestive_of 显式标注了 19035 次推断边**：pneumonia(4690)、atelectasis(3586)、edema(1490)。**数据本身埋着 ground-truth 的跨概念边**，是第一优先级知识源。
- **案例库极稠密**：14 万样本 + impression median 1，几乎任何 finding 组合都能找到近邻——检索器用最简单的稠密检索即可。

### 2.4 对标工作的缺口

- **CLARIFID**：Findings→Impression 两步推理 + CheXbert reward，但推理过程仍是**黑盒自然语言**。→ 我们补"结构化中间表示"。
- **DeepRare (Nature 2026)**：agentic LLM 达成可追溯推理，但可解释性停在 **citation 级**；其失败模式（41% 权重错误、38.5% 表型混淆）证明自由文本推理链无法保证事实级结构正确性。→ 我们补"fact 级结构正确性"。
- **ReXKG**：放射学知识图暴露盲区。→ 我们用它补跨概念边。

---

## 3. 方法概述（架构）

```
                ┌─────────────────────────────────────────────────────────┐
                │              RadGraph 结构化 finding 图 F                │
                └──────────────────────────────┬──────────────────────────┘
                                               │
            ┌──────────────────────────────────┼──────────────────────────┐
            ▼                                  ▼                          ▼
   ① Selection 选择                  ② Case Retrieval 案例检索   ③ Knowledge Lookup 知识查询
   (规则/检索,[复用v3])             (稠密检索 TopK 结构化对)     (语料内 suggestive_of 统计优先)
            │                                  │                          │
            └──────────────┬───────────────────┴──────────────────────────┘
                           ▼
        ④ Abstraction + Inference  (知识引导 LLM, schema 约束输出 5-元组)
                           │ Ĝ_cand
                           ▼
        ⑤ Self-Reflective Verification 自反思验证环  ← 借鉴 DeepRare
          (前置: location 规范化) 逐条 fact 结构化验证; 不过则回环修正
                           │ Ĝ
                           ▼
        ⑥ 结构化 impression Ĝ + 每 fact 溯源 (支撑 F 节点/知识/参考病例)
                           │
                           ▼
        ⑦ Verbalize (小 LLM) → 自然语言 impression
                           │
                  在 Ĝ 层做结构化指标评估
```

**设计原则**：
1. 结构化 fact 是内核（差异化），agentic 推理 + 自反思是引擎（解决稀疏）。
2. 能确定的不用 LLM（Selection / 位置继承用规则）；会涌现的才用 LLM（Abstraction/Inference）。
3. 评估与生成解耦：指标打在 Ĝ 上。

---

## 4. 各组件设计

### 4.0 共享预处理：head 归一化 + location 规范化（前置必做）

数据画像暴露两个必须先解决的数据质量问题，否则下游检索和验证都会失真。

**(a) head 归一化**（finding head 词表 12532 unique，碎片化严重）

| 现象 | 例子（画像频次） | 处理 |
|---|---|---|
| 单复数 | opacity 11578 / opacities 8670；effusion 64092 / effusions 16539 | 词形还原（lemmatize）→ opacity / effusion |
| 大小写 | Chronic / chronic | 全小写 |
| 极保守别名 | lungs/lung（保留 v1 审核表） | 复用 v1 alias 表 |

> **职责边界**：只做形态归一，**不合并左右侧、不改 assertion**（与 v1 审核式归一化一致）。归一化映射表固化为一份 `head_norm.json`，所有下游模块共用。

**(b) location 规范化 + laterality 抽取**（location 是自由文本，14431 unique）

数据画像：laterality token 频次 right 49090 / left 45047 / bilateral 18178 / unilateral 8（几乎只有 3 类）。这些 token **埋在 location 文本里**（"left pleural"、"interstitial bilaterally"），verifier 无法直接用。

规范化把每个 location 文本解析为结构化三元组：

```
location_struct = ( laterality ∈ {left, right, bilateral, none},
                    region      ∈ {归一解剖区},      // 用一份受控解剖词表对齐
                    raw         = 原始文本 )          // 保留用于溯源
```

- laterality 抽取规则：`bilateral|bilaterally` → bilateral；含 `left` 且不含 `right` → left；反之 right；含 `unilateral` → 标记需人工/LLM 判定；其余 → none。
- region 对齐：`[待验证]` 维护一份 ~50 条的解剖词表（lung/lobe/pleura/mediastinum/cardiac/bone…），location 文本模糊匹配到 region。
- 输出一份 `location_norm.json` 缓存，避免重复解析。

> 这两份归一化是 §4.2 检索召回和 §4.5 laterality check 的**前置条件**，不解决则两者都失真。

---

### 4.1 Selection 选择（确定性，[复用v3]）

从 F 中选出"应进入 impression"的 finding 子集，标注 `keep_mode ∈ {identity, abstract, drop}`。
- identity 保留的 fact 直接成为 Ĝ 候选（assertion/head/locations 原样继承）。
- 位置继承：identity 默认启用，transformed 需审核批准（[复用v3]）。
- 职责边界：只做"选/不选 + 是否原样保留"，**不做跨概念推断**（那是 ④ 的事）。

---

### 4.2 Case Retrieval 案例检索（★细化，基于数据画像）

**目标**：把 14 万结构化对当案例库，输入新 F → 检索 TopK 结构化相似对，作为 ④ 的 few-shot 证据与 ⑤ 的验证参照。**稠密检索即可，不上对比学习训练**（数据画像证明库已极稠密）。

**4.2.1 序列化格式（喂给检索器/LLM 的 F 文本表示）**

`findings_graph_compact` 序列化为带 assertion 前缀的紧凑行，归一化后 head/region：

```
[FINDING]
+ present: opacity (loc: left lower lobe; mod: mild)
+ present: consolidation (loc: left lower lobe)
+ present: air bronchogram (loc: left lower lobe)
- absent: effusion
- absent: pneumothorax
= uncertain: nodule (loc: right upper lobe)
```

- 前缀 `+/-/=` 对应 present/absent/uncertain（数据画像确认闭合 3 类）。
- head 经 §4.0(a) 归一；location 经 §4.0(b) 规范化，显式标 `(loc: laterality region)`。
- impression 端同理序列化，额外保留 `suggestive_of`。

**4.2.2 检索方案：两阶段（BM25 + 神经重排）**

> 选择两阶段而非纯向量：BM25 对罕见 head（如 "gangrene"）召回稳，神经重排提语义相似。`[待验证]` 是否纯向量更优。

- **Stage 1 召回**：BM25 over 序列化 finding 文本，取 TopM（M=50）。
  - 索引键：归一化 head + laterality + region（不含 assertion？`[待验证]` —— 不含则召回"结构相似但阴阳不同"的负例，对验证环反而有用）。
- **Stage 2 重排**：句向量（`[待验证]` `PubMedBERT` / `RadBERT` / `bge-base`）cos 相似，取 TopK（K=5）。
  - 向量输入：整段序列化 finding。
  - 重排时不排除 Stage1 召回的"阴阳不同"样本——它们作为对照例喂给 LLM，提示"这种 finding 也可能导出相反结论"。

**4.2.3 评估检索器（必做，否则检索质量无据）**

在 14 万对上构造检索评测集：随机抽 1000 finding 作 query，以"同 impression head 集合 Jaccard ≥ 0.5"为相关判定，报 **Recall@K**（K=1,5,10）。这是 §4.2 上线前的 gate。

**4.2.4 输出给下游的接口**

```python
retrieved = [
    {"rank": 1, "id": "...", "findings_serialized": "...", "impression_serialized": "...",
     "impression_facts": [ {head, assertion, locations, modifiers, suggestive_of}, ... ]},
    ...  # TopK
]
```

---

### 4.3 Knowledge Lookup 知识查询（★细化，[改] 优先级调整）

数据画像改变了知识源优先级——**语料内 suggestive_of 统计是第一优先级**，因为它就是标注好的 ground-truth 推断边。

**知识源（按优先级）**：

1. **语料内 suggestive_of 统计**（最强，画像已证 19035 条）：
   - 离线构建：`finding_head → suggestive_of_head` 的条件概率表，带频次与 assertion。
   - 例：`opacity → suggestive_of pneumonia (4690次, 多为 uncertain)`。
   - 用途：直接给 ④ 的 Inference 提供"从哪些观察可推断哪些诊断"的候选。
2. **CheXpert 14 类标签**：作为合法 impression head 的**受控词表**（pneumonia/atelectasis/effusion/cardomegaly/edema/…），约束 ④ 输出 head 落在合法集，降幻觉。数据画像的 impression head top20 几乎全在这 14 类内。
3. **UMLS / SNOMED CT**：finding–disease 关联，补语料没覆盖的边。`[待验证]` 是否必要。
4. **ReXKG 式自建 KG**：`[TODO]` 后续可选，把语料内 suggestive_of + 共现统计升维成图。

> `[改]` v0.1 把外部本体排第一，v0.2 改成语料内 suggestive_of 第一——因为数据画像证明它是现成、稠密、带标注的推断边。

---

### 4.4 Abstraction + Inference（知识引导 LLM，schema 约束）

**替代"规则映射"的核心环节，负责所有涌现/抽象 fact。**

- **输入**：F_sel + TopK 检索病例 + 知识查询结果（suggestive_of 候选 + CheXpert 词表）。
- **输出**：候选 fact 集 Ĝ_cand（合法 5-元组，1-3 条，匹配 impression median=1）。
- **机制**：
  1. LLM（`[待验证]` GPT-4o / Llama-3-Med / 本地 7B）做软推理，完成"实变→pneumonia"这类涌现推断——其依据来自 §4.3 的 suggestive_of 候选。
  2. **schema 约束**：JSON schema / grammar 约束输出 5-元组，head 必须落在 CheXpert 14 类（硬约束 grammar 优先，`[待验证]` vs 软约束 prompt）。
  3. **few-shot**：4.2 检索到的结构化对作 in-context 示例。

> schema 约束的 5-元组输出本身即解释中间层——把 LLM 推理钉在可计数结构上，不再是黑盒自然语言。

---

### 4.5 Self-Reflective Verification 自反思验证环（★细化，基于数据画像）

**借鉴 DeepRare，但验证对象是结构化 5-元组（非自由文本）——这是对 DeepRare 缺失的"结构化正确性层"的补齐。**

**前置**：location 必须经 §4.0(b) 规范化，laterality 才可检查。

**4.5.1 check 函数清单（每维独立打分，正交于 NLG 指标）**

| check | 输入 | 判定逻辑 | 失败示例 | 修正动作 |
|---|---|---|---|---|
| `laterality` | g, F | g 的 laterality 必须被 F 支撑观察的 laterality 覆盖（g=left ⇒ F 须有 left 观察支撑） | F 只有 right opacity，g 却 left pneumonia | re-locate：laterality 改成 F 支撑侧；或 drop |
| `assertion` | g, F | 阴阳性一致性：g=absent pneumonia ⇒ F 不应有强阳性肺炎征象；g=present ⇒ 须有支撑 | F 阴性，g 却 present pneumonia（无证据推断） | flip-assertion；或 drop |
| `evidence` | g, F, retrieved, knowledge | g.head 是否有来源：(a) F 中 identity 继承；(b) F 中 suggestive_of；(c) 检索病例支持；(d) 知识候选。四源皆无 → fail | 凭空出现的罕见 head | drop；或降 assertion 为 uncertain |
| `head_legal` | g | g.head ∈ CheXpert 14 类（或语料 head 词表） | 幻觉 head | 映射到最近合法 head；或 drop |
| `location_legal` | g | g.locations 的 region ∈ 解剖词表 | 非法解剖位 | 映射或 drop |
| `conflict` | g, Ĝ\{g} | 不存在互斥 fact（如 present 与 absent 同 head） | 同 head 阴阳性矛盾 | 保留证据更强者，drop 另一 |
| `redundancy` | g, Ĝ\{g} | 无重复 fact（同 head+assertion+loc） | 重复 | merge |

**4.5.2 修正动作集**（[复用v3] 编辑操作扩展）：`{re-locate, flip-assertion, drop, merge, abstract, add}`

**4.5.3 终止**：所有 fact 通过，或达最大迭代 T（`[待验证]` T=2，因为 impression median=1，验证开销小）。

**4.5.4 伪代码**

```text
function VerifyAndRepair(F, Ĝ_cand, retrieved, knowledge, T):
    Ĝ ← Ĝ_cand
    for t in 1..T:
        issues ← []
        for g in Ĝ:
            for name, check in CHECKS:                     # 4.5.1 七个 check
                res ← check(g, F, retrieved, knowledge, Ĝ)
                if not res.ok:
                    issues.append((g, name, res, res.suggested_action))
        if issues is empty: break
        Ĝ ← LLM_repair(Ĝ, issues, F, retrieved, knowledge)  # schema 约束的修正
    return Ĝ, trace(issues)                                  # trace 即溯源
```

> **关键差异于 DeepRare**：DeepRare 验证自由文本假设，41% 权重错误源于此；我们验证结构化 5-元组，每维可确定性/半确定性检查——laterality/assertion/evidence 三类错误正是 NLG 指标抓不住、而我们能抓的。

---

### 4.6 Verbalize 组装自然语言（解耦层）

- 输入：验证后 Ĝ；输出：自然语言 impression。
- 小 LLM，schema-guided（按 5-元组拼装，保留左右侧/阴阳性/位置）。
- 定位：纯表达层，**impression 正确性已在 Ĝ 层判定**，verbalize 瑕疵不影响临床正确性。

---

## 5. 知识源与数据资产

| 资产 | 角色 | 状态 |
|---|---|---|
| `rexgradient_radgraph_structured_v3_full.jsonl` | 案例库 + 评估集 + few-shot 池 | 已有 |
| 语料内 suggestive_of 统计表 | 第一优先级知识源 | 离线构建（§4.3） |
| CheXpert 14 标签 | 合法 impression head 受控词表 | 公开 |
| `head_norm.json` | head 归一化映射 | `[TODO]` §4.0(a) |
| `location_norm.json` | location 规范化缓存 | `[TODO]` §4.0(b) |

---

## 6. 可解释性机制

| 可解释性轴 | 含义 | 我们覆盖 |
|---|---|---|
| Citation 级溯源性 | 证据来自哪条病例 | 辅助覆盖（检索病例/知识引用） |
| **Fact 级结构正确性** ★ | 左右侧/阴阳性/位置对不对 | **核心覆盖（差异点）** |
| Fact 级归因 | 每 g 支撑它的 F 节点/知识/参考病例 | 核心覆盖（§4.5 trace） |

每个 impression fact 附溯源元数据：`supporting_findings`、`knowledge_used`、`reference_cases`。支持后续放射科医生盲评（借鉴 DeepRare 专家评估范式）。

---

## 7. 学习与优化（最小化学习）

| 模块 | 是否学习 | 方式 |
|---|---|---|
| Selection | 否 | 规则 [复用v3] |
| Case Retrieval 编码器 | 否 | 零样本 BM25 + 预训练句向量（画像证稠密，不需训练）`[待验证]` |
| Abstraction+Inference LLM | 否/轻量 | training-free few-shot；`[待验证]` 轻量 SFT 稳定 5-元组输出 |
| Verifier | 半 | 规则确定性 check + LLM 语义判官 |
| Verbalizer | `[待验证]` | few-shot / 轻量 SFT |

---

## 8. 评估方案

### 8.1 自动评估（主表）
在 Ĝ 层算 §1.3 五指标，对比：规则基线(v3)、CLARIFID-style 端到端、纯 LLM 直接生成、本方法消融。

### 8.2 消融
去 Case Retrieval；去 Knowledge Lookup；去 Verification；去 schema 约束。预期：验证环与 schema 约束对 location_error_rate / exact_fact_f1 提升最大。

### 8.3 专家评估
放射科医生对（结构化 impression + 溯源）盲评：事实正确性认同率、溯源性认同率。

### 8.4 错误分析（对标 DeepRare 失败模式）
统计本方法在"左右侧错误/阴阳性错误/位置错误/无证据推断"上的分布，论证结构化中间表示相对自由文本推理链的优势。

---

## 9. 实现备注（向开发文档演化）

- `[TODO]` 数据流：JSONL → 内存 fact 对象（Python dataclass）→ 各模块统一接口
- `[TODO]` schema 约束选型：JSON schema 校验 / outlines / guidance / xgrammar
- `[TODO]` baseline 复现：CLARIFID、纯 LLM 直接生成
- `[TODO]` 实验配置记录模板（数据切分、模型版本、prompt、温度、T）

---

## 10. 与现有工作的定位

| 工作 | 可解释性轴 | 与我们的关系 |
|---|---|---|
| CLARIFID (2025) | 结构化目标(CheXbert)，推理黑盒 | 借其结构化目标，补结构化中间表示 |
| DeepRare (Nature 2026) | citation 级溯源性 | 借自反思环+案例检索+专家评估；补 fact 级结构正确性 |
| ReXKG (2024) | 知识盲区暴露 | 作知识源/跨概念边基础设施 |
| RADO (ACM 2025) | 安全约束 LLM 推理 | 对照"结构化约束防黑盒" |

**一句话定位**：我们针对 CXR finding→impression，保留 LLM 跨概念推断能力（解决规则稀疏天花板），但用**结构化 5-元组 fact 作显式中间表示 + 自反思结构化验证**，使阴阳性/左右侧/位置错误可被直接评估与归因——这是对 CLARIFID（结构化目标但黑盒推理）与 DeepRare（可追溯但仅 citation 级）在**事实级结构正确性维度**上的补充。

---

## 附录 A：数据画像（来自 `profile_data.py`，N=140000）

> 决定本文档所有 `[待验证]`/参数的事实依据，固化于此防止丢失。

- **样本数**：140000
- **Assertion 闭合集**：present / absent / uncertain（`measurement::*` 视为噪音合并）
- **Finding head**：12532 unique；碎片化（opacity 11578 vs opacities 8670；effusion 64092 vs effusions 16539）→ 必须 §4.0(a) 归一化
- **Impression head**：6745 unique；top: disease 37007, atelectasis 16047, pneumonia 10364, edema 10072 → 几乎全在 CheXpert 14 类内
- **涌现事件**：152930 次（impression head 不在同样本 finding heads）；top: disease 30897, pneumonia 7828, process 6111 → 规则挖不出的核心证据
- **suggestive_of**：19035 条推断边；pneumonia 4690, atelectasis 3586, edema 1490 → 第一优先级知识源
- **Location**：14431 unique 自由文本；laterality token 频次 right 49090 / left 45047 / bilateral 18178 / unilateral 8 → 必须 §4.0(b) 规范化抽取 laterality
- **Fact 数/样本**：finding min0 p506 mean5.92 max49；impression min0 p501 mean1.95 max23 → impression 极简，验证环代价可承受
- **词表重叠**：finding∩impression = 4135；impression-only = 2610 → 涌现 head 占比显著

---

## 变更日志
- **v0.2**：基于真实数据画像细化 §4.0/§4.2/§4.3/§4.5；新增附录 A 数据画像；修正知识源优先级（语料内 suggestive_of 提至第一）。
- v0.1：初版方法草稿。

### 下一步
1. 实现 §4.0(a) head 归一化 + §4.0(b) location 规范化，产出 `head_norm.json`/`location_norm.json`
2. 构建 §4.3 语料内 suggestive_of 统计表
3. 跑通 §4.2 最小检索（BM25 only）+ Recall@K 评测，作为检索 gate
4. 跑通最小 pipeline 下限（无检索/无验证，纯 LLM+schema）测 fact 指标起点
