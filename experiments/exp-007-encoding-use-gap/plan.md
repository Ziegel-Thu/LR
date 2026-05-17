# Exp-007: Encoding ≠ Use 量化

## 研究问题

在真实 LLM 中，probe 能解码的信息有多大比例是模型实际用于推理的？"encoding-use gap" 有多大？

## 动机

- Braun 2025 (ICML) 在 deep linear networks 中解析证明 encoding ≠ use
- 但没人在真实 LLM 上系统量化这个 gap
- 如果 gap 很大 → 大量 probing 研究的因果推论需要修正
- Wu 2025 AxBench 暗示 SAE 也有类似问题

## 假设

1. 存在大量"ghost information"——probe 高准确率但 ablation 无影响的 case
2. Ghost information 比例在中间层最高
3. 某些特征类型（语法 vs 语义 vs 世界知识）的 gap 大小不同

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-1.4B (pilot), Pythia-2.8B (full) |
| 特征 | 15 个语言/语义特征（见下） |
| 每层 | 训练 linear probe → 记录 accuracy |
| 干预 | 对 probe 子空间做零化投影（AMNESIC-style）→ 记录 loss 变化 |
| 数据 | WikiText-103 validation, 2000 sentences |

### 特征列表
1. 词性 (POS: noun/verb/adj/etc)
2. 命名实体 (NER: person/org/loc)
3. 句法深度 (parse tree depth)
4. 时态 (past/present/future)
5. 数 (singular/plural)
6. 情感 (positive/negative)
7. 形式程度 (formal/informal)
8. 事实性 (factual/opinion)
9. 语言 (English/non-English tokens)
10. 大小写 (capitalized/not)
11. 频率 (high/low frequency word)
12. 主题 (science/politics/sports/etc)
13. 逻辑关系 (causal/temporal/contrast)
14. 否定 (negated/not negated)
15. 引用 (quoted/not quoted)

## 核心分析

### Encoding-Use 散点图
- X 轴: probe accuracy (encoding strength)
- Y 轴: Δloss from ablation (use strength)
- 每个点 = 一层 × 一个特征

### 四象限解读
| | Ablation 影响大 | Ablation 影响小 |
|---|---|---|
| Probe 准确率高 | **编码且使用** | **Ghost information** ← 关键发现 |
| Probe 准确率低 | 不应出现 | 不编码不使用 |

### Ghost ratio
- ghost_ratio = |{(layer, feature): probe_acc > 0.8 AND Δloss < threshold}| / |{(layer, feature): probe_acc > 0.8}|
- 整体 ghost ratio + 按特征类型分层

## 成功标准

- Ghost ratio > 30% → 重要发现
- Ghost ratio < 10% → encoding ≈ use in practice
- 两种结果都有论文价值

## 算力估计

- Probe 训练: 15 features × 32 layers × ~2min = ~16 GPU-hours
- Ablation: 15 × 32 × inference = ~8 GPU-hours
- 可在本地 pilot (Pythia-410M, 5 features, 5 layers)
