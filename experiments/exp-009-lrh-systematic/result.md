# Exp-009: LRH 系统性测试 — Phase 1+2 结果

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-1.4B, 24 layers, d=2048 |
| 概念 | 9 有效（1 跳过: hyphenated 太少）|
| 方法 | Difference-in-means 每层提取方向 |
| 数据 | WikiText-103 validation, 176K tokens |
| 节点 | jiagpu4 GPU 0 |

## Phase 1: 跨层一致性

### 相邻层余弦相似度

| 概念 | 平均相邻 cos |
|------|-------------|
| subword | 0.950 |
| plural | 0.947 |
| numeric | 0.944 |
| punctuation | 0.938 |
| past_tense | 0.935 |
| short | 0.926 |
| starts_vowel | 0.926 |
| capitalized | 0.925 |
| title_case | 0.924 |

### 发现

**概念方向跨层高度一致（mean cos ≈ 0.93）**。这支持 LRH 的一个关键预测：同一概念在不同层用相似的方向表示。没有发现"方向突变层"。

## Phase 2: 概念-概念角度矩阵

### 核心指标

| 指标 | 值 |
|------|-----|
| Off-diagonal mean |cos| | **0.829** |
| Near-orthogonal pairs (<0.1) | **0 / 36 (0%)** |

### 解读

**概念方向之间高度非正交**。9 个概念的方向余弦相似度平均 0.83，没有任何一对接近正交。

这是一个**重要的负面发现**：
1. **挑战 superposition 的正交 packing 假设**：Elhage 2022 预测高维空间中概念应近似正交地 pack。实际数据显示完全不是。
2. **可能原因**：heuristic 标签之间有 correlation（如 capitalized ↔ title_case），导致 DiffMeans 方向也相关。需要用语义上独立的概念重做。
3. **也可能是真实现象**：模型可能不用正交方向编码概念，而是用共享子空间。

## 局限

- 概念标签是 heuristic（token-level 形态学特征），非语义概念
- 多个概念之间有自然 correlation（如 capitalized/title_case, short/punctuation）
- 只测了 Pythia-1.4B 一个模型
- Phase 3 (Mamba) 和 Phase 4 (training dynamics) 未做

## 下一步

- [ ] 用语义上独立的概念重做（需 NLP 标注）
- [ ] Phase 3: Mamba-130M 上重复
- [ ] Phase 4: Pythia checkpoints 训练动态
- [ ] 用 PCA 去除概念间 correlation 后重新分析
