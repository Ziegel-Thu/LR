# Exp-009: LRH 系统性测试 — Phase 1+2+3 结果

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

---

## Phase 4: 训练动态（概念方向涌现）

### 配置
- Pythia-1.4B, 7 checkpoints: step 1K/5K/10K/50K/100K/143K/main
- 5 concepts, difference-in-means at each checkpoint × each layer

### Loss 曲线
| Step | Val Loss |
|------|---------|
| 1K | 7.988 |
| 5K | 7.126 |
| 10K | 6.931 |
| 50K | 6.461 |
| 100K | 6.357 |
| 143K | 6.376 |

### 方向涌现：cos(step1K_direction, final_direction)

| 概念 | cos @ step1K |
|------|-------------|
| numeric | **0.634** |
| subword | 0.525 |
| capitalized | 0.525 |
| plural | 0.484 |
| short | 0.370 |

### 关键发现

**概念方向在训练极早期就已部分形成（step 1K, cos=0.37-0.63）。**

- 没有 sharp phase transition — 方向是渐进涌现的
- `numeric` 方向最早形成（cos=0.63），`short` 最慢（cos=0.37）
- 这与 loss 曲线一致：loss 也是连续下降而非突变

**解读**：概念编码不是"涌现"的——它从训练一开始就在逐步建立。这挑战了"能力突然涌现"的叙事，至少在 token-level 概念上如此。

---

## Phase 3: Mamba 上的 LRH

### 配置
- Mamba-130M vs Pythia-160M（同规模对照）
- 7 个概念，difference-in-means 每层提取方向

### 跨架构概念方向对齐

| 概念 | mean cos(Mamba, Pythia) | max |cos| |
|------|------------------------|-----------|
| capitalized | 0.020 | 0.042 |
| numeric | -0.028 | 0.064 |
| punctuation | 0.027 | 0.082 |
| plural | 0.008 | 0.053 |
| short | 0.009 | 0.062 |
| subword | -0.023 | 0.051 |
| starts_vowel | -0.030 | 0.076 |

### 关键发现

**概念方向在 Mamba 和 Pythia 之间完全不对齐（mean cos ≈ 0）！**

- 所有概念的 mean cosine 都在 ±0.03 以内（接近随机）
- max |cos| 也仅 0.04-0.08（768 维空间中的随机 expected max ≈ 0.05-0.08）
- **结论：虽然 exp-003 显示两种架构的 kNN 几何相似（overlap > 0.7），但概念编码的方向完全不同**

这与 exp-008 Phase 2（SAE 特征 MMCS=0.13）完全一致：
- **几何相似 ≠ 方向对齐**
- 两种架构用不同的方向编码相同的信息
