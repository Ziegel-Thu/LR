# Exp-018: Linear Representation Hypothesis on SSM Architectures

## 目的

测试 Linear Representation Hypothesis (LRH) 是否在 SSM 架构（Mamba、RWKV）上成立。
LRH 在 Transformer 上已有大量验证，但尚无论文在 SSM/RNN 架构上测试。

## 假设

1. **Cross-layer consistency**: 如果 LRH 成立，概念方向在不同层之间应保持一致（高 cosine similarity）
2. **Concept orthogonality**: LRH + superposition 预测不同概念方向应近似正交（低 cosine similarity）
3. **Direction stability**: 概念方向应在不同数据子集上保持稳定（高 cosine similarity）

## 背景

- Exp-009 在 Pythia 上的发现：cross-layer cos=0.93, concept non-orthogonality cos=0.83
- Exp-007 发现 Mamba ghost ratio 89.8%（编码了信息但不使用）
- Cross-architecture alignment: Mamba↔Pythia concept directions cos≈0

## 方法

### 数据

复用 exp-007 缓存的 hidden states（WikiText-103 validation, 176,213 tokens）：

| 模型 | 节点 | 路径 | 层数 | d_model |
|------|------|------|------|---------|
| Pythia-2.8B | jiagpu4 | `/nvmessd/lifanhong/LR-env/exp007_28b/` | 32 | 2560 |
| RWKV-3B | jiagpu4 | `/nvmessd/lifanhong/LR-env/exp007-rwkv3b/` | 32 | 2560 |
| Mamba-2.8B | jiagpu5 | `/nvmessd/lifanhong/LR-env/exp007-mamba28b/` | 64 | 2560 |

### 特征（11个 token-level heuristic features）

is_capitalized, is_plural, is_high_freq, is_numeric, is_punctuation, is_short,
is_stopword, is_rare, is_title_case, starts_with_space, is_subword

### 方向提取

Logistic Regression probe（sklearn, max_iter=1000, C=1.0），取 `coef_[0]` 作为概念方向。

### 测试

1. **Cross-layer consistency**: 每个 feature 在各层训练 probe，计算层间 probe 方向的 pairwise cosine similarity
2. **Concept orthogonality**: 每层上，计算不同 feature probe 方向之间的 pairwise cosine similarity
3. **Direction stability**: 50/50 split 数据，分别训练 probe，比较两个方向的 cosine similarity

### 层采样

均匀采样 8 层（Pythia/RWKV 32 层采 8 层，Mamba 64 层采 8 层）。

## 执行

- jiagpu4: Pythia-2.8B + RWKV-3B
- jiagpu5: Mamba-2.8B（数据仅在 jiagpu5）
- 输出: `/nvmessd/lifanhong/LR-env/exp018-lrh/{pythia28b,rwkv3b,mamba28b}/`

## 预期结果

- Pythia 应复现 exp-009 的 high cross-layer consistency
- Mamba/RWKV 是否也有 high cross-layer consistency？
- Mamba 的 ghost info 现象是否反映在 LRH 测试中？
- Concept orthogonality 是否因架构而异？
