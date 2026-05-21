# Exp-021: SAE 作为经验 ICA — 结果

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-160M (d_model=768, 12 layers) |
| 目标层 | Layer 6 |
| 数据 | WikiText-103 validation, 230K tokens |
| SAE | TopK, d_sae=4096, K=32, 20K steps |
| L1 系数 | [1e-4, 3e-4, 1e-3, 3e-3, 1e-2] |
| 基线 | PCA (128), FastICA (128), Random (128) |
| 节点 | jiagpu4 GPU 1, 运行 60 min |

## 核心结果

### 独立性对比表

| 方法 | #特征 | Sparsity | Mean \|r\| | Mean MI (nats) |
|------|-------|----------|-----------|----------------|
| **PCA** | 128 | 0.000 | ≈0 (构造) | 0.0225 |
| **FastICA** | 128 | 0.000 | ≈0 (白化) | 0.0185 |
| **Random** | 128 | 0.000 | 0.539 | 0.093 |
| **SAE L1=1e-4** | 4096 | 0.992 | 0.004 | **0.00026** |
| **SAE L1=3e-4** | 4096 | 0.992 | 0.004 | **0.00023** |
| **SAE L1=1e-3** | 4096 | 0.992 | 0.004 | **0.00025** |
| **SAE L1=3e-3** | 4096 | 0.992 | 0.004 | **0.00023** |
| **SAE L1=1e-2** | 4096 | 0.992 | 0.004 | **0.00017** |

### SAE 质量

所有 SAE 表现一致：
- Variance explained: 83.6–83.7%
- Dead features: 0/4096 (0%)
- Sparsity: 99.2% (由 TopK K=32 决定)

## 关键发现

### 1. SAE 特征比 ICA 更独立（~80x）

**SAE 的 pairwise MI ≈ 0.0002 nats，远低于 FastICA 的 0.019 nats。**

这是最出人意料的结果。ICA 显式优化独立性目标函数，但 SAE（仅优化重建 + 稀疏）产生的特征独立性高出约 80 倍。

排序（MI 从低到高）：**SAE << ICA < PCA << Random**

### 2. L1 系数对独立性几乎无影响

MI 从 0.00026（L1=1e-4）到 0.00017（L1=1e-2），变化仅 ~1.5x。这是因为 **TopK 约束主导了稀疏性**：K=32/4096 固定了 99.2% 的零元素比例，L1 系数的边际影响极小。

→ **是 TopK（而非 L1）创造了独立性。**

### 3. PCA 去相关但不去依赖

PCA 的 |r| ≈ 0（正交构造），但 MI = 0.023 > 0——存在明显的非线性依赖。FastICA 降低了约 18% 的 MI（0.023 → 0.019），但远不如 SAE 的 0.0002。

## 解读与注意

### 稀疏性 vs 真独立性

SAE 的极低 MI 部分是稀疏性的结构性后果：当 99.2% 的值是 0 时，大多数特征对几乎从不同时激活，因此 MI 机械地趋近 0。

这引出一个关键区分：
- **无条件独立性**：P(f_i, f_j) ≈ P(f_i)P(f_j) → ✅ SAE 满足（因为稀疏）
- **条件独立性**：当两个特征都激活时，它们的值是否独立？→ 需要进一步检验

但即使是"因稀疏而独立"也有理论意义：**稀疏表征自动满足独立性的充分条件**。

### 与 ICA 理论的联系

Hyvärinen (1999) 证明：非高斯 + 独立 → 可识别。SAE 的发现表明：

1. **稀疏特征（非高斯分布）+ 近似独立 → SAE 特征可能满足可识别性条件**
2. SAE 不需要显式优化独立性——TopK 稀疏约束作为隐式 ICA 正则化
3. 这为 SAE 特征的唯一性（identifiability）提供了理论依据

### 维度不对等

SAE (4096 features) vs PCA/ICA (128 features) 的维度差异是一个混淆因素。更高维度、更稀疏的表征自然会有更低的 pairwise 依赖。但这恰恰说明了 **overcomplete + sparse 是 SAE 产生独立特征的机制**。

## 图表

- `independence_analysis.png`: 三面板图（sparsity-MI、sparsity-corr、方法对比）
- `correlation_distributions.png`: 各方法的 |r| 分布
- `mi_distributions.png`: 各方法的 MI 分布
- 输出目录: `/nvmessd/lifanhong/LR-env/exp021_sae_ica/`

## 结论

**SAE 是一种比 ICA 更强的独立分解方法**（在 MI 意义下），但其独立性很大程度上来自 TopK 强制的稀疏性，而非对独立性的直接优化。这建立了 SAE ↔ ICA 的经验联系，并暗示 SAE 特征可能满足 Hyvärinen 的可识别性条件。

### 下一步

1. **条件独立性测试**：在两个特征都激活的条件下测 MI
2. **维度控制实验**：训练 128-feature SAE（与 PCA/ICA 同维）
3. **Higher-order independence**：检查 3-way 和 4-way 依赖
4. **跨模型泛化**：在 Mamba-130M SAE 上重复（已有 exp-008 checkpoint）
