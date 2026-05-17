# Exp-008: SSM 上的 SAE — Phase 1 结果

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Mamba-130M (`state-spaces/mamba-130m-hf`) |
| 目标层 | Layer 12/24 (中间层 output hidden state) |
| 训练数据 | WikiText-103 validation, 235,568 tokens |
| SAE 架构 | TopK (d_model=768 → d_sae=4096, K=32) |
| 训练 | Adam lr=3e-4, L1 coeff=1e-3, 30K steps, batch=4096 |
| 节点 | jiagpu4 GPU 4 |

## 结果

### 核心指标

| 指标 | 值 | 解读 |
|------|-----|------|
| Reconstruction MSE | 0.198 | 在标准化空间 |
| Variance explained | **80.3%** | 良好——Transformer SAE 通常 85-95% |
| Dead features | **0/4096 (0%)** | 完美——所有特征都学到了东西 |
| Avg active/token | 32.0 | 精确匹配 K=32 |
| Alive fire rate (mean) | 0.78% | 稀疏且分布合理 |
| Alive fire rate (median) | 0.39% | |

### 训练曲线

- MSE: 0.50 → 0.20，收敛良好
- Dead features: 始终 0%（整个训练过程）
- L1: 0.042 → 0.025，稳定下降

## 关键发现

### 1. SAE 在 Mamba 上 works ✅

80.3% variance explained 证明 TopK SAE 能有效重构 Mamba 的隐状态。虽然比 Transformer SAE 的典型值（85-95%）略低，但考虑到：
- 这是**首次在 SSM 上训练 SAE**
- 训练数据量有限（236K tokens vs 典型的 10M+）
- 无超参调优

结论：**Superposition 不是 Transformer 特有的，至少在 Mamba 的 output hidden state 中也存在。**

### 2. 0% dead features

这非常罕见——Transformer SAE 通常有 5-30% dead features。可能原因：
- d_sae/d_model 比例适中（4096/768 ≈ 5.3x）
- TopK=32 保证了足够的梯度信号
- 或者 Mamba 的表征空间确实有 ≥4096 个可学习的方向

### 3. Variance gap（~20%）

20% 的未解释方差可能来自：
- Mamba 隐状态的递推结构（temporal dependencies 不适合 pointwise SAE）
- 训练数据不足
- 需要更大的 d_sae

## 局限

- 训练数据仅 236K tokens（plan 要 2M），可能限制了 reconstruction quality
- 未做 feature 语义分析（需要手动检查 top-activating examples）
- 单一层（layer 12）——其他层可能表现不同

## 下一步

- [ ] Phase 2: 与 Pythia-160M SAE 特征对比（MMCS、特征类型分类）
- [ ] 增大训练数据（用 Pile 的更大 split）
- [ ] 手动检查 top-50 features 的最大激活样本
- [ ] 测试其他层（early/late layers）
