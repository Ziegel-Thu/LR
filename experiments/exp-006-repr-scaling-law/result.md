# Exp-006: 表征质量 Scaling Law — Phase 1 结果

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Pythia 70M / 160M / 410M / 1B / 1.4B / 2.8B / 6.9B (全 7 scale) |
| 数据 | WikiText-103 validation, 1728 stimuli, len>50 |
| 序列长度 | 256 tokens, mean-pooled |
| 度量 | CKA, shape metric (Williams), mutual kNN (k=10), TwoNN ID, MLE ID, stable rank, effective rank |
| 节点 | jiagpu4 |

## 结果

### Intra-model 度量（中间层）

| 模型 | Params | Stable Rank | Effective Rank | TwoNN ID | MLE ID |
|------|--------|-------------|----------------|----------|--------|
| Pythia-70M | 70M | 1.5 | 6.7 | 15.6 | 12.5 |
| Pythia-160M | 160M | 1.2 | 3.0 | 14.8 | 11.8 |
| Pythia-410M | 410M | 1.0 | 1.2 | 13.7 | 10.1 |
| Pythia-1B | 1B | 1.0 | 1.3 | 14.5 | 11.3 |
| Pythia-1.4B | 1.4B | 1.0 | 1.5 | 15.3 | 12.9 |
| Pythia-2.8B | 2.8B | 1.3 | 5.8 | 14.7 | 14.5 |
| Pythia-6.9B | 6.9B | 1.1 | 2.3 | 13.3 | 13.0 |

### Cross-model 度量（相邻 scale pair，best layer）

| Pair | CKA | kNN | Shape Distance |
|------|-----|-----|----------------|
| 70M ↔ 160M | 0.9726 | 0.7692 | 0.3523 |
| 160M ↔ 410M | 0.9937 | 0.7861 | 0.2509 |
| 410M ↔ 1B | 0.9943 | 0.7980 | 0.0976 |
| 1B ↔ 1.4B | 0.9945 | 0.8428 | 0.1016 |
| 1.4B ↔ 2.8B | 0.9950 | 0.8428 | 0.1961 |
| 2.8B ↔ 6.9B | 0.9934 | 0.8473 | 0.2046 |

### Power Law 拟合: s(N) = a · N^β

| 度量 | β | R² | 解读 |
|------|---|-----|------|
| kNN overlap | 0.029 | **0.897** | 最强 scaling，近乎 power law |
| CKA | 0.005 | 0.495 | 中等拟合，接近饱和 |
| Shape distance | 1.656 | 0.000 | 无 power law |
| Stable rank | 0.001 | -0.010 | 无 scaling 趋势 |
| Effective rank | 0.001 | -0.001 | 无 scaling 趋势 |
| TwoNN ID | 0.001 | -0.034 | 无 scaling 趋势 |
| MLE ID | 0.034 | 0.218 | 弱正趋势 |

## 关键发现

### 1. kNN 是唯一呈 clean power law 的度量 (R²=0.90) ✅

跨 scale 的 kNN overlap 从 0.77 (70M↔160M) 单调增长到 0.85 (2.8B↔6.9B)，拟合 power law R²=0.90。β=0.029 表明增长很慢——对数 scale 上接近线性。

这验证了 Huh 2024 的定性发现（bigger → more aligned），并**首次给出了定量幂律**。

### 2. CKA 接近饱和（>0.99），区分度弱

所有 pair 的 CKA 都 >0.97，在大 scale 几乎不变（0.993-0.995）。CKA 作为 scaling law 的指标太饱和了。

### 3. Intra-model 几何量（ID、rank）无 scaling

Stable rank ≈ 1.0-1.5，effective rank ≈ 1-7，TwoNN ID ≈ 13-16——全部在一个小范围内波动，无 systematic trend。

**解读**：这些度量反映的是表征的内部几何（维度、秩），不随模型规模 monotonically 变化。表征"质量"的 scaling 体现在**跨模型对齐度**（kNN）而非内部结构。

### 4. Shape distance 非单调

Shape distance 在 410M↔1B 最小（0.098），但在 1.4B↔2.8B 和 2.8B↔6.9B 反弹到 0.20。可能因为 d_model 跳变（2048→2560→4096）影响 Procrustes 对齐。

## 生成的图表

- `figures/intra_scaling.png` — 4 个 intra 度量 vs log(params) + power law fit
- `figures/cross_scaling.png` — CKA/kNN/shape vs geometric mean params
- `figures/depth_profiles.png` — 7 个模型的逐层度量深度曲线

## 结论

**表征 scaling law 存在，但仅在跨模型对齐度（kNN）上呈 power law。** 内部几何量（ID、rank）没有 clean scaling。这是一个有价值的发现：

1. 为 Platonic Representation Hypothesis 提供了**定量 scaling law**（之前只有定性观察）
2. kNN 是最敏感的 metric（vs CKA 饱和、shape 非单调）
3. β ≈ 0.03 意味着需要 ~10x scale 才能获得显著的对齐提升

## 下一步

- [ ] Phase 2: 多架构 scaling（已有 exp-003 reps，加 shape/ID/rank 度量）
- [ ] Phase 3: representation alignment vs validation loss 关系
- [ ] 增加 n_perms 做 z-score permutation null（当前是 raw 值）
