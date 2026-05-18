# Exp-012: 几何量对比 + ID 估计器 Benchmark — 结果

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Pythia 70M → 6.9B（7 scale，复用 exp-006 reps）|
| 度量 | TwoNN ID, MLE ID (K=10/50/100/500), stable rank, effective rank, participation ratio |
| 分析层 | 中间层（每模型 n_layers//2）|

## Phase 1: 多度量对比

### 中间层度量值

各度量在模型规模上**没有 monotonic trend**（与 exp-006 Phase 1 一致）。

## Phase 2: ID 估计器一致性

### 估计器间 Spearman rank 相关

| | TwoNN | MLE-10 | MLE-50 | MLE-100 | MLE-500 |
|---|---|---|---|---|---|
| TwoNN | 1.000 | 0.571 | 0.429 | 0.107 | 0.107 |
| MLE-10 | 0.571 | 1.000 | 0.714 | 0.393 | 0.214 |
| MLE-50 | 0.429 | 0.714 | 1.000 | 0.857 | 0.679 |
| MLE-100 | 0.107 | 0.393 | 0.857 | 1.000 | 0.893 |
| MLE-500 | 0.107 | 0.214 | 0.679 | 0.893 | 1.000 |

### 关键发现

**ID 估计器严重不一致！**
- TwoNN 和 MLE-K500 的 rank correlation 仅 **0.107**——几乎无关
- 相邻 K 值（如 MLE-100 vs MLE-500）correlation = 0.893，但远近 K 之间差异巨大
- **结论：ID 作为表征度量需要标注估计器 + K 值，不同设置可能给出矛盾结论**

## Phase 3: 度量 vs Validation Loss

| 度量 | Pearson r | p-value |
|------|-----------|---------|
| TwoNN ID | 0.554 | 0.197 |
| Participation ratio | 0.538 | 0.213 |
| Stable rank | 0.521 | 0.230 |
| Effective rank | 0.392 | 0.384 |
| MLE ID (K=10) | 0.182 | 0.696 |
| MLE ID (K=100) | -0.402 | 0.372 |
| MLE ID (K=500) | -0.170 | 0.716 |

### 关键发现

**没有任何几何量与 loss 显著相关**（所有 p > 0.19）。

这与 exp-006 Phase 3 形成鲜明对比：
- **跨模型对齐度（kNN）vs loss**: r = -0.952 ✅
- **单模型几何量 vs loss**: r < 0.55, all p > 0.19 ❌

**解读**：表征的"质量"不在于内部几何（维度、秩），而在于跨模型的对齐度。Loss 下降改善的是表征的"外部关系"而非"内部结构"。

## 局限

- 仅测中间层，不同层可能不同
- 7 个数据点（7 个 Pythia scale）统计检验力弱
- 未包含 GeoMLE（需额外安装）

## 下一步

- [ ] Phase 3: Pythia-1.4B checkpoints 训练动态（hunchback 形成）
- [ ] 增加更多模型/架构以提高统计检验力
