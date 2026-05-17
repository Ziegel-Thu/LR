# Exp-012: 几何量对比 + ID 估计器 Benchmark

## 研究问题

1. 哪种几何量最能预测泛化？ID vs stable rank vs effective rank？
2. 不同 ID 估计器在同一表征上是否一致？
3. Hunchback 在训练过程中如何形成？

## Phase 1: 多几何量对比

| 参数 | 值 |
|------|-----|
| 模型 | Pythia 70M~6.9B（共享 exp-006 表征）|
| 度量 | TwoNN ID, MLE ID (K=10/50/100/500), stable rank, effective rank, participation ratio |
| 节点 | jiagpu（和 exp-006 共享基础设施）|

分析：度量间相关矩阵、每种度量 vs loss/accuracy 的 correlation。

## Phase 2: ID 估计器对比

同 Phase 1 表征，对每层每模型跑 TwoNN / MLE(多 K) / GeoMLE。
核心输出：估计器间 rank correlation、K sensitivity 曲线。

## Phase 3: Hunchback 训练动态

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-1.4B, ~15 个 checkpoint (log-spaced) |
| 节点 | jiagpu |

每层 ID 的训练时间序列，和 loss 曲线叠加。

## 成功标准与预期结果

- stable rank 在跨架构 setup 上仍碾压 ID → 挑战 ID 主导叙事
- ID 在跨架构上赢回来 → ID 适合跨架构、stable rank 适合同架构，两者互补
- 估计器一致 → 领域结论 robust
- 估计器不一致 → 需标注每种估计器适用范围
- Hunchback 有 sharp phase transition → "表征涌现"的几何证据
- Hunchback 渐进形成 → 表征结构是连续演化的
