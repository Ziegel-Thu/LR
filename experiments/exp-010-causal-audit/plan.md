# Exp-010: 因果抽象方法论审计

## 研究问题

现有 circuit discovery 方法论（activation patching, ablation, IIA）的结论有多可靠？有多少是方法论 artifact？

## 动机

- Sutter 2025 证明随机网络也能 80%+ IIA → 因果抽象可能 trivial
- Wang 2022 IOI circuit 只解释 87% → 13% "暗物质"去哪了
- Zhang & Nanda 2024 发现 denoising ≠ noising patching → 方向性不对称
- Ablation baseline（mean/zero/resample）从未被论证 → 隐藏的自由度
- 没人量化过 off-distribution intervention 的严重程度

## Phase 1: Off-distribution 量化

| 参数 | 值 |
|------|-----|
| 模型 | GPT-2 small (124M) |
| 任务 | IOI (Wang 2022 的 setup) |
| 节点 | 本地 |

对每个 attention head 做 activation patching，计算 patched activation 的 Mahalanobis distance。画 distance vs IIA 可靠性的关系。

## Phase 2: Ablation Baseline 对比

三种 baseline（mean / zero / resample）分别做 IOI circuit discovery，比较 circuit topology 差异和 faithfulness gap。

## Phase 3: Trivialization Baseline

随机初始化 GPT-2 small 做 IOI circuit discovery，作为 Wang 2022 的对照。

## Phase 4: 暗物质探索

1. MLP 纳入 circuit search（逐 MLP 层 ablation）
2. 组合 ablation（circuit 外 head pairs/triples）
3. Circuit 外 random ablation scaling（Δloss 是否随 k 线性增长）

## 算力

全在 GPT-2 small 上，本地可跑，总计 ~12 GPU-hours
