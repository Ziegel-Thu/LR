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
| 节点 | 待定（等模型资产 checklist，服务器有就服务器跑，没有就本地跑）|

对每个 attention head 做 activation patching，计算 patched activation 的 Mahalanobis distance。画 distance vs IIA 可靠性的关系。

## Phase 2: Ablation Baseline 对比

三种 baseline（mean / zero / resample）分别做 IOI circuit discovery，比较 circuit topology 差异和 faithfulness gap。

## Phase 3: Trivialization Baseline

随机初始化 GPT-2 small 做 IOI circuit discovery，作为 Wang 2022 的对照。

## Phase 4: 暗物质探索

1. MLP 纳入 circuit search（逐 MLP 层 ablation）
2. 组合 ablation（circuit 外 head pairs/triples）
3. Circuit 外 random ablation scaling（Δloss 是否随 k 线性增长）

## 成功标准与预期结果

- 换 baseline 后 circuit 大变 → 现有结论部分是 artifact（高影响力负面结果）
- 换 baseline 后 circuit 不变 → 方法论 robust，给社区 reassurance
- 随机网络 circuit ≈ 训练 circuit → circuit discovery 方法论需重新审视
- 随机网络 circuit 完全不同 → circuit 确实捕获了训练学到的结构
- MLP 纳入后 87% → 95%+ → 暗物质在 MLP 里，circuit search 需扩展
- MLP 纳入后仍 ~87% → 暗物质是 distributed 的，稀疏 circuit 假设有根本局限
