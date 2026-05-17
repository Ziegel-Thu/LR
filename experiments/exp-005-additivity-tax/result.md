# Exp-005: Additivity Tax — 结果

## Phase 1: 自然训练网络（IGA ≈ 2.0）

30 个随机种子的 2-hidden-layer ReLU 网络，每种局部行为训练 30 个模型。
Φ^α = (1-α)·IG + α·SHAP，α 从 0 到 1。

| Pair | α=0.0 | α=0.5 | α=1.0 |
|------|:-----:|:-----:|:-----:|
| pos vs neg | 2.000 | 2.000 | 2.000 |
| pos vs quad | 2.000 | 2.000 | 2.000 |
| pos vs flat | 2.000 | 2.000 | 2.000 |
| neg vs quad | 1.667 | 1.700 | 1.567 |
| quad vs flat | 1.933 | 1.967 | 1.967 |

**结论**：所有 α 的 IGA 都远高于 1.0。SHAP (α=1) 在自然训练网络上完全能区分不同局部行为。Bilodeau 的不可能性在 average-case 下不触发。

## Phase 2: Adversarial Bilodeau 构造（IGA ≈ 1.1-1.2）

构造分段可加模型，通过调节远场值使不同局部行为的 SHAP attributions 相同。

| Pair | α=0.0 | α=0.5 | α=1.0 |
|------|:-----:|:-----:|:-----:|
| pos vs neg | 1.200 | 1.233 | 1.100 |
| pos vs quad | 1.333 | 1.200 | 1.167 |
| pos vs flat | 1.167 | 1.133 | 1.133 |
| neg vs quad | 1.200 | 1.133 | 1.200 |
| quad vs flat | 1.267 | 1.167 | 1.167 |

**结论**：Adversarial 构造下 IGA 降至 1.1-1.3（接近随机猜测的 1.0）。Bilodeau 的 worst-case 不可能性确实成立。

## 核心发现

**Bilodeau 的不可能性是 worst-case 定理，不是 average-case 定理。**

| 场景 | IGA (SHAP) | 含义 |
|------|:----------:|------|
| 自然训练网络 | ≈ 2.0 | 完美可区分 |
| Adversarial 构造 | ≈ 1.1 | 接近随机猜测 |

对比差 ~0.9，说明远场对消机制虽然数学上可行，但在自然训练的网络中不会自发出现——自然训练的网络的远场行为与局部行为是相关的（由同一个训练目标决定），不会恰好对消。

## 下一步

1. **Smoothed analysis**：在 adversarial 模型上加不同强度的随机扰动，看 IGA 如何从 ~1.1 回升到 ~2.0——找到 phase transition 点
2. **何时自然出现 adversarial failure**：对抗训练 / RLHF / model editing 后的网络是否更接近 worst-case？
3. **理论化**：定义 "attribution condition number"，量化从 average-case 到 worst-case 的距离
