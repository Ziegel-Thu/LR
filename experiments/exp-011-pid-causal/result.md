# Exp-011: 信息 vs 因果 — MI 预测 Ablation 效果

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | GPT-2 small, layer 5 |
| 分析 | Top-50 neurons (by variance) |
| MI | Binned estimation, MI(neuron_act, next_token_logprob) |
| 因果 | Zero ablation per neuron → Δloss |

## 结果

| 指标 | 值 |
|------|-----|
| Spearman ρ(MI, |Δloss|) | **0.360** (p=0.010) |
| Pearson r | **0.437** (p=0.002) |

## 解读

**MI 和因果效应之间存在显著但中等强度的正相关。**

- ρ=0.36 说明信息量大的 neuron 倾向于因果效应也大，但远非完美预测
- p=0.01 统计显著
- 40% 的信息无法通过因果效应解释（complement of r²=0.19）
- **结论：信息（MI）可以粗略预测因果性，但 encoding ≠ use 的 gap 在 neuron 层面也存在**

这与 exp-007 的 ghost ratio=71%（层×feature 粒度）一致：信息和因果是相关但不等同的。

## 局限

- 仅 50 neurons at 1 layer
- Binned MI 估计粗糙（受 bin 数影响）
- 未做 full PID 分解（unique/redundant/synergistic）
