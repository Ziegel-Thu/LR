# Exp-011: PID 预测因果效果

## 研究问题

PID 分解出的 unique information 是否预测因果干预效果？

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | GPT-2 small (124M) |
| 节点 | 待定（等模型资产 checklist）|

## 方法

1. 量化激活到 2-3 bit
2. 用 nninfo 计算 5 neurons 关于 target token 的 PID（unique info per neuron）
3. 逐个 neuron 做 zero ablation，记录 Δloss
4. 比较 unique info rank vs ablation effect rank（Spearman ρ）

## 成功标准

- ρ > 0.7 → info 预测 causation
- ρ < 0.3 → PID 不预测因果

## 算力

极轻量：~1 GPU-hour，本地可跑
