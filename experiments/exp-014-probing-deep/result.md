# Exp-014: Probing 深层问题 — Phase 1 梯度对齐测试

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | GPT-2 small, 12 layers |
| 概念 | 5 个 (capitalized, numeric, punctuation, short, plural) |
| 方法 | 每层训练 linear probe → cos(probe_direction, loss_gradient) |

## 结果

| 概念 | Best Probe Acc | Mean |cos(w, g)| |
|------|---------------|---------------------|
| numeric | 1.000 | 0.059 |
| capitalized | 0.997 | 0.069 |
| punctuation | 0.999 | 0.044 |
| short | 0.983 | 0.043 |
| plural | 0.979 | 0.067 |

## 关键发现

**Probe 方向和 loss 梯度方向几乎正交（mean |cos| ≈ 0.05）！**

- 所有概念的 cos(w, g) 都接近 0（0.04-0.07）
- Probe 完美解码信息（acc > 0.98），但 probe 方向和模型 loss 梯度无关
- **结论：梯度对齐 NOT a free causal proxy**

### 解读

这验证了 exp-007 的 ghost information 发现的机制：
- Probe 找到了信息存在的方向
- 但这个方向和模型实际优化的方向（梯度）无关
- 信息是"附带编码"的，不参与 loss 优化
- 这从另一个角度证实了 encoding ≠ use

## 局限

- Loss gradient 是全局的，不是针对特定概念的
- 应该用 concept-specific gradient（如对特定 token 的 loss）
- 仅 5 个 token-level 概念
