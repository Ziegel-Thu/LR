# Exp-020: Steering Vector OOD Boundary Detection — 结果

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-2.8B (`EleutherAI/pythia-2.8b-deduped`), 32 layers, d=2560 |
| 数据 | WikiText-103 validation, 100 sentences |
| Steering 方向 | 11 个 exp-007 probe features + 3 个随机方向 |
| α 范围 | [-5, +5], 21 等间距点 |
| 节点 | jiagpu4 GPU 2 |
| 运行时间 | ~15min |
| Baseline loss | 6.321 |

## 结果总览

| Feature | 类型 | Layer | Probe Acc | Max\|Δloss\| | Max KL | OOD \|α\| |
|---------|------|-------|-----------|-------------|--------|-----------|
| is_title_case | USED | L1 | 1.000 | 0.049 | 0.888 | 4.0 |
| is_capitalized | USED | L1 | 1.000 | 0.030 | 0.697 | 4.0 |
| is_plural | GHOST | L0 | 0.998 | 0.147 | 0.340 | >5.0 |
| is_numeric | GHOST | L0 | 1.000 | 0.075 | 0.206 | >5.0 |
| is_short | GHOST | L0 | 1.000 | 0.018 | 0.162 | >5.0 |
| is_subword | GHOST | L1 | 1.000 | 0.015 | 0.136 | >5.0 |
| is_stopword | USED | L0 | 1.000 | 0.060 | 0.133 | >5.0 |
| starts_with_space | GHOST | L2 | 1.000 | 0.019 | 0.118 | >5.0 |
| is_high_freq | GHOST | L1 | 0.959 | 0.009 | 0.078 | >5.0 |
| is_punctuation | USED | L12 | 1.000 | 0.086 | 0.021 | >5.0 |
| is_rare | USED | L9 | 0.973 | 0.048 | 0.016 | >5.0 |
| random_0 | RANDOM | L16 | — | — | 0.004 | >5.0 |
| random_1 | RANDOM | L16 | — | — | 0.004 | >5.0 |
| random_2 | RANDOM | L16 | — | — | 0.004 | >5.0 |

## 关键发现

### 1. OOD boundary 是"软"的，不是"硬"的

在 α∈[-5,+5] 范围内，没有观察到 KL 散度的突然跳变（spike）。所有 feature 的 KL-α 曲线都是**平滑递增**的，近似二次或指数增长。这与"存在一个临界点"的假设不符——模型行为的偏移是连续渐变的。

只有 is_capitalized 和 is_title_case 在 α≈4 时 KL 超过阈值（0.5 nats），但这是渐进增长的结果，不是突变。

### 2. Steering 效果量级分三档

| 类别 | Max KL (α=5) | 代表 |
|------|-------------|------|
| **强效** | 0.5–0.9 nats | is_capitalized, is_title_case（L1, USED） |
| **中效** | 0.05–0.35 nats | is_plural, is_numeric, is_short, is_subword, is_stopword, starts_with_space（L0-L2） |
| **弱效** | <0.02 nats | is_punctuation (L12), is_rare (L9) |
| **无效** | ~0.004 nats | random directions (L16) |

### 3. Layer 位置是最强的预测因子

Steering 效果与 **intervention 层的深度**高度相关，而非 ghost/used 分类：

- **L0–L2 的 feature direction**: max KL 0.08–0.89 → 浅层扰动通过更多 transformer 层传播
- **L9–L12 的 feature direction**: max KL 0.016–0.021 → 深层扰动被 LN + attention 快速吸收
- **L16 随机方向**: max KL ~0.004 → 方向性仍有影响（vs 同层 feature 若有的话会更高）

这说明 **OOD boundary 主要由干预位置决定，而非信息的因果重要性**。

### 4. Ghost vs Used：预期之外的结果

原假设：ghost features 的 OOD boundary 更窄。实际结果：

- **Ghost features 的 KL 并不比 Used features 更高**
  - 在同一层（L0），ghost is_plural (0.34) > used is_stopword (0.13)
  - 在同一层（L1），ghost is_subword (0.14) < used is_capitalized (0.70)
- 没有系统性差异——两组 feature 的 KL 范围重叠
- **结论：ghost/used 分类不决定 steering 的 OOD boundary**

### 5. 不对称性：正向 steering 效果更强

几乎所有 feature 呈现 **正 α 方向 KL 更大** 的不对称性：

| Feature | KL(α=-5) | KL(α=+5) | 正/负比 |
|---------|----------|----------|---------|
| is_capitalized | 0.179 | 0.697 | 3.9× |
| is_title_case | 0.128 | 0.888 | 6.9× |
| is_plural | 0.098 | 0.340 | 3.5× |

正方向（增强 feature）比负方向（抑制 feature）的效果更大。这可能因为：
- probe W 的正方向对应 label=1 方向（feature 存在）
- 负方向将表征推向"feature 不存在"，这已经是多数 token 的状态
- 正方向将表征推向更极端的 feature 编码 → 更远离自然分布

### 6. 有趣的 loss 非单调性

is_capitalized 和 is_title_case 在小正 α 时 loss **先下降再上升**：

- is_capitalized: loss 从 6.321 降到 6.296（α=1.5），然后升到 6.778（α=5）
- is_title_case: loss 从 6.321 降到 6.302（α=1.5），然后升到 6.950（α=5）

小幅度 steering 可以 **改善** 模型表现（loss 下降 ~0.02），但大幅度时急剧恶化。这暗示模型的 feature encoding 略微 under-emphasized 了这些方向。

## 图表

- `steering_sweep_all.png`: 所有 feature 的 KL-α 和 loss-α 曲线
- `ood_boundary_comparison.png`: Ghost vs Used vs Random 的 OOD boundary 对比
- `individual/*.png`: 每个 feature 的独立 loss/KL 图

所有图表位于 jiagpu4: `/nvmessd/lifanhong/LR-env/exp020-steering/`

## 结论

1. **Steering 的 OOD boundary 是连续渐变的**，没有尖锐的临界点。"boundary"更像是一个宽阔的过渡带。
2. **干预层的深度是 steering 效果的主要决定因素**——浅层干预传播更远，效果更强。
3. **Ghost/used 分类与 OOD boundary 无显著关系**——这出乎意料，说明 ghost information 的"ghostness"不体现在扰动敏感性上。
4. **小幅度 steering 可以改善 loss**——说明某些 probe 方向确实指向模型可以更好利用的信息。
5. **随机方向几乎无效果**——验证了 probe 方向的特殊性（不仅仅是添加噪声）。

## 下一步建议

- 扩大 α 范围到 [-20, +20] 看是否有硬边界
- 对比不同层的同一 feature（而非只用 best layer）
- 在 Gemma-2-2B 上重复，对比不同架构的 steering 敏感性
- 研究多方向同时 steering 的交互效应
