# Exp-003: 跨架构柏拉图收敛 Pilot 结果

## 配置
| 参数 | 值 |
|------|-----|
| 模型 | Pythia-160M, Mamba-130M, RWKV-4-169M |
| 训练数据 | 全部 The Pile 300B tokens |
| stimuli | 500 sentences (WikiText-103 validation, len>50) |
| 序列长度 | 128 tokens |
| 表征 | 每层 residual stream, mean-pooled |
| 度量 | mutual k-NN (k=10) + linear CKA |
| null calibration | 50 permutations |
| 设备 | CPU |

## 核心结果

### z-score 总结（远高于 null）
| Pair | Best kNN z | Best CKA z |
|------|-----------|-----------|
| Pythia ↔ Mamba | **379.4** | **254.4** |
| Pythia ↔ RWKV | **353.5** | **286.5** |
| Mamba ↔ RWKV | **436.0** | **312.8** |

**所有 z-score 都在 250-436 之间**——远超显著性阈值（z>5）。即使在 130M-170M 这样的小模型上，三种完全不同的架构学到的表征也高度相似。

### 按深度的相似性曲线（见 figures/cross_arch_pilot.png）

**mutual k-NN（左列）**：
- 三对比较都呈现 **U 形曲线**：早层和晚层相似度高（0.6-0.75），中间层下降到 0.4-0.5
- 所有层的 raw 值都远高于 null（~0.02）
- Mamba ↔ RWKV 相似度最高（两个都是非 attention 架构）

**CKA（右列）**：
- 与 kNN 趋势一致但更极端：embedding 层 CKA ~0.85，中间层降到 0.2，最后层回升到 0.65-0.82
- CKA 在中间层的骤降可能反映架构差异最大的处理阶段

### 深度曲线的解读

U 形曲线说明：
1. **早层收敛**：不同架构在 embedding/浅层学到相似的 token 表征（可能因为共享 tokenizer + 训练数据）
2. **中间层分化**：序列处理机制不同（attention vs SSM vs linear attention）导致中间表征分化
3. **晚层重新收敛**：为了预测同样的 next token，不同架构的输出层表征再次对齐

这直接支持 Hoang et al. (arXiv:2510.06640) 的发现："Transformers 在早/中层快速同质化 token 表征，SSMs 在深层才同质化"。

### 三对比较的相对关系

Mamba ↔ RWKV 相似度 > Pythia ↔ Mamba ≈ Pythia ↔ RWKV

两个非 attention 架构（Mamba, RWKV）之间比它们各自与 Transformer 之间更相似——**架构归纳偏置确实影响表征几何**。

## 分析

### 支持柏拉图假说的证据
- 即使 130M-170M 的小模型，跨架构 mutual kNN 已达 0.4-0.75（null 只有 0.02）
- 这意味着**不同架构学到了高度相似的局部邻域结构**

### 但仍需验证的
- 这只是一个 scale 点，无法拟合 scaling law
- 需要在 {410M, 1.4B, 2.8B, 7B} 上重复，看相似度是否随 scale 增大
- CKA 的中间层骤降需要 null-calibration 后确认（Aristotelian critique）
- 500 stimuli 偏少，可能有采样噪声

## 结论

**Pilot 成功：三种架构的表征确实高度相似，信号清晰。** 值得在服务器上做完整 scaling 实验。

## 下一步
- 服务器上跑 Pythia/Mamba/RWKV 的 {410M, 1.4B, 2.8B, 7B} 完整 scaling ladder
- 拟合 s(N) = s∞ - a·N^(-β) 看收敛速率
