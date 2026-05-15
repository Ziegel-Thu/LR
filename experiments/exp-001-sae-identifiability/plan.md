# Exp-001: SAE 可识别性 Pilot 实验

## 目的

验证 Paulo & Belrose (arXiv:2501.16615) 发现的 SAE 可识别性问题在小模型上是否存在：不同随机种子训练的 SAE 能发现多少共同特征？

## 假设

1. 即使在小模型（Pythia-160M）上，不同种子训练的 SAE 也只能共享部分特征（<60%）
2. 共享率可能与字典大小（expansion factor）负相关
3. 共享率可能与稀疏度（TopK 的 K）有关，存在非单调关系

## 方法/配置

### 模型与数据
- **模型**: Pythia-160M (EleutherAI/pythia-160m-deduped)
- **层**: Layer 6 MLP output, d_model = 768
- **数据**: The Pile 验证集或 FineWeb 子集, 5M tokens
- **序列长度**: 1024 tokens

### SAE 配置
- **架构**: TopK SAE
- **字典大小**: 4096 latents (expansion factor ≈ 5.3×)
- **K**: 32
- **训练**: Adam, lr 3e-4, batch 4096 tokens, 5M tokens
- **种子数**: 5 个不同随机种子

### 评估指标
- **MMCS** (Mean Max Cosine Similarity): 字典列两两最大余弦相似度的均值
- **Hungarian matching**: 最优匹配后 cos > 0.7 的特征对比例（"共享率"）
- **重建质量**: NMSE (normalized MSE)
- **稀疏度**: 平均 L0

### 计算资源
- 本地 M4 Max 48GB
- 预计每个 SAE 训练 30-60 分钟
- 总计 ~4-5 小时

## 预期结果

- 如果共享率 ~30%: 问题在小模型上已存在，值得深入
- 如果共享率 >80%: 问题可能跟模型规模有关，需要在更大模型上验证
- 两种结果都有价值
