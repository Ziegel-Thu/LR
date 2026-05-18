# DR-022：表征分析工具如何辅助机器人策略训练？

## 动机

表征分析（probing、CKA、ID、SAE 等）通常被当作事后诊断工具。但最近出现了一些把表征分析**融入训练流程**的工作：
- Lei et al. 2026 "Mechanistic Analysis of Sim-and-Real Co-Training" 用 Structured Representation Alignment (SRA) 诊断 co-training 机制，然后设计 CFG-ADDA 提升 ~20%
- ReVLA 用 probe 诊断 DINOv2 的 catastrophic forgetting → OOD 提升 77%
- D²PPO 用 cluster analysis 发现 representation collapse → dispersive loss 修复
- RS-CL 用 proprio 距离做 contrastive loss 缩小模态 gap

我们想把这条线拉到更系统的层面——**表征分析能否成为机器人训练的标准工具？**

## 任务

### 1. 已有的"表征分析 → 改善训练"案例

全面列出所有已发表的、用表征分析方法辅助机器人训练的工作。包括：
- 用 probe 诊断训练问题（catastrophic forgetting、模态不平衡、representation collapse）
- 用 CKA/Procrustes 做蒸馏 loss 或迁移选择
- 用 domain alignment 分析（如 SRA）指导 co-training / sim-to-real
- 用几何量（ID、manifold capacity）做 early stopping 或 curriculum design
- 用 SAE 特征分析指导 data augmentation 或 reward shaping
- 不限于机器人——CV/NLP 中的类似案例也列出（可迁移到机器人）

### 2. Lei et al. 2026 (SRA) 的方法论细节

- Structured Representation Alignment 具体怎么定义和测量的？
- CFG-ADDA 的训练流程？
- 这套方法可以推广到其他场景吗（不只是 sim-real，还有 cross-embodiment、cross-modality）？

### 3. 可工具化的表征分析方法

哪些表征分析方法**实用性最强**（计算便宜、不需要额外标注、在线可用）？
- 训练中实时监控的（如 probe accuracy 作为 callback）
- 训练后诊断的（如 CKA 做 checkpoint 选择）
- 指导数据混合比例的（如 domain alignment 度量）
- 指导架构选择的（如 modality gap 分析）

### 4. 还没人做但应该做的

综合 DR-019/020/021 的空白，最有价值的"表征分析 × 机器人训练"方向：
- CKA/Procrustes 做 VLA 蒸馏 loss（CV/NLP 已有但机器人没有）
- Probe suite 做训练 diagnostic（像 loss curve 一样持续监控）
- 模态 gap 分析指导多模态融合策略
- Cross-embodiment alignment 分析指导数据混合
- SAE 特征分析区分 memorized vs generalizable features → 指导数据选择

### 5. 如果我们要做一个"机器人训练的表征分析工具包"

应该包含哪些功能？面向什么用户？需要什么接口？已有的开源工具（如 SAE-lens、TransformerLens、nninfo）能复用多少？
