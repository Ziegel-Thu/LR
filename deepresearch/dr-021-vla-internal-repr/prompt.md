# DR-021：VLA 模型内部表征结构

## 动机

VLA（Vision-Language-Action）模型把视觉、语言、本体感觉、动作等多种模态压入同一个 Transformer，但内部表征结构几乎未被研究。π0/π0.5 的 SAE 发现大部分特征是 memorized trajectories（Swann 2026），OpenVLA 被 probe 过但只有两篇论文。我们想了解 VLA 内部到底发生了什么。

## 任务

### 1. 已有的 VLA 内部表征分析

全面列出：OpenVLA、π0/π0.5、Octo、RT-1/RT-2、Diffusion Policy、ACT、Gato 等模型上做过的 probing / SAE / 因果分析 / 几何分析。每个列出论文、方法、核心发现。

### 2. VLA 内部的 modality gap

vision token、language token、proprio token、action token 在 latent space 中的几何关系——有没有人测过？
- 类似 CLIP modality gap 的 cone-effect / cosine distance？
- 不同模态的 token 在残差流中的分布？
- 融合发生在哪一层？

### 3. Action token 如何改变表征空间

VLA 和纯 LLM 的对比：
- 加入 action 预测后，语言/视觉表征怎么变了？
- 预训练的语言知识保留了多少？（Häon 2025 发现 ~25% FFN neurons 被改造用于 action）
- 有没有 representation collapse 的证据？

### 4. "Memorized trajectories" 的普遍性

π0.5 的 SAE 发现大部分 feature 是 memorized trajectories：
- 这在其他 VLA（OpenVLA、Octo、Diffusion Policy、ACT）上是否也成立？
- 怎么区分 memorized trajectory feature vs generalizable motion primitive feature？
- 对 VLA 的泛化能力意味什么？

### 5. 表征分析诊断训练问题

- Representation collapse（Kachaev 2025 在 OpenVLA 上发现）
- 模态不平衡（vision dominance）
- Catastrophic forgetting（ReVLA 用 probe 诊断）
- 有没有系统的"VLA 训练健康检查"方法论？

### 6. 最值得做的实验
