# Exp-014: Probing 深层问题实验集

## 概述

从 brainstorm 中挑选的最有价值的 probing 方法论实验，围绕 encoding vs use、信息生命周期、信息来源分解、理解模型 probing 四个方向。

---

## Phase 1: 梯度对齐测试（最便宜，最先做）

### 问题
Probe 方向和模型梯度方向的夹角能否作为 encoding-use gap 的免费指标？

### 方法
- 模型：GPT-2 small / Pythia-1.4B
- 对每层每个概念：
  - 训练 linear probe → 得到 probe 方向 w
  - 前向+反向 → 得到 loss 对该层激活的梯度 g
  - 计算 cos(w, g)
- 和 exp-007 的 ablation effect 比较：cos(w,g) 高的特征是否 ablation effect 也大？

### 预期结果
- cos(w,g) 和 ablation effect 强正相关 → 梯度对齐是免费的因果 proxy，不需要做昂贵的 ablation
- 不相关 → 梯度方向和因果方向是不同的东西

---

## Phase 2: 信息生命周期（IOI 上的 probe+ablation 双曲线）

### 问题
"subject identity" 信息在 GPT-2 small 各层的编码强度和因果重要性如何变化？

### 方法
- 模型：GPT-2 small，IOI 任务
- 概念："重复出现的名字是谁"
- 每层：probe accuracy + AMNESIC ablation Δloss
- 画双曲线，和 Wang 2022 已知 circuit（Name Mover heads 在 L9-10）对照

### 预期结果
- 双曲线峰对齐 → encoding=use 在 IOI 上成立，lifecycle 和 circuit 互印证
- probe 峰在前、ablation 峰在后 → 信息先编码后消费，中间有"搬运"过程
- probe 持续高但 ablation 只在特定层高 → ghost information 大量存在
- 8 种 lifecycle 分类法的初步验证

---

## Phase 3: 五方向 Head-to-Head

### 问题
同一概念的 5 种方向提取方法，哪个最接近因果方向？

### 方法
- 模型：GPT-2 small 或 Pythia-1.4B
- 概念：3-5 个（情感、时态、大写...）
- 5 种方向：linear probe / DiffMeans / DAS / Park causal inner product / SAE feature
- 每种方向：ablation Δloss + steering 效果
- 计算两两夹角矩阵

### 预期结果
- DAS/causal 方向 ablation 最大 → 因果方法优于统计方法
- 所有方向夹角小 → 对简单概念方向提取方法不重要
- 方向夹角大且因果效果差异大 → 方向选择是 probing 可靠性的关键因素

---

## Phase 4: 三来源分解 Pilot

### 问题
Probe 信号有多少来自输入 bleed-through、多少来自模型计算？

### 方法
- 模型：Pythia-1.4B（训练好的）+ 随机初始化 Pythia-1.4B
- 概念：5 个（含 token-level 和 cross-token 各半）
- 三级 baseline：
  1. Probe on 输入 embedding（layer 0）→ 输入贡献
  2. Probe on 随机网络中间层 → 架构贡献
  3. Probe on 训练网络中间层 - 前两者 → 模型计算贡献
- 每层做，画三个分量的 stacked area chart

### 预期结果
- Token-level 概念（大写/标点）→ 输入贡献占主导（bleed-through）
- Cross-token 概念（情感/主题）→ 模型计算贡献大
- 随机网络 ≈ 输入 embedding → 架构贡献小
- 随机网络 >> 输入 embedding → 架构本身创造了某种结构

---

## Phase 5: 理解模型 Probing

### 问题
声称"理解"的模型能否 probe 出抽象概念？

### 方法

**5a. OthelloGPT — 策略 + 多步前瞻 probing**
- 已知 board state 可 probe（Nanda 2023）
- 新问题：能否 probe 出战略概念？（机动性、frontier discs、稳定边角、中心控制、tile stability）
- 复制 Jenner 2024 在 Leela 上的双线性 2 步前瞻 probe 到 OthelloGPT
- 对照：随机走子训练的 OthelloGPT vs 强模型
- DR-018：Yuan 2025 发现两跳预测 <70%，需验证

**5b. DreamerV3 — RSSM 系统性探测**
- DR-018 确认：**从未被正式 probe 过——最大低垂果实**
- 在 h_t (GRU 确定状态) 和 z_t (32×32 categorical) 上分别做 linear/MLP probe
- 目标变量：位置、速度、奖励预测、对象身份
- 用 MDL probe 区分编码 vs 随机基线
- 因果追踪（ROME 风格）定位哪些 RSSM 单元中介奖励预测
- 关键判据：若 z_t 线性可解码 → 因子化压缩（world model）；若只有 MLP 能解码 → 分布式压缩（state compressor）

**5c. Coconut — 反事实因果探测**
- DR-018 发现：第三方复现否定了 "BFS 叠加"（零 latent token 仍 96.6%）
- 每个 continuous thought 位置训练中间实体 probe
- 沿 probe 方向做反事实干预——修改思维 k 的方向，最终答案是否变？
- 对照：零向量/随机向量占位符 → 若仍正确则 latent thought 是占位符
- 扩展到 Huginn：每次递归迭代做相同测试

**5d. CLIP — 理解 vs 模式匹配**
- 设计对比样本："dog biting man" vs "man biting dog"（语义不同但词相同）
- CLIP 表征能否区分？如果不能 → 只是 bag-of-words 模式匹配
- DR-018 发现：Winoground/ARO 已证实 CLIP 组合性失败；Khajuria 2025 发现单模态内可解码但跨模态失败

**5e. V-JEPA 2 — 守恒量 probing**
- DR-018 发现：Joseph 2026 已证明速度/加速度可线性读出，但**守恒量从未被 probe**
- 设计视频：两体弹性/非弹性碰撞，碰撞前后分别 probe 动量/动能
- 关键判据：若弹性碰撞中 probe 出的"动量向量"碰撞后守恒 → 模型有动力学理解；若不守恒 → 只有运动学统计
- 沿守恒方向做因果干预（强制守恒）→ 下游 VoE 判断是否变？
- 对照：随机 ViT + VideoMAE-v2

**5f. 跨模型统一抽象概念基准（DR-018 实验 5）**
- 四类抽象任务：守恒 / 对称 / 反事实 / 规划
- 应用于：V-JEPA 2, DreamerV3, OthelloGPT, Chess-GPT, CLIP, LLaMA, Coconut, Huginn
- 统一 MDL + 选择性度量 + 同架构随机对照
- 首次产出跨模型可比较的"抽象理解地图"

### 预期结果
- OthelloGPT 策略 probe 成功 → 不只是记棋盘，还有评估；多步前瞻 <70% → 和 Yuan 2025 一致
- DreamerV3 z_t 线性可解码 → world model；只 MLP 可解码 → state compressor
- Coconut 反事实干预改变答案 → 真推理；占位符 baseline 也对 → 快捷方式（和已有第三方复现一致）
- CLIP 组合性失败 → 确认已有文献；单模态可解码 → 失败在对齐阶段
- V-JEPA 2 守恒量守恒 → 动力学理解；不守恒 → 只有运动学
- 跨模型基准 → 首次可比较的"抽象理解地图"
- **所有正面/反面结果都有论文价值**

---

## 算力估计

| Phase | 模型 | 节点 | 时间 |
|-------|------|------|------|
| 1 梯度对齐 | GPT-2 small | 待定 | ~2h |
| 2 Lifecycle | GPT-2 small | 待定 | ~4h |
| 3 五方向 | GPT-2 small | 待定 | ~8h |
| 4 三来源 | Pythia-1.4B | 待定 | ~6h |
| 5a OthelloGPT | OthelloGPT | 待定 | ~2h |
| 5b DreamerV3 | DreamerV3 | 待定 | ~8h |
| 5c Coconut | Coconut | 待定 | ~4h |
| 5d CLIP | ViT-B/32 | 待定 | ~2h |
| 5e V-JEPA 2 | ViT | 待定 | ~8h |
