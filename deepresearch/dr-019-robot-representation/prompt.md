# DR-019：机器人模型的表征分析——多模态融合、具身差异与训练辅助

## 动机

机器人模型的训练数据和模态组合与纯语言/视觉模型有本质不同——它们直接和物理世界交互，有丰富的多模态输入（视觉、深度、触觉、本体感觉、动作），而且不同机器人有不同的"身体"（embodiment）。这为表征分析提供了独特的研究角度。同时，表征分析不只是事后诊断——它可以辅助训练（如用表征质量指标做 early stopping、用表征对齐做迁移/蒸馏、用 probing 诊断训练失败等）。

## 任务

### 1. 机器人领域的主流模型和训练范式

列出当前主要的机器人学习模型/框架，包括但不限于：
- 视觉-语言-动作模型（RT-1/RT-2、Octo、OpenVLA、π₀ 等）
- Diffusion Policy / Action Chunking
- 大规模预训练+微调范式（Open X-Embodiment 等）
- V-JEPA 2-AC、RoboCat、Gato 等
- 世界模型方法（UniSim、DreamerV3 用于机器人等）

每个模型：输入模态、输出模态、架构、开源否、规模。

### 2. 不同模态组合的表征

- 无本体（只有视觉和语言）vs 有本体（加关节角度/力矩）vs 有触觉：
  - 表征空间有什么结构性差异？
  - 加入本体感觉后表征如何变化？
  - 有没有人用 probing / 几何分析比较过不同模态组合的表征？
- 多模态融合在模型内部是怎么发生的？
  - 视觉和 proprioception 在哪一层对齐？
  - 触觉信息如何被整合？
  - 有没有类似 CLIP modality gap 的现象？

### 3. 跨具身（cross-embodiment）的表征

- 不同机器人硬件训练的模型，表征是否收敛？
- Open X-Embodiment 等多机器人数据集上的表征对齐分析
- 和 Platonic Representation Hypothesis 在 embodiment 维度上的关系

### 4. 表征分析对训练的辅助作用

- 有没有人用表征质量指标（ID、probe accuracy、CKA 等）辅助训练？比如：
  - 用表征对齐度做迁移学习的选择依据？
  - 用 probing 诊断训练中的模态不平衡/信息瓶颈？
  - 用表征几何做 early stopping / curriculum design？
  - 用表征分析指导 reward shaping 或 data augmentation？
- 表征蒸馏：大模型→小模型的表征对齐作为蒸馏目标（CKA loss / Procrustes loss）？
- 表征分析在 sim-to-real 迁移中的角色？

### 5. 已有的表征分析工作

- 有没有人对机器人模型做过 probing、因果干预、SAE、几何分析？
- OpenVLA 的 Molinari 2025 之外还有什么？
- 哪些模型被分析过、哪些是空白？

### 6. 独特机会

机器人模型为表征分析提供了什么纯语言/视觉模型没有的独特机会？
- 物理 ground truth 可直接测量
- 因果交互（动作→环境变化→反馈）
- 不同 embodiment 的天然对照
- 安全关键性（表征失败 → 物理损害）
