# Brainstorm 综合摘要：Top Experiment Ideas & Cross-Insights

> 5 个 brainstorm agents 的精华提炼。每个方向提取 3 个最 actionable 的实验，附跨方向洞察。

---

## P1: Encoding vs Use 方向对比

### 1. 五方向 Head-to-Head 对决（Pilot）
- **描述**：在同一模型 × 同一概念 × 同一评估协议下，比较 DiffMeans / Probe / Park Causal / DAS / SAE 五种方向提取方法的 detection / ablation / steering 表现。
- **价值**：从未有人在统一框架下做过这五种方法的 head-to-head，尤其 Park causal direction vs DAS 从未同台比较。这是"方向提取方法的 Rosetta Stone"。
- **模型/数据**：Pythia-1.4B，5 个二值概念（情感/数/时态/大小写/否定），3 层（早/中/晚）
- **可行性**：🟢 高（~12 GPU-hours，复用 exp-007/009 基础设施，1-2 天完成）

### 2. 方向间角度分析
- **描述**：对同一概念的 5 种方向，计算两两余弦相似度。核心问题：Park causal direction 和 DAS direction 是否对齐？
- **价值**：如果 Park ⊥ DAS → 两种"因果"定义不兼容（重大发现）。如果 Park ≈ DAS ≫ Probe → 因果修正是一致的。
- **模型/数据**：同上，附加计算量极小
- **可行性**：🟢 高（P1-Pilot 的副产品）

### 3. 方向因果性的训练动态
- **描述**：利用 Pythia 143 个 checkpoint，追踪 DiffMeans vs DAS 方向的 detection 和 ablation 随训练的变化。核心问题：encoding 先于 use 出现？
- **价值**：首次给出 "encoding 何时变成 use" 的 empirical 描述，直接检验 Braun 的 dissociation 是否有训练动态版本。
- **模型/数据**：Pythia-1.4B checkpoints，3 概念 × 2 方法
- **可行性**：🟡 中（计算量较大，但 Pythia checkpoints 公开可用）

---

## P2: "谁的信息"——来源分解

### 1. 三级 Baseline 分解（embedding → random → trained）
- **描述**：在 layer 0（embedding）和 5 个随机初始化网络上做 probe，将 probe accuracy 分解为 input bleed-through / 架构贡献 / 训练贡献三个来源。
- **价值**：首次系统化地将 probe accuracy 归因到三个来源。如果 bleed-through ratio > 50% → 大量 probing 文献的结论需要重新审视。
- **模型/数据**：Pythia-1.4B + 5 个随机种子初始化，15 个概念
- **可行性**：🟢 高（~12 GPU-hours，embedding baseline 和 random network 实现简单）

### 2. Standard × Encoding Probe 交叉分析
- **描述**：对每个概念 × 每层，同时计算 standard probe accuracy（d）和 Gubian encoding probe R²（e）。d 高但 e 低 = bleed-through 的直接证据。
- **价值**：比 acc(0) baseline 更精确地识别 bleed-through。Gubian encoding probe 在 LLM 上的首次应用。
- **模型/数据**：Pythia-1.4B，15 概念 × 32 层
- **可行性**：🟢 高（~16 GPU-hours，主要是 probe 训练）

### 3. Cross-Token Integration 概念谱系
- **描述**：设计从纯 token-level（大小写）到纯 sentence-level（蕴含/共指/反讽）的概念梯度，验证 bleed-through 与概念类型的关系。
- **价值**：给出"什么类型的概念最容易被 bleed-through 污染"的清晰分类。golden test cases（如 "wonderful" → "not wonderful" → 反讽句）特别有诊断力。
- **模型/数据**：Pythia-1.4B，16 个精选概念（4 级别 × 4）
- **可行性**：🟢 高（概念选择和标注是主要工作）

---

## P3: 信息生命周期

### 1. Lifecycle × Circuit（GPT-2 small IOI）
- **描述**：在 IOI task 上做全层 probe + ablation 双曲线，看 lifecycle 曲线能否自动重现已知的 IOI circuit 结构。
- **价值**：如果 lifecycle 曲线在 layer 7-8（S-inhibition heads）处 ablation 峰值与已知 circuit 完全匹配 → 桥接 probing 和 mechanistic interpretability 两个社区。proof of concept。
- **模型/数据**：GPT-2 small，IOI task（有完整的 ground truth circuit）
- **可行性**：🟢 高（~2 GPU-hours，模型小，ground truth 已有）

### 2. Lifecycle Taxonomy（8 种模式分类）
- **描述**：10 个概念 × 32 层的全扫描，将 dual-curve pattern 分类为 8 种模式（Ghost / Clean Lifecycle / Late Consumer / Distributed Worker / Re-encoder / Catalyst / Emergent / Paradox）。
- **价值**：建立信息 lifecycle 的分类学，每种模式有清晰的理论解读。exp-007 pilot 已显示信号（numeric 在 layer 6 Δloss=0.174 但晚期归零）。
- **模型/数据**：Pythia-1.4B，10 概念 × 32 层
- **可行性**：🟢 高（~24 GPU-hours，是 exp-007 的自然扩展）

### 3. AMNESIC Cascade（概念间因果流）
- **描述**：在层 ℓ ablate 概念 A 的方向，测下游其他概念 B 的 detection 变化。自动发现概念依赖图。
- **价值**：如果 ablate "number" 导致下游 "subject-verb agreement" detection 下降 → 自动发现概念间的因果依赖。全新的分析方法。
- **模型/数据**：GPT-2 small，10 概念
- **可行性**：🟡 中（~25 GPU-hours，实现需要精心设计 cascade protocol）

---

## P4: Probing "理解" 模型

### 1. DreamerV3 Categorical Anatomy
- **描述**：第一篇 DreamerV3 interpretability 研究。扫描 32 个 categorical variables × 32 classes，测每个 slot 与 (agent position / reward / object / velocity / environment) 的 mutual information。直接 flip categorical values 做因果干预。
- **价值**：DreamerV3 解释性研究**完全空白**。离散 categorical 的干预比连续表征更 clean。如果 slot 高度 specialized → 离散瓶颈自发 disentangle。
- **模型/数据**：DreamerV3 (Atari)，代码完全开源
- **可行性**：🟢 高（1-2 个月，模型小，实验 turnaround 快）

### 2. V-JEPA 2 守恒量 Probing
- **描述**：在合成物理视频（billiards/bouncing balls）上，对 V-JEPA 2 每层 patch representation 训练 linear probe 预测总动量/总能量。对比 random init / VideoMAE / temporal CLIP。
- **价值**：如果守恒量线性可 probe → self-supervised video prediction 能诱导抽象物理理解。如果需要 nonlinear probe → 模型"知道"物理量但没"理解"守恒定律。V-JEPA 2 是 FAIR flagship → 高 visibility。
- **模型/数据**：V-JEPA 2（权重开源），pymunk/pybullet 生成合成数据
- **可行性**：🟡 中（需要搭建合成视频 pipeline，2-3 个月）

### 3. Counterfactual Probing Protocol（L3 理解测试）
- **描述**：形式化三级测试框架：L1 Encoding → L2 Causal Use → L3 Understanding（改变概念值后模型预测是否**正确**地改变）。L3 需要 ground-truth physics/logic model 作为 oracle。
- **价值**：现有文献只到 L2（encoding ≠ use）。L3 是全新的方法论贡献。不只问"removing C does it hurt?"，而是 "changing C to C', does the prediction change *correctly*?"。
- **模型/数据**：V-JEPA 2 / DreamerV3 / OthelloGPT 做示范
- **可行性**：🟡 中（方法论新颖，需要精心设计 counterfactual + oracle，2-3 个月）

---

## Gap: Encoding = Use 的条件

### 1. Gradient Alignment Test（最 novel）
- **描述**：对每个概念 × 每层，计算 cos(probe_dir, ∇_h L)。如果此 cosine 和 ablation effect 强相关 → 梯度对齐是一个简单、免 ablation 的 faithfulness metric。
- **价值**：**没人做过这个**。如果 work → 任何 probe 可以用一次 forward+backward pass 就得到 "trustworthiness score"，直接改变 probing 实践。
- **模型/数据**：Pythia-1.4B，piggyback on exp-007
- **可行性**：🟢 高（计算成本极低，是 exp-007 的副产品）

### 2. Joint Training 方向比较
- **描述**：对每个概念 × 每层，训练 probe / DiffMeans / ReFT-r1 三种方向，分别做 steering 和 ablation。AxBench 已有证据：probe=0.098, ReFT-r1=0.543, DiffMeans=0.239。
- **价值**：验证"方向选择方法比模型本身更重要"这一核心假说。如果 ReFT-r1 >> probe → joint training 是关键条件。
- **模型/数据**：Pythia-1.4B，12 概念 × 32 层 × 3 方向
- **可行性**：🟢 高（Experiment A 的 ~3x 计算量）

### 3. Ghost Phase → Use Phase 训练动态
- **描述**：Pythia-1.4B × ~20 checkpoints（log-spaced），5 概念 × 32 层 × probe + ablation，画 ghost ratio vs training step。
- **价值**：核心假说检验——每个概念是否经历 ghost phase → use phase transition？如果 yes → encoding ≠ use 是**暂时的训练现象**而非**结构性问题**。
- **模型/数据**：Pythia-1.4B 20 个 checkpoint
- **可行性**：🟡 中（~480 GPU-hours，大但可行，可并行化）

---

## 🔗 跨 Brainstorm 洞察

### Insight 1: 训练动态是统一线索
P1（方向何时变因果）、P3（信息何时被消费）、Gap（ghost phase 假说）三个方向都指向同一个实验：**用 Pythia checkpoints 追踪 encoding-use gap 的训练演变**。这是一个"做一次回答三个问题"的实验。

### Insight 2: IOI Circuit 是理想的 Ground Truth Testbed
P3（lifecycle 重现 circuit）和 Gap（ground truth 验证条件）都推荐在 GPT-2 small 的 IOI circuit 上做 pilot。因为有完整的 known circuit，可以验证所有 predictions。**应优先用 IOI 做 proof of concept。**

### Insight 3: Transformer vs SSM 的架构对比是高价值
P2（bleed-through 在 SSM 中衰减更快？）、P3（SSM lifecycle 更"干净"？）、Gap（Mamba ghost ratio < Transformer？）三个方向都做出了 SSM 应该有更小 encoding-use gap 的预测。**一个 Pythia-160M vs Mamba-130M 的实验同时检验三个假说。**

### Insight 4: 概念类型决定一切
P1（低级特征 encoding ≈ use，高级语义 gap 大）、P2（token-level 概念 bleed-through 高）、Gap（surface features ghost 概率高）三个方向一致预测：**概念在抽象层级中的位置是 encoding-use gap 的主要决定因素**。exp-007 pilot 已有初步支持（capitalized 是 ghost，numeric 有 ablation effect）。

### Insight 5: 梯度对齐连接方向比较和可信度
P1（哪种方向最因果？→ Jacobian range 分析）和 Gap（cos(probe, gradient) 作为 faithfulness metric）本质上是同一个 idea：**因果方向 = 模型 loss landscape 中的敏感方向**。梯度对齐可能是统一五种方向提取方法的理论基础。

### Insight 6: 方向提取方法比模型更重要
P1（5 种方向给出不同因果效果）和 Gap（ReFT-r1 steering 5.5x > probe steering）都指向：**"如何提取方向"可能比"模型编码了什么"更关键**。这对整个 probing 文献的 methodology 有深远影响。

---

## 📊 优先级排序（综合 feasibility × impact × novelty）

| 排名 | 实验 | 来源 | 理由 |
|------|------|------|------|
| 1 | Gradient alignment test | Gap | 成本极低，novelty 极高，可能改变 probing practice |
| 2 | Lifecycle × IOI circuit | P3 | 成本低，ground truth 验证，桥接两个社区 |
| 3 | 五方向 head-to-head pilot | P1 | 核心创新点，1-2 天完成，直接成文 |
| 4 | 三级 baseline 分解 | P2 | 简单有效，如果 bleed-through > 50% 立刻有 story |
| 5 | DreamerV3 categorical anatomy | P4 | 完全空白领域，turnaround 快 |
| 6 | Transformer vs SSM ghost ratio | P2+P3+Gap | 一个实验回答三个问题 |
| 7 | Training dynamics (Pythia checkpoints) | P1+P3+Gap | 一个实验回答三个问题，但算力大 |
| 8 | Encoding × Standard probe 交叉 | P2 | Gubian 方法首次 LLM 应用 |
| 9 | Counterfactual probing (L3) | P4 | 方法论贡献大，但实现复杂 |
| 10 | V-JEPA 2 物理 probing | P4 | 高 visibility，但需建 pipeline |

---

## 🎯 推荐执行路径

**Phase 0（本地 Mac，1-2 天）**：
- OthelloGPT mobility probe（P4 quick win，验证 pipeline）
- Gradient alignment test（Gap #1，极低成本）

**Phase 1（jiagpu，1 周）**：
- Lifecycle × IOI circuit（P3 #1）
- 五方向 pilot（P1 #1）
- 三级 baseline 分解（P2 #1）

**Phase 2（jiagpu，2-3 周）**：
- 完整 exp-007 扩展 + lifecycle taxonomy
- Transformer vs Mamba 对比
- Joint training 方向比较

**Phase 3（根据信号强度选择）**：
- DreamerV3 anatomy 或 V-JEPA 2 physics probing
- Training dynamics with Pythia checkpoints
- 理论框架（SGD → encoding=use chain）
