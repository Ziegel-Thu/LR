# P3 Brainstorm: Information Lifecycle — 逐层追踪编码→使用→消费

> 2025-07 brainstorming，基于 exp-007 pilot 数据和 OPEN-PROBLEMS 分析

## 核心思想

对给定概念 C，画两条曲线：
- **Probe curve**: P(l) = probe accuracy at layer l（编码强度）
- **Ablation curve**: A(l) = Δloss from ablating concept direction at layer l（因果重要性）

这两条曲线的关系揭示信息的**生命周期**——不仅是"在哪里"（probing），也不仅是"有没有用"（ablation），而是"什么时候编码、什么时候使用、什么时候丢弃"。

**为什么这是新的？**
- Probing 文献只画 P(l)，不做 ablation
- Circuit 文献只做 ablation，不画 probe profile
- 没人**在同一概念上同时做两者**并系统比较 across layers
- exp-007 pilot 首次做了这个，但只有 5 层 × 5 个 surface feature，需要大幅扩展

---

## 一、回答关键设计问题

### 1. 什么概念值得追踪？

**选择标准**：(a) 有清晰 ground truth label (b) 对模型行为有可测量影响 (c) 覆盖不同抽象层次

| 层次 | 概念 | Ground Truth | 行为测量 | 预期 lifecycle |
|------|------|-------------|----------|---------------|
| **Surface** | 大小写 (capitalized) | token 字面 | 下一 token 大小写预测 | 早期编码，快速 ghost |
| **Surface** | 数字 (numeric) | token 包含数字 | 数值运算 | 早期编码，中间层使用 |
| **Morphological** | 名词单复数 (number) | 标注 / 启发式 | subject-verb agreement | 中间层编码+使用，晚期消费 |
| **Syntactic** | 词性 (POS) | SpaCy 标注 | 句法合法性 | 早→中层编码，可能 ghost |
| **Syntactic** | 句法深度 (tree depth) | parser | 嵌套结构理解 | 中间层渐进编码 |
| **Semantic** | 情感 (sentiment) | 标注器 | 生成延续的情感一致性 | 中→晚层编码 |
| **Relational** | IOI: S-identity | 构造数据 | IO vs S 输出概率 | 已知 circuit 可预测 |
| **Relational** | IOI: IO-identity | 构造数据 | 正确名字概率 | 已知 circuit 可预测 |
| **Relational** | IOI: position signal | 构造数据 | 位置依赖的输出 | 头几层编码 |
| **World knowledge** | 实体类型 (entity type) | NER 标注 | 事实回忆 | 中→晚层，可能分布式 |
| **World knowledge** | 首都-国家关系 | 构造 prompt | 事实完成 | 晚层使用 |
| **Task-specific** | Greater-than | 构造 prompt | 数值比较 | 已知 circuit |

**推荐优先级**：
1. **IOI 三概念** (S-identity, IO-identity, position) — 有 Wang 2022 circuit 作对照，可验证
2. **Number agreement** — 经典 probing 任务，行为影响清晰
3. **POS + sentiment** — 对比低层 vs 高层概念的生命周期差异
4. **Greater-than** — 第二个有 circuit 的任务，验证 lifecycle-circuit 联系的 robustness

### 2. 什么模型？

| 模型 | 优势 | 劣势 | 用途 |
|------|------|------|------|
| **GPT-2 small (117M)** | IOI circuit 已知 (Wang 2022), 12 层, 快速迭代 | 只有 Transformer, 太小可能没有复杂 lifecycle | 核心实验 1: lifecycle ↔ circuit 验证 |
| **Pythia-1.4B** | 32 层, 更精细的 lifecycle, 有 checkpoints | 无已知 circuit | 核心实验 2: lifecycle taxonomy + 训练动态 |
| **Pythia-160M** | 和 Mamba-130M 规模匹配, 12 层 | 太小可能信号弱 | 跨架构对比的 Transformer baseline |
| **Mamba-130M** | SSM 架构, 24 层, 无 attention | 无 clean unembedding, ablation 需适配 | 跨架构对比 |
| **Pythia-70M→6.9B** | ladder 可拟合 scaling law | 计算量大 | 如果核心结果有趣才做 |

**推荐策略**：GPT-2 small (验证性) → Pythia-1.4B (主实验) → Pythia-160M vs Mamba-130M (跨架构)

### 3. 怎么 ablate？

| 方法 | 原理 | 优势 | 劣势 | 适用场景 |
|------|------|------|------|---------|
| **AMNESIC (project out probe direction)** | 投影移除 probe 学到的方向 | 概念特异性：只移除目标信息 | probe 方向可能不准 | **默认选择** |
| **DiffMeans ablation** | 投影移除 μ₊ - μ₋ 方向 | 无需训练 probe | 假设线性可分 | 验证性对比 |
| **DAS direction** | 投影移除 DAS 学到的方向 | 更因果 (Park 2024) | 计算量大 | 验证性对比 |
| **Mean ablation** | 替换为数据集均值 | 最常见 | 非概念特异性，over-estimate effect | 仅作 baseline |
| **Resample ablation** | 替换为同概念不同实例的 activation | 最干净的因果推断 | 需要 paired data | IOI 等构造任务 |
| **Zero ablation** | 设为零 | 最简单 | 严重 OOD（LayerNorm 后尤甚）| **不推荐** |

**推荐**：
- **主方法**: AMNESIC（概念特异性，和 probe 自然配对）
- **验证**: DiffMeans（检查 probe 方向 vs mean difference 方向一致性）
- **对照**: Mean ablation（检查概念特异性 vs 全信息移除的差异）
- 对 IOI: 加 resample ablation（有 paired 构造数据）

**关键细节**：AMNESIC 投影出的是**一维子空间**，这意味着我们假设概念用一个方向编码（LRH）。可以扩展到 k 维（前 k 个奇异向量），但一维是起步。

### 4. 双曲线的所有可能 pattern 及解读

以下用 (P, A) 描述 probe 和 ablation 曲线在给定层的状态（H=高, L=低, ↑=上升, ↓=下降）:

#### Pattern 1: **Ghost（幽灵信息）**
```
P: ————————————————  (持续高)
A: __——______________  (早期小 peak 后消失)
```
- **解读**: 信息从 embedding 就编码了，模型短暂使用后不再需要，但信息因残差流持续存在
- **类比**: 写在黑板角落的笔记——一直在那但没人看
- **exp-007 pilot**: `capitalized` 就是这个 pattern
- **理论意义**: 最直接支持 Braun 2025 的 encoding ≠ use

#### Pattern 2: **Clean Lifecycle（干净生命周期）**
```
P: ___/‾‾‾‾‾\___  (先升后降)
A: _____/‾‾\____  (peak 略晚于 P 的 peak)
```
- **解读**: 概念在某些层被主动计算出来（P 上升），被下游计算使用（A peak），然后被新计算覆盖（P 下降）
- **类比**: 中间产物——被合成、使用、降解
- **预期出现**: 句法特征（POS 在早期层计算，用于理解句法后被语义覆盖）
- **理论意义**: 信息被 "consumed"，支持 processing-as-computation 叙事

#### Pattern 3: **Late Consumer（晚期使用）**
```
P: ——/‾‾‾‾‾‾‾‾‾‾  (早期编码，持续到最后)
A: ___________/‾‾  (只在最后几层才有 effect)
```
- **解读**: 信息被保留在残差流中，直到接近 output 时才用于预测
- **类比**: 行李——一路带着但到目的地才打开
- **预期出现**: 世界知识（entity type 一直编码，最后才用于预测 next token）

#### Pattern 4: **Distributed Worker（分布式使用）**
```
P: ——/‾‾‾‾‾‾‾‾‾‾  (持续编码)
A: __/‾‾‾‾‾‾‾‾‾\  (多层都有中等 effect)
```
- **解读**: 信息在多个层持续被使用
- **类比**: 电力——持续供应，持续消耗
- **预期出现**: number agreement（多层 attention 都需要检查主谓一致）

#### Pattern 5: **Re-encoder（重新编码）**
```
P: ——/‾\__/‾‾‾‾  (有 dip 后恢复)
A: ___/‾\__/‾\__  (多个 peaks)
```
- **解读**: 信息被变换、部分丢失、用新表征重新编码
- **类比**: 翻译——原文用完，译文重新产生
- **预期出现**: 语义特征（从词汇语义→句子级语义的重新编码）
- **理论意义**: 直接支持 iterative refinement 叙事（tuned lens 的直觉）

#### Pattern 6: **Catalyst（催化剂）**
```
P: ——/‾‾‾‾‾‾‾‾‾‾  (一直高)
A: ____/‾‾‾\____  (中间 peak，两端低)
```
- **解读**: 信息被使用但不被消耗——用完后仍然完整保留
- **类比**: 催化剂——参与反应但自身不变
- **预期出现**: 位置信息（每层都可以读取，特定层集中使用）
- **注意**: 和 Pattern 1 (Ghost) 的区别在于 A 确实有 non-trivial peak

#### Pattern 7: **Emergent（涌现）**
```
P: ____________/‾‾  (只在晚期出现)
A: _____________/‾  (紧跟 P 出现)
```
- **解读**: 概念是晚期层计算的结果，一旦产生立即使用
- **类比**: 结论——推理后才出现，出现后立即行动
- **预期出现**: 高阶语义（如 "讽刺"——需要句法+语义综合后才能识别）

#### Pattern 8: **Paradox（悖论）**
```
P: ___________\__  (probe 下降)
A: ___________/‾‾  (ablation 上升)
```
- **解读**: probe 读不到但 ablation 有效——信息可能以 **非线性** 方式编码
- **理论意义**: 直接挑战 LRH；或者暗示 probe 方向不准而信息以其他形式存在
- **如果观察到**: 需要用非线性 probe 验证

### 5. 与已知 circuit 的联系

**IOI Circuit (Wang 2022, GPT-2 small)**:

| Circuit Component | Layers | 功能 | 预测: P(l) | 预测: A(l) |
|-------------------|--------|------|-----------|-----------|
| Duplicate token heads | 0-1 | 检测 token 是否重复出现 | position signal P 高 | position signal A 中等 |
| Previous token heads | 0-2 | 传递前一个 token 信息 | — | — |
| S-Inhibition heads | 7-8 | 抑制 S(ubject) 名字 | S-identity P 高, IO-identity P 高 | S-identity A **peak** |
| Name Mover heads | 9-11 | 将正确名字 copy 到 output | IO-identity P 保持 | IO-identity A **peak** |
| Backup Name Movers | 9-11 | 冗余路径 | IO-identity P 保持 | 可能增加 A 的宽度 |

**具体预测**：
1. **S-identity probe** 应在 layer 0-3 快速升高（embedding 已包含），但 **ablation peak** 应在 layer 7-8（S-inhibition 使用它）
2. **IO-identity probe** 应持续高，但 **ablation peak** 应在 layer 9-11（name mover 使用它）
3. **Position signal ablation** 应在 layer 0-2 peak（duplicate token heads 需要它）

**如果预测成立**：lifecycle 曲线可以**从 probe 数据自动发现 circuit 的层级结构**，无需手动分析
**如果不成立**：说明 circuit 只捕获了部分计算，lifecycle 揭示了 circuit 遗漏的信息流

### 6. 跨架构对比

**Transformer vs Mamba 的 lifecycle 差异预测**：

| 特性 | Transformer | Mamba | 影响 |
|------|-------------|-------|------|
| 残差流 | 加法残差 → 信息不主动消失 | 递推状态 → 信息被 overwrite | Mamba 的 P(l) 下降更快 |
| 信息读取 | Attention 可以 look back | 只有 state → 无 look-back | Mamba 必须更早编码 |
| 层间通信 | MLP + attention | SSM + MLP | 不同的编码通道 |

**具体预测**：
- **Ghost ratio**: Transformer > Mamba（残差流保留更多无用信息）
- **Lifecycle 宽度**: Transformer 更宽（信息活得更久）
- **Ablation peak 位置**: Mamba 更集中（不能 re-read，必须用了就用）
- **消费模式**: Transformer 催化剂更多，Mamba clean lifecycle 更多

**技术挑战**：
- Mamba 没有 clean 的 "layer output" hook 位置——需要 hook SSM state 或 block output
- Mamba 的 "probe direction" 可能更难找（selective scan 使表征更 entangled）
- 对 Mamba 做 AMNESIC 是否合理？投影出方向后 SSM 状态转移可能更不稳定

### 7. 训练动态

**Pythia checkpoints (154 个, step 0 → 143000)**：

追踪概念 C 在不同训练阶段的 lifecycle 变化：

```
训练早期:   P 低, A 低   — 概念尚未被学会
训练中期:   P 高, A 低   — 编码了但还没学会使用（Ghost 阶段）
训练晚期:   P 高, A 高   — 编码且使用（Mature 阶段）
```

**核心假说**: **每个概念都经历 "ghost phase"** —— 先学会编码，后学会使用。

**可测试推论**：
1. Ghost phase 的长度和概念复杂度正相关（surface feature 的 ghost phase 短，语义 feature 长）
2. Ghost phase 的结束和 loss 下降的 phase transition 对应（类似 grokking）
3. 过拟合时：ghost ratio 增加（模型编码更多不用的信息）

**实验设计**：
- 选 5 个概念 × Pythia-1.4B 的 ~20 个 checkpoint（log-spaced）
- 每个 checkpoint 做全 32 层 probe + ablation
- 画 "lifecycle evolution" 动画/热力图

**和 grokking 的联系**：
- Grokking: test acc 突然跳升 → 可能对应某个概念从 ghost→used 的 transition
- 如果能在 lifecycle 曲线上看到 grokking → 这就是 grokking 的 mechanistic 解释

### 8. 多概念同时追踪

**核心问题**：概念的 lifecycle 是独立的还是耦合的？

**设计**：同时追踪 5 个概念 {C₁, ..., C₅}，检查：

1. **时序耦合**: C₁ 的 ablation peak 和 C₂ 的 probe 变化是否同步？
   - 如果 ablate C₁ at layer L → C₂ probe drops at layer L+1 → C₂ 依赖 C₁
   - 这就是**信息依赖图**的自动发现

2. **空间耦合**: 不同概念的 probe direction 在 layer L 是否正交？
   - 正交 → 独立编码
   - 共线 → 纠缠（superposition?）
   - cosine similarity 随层变化 → 纠缠/解纠缠的动态

3. **Resource competition**: 当同时 ablate C₁ 和 C₂ 时，效果是否 > 单独 ablate 之和？
   - 超加性 → 冗余（一个概念被 ablate 后另一个 compensate）
   - 亚加性 → synergy（两个概念共享通道）

**推荐概念组合** (IOI task):
- {S-identity, IO-identity, position, number, sentiment}
- 前三个在 IOI circuit 中有已知交互，后两个是 "旁观者"

### 9. 潜在 confounds

| Confound | 描述 | 影响 | 缓解 |
|----------|------|------|------|
| **残差流粘滞** | Transformer 的加法残差流使信息永不消失 | P(l) 只升不降，lifecycle 被遮蔽 | 用 normalized probe（减去前一层方向投影）|
| **LayerNorm 补偿** | LN 在 ablation 后重新归一化，部分恢复信息 | A(l) 被低估 | 在 LN 之后做 ablation（而非之前）|
| **Probe 方向漂移** | Probe 在每层学到不同方向 → 投影出的不是同一概念 | lifecycle 实际上追踪的是"方向"而非"概念" | 计算相邻层 probe 方向 cosine；用 concept direction 一致性 (exp-009) 的结果来校准 |
| **Ablation OOD** | 投影后的 activation 偏离 on-distribution manifold | A(l) 可能反映 OOD 效应而非信息缺失 | 计算 ablated activation 的 Mahalanobis distance，过大的 A 不可信 |
| **Label 噪声** | Ground truth 标签不完美 → P(l) 有 ceiling | P(l) 和 A(l) 的绝对值不可比 | 用 rank 而非绝对值；或 calibrate P(l) by control task |
| **多维编码** | 概念用多个方向编码 → 投影一个方向不够 | A(l) 被低估 | 用 top-k 奇异方向做 AMNESIC；看 k 的影响 |
| **间接因果链** | A(l) 反映的可能是 l+1 到 l+n 的累积效果而非 l 本身 | A(l) 归因不够 precise | 比较 "layer-specific ablation" vs "layer-and-all-downstream ablation" |

**最重要的 confound**: **残差流粘滞**。对 Transformer 来说，一旦信息写入残差流，除非被显式覆盖（MLP 的非线性叠加可能部分覆盖），它就一直在。这意味着 P(l) 天然单调递增或平台，很少下降。

**解决方案**: 定义 **incremental probe accuracy** ΔP(l) = P(l) - P(l-1)，而非 P(l) 本身。ΔP(l) > 0 意味着 layer l 主动增加了信息；ΔP(l) ≈ 0 意味着信息被动保留；ΔP(l) < 0 意味着信息被覆盖。

### 10. 什么让这能发表？

#### Minimum Viable Result (投稿 Workshop)
- **设置**: GPT-2 small, 3-5 个概念, 12 层, AMNESIC ablation
- **展示**: 不同概念确有不同 lifecycle pattern（至少 2 种 qualitatively 不同的 pattern）
- **亮点**: 首次系统化 probe + ablation 双曲线
- **叙事**: "Beyond probing: tracking the information lifecycle in Transformers"

#### Strong Result (投稿 ICML/NeurIPS)
1. **Lifecycle taxonomy**: 在多个概念 × 多个模型上建立 ≥4 种 pattern 分类
2. **Circuit prediction**: lifecycle 曲线在 IOI 上准确预测已知 circuit 结构
3. **Ghost ratio 统计**: 系统量化不同类型概念的 ghost ratio（和 Braun 2025 对话）
4. **Cross-architecture**: Transformer vs Mamba 有定性差异（至少一个 prediction confirmed）

#### Home Run (投稿 ICML/NeurIPS oral)
以上全部 + 以下任一：
- **Training dynamics**: ghost phase 是 universal 的，长度可预测
- **Lifecycle-based circuit discovery**: 从 lifecycle 曲线自动发现 circuit，和 Wang 2022 手工发现的一致
- **信息预算守恒**: sum of A(l) across layers ≈ total concept importance，存在"信息守恒定律"

### 11. Novel Twists（超越基础双曲线）

#### Twist 1: Lifecycle Velocity（信息处理速度）
定义 v_P(l) = dP/dl, v_A(l) = dA/dl（离散导数）
- v_P > 0: 信息正在被创建
- v_P < 0: 信息正在被覆盖/遗忘
- v_A > 0: 信息正在被动员使用
- v_A < 0: 使用完成，信息被释放

画 (v_P, v_A) 的 phase portrait → 每个概念在 layer 空间的 trajectory 是一个**信息状态机**

#### Twist 2: AMNESIC Cascade（级联消融）
- 在 layer L 做 AMNESIC，然后 re-probe at layer L+1, L+2, ...
- 如果 P(L+1) 恢复 → 信息有 **冗余编码**（另一条路径重新计算了）
- 如果 P(L+1) 仍低 → 信息真的被删除了
- 这量化了 **redundancy vs necessity** — circuit 的 "backup" 概念的实证版

#### Twist 3: Cross-concept Information Flow
- Ablate concept C₁ at layer L → measure Δprobe for C₂ at layers L+1 到 L_max
- 构建 **concept-layer 因果图**: C₁@L → C₂@L' 表示概念间的计算依赖
- 这是一种 **bottom-up circuit discovery**，不需要任何先验知识

#### Twist 4: Lifecycle Entropy
- 将 A(l) 归一化为概率分布: p(l) = A(l) / Σ A(l)
- 计算 H = -Σ p(l) log p(l)
- H 低 → 信息使用集中在少数层（localized processing）
- H 高 → 信息使用分布在多层（distributed processing）
- 可以用 H 作为**概念复杂度**的度量，和认知科学中的信息加工深度比较

#### Twist 5: Information Budget
- 对每个概念 C: Total_Use(C) = Σ_l A(l), Total_Encoding(C) = max_l P(l)
- 画 Total_Use vs Total_Encoding 散点图
- 斜率 = 信息利用效率 η
- 如果 η 在训练过程中增加 → 模型在学会"少编码、多使用"

#### Twist 6: Lifecycle Similarity as Concept Similarity
- 两个概念 C₁, C₂ 的 lifecycle 向量: [P(1), A(1), P(2), A(2), ...]
- Lifecycle 相似性 = 这些向量的 cosine
- **假说**: lifecycle 相似的概念 functionally 相关
- 从 lifecycle 向量做聚类 → 自动发现概念的功能分组
- 和 LRH (exp-009) 的概念方向相似性对比 → 几何相似 vs 功能相似

#### Twist 7: Predicting Ablation from Probing
- 训练一个 meta-model: 从 P(l) profile 预测 A(l) profile
- 如果 P→A 可预测 → encoding 和 use 有 lawful relationship → probing 范式 partially 合法
- 如果不可预测 → encoding 和 use 真的解耦 → Braun 2025 的 non-linear 版本成立
- **最有理论价值的 twist** — 直接回答 "probing 能不能推断因果"

---

## 二、具体实验设计

### Experiment A: Lifecycle × Circuit 验证（GPT-2 small + IOI）

**目的**：验证 lifecycle 双曲线能否自动 "rediscover" 已知 circuit 结构

**模型**: GPT-2 small (12 layers, 117M)

**Task**: IOI (Indirect Object Identification)
- 输入: "When Mary and John went to the store, John gave a drink to"
- 正确输出: "Mary" (IO)

**追踪概念**:
1. S-identity: "当前名字 = subject"
2. IO-identity: "当前名字 = indirect object"
3. Duplicate detection: "当前 token 之前出现过"
4. Position: "当前 token 在哪个槽位 (S1/IO/S2/end)"

**方法**:
- 构造 100 个 IOI prompt（用 Wang 2022 的 template）
- 每层: 训练 linear probe → 记录 P(l)
- 每层: AMNESIC ablation → 记录 A(l) = Δ(logit_IO - logit_S)

注意：这里 A(l) 应该测 **logit difference** 而不是 loss，因为 IOI 的 behavioral metric 是 logit(IO) - logit(S)

**对照**:
- 和 Wang 2022 Table 1 的 circuit 组件位置比较
- 用 Conmy 2023 (ACDC) 的自动 circuit discovery 结果比较

**Ablation 方法**: AMNESIC + resample (IOI 有天然 paired data: 交换 S 和 IO)

**成功标准**:
- A(l) peak 出现在 circuit 预测的位置 (±1 layer)
- lifecycle pattern 可以区分 circuit 内 vs 外的层

**算力**: ~2 GPU-hours (GPT-2 small 很快), 可在 jiagpu 任一节点

**依赖**: TransformerLens (已在 jiagpu4 环境)

---

### Experiment B: Lifecycle Taxonomy（Pythia-1.4B, 全 32 层）

**目的**：在多种概念上建立 lifecycle 的系统分类

**模型**: Pythia-1.4B (32 layers)

**概念** (10 个，覆盖 5 个抽象层次):

| # | 概念 | 类型 | Ground Truth | 行为指标 |
|---|------|------|-------------|----------|
| 1 | Capitalized | surface | token 字面 | 大写字母预测 |
| 2 | Numeric | surface | token 包含数字 | 数值 token 预测 |
| 3 | POS: noun/verb | morphological | SpaCy | 词性 bigram 合法性 |
| 4 | Number: sg/pl | morphological | SpaCy | agreement 预测 |
| 5 | Tree depth | syntactic | SpaCy parser | 嵌套理解 |
| 6 | Named entity | semantic | SpaCy NER | 实体相关预测 |
| 7 | Sentiment | semantic | 标注器 | 情感一致延续 |
| 8 | Formal/informal | pragmatic | 简单规则 | 语域一致延续 |
| 9 | Content word | lexical | 频率+POS | 信息密度 |
| 10 | Negation scope | logical | 启发式规则 | 逻辑关系 |

**方法**:
- 数据: WikiText-103 validation, 2000 sentences（和 exp-007 plan 一致）
- 每个概念 × 每层: 训练 linear probe → P(l)
- 每个概念 × 每层: AMNESIC ablation → A(l) = Δloss
- 附加: DiffMeans ablation 作为对照

**分析**:
1. 画 10 × 2 条曲线（probe + ablation for each concept）
2. 对每个概念分类到 §4 的 pattern taxonomy
3. 计算 ghost ratio（和 exp-007 直接对比）
4. 计算 lifecycle entropy（Twist 4）
5. 计算 cross-concept lifecycle similarity（Twist 6）
6. 尝试 P→A prediction（Twist 7）

**成功标准**:
- 至少 3 种不同 lifecycle pattern 在 10 个概念中出现
- surface concepts (1-2) 和 semantic concepts (6-7) 的 lifecycle qualitatively 不同
- ghost ratio 按概念类型有系统差异

**算力**: 10 concepts × 32 layers × ~3min = ~16 GPU-hours（probe）+ ~8 GPU-hours（ablation）= ~24 GPU-hours

**注意**: 这实质是 **exp-007 的全量版 + lifecycle 分析框架**。可以复用 exp-007 pilot 的代码，扩展到全层 + 更多概念。

---

### Experiment C: 跨架构 Lifecycle 对比（Pythia-160M vs Mamba-130M）

**目的**：Transformer 和 SSM 是否有不同的信息处理策略？

**模型**: Pythia-160M (12 layers) vs Mamba-130M (24 layers)
- 参数量匹配，层数不同 → 用 normalized depth (0-1) 做横轴

**概念**: 从 Exp B 中选 5 个信号最强的概念

**技术挑战**:

Mamba 的 ablation 需要适配：
```python
# Transformer: hook on residual stream after layer
# Mamba: hook on block output (类似 residual stream)
# Mamba 的 block = SSM + MLP，可以 hook block output
```

Mamba 的 probe 也需要确认：
- Mamba 的 hidden state 和 Transformer 的残差流不同
- 需要分别 probe **block output**（残差流等价）和 **SSM internal state**

**分析**:
1. 对每个概念画 lifecycle（normalized depth on x-axis）
2. 比较 ghost ratio: Transformer vs Mamba
3. 比较 lifecycle entropy: 哪个更集中？
4. 比较 ablation peak 位置: Mamba 是否更早使用信息？

**预测**（可证伪）:
- Mamba ghost ratio < Transformer（残差流粘滞 → ghost）
- Mamba lifecycle entropy < Transformer（使用更集中）
- Mamba probe curve 在 normalized depth > 0.5 后下降更快（state overwrite）

**算力**: ~12 GPU-hours per model × 2 = ~24 GPU-hours

---

### Experiment D: Lifecycle 训练动态（Pythia-1.4B checkpoints）

**目的**：概念的 lifecycle 如何在训练过程中演化？是否存在 universal 的 ghost phase？

**模型**: Pythia-1.4B, 20 checkpoints (log-spaced from step 0 to 143000)
- steps: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000

**概念**: 从 Exp B 中选 5 个最有信号的概念

**方法**:
- 每个 checkpoint × 每个概念 × 每层: probe + ablation
- 产出: 一个 (layer × checkpoint) 热力图 for P 和 A，共 10 张

**分析**:
1. **Ghost phase detection**: 对每个概念，找最早的 checkpoint 使 P > 0.8 (encoding threshold) 和最早的 checkpoint 使 A > threshold (use threshold)。差值 = ghost phase 长度
2. **Phase transition**: 从 ghost→used 的转换是 gradual 还是 sudden？如果 sudden → grokking-like dynamics
3. **概念出现顺序**: 哪些概念先被编码？先被使用？
4. **早期训练的 lifecycle 形状**: 是否所有概念一开始都是 ghost，然后逐渐变成 used？

**核心假说**:
> **Ghost Phase Universality**: 每个概念在训练过程中都经历 encoding-before-use 阶段，阶段长度与概念抽象程度正相关

**成功标准**:
- ≥3/5 概念有可检测的 ghost phase
- Ghost phase 长度按概念类型排序: surface < morphological < semantic

**算力**: 5 concepts × 32 layers × 20 checkpoints × ~1.5min = ~40 GPU-hours
- 可并行：不同 checkpoint 是独立的，可用 8 GPU 分摊

---

### Experiment E: Cross-concept Causal Flow（概念间因果流）

**目的**：概念的 lifecycle 是独立的还是耦合的？能否自动发现概念间的计算依赖？

**前置**: Experiment A 或 B 完成后

**方法** (AMNESIC Cascade, Twist 2 + Twist 3):
1. 选 5 个概念 {C₁, ..., C₅}
2. 对每个 C_i, 每个 layer L:
   - Ablate C_i at layer L (AMNESIC)
   - Re-probe **all other concepts** at layers L+1, L+2, ..., L_max
   - 记录 ΔP_j(l) = P_j(l|ablate C_i@L) - P_j(l|no ablation) for j ≠ i
3. 构建因果图: edge C_i@L → C_j@L' if ΔP_j(L') is significant

**产出**: 一个 (concept × layer) → (concept × layer) 的因果矩阵
- 可视化为 "information flow diagram"
- 和 IOI circuit diagram 对比（如果在 GPT-2 small 上做）

**成功标准**:
- 因果图中的边 non-trivially sparse（不是全连接）
- 在 IOI 上：因果图的 structure 和已知 circuit 对应

**算力**: 5 concepts × 12 layers × 5 re-probes = 300 probe evaluations → ~10 GPU-hours
- 但需要 5 × 12 次 ablation forward pass → ~15 GPU-hours

**风险**: Cascade ablation 可能引入太多噪声，信号弱

---

## 三、实验优先级和执行计划

### Phase 1: Proof of Concept（2 周）
**Exp A (Lifecycle × Circuit)** on GPT-2 small
- 最少工作量验证核心想法
- 有 circuit ground truth → 可立即判断是否有价值
- 代码基于 exp-007 pilot + TransformerLens

### Phase 2: Systematic Study（3-4 周）
**Exp B (Lifecycle Taxonomy)** on Pythia-1.4B
- exp-007 的全量版，加 lifecycle 分析框架
- 建立 pattern taxonomy
- 代码基于 exp-007 pilot 扩展

### Phase 3: Extensions（各 2 周，可并行）
- **Exp C (Cross-architecture)**: 如果 Phase 1-2 结果有趣
- **Exp D (Training dynamics)**: 如果 Phase 2 发现 ghost pattern
- **Exp E (Cross-concept flow)**: 如果 Phase 1 的 IOI 结果和 circuit 对应

### 和现有实验的关系

| 现有实验 | 和 P3 的关系 |
|---------|-------------|
| exp-007 | **直接前身**——P3 是 exp-007 的 lifecycle 扩展 |
| exp-009 (LRH systematic) | 概念方向一致性数据可用于 validate probe direction stability (confound 3) |
| exp-003 (cross-arch) | 跨架构基础设施可复用 |
| exp-008 (SAE on SSM) | Mamba 模型加载/hook 代码可复用 |
| PLAN.md 任务 10 (因果抽象审计) | OOD intervention 量化和 P3 的 confound 4 共享方法论 |

---

## 四、最有杀伤力的 Narrative

如果所有实验都 work out，论文叙事:

> **Title**: "The Information Lifecycle: Tracking Encoding, Use, and Consumption in Neural Networks"
>
> **One-sentence pitch**: 我们通过在每一层同时测量 probe accuracy 和 ablation effect，首次追踪了神经网络中概念信息的完整生命周期——从编码到使用到消费——发现了 X 种不同的 lifecycle pattern，其中 Y% 的可解码信息是 ghost information（编码但不使用），并证明 lifecycle 曲线可以自动 rediscover 已知 circuit 结构。
>
> **Why this matters**:
> 1. Probing tells you "where" but not "how" — lifecycle fills the gap
> 2. Circuits tell you "how" but require manual analysis — lifecycle automates it
> 3. Ghost information quantifies the reliability of the entire probing paradigm
> 4. Cross-architecture comparison reveals fundamentally different processing strategies

**最强的 figure**: 一张 12-panel 图，每 panel 是一个概念的 lifecycle (P+A 双曲线)，标注 pattern type，底部叠加 IOI circuit 的已知组件位置。读者一眼看到 lifecycle 和 circuit 的对应。

---

## 五、风险清单

| 风险 | 影响 | 概率 | 缓解 |
|------|------|------|------|
| Probe accuracy 太高（ceiling），看不出差异 | lifecycle 全是 flat ghost | 中 | 用更难的概念（语义/逻辑），或用 mutual information 代替 accuracy |
| AMNESIC ablation effect 太小（noise floor） | A(l) 全是噪声 | 中 | 增大样本量；用 mean ablation 确认确有信号 |
| 所有概念 lifecycle 都一样 | 没有 taxonomy | 低 | pilot 已看到差异 |
| Lifecycle 和 circuit 不对应 | 失去最强叙事 | 中 | 仍然有独立价值（lifecycle taxonomy + ghost ratio） |
| Mamba ablation 实现困难 | Exp C 延迟 | 中 | 先搞定 Transformer 实验 |
| 计算量太大 | 跑不完 | 低 | 优先级清晰；可选择性跳过 Exp D/E |
