# DR-010: 表征分析方法的互补性理论——从不可能性到最优组合

## 背景

### 不可能性的现状

我们已经整理了四个独立的不可能性/no-free-lunch 结果（Bilodeau PNAS'24, Han NeurIPS'22, Sutter NeurIPS'25, Kleinberg NeurIPS'02），并在自己的 formalization 中证明了 "additivity（而非 completeness）是 attribution 不可能性的根源"。

### 关键实验发现（just obtained）

我们刚跑完一个实验（exp-005），测试 Φ^α = (1-α)·IG + α·SHAP（α 从 0 到 1）对不同局部行为的可辨识性。

**理论预测**：α=0 (IG, completeness only) 时 IGA>1（可区分）；α=1 (SHAP, completeness + additivity) 时 IGA=1（不可区分，等于随机猜测）。

**实际结果**：**所有 α 的 IGA 都接近 2.0，几乎没有下降。** 即使纯 SHAP (α=1) 也能完美区分不同局部行为。

**解释**：Bilodeau 的不可能性是 **worst-case** 结果——需要精心构造的 adversarial 模型（远场行为专门对消局部信号）。在自然训练的网络上，这种对消不会发生。

### 核心问题转变

这个实验发现改变了问题的性质：从 "能不能" 变成 "在什么分布下能/不能"。具体地：

1. **Worst-case vs average-case gap**：Bilodeau 的不可能性在 worst-case 下成立，但 average-case（自然训练的网络）下不成立。这个 gap 有多大？什么条件下 worst-case 是 tight 的（即自然就会遇到 adversarial 模型）？

2. **方法互补性**：如果单一方法在 worst-case 下不行，但多个方法的组合能否缩小 worst-case？比如，IG 和 SHAP 各自有 adversarial failure modes，但它们的 failure sets 是否有交集？如果没有，联合使用就能克服不可能性。

3. **Arrow 定理之后的 mechanism design 类比**：Arrow 的不可能性定理之后，mechanism design 通过放松公理（如 Gibbard-Satterthwaite 只要求 strategy-proofness）或改变目标（如 VCG 追求效率而非所有 Arrow 公理）实现了正面结果。在表征分析中，类似的正面结果是什么？

## 需要调研的问题

### 1. Worst-case vs average-case impossibility

在统计学习理论和其他领域中，有哪些 "worst-case impossible but average-case tractable" 的著名例子？特别是：
- **压缩感知 (Compressed Sensing)**：一般的稀疏恢复是 NP-hard (worst-case)，但在 RIP 条件下（random matrices 满足）变成多项式时间。这个结构与我们的问题是否类似？
- **PAC learning vs agnostic learning**：agnostic learning 中的 impossibility 是否有 average-case relaxation？
- **Minimax vs Bayes risk**：当 prior 集中在 "non-adversarial" 模型上时，Bayes risk 可以远小于 minimax risk。是否有精确的量化？
- 是否有已知的工作将 Bilodeau 或类似的 attribution impossibility 做了 average-case 分析？

### 2. 方法互补性的数学理论

有没有关于"多个弱方法联合变强"的严格数学理论？候选：
- **Boosting 理论**（AdaBoost 等）：多个 weak learners 组合成 strong learner。但这是在同一个任务上——我们需要的是不同任务/公理的互补。
- **Multi-view learning / Co-training**：多个视角提供互补信息。Blum & Mitchell 1998 的 co-training 理论是否有启示？
- **Ensemble diversity theory**：Krogh & Vedelsby 1995 的 ambiguity decomposition，Hansen & Salamon 1990 的 ensemble diversity。这些能否推广到 attribution/decomposition 的互补性？
- **Information-theoretic bounds on combination**：如果方法 A 提供 $I_A$ bits 关于行为 $B$ 的信息，方法 B 提供 $I_B$ bits，联合最多提供 $I_A + I_B - I_{AB}$ bits（其中 $I_{AB}$ 是冗余）。是否有已知的工作量化不同 interpretability 方法之间的信息冗余？

### 3. Arrow 定理之后的正面结果

Arrow 的不可能性定理（1951）之后，social choice theory 产生了大量正面结果：
- **Gibbard-Satterthwaite**：放松一个公理后得到可能的机制
- **VCG mechanism**：在准线性偏好下实现效率
- **Sen's impossibility of a Paretian liberal**：通过限制偏好域恢复可能性
- **Single-peaked preferences**（Black's theorem）：在受限域上 majority rule 满足所有 Arrow 公理

在表征分析中，是否有类似的正面结果？具体地：
- 如果放松 additivity（保留 completeness），最优方法是什么？（我们的 formalization §5.5 暗示梯度类方法可能是答案）
- 如果限制模型类（如只考虑 $L$-Lipschitz 网络），不可能性是否消失？
- 如果允许方法有随机性（如 smoothed SHAP），不可能性是否减弱？

### 4. "方法 portfolio" 的最优设计

给定 $K$ 种方法 $\Phi_1, \ldots, \Phi_K$（如 IG, SHAP, SAE, activation patching, linear probe），如何设计最优的组合策略？
- 是否有 **minimax optimal portfolio** 的理论？即选择权重 $w_1, \ldots, w_K$ 使得 worst-case identifiability gap 最小。
- 是否类似于 **robust portfolio theory**（Markowitz 投资组合的 robust 版本）？
- 是否有人在 XAI/interpretability 领域研究过 "explanation portfolio" 或 "multi-method verification"？

### 5. 因果角度：何时 adversarial failure 自然出现？

我们的实验表明 adversarial failure 在自然训练网络上不出现。但有些场景可能更接近 worst-case：
- **对抗性训练 (adversarial training)**：对抗性训练的模型是否更容易触发 Bilodeau 型 failure？
- **RLHF / alignment fine-tuning**：RLHF 后的模型是否可能发展出"远场对消"行为（表面上的行为改变不反映在 attribution 中）？
- **Model editing (ROME/MEMIT)**：精确编辑模型的某个事实后，attribution 方法是否能检测到？这是 Bilodeau 构造的自然版本。
- 是否有已知的工作研究了 "when do attribution methods fail in practice"？

### 6. 量化 "worst-case gap" 的工具

有没有数学工具可以精确量化"从 worst-case 到 average-case 的 gap"？
- **Smoothed analysis** (Spielman & Teng)：smoothed complexity 作为 worst-case 和 average-case 之间的中间概念。是否适用于 attribution impossibility？
- **Distributional robustness**：在 $\varepsilon$-Wasserstein ball 内的最坏分布上的 identifiability。当 $\varepsilon$ 从 0（average-case）变到 $\infty$（worst-case）时 IGA 如何变化？
- **Condition numbers for attribution**：类比数值分析中的条件数，定义 attribution 的条件数为"最小扰动使 identifiability 崩溃的大小"。

## 输出期望

1. **每个问题的文献综述**（附精确引用）
2. **"worst-case vs average-case" gap 最有前景的量化工具推荐**
3. **方法互补性理论中最接近我们需求的框架**
4. **Arrow 定理后正面结果的类比分析**——哪些策略最可能在表征分析中复现
5. **一个具体的研究计划**：如何将实验发现（SHAP 在 average-case 下 works）和理论不可能性（worst-case）之间的 gap 转化为一篇论文
6. **关键论文列表**（按优先级排序，附 arXiv ID）
