# DR-009: Squeeze Conjecture 的数学化——统一 Bilodeau 和 Sutter 的复杂度参数

## 背景

我们在研究神经网络表征分析方法的不可能性定理。已有两个方向相反的不可能性结果：

1. **Bilodeau et al. (arXiv:2212.11870, PNAS 2024)**：当归因方法满足 completeness + linearity（输出空间"太简单"）时，无法区分不同的局部行为——spec + sens ≤ 1（等于随机猜测）。

2. **Sutter et al. (arXiv:2507.08802, NeurIPS 2025 Spotlight)**：当 alignment map 不受限制（输出空间"太复杂"）时，任何神经网络都可以被映射到任何算法——causal abstraction 变得 trivial。

这两个结果形成一个 **squeeze**：
- Finiteness 太强 → ¬Identifiability (Bilodeau)
- Finiteness 太弱 → ¬Non-triviality (Sutter)

**核心猜想 (Squeeze Conjecture)**：不存在"恰好合适"的中间复杂度。形式化为：定义 $\Phi: M \times D \to F_c$，其中 $c$ 参数化 $F_c$ 的复杂度。存在 $c_0, c_1$ 使得 $c < c_0$ 时 identifiability 失败，$c > c_1$ 时 non-triviality 失败，且 $c_0 > c_1$（可行区间为空）。

**问题**：目前两个结果用完全不同的"复杂度"概念——Bilodeau 用输出空间的维度/结构约束（completeness + linearity），Sutter 用 alignment map 的函数类复杂度（线性 vs 非线性 vs 无约束）。要统一它们，需要找到**同一个复杂度参数** $c$ 能同时参数化两个方向。

## 需要调研的问题

### 1. 统一复杂度度量

有哪些数学工具可以统一"输出结构约束"和"映射函数类复杂度"？候选：
- **Rademacher complexity** / **Gaussian complexity**：能否用来同时度量 attribution 输出空间的表达力和 alignment map 的函数类丰富度？
- **VC 维 / fat-shattering dimension**：是否适用于连续输出的 attribution 方法？
- **Metric entropy / covering number**：能否统一 $\mathbb{R}^{d \times m}$（Bilodeau 的输出空间）和 bijection class（Sutter 的 alignment map 空间）？
- **描述复杂度 / Kolmogorov complexity**：是否有 computability-theoretic 的统一？
- **率失真理论 (Rate-Distortion Theory)**：$\Phi$ 可以视为一个信道，$c$ 是信道容量。Bilodeau 说容量太低时信息丢失；Sutter 说容量太高时信息冗余（无法区分信号和噪声）。这是否有精确的 Rate-Distortion 对应？

### 2. 已有的 "squeeze" 类结果

在其他数学领域，是否有类似的 "太弱失去 X，太强失去 Y，中间不存在" 的不可能性定理？特别是：
- **统计学习理论**中的 bias-variance tradeoff 是否有公理化版本证明最优点不存在？
- **信息论**中是否有类似的 channel capacity impossibility？
- **社会选择理论**（Arrow 定理的推广）中是否有参数化版本？
- **拓扑学**中的不动点定理是否有类似结构？

### 3. Sutter 的假设松弛

Sutter 的 trivialization 定理依赖 5 个假设（countable input space, input-injectivity, strict output-surjectivity, matchable partial-orderings, DNN solves the task）。当这些假设被逐一放松时：
- 哪些假设是"本质的"（去掉后定理不成立）？
- 有没有后续工作（2025-2026）研究了假设松弛后的版本？
- 特别是 **input-injectivity**：有维度瓶颈的架构（如 autoencoder）不满足这个假设。是否有替代的 trivialization 结果？

### 4. Bilodeau 的推广

Bilodeau 的不可能性依赖 completeness + linearity。如果放松 linearity（允许非线性 attribution），保留 completeness：
- 是否有已知的不可能性结果？
- **我们自己的工作**（mathproblem/impossibility-theorem/formalization_v3.md §5.5）证明了 completeness alone 不导致不可能性（通过梯度构造）。是否有人独立得出类似结论？
- 在 completeness + "弱于 linearity 的某种结构约束" 下，不可能性的边界在哪里？

### 5. 信息论框架

能否将整个问题重新表述为信息论问题？
- 定义 $I(\Phi; B)$ = $\Phi$ 的输出关于行为 $B$ 的互信息
- Bilodeau: 当 $\Phi$ 的输出结构受限时，$I(\Phi; B) = 0$（对 worst-case 模型）
- Sutter: 当 $\Phi$ 不受限时，$I(\Phi; B) = H(B)$ 但 $\Phi$ 无法区分 correct vs wrong $B$
- 是否有已知的信息论框架能精确刻画这个 tradeoff？
- **Rate-Distortion with side information** 是否是正确的数学模型？

### 6. 计算复杂度视角

是否可以从计算复杂度角度论证 squeeze？比如：
- 即使存在"恰好合适"的 $c$，计算 $\Phi$ 可能是 NP-hard 的
- 这类似于 mechanism design 中的"价格"：即使 Arrow 定理有 workaround（VCG 机制），VCG 在一般情况下是 computationally intractable 的
- 是否有 representation learning 方法的计算复杂度下界？

## 输出期望

1. **每个问题的现有文献综述**（附 arXiv ID 或出版信息）
2. **最有前景的统一复杂度度量的推荐**（附理由）
3. **已有的 squeeze 类定理的精确陈述**（如果存在的话）
4. **对 Squeeze Conjecture 可证性的判断**——是否有足够的数学工具来尝试证明？卡点在哪里？
5. **具体的下一步建议**——该读哪些论文？该用什么数学工具？
