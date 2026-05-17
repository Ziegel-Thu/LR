# 表征分析的不可能性定理：公理化前期工作

> 探索性工作，2026-05-17（修正稿 v3）

---

## 目录

1. [四篇论文的精确定理提取](#1-四篇论文的精确定理提取)
2. [统一数学框架](#2-统一数学框架)
3. [公理系统](#3-公理系统)
4. [已知的不相容对](#4-已知的不相容对)
5. [Mini Impossibility 定理](#5-mini-impossibility-定理)
6. [Open Gaps 与实验设计](#6-open-gaps-与实验设计)
7. [下一步建议](#7-下一步建议)

---

## 1. 四篇论文的精确定理提取

### 1.1 Bilodeau et al. (arXiv:2212.11870, PNAS 2024)

**论文标题**: Impossibility Theorems for Feature Attribution

**数学结构**:

- **输入空间**: Feature space $\mathcal{X} \subseteq \mathbb{R}^d$, output space $\mathcal{Y} \subseteq \mathbb{R}^m$, model class $\mathcal{F} \subseteq (\mathcal{X} \to \mathcal{Y})$
- **方法的形式定义**: Feature attribution method $\Phi: \mathcal{F} \times \mathcal{P}(\mathcal{X}) \times \mathcal{X} \to \mathbb{R}^{d \times m}$
  - 三个输入：模型 $f$、baseline distribution $\mu$、observation $x$
  - 输出：$d \times m$ 矩阵，每个元素 $\Phi(f, \mu, x)_{jk}$ 表示 feature $j$ 对 output $k$ 的归因

**公理精确陈述**:

**Definition (Completeness).** $\Phi$ is *complete* iff for all $f \in \mathcal{F}$, $\mu \in \mathcal{P}(\mathcal{X})$, $x \in \mathcal{X}$:

$$\sum_{j \in [d]} \Phi(f, \mu, x)_j = f(x) - \mathbb{E}_{x' \sim \mu}[f(x')]$$

**Definition (Linearity/Additivity).** $\Phi$ is *linear* (called "additive" in paper) iff for all collections $f^{(1)}, \ldots, f^{(d)}: \mathbb{R} \to \mathcal{Y}$, if $f(x) = \sum_j f^{(j)}(x_j) \in \mathcal{F}$, then:

$$\Phi(f, \mu, x)_j = \Phi(f^{(j)}, \mu_j, x_j)$$

**假设**:

**Assumption 1 (Baseline non-degeneracy).** For observation $x$, feature $j$, radius $\delta > 0$, baseline $\mu$: there exist $a_j, b_j$ s.t.
$$[x_j - \delta, x_j + \delta] \subseteq (a_j, b_j) \subseteq \{z_j : z \in \mathcal{X}\}$$
and $\mu_j((a_j, b_j) \setminus [x_j - \delta, x_j + \delta]) > 0$。

即 baseline 分布在感兴趣的邻域外有支撑。

**Assumption 2 (Model richness).** For any observation $x$, feature $j$, radius $\delta > 0$, and any target behavior $h: [x_j - \delta, x_j + \delta] \to \mathcal{Y}$, the model class contains all models that locally behave as $h$ and are piecewise linear outside:

$$\{f : \forall z \in \mathcal{N}, f(z) = h(z_j), \quad f|_{\mathcal{N}^c} \in \text{PwL}_2|_{\mathcal{N}^c}\} \subseteq \mathcal{F}$$

where $\mathcal{N} = [x_j - \delta, x_j + \delta] \times \mathcal{X}_{[d]\setminus\{j\}}$. 对于 ReLU 网络，这个假设自然成立。

**主定理 (Theorem 1)**:

> Fix any $x \in \mathcal{X}$, $j \in [d]$, $\delta > 0$, $\mu \in \mathcal{P}(\mathcal{X})$, and two local behaviors $h_0, h_1: [x_j - \delta, x_j + \delta] \to \mathbb{R}$. Under Assumptions 1 & 2, let
>
> $$\mathcal{F}_0 = \{f \in \mathcal{F} : \forall z \in \mathcal{N}, f(z) = h_0(z_j)\}, \quad \mathcal{F}_1 = \{f \in \mathcal{F} : \forall z \in \mathcal{N}, f(z) = h_1(z_j)\}$$
>
> For any **complete and linear** $\Phi$ and any hypothesis test $T: \mathbb{R}^{d \times m} \to [0,1]$:
>
> $$\text{specificity}_{\Phi,\mu,x}(T) \leq 1 - \text{sensitivity}_{\Phi,\mu,x}(T)$$

其中 specificity $= \inf_{f \in \mathcal{F}_0}[1 - T(\Phi(f,\mu,x))]$，sensitivity $= \inf_{f \in \mathcal{F}_1} T(\Phi(f,\mu,x))$。

**不可能性的精确含义**: 用完整且线性的归因方法推断 counterfactual model behavior 时，最优假设检验的 specificity + sensitivity ≤ 1，这恰好等于 random guessing 能达到的水平。换言之，任何基于 SHAP 或 Integrated Gradients 输出的推断策略都不能比随机猜测做得更好。

**证明的核心机制**: 构造两个模型 $f_0 \in \mathcal{F}_0$, $f_1 \in \mathcal{F}_1$，它们在邻域内有不同的局部行为，但 completeness + linearity 迫使 $\Phi(f_0, \mu, x) = \Phi(f_1, \mu, x) = \mathbf{0}$（或任意预设值）。关键在于可以通过调节模型在邻域外（baseline 有支撑的区域）的行为来"对消"掉局部行为对归因的影响。

---

### 1.2 Han, Srinivas & Lakkaraju (arXiv:2206.01254, NeurIPS 2022)

**论文标题**: Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post Hoc Explanations

**数学结构**:

- **输入空间**: Black-box model $f: \mathcal{X} \to \mathcal{Y}$，interpretable model class $\mathcal{G}$（通常是线性模型），input domain $\mathcal{X}$
- **方法的形式定义**: Local Function Approximation (LFA) framework:

$$g^* = \arg\min_{g \in \mathcal{G}} \mathbb{E}_{\xi \sim \mathcal{Z}}[\ell(f, g, x_0, \xi)]$$

其中 $\mathcal{Z}$ 是 $x_0$ 附近的邻域分布，$\ell$ 是损失函数。

**关键量**:

$$d(f, \mathcal{G}) = \min_{g \in \mathcal{G}} \max_{x \in \mathcal{X}} \ell(f, g, x_0, x)$$

$f$ 到可解释模型类 $\mathcal{G}$ 的"距离"，灵感来自 Hausdorff 距离。

> **勘误**：原文及论文部分表述中第三个参数写为 $0$，应为 $x_0$（参考点），下同。

**NFL 定理 (Theorem 1)**:

> Consider explaining $f$ around $x_0$ using $g \in \mathcal{G}$ and loss $\ell$, where $d(f, \mathcal{G}) = \min_{g \in \mathcal{G}} \max_{x \in \mathcal{X}} \ell(f, g, x_0, x)$.
>
> Then, for any explanation $g^*$ on neighborhood $\xi_1 \sim \mathcal{Z}_1$ satisfying $\max_{\xi_1} \ell(f, g^*, x_0, \xi_1) \leq \epsilon$, there always exists another neighborhood $\xi_2 \sim \mathcal{Z}_2$ such that:
>
> $$\max_{\xi_2} \ell(f, g^*, x_0, \xi_2) \geq d(f, \mathcal{G})$$

**不可能性的精确含义**: 当 $f \notin \mathcal{G}$（即 $d(f, \mathcal{G}) > 0$），任何在某个邻域上表现好的解释必然在另一个邻域上表现差，误差下界为 $d(f, \mathcal{G})$。不存在在所有邻域上同时最优的解释方法。

**证明机制**: 对任意 $g^*$，构造 adversarial input $x_{\text{adv}} = \arg\max_x \ell(f, g^*, x_0, x)$，然后构造包含此点的邻域 $\mathcal{Z}_2 = \text{Uniform}(x_0, x_{\text{adv}})$。由定义：
$$\max_{\xi_2} \ell(f, g^*, x_0, \xi_2) = \ell(f, g^*, x_0, x_{\text{adv}}) = \max_x \ell(f, g^*, x_0, x) \geq d(f, \mathcal{G})$$

**LFA 统一已有方法**: LIME、KernelSHAP、Occlusion、SmoothGrad、Integrated Gradients、Vanilla Gradients、C-LIME 都是 LFA framework 的特例（不同的 $\mathcal{G}$、$\mathcal{Z}$、$\ell$、perturbation operator $\oplus$ 选择）。

---

### 1.3 Sutter et al. (arXiv:2507.08802, NeurIPS 2025 Spotlight)

**论文标题**: The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?

**数学结构**:

- **输入空间**: Algorithm $\mathcal{A}$（实现 $f_{\text{task}}: \mathcal{X} \to \mathcal{Y}$ 的 DAG）与 DNN $\mathcal{N}$（实现 $f_N: \mathcal{X} \to \mathcal{Y}$ 的神经网络）的配对 $(\mathcal{A}, \mathcal{N})$
- **方法的形式定义**: Alignment map $\phi: \mathcal{H} \to \mathcal{H}$（双射、逐层可分解），加上 abstraction map $\tau$ 将 DNN 状态映射到算法状态

**关键定义链**:

1. **$\tau$-abstraction** (Def 1, from Beckers 2018): $\tau$ 是 surjective map，满足 (a) $\tau$ 满射到 algorithm 所有状态；(b-c) 干预一致性——在 algorithm 中的干预等价于在 DNN 中对应的干预
2. **Distributed abstraction** (Def 5): $\tau$ 可分解为逐层的 alignment map $\phi = [\phi_\ell]_{\ell=0}^{L+1}$，每层独立变换隐状态
3. **Input-restricted distributed abstraction** (Def 6): 干预限制为 input-reachable 的那些（即由实际输入通过网络传播产生的隐状态），而非任意常数干预

**假设**:

1. **Countable input-space**: $|\mathcal{X}|$ countable（语言模型、图像分类满足）
2. **Input-injectivity in all layers**: $f_{:\ell}$ (从输入到第 $\ell$ 层的映射) 对所有层单射——不丢失输入信息
3. **Strict output-surjectivity in all layers**: $\tau_{\text{out}} \circ f_{\ell:}$ 对所有层严格满射——每层都能产生所有目标输出
4. **Matchable partial-orderings**: 存在 DNN 神经元的 partition 匹配 algorithm 的偏序结构
5. **DNN solves the task**: $f_{\text{task}}(x) = \arg\max_y p_N(y|x)$ 对所有 $x \in \mathcal{X}$

> **注意**：Assumption 2（input-injectivity for all layers）在实际大型网络中是非平凡的——许多架构含有维度瓶颈（如 autoencoder 结构、pooling 层）。该假设的限制性在 Section 6 的 Open Gap 3 中进一步讨论。

**主定理 (Theorem 1)**:

> Given **any** algorithm $\mathcal{A}$ and **any** neural network $\mathcal{N}$ such that Assumptions 1–5 hold, we can show that $\mathcal{A}$ is an input-restricted distributed abstraction of $\mathcal{N}$.

**不可能性的精确含义**: 方向与前两篇相反。这里不是说"好方法不存在"，而是说**当 alignment map 不受限制时，任何 DNN 都可以被任何 algorithm 抽象**——causal abstraction 变成 vacuous（空洞）。

这导致了 **non-linear representation dilemma**:
- 限制为线性 alignment → causal abstraction 有意义，但可能遗漏非线性编码的信息
- 允许非线性 alignment → causal abstraction 变得 trivial，失去区分力

**证明核心机制**: 利用 injectivity（DNN 的隐状态唯一确定输入）和 surjectivity（每层都能产生任意输出），递归地构造逐层 alignment map。对每个 input-restricted intervention，可以找到一个 bijective $\phi_\ell$ 使得变换后的隐状态恰好满足 interventional consistency。由于 input-restricted interventions 是 countable 的，而隐状态空间是连续的（uncountable），有足够的"自由度"来实现任意映射。

**实验验证**: 在 randomly initialized language models 上，使用非线性 alignment maps (RevNets) 在 IOI task 上达到 100% interchange intervention accuracy。

---

### 1.4 Kleinberg (NeurIPS 2002)

**论文标题**: An Impossibility Theorem for Clustering

**数学结构**:

- **输入空间**: Finite set $S$（$|S| \geq 2$），distance function $d: S \times S \to \mathbb{R}_{\geq 0}$（满足 $d(i,j) = 0 \iff i = j$）
- **方法的形式定义**: Clustering function $f: \mathcal{D}(S) \to \Pi(S)$，从 $S$ 上所有距离函数的集合映射到 $S$ 的所有 partition 的集合

**公理精确陈述**:

**Axiom 1 (Scale-Invariance).** For any distance function $d$ and any $\alpha > 0$:

$$f(d) = f(\alpha \cdot d)$$

**Axiom 2 (Richness).** For every partition $\mathcal{P}$ of $S$, there exists a distance function $d$ such that:

$$f(d) = \mathcal{P}$$

**Axiom 3 (Consistency).** Let $\Gamma = f(d)$. If $d'$ is a distance function such that:
- For all $i, j$ in the **same** cluster of $\Gamma$: $d'(i,j) \leq d(i,j)$
- For all $i, j$ in **different** clusters of $\Gamma$: $d'(i,j) \geq d(i,j)$

then $f(d') = \Gamma$。

**不可能性定理 (Theorem 3.1)**:

> For $|S| \geq 2$, there is **no** clustering function $f$ that simultaneously satisfies Scale-Invariance, Richness, and Consistency.

**不可能性的精确含义**: 这三个看似温和、独立合理的公理不可能同时满足。Scale-invariance 要求聚类不受距离的绝对尺度影响；Richness 要求方法足够灵活以产出任何聚类；Consistency 要求加强类内距离/类间距离后聚类不变。

**与表征分析的关联**: SAE feature discovery 本质上是一个聚类/字典学习问题（将 activation space 分解为 clusters/directions）。Kleinberg 的定理意味着任何将 activation vectors 聚类为 features 的方法都无法同时满足这三条公理。

---

## 2. 统一数学框架

### 2.1 共同结构抽取

四篇论文都涉及以下共同结构：

| 组件 | Bilodeau | Han | Sutter | Kleinberg |
|------|----------|-----|--------|-----------|
| **被分析对象** $M$ | Model $f \in \mathcal{F}$ | Black-box $f$ | DNN-Algorithm pair $(\mathcal{N}, \mathcal{A})$ | Distance function $d$ |
| **上下文/数据** $D$ | $(\mu, x)$ | $(x_0, \mathcal{Z})$ | Input set $\mathcal{X}$ | (无) |
| **方法输出** $F$ | Attribution $\in \mathbb{R}^{d \times m}$ | Linear model $g \in \mathcal{G}$ | Abstraction $(\phi, \tau)$ | Partition $\Gamma \in \Pi(S)$ |
| **目标行为** $B$ | Local model behavior $h$ | $f$ restricted to $\mathcal{Z}$ | "Does $\mathcal{A}$ explain $\mathcal{N}$?" | "True" clustering structure |
| **不相容公理** | Completeness + Linearity + Richness → ¬Identifiability | Faithfulness_local + Universality_neighborhoods → failure | Unrestricted maps + Regularity → Triviality | Scale-inv + Richness + Consistency → ⊥ |

### 2.2 统一定义

**Definition (Representation Decomposition Method).** 一个 *表征分解方法* (representation decomposition method) 是一个元组 $(M, D, F, O, \Phi, B)$，其中：

- $M$ 是 **模型空间** (model space)——被分析的对象全体
- $D$ 是 **上下文空间** (context space)——辅助输入（数据、邻域、baseline）
- $F$ 是 **特征空间** (feature space)——分解/分析的输出
- $O$ 是 **行为空间** (behavior space)——我们想要推断的目标
- $\Phi: M \times D \to F$ 是 **分析映射** (analysis map)
- $B: M \times D \to O$ 是 **行为映射** (behavior map)

**目标**: 从 $\Phi(m, d)$ 推断 $B(m, d)$。

### 2.3 四篇论文到统一框架的映射

| 论文 | $M$ | $D$ | $F$ | $O$ | $\Phi$ | $B$ |
|------|-----|-----|-----|-----|--------|-----|
| Bilodeau | $\mathcal{F} \subseteq (\mathcal{X} \to \mathcal{Y})$ | $\mathcal{P}(\mathcal{X}) \times \mathcal{X}$ | $\mathbb{R}^{d \times m}$ | $(\mathcal{X}_j \to \mathcal{Y})$ | Attribution $\Phi(f,\mu,x)$ | Local behavior $f\|_{\mathcal{N}}$ |
| Han | $(\mathcal{X} \to \mathcal{Y})$ | $\mathcal{X} \times \text{Nbhd}$ | $\mathcal{G}$ (linear models) | $f\|_{\mathcal{Z}}$ | $g^* = \text{LFA}(f, \mathcal{Z})$ | $f$ restricted to $\mathcal{Z}$ |
| Sutter | DNN class | $\mathcal{X}$ (inputs) | $\{(\phi, \tau)\}$ (alignments) | $\{0, 1\}$ (aligned or not) | Best alignment map | Does $\mathcal{A}$ genuinely abstract $\mathcal{N}$? |
| Kleinberg | $\mathcal{D}(S)$ (distances) | $\emptyset$ | $\Pi(S)$ (partitions) | $\Pi(S)$ | Clustering $f(d)$ | "True" partition |

---

## 3. 公理系统

### 3.1 五个公理

在统一框架 $\Phi: M \times D \to F$ 下，我们区分五个公理层次。关键的改进是将原先混淆的 "Faithfulness" 拆分为 **Identifiability**（信息论层面）和 **Structural Faithfulness**（代数/几何层面）。

---

**Axiom 1: Identifiability（可辨识性）**

$\Phi$ 在行为等价类上是单射——不同行为的模型产生不同的分析输出：

$$B(m_1, d) \neq B(m_2, d) \implies \Phi(m_1, d) \neq \Phi(m_2, d)$$

这是纯粹的信息论要求：$\Phi(\cdot, d)$ 不丢失关于 $B(\cdot, d)$ 的信息。

**各论文中的实例化**:
- Bilodeau: hypothesis test 的 spec + sens $> 1$（比 random guessing 好）
- Han: 解释 $g^*$ 在邻域上的误差足够小
- Sutter: IIA = 100% 且 non-trivial
- Kleinberg: （隐含——不同 "true" partition 应产出不同聚类）

---

**Axiom 2: Structural Faithfulness（结构忠实性）**

$\Phi$ 的输出不仅编码了 $B$ 的信息，而且以特定的结构化方式编码——输出满足特定的代数/几何约束，这些约束反映了"解释"的语义要求。

Structural Faithfulness 严格强于 Identifiability：它不仅要求 $\Phi$ 能区分不同行为，还要求区分的**方式**满足特定结构。

**各论文中的实例化**:

| 论文 | Structural Faithfulness 的具体形式 |
|------|-----------------------------------|
| Bilodeau | Completeness: $\sum_j \Phi_j = f(x) - \mathbb{E}_\mu[f]$ |
| Han | $g^* \in \mathcal{G}$ 且 $g^*$ 近似 $f$（LFA 最小化） |
| Sutter | $\tau$ 满足 interventional consistency |
| Kleinberg | Consistency: cluster-strengthening 不改变输出 |

**与 Identifiability 的区别**：Identifiability 只要求"不同行为 → 不同输出"，不管输出的具体结构。Structural Faithfulness 进一步要求输出具备可解释的语义（如归因之和等于输出差）。一个 hash 函数可以满足 Identifiability（它是单射），但不满足 Structural Faithfulness（hash 值没有归因语义）。

---

**Axiom 3: Generality（通用性/Richness）**

$M$ 和 $D$ 足够丰富，$\Phi$ 在整个空间上有定义且非退化。

- **Model richness**: $M$ 包含"足够复杂"的模型（如所有 piecewise linear functions、所有满足正则性条件的 DNN）
- **Context universality**: $\Phi$ 对所有 $d \in D$ 都有定义（对所有 neighborhoods、所有 baselines、所有 distance functions）
- **Output surjectivity (Richness)**: $\Phi(M \times D) = F$ 或 $\Phi$ 的像"覆盖" $F$

**各论文中的实例化**:
- Bilodeau: Model class 包含 piecewise linear functions (Assumption 2)
- Han: $f$ 可以是任意非线性函数；结果对所有 $\ell$, $\mathcal{G}$, $\mathcal{Z}_1$ 成立
- Sutter: 任意 algorithm + 满足 5 个假设的任意 DNN
- Kleinberg: Richness axiom——任何 partition 都可达

---

**Axiom 4: Non-triviality（非平凡性）**

$\Phi$ 提供的信息不是 trivially obtainable 的——存在 $m_1, m_2$ 使得 $\Phi(m_1, d) \neq \Phi(m_2, d)$，且这个区分有实际意义。

更强的形式：$\Phi$ 不会把所有 $(m, d)$ pair 都映射到相同的输出。

**各论文中的实例化**:
- Bilodeau: 要求 hypothesis test 能区分 $\mathcal{F}_0$ 和 $\mathcal{F}_1$
- Han: 解释应当在某个邻域上忠实（$\leq \epsilon$）
- Sutter: 主定理的**否定面**——当 alignment map 无约束时，Non-triviality 被违反（任何 algorithm 都 align 任何 DNN）
- Kleinberg: 隐含在 Richness 中——方法能产出不同的聚类

---

**Axiom 5: Finiteness / Simplicity（有限性/简洁性）**

$F$ 的复杂度有界，$\Phi$ 的输出结构受限。

- **Output structure**: $F$ 是有限维的（$\mathbb{R}^{d \times m}$）、线性的（$\mathcal{G}$ = 线性模型）、或 scale-free 的
- **Method structure**: $\Phi$ 本身满足结构约束（可加性、scale-invariance）

**各论文中的实例化**:
- Bilodeau: Additivity（归因在可加模型上的分解结构）——这是**不可能性的真正来源**（见 Section 5.5）
- Han: $\mathcal{G}$ 是线性模型类（比 $f$ 简单）
- Sutter: **反面**——当不施加 finiteness（允许任意复杂的 alignment map）时，faithfulness 变得 trivial
- Kleinberg: Scale-invariance（输出不依赖于距离的绝对尺度）

---

### 3.2 公理间的层级关系

```
Structural Faithfulness ⟹ Identifiability（前者严格强于后者）

       Structural Faithfulness
              ↕ (tension)
      Finiteness + Generality
              ↕ (tension)
         Non-triviality
```

**关键张力**:
- Structural Faithfulness + Finiteness + Generality → ¬Identifiability (Bilodeau)
- Structural Faithfulness + Generality → ¬Universality-across-neighborhoods (Han)
- ¬Finiteness + Generality → ¬Non-triviality (Sutter)
- Scale-invariance + Richness + Consistency → ⊥ (Kleinberg)

---

## 4. 已知的不相容对

### 4.1 从四篇论文提取的不相容关系

**不相容性 I**: {Str. Faithfulness (completeness), Finiteness (additivity), Generality (rich models)} → ¬Identifiability

- **来源**: Bilodeau et al.
- **含义**: Complete + additive attributions 无法区分不同局部行为
- **形式**: spec + sens ≤ 1 (random guessing)
- **覆盖的方法**: SHAP, Integrated Gradients

**不相容性 II**: {Str. Faithfulness (local accuracy), Generality (all neighborhoods)} → bounded failure

- **来源**: Han et al.
- **含义**: 在一个邻域上准确的解释在另一个邻域上误差不小于 $d(f, \mathcal{G})$
- **前提**: $f \notin \mathcal{G}$（被解释的函数不在可解释模型类中）
- **覆盖的方法**: 所有 LFA 类方法（LIME, SHAP, SmoothGrad, ...）

**不相容性 III**: {Identifiability (interventional consistency), Generality (any alg-DNN pair), ¬Finiteness (unrestricted maps)} → ¬Non-triviality

- **来源**: Sutter et al.
- **含义**: 无约束的 causal abstraction 是 vacuous 的
- **等价表述**: 如果同时要求 identifiability, generality, 和 non-triviality，则必须施加 finiteness (如 linearity constraint on alignment maps)
- **覆盖的方法**: DAS, causal abstraction (无线性约束时)

**不相容性 IV**: {Scale-Invariance, Richness, Consistency} → ⊥

- **来源**: Kleinberg
- **含义**: 三条公理不可能同时满足
- **映射到统一框架**: {Finiteness (scale-inv), Generality (richness), Str. Faithfulness (consistency)} → ⊥
- **覆盖的方法**: 所有 clustering functions

> **关于 Kleinberg 映射的坦率说明**：将 Scale-Invariance 归入 "Finiteness"、Consistency 归入 "Structural Faithfulness" 带有一定程度的概念延伸。Scale-invariance 本质是对称性约束而非复杂度上界；Consistency 是单调稳定性条件，与"从 $\Phi$ 推断 $B$"意义上的忠实性有距离。此映射更多是启发式类比，而非严格蕴含。

### 4.2 不相容性的统一视角

所有四个不相容性都可以表述为以下 meta-pattern:

> **当分析方法的输出空间 $F$ 相对于模型空间 $M$ "太简单"时，无法同时忠实地反映模型行为并对所有模型有效。当输出空间不受限制时，方法变得 trivially "忠实"但失去区分力。**

这形成一个 **squeeze**:
- 约束太强（Finiteness ↑）：方法失去 Identifiability (Bilodeau, Han, Kleinberg)
- 约束太弱（Finiteness ↓）：方法失去 Non-triviality (Sutter)

**Finiteness 是关键调节器**: 它控制了 $F$ 相对于 $M$ 的复杂度比。最优点——既 faithful 又 non-trivial——可能不存在（这正是 impossibility theorem 要证明的）。

### 4.3 不相容对总结表

| 公理组合 | 来源 | 结论 |
|----------|------|------|
| Str. Faithfulness + Finiteness + Generality | Bilodeau | ¬Identifiability |
| Str. Faithfulness + Context-universality | Han | ¬(同时 faithful everywhere) |
| Identifiability + Generality + (¬Finiteness) | Sutter | ¬Non-triviality |
| Finiteness + Generality + Str. Faithfulness | Kleinberg | ⊥ (不可能) |
| Identifiability + Str. Faithfulness + Generality | *conjectured* | ¬Finiteness (需要无界复杂度) |
| Identifiability + Finiteness + Generality | *conjectured* | ¬Str. Faithfulness |

---

## 5. Mini Impossibility 定理

### 5.1 定理陈述

**Theorem (Mini Impossibility for Attribution under Additivity).**

**Setting.** 令 $\mathcal{X} = \mathbb{R}^d$，$\mathcal{Y} = \mathbb{R}$。令 $M \subseteq (\mathcal{X} \to \mathcal{Y})$ 为模型类。固定上下文 $\mathbf{d} = (\mu, x, \delta)$，其中 $\mu \in \mathcal{P}(\mathcal{X})$ 为 baseline 分布，$x \in \mathcal{X}$ 为观测点，$\delta > 0$ 定义邻域 $\mathcal{N}_j = \{z \in \mathcal{X} : |z_j - x_j| \leq \delta\}$。

令 $\Phi: M \times D \to \mathbb{R}^d$ 为归因方法。对特征 $j$ 和局部行为 $h: [x_j - \delta, x_j + \delta] \to \mathbb{R}$，定义

$$M_h^{(j)} = \{m \in M : m(z) = h(z_j) \text{ for all } z \in \mathcal{N}_j\}$$

为所有在 $\mathcal{N}_j$ 上沿特征 $j$ 表现为 $h$ 的模型之集。

**公理：**

**(A1) Identifiability.** 存在 inference function $I: \mathbb{R}^d \to \{0,1\}$，对任意两个可区分的局部行为 $h_0 \neq h_1$ 及任意特征 $j \in [d]$：

$$\text{spec}(I) + \text{sens}(I) > 1$$

其中

$$\text{spec}(I) := \inf_{m \in M_{h_0}^{(j)}} [1 - I(\Phi(m, \mathbf{d}))], \qquad \text{sens}(I) := \inf_{m \in M_{h_1}^{(j)}} I(\Phi(m, \mathbf{d}))$$

即假设检验严格优于随机猜测。

**(A2) Completeness + Additivity.** 分两部分：

**(A2a) Completeness.** 对所有 $m \in M$：

$$\sum_{j=1}^d \Phi(m, \mathbf{d})_j = m(x) - \mathbb{E}_{x' \sim \mu}[m(x')]$$

**(A2b) Additivity.** 记 $M_{\text{add}} = \{m \in M : m(x) = \sum_{j=1}^d m_j(x_j) \text{ for some univariate } m_1, \ldots, m_d\}$。对任意 $m \in M_{\text{add}}$：

$$\Phi(m, \mathbf{d})_j = \phi_j(m_j, \mu_j, x_j)$$

其中 $\phi_j$ 仅依赖于 $m$ 的第 $j$ 个分量 $m_j$、$\mu$ 的第 $j$ 个边际 $\mu_j$、以及 $x_j$。

> **范围注记**：(A2a) 约束所有模型；(A2b) 仅约束可加子类 $M_{\text{add}}$。这个区分对证明的正确性至关重要。

**(A3) Model Richness.** 分两部分：

**(A3a) Additive richness.** 对任意特征 $j$ 和任意连续函数 $g_j: \mathbb{R} \to \mathbb{R}$，存在可加模型 $m \in M_{\text{add}}$ 满足 $m(x) = g_j(x_j)$（即 $m_k \equiv 0$ for $k \neq j$，$m_j = g_j$）。

**(A3b) Baseline non-degeneracy.** 对每个特征 $j$，存在 $a_j < x_j - \delta < x_j + \delta < b_j$ 使得

$$\mu_j\big((a_j, b_j) \setminus [x_j - \delta, x_j + \delta]\big) > 0$$

**Claim.** (A1)、(A2)、(A3) 不能同时满足。

---

### 5.2 证明

我们证明 (A2) + (A3) $\implies$ $\lnot$(A1)。

**Step 1. 在可加单特征模型上确定归因公式。**

取特征 $j$，考虑单特征可加模型 $m(x) = m_j(x_j)$（即 $m_k \equiv 0$ for $k \neq j$）。由 (A3a)，对任意连续 $m_j$，此模型属于 $M_{\text{add}} \subseteq M$。

由 (A2b)：$\Phi(m, \mathbf{d})_j = \phi_j(m_j, \mu_j, x_j)$，且 $\Phi(m, \mathbf{d})_k = \phi_k(0, \mu_k, x_k)$ for $k \neq j$。

由 (A2a)：

$$\phi_j(m_j, \mu_j, x_j) + \sum_{k \neq j} \phi_k(0, \mu_k, x_k) = m_j(x_j) - \mathbb{E}_{\mu_j}[m_j(X_j)]$$

其中右端使用了 $m$ 仅依赖 $x_j$ 的事实，故 $\mathbb{E}_\mu[m] = \mathbb{E}_{\mu_j}[m_j]$。

定义 $C_j := \sum_{k \neq j} \phi_k(0, \mu_k, x_k)$（不依赖于 $m_j$ 的常数）。则

$$\phi_j(m_j, \mu_j, x_j) = m_j(x_j) - \mathbb{E}_{\mu_j}[m_j] - C_j \qquad (\star)$$

**确定 $C_j$**：取 $m_j \equiv 0$（零函数），模型为 $m \equiv 0$。由 (A2a)：$\sum_k \Phi(0, \mathbf{d})_k = 0$。由 (A2b)：$\Phi(0, \mathbf{d})_k = \phi_k(0, \mu_k, x_k)$ for all $k$。故 $\sum_k \phi_k(0, \mu_k, x_k) = 0$。

对 $(\star)$ 代入 $m_j \equiv 0$：$\phi_j(0, \mu_j, x_j) = 0 - 0 - C_j = -C_j$。

同时 $\sum_k \phi_k(0, \mu_k, x_k) = 0$ 即 $\phi_j(0, \mu_j, x_j) + C_j = 0$，即 $-C_j + C_j = 0$——自洽但不能确定 $C_j$ 的值。

**关键观察**：$C_j$ 的具体值不影响证明。对任意两个函数 $m_j', m_j''$，由 $(\star)$：

$$\phi_j(m_j', \mu_j, x_j) - \phi_j(m_j'', \mu_j, x_j) = \big[m_j'(x_j) - \mathbb{E}_{\mu_j}[m_j']\big] - \big[m_j''(x_j) - \mathbb{E}_{\mu_j}[m_j'']\big]$$

$C_j$ 在差中消去。Step 2 只需要控制归因的**差**（使两个模型归因相等），因此 $C_j$ 无关紧要。等价地，不妨取归因的"零点"使得 $C_j = 0$，则

$$\Phi(m, \mathbf{d})_j = m_j(x_j) - \mathbb{E}_{\mu_j}[m_j] \qquad \text{for all single-feature additive } m(x) = m_j(x_j) \quad (\star\star)$$

> **适用范围声明**：$(\star\star)$ **仅对可加单特征模型成立**。对一般非可加模型，(A2b) 不适用，$\Phi$ 的行为不被 $(\star\star)$ 约束。

---

**Step 2. 在可加模型内，局部行为与归因解耦。**

固定特征 $j$，取两个不同的局部行为 $h_0, h_1: [x_j - \delta, x_j + \delta] \to \mathbb{R}$，$h_0 \neq h_1$。取任意目标归因值 $v \in \mathbb{R}$。

**构造**：对 $i \in \{0, 1\}$，定义单特征模型 $m_i(x) = g_i(x_j)$，其中 $g_i: \mathbb{R} \to \mathbb{R}$ 为如下**连续**函数。选取辅助参数 $\eta > 0$，满足 $\eta < \min\big(\delta, \frac{b_j - (x_j + \delta)}{2}, \frac{(x_j - \delta) - a_j}{2}\big)$（确保以下过渡区间不重叠）。定义：

$$g_i(t) = \begin{cases} h_i(t) & t \in [x_j - \delta, x_j + \delta] \\[4pt] h_i(x_j + \delta) + \dfrac{t - (x_j + \delta)}{\eta}\big(\lambda_i - h_i(x_j + \delta)\big) & t \in (x_j + \delta, x_j + \delta + \eta] \\[4pt] h_i(x_j - \delta) + \dfrac{(x_j - \delta) - t}{\eta}\big(\lambda_i - h_i(x_j - \delta)\big) & t \in [x_j - \delta - \eta, x_j - \delta) \\[4pt] \lambda_i & t \in [a_j + \eta, x_j - \delta - \eta] \cup [x_j + \delta + \eta, b_j - \eta] \setminus \text{(transition bands)} \\[4pt] \lambda_i \cdot \dfrac{t - a_j}{\eta} & t \in [a_j, a_j + \eta] \\[4pt] \lambda_i \cdot \dfrac{b_j - t}{\eta} & t \in [b_j - \eta, b_j] \\[4pt] 0 & t \notin (a_j, b_j) \end{cases}$$

> **连续性说明**：$g_i$ 在所有拼接点 $x_j \pm \delta$、$x_j \pm \delta \pm \eta$、$a_j$、$a_j + \eta$、$b_j - \eta$、$b_j$ 处均连续（各段在边界处的值相等），因此 $g_i \in C^0(\mathbb{R})$，满足 (A3a) 对连续函数的要求。

由 $(\star\star)$：

$$\Phi(m_i, \mathbf{d})_j = g_i(x_j) - \mathbb{E}_{\mu_j}[g_i]$$

将 $\mathbb{E}_{\mu_j}[g_i]$ 分解为局部、远场、过渡带三部分的积分：

$$\mathbb{E}_{\mu_j}[g_i] = \underbrace{\int_{x_j - \delta}^{x_j + \delta} h_i(t) \, d\mu_j(t)}_{=: A_i} + \lambda_i \cdot p_\eta + R_i(\eta)$$

其中 $p_\eta := \mu_j\big((a_j + \eta, b_j - \eta) \setminus [x_j - \delta - \eta, x_j + \delta + \eta]\big)$，$R_i(\eta)$ 是过渡带上的积分贡献（有界，且 $R_i(\eta) \to 0$ as $\eta \to 0$）。

由于 $g_i(x_j) = h_i(x_j)$：

$$\Phi(m_i, \mathbf{d})_j = h_i(x_j) - A_i - \lambda_i \cdot p_\eta - R_i(\eta)$$

**关键**：由 (A3b)，当 $\eta$ 足够小时 $p_\eta > 0$（因为 $\mu_j((a_j, b_j) \setminus [x_j - \delta, x_j + \delta]) > 0$，而从该集合中去掉测度趋于零的过渡带后仍有正测度）。选取

$$\lambda_i = \frac{h_i(x_j) - A_i - R_i(\eta) - v}{p_\eta}$$

> **注记（隐式依赖的可解性）**：上式中 $R_i(\eta) = \int_{\text{transition bands}} g_i(t)\, d\mu_j(t)$ 本身依赖于 $\lambda_i$（因为 $g_i$ 在过渡带上取的值介于 $h_i$ 边界值和 $\lambda_i$ 之间）。设过渡带上的贡献可分解为 $R_i(\eta) = \lambda_i \cdot q_\eta^{(1)} + q_\eta^{(2)}$，其中 $q_\eta^{(1)} = \mu_j$-加权的与 $\lambda_i$ 线性相关部分（来自线性插值段），$q_\eta^{(2)}$ 是与 $h_i$ 有关的常数项。代入后得到：
> $$\lambda_i(p_\eta + q_\eta^{(1)}) = h_i(x_j) - A_i - q_\eta^{(2)} - v$$
> 由于 $p_\eta > 0$（已论证）且 $q_\eta^{(1)} \geq 0$（过渡带上 $g_i$ 的 $\lambda_i$ 系数非负），故 $p_\eta + q_\eta^{(1)} > 0$，$\lambda_i$ 有唯一解。当 $\eta \to 0$ 时 $q_\eta^{(1)} \to 0$，$q_\eta^{(2)} \to 0$，$\lambda_i$ 退化为 $\frac{h_i(x_j) - A_i - v}{p_0}$，其中 $p_0 = \mu_j((a_j, b_j) \setminus [x_j-\delta, x_j+\delta]) > 0$。

则 $\Phi(m_i, \mathbf{d})_j = v$。

同时，$\Phi(m_i, \mathbf{d})_k = \phi_k(0, \mu_k, x_k)$ for $k \neq j$（因 $m_i$ 是单特征模型，第 $k$ 分量为零），不依赖于 $i$。

因此 $\Phi(m_0, \mathbf{d}) = \Phi(m_1, \mathbf{d}) =: \mathbf{v}$。

由 (A3a)，上述 $g_i$（连续单变量函数）对应的单特征模型属于 $M_{\text{add}} \subseteq M$。由构造，$m_i \in M_{h_i}^{(j)}$（$m_i$ 在邻域内行为为 $h_i$）。

---

**Step 3. 不可能有优于随机猜测的推断。**

由 Step 2，对任意 inference function $I: \mathbb{R}^d \to \{0,1\}$：

$$\text{spec}(I) = \inf_{m \in M_{h_0}^{(j)}} [1 - I(\Phi(m, \mathbf{d}))] \leq 1 - I(\mathbf{v})$$

$$\text{sens}(I) = \inf_{m \in M_{h_1}^{(j)}} I(\Phi(m, \mathbf{d})) \leq I(\mathbf{v})$$

其中不等号成立是因为 $m_0 \in M_{h_0}^{(j)}$，$m_1 \in M_{h_1}^{(j)}$，$\inf$ 不超过任一具体元素处的值。

相加：$\text{spec}(I) + \text{sens}(I) \leq 1$。与 (A1) 的要求 $> 1$ 矛盾。$\square$

---

### 5.3 定理的实际覆盖范围：超越可加模型

上述证明只构造了**可加模型**作为 adversarial 实例。一个自然的疑问是：这是否意味着定理仅对可加模型类有效？

**答案：否。定理覆盖任意包含可加子类的模型空间。**

论证如下。(A1) 中的 spec 和 sens 使用 $\inf$ 遍历整个 $M_{h_i}^{(j)}$，而

$$M_{h_i}^{(j)} \cap M_{\text{add}} \subseteq M_{h_i}^{(j)}$$

因此

$$\inf_{m \in M_{h_i}^{(j)}} \leq \inf_{m \in M_{h_i}^{(j)} \cap M_{\text{add}}}$$

Step 2 证明了右端已经足以使 spec + sens $\leq 1$。更直接地说：由于 $m_0 \in M_{h_0}^{(j)}$（不仅仅在 $M_{\text{add}}$ 中），Step 2 构造的 adversarial 实例 $m_0, m_1$ 直接属于 $M_{h_0}^{(j)}$ 和 $M_{h_1}^{(j)}$，因此 Step 3 的论证对整个 $M$ 直接成立——$\inf$ 被这些具体实例所约束。

直觉：**可加模型是每个行为类 $M_{h_i}^{(j)}$ 中的 "最难案例"——它们利用远场自由度来对消局部信号。任何推断策略，无论它在非可加模型上多强，都必须面对这些最难案例。**

---

### 5.4 边际公式能否推广到非可加模型？

原文（v1）隐含地假设了 $\Phi(m,\mathbf{d})_j = m_j(x_j) - \mathbb{E}_{\mu_j}[m_j]$ 对一般模型也成立。这是**错误的**。显式反例如下：

**Counterexample.** 取 $d=2$。定义 $\Phi$ 为分段函数：

$$\Phi(m, \mathbf{d}) = \begin{cases} \big(m_1(x_1) - \mathbb{E}_{\mu_1}[m_1],\; m_2(x_2) - \mathbb{E}_{\mu_2}[m_2]\big) & \text{if } m \in M_{\text{add}},\; m = m_1 + m_2 \\[6pt] \big(m(x) - \mathbb{E}_\mu[m],\; 0\big) & \text{if } m \notin M_{\text{add}} \end{cases}$$

**验证**:

- **(A2a) Completeness**：对 $m \in M_{\text{add}}$，$\Phi_1 + \Phi_2 = [m_1(x_1) - \mathbb{E}[m_1]] + [m_2(x_2) - \mathbb{E}[m_2]] = m(x) - \mathbb{E}[m]$。对 $m \notin M_{\text{add}}$，$\Phi_1 + \Phi_2 = m(x) - \mathbb{E}[m] + 0 = m(x) - \mathbb{E}[m]$。两种情况均满足。
- **(A2b) Additivity**：对 $m \in M_{\text{add}}$，$\Phi_j$ 仅依赖 $m_j, \mu_j, x_j$。满足。
- **非边际公式**：对 $m \notin M_{\text{add}}$，$\Phi_2 = 0$，所有归因被倾倒到第一个特征中。

这说明从 (A2a) + (A2b) **无法推导出**边际公式对一般模型成立。但如 Section 5.3 所论证，这不影响不可能性定理本身。

---

### 5.5 不可能性的根源诊断：Additivity 而非 Completeness

自然的下一个问题：如果 $\Phi$ 只满足 completeness (A2a) 而**不**满足 additivity (A2b)，可辨识性是否可能？

**Proposition.** 仅 completeness 不导致不可能性。存在满足 (A2a) 且达到 spec + sens $> 1$ 的 $\Phi$（在适当正则的模型类上）。

**Proof (by construction).** 取 $d=2$。将模型类限制为 $M \cap C^1(\mathcal{X})$（连续可微函数），并将 (A3a) 中的 "连续函数" 相应替换为 "$C^1$ 函数"（即 additive richness 要求 $g_j \in C^1$）。定义

$$\Phi(m, \mathbf{d})_1 := \frac{\partial m}{\partial x_1}\bigg|_x, \qquad \Phi(m, \mathbf{d})_2 := m(x) - \mathbb{E}_\mu[m] - \frac{\partial m}{\partial x_1}\bigg|_x$$

**验证**:

- **(A2a) Completeness**: $\Phi_1 + \Phi_2 = m(x) - \mathbb{E}[m]$。满足。
- **不满足 (A2b) Additivity**: 对可加 $m(x) = m_1(x_1) + m_2(x_2)$，$\Phi_1 = m_1'(x_1)$，一般不等于 $m_1(x_1) - \mathbb{E}_{\mu_1}[m_1]$。

此时 $\Phi_1$ 编码了 $m$ 在 $x$ 处沿 $x_1$ 方向的梯度——这是**真正的局部信息**。对两个 $C^1$ 局部行为 $h_0, h_1$ 满足 $h_0'(x_1) \neq h_1'(x_1)$，对所有 $m \in M_{h_i}^{(1)}$，$\Phi_1 = h_i'(x_1)$（无论远场行为如何，因为梯度仅由 $x$ 处的局部信息决定）。因此 $I(\Phi) = \mathbb{1}[\Phi_1 = h_1'(x_1)]$ 给出 spec + sens $= 2 > 1$。$\square$

> **正则性假设注记**：此构造需要 $m \in C^1(\mathcal{X})$，而原始 (A3a) 只要求连续 $g_j$（连续函数未必可微）。因此该 Proposition 的适用范围是 $C^1$ 模型类，而非全部连续模型类。结论的**定性含义**不受影响：它证明了 completeness alone 不排斥可辨识性，关键限制来自 additivity。定量上，$h_0'(x_1) \neq h_1'(x_1)$ 的要求将可区分的行为对限制为在 $x_1$ 处导数不同的那些。

**结论：Additivity 是不可能性的真正来源，而非 Completeness。**

机制解释：Additivity 迫使 $\Phi$ 在可加模型上通过全局积分 $\mathbb{E}_{\mu_j}[m_j]$ 编码信息。这个积分混合了邻域内（局部）和邻域外（远场）的模型行为。远场行为因此获得了"对消"局部信号的能力。Completeness 本身不强加这种混合。

---

### 5.6 综合评价

**已证明的**:
1. (A2) + (A3) → ¬(A1)：证明完整，无逻辑跳跃。构造的 $g_i$ 为连续函数，满足 (A3a)。
2. 定理覆盖所有包含可加子类的模型空间（Section 5.3）。
3. 边际公式不能推广到非可加模型（Section 5.4 反例，显式分段定义）。
4. Completeness alone 不导致不可能性（Section 5.5 构造，在 $C^1$ 模型类上）。
5. Additivity 是不可能性的核心公理。

**未证明/未覆盖的**:
- 未超越 Bilodeau 的技术框架（证明策略相同：远场对消）
- 未覆盖 Han 的 NFL（涉及 neighborhood-dependent faithfulness）
- 未覆盖 Sutter 的 trivialization（方向相反）
- 未覆盖 Kleinberg 的三公理绝对不相容
- 统一不可能性定理仍是 open conjecture

---

## 6. Open Gaps 与实验设计

### 6.1 Open Gap 1: Additivity Tax 的精确量化

**问题陈述**：Section 5.5 证明了 completeness alone 不阻碍可辨识性，而 additivity 是致命的。但**在 completeness 和 completeness + additivity 之间有多大空间？** 能否精确刻画 additivity 带来的"可辨识性损失"？

**形式化**：定义 $\text{IGA}(\Phi) := \sup_{h_0 \neq h_1, I} [\text{spec}(I) + \text{sens}(I)]$（identifiability gap: 最优推断策略能达到的最大 spec + sens）。

- Section 5.2 证明了 $\Phi$ 满足 (A2) 时 $\text{IGA} \leq 1$。
- Section 5.5 构造了仅满足 (A2a) 且 $\text{IGA} = 2$ 的 $\Phi$（在 $C^1$ 模型类上）。

**Open**：对满足 (A2a) 的 $\Phi$ 施加**不同程度**的 additivity（如在 $M_{\text{add}}$ 的子集上要求可分解），$\text{IGA}$ 如何变化？是否存在 sharp phase transition？

**实验设计 (Experiment 1: Additivity-Identifiability Tradeoff Curve)**

1. **Setup**：取 $d=10$，$\mathcal{X} = [0,1]^{10}$。生成 $N=1000$ 个 two-hidden-layer ReLU 网络作为模型类。固定 $x$, $\mu = \text{Uniform}(\mathcal{X})$, $\delta = 0.1$。

2. **Method family**：参数化归因方法为

$$\Phi^\alpha(m, \mathbf{d})_j = (1-\alpha) \cdot \text{IG}_j(m, x', x) + \alpha \cdot \text{SHAP}_j(m, \mu, x)$$

其中 $\alpha \in [0,1]$，$\text{IG}_j$ 是 Integrated Gradients（以固定 baseline $x'$ 为参考点），$\text{SHAP}_j$ 是 SHAP value。

> **为何使用 IG 而非 vanilla gradient**：Vanilla gradient $\nabla_j m(x)$ **不满足 completeness**——一般地 $\sum_j \frac{\partial m}{\partial x_j}\big|_x \neq m(x) - \mathbb{E}_\mu[m]$（等号仅对线性模型成立）。Integrated Gradients 满足 completeness（由路径积分基本定理：$\sum_j \text{IG}_j = m(x) - m(x')$），因此是一个合法的 $\alpha = 0$ 基准。
>
> **IG 与 SHAP 在 additivity 上的差异**：对可加模型 $m = \sum_j m_j$，SHAP 给出 $\text{SHAP}_j = m_j(x_j) - \mathbb{E}_{\mu_j}[m_j]$，满足 (A2b)（因为 SHAP 是以 baseline 分布 $\mu$ 为参照的边际贡献）。IG 给出 $\text{IG}_j = m_j(x_j) - m_j(x_j')$（以单一 baseline 点 $x'$ 为参照），当 $x' \neq \mathbb{E}_\mu[X]$ 或 $m_j$ 非线性时，这不等于 $m_j(x_j) - \mathbb{E}_{\mu_j}[m_j]$。因此 IG 满足 completeness 但**不**满足 (A2b) 中以 $\mu_j$ 为参照的 additivity。

$\alpha=0$ 是纯 IG（满足 completeness，不满足 additivity），$\alpha=1$ 是 SHAP（满足两者）。

> **归一化注记**：$\Phi^\alpha$ 对每个 $\alpha$ 均满足 completeness，因为 IG 和 SHAP 各自满足 completeness 且二者的凸组合保持加和性质（$\sum_j \Phi^\alpha_j = (1-\alpha)[m(x) - m(x')] + \alpha[m(x) - \mathbb{E}_\mu[m]]$）。若要使两端的 completeness 总量一致，可取 $x' = \mathbb{E}_\mu[X]$ 作为 IG 的 baseline，此时两者的 completeness 总量均为 $m(x) - \mathbb{E}_\mu[m]$。

3. **Metric**：对每个 $\alpha$，估计 $\text{IGA}(\Phi^\alpha)$——对随机生成的 $(h_0, h_1)$ 对，找使 $\Phi$ 值相同的模型对，计算最优 spec + sens。

4. **预测**：
   - $\alpha = 0$: $\text{IGA} > 1$（IG 编码局部路径信息，不被远场完全对消）
   - $\alpha = 1$: $\text{IGA} = 1$（SHAP 的不可能性）
   - 中间 $\alpha$: 应存在 critical $\alpha^*$，低于它 $\text{IGA} > 1$

5. **技术细节**：
   - 估计 IGA 的上界：对每个 $(h_0, h_1)$ 对，搜索 $M_{h_0}$ 和 $M_{h_1}$ 中使 $\|\Phi(m_0) - \Phi(m_1)\|$ 最小的模型对。若最小距离 $> 0$，则 IGA $>1$。
   - 估计 IGA 的下界：使用 Section 5.2 的显式构造，验证 adversarial 可加模型确实使 $\Phi$ 值重合。
   - $\alpha$ 扫描：$[0, 0.1, 0.2, \ldots, 1.0]$，每个点 100 次随机 $(h_0, h_1)$ 对。

---

### 6.2 Open Gap 2: "Squeeze Theorem" 的可行性

**问题陈述**：Section 4.2 提出了 squeeze 图景——Finiteness 太强失去 identifiability，太弱失去 non-triviality。能否严格证明不存在"恰好合适"的中间复杂度？

**形式化 (Conjecture: Finiteness–Non-triviality Squeeze)**：

定义 representation decomposition method $\Phi: M \times D \to F_c$，其中 $c \in \mathbb{R}_{>0}$ 参数化 $F_c$ 的复杂度。

- **Lower bound** (Bilodeau type): $\exists c_0$ s.t. $c < c_0 \implies$ identifiability fails
- **Upper bound** (Sutter type): $\exists c_1$ s.t. $c > c_1 \implies$ non-triviality fails
- **Impossibility**: $c_0 > c_1$ (feasible range empty)

**实验设计 (Experiment 2: Complexity–Faithfulness Phase Diagram)**

在 causal abstraction 语境下直接测试 Sutter 的 dilemma：

1. **Setup**：在 Indirect Object Identification (IOI) task 上训练一个 2-layer transformer。用一个已知的 ground-truth algorithm（"IOI circuit"）作为 $\mathcal{A}$。

2. **Complexity parameterization**：使用不同复杂度的 alignment maps：
   - $c=1$: 线性 probe（affine map）
   - $c=2$: 2-layer MLP
   - $c=3$: 4-layer MLP
   - $c=4$: 8-layer MLP
   - $c=5$: RevNet（可逆网络，接近无约束）

3. **Metrics**：
   - **Faithfulness**: Interchange Intervention Accuracy (IIA) on the correct algorithm
   - **Non-triviality**: IIA on a **random/wrong** algorithm（如果也高，说明 trivial）

4. **预测**（基于 Sutter 的理论）：
   - 低复杂度 ($c=1$): IIA on correct alg < 100%（忠实性不够）
   - 高复杂度 ($c=5$): IIA on correct alg ≈ IIA on wrong alg ≈ 100%（trivial）
   - 中间复杂度：IIA correct $>$ IIA wrong（sweet spot，如果存在）

5. **关键观察**：如果对所有 $c$，要么 IIA correct $<$ threshold 要么 IIA wrong $>$ threshold，则 squeeze conjecture 得到实验支持。

6. **Controls**：
   - 对 randomly initialized（未训练）网络重复实验（Sutter 论文已示有 trivial alignment）
   - 对训练好的网络用不同随机种子重复，检查 alignment map 的稳定性

---

### 6.3 Open Gap 3: Sutter 假设在实际网络中的成立范围

**问题陈述**：Sutter 的定理依赖 5 个假设，特别是 **input-injectivity**（所有层的 input-to-hidden map 是单射）。这在有维度瓶颈的架构中不成立。当假设被违反时，trivialization 是否仍然发生？

**实验设计 (Experiment 3: Injectivity Violation and Trivialization)**

1. **Setup**：训练一系列架构，系统性地违反 injectivity：
   - (a) Standard transformer（各层等宽，injectivity 可能近似成立）
   - (b) Bottleneck transformer（中间层维度 $d_\ell = d/4$，强制信息压缩）
   - (c) Autoencoder 结构（$d \to d/8 \to d$）
   - (d) 带 pooling 的 CNN

2. **Metric**：对每个架构，用 RevNet alignment map 检查：
   - IIA on correct algorithm
   - IIA on random algorithm (trivialization 指标)

3. **预测**：
   - 架构 (a): trivialization 应出现（近似满足 Sutter 假设）
   - 架构 (b)-(d): trivialization 可能减弱（injectivity 破坏使得 hidden states 不再唯一确定 input，Sutter 构造失效）

4. **理论意义**：如果 bottleneck 确实减弱了 trivialization，这暗示信息压缩是一种自然的"finiteness"约束，可能是 squeeze theorem 中 $c_1$ 的具体实例化。

---

### 6.4 Open Gap 4: SAE 非唯一性是否是不可能性的实证表现

**问题陈述**：Paulo & Belrose (arXiv:2501.16615) 报告 SAE 的 131K latents 中仅约 30% 在不同种子间共享。这是否可以理论性地联系到 Bilodeau 型不可能性？

**假说**：SAE 训练本质上是一个 additive decomposition（将 activation 分解为 $\sum_k c_k \mathbf{d}_k$，字典 $\mathbf{d}_k$ 类似于可加分量）。如果 SAE 的字典学习类似于 complete + additive attribution，则 Bilodeau 型不可能性预测：**字典方向不被激活模型的局部行为唯一确定**——不同的远场（训练数据分布的不同子集）可以导向不同的字典，即使局部行为（模型在某个子空间上的行为）相同。

**实验设计 (Experiment 4: Adversarial Seed Divergence)**

1. **Setup**：在 GPT-2 small 的某个中间层训练 10 个 SAE（不同随机种子），取 $k=4096$ latents。

2. **Reproduce Paulo & Belrose**：计算 pairwise feature matching（cosine similarity > 0.9 为 "shared"），确认 ~30% shared rate。

3. **关键新分析——连接到 Bilodeau 构造**：

   (a) **远场依赖测试**：对每个 non-shared feature $\mathbf{d}_k$（仅出现在某些种子中），检查它在训练过程中被什么数据"塑造"。具体地，计算 $\mathbf{d}_k$ 的 top-activating examples，然后检查这些 examples 是否集中在特定的数据子分布上。

   (b) **对消构造验证**：取两个 SAE（种子 $s_1$, $s_2$），找一个 non-shared feature $\mathbf{d}_{k_1}^{(s_1)}$（出现在 $s_1$ 但不在 $s_2$）。构造一个 adversarial activation $\mathbf{a}$ 使得：
   - $\mathbf{a}$ 在 $\mathbf{d}_{k_1}^{(s_1)}$ 方向上有强投影
   - $\mathbf{a}$ 在 $s_2$ 的所有 features 上的投影模式完全不同
   - 但两个 SAE 对 $\mathbf{a}$ 的重构误差相当

   如果这样的 $\mathbf{a}$ 容易找到，说明 SAE 分解确实类似于"远场对消"现象。

4. **量化联系**：计算一个 "identifiability gap" 指标——对 $S$ 个种子的 SAE，同一个 activation 能被多少种不同的 feature 组合同等好地解释。

---

### 6.5 Open Gap 5: 超越 Bilodeau 的统一不可能性

**问题陈述**：当前证明本质上是 Bilodeau 的简化版。真正的统一不可能性需要同时覆盖 Han 的 NFL、Sutter 的 trivialization、和 Kleinberg 的三公理不相容。

**需要回答的理论问题**：

1. **Han 的 NFL 能否纳入 additivity 框架？** Han 的 NFL 适用于所有 LFA 方法（不仅仅是 additive 的），其不可能性来源是 $f \notin \mathcal{G}$（approximation gap），而非 additivity。这是否意味着 Han 的不可能性是**独立于** Bilodeau 的？还是存在更深层的统一？

   **Tentative answer**: 二者的不可能性源自不同机制。Bilodeau: 输出结构（additivity）导致信息丢失。Han: 模型复杂度 gap ($f \notin \mathcal{G}$) 导致邻域间的 faithfulness 冲突。统一它们可能需要一个同时参数化"输出结构约束"和"模型-解释复杂度 gap"的框架。

2. **Sutter 的 trivialization 能否视为 "anti-Bilodeau"？** Bilodeau: 约束太强 → 信息丢失。Sutter: 约束太弱 → 信息冗余。这暗示了一个 **duality**——能否形式化这个对偶？

   **Candidate formalization**: 定义 $\text{Info}(\Phi) := $ $\Phi$ 传递的关于 $B$ 的互信息（在某个先验下）。Bilodeau 证明当 $F$ "太小" 时 $\text{Info} = 0$；Sutter 证明当 $F$ "太大" 时 $\text{Info} = H(B)$ 但 $\Phi$ 不区分正确/错误的 $B$。统一不可能性 = $\nexists F$ 使得 $0 < \text{Info} < H(B)$ 且 Info 聚焦在正确的 $B$ 上。这本质上就是 Squeeze Conjecture 的信息论版本。

3. **Kleinberg 是否有独立的推广路径？** Kleinberg 的证明不依赖"远场对消"或"维度间隙"，而是纯组合性的（三条公理在有限集上不相容）。将其推广到连续空间的 feature discovery 需要找到类似的组合不可能性论证。

   **Possible approach**: 将 activation space discretize 为有限 partition（如 k-means with $k$ clusters），将 scale-invariance 翻译为 "rescaling activations doesn't change feature assignment"，richness 翻译为 "any feature assignment is achievable"，consistency 翻译为 "strengthening within-feature coherence doesn't change assignment"。在此离散化下 Kleinberg 直接适用，但需要论证离散化过程不引入 artifact。

---

### 6.6 实验优先级排序

| 实验 | 难度 | 理论价值 | 实践价值 | 建议优先级 |
|------|------|----------|----------|------------|
| Exp 1: Additivity–Identifiability Tradeoff | 中 | 高——验证 additivity 是否是唯一来源 | 高——指导归因方法选择 | **1** |
| Exp 2: Complexity–Faithfulness Phase Diagram | 高 | 极高——直接检验 squeeze conjecture | 高——指导 mech interp 方法学 | **2** |
| Exp 4: SAE Seed Divergence Analysis | 中 | 高——连接理论到 SAE 实践 | 极高——直接回应 SAE 可靠性争议 | **2** |
| Exp 3: Injectivity Violation | 中 | 中——测试 Sutter 假设边界 | 中——影响架构选择 | **3** |

---

## 7. 下一步建议

### 7.1 短期（1-2 个月）

1. **执行 Experiment 1 和 4**——这两个实验可以在现有基础设施上完成（标准 PyTorch + SAE 训练代码），且直接检验核心理论预测。

2. **精确定义 "representation decomposition method" for neural networks**
   - 形式化 $\Phi: \text{NN}(d_{\text{in}}, d_{\text{out}}) \times \text{Data}(\mathcal{X}) \to \text{Dict}(k, d)$
   - 其中 $\text{Dict}(k, d)$ 是 $k$ 个 $d$ 维特征向量的字典
   - 涵盖 SAE, PCA, sparse coding, concept vectors 等
   - 定义 "行为" $B$ 为模型的 causal/interventional behavior

3. **写出 Section 5.5 的完整推广**——将 "completeness alone is fine" 从 $d=2$ 构造推广到一般 $d$，并严格刻画哪些 $\Phi$ 结构（介于 completeness 和 completeness + additivity 之间）是安全的。

### 7.2 中期（3-6 个月）

4. **执行 Experiment 2 和 3**——需要较多 GPU 资源（训练 transformer + 各种 alignment maps）。

5. **验证 Squeeze Conjecture (Approach A)**
   - 对 $F_c$ 的复杂度做精确参数化（如字典大小 $k$、alignment map 的 Lipschitz 常数）
   - 用 Bilodeau 的技术证明 lower bound $c_0$
   - 用 Sutter 的技术证明 upper bound $c_1$
   - 关键问题：$c_0$ 和 $c_1$ 是否依赖于 $M$ 的具体结构？

6. **Category-theoretic formulation**
   - 将 $\Phi$ 视为 functor from "model category" to "feature category"
   - Identifiability = faithful functor; Structural Faithfulness = full functor
   - Non-triviality = non-trivial image
   - 不可能性 = 不存在同时 full and faithful 的 functor
   - 参考 Englberger & Dhami (arXiv:2510.05033) 的 Markov category 框架

### 7.3 长期目标

7. **完整的不可能性定理**
   - 五个公理 {Identifiability, Structural Faithfulness, Generality, Non-triviality, Finiteness}
   - 证明 **no measurable** $\Phi$ 满足全部五个
   - 或者证明至多满足其中 $k$ 个（确定 $k$）
   - 对不同的 "表征分析任务类"（attribution, decomposition, abstraction, clustering），确定各自的不可能性 frontier

8. **正面结果：可能性定理**
   - 如果放松 additivity，最优方法是什么？（Section 5.5 暗示梯度类方法是候选）
   - 如果放松 generality（限制为特定模型类），不可能性是否消失？
   - 类比 Arrow 定理后的 mechanism design：一旦知道不可能，可以设计 "次优但最好" 的方法
   - 实际指导：多方法 portfolio 的理论最优组合

---

## 附录 A: 技术细节补充

### A.1 Bilodeau 证明的关键引理

**Interior Freedom Lemma (Theorem A.2 in paper):**

> Fix $x, j, \delta, \mu$, local behavior $h$. Under Assumptions 1 & 2, for every $v \in \mathbb{R}^{d \times m}$, there exists $m \in M$ such that:
> 1. $\Phi(m, \mu, x)_{[d]} = v$ for any complete and linear $\Phi$（此处 $[d] = \{1, \ldots, d\}$ 表示全部特征分量）
> 2. $m(z) = h(z_j)$ for all $z$ with $|z_j - x_j| \leq \delta$

即：可以在保持局部行为不变的前提下，让任何 complete + linear 方法输出任意预设值。

这个引理的本质是 **completeness 和 linearity 一起构成了一个 under-determined 线性系统**: 方程数（completeness 提供 1 个约束/feature）少于自由度数（far-field 行为的维度），因此解不唯一。

### A.2 Sutter 证明的关键步骤

**逐层构造**: 对每一层 $\ell$：

1. 由 injectivity，input-restricted hidden states 是 countable set（因为 $\mathcal{X}$ countable）
2. 由 surjectivity，对每个目标输出存在 hidden state 可以产生它
3. 对 countable 个 input-restricted interventions，逐一定义 $\phi_\ell$ 的值使得 interventional consistency 成立
4. 由 hidden state space 是 uncountable（$\mathbb{R}^{d_\ell}$）而 constraints 是 countable 的，总能将 $\phi_\ell$ 扩展为整个空间上的 bijection

关键：**countable constraints in uncountable space** 给了足够的自由度。这与 Bilodeau 的 "under-determined linear system" 是同一种 **capacity surplus** 的体现。

### A.3 共同的数学机制

四篇论文的不可能性都源于同一个根本原因：

> **The method's output is constrained (by axioms) to live in a space of lower effective dimension than the space of possible model behaviors.**

- Bilodeau: attributions ∈ $\mathbb{R}^d$, but local behaviors ∈ function space (∞-dim)
- Han: linear models ∈ $\mathbb{R}^d$, but $f$ ∈ larger function space
- Sutter: alignment maps ∈ uncountable function space, but constraints are countable → inverse direction, too many maps
- Kleinberg: partition space is finite, but the constraints (3 axioms) over-determine the system

**关于 $\dim_\text{eff}$ 的严格化建议**: 上述叙述使用了未定义的 "有效维度" $\dim_\text{eff}$。若要将其发展为定量结果（如 Squeeze Conjecture），需要为不同对象选择具体度量：

| 对象类型 | 建议度量 |
|----------|----------|
| 有限维向量空间 | 线性代数维度 $\dim(V)$ |
| 函数空间 / 模型类 | VC 维 / Rademacher 复杂度 / metric entropy |
| 离散 / 组合对象 | $\log_2 |\cdot|$ (bits) |
| Countable vs. uncountable 对比 | 基数 $|\cdot|$ |

不同选择导致不同的 "维度间隙" 解释。Squeeze Conjecture 最可能需要 metric entropy 或 Rademacher 复杂度，因为它们同时适用于有限维和无穷维空间。

---

## 附录 B: 参考文献

1. Bilodeau, Jaques, Koh, Kim. *Impossibility Theorems for Feature Attribution.* arXiv:2212.11870. PNAS 2024.
2. Han, Srinivas, Lakkaraju. *Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post Hoc Explanations.* arXiv:2206.01254. NeurIPS 2022.
3. Sutter, Minder, Hofmann, Pimentel. *The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?* arXiv:2507.08802. NeurIPS 2025 Spotlight.
4. Kleinberg. *An Impossibility Theorem for Clustering.* NeurIPS 2002. pp. 463–470.
5. Tang, Meng, Gandelsman. *A Unified Theory of Sparse Dictionary Learning.* arXiv:2512.05534. 2025.
6. Paulo, Belrose. *How SAE Features are Not Reproducible.* arXiv:2501.16615. 2025.
7. Geiger et al. *Causal Abstraction for Faithful Model Interpretation.* arXiv:2301.04709. 2023.
8. Klabunde et al. *Similarity of Neural Network Representations Revisited.* arXiv:2305.06329. ACM Computing Surveys 2025.
9. Englberger, Dhami. *A Category-Theoretic Framework for Causal Abstraction.* arXiv:2510.05033. 2025.
10. Arrow. *Social Choice and Individual Values.* 1951.

---

## 附录 C: 修正日志

| 版本 | 日期 | 修正内容 |
|------|------|----------|
| v1 | 2026-05-17 | 初稿 |
| v2 | 2026-05-17 | §1.2: 勘误 $0 \to x_0$（三处）。§3.1: 重组公理——区分 Identifiability 与 Structural Faithfulness。§5.1: (A1) 从 $\Pr_{\text{Uniform}}$ 改为 $\inf$；(A2) 拆分为 (A2a)+(A2b) 并标注适用范围；(A3) 弱化为 additive richness + baseline non-degeneracy。§5.2: 证明完全限制在可加模型内，显式处理常数项 $C_j$，给出远场调节的闭式构造。§5.3-5.5: 新增——覆盖范围分析、边际公式反例、根源诊断（additivity 而非 completeness）。§6: 新增——Open Gaps 与实验设计。§A.3: 新增 $\dim_\text{eff}$ 严格化建议。|
| v3 | 2026-05-17 | **Fix 1 (§5.2 Step 2 连续性)**：将 $g_i$ 从分段常数改为连续分段线性函数，在 $x_j \pm \delta$、$a_j$、$b_j$ 处引入宽度 $\eta$ 的线性过渡带，使 $g_i \in C^0(\mathbb{R})$，严格满足 (A3a)。$\lambda_i$ 的选取公式修改为补偿过渡带积分贡献 $R_i(\eta)$，并说明 $\eta$ 足够小时 $p_\eta > 0$ 仍成立。**Fix 2 (§6.1 Experiment 1)**：$\alpha = 0$ 的基准方法从 vanilla gradient 改为 Integrated Gradients (IG)。新增注记解释 IG 满足 completeness 的原因（路径积分基本定理），以及 IG 与 SHAP 在 additivity 上的差异（单一 baseline $x'$ vs. baseline 分布 $\mu$）。**Fix 3 (§5.5 可微性)**：显式声明模型类限制为 $M \cap C^1(\mathcal{X})$，$h_0, h_1$ 限制为可微且 $h_0'(x_1) \neq h_1'(x_1)$。注明 (A3a) 从 "连续" 变为 "$C^1$"，不影响定性结论但限制了定量适用范围。**Fix 4 (§5.4 反例)**：$\Phi$ 改为显式的 $\begin{cases} \ldots & m \in M_{\text{add}} \\ \ldots & m \notin M_{\text{add}} \end{cases}$ 分段定义，消除歧义。|
