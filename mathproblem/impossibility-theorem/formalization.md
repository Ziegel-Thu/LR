# 表征分析的不可能性定理：公理化前期工作

> 探索性工作，2026-05-17

---

## 目录

1. [四篇论文的精确定理提取](#1-四篇论文的精确定理提取)
2. [统一数学框架](#2-统一数学框架)
3. [公理系统草稿](#3-公理系统草稿)
4. [已知的不相容对](#4-已知的不相容对)
5. [Mini Impossibility 尝试](#5-mini-impossibility-尝试)
6. [下一步建议](#6-下一步建议)

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

$$d(f, \mathcal{G}) = \min_{g \in \mathcal{G}} \max_{x \in \mathcal{X}} \ell(f, g, 0, x)$$

$f$ 到可解释模型类 $\mathcal{G}$ 的"距离"，灵感来自 Hausdorff 距离。

**NFL 定理 (Theorem 1)**:

> Consider explaining $f$ around $x_0$ using $g \in \mathcal{G}$ and loss $\ell$, where $d(f, \mathcal{G}) = \min_{g \in \mathcal{G}} \max_{x \in \mathcal{X}} \ell(f, g, 0, x)$.
>
> Then, for any explanation $g^*$ on neighborhood $\xi_1 \sim \mathcal{Z}_1$ satisfying $\max_{\xi_1} \ell(f, g^*, x_0, \xi_1) \leq \epsilon$, there always exists another neighborhood $\xi_2 \sim \mathcal{Z}_2$ such that:
>
> $$\max_{\xi_2} \ell(f, g^*, x_0, \xi_2) \geq d(f, \mathcal{G})$$

**不可能性的精确含义**: 当 $f \notin \mathcal{G}$（即 $d(f, \mathcal{G}) > 0$），任何在某个邻域上表现好的解释必然在另一个邻域上表现差，误差下界为 $d(f, \mathcal{G})$。不存在在所有邻域上同时最优的解释方法。

**证明机制**: 对任意 $g^*$，构造 adversarial input $x_{\text{adv}} = \arg\max_x \ell(f, g^*, 0, x)$，然后构造包含此点的邻域 $\mathcal{Z}_2 = \text{Uniform}(x_0, x_{\text{adv}})$。由定义：
$$\max_{\xi_2} \ell(f, g^*, x_0, \xi_2) = \ell(f, g^*, 0, x_{\text{adv}}) = \max_x \ell(f, g^*, 0, x) \geq d(f, \mathcal{G})$$

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

## 3. 公理系统草稿

### 3.1 五个候选公理

在统一框架 $\Phi: M \times D \to F$ 下，我们提出以下五个公理方向：

---

**Axiom 1: Faithfulness（忠实性）**

$\Phi(m, d)$ 忠实地反映 $m$ 在上下文 $d$ 下的行为。形式化为：存在 inference map $I: F \to O$ 使得

$$I(\Phi(m, d)) = B(m, d) \quad \forall (m, d) \in M \times D$$

等价地（弱版本）：若 $B(m_1, d) \neq B(m_2, d)$，则 $\Phi(m_1, d) \neq \Phi(m_2, d)$。

**各论文中的实例化**:
- Bilodeau: spec + sens > 1（比 random guessing 好）
- Han: $\ell(f, g^*, x_0, \xi) \leq \epsilon$ 对所有 $\xi \in \mathcal{Z}$
- Sutter: IIA = 100% 且 non-trivial（非所有 pair 都 align）
- Kleinberg: Consistency（加强 cluster structure 后结果不变）

---

**Axiom 2: Identifiability（可辨识性）**

不同行为的模型产生不同的特征分解。

$$B(m_1, d) \neq B(m_2, d) \implies \Phi(m_1, d) \neq \Phi(m_2, d)$$

这是 Faithfulness 的一个更强形式。在 Bilodeau 的框架中，这对应于 hypothesis test 的 discriminating power。

**与 faithfulness 的区别**: Faithfulness 要求 $\Phi$ 的输出可以被用来推断 $B$；Identifiability 要求 $\Phi$ 在 behavioral equivalence classes 上是单射。Faithfulness 还要求输出的"格式"适合推断（如 attributions 的和等于输出差）。

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
- **Method structure**: $\Phi$ 本身满足结构约束（完备性、可加性、scale-invariance）

**各论文中的实例化**:
- Bilodeau: Completeness + Linearity（attributions 满足加法和对消结构）
- Han: $\mathcal{G}$ 是线性模型类（比 $f$ 简单）
- Sutter: **反面**——当不施加 finiteness（允许任意复杂的 alignment map）时，faithfulness 变得 trivial
- Kleinberg: Scale-invariance（输出不依赖于距离的绝对尺度）

---

### 3.2 公理间的逻辑关系

```
            Faithfulness ←→ Identifiability（强版本）
                ↕                    ↕
           Generality    ←→     Non-triviality
                ↕                    ↕
           Finiteness    ←→    (provides structure)
```

**关键张力**:
- Faithfulness + Finiteness + Generality → ¬Identifiability (Bilodeau)
- Faithfulness + Generality → ¬Universality-across-neighborhoods (Han)
- ¬Finiteness + Generality → ¬Non-triviality (Sutter)
- Scale-invariance + Richness + Consistency → ⊥ (Kleinberg)

---

## 4. 已知的不相容对

### 4.1 从四篇论文提取的不相容关系

**不相容性 I**: {Faithfulness (completeness), Finiteness (linearity), Generality (rich models)} → ¬Identifiability

- **来源**: Bilodeau et al.
- **含义**: Complete + linear attributions 无法区分不同局部行为
- **形式**: spec + sens ≤ 1 (random guessing)
- **覆盖的方法**: SHAP, Integrated Gradients

**不相容性 II**: {Faithfulness (local accuracy), Generality (all neighborhoods)} → bounded failure

- **来源**: Han et al.
- **含义**: 在一个邻域上准确的解释在另一个邻域上误差不小于 $d(f, \mathcal{G})$
- **前提**: $f \notin \mathcal{G}$（被解释的函数不在可解释模型类中）
- **覆盖的方法**: 所有 LFA 类方法（LIME, SHAP, SmoothGrad, ...）

**不相容性 III**: {Faithfulness (interventional consistency), Generality (any alg-DNN pair), ¬Finiteness (unrestricted maps)} → ¬Non-triviality

- **来源**: Sutter et al.
- **含义**: 无约束的 causal abstraction 是 vacuous 的
- **等价表述**: 如果同时要求 faithfulness, generality, 和 non-triviality，则必须施加 finiteness (如 linearity constraint on alignment maps)
- **覆盖的方法**: DAS, causal abstraction (无线性约束时)

**不相容性 IV**: {Scale-Invariance, Richness, Consistency} → ⊥

- **来源**: Kleinberg
- **含义**: 三条公理不可能同时满足
- **映射到统一框架**: {Finiteness (scale-inv), Generality (richness), Faithfulness (consistency)} → ⊥
- **覆盖的方法**: 所有 clustering functions

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
| Faithfulness + Finiteness + Generality | Bilodeau | ¬Identifiability |
| Faithfulness + Context-universality | Han | ¬(同时 faithful everywhere) |
| Faithfulness + Generality + (¬Finiteness) | Sutter | ¬Non-triviality |
| Finiteness + Generality + Faithfulness | Kleinberg | ⊥ (不可能) |
| Faithfulness + Identifiability + Generality | *conjectured* | ¬Finiteness (需要无界复杂度) |
| Identifiability + Finiteness + Generality | *conjectured* | ¬Faithfulness |

---

## 5. Mini Impossibility 尝试

### 5.1 定理陈述

**Theorem (Mini Impossibility for Representation Decomposition — Attempt).**

**Setting.** Let $\mathcal{X} = \mathbb{R}^d$, $\mathcal{Y} = \mathbb{R}$. Let $M$ be a model class $M \subseteq (\mathcal{X} \to \mathcal{Y})$. Fix a "context" $d = (\mu, x, \delta)$ where $\mu \in \mathcal{P}(\mathcal{X})$ is a baseline distribution, $x \in \mathcal{X}$ is an observation, and $\delta > 0$ defines a neighborhood $\mathcal{N}_j = [x_j - \delta, x_j + \delta]$ for each feature $j \in [d]$.

Let $\Phi: M \times D \to F$ be a representation decomposition method.

**Axioms:**

**(A1) Faithfulness.** $\Phi(m, d)$ captures the local behavior of $m$ near $x$: there exists an inference function $I: F \to \{0, 1\}$ such that for any pair of distinguishable behaviors $b_0, b_1$:

$$\Pr_{m \sim \text{Uniform}(M_{b_0})}[I(\Phi(m, d)) = 0] + \Pr_{m \sim \text{Uniform}(M_{b_1})}[I(\Phi(m, d)) = 1] > 1$$

(i.e., better than random guessing)

**(A2) Structural Decomposability.** $\Phi$ decomposes additively over features: if $m(x) = \sum_j m_j(x_j)$ is additive, then $\Phi(m, d)$ decomposes as $\Phi(m, d)_j = \phi(m_j, d_j)$ for some univariate analysis $\phi$.

Additionally, $\Phi$ satisfies a bookkeeping constraint: $\sum_j \Phi(m, d)_j = m(x) - \mathbb{E}_\mu[m]$.

**(A3) Model Richness.** For any local behavior $h: \mathcal{N}_j \to \mathcal{Y}$ and any "far-field" piecewise linear function on $\mathcal{N}_j^c$, the combined function is in $M$. (Satisfied by any ReLU network class of sufficient width/depth.)

**Claim**: (A1) + (A2) + (A3) are jointly unsatisfiable.

### 5.2 证明草稿

**Proof (adapting Bilodeau et al.).**

我们展示 (A2) + (A3) → ¬(A1)。

**Step 1: Attributions are determined by far-field behavior.**

Fix feature $j$ and consider an additive model $m(x) = m_j(x_j) + m_{-j}(x_{-j})$ where $m_{-j} \equiv 0$. By (A2):

$$\Phi(m, d)_j = \phi(m_j, d_j)$$

By the bookkeeping constraint in (A2):

$$\phi(m_j, d_j) = m_j(x_j) - \mathbb{E}_{\mu_j}[m_j]$$

Now, consider $m_j$ restricted to a general (not necessarily additive) model. By (A2) (decomposability + completeness):

$$\Phi(m, d)_j = m_j(x_j) - \mathbb{E}_{\mu_j}[m_j(X_j')]$$

The attribution for feature $j$ depends on $m$ only through the **marginal expectation** $\mathbb{E}_{\mu_j}[m_j]$, which integrates over the entire support of $\mu_j$——包括邻域外的区域。

**Step 2: Local behavior is decoupled from attributions.**

By (A3), for any fixed local behavior $h$ on $\mathcal{N}_j$ and any desired attribution value $v$, we can construct a model $m \in M$ with:
- $m|_{\mathcal{N}_j} = h$（局部行为固定）
- $\mathbb{E}_{\mu_j}[m_j]$ 取任意值（通过调节 $\mathcal{N}_j$ 外的行为）

具体构造：设 $\mu_j$ 在 $\mathcal{N}_j$ 外的某个区间 $(a, b)$ 上有正测度（由 baseline non-degeneracy 保证）。定义 $m$ 在 $(a, b)$ 上为斜率 $\lambda$ 的线性函数。调节 $\lambda$ 可以让 $\mathbb{E}_{\mu_j}[m_j]$ 取任意实数值，从而让 $\Phi(m, d)_j$ 取任意值。

**Step 3: Impossibility of faithful inference.**

取两个不同的局部行为 $h_0 \neq h_1$。

对 $h_0$: 由 Step 2，存在 $m_0 \in M_{h_0}$ 使得 $\Phi(m_0, d) = \mathbf{v}$（任意预设值）。

对 $h_1$: 同理，存在 $m_1 \in M_{h_1}$ 使得 $\Phi(m_1, d) = \mathbf{v}$（相同的预设值）。

则 $\Phi(m_0, d) = \Phi(m_1, d) = \mathbf{v}$，但 $B(m_0, d) \neq B(m_1, d)$。

对任意 inference function $I$:

$$\text{spec}(I) + \text{sens}(I) = [1 - I(\mathbf{v})] + I(\mathbf{v}) = 1$$

这恰好是 random guessing 的水平。$\square$

### 5.3 评估

**成功的部分**:
- 上述证明是完整的，在 (A2) + (A3) 的特定实例化下确实给出了不可能性
- 覆盖了 Bilodeau 的核心结果（complete + linear → can't beat random guessing）
- 证明思路清晰：structural decomposability 使得 attributions 受远场行为控制，而 model richness 使得远场行为可以自由调节

**失败/不足的部分**:
- **未超越 Bilodeau**: 上述证明本质上是 Bilodeau et al. 的简化版，没有实质性地推广
- **未覆盖 Han**: Han 的 NFL 涉及不同的结构——neighborhood-dependent faithfulness，而非 completeness + linearity 的对消
- **未覆盖 Sutter**: Sutter 的 trivialization 方向相反（过于表达 vs. 表达不足）
- **未覆盖 Kleinberg**: Kleinberg 的三公理（scale-invariance, richness, consistency）不直接映射到 completeness + linearity

**核心卡点**:

1. **两种不可能性方向不统一**: Bilodeau/Han/Kleinberg 是 "finiteness → ¬faithfulness" 方向，Sutter 是 "¬finiteness → ¬non-triviality" 方向。要统一它们，需要证明 **不存在 "恰好合适" 的中间复杂度**——这需要对 $F$ 的复杂度做更精细的参数化。

2. **公理不够 uniform**: 四篇论文的 "faithfulness" 含义不同——Bilodeau 是 completeness 约束，Han 是 local approximation accuracy，Sutter 是 interventional consistency，Kleinberg 是 consistency under strengthening。统一需要找到一个覆盖所有这些的 meta-axiom。

3. **缺少 "representation decomposition" 的内在公理**: 当前公理更多是 post-hoc explanation 的公理。真正的 "表征分解"（如 SAE、PID）需要额外的结构——如字典的稀疏性、latent code 的统计独立性——这些在四篇论文中没有直接出现。

### 5.4 更有希望的 Mini Impossibility 方向

**Approach A: Squeeze Theorem**

**Conjecture (Finiteness-Non-triviality Squeeze).** 定义 representation decomposition method $\Phi: M \times D \to F_c$，其中 $c \in \mathbb{R}_{>0}$ 参数化 $F_c$ 的复杂度（如 alignment map 的参数数量、字典的大小、模型类的 VC 维）。

- **Lower bound** (from Bilodeau/Kleinberg type): $\exists c_0 > 0$ such that if $c < c_0$, then $\Phi$ cannot be faithful (identifiability fails)
- **Upper bound** (from Sutter type): $\exists c_1 < \infty$ such that if $c > c_1$, then $\Phi$ is trivial (non-triviality fails)
- **Impossibility**: Prove $c_0 > c_1$, i.e., the feasible range is empty

**Approach B: Axiomatic (Kleinberg style)**

**Conjecture.** No representation decomposition method $\Phi: M \to F$ (for $M$ = neural networks, $F$ = "feature decompositions") can simultaneously satisfy:

1. **Reconstruction Faithfulness**: $m$ can be approximately recovered from $\Phi(m)$
2. **Identifiability under re-training**: If $m_1, m_2$ implement the same function but with different weights, then $\Phi(m_1) \sim \Phi(m_2)$ (up to symmetries)
3. **Feature Universality**: For any conceivable "feature" (linear direction, circular representation, manifold), $F$ can express it
4. **Decomposition Consistency**: Strengthening the "separation" between features in $m$ does not change $\Phi(m)$

这类似于 Kleinberg 的三公理框架，但专门针对 neural network representation decomposition。

**Challenge**: 需要严格定义每个公理在 neural network representation 语境下的含义。

---

## 6. 下一步建议

### 6.1 短期（1-2 个月）

1. **精确定义 "representation decomposition method" for neural networks**
   - 形式化 $\Phi: \text{NN}(d_{\text{in}}, d_{\text{out}}) \times \text{Data}(\mathcal{X}) \to \text{Dict}(k, d)$
   - 其中 $\text{Dict}(k, d)$ 是 $k$ 个 $d$ 维特征向量的字典
   - 涵盖 SAE, PCA, sparse coding, concept vectors 等
   - 定义 "行为" $B$ 为模型的 causal/interventional behavior

2. **验证 Approach A (Squeeze Theorem)**
   - 对 $F_c$ 的复杂度做精确参数化（如字典大小 $k$、alignment map 的 Lipschitz 常数）
   - 用 Bilodeau 的技术证明 lower bound $c_0$
   - 用 Sutter 的技术证明 upper bound $c_1$
   - 关键问题：$c_0$ 和 $c_1$ 是否依赖于 $M$ 的具体结构？

3. **研究 Kleinberg 公理的直接推广**
   - 将 scale-invariance, richness, consistency 直接翻译到 activation space 中的 feature discovery
   - Distance function → activation similarity; Partition → feature assignment; Consistency → robustness to feature strengthening

### 6.2 中期（3-6 个月）

4. **Category-theoretic formulation**
   - 将 $\Phi$ 视为 functor from "model category" to "feature category"
   - Faithfulness = full functor; Identifiability = faithful functor
   - Non-triviality = non-trivial image
   - 不可能性 = 不存在同时 full and faithful 的 functor
   - 参考 Englberger & Dhami (arXiv:2510.05033) 的 Markov category 框架

5. **连接 SAE 非唯一性**
   - Paulo & Belrose (arXiv:2501.16615) 的 seed-dependence 结果（131K latents 中仅 30% shared across seeds）
   - 这是否是 Identifiability 不可能性的实证证据？
   - Tang et al. (arXiv:2512.05534) 的 biconvex non-identifiability 是否可以严格转化为公理性不可能？

### 6.3 长期目标

6. **完整的不可能性定理**
   - 五个公理 {Faithfulness, Identifiability, Generality, Non-triviality, Finiteness}
   - 证明 **no measurable** $\Phi$ 满足全部五个
   - 或者证明至多满足其中 $k$ 个（确定 $k$）
   - 对不同的 "表征分析任务类"（attribution, decomposition, abstraction, clustering），确定各自的不可能性 frontier

7. **正面结果：可能性定理**
   - 如果放松一个公理，最优方法是什么？
   - 类比 Arrow 定理后的 mechanism design：一旦知道不可能，可以设计 "次优但最好" 的方法
   - 实际指导：多方法 portfolio 的理论最优组合

---

## 附录 A: 技术细节补充

### A.1 Bilodeau 证明的关键引理

**Interior Freedom Lemma (Theorem A.2 in paper):**

> Fix $x, j, \delta, \mu$, local behavior $h$. Under Assumptions 1 & 2, for every $v \in \mathbb{R}^{d \times m}$, there exists $m \in M$ such that:
> 1. $\Phi(m, \mu, x)_{[J]} = v$ for any complete and linear $\Phi$
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

**统一表述**: 令 $\dim_{\text{eff}}(F)$ 为输出空间的有效维度，$\dim_{\text{eff}}(\text{constraints})$ 为公理施加的约束数量，$\dim_{\text{eff}}(B)$ 为行为空间的维度。

- $\dim_{\text{eff}}(F) < \dim_{\text{eff}}(B)$ 且 constraints 不足以唯一确定 $\Phi$: **identification failure** (Bilodeau, Han)
- $\dim_{\text{eff}}(F) > \dim_{\text{eff}}(\text{constraints})$: **trivialization** (Sutter)
- 公理相互矛盾（no feasible $\Phi$）: **absolute impossibility** (Kleinberg)

---

## 附录 B: 参考文献

1. Bilodeau, Jaques, Koh, Kim. *Impossibility Theorems for Feature Attribution.* arXiv:2212.11870. PNAS 2024.
2. Han, Srinivas, Lakkaraju. *Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post Hoc Explanations.* arXiv:2206.01254. NeurIPS 2022.
3. Sutter, Ryser, Pimentel. *The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?* arXiv:2507.08802. NeurIPS 2025 Spotlight.
4. Kleinberg. *An Impossibility Theorem for Clustering.* NeurIPS 2002. pp. 463–470.
5. Tang, Meng, Gandelsman. *A Unified Theory of Sparse Dictionary Learning.* arXiv:2512.05534. 2025.
6. Paulo, Belrose. *How SAE Features are Not Reproducible.* arXiv:2501.16615. 2025.
7. Geiger et al. *Causal Abstraction for Faithful Model Interpretation.* arXiv:2301.04709. 2023.
8. Klabunde et al. *Similarity of Neural Network Representations Revisited.* arXiv:2305.06329. ACM Computing Surveys 2025.
9. Englberger, Dhami. *A Category-Theoretic Framework for Causal Abstraction.* arXiv:2510.05033. 2025.
10. Arrow. *Social Choice and Individual Values.* 1951.
