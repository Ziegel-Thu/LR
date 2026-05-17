# 不可能性/理论基础分支：综合方法论分析

> 2025-07-18 生成

---

## 目录

- [A–D. 逐篇分析](#a-d-逐篇分析)
  - [Paper 1: Bilodeau et al. (2024)](#paper-1-bilodeau-et-al-2024)
  - [Paper 2: Locatello et al. (2019)](#paper-2-locatello-et-al-2019)
  - [Paper 3: Sutter et al. (2025)](#paper-3-sutter-et-al-2025)
- [E. 三篇不可能性结果的共同结构](#e-三篇不可能性结果的共同结构)
- [F. 我们的 5-axiom 框架与各论文的映射](#f-我们的-5-axiom-框架与各论文的映射)
- [G. Worst-case vs Average-case gap](#g-worst-case-vs-average-case-gap)
- [H. 放松公理获得正面结果](#h-放松公理获得正面结果)
- [I. "Everyone does X but nobody does Y"](#i-everyone-does-x-but-nobody-does-y)

---

## A–D. 逐篇分析

### Paper 1: Bilodeau et al. (2024)

**Impossibility Theorems for Feature Attribution** (PNAS 2024, arXiv:2212.11870)

#### A. 精确定理陈述

**Main Theorem (Theorem 1).** 固定 $x \in \mathcal{X}$, $j \in [d]$, $\delta > 0$, baseline $\mu \in \mathcal{P}(\mathcal{X})$, 两个局部行为 $h_0, h_1: [x_j - \delta, x_j + \delta] \to \mathbb{R}$。令

$$\mathcal{F}_i = \{f \in \mathcal{F} : \forall z \in \mathcal{N}_j,\, f(z) = h_i(z_j)\}, \quad i \in \{0,1\}$$

在 Assumptions 1–2 下，对**任何** complete + additive 归因方法 $\Phi$ 和**任何**假设检验 $T: \mathbb{R}^{d \times m} \to [0,1]$：

$$\text{specificity}_{\Phi,\mu,x}(T) + \text{sensitivity}_{\Phi,\mu,x}(T) \leq 1$$

即推断局部行为的能力**不优于随机猜测**。

**Interior-freedom theorem.** 对任意目标 attribution $v \in \mathbb{R}^{|S| \times m}$ 和任意局部行为 $h$，存在 $f \in \mathcal{F}$ 使得 $f$ 在邻域内行为为 $h$ 且 $\Phi(f, \mu, x) = v$。

**Positive result (Proposition).** 若 $\nabla f(x)$ 存在，vanilla gradient 对 local-stability task 可达到 spec = sens = 1（完美推断）。

**Sample complexity.** 直接查询模型 $O((\delta^{-1})^d)$ 次即可区分局部行为（绕过归因方法）。

**覆盖的方法**: SHAP (marginal), Integrated Gradients — 都是 complete + additive 的。Vanilla gradient、LIME 不在 covered class 中。

#### B. 证明技术

**构造性 adversarial counterexample**。核心三步：

1. **Interior freedom**: 对任意局部行为模板 $h$ 和任意目标 attribution $v$，构造 piecewise linear 函数 $f$，在邻域内 $= h$，在远场调整使 completeness + additivity 迫使 $\Phi(f) = v$。
2. **Cancellation mechanism**: 构造 $f_0 \in \mathcal{F}_0$, $f_1 \in \mathcal{F}_1$，两者在邻域内行为不同（$h_0$ vs $h_1$），但远场参数调节使 $\Phi(f_0) = \Phi(f_1)$。
3. **Testing bound**: 由于 $\Phi$ 无法区分 $f_0, f_1$，任何 test 的 spec + sens $\leq 1$。

数学工具：分段线性函数构造、measure theory（baseline 分布的支撑）、线性代数（解远场参数使积分匹配）。**无 functional analysis、无拓扑方法。**

#### C. 关键假设

| 假设 | 精确条件 | Essential? | 实际满足 |
|------|----------|:----------:|----------|
| **Assumption 1** (Baseline non-degeneracy) | $\mu_j$ 在 $[x_j-\delta, x_j+\delta]$ 外有正测度支撑 | ✅ Essential | 连续分布通常满足；全零 baseline 也满足（离目标点远） |
| **Assumption 2** (Model richness) | $\mathcal{F}$ 包含所有局部匹配 $h$ 且远场 2-piecewise-linear 的模型 | ✅ Essential | ReLU 网络自然满足；任何足够 expressive 的 piecewise-linear class |
| **Completeness** | $\sum_j \Phi_j = f(x) - \mathbb{E}_\mu[f]$ | Cosmetic — 可放松后保留 identifiability | SHAP, IG 满足 |
| **Additivity** | 可加模型上 $\Phi_j$ 仅依赖 $f_j, \mu_j, x_j$ | ✅ **不可能性的真正来源** | SHAP, IG 满足；vanilla gradient 不满足 |

#### D. 局限与未来工作

- **Conditional SHAP** 未被覆盖（需要不同证明技术）
- Assumption 2 是充分条件，可弱化至 Lipschitz piecewise linear
- 未来方向：开发 task-specific 方法；建立 performance guarantees + computational efficiency
- 作者建议直接 query 模型（sample complexity）而非依赖归因

---

### Paper 2: Locatello et al. (2019)

**Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations** (ICML 2019 Best Paper)

#### A. 精确定理陈述

**Theorem 1 (唯一的形式化定理).** 对任意 $d > 1$，若 $z \sim P$ 满足 $p(z) = \prod_{i=1}^d p(z_i)$（独立先验），则存在无穷族双射

$$f: \text{supp}(z) \to \text{supp}(z)$$

满足：
1. $\frac{\partial f_i(u)}{\partial u_j} \neq 0$ a.e. 对所有 $i, j$（$z$ 与 $f(z)$ **完全 entangled**）
2. $P(z \leq u) = P(f(z) \leq u)$ 对所有 $u$（同分布，观测上不可区分）

**含义**：不存在仅从观测数据 $x$ 学习 disentangled representation 的无监督方法——任何"正确"的 latent $z$ 都对应无穷个同样合法但完全 entangled 的替代表征 $f(z)$。

#### B. 证明技术

**显式构造**，三步管道：

1. **CDF 变换** $g$：$g_i(v) = P(z_i \leq v_i)$ 将每个坐标映射到 $\text{Unif}(0,1)$
2. **Gaussian 化** $h$：逆正态 CDF，得到独立标准正态
3. **正交混合** $A$：选择所有元素非零的正交矩阵（Householder 构造），旋转后仍为标准正态但完全混合
4. **逆变换** $f = g^{-1} \circ h^{-1} \circ A \circ h \circ g$

数学工具：概率积分变换、逆 CDF、正交矩阵理论、Jacobian 链式法则。**非常初等但精巧。**

#### C. 关键假设

| 假设 | 精确条件 | Essential? | 备注 |
|------|----------|:----------:|------|
| **独立先验** | $p(z) = \prod_i p(z_i)$ | ✅ Essential | CDF 分解的基础；若先验有依赖结构，可利用此打破对称性 |
| **$d > 1$** | 至少 2 维 | ✅ Essential | 1 维无 entanglement 概念 |
| **密度存在** | $p(z)$ 有密度 | ✅ Essential | 概率积分变换需要 |
| **双射等价** | $f$ 是 bijection 且保持边际分布 | ✅ 定义性假设 | |
| **无监督** | 只观测 $x$，不观测 $z$、无干预 | ✅ 核心限制 | 放松此条件（加监督/干预）可恢复 identifiability |

**"无监督"精确含义**：训练时只有 $x \sim P(x) = \int P(x|z)p(z)dz$ 的样本，无 factor labels、无 paired interventions、无其他监督信号。

#### D. 局限与未来工作

**论文自述局限**：
- 仅用合成图像数据（dSprites, Cars3D, etc.）
- 仅 toy setting + uniformly distributed factors
- 无 confounders，独立离散 factors
- 仅 convolutional 架构
- 未扫描 latent size, optimizer, batch size

**未来方向**：
- 明确 inductive biases 的作用
- 研究 implicit/explicit supervision
- 探究 disentanglement 是否有助于下游任务
- 连接因果推断 / independent causal mechanisms

**实验发现**（大规模 12,800 模型实验）：
- 无方法一致性胜出
- **random seed 和 hyperparameters 的影响 > model architecture 的影响**
- 无监督模型选择是核心未解问题

#### 与 nonlinear ICA 的关系

论文显式指出：结果类似 nonlinear ICA 的不可辨识性——线性 ICA 可辨识（up to permutation + scaling），但非线性 ICA 一般不可辨识。Locatello 的定理将此推广到 VAE 框架。

---

### Paper 3: Sutter et al. (2025)

**The Non-Linear Representation Dilemma** (NeurIPS 2025 Spotlight)

#### A. 精确定理陈述

**Theorem 1 (Main).** 给定**任意** algorithm $\mathcal{A}$ 和**任意** DNN $\mathcal{N}$，若 Assumptions 1–5 成立，则 $\mathcal{A}$ 是 $\mathcal{N}$ 的 input-restricted distributed abstraction。

即：**任何（满足假设的）网络都可被任何算法所抽象** → causal abstraction 变得 vacuous。

**Non-linear representation dilemma**:
- 限制 alignment 为线性 → causal abstraction 有意义但可能遗漏非线性编码
- 允许 alignment 为非线性 → causal abstraction 变 trivial（失去区分力）

**支撑引理**（appendix）：
- Lemma: 前一层的 abstraction condition 可提升到后一层（post-intervention lifting）
- Theorem: input-restricted interventions 集合 countable（关键技术引理）
- Theorem: strict output-surjectivity → 每个目标类的 preimage 不可数
- Theorem: Transformers at init 几乎必然逐层 injective

#### B. 证明技术

**集合论构造 + 逐层归纳**：

1. 逐层定义 alignment map $\phi_\ell$
2. 核心技巧：**countable domain + uncountable codomain**
   - Input-restricted interventions → countable 个 hidden states
   - Hidden state space $\mathbb{R}^n$ → uncountable
   - 因此可对每个 input-restricted state 自由指定 $\phi_\ell$ 的值
3. 对每一层：在 countable 的 input-restricted states 上定义 injective mapping，再扩展为全局 bijection
4. Cantor-style injection/bijection + 逐层归纳 + 部分逆

**不用**范畴论、模型论。纯集合论 + 线性代数 + 概率论（用于 injectivity at init）。

#### C. 关键假设

| # | 假设 | 精确条件 | Essential? | 实际满足 |
|---|------|----------|:----------:|----------|
| 1 | Countable input space | $|\mathcal{X}|$ countable | ✅ | 语言模型 ✅（有限字符串），固定分辨率图像 ✅，连续 $\mathbb{R}^n$ ✗ |
| 2 | Input-injectivity (所有层) | $f_{:\ell}$ injective ∀ℓ | ✅ | Transformers at init: 几乎必然 ✅; 含 bottleneck/pooling 的架构 ✗; LayerNorm 不影响（论文有专门引理） |
| 3 | Strict output-surjectivity (所有层) | $\tau_{\text{out}} \circ f_{\ell:}$ strictly surjective ∀ℓ | ✅ | 大网络通常 ✅; softmax bottleneck 可能 ✗ |
| 4 | Matchable partial orderings | 存在神经元 partition 匹配 algorithm DAG | 结构性 | 足够宽的网络 ✅; 窄网络可能 ✗ |
| 5 | DNN solves the task | $f_{\text{task}}(x) = \arg\max_y p_N(y|x)$ ∀x | ✅ | 仅 task-solving 模型；随机初始化网络 ✗ |

**关键洞察**：Assumption 2（injectivity）是最非平凡的。论文证明 Transformers 在随机初始化时几乎必然满足。但训练后维度瓶颈可能破坏此性质。

#### D. 局限与未来工作

- 定理"依赖一种过拟合形式"——alignment map 对每个 input 单独拟合
- Learned maps 可能经验上泛化，但理论不解释何时
- Assumptions 2–3 可能在实际中失败
- 未来：研究 learned alignment maps 为何泛化；探索 injectivity/surjectivity 的失败模式

---

## E. 三篇不可能性结果的共同结构

### 统一 pattern: **Method-Model-Mismatch Trilemma**

三篇论文都证明了同一 meta-pattern：

> **当分析工具的复杂度与被分析对象的复杂度不匹配时，三类"好性质"不可兼得。**

| 性质 | Bilodeau | Locatello | Sutter |
|------|----------|-----------|--------|
| **Faithfulness** | Completeness + Additivity | 保持 true latent structure | Interventional consistency |
| **Generality** | Rich model class (piecewise linear) | 任意独立先验 | 任意 alg-DNN pair |
| **Identifiability** | 区分不同局部行为 | 恢复 true disentangling | 区分 correct vs wrong algorithm |

**不可能性 = 三者不可兼得**：
- Bilodeau: Faithfulness + Generality → ¬Identifiability
- Locatello: Faithfulness + Generality → ¬Identifiability（unsupervised setting）
- Sutter: Faithfulness + Generality → ¬Discriminability（trivialization）

### 共同证明策略

三篇都使用**构造性证明**——显式构建 counterexample，而非存在性论证：

| 论文 | 构造什么 | 怎么构造 | 核心自由度 |
|------|----------|----------|-----------|
| Bilodeau | 两个模型，局部行为不同但 attribution 相同 | 调节远场 piecewise-linear 参数 | 远场函数值（$\lambda_i$） |
| Locatello | 两个 latent representations，完全 entangled 但同分布 | CDF → Gaussian → 正交旋转 → 逆变换 | 正交矩阵 $A$（有无穷多选择） |
| Sutter | 对任意 algorithm 构造 trivializing alignment map | 逐层 Cantor-style injection | countable vs uncountable 的维度差 |

### 共同机制：**维度/容量不匹配**

- **Bilodeau**: Attribution $\in \mathbb{R}^{d \times m}$（有限维）vs 模型行为（无穷维函数空间）→ 信息必然丢失
- **Locatello**: 观测分布（$d$-维 marginal）vs latent structure（$d!$ 种 permutations + 连续变换族）→ under-determined
- **Sutter**: Input-restricted states（countable）vs hidden state space（uncountable $\mathbb{R}^n$）→ 有太多自由度来构造 trivial alignment

---

## F. 我们的 5-axiom 框架与各论文的映射

我们的框架定义 5 个公理：**Identifiability, Structural Faithfulness, Generality, Non-triviality, Finiteness**。

### 映射表

| 我们的公理 | Bilodeau | Locatello | Sutter |
|------------|----------|-----------|--------|
| **A1: Identifiability** | spec + sens > 1（被否定） | 恢复 true latent（被否定） | IIA on correct alg ≠ IIA on wrong alg（被否定） |
| **A2: Structural Faithfulness** | Completeness | 保持 factorial structure | Interventional consistency |
| **A3: Generality** | Assumption 2 (model richness) | 任意独立先验 | Assumptions 1–5 (any qualifying DNN-alg pair) |
| **A4: Non-triviality** | ⟨隐含于 hypothesis testing setup⟩ | ⟨隐含于 disentanglement 定义⟩ | **被直接否定**（trivialization = 失去 non-triviality） |
| **A5: Finiteness** | Additivity（方法结构约束） | 无监督（无额外信息源） | **被放松**（unrestricted alignment → trivialization） |

### 不相容关系的统一表述

| 来源 | 不相容公理组 | 被否定的公理 |
|------|-------------|-------------|
| Bilodeau | A2 + A5 + A3 | → ¬A1 |
| Locatello | A2 + A3 (+ unsupervised) | → ¬A1 |
| Sutter | A2 + A3 + ¬A5 | → ¬A4 |

### 我们的框架**增加了什么**

1. **Faithfulness 的拆分**：将 Bilodeau 的 "completeness + additivity" 拆为 Structural Faithfulness (A2) 和 Finiteness (A5)。这揭示了 **additivity 是 A5（方法复杂度约束）而非 A2（忠实性）**——是不可能性的真正来源。

2. **Squeeze 图景**：Finiteness 作为调节器，A5↑（约束太强）→ ¬A1（Bilodeau），A5↓（约束太弱）→ ¬A4（Sutter）。这是各论文单独看不到的结构。

3. **Mini Impossibility Theorem**：我们在 formalization.md §5 给出了统一框架下的独立证明，且进一步诊断出 additivity（而非 completeness）是不可能性的核心。

4. **Positive result via axiom relaxation**：formalization.md §5.5 证明了放松 A5（去掉 additivity，保留 completeness）可恢复 identifiability（用 vanilla gradient 编码局部信息）。

5. **Locatello 的映射揭示缺失环节**：Locatello 的 "unsupervised" 条件在我们的框架中跨越 A3（generality of priors）和 A5（无监督 = 有限信息源）。这暗示 **semi-supervised disentanglement 应该对应 A5 的中间放松**——与 Khemakhem et al. (2020) 的 iVAE 正好吻合。

---

## G. Worst-case vs Average-case Gap

### Bilodeau: 已实验验证存在巨大 gap

我们的 exp-005 直接测试了这个 gap：

| 场景 | IGA (SHAP) | 含义 |
|------|:----------:|------|
| 自然训练网络 | ≈ 2.0 | **完美可区分** |
| Adversarial 构造 | ≈ 1.1 | 接近随机猜测 |
| Natural→Adversarial 插值 β∈[0, 0.9] | 2.0 | 保持完美 |
| β=1.0 | ~1.4-1.9 | 才开始崩溃 |

**结论**：Bilodeau 的 adversarial failure 极其脆弱——保留 10% 自然成分即完全失效。

### Locatello: 同样存在 gap，但更微妙

**定理本身是 worst-case**：对任意独立先验，存在无穷多等价的 entangled representations。但：

- 实验中，**特定 inductive biases（如 β-VAE 的 KL 惩罚）在某些 datasets 上 reliably 产生 disentangled representations**
- random seeds 的影响很大 → 说明 landscape 中有"好"basin 和"差"basin
- 论文 12,800 模型实验显示：**自然训练不一定掉入 adversarial 等价类**

**推测**：Locatello 存在类似 Bilodeau 的 avg-vs-worst gap——自然训练的 VAE 不会自发选择 entangled representation，因为 Gaussian decoder + factorial prior 的 inductive bias 偏好 axis-aligned（即 disentangled）方向。但这个 gap 未被定量实验测量。

**关键区别**：Bilodeau 的 gap 源于"远场对消不会自然发生"；Locatello 的 gap 源于"正交旋转不是自然训练的稳定不动点"。机制不同但结果类似。

### Sutter: Gap 性质不同

Sutter 的定理不是 "对特定网络存在 adversarial failure"，而是 **"对所有满足假设的网络，trivialization 都成立"**。因此：

- **理论上无 gap**：不是 worst-case 过 construction，而是 universal conditional on assumptions
- **实验上有 gap**：论文用 RevNet（高容量）在**随机初始化** LM 上达到 100% IIA，但在 **trained** LM 上是否同样容易 trivialize 是 open question
- **真正的 gap 在 Assumption 2（injectivity）**：训练后的网络可能不满足 all-layer injectivity（维度瓶颈），此时定理不适用

**总结**：

| 论文 | 定理类型 | Avg-vs-worst gap? | Gap 的来源 |
|------|----------|:------------------:|-----------|
| Bilodeau | Worst-case ∃ adversarial pair | ✅ 巨大（IGA 2.0 vs 1.1） | 远场对消不自然发生 |
| Locatello | Worst-case ∃ entangled equiv | ✅ 存在（inductive bias 帮助） | 正交旋转非自然极值点 |
| Sutter | Universal ∀ (under assumptions) | ⚠️ 条件性的 | Gap 在 assumptions 是否成立 |

---

## H. 放松公理获得正面结果（Ackerman-Ben-David pattern）

Ackerman & Ben-David (2009) 对 Kleinberg 的回应开创了一种 pattern：不可能性定理的价值不在于"停止研究"，而在于**指明哪个公理可放松以获得正面结果**。

### 各论文的放松方向

| 论文 | 放松哪个公理 | 获得什么 | 已有工作 |
|------|-------------|----------|----------|
| **Bilodeau** | A5: 去掉 Additivity | 恢复 Identifiability（我们的 formalization §5.5） | Vanilla gradient 满足 completeness 不满足 additivity，可完美推断 |
| | A3: 限制 model class | 对"自然"模型 SHAP 仍有效 | 我们的 exp-005 |
| | 增加 queries | Sample complexity $O(\delta^{-d})$ | Bilodeau §5 |
| **Locatello** | A5: 加监督信号 | iVAE: identifiable up to affine | Khemakhem et al. (2020) |
| | A3: 限制先验族 | Exponential family → identifiable | Khemakhem et al. (2020) |
| | A2: 弱化 disentanglement 定义 | Slot-based / object-centric | Locatello et al. (2020) |
| **Sutter** | A5: 限制 alignment 为线性 | Causal abstraction non-trivial | DAS (Geiger et al.) |
| | A3: 放松 assumption 2 (injectivity) | Bottleneck → alignment 受限 | Open |
| | A1: 接受 approximate identifiability | IIA < 100% 但 > random | Standard practice |

### 我们的 formalization 提供的新放松

**Proposition (formalization.md §5.5)**：仅 Completeness（A2a）不导致不可能性。定义

$$\Phi_1 = \frac{\partial m}{\partial x_1}\bigg|_x, \quad \Phi_2 = m(x) - \mathbb{E}_\mu[m] - \Phi_1$$

满足 Completeness 但不满足 Additivity，且对 $h_0'(x_1) \neq h_1'(x_1)$ 达到 spec + sens = 2。

**含义**：Additivity 是可放松的代价最低的公理。放松后仍保持 completeness（归因总和等于输出差），但归因不再被远场积分"污染"。

---

## I. "Everyone does X but nobody does Y"

### Everyone does:

1. **Everyone proves impossibility via construction, nobody proves it via information theory.**
   三篇都显式构建 counterexample（adversarial model pair / entangled reparametrization / trivializing map）。没有论文给出信息论下界（如 mutual information bound $I(\Phi; B) \leq \ldots$）。这意味着我们不知道 impossibility 有多"tight"——是紧的还是证明技术的 artifact。

2. **Everyone states worst-case results, nobody quantifies the average-case gap.**
   Bilodeau 给了一个 average-case observation（Proposition: 错误率 ∈ (0.25, 0.5) 在简单分布上），但没有 formal average-case theorem。Locatello 做了大量实验但没有 average-case 理论。Sutter 完全是 worst-case。**我们的 exp-005 是目前唯一系统性地定量测量 avg-vs-worst gap 的工作。**

3. **Everyone assumes model richness, nobody studies what happens with bounded-complexity models.**
   Bilodeau 需要 piecewise-linear richness，Locatello 需要任意先验，Sutter 需要 injectivity。没有人研究 **bounded-depth / bounded-width networks 上不可能性是否消失**。这是一个巨大的 gap——实际网络有有限容量。

4. **Everyone treats the method and the model symmetrically, nobody breaks the symmetry.**
   不可能性源于 "方法太简单 OR 方法太复杂"。但实际 ML 中，方法和模型不是独立选择的——方法可以 adapt to specific model。没有人研究 **model-aware attribution**（根据具体模型的结构调整方法）是否能绕过不可能性。

### Nobody does:

5. **Nobody connects the three impossibilities into a unified impossibility.**
   Bilodeau、Locatello、Sutter 在不同社区（XAI、representation learning、mech interp），互相几乎不引用。没有论文指出它们是**同一 meta-theorem 的不同实例**。我们的 5-axiom framework 是第一个尝试。

6. **Nobody provides a constructive "what to do instead" theorem.**
   每篇都说 "future work: develop better methods"，但没有给出 formal positive result 说明什么样的方法在什么条件下可行。唯一接近的是 Bilodeau 的 sample complexity，但那是暴力搜索而非结构化方法。

7. **Nobody tests impossibility predictions on real (pretrained) models.**
   Bilodeau 用 synthetic ReLU nets，Locatello 用 synthetic images + VAEs，Sutter 用 random-init LMs。没有人在 **pretrained GPT / BERT / ViT** 上验证不可能性是否确实触发。我们的 exp-005 也用 synthetic nets——在 pretrained 模型上的验证仍是 open。

8. **Nobody formalizes the computational cost of "escaping" impossibility.**
   放松公理可以恢复 identifiability，但代价是什么？Sample complexity 增长多快？Computation 增长多快？没有 complexity-theoretic 分析。

---

## 附录：引用关系

```
Bilodeau (2024) ──cites──→ Kleinberg (2002) [inspiration]
                ──cites──→ Arrow (1950) [axiom incompatibility pattern]
                  ↑
                  │ 未互引
                  ↓
Locatello (2019) ──cites──→ Hyvärinen & Pajunen (1999) [nonlinear ICA]
                  ↑
                  │ 未互引
                  ↓
Sutter (2025)   ──cites──→ Geiger et al. (2024) [causal abstraction]
                ──cites──→ Huang et al. (2024) [RAVEL, probing]

我们的 formalization ──cites──→ 全部三篇 + Kleinberg + Han
                     ──adds──→ 5-axiom framework + Mini Impossibility + Squeeze conjecture
```
