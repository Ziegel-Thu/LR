# Information Theory 分支：综合方法论分析

> 2025-07-18 生成

---

## 目录

- [A–D. 逐篇分析](#a-d-逐篇分析)
  - [Paper 1: Ehrlich et al. (2023)](#paper-1-ehrlich-et-al-2023)
  - [Paper 2: Saxe et al. (2019)](#paper-2-saxe-et-al-2019)
- [E. What Died and What Survived](#e-what-died-and-what-survived)
- [F. The Scalability Problem](#f-the-scalability-problem)
- [G. Cross-Branch Connections](#g-cross-branch-connections)
- [H. "Everyone Does X but Nobody Does Y"](#h-everyone-does-x-but-nobody-does-y)

---

## A–D. 逐篇分析

### Paper 1: Ehrlich et al. (2023)

**A Measure of the Complexity of Neural Representations based on Partial Information Decomposition** (TMLR 2023)

#### A. 精确方法

**核心量：Representational Complexity C**

$$C := \frac{1}{I(T:\mathbf{S})} \sum_\alpha \Pi(T:\mathbf{S}_\alpha) \cdot m(\alpha)$$

其中 $m(\alpha) = \min_{\mathbf{a} \in \alpha} |\mathbf{a}|$ 是 synergy degree（最小 antichain 元素大小）。直觉：**平均来看，需要同时观察多少个 neuron 才能获取一条 task-relevant 信息？**

取值范围：$C = 1$（所有信息可从单个 neuron 获取）到 $C = n$（信息纯在 $n$-way synergy 中）。

**PID measure 选择**：使用 $I^{sx}_\cap$（Makkeh et al. 的 shared-exclusions measure），理由：
- 连续且可微
- 满足 PID 所需公理
- 可分解为 informative / misinformative 两部分

**MI / PID 估计**：不使用 MI 估计器。因为网络被量化为有限字母表，概率分布可直接通过频率统计得到，MI / PID 是精确计算（非估计）。这是回避 Saxe 批评的关键设计。

**量化方案**：
- 8 levels（3 bits）或 4 levels（2 bits）per neuron
- 公式：$\ell = \epsilon \lfloor (\hat\ell - \sigma_{\min}) / \epsilon \rceil + \sigma_{\min}$，$\epsilon = (\sigma_{\max} - \sigma_{\min}) / (n_{\text{bins}} - 1)$
- **训练时**用 stochastic quantization（forward pass 随机舍入），**评估时**用确定性量化
- 关键论点：量化是网络 forward pass 的一部分，不是 post-hoc analysis artifact

**网络架构**：

| 设置 | 架构 | 激活 | 数据集 | 训练 |
|------|------|------|--------|------|
| MNIST default | FC: 784→...→5→5→5→10 | tanh (hidden), softmax (output) | 60K train, 60K test (MNIST+QMNIST) | SGD, bs=64, lr=0.01, 10⁵ epochs, 20 runs |
| MNIST binary | 同上但 output=4 (binary) | tanh + sigmoid | 同上 | 同上 |
| MNIST reduced | 同上但 4 quant levels | tanh + softmax | 同上 | 同上 |
| MNIST shuffled | 同上 | tanh + softmax | 标签随机打乱 | 同上, 5 runs |
| CIFAR10 | 3 conv (32/64/128, k=3, maxpool+ReLU) → FC: 2048→128→32→5→5→10 | ReLU (conv) + tanh (FC) | 50K images | SGD, bs=64, lr=0.005, 10⁴ epochs, 10 runs |

**Subsampling**：从大层随机选 $\hat{n}$ 个 neurons 做 PID。作者明确说**不能提供 bound**，会低估或高估，"not suitable"。

**Coarse-graining**：把 $d$ 个 neurons 合成一个高维 super-variable。
- Atom mapping 定理：$\Pi(T:\tilde{\mathbf{S}}_{\tilde\Phi}) = \sum_{\Phi: \Phi \circ f^{-1} = \tilde\Phi} \Pi(T:\mathbf{S}_\Phi)$
- Synergy degree bound：$m(\tilde\Phi) \le m(\Phi) \le d \cdot m(\tilde\Phi)$
- **Complexity bound**：$C(T:\tilde{\mathbf{S}}) \le C(T:\mathbf{S}) \le d \cdot C(T:\tilde{\mathbf{S}})$

**计算限制**：PID atoms 数量按 Dedekind numbers 增长（super-exponential），**6+ sources 的 full PID 已不可行**。

#### B. 关键结果

**Toy examples**（验证量 C 的直觉）：

| 编码方式 | C 值 | 预期 |
|----------|------|------|
| One-hot (4 classes in 4 neurons) | 1.00 | 最简单 |
| Binary/pair encoding (4 values in 4 neurons) | 1.21 | 稍复杂 |
| 16 states in 4 binary neurons | 1.67 | 中等 |
| XOR of two 8-level neurons | 1.89 | 接近上限 2 |
| Parity of 3 thresholded neurons | 2.83 | 接近上限 3 |
| 10-class one-hot output | 1.46 | — |

**训练中的 C 变化**：
- MNIST 小隐层 ($L_3, L_4, L_5$, 各 5 neurons)：**C 随训练单调下降**，且**层越深 C 越低**
- Binary-output MNIST：前 ~10 epochs 先升后降
- CIFAR10 后两层：前 ~10 epochs plateau，之后下降
- Shuffled labels：训练集 C 下降但测试集不下降（符合预期）

**与其他 measure 对比**：
- Reing representational complexity $C_{\text{Reing}}$：也下降，但不单调、值更高、不保证跨层 ordering
- Local intrinsic dimension：也随训练下降，但不保证跨层单调

**信息平面**：作者没画传统 information plane。他们明确批评：连续网络里 $I(X;T)$ 要么 infinite 要么固定等于标签 entropy，传统 info plane 分析有方法论缺陷。

#### C. 明确局限 + future work

**不能做的**：
- ≥6 sources 的 full PID 计算不可行
- Subsampling 无 bound
- Coarse-graining bound 对大网络"impractically loose"
- 当前方法只适用于 intrinsically discrete（量化）系统

**Future work**：
- Train vs test C 的差异（overfitting 诊断）
- 不同 class 的 C 差异
- 训练早期的 restructuring
- 拆到 backbone atoms 层面
- C 的下界推导
- 换 target 为 input（而非只看 label）
- 推广到 RNN / 生物 / 生态系统

#### D. 方法论缺口

1. **量化依赖性**：结果强依赖量化方案（8 vs 4 levels），但无系统敏感性分析
2. **PID measure 依赖**：核心结果用 $I^{sx}_\cap$，不保证对其他 PID measure 不变
3. **Scalability gap**：真正的大层（128+ neurons）无法直接计算，coarse-graining bound 太松
4. **Train-set bias**：用 train set 评估，绑定到训练分布
5. **Target choice**：只看 label，不看 input 或中间目标
6. **无因果声明**：低 C 被解释为"difficulty of access"，但未证明低 C → 更好 generalization
7. **编码依赖**：output 编码不同（one-hot vs binary）会改变 C，说明 C 是 representation-dependent，不是纯任务常数

---

### Paper 2: Saxe et al. (2019)

**On the Information Bottleneck Theory of Deep Learning** (ICLR 2018 / JSTAT 2019)

#### A. 精确方法

**测试的 3 个 IB claims**：
1. 深网训练有 fitting phase + compression phase 两个阶段
2. Compression 因果地解释了 generalization
3. Compression 源自 SGD 的 diffusion-like behavior

**网络架构**：

| 设置 | 架构 | 激活 | 数据集 |
|------|------|------|--------|
| Shwartz-Ziv 复现 (tanh) | FC: 12→10→7→5→4→3→2 | tanh hidden, sigmoid output | IBnet: 12-dim, 4096 patterns, binary labels |
| Shwartz-Ziv (ReLU) | 同上 | ReLU hidden, sigmoid output | 同上 |
| MNIST (tanh/ReLU) | 784→1024→20→20→20→10 | tanh 或 ReLU | MNIST |
| Deep linear | 1 或 5 hidden layers × 50 units | linear | Synthetic teacher-student |
| Irrelevant subspace | linear, 30 relevant + 70 irrelevant dim | linear | Synthetic |

**MI 估计方法**（多种对比）：
- **Binning**：tanh 用 30 equal bins in [-1, 1]；ReLU 用 100 bins over observed min/max
- **KDE / mixture-of-Gaussians**（Kolchinsky & Tracey 2017）
- **Kraskov k-NN estimator**
- 对 linear network：加入 Gaussian noise $T = WX + \varepsilon$，有 MI 闭式解

**对 binning 的核心批评**：
- 对 deterministic continuous network，$I(X;T)$ 理论上是 **infinite**（或等于 $H(X)$），必须靠 binning 或 added noise 才能得到有限值
- 但 **binning/noise 是 analysis artifact，不是网络实际运算**
- 不同 binning scheme 可以得出完全不同的 MI 轨迹（例如按 net-input 方式定 bin，tanh 的 compression 消失）
- 30–100 bins 太粗，与真正 digital precision 差距很大

**关键实验**：
1. tanh vs ReLU 在 IBnet 上的 information plane 对比
2. Minimal 3-neuron analytical model：Gaussian input 过 tanh/ReLU，对 binned $I(T;X)$ 有精确公式
3. Deep linear teacher-student：compression vs generalization 独立变化
4. Full-batch GD vs SGD：两者都能产生 compression（在 tanh）
5. Task-relevant vs task-irrelevant 输入拆分

#### B. 关键结果

**ReLU vs tanh 的根本差异**：
- **tanh**：$I(X;T)$ 先升后降 → 出现 compression phase。机制：大权重让 tanh **双边饱和**，把输入映射到少数几个 bin，分布 entropy 下降
- **ReLU**：$I(X;T)$ **单调增加**，不压缩。机制：ReLU 单边饱和，非零部分随权重扩散，entropy 不降
- **Linear**：$I(X;T)$ 单调增加
- MNIST 上结论一致（KDE / binning / k-NN 三种方法都确认）

**Compression ≠ Generalization**：
- Compress 但不 generalize well（tanh overfitting case）
- 不 compress 也能 generalize well（ReLU / linear with good test performance）
- **没有清楚的 causal link**

**Full-batch GD vs SGD**：
- 两者在 tanh 上都产生类似 compression
- 因此 **SGD diffusion 不是 compression 的必要原因**
- Gradient SNR 的两阶段转变在所有 activation 中都出现，但与 compression 无关

**Task-irrelevant information**：
- 全局 $I(X;T)$ 可能仍上升
- 但对 irrelevant subspace 的 MI **会下降**
- 这个 compression 与 fitting **同时发生**，不是后续单独的 compression phase

**关键数值/轨迹**：
- 某些 binning 设定下 MI 会被 pin 到 $\log_2(P)$（例如 machine precision binning）
- MNIST：$H(X) = \log_2 10000$ 作为上界

#### C. 明确局限 + future work

**没有否定的**：
- **不否定 IB 作为一般 optimization objective / principle**
- 不否定 IB 在 stochastic networks、显式正则、或新训练算法中的价值
- 不否定 task-irrelevant information 的压缩确实会发生

**否定的**：
- IB 作为**解释当前深网训练 dynamics + generalization 的具体 descriptive theory**
- 具体地：三个 claims 在一般情况下不成立

**Future work**：
- 更系统研究真正 stochastic/regularized networks
- 研究 explicit noise assumptions 下的 IB
- 探索输入中"高方差方向是否更 task-relevant"

#### D. 方法论缺口

1. **主要针对 deterministic continuous nets + post-hoc binning**，对真正 stochastic representations 或不同 random-variable 定义结论不直接等价
2. **没有测试 quantized networks**：Ehrlich 2023 用量化让 MI well-defined，绕过了 Saxe 的主要批评
3. **所有 tanh 实验都用较窄的 binning**（30 bins），不同 bin 数的 robustness 虽有讨论但未系统比较
4. **只测了 FC 和 linear networks**，未涉及 CNN / Transformer / 现代架构
5. **未涉及 PID**：Saxe 只用 pairwise MI，无法区分 redundancy 和 synergy

---

## E. What Died and What Survived

### 💀 What Died with Saxe 2018

**IB 作为训练 dynamics 的 descriptive theory 死了：**

1. **两阶段叙事**（fitting → compression）：只在 tanh + binning 下出现，是 activation saturation 的 artifact，不是普遍 learning principle
2. **"Compression explains generalization"**：ReLU 网络不压缩但泛化得很好；tanh 网络可以压缩但仍 overfit。两者无因果关系
3. **"SGD diffusion causes compression"**：full-batch GD 也产生相同 compression。SGD 的随机性不是压缩的原因

**具体后果**：传统 information plane（$I(X;T)$ vs $I(T;Y)$）作为训练诊断工具基本废弃。连续 deterministic 网络的 $I(X;T)$ 要么 infinite 要么是 binning artifact。

### ✅ What Survived

1. **IB 作为 optimization objective**：Variational Information Bottleneck (VIB, Alemi et al. 2017) 等方法把 IB 作为显式目标函数，这完全不受 Saxe 批评影响
2. **Task-irrelevant info compression 确实发生**：Saxe 自己验证了这一点。只是它与 fitting 同时发生，不是后续独立阶段
3. **Information theory 作为分析工具的合法性**：关键是要让 MI well-defined——Ehrlich 的量化方案正是这个方向
4. **PID 作为比 MI 更细粒度的工具**：Saxe 的批评只涉及 pairwise MI。PID 提供了全新维度（redundancy vs synergy），不受"compression 是否发生"这个问题的限制

### 🔑 关键传承

Saxe → Ehrlich 的逻辑链条：
- Saxe 证明：post-hoc binning 的 MI 不可信 → **必须让离散化成为网络的一部分**
- Ehrlich 回应：用 stochastic quantization 训练网络，量化是 forward pass 的一部分 → MI/PID well-defined
- Ehrlich 还做了更聪明的事：不再追问"compression 是否发生"这个已死问题，转而问"information 如何**分布**在 neurons 之间"——这是一个独立于 compression 叙事的新问题

---

## F. The Scalability Problem

### Ehrlich 能走多远？

**当前上限**：Full PID 只能算 ≤5 sources（neurons）。

| n (sources) | PID atoms 数 | 可行性 |
|-------------|-------------|--------|
| 2 | 3 | trivial |
| 3 | 18 | easy |
| 4 | 166 | feasible |
| 5 | 7579 | feasible but slow |
| 6 | 7828352 (Dedekind) | impractical |
| 7 | ~2.4 × 10¹⁰ | impossible |

**Coarse-graining 的真实 gap**：假设要分析一个 128-neuron 层，coarse-grain 到 5 super-variables（每个含 ~26 neurons）：
- 下界 $C(T:\tilde{\mathbf{S}})$ 精确
- 上界 $d \cdot C(T:\tilde{\mathbf{S}}) = 26 \cdot C_{\text{coarse}}$
- 若 $C_{\text{coarse}} = 1.5$，则 $1.5 \le C_{\text{true}} \le 39$——这个 bound 几乎无信息量

### 对 Transformer 层需要什么？

一个 Transformer 的 hidden dim 通常 768–4096。要计算 representational complexity：

1. **Token-level**：每个 token 的 representation 是 $d$-dim vector。需要选 target（例如 next-token prediction label）和 sources（$d$ 个 neuron 维度）。$d = 768$ → 完全不可行
2. **Head-level**：注意力头作为 sources（例如 12 heads/layer），需要量化每个 head 的输出为有限字母表。12 heads → 6+ sources → 仍超出 PID 极限
3. **Layer-level**：把整层当作一个 source 的 sequence，但这失去了 PID 的意义

**需要的突破**（按可能性排序）：
1. **近似 PID 算法**：不精确计算所有 atoms，而是估计 C 的上下界。目前无此类算法
2. **Structured PID**：利用网络结构（如 attention pattern 的稀疏性）减少搜索空间
3. **连续 PID 的高效估计器**：$I^{sx}_\cap$ 有 continuous generalization，但无高效 estimator
4. **不同粒度的 decomposition**：不以 individual neurons 为 sources，而以 groups（如 SAE features、attention heads）为单位

### 诚实评估

以当前方法，Ehrlich 的 PID-based C 在以下场景有价值：
- ✅ 量化后的小层（≤5 neurons）：精确计算
- ⚠️ 量化后的中层（~10-50 neurons）：coarse-graining 给出有意义但宽松的 bounds
- ❌ 大层（128+）：bounds 太松，无实际信息
- ❌ 现代 Transformer：完全不可行

---

## G. Cross-Branch Connections

### C 与 LRH linearity 的关系

**Linear Representation Hypothesis** 认为概念被编码为线性方向。从 PID 角度：
- 如果信息确实沿线性方向编码，那么每个概念的信息应主要在 **redundancy**（$m=1$）中，因为线性方向意味着单个 neuron 的线性组合已经够用 → **C 应接近 1**
- 如果 C >> 1（大量 synergy），说明信息不能从单个 neuron 获取，暗示**非线性或分布式编码** → 与 LRH 的简单版本矛盾
- 但注意：LRH 说的是"概念方向存在"，C 说的是"从个别 neuron 能否获取信息"。一个线性方向可以跨越多个 neuron 的联合空间而仍然是线性的，此时 C 可能 > 1
- **预测**：如果 LRH 成立的层，C 应较低（多数信息可从少数 neuron 获取或冗余编码）

### C 与 Intrinsic Dimension (ID) 的关系

- Ehrlich 论文比较了 C 与 local intrinsic dimension，发现两者都随训练下降
- ID 测量的是 representation manifold 的有效维度，C 测量的是 information 的分布方式
- 理论上：低 ID → representation 集中在低维 subspace → 可能对应低 C（少数 neuron 就够）
- 但 ID 不区分 redundancy 和 synergy：即使 ID 低，信息可能仍在几个 neuron 的 synergy 中
- **关键区别**：ID 是几何量（manifold），C 是信息论量（分布）。两者的相关性是有趣的经验问题

### C 与 SAE features 的关系

- SAE 的目标是找到 sparse, monosemantic features
- 如果 SAE 成功，每个 feature direction 应独立携带信息 → 在 SAE feature basis 下 C ≈ 1
- **可操作的实验**：计算原始 neuron basis 下的 C 与 SAE feature basis 下的 C，看 SAE 是否真正"解耦"了 synergy
- 如果 SAE basis 下 C 仍然高，说明 SAE features 之间仍有 synergistic interaction → SAE 不够 complete
- **与 exp-001 (SAE identifiability) 的连接**：如果 SAE features 不唯一，那么 C 在不同 SAE basis 下可能不同

### C 与不可能性定理的关系

- Bilodeau et al. (2024) 的不可能性定理说：任何满足基本公理的 feature attribution 方法在 worst case 下可以任意误导
- C 提供了一个 complementary 视角：C 高的表征本来就"难以 attribute"——信息分布在 synergy 中，任何 single-feature attribution 都会遗漏
- **连接点**：C 可作为 feature attribution 可靠性的**先验指标**——C 低的层更适合做 attribution，C 高的层天然抵抗任何 attribution 方法

---

## H. "Everyone Does X but Nobody Does Y"

### Everyone 用 MI 量化层间信息流，Nobody 用 PID 区分 redundancy 和 synergy
- Shwartz-Ziv & Tishby (2017) 以来大量工作画 information plane
- 但 MI 混淆了 redundant 和 synergistic 贡献（Williams & Beer 2010 的核心论点）
- Ehrlich 2023 是少有的真正用 PID 的工作，但受限于 scalability
- **Gap**：需要 scalable PID 估计器

### Everyone 争论 compression 是否发生，Nobody 问 information structure 如何变化
- Saxe 2018 以来的讨论集中在"是否有 compression phase"
- 但这是一个 coarse 的问题——即使 total MI 不变，信息的**内部结构**（redundancy vs synergy 分布）可能在剧烈变化
- Ehrlich 的 C 正是回答这个更细粒度问题的工具
- **Gap**：需要 track PID atoms 的动态，不仅仅是 total C

### Everyone 用 binning/KDE 估计 MI，Nobody 在网络的 actual computational graph 中定义 MI
- Saxe 的批评核心：binning 不是网络运算的一部分
- Ehrlich 的量化方案是一个回应，但量化改变了网络本身
- **真正缺失的**：在不改变网络的前提下，找到 MI 的 well-defined 版本。可能方向：
  - 条件 MI（conditioning on finite output）
  - Functional MI（只关心与下游相关的信息）
  - Usable information (Xu et al. 2020)

### Everyone 分析小网络上的 toy tasks，Nobody 在生产级模型上做 information-theoretic 分析
- Ehrlich：MNIST / CIFAR10 上的 5-neuron 层
- Saxe：IBnet (12-dim, 4096 patterns) 和 小 MNIST 网络
- **巨大 gap**：GPT/LLaMA 级别的模型没有任何 PID 分析
- 但这不仅仅是"需要更大 GPU"——根本问题是 PID 对 source 数量的 Dedekind number scaling

### Everyone 关注 label information，Nobody 分析 inter-layer information structure
- Ehrlich 和 Saxe 都以 $I(T;Y)$（T = hidden layer, Y = label）为核心
- 但层间信息传递（$I(T_l; T_{l+1})$）和表征间的关系结构几乎无人用 PID 分析
- **Gap**：PID 分析 "一个层如何利用上一层的信息"——这与 residual connections、skip connections 等架构选择直接相关

### Everyone 在固定 basis (neuron basis) 中做分析，Nobody 检验 basis 选择的影响
- PID 结果依赖 source 的定义。Neuron basis、PCA basis、SAE basis 会给出不同的 PID 分解
- **可操作的问题**：哪个 basis 使 C 最小？这个 optimal basis 是否与 SAE 或 PCA 一致？
- 这个问题本身可能产生新的 representation quality metric
