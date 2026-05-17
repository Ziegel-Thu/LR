# DR-013: 表征学习/表征分析的论文地图——每个分支的核心定义与隐含假设

## 动机

我们要在表征学习/表征分析领域找到值得定义的新问题。好问题往往藏在已有论文的**核心定义中的隐含假设**里（如 Sutter 发现 causal abstraction 的定义没有限制 alignment map 的复杂度）。但前提是我们得**广泛了解**每个分支有哪些核心定义。

## 任务

为以下每个分支，列出该分支的 **top 5-10 篇最重要的论文**（按影响力排序），对每篇给出：

1. **一句话概括**：这篇论文做了什么
2. **核心定义/形式化**：它引入的最重要的数学定义或框架是什么（用数学符号简述）
3. **隐含假设**：这个定义中有哪些**未被明确讨论或证明合理性**的假设？哪些假设如果去掉，结论可能不成立？
4. **后续挑战**：是否有后续工作质疑过这些假设？结果如何？

## 分支列表

### 1. 表征相似度度量
CKA, CCA, SVCCA, RSA, Procrustes, mutual kNN, shape metrics 等。
- 起源论文、核心 survey、最新进展
- 特别关注：每种度量的不变性群假设

### 2. Sparse Autoencoders (SAE) 与字典学习
SAE 用于 mech interp、transcoders、crosscoders、feature splitting/absorption。
- Cunningham 2023, Bricken 2023, Paulo-Belrose 2025, Tang 2025, Engels 2024 (多维特征) 等
- 特别关注：线性叠加假设 (superposition hypothesis) 的精确陈述和已知反例

### 3. 因果抽象与机械可解释性
Causal abstraction, activation patching, circuit analysis, DAS。
- Geiger 系列、Conmy (ACDC)、Wang (IOI circuit)、Sutter 2025 等
- 特别关注：alignment map 的限制条件、intervention 的合法性

### 4. 线性表征假说 (Linear Representation Hypothesis)
概念方向、probing、steering vectors、representation engineering。
- Mikolov 2013, Park-Choe-Veitch 2023, Zou 2023 (RepE), Turner 2023 (activation addition) 等
- 特别关注："概念 = 方向"这个假设在什么条件下成立/失败

### 5. Probing 与 Lens
Linear probes, structural probes, logit lens, tuned lens, MDL probing。
- Alain-Bengio 2016, Hewitt-Manning 2019, Belinkov 2022, Belrose 2023 (tuned lens) 等
- 特别关注：probe 的 selectivity 问题（Hewitt-Liang 2019），"能 decode ≠ 模型在用"

### 6. 信息论方法
PID (Partial Information Decomposition), mutual information probing, information bottleneck。
- Williams-Beer 2010, Ehrlich 2022, Shwartz-Ziv & Tishby 2017 (information bottleneck) 等
- 特别关注：redundancy lattice 的选择（不同选择给不同答案），information bottleneck 争议

### 7. 拓扑/几何方法
Persistent homology, TDA for neural networks, intrinsic dimension estimation, manifold hypothesis。
- Watanabe-Yamana 2021, Facco 2017 (TwoNN), Ansuini 2019 (ID in DNNs) 等
- 特别关注：manifold hypothesis 的精确陈述和实证支持/反对

### 8. Disentanglement 与可识别性
解纠缠表征、ICA、nonlinear ICA、因果表征学习。
- Locatello 2019 (impossibility), Khemakhem 2020 (iVAE), Schölkopf 2021 (causal representation learning) 等
- 特别关注：Locatello 的不可能性和后续如何通过归纳偏置绕过

### 9. 表征的 Scaling Laws
表征随模型/数据规模变化的规律、柏拉图表征假说、收敛/发散。
- Huh 2024 (Platonic Representation Hypothesis), Nguyen 2020 (wide networks similarity) 等
- 特别关注：收敛到"同一个表征"这个 claim 的精确含义和已知限制

### 10. 不可能性/No-Free-Lunch 定理
已有的各种不可能性结果（不限于我们已知的四篇）。
- Kleinberg 2002, Bilodeau 2024, Han 2022, Sutter 2025, Locatello 2019, 还有哪些？
- 特别关注：每个不可能性定理之后的**生产性反驳**（Ackerman-Ben-David 式的"换公理后的正面结果"）

## 输出格式

对每个分支：

```
### 分支名
#### 核心论文（按影响力排序）
1. **Author Year** — 一句话概括
   - 核心定义：[数学符号简述]
   - 隐含假设：[未被证明合理性的假设]
   - 后续挑战：[是否有人质疑？]
2. ...

#### 该分支中最值得追问的开放假设
- [具体假设] — 为什么值得追问，如果去掉会怎样
```

## 最终总结

在所有分支的隐含假设中，哪 10 个**最值得质疑**？标准：
1. 广泛使用但很少被讨论
2. 如果去掉，可能导致类似 Sutter/Locatello 式的颠覆性结果
3. 有实验可以检验
4. 与我们已有的实验数据（exp-001~005）有关联
