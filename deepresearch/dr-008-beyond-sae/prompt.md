# 神经网络表征分解与分析方法：统一框架的可能性

## 背景

SAE（Sparse Autoencoder）已成为 mechanistic interpretability 的主流工具，但它有根本性的局限：线性假设、不可识别性（不同种子学到不同特征）、可扩展性问题。与此同时，表征分析领域存在大量看似无关的方法（probing、CKA、PID、activation patching、logit lens 等），它们各自从不同角度观测同一个对象——神经网络的内部表征。

Klabunde et al. (arXiv:2305.06329) 的一个精彩贡献是：把所有表征相似性度量按不变性群分类（PT ⊂ OT ⊂ ILT ⊂ AT），揭示了"这些度量为什么给出不同答案"的数学本质。这种统一视角极大地推进了领域的理解。

我想知道：**对于更广泛的表征分析工具箱，是否存在类似的统一数学框架？**

## 请研究的内容

### 一、方法全景

现有的表征分解/分析方法有哪些？每种方法的核心思想、数学基础、它暗含了什么假设（类比 invariance group 的思路——每种方法暗含了对"什么是特征"的不同定义）？

覆盖但不限于：SAE、Probing、Logit/Tuned Lens、Activation Patching、Transcoders、CCA/PCA/CKA、PID、拓扑方法（persistent homology）、因果抽象、representation engineering / steering vectors、非线性特征分解。

### 二、数学关系

这些方法之间的数学关系是什么？

- 哪些方法在特定条件下等价？（比如 linear CKA = RV coefficient）
- 哪些方法是另一些的推广或特例？
- 它们各自对应什么样的"表征观"？（线性子空间 vs 流形 vs 拓扑 vs 信息论 vs 因果）
- Klabunde 用 invariance group 统一了相似性度量。能否用一个更大的数学框架（如信息几何、范畴论、sheaf theory）统一所有这些方法？

### 三、前沿问题

当前最需要解决的工具/方法层面的研究问题：
- 非线性特征（Engels et al. 发现的圆形/多维特征）需要什么新工具？
- SAE 的可识别性问题有没有数学上的根本解？
- 能否建立"表征分析方法的不可能定理"——类似 Arrow 不可能定理，证明不存在同时满足所有期望性质的单一方法？
- 每个问题给出：问题陈述、重要性、当前进展与 gap

给出关键论文的 arXiv ID。
