# DR-013: 表征学习/表征分析领域论文地图

## 动机

我们在研究神经网络表征的数学结构。在决定研究方向之前，需要先对整个领域有全面的 overview——每个分支有哪些重要论文，做了什么，当前进展到哪里了。

## 任务

为以下每个分支，列出该分支**最重要的论文**（按影响力/时间排序），对每篇给出一句话概括。区分开创性论文、关键进展、最新前沿。

## 分支列表

### 1. 表征相似度度量
CKA, CCA, SVCCA, RSA, Procrustes, mutual kNN, shape metrics, centered kernel alignment 等。

### 2. Sparse Autoencoders (SAE) 与字典学习
SAE 用于 mech interp、transcoders、crosscoders、superposition hypothesis、feature splitting/absorption。

### 3. 因果抽象与机械可解释性
Causal abstraction, activation patching, circuit analysis, DAS, interchange interventions。

### 4. 线性表征假说 (Linear Representation Hypothesis)
概念方向、steering vectors、representation engineering、concept bottleneck。

### 5. Probing 与 Lens
Linear probes, structural probes, logit lens, tuned lens, MDL probing。

### 6. 信息论方法
PID, mutual information probing, information bottleneck, redundancy/synergy。

### 7. 拓扑/几何方法
Persistent homology, TDA, intrinsic dimension estimation, manifold hypothesis, neural manifolds。

### 8. Disentanglement 与可识别性
解纠缠表征、ICA、nonlinear ICA、因果表征学习、identifiability。

### 9. 表征的 Scaling Laws 与收敛
Platonic representation hypothesis、表征随规模变化的规律、跨模态收敛。

### 10. 不可能性与理论基础
不可能性定理、no-free-lunch、公理化框架、表征学习的理论基础。

## 输出格式

对每个分支：

```
### 分支名

**开创性论文**
- Author (Year) — 一句话概括 [arXiv ID / venue]

**关键进展**
- Author (Year) — 一句话概括

**最新前沿 (2024-2026)**
- Author (Year) — 一句话概括

**当前状态**：一段话总结该分支目前走到了哪里，主要的 open questions 是什么
```

## 最终总结

1. 哪些分支最活跃、进展最快？
2. 哪些分支之间有**未被充分探索的交叉**？
3. 整个领域目前最大的 3-5 个 open questions 是什么？
