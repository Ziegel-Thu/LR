# Klabunde et al. 2023 - Similarity of Neural Network Models

**论文**: Similarity of Neural Network Models: A Survey of Functional and Representational Measures  
**作者**: Max Klabunde, Tobias Schumacher, Markus Strohmaier, Florian Lemmerich  
**来源**: ACM Computing Surveys, 2025 (arXiv: 2305.06329)  
**TeX 源码**: `literature/klabunde-2023-nn-similarity/`

## 核心思想

系统综述了两种互补的神经网络相似性度量视角：
1. **Representational Similarity（表征相似性）**：比较中间层的 activations
2. **Functional Similarity（功能相似性）**：比较模型的 outputs

两者互补：functionally similar 的模型可能有不同的 representations，反之亦然。

## Representational Similarity Measures 分类

### 1. CCA-Based（典型相关分析）
- **CCA**: 找到使两个表征最大相关的线性组合，invariant to affine transformations
- **SVCCA**: 先做 PCA 去噪，再 CCA；invariant to OT + IS + TR
- **PWCCA**: 用重要性加权 canonical correlations；asymmetric

### 2. Alignment-Based（对齐）
- **Orthogonal Procrustes**: 找最优正交变换对齐两个表征，是 distance metric；invariant to OT
- **Generalized Shape Metrics**: 基于统计形状分析，可调节 invariance 范围（α=0 时 invariant to AT，α=1 时 invariant to OT+IS+TR）
- **Soft Matching Distance**: 推广 Permutation Procrustes 到不同维度，基于 2-Wasserstein 距离
- **Linear Regression (R²)**: 用线性回归预测一个表征从另一个，用 R² 衡量
- **Correlation Match**: 将两个表征的 neurons 一对一匹配，计算平均相关

### 3. RSM-Based（表征相似性矩阵）
- **RSA**: 先算 pairwise similarity matrix，再比较两个 RSM（可用 Spearman/Pearson 等）
- **CKA (Centered Kernel Alignment)**: 用 kernel 算 RSM，再用 HSIC 比较；linear CKA invariant to OT+IS。**被广泛使用但存在争议**（对 perturbation 不够敏感）
- **Distance Correlation**: 非线性相关度量
- **Normalized Bures Similarity**: 源自量子信息论
- **GULP**: 通过 ridge regression 的泛化差异衡量相似性，与 CCA 有联系

### 4. Neighborhood-Based（近邻）
- **k-NN Jaccard Similarity**: 比较两个表征中每个 instance 的 k 近邻集合的 Jaccard 相似度
- **Second-Order Cosine Similarity**: 比较 instance 到邻居的相似度向量
- **Rank Similarity**: 考虑近邻排序的一致性

### 5. Topology-Based（拓扑）
- **RTD (Representation Topology Divergence)**: 基于 persistent homology 比较表征的拓扑结构
- **IMD (Intrinsic Multi-scale Distance)**: 多尺度拓扑比较

### 6. Descriptive Statistics（描述性统计）
- Magnitude, Concentricity, Uniformity, Tolerance 等单模型统计量

## Functional Similarity Measures 分类

### 1. Performance-Based
- 直接比较 accuracy 等性能指标的差异——简单但粗糙

### 2. Hard Prediction-Based
- **Disagreement/Churn**: 硬预测不一致的比例——最直观
- **Cohen's Kappa**: 修正随机一致性
- **Error-Corrected Disagreement**: 修正模型准确率的影响
- **Label Entropy**: 多模型预测的熵

### 3. Soft Prediction-Based
- **Jensen-Shannon Divergence**: 比较概率分布
- **Prediction Difference**: 多模型预测方差
- **Surrogate Churn**: Disagreement 的软化版本

### 4. Gradient/Adversarial-Based
- **Adversarial Example Transferability**: 对抗样本在模型间的迁移性
- **Saliency Map Cosine Similarity**: 比较梯度 saliency map

### 5. Stitching-Based
- 将一个模型的底层与另一个模型的顶层拼接，训练 stitching layer 后看性能损失

## 关键发现

### 表征 vs 功能相似性的关系
- **不能简单等同**：functional dissimilarity → representational dissimilarity（at some layer），但反之不成立
- 两者得分一般不相关，需要结合使用

### 度量选择建议
- **Functional measures**: 没有根本性的错误选择，推荐用多个 measures 并控制 confounding factors
- **Representational measures**: 选择更困难
  - 先根据 invariance 需求过滤（看 Table 1）
  - CKA 使用广泛但对 perturbation 不敏感（Davari et al. 2022 批评）
  - Orthogonal Procrustes 与 functional behavior 的相关性通常高于 CKA
  - 没有一个 measure 在所有测试中表现最好（ReSi benchmark）

### CKA 的局限
- 可以通过移动单个 instance 的表征来将 CKA 从 1 降到接近 0，而不影响分类性能
- 移除主成分导致 accuracy 下降 15% 时，CKA 仍显示高相似性

## 与本研究的关联

- 提供了衡量表征质量和表征比较的完整工具箱
- Representation similarity 的 invariance 层级（PT ⊂ OT ⊂ ILT ⊂ AT）是理解表征学习目标的重要框架
- 表征学习中常用的 contrastive learning 产生的表征，其相似性度量需要考虑 invariance 选择
- 可用于评估不同表征学习方法学到的表征之间的差异

## 重要参考

- ReSi benchmark: https://github.com/mklabunde/resi（24 个 representational similarity measures 的实现）
- sim_metric: https://github.com/js-d/sim_metric
- similarity-repository: https://github.com/nacloos/similarity-repository
