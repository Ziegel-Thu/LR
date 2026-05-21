# exp-022: Full Cross-Architecture Procrustes Similarity Map

## 目的

用 Procrustes distance 构建 16 个模型（Pythia / Mamba / RWKV）之间的完整相似度矩阵，作为 exp-003 CKA 分析的补充。Procrustes distance 考虑旋转不变性，是更 principled 的表征比较度量。

## 假设

1. 同架构家族内部 Procrustes 距离更小（architecture clustering）
2. 模型规模增大时，跨架构距离减小（convergence hypothesis）
3. Procrustes distance 与 CKA 高度相关（如果是，CKA 是好的 proxy）

## 数据

exp-003 表征文件，jiagpu4 SSD `/nvmessd/lifanhong/LR-env/exp003_reps/`。
每个文件形状 (n_layers, 1728, d_model)。

## 方法

对每对模型 (A, B)：
1. 取中间层 (layer n_layers // 2)
2. 随机采样 500 个句子
3. 如果 d_model 不同，用 PCA 投影到 min(d1, d2) 维
4. 计算 orthogonal Procrustes distance：中心化+归一化后求最优旋转
5. 同时计算 linear CKA 做对比

## 输出

- `procrustes_matrix.npy` — N×N Procrustes distance matrix
- `cka_matrix.npy` — N×N CKA similarity matrix
- `model_names.json` — 模型名称列表
- `results_summary.json` — 聚合统计

## 执行

- 节点：jiagpu4（CPU only，数据已在 SSD）
- 预计时间：~5 分钟（16×16=256 对，每对很快）
