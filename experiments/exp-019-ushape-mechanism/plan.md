# exp-019: U-shape Depth Curve Mechanism Analysis

## 目的

解释表征相似性（CKA、kNN alignment）在深度方向上呈现 U 形曲线的机制。这一现象在所有模型族（Transformer、SSM、RWKV）和所有规模上都被观察到，但至今没有令人信服的解释。

## 假设

- **H1: Bottleneck compression** — 中间层压缩信息（ID/rank 下降），早/晚层保持较高维度
- **H2: Task specialization** — 中间层发展任务特定特征（与输入/输出差异大），早/晚层保留共享结构
- **H3: Residual stream dominance** — 早/晚层被 residual stream 主导（与 input/output 相似），中间层加入最多新信息

## 数据

使用 exp-003 提取的表征文件（jiagpu4 SSD `/nvmessd/lifanhong/LR-env/exp003_reps/`），覆盖：
- Pythia 系列：70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B
- Mamba 系列：130M, 370M, 1.4B, 2.8B
- RWKV 系列：169M, 430M, 1.5B, 3B, 7B

每个文件形状 (n_layers, 1728, d_model)，取 500 句子子集分析。

## 分析指标（per-layer）

1. **Intrinsic Dimensionality (TwoNN)** — 表征的内在维度
2. **Effective Rank (Stable Rank)** — ||M||_F² / ||M||_2²
3. **Consecutive CKA** — CKA(layer_i, layer_{i+1})，每层变化幅度
4. **Input Similarity** — CKA(layer_i, layer_0)
5. **Output Similarity** — CKA(layer_i, layer_last)
6. **Isotropy** — 随机 pair 的平均 cosine similarity

## 方法

- CPU-only 分析，jiagpu4 运行
- 500 句子子集加速计算
- 所有结果保存为 JSON，支持后续绘图

## 节点

jiagpu4
