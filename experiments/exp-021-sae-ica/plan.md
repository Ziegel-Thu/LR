# Exp-021: SAE 作为经验 ICA — 桥接 SAE 实践和可识别性理论

## 研究问题

SAE（Sparse Autoencoder）分解的特征是否满足统计独立性？如果是，SAE 可被视为一种经验 ICA（Independent Component Analysis），为 SAE 提供理论根据。

## 动机

- SAE 将神经激活分解为稀疏特征，ICA 将信号分解为独立成分
- 两者的联系少有人探讨：稀疏性是否暗示独立性？
- 如果 SAE features 统计独立 → SAE 是一种经验 ICA → 和 identifiability 理论（Hyvärinen 等）接轨
- 如果不独立 → SAE 找到的是别的结构（如 sparse coding）

## 假设

1. SAE 特征的 pairwise MI 低于 PCA 成分（PCA 只保证不相关）
2. SAE 特征的 MI 接近 FastICA 成分（ICA 显式优化独立性）
3. 更高稀疏性 → 更低 pairwise MI（sparsity 促进 independence）

## 方法

### 模型与数据
| 参数 | 值 |
|------|-----|
| 模型 | Pythia-160M (d_model=768, 12 layers) |
| 目标层 | Layer 6 (middle) |
| 数据 | WikiText-103 validation, ~500K tokens |
| 节点 | jiagpu4 GPU 1 |

### Phase 1: 提取激活
- 复用 exp-008 缓存的 Pythia-160M layer 6 激活（235K tokens）
- 或重新缓存 500K tokens（如需更大样本）

### Phase 2: 训练多个 SAE（L1 系数扫描）
- 架构: TopK SAE, d_sae=4096, K=32（同 exp-008）
- L1 系数: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
- 每个 20K steps, Adam lr=3e-4

### Phase 3: 基线方法
- **PCA**: 全 768 主成分（正交但非独立）
- **FastICA**: 128 独立成分（显式优化独立性）
- **Random projection**: 128 随机方向（零假设）

### Phase 4: 独立性分析
对每种分解方法，计算：

| 指标 | 含义 |
|------|------|
| Mean \|Pearson r\| | 线性相关性（2000 对抽样） |
| Mean pairwise MI | 非线性依赖（500 对抽样, histogram-based） |
| Sparsity | 零激活比例 |

### Phase 5: 可视化
- Sparsity vs MI 曲线（SAE L1 扫描）
- 方法对比柱状图
- 相关系数分布直方图

## 预期结果

| 方法 | 相关性 | MI | 原因 |
|------|--------|-----|------|
| PCA | ≈0 (构造) | 中等 | 正交≠独立 |
| FastICA | ≈0 | 低 | 优化独立性 |
| Random | 低 | 低 | 随机方向弱依赖 |
| SAE (低L1) | ? | ? | 重建优先 |
| SAE (高L1) | ? | 低? | 稀疏≈独立? |

## 成功标准

- 完成分析 = 有定量对比表和图
- 正面结果 = SAE 的 MI 随 L1 降低，趋近 ICA 水平
- 负面结果 = MI 不随 L1 变化，SAE ≠ ICA（同样有意义）

## 算力估计

- 激活缓存: ~15 min
- SAE 训练: ~8 min × 5 = ~40 min
- PCA/ICA: ~5 min
- 分析: ~10 min
- 总计: ~1.5 GPU-hours
