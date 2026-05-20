# Exp-015: OthelloGPT Strategy Probing — Results

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | OthelloGPT Synthetic (8 layers, d=512, ~26M params) |
| 数据 | 4998 自生成随机对局, 20000 positions |
| Probe | sklearn LogisticRegression (linear, max_iter=200) |
| 节点 | jiagpu4 GPU 0 |

## 结果

### 逐层 Probe 准确率

| Layer | Board State (64-sq mean) | Frontier Discs | Mobility R² |
|-------|-------------------------|---------------|------------|
| embed | 0.599 | 0.737 | 0.622 |
| L0 | 0.728 | 0.943 | 0.665 |
| L1 | 0.729 | 0.955 | 0.684 |
| L2 | 0.727 | 0.959 | 0.688 |
| L3 | 0.725 | 0.964 | 0.692 |
| L4 | 0.724 | **0.967** | 0.704 |
| L5 | 0.720 | 0.963 | 0.730 |
| L6 | 0.719 | 0.946 | **0.731** |
| L7 | 0.718 | 0.942 | 0.711 |

### 关键发现

1. **战略概念可线性解码**：Frontier discs 在 L4 达到 96.7% 准确率，Mobility R²=0.73
2. **Board state 线性 probe 约 72%**：与 Li et al. 用 MLP probe 98% 一致（非线性 probe 明显更好）
3. **不同概念的最佳层不同**：
   - Frontier discs 峰值在 L3-L4（中层）
   - Mobility 峰值在 L5-L6（深层）
   - Board state 在 L0-L1 即达峰，之后缓慢下降
4. **Embedding 层已包含显著信息**：frontier 73.7%, mobility R²=0.62

### 与 LLM 对比

OthelloGPT 的 frontier disc probe (96.7%) 远高于 LLM 的 token 特征 probe，
且 board state 的线性 probe 准确率 (72%) 远低于 MLP probe (98%)——
说明 OthelloGPT 的表征更加**非线性**，而 LLM 更偏线性。

## 下一步

- [ ] 用 MLP probe 替代线性 probe，复现 Li et al. 的 98%
- [ ] 添加更多战略概念：tile stability, corner/edge control, disc parity
- [ ] Ablation: 投影掉 frontier/mobility 方向，测下一步预测准确率变化
- [ ] 对比 synthetic vs championship model
