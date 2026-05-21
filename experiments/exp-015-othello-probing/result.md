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

---

## Phase 2: MLP Probes + Ablation

### 配置

| 参数 | 值 |
|------|-----|
| Probe | 2-layer MLP: Linear(512→256) → ReLU → Linear(256→n) |
| 优化器 | Adam, lr=1e-3, 20 epochs, batch_size=256 |
| 数据 | 同 Phase 1 (4998 games, 20000 positions) |
| 节点 | jiagpu4 GPU 0 |

### MLP Probe 结果

| Layer | Board State | Frontier | Mob. R² | Corner | Edge R² | Disc Adv. R² |
|-------|------------|----------|---------|--------|---------|-------------|
| embed | 0.616 | 0.734 | 0.622 | 0.848 | 0.726 | -0.012 |
| L0 | 0.836 | 0.972 | 0.655 | **0.971** | **0.795** | 0.029 |
| L1 | 0.860 | 0.980 | 0.673 | 0.965 | 0.797 | 0.075 |
| L2 | 0.872 | 0.980 | 0.674 | 0.960 | 0.794 | 0.062 |
| L3 | **0.877** | 0.983 | 0.678 | 0.957 | 0.794 | 0.098 |
| L4 | 0.873 | **0.983** | 0.694 | 0.951 | 0.772 | **0.101** |
| L5 | 0.859 | 0.982 | 0.719 | 0.945 | 0.763 | 0.035 |
| L6 | 0.824 | 0.977 | **0.724** | 0.937 | 0.746 | 0.025 |
| L7 | 0.721 | 0.971 | 0.712 | 0.934 | 0.722 | -0.007 |

### Ablation (Encoding ≠ Use)

在最佳 probe 层投影掉 probe 子空间，测 next-move prediction 变化：

| Concept | Layer | Baseline Acc | Ablated Acc | Δacc |
|---------|-------|-------------|------------|------|
| Frontier | L4 | 0.116 | 0.107 | **-0.010** |
| Mobility | L6 | 0.116 | 0.119 | +0.003 |

### 关键发现

1. **MLP probe 大幅提升 board state**：L3 达到 87.7%（线性 72.5% → MLP 87.7%），
   但仍低于 Li et al. 的 98%——可能因为我们用随机对局而非真实对局数据
2. **新战略概念可解码**：
   - Corner occupancy: 97.1% @ L0（非常早就编码了）
   - Edge control: R²=0.80 @ L0-L1
   - Disc advantage: R²仅 0.10，模型几乎不编码原始盘面差
3. **Board state 峰值移到 L3**（线性时在 L0），说明 MLP 能捕获更高层的非线性表征
4. **Ablation 结果**：
   - Frontier 投影掉后 next-move acc 下降 1%——模型确实使用 frontier 信息
   - Mobility 无显著变化——编码但未被因果使用（ghost representation）
   - 基线 next-move acc 仅 11.6%（随机对局难预测下一步）

---

## Phase 3: Championship vs Synthetic + Tile Stability Probing

### 配置

| 参数 | 值 |
|------|-----|
| 模型 | Synthetic + Championship (同架构 8L/d=512) |
| 数据 | 4998 随机对局, 20000 positions（两模型共用） |
| Probe | Linear (LogisticRegression / Ridge) |
| Stability | Directional-anchor 算法（4 轴迭代传播） |
| 节点 | jiagpu4 GPU 0 |

### Synthetic vs Championship 逐层对比

| Layer | Board State |  | Frontier |  | Mobility R² |  | Stability |  |
|-------|:-----------:|:-:|:--------:|:-:|:-----------:|:-:|:---------:|:-:|
|       | Synth | Champ | Synth | Champ | Synth | Champ | Synth | Champ |
| L0 | 0.728 | **0.736** | **0.943** | 0.879 | **0.665** | 0.639 | **0.984** | 0.973 |
| L1 | 0.729 | **0.739** | **0.955** | 0.904 | **0.684** | 0.648 | **0.985** | 0.976 |
| L2 | 0.727 | **0.732** | **0.959** | 0.906 | **0.688** | 0.643 | **0.985** | 0.977 |
| L3 | **0.725** | 0.725 | **0.964** | 0.903 | **0.692** | 0.640 | **0.986** | 0.976 |
| L4 | **0.724** | 0.718 | **0.967** | 0.895 | **0.704** | 0.627 | **0.985** | 0.975 |
| L5 | **0.720** | 0.708 | **0.963** | 0.885 | **0.730** | 0.614 | **0.983** | 0.974 |
| L6 | **0.719** | 0.701 | **0.946** | 0.876 | **0.731** | 0.604 | **0.982** | 0.973 |
| L7 | **0.718** | 0.694 | **0.942** | 0.866 | **0.711** | 0.600 | **0.979** | 0.972 |
| embed | 0.599 | 0.598 | 0.737 | 0.737 | 0.622 | 0.622 | 0.970 | 0.970 |

### Best-layer 汇总

| Concept | Synth (best) | Champ (best) | Δ | Winner |
|---------|:------------:|:------------:|:---:|:------:|
| Board State | 0.729 @ L1 | **0.739 @ L1** | +0.010 | **CHAMP** |
| Frontier | **0.967 @ L4** | 0.906 @ L2 | −0.061 | **SYNTH** |
| Mobility R² | **0.731 @ L6** | 0.648 @ L1 | −0.083 | **SYNTH** |
| Stability | **0.986 @ L3** | 0.977 @ L2 | −0.009 | **SYNTH** |

### 关键发现

1. **假设被推翻**：Championship 模型的战略表征并不比 Synthetic 更好。
   除 board state 微弱领先 (+1%) 外，Frontier/Mobility/Stability 全面落后于 Synthetic。

2. **Board state 是 championship 唯一优势**：Champ L1 = 73.9% vs Synth L1 = 72.9%，
   但差距很小。Championship 训练在更窄的棋局分布上（强手对局），可能导致 probe
   在随机对局数据上泛化更差。

3. **Synthetic 的 frontier/mobility 优势显著**：
   - Frontier: 96.7% vs 90.6%（Δ=6.1%）
   - Mobility: R²=0.731 vs 0.648（Δ=0.083）
   - 随机对局 mobility 变化更大，迫使模型更充分编码这些特征

4. **Tile stability 探针结果极好**：
   - Synthetic 最佳 98.6% @ L3, Championship 97.7% @ L2
   - 几乎在所有层都 >97%——stability 是一个非常"线性"的特征
   - 64 个方格全部有足够方差用于训练 probe（稳定性在整盘都有分布）

5. **Embedding 层两模型完全一致**：因为共享相同的 tokenization，
   embedding 层差异 <0.001——验证了实验一致性

6. **Championship 的表征更浅**：其最佳层通常在 L1-L2，
   而 Synthetic 的最佳层分布更广（L1-L6）——暗示 championship 模型
   在浅层就确定策略，深层表征退化更快

## 下一步

- [x] ~~对比 synthetic vs championship model~~
- [ ] 用 championship 对局数据（而非随机对局）重新评估两模型
- [ ] Stability ablation: 投影掉 stability 方向，测 next-move Δ
- [ ] 更多 ablation targets（board state, corner）
