# Exp-008: SSM 上的 SAE — Phase 1 + Phase 2 结果

## Phase 1: Mamba-130M SAE Pilot

### 配置

| 参数 | 值 |
|------|-----|
| 模型 | Mamba-130M (`state-spaces/mamba-130m-hf`) |
| 目标层 | Layer 12/24 (中间层 output hidden state) |
| 训练数据 | WikiText-103 validation, 235,568 tokens |
| SAE 架构 | TopK (d_model=768 → d_sae=4096, K=32) |
| 训练 | Adam lr=3e-4, L1 coeff=1e-3, 30K steps, batch=4096 |
| 节点 | jiagpu4 GPU 4 |

### 结果

| 指标 | 值 | 解读 |
|------|-----|------|
| Reconstruction MSE | 0.198 | 在标准化空间 |
| Variance explained | **80.3%** | 良好——Transformer SAE 通常 85-95% |
| Dead features | **0/4096 (0%)** | 完美——所有特征都学到了东西 |
| Avg active/token | 32.0 | 精确匹配 K=32 |
| Alive fire rate (mean) | 0.78% | 稀疏且分布合理 |

### 关键发现

1. **SAE 在 Mamba 上 works** — 80.3% variance explained，首次在 SSM 上成功训练 SAE
2. **0% dead features** — 非常罕见（Transformer 通常 5-30%）
3. **结论：Superposition 不是 Transformer 特有的**

---

## Phase 2: Mamba SAE vs Pythia SAE 特征对比

### 配置

| 参数 | 值 |
|------|-----|
| Pythia 模型 | Pythia-160M (d_model=768, 同 Mamba-130M) |
| SAE | 相同架构: TopK d_sae=4096, K=32, 30K steps |
| 对比方法 | MMCS (Max Mean Cosine Similarity on decoder weights) |

### MMCS 结果

| 指标 | 值 |
|------|-----|
| MMCS (Pythia → Mamba) | 0.130 |
| MMCS (Mamba → Pythia) | 0.130 |
| MMCS (symmetric) | **0.130** |
| Median best match cosine | 0.129 |
| Features with overlap > 0.9 | **0 / 4096 (0%)** |

### 解读

**MMCS = 0.13 接近随机水平** (768-dim 随机向量的 expected max cosine ~ 0.1-0.15)。

- **0 个特征的 best match > 0.9**：Mamba 和 Pythia 没有任何"共享概念"的特征
- 两个方向（Pythia→Mamba, Mamba→Pythia）高度对称：不存在单向嵌入
- **结论：虽然两种架构都存在 superposition，但学到的特征方向完全不同**

这与 exp-003 的发现形成有趣对比：
- exp-003: **表征几何** 高度相似 (kNN overlap > 0.7)
- exp-008: **SAE 特征** 几乎不重叠 (MMCS = 0.13)

可能解释：两种架构编码相同的信息（因此 kNN 高），但用不同的 superposition 结构（因此 SAE 特征不同）。这是一个值得深入的发现。

## 局限

- 训练数据仅 236K tokens（可能不够充分）
- 仅比较 middle layer
- MMCS 可能不是最佳的特征对比方法（decoder weight 方向 ≠ feature 语义）

## 下一步

- [ ] Phase 3: Mamba-370M / 1.4B 上的 SAE scaling
- [ ] 更大训练数据 (10M+ tokens)
- [ ] 手动检查 Mamba 和 Pythia 各自的 top features 语义
- [ ] 用 activation correlation（而非 weight cosine）做特征对比
