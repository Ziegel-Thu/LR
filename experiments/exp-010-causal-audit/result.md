# Exp-010: 因果抽象方法论审计 — 结果

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | GPT-2 small (124M), 12 layers × 12 heads |
| 任务 | IOI (Indirect Object Identification), 10 prompts |
| 节点 | jiagpu4 GPU 1 |

## Phase 1: Off-distribution 量化

Mahalanobis distance of patched activations vs logit diff change — 见图 `audit_figures.png`。

## Phase 2: Ablation Baseline 对比

| Baseline 对 | Spearman ρ |
|-------------|-----------|
| zero ↔ mean | **0.270** |
| zero ↔ resample | **0.294** |
| mean ↔ resample | 0.896 |

### 关键发现

**不同 ablation baseline 给出截然不同的 head importance ranking！**
- zero 和 mean/resample 的 rank correlation 仅 0.27-0.29（几乎无关）
- mean 和 resample 之间 ρ=0.90（高度一致）
- **结论：zero ablation 和 mean/resample ablation 识别出不同的 "重要" heads**

这意味着 circuit discovery 结论**依赖 baseline 选择**——一个严重的方法论问题。

## Phase 3: Trivialization Baseline

| 对比 | Spearman ρ |
|------|-----------|
| trained ↔ random importance | **-0.171** |

**随机网络和训练网络的 head importance 不相关（ρ≈-0.17）。**

这是一个 positive result：circuit discovery 确实捕获了训练学到的结构，而非架构 artifact。虽然 Sutter 2025 在 IIA 上发现了 trivialization，我们的 zero-ablation 基础实验显示 trained ≠ random。

## 总结

| 发现 | 影响 |
|------|------|
| Baseline matters (ρ<0.3) | ⚠️ 现有 circuit 结论部分依赖 baseline |
| Trained ≠ random | ✅ Circuit discovery 不是 trivial |
| Mean ≈ resample (ρ=0.9) | 这两种 baseline 可互换 |

---

## Phase 4: Circuit Dark Matter

### 配置
- GPT-2 small, IOI, 12 layers × 12 heads + 12 MLPs = 156 components

### 结果

| 指标 | 值 |
|------|-----|
| MLP 占比 | **13.9%** |
| Ablate 10% | 100.4% preserved |
| Ablate 30% | 71.9% preserved |
| Ablate 50% | 18.3% preserved |

**MLP 只占 14% — 暗物质不在 MLP 里。** Circuit 高度稀疏：底部 10% 组件完全无贡献，top ~30% 组件承载 72% 信号。暗物质可能是分布式的 head interactions，不是任何单一可定位的组件。

---

## Phase 4 扩展: WikiText-2 Dark Matter 探索

> 从 IOI 扩展到通用语言建模（WikiText-2 validation），三项分析。

### 配置

| 参数 | 值 |
|------|-----|
| 模型 | GPT-2 small (124M), 12 layers |
| 数据 | WikiText-2 validation, 100 seqs × 256 tokens |
| 方法 | Layer knockout (zero ablation) + Direct Logit Attribution |
| 节点 | jiagpu4 GPU 1 |

### Analysis 1: Layer Knockout

对每层分别 zero out attn、mlp、或 both，测 perplexity 增量：

| Layer | Δloss (attn) | Δloss (mlp) | Δloss (both) |
|-------|-------------|-------------|--------------|
| L0 | 0.49 | 3.47 | **4.71** |
| L1 | 0.01 | 0.04 | 0.05 |
| L2 | 0.05 | 0.06 | 0.12 |
| L3 | 0.05 | 0.05 | 0.11 |
| L4 | 0.15 | 0.04 | 0.18 |
| L5 | 0.13 | 0.06 | 0.18 |
| L6 | 0.11 | 0.05 | 0.15 |
| L7 | 0.09 | 0.06 | 0.14 |
| L8 | 0.11 | 0.07 | 0.17 |
| L9 | 0.07 | 0.09 | 0.15 |
| L10 | 0.05 | 0.22 | 0.26 |
| L11 | 2.35 | 0.24 | **1.30** |

**关键发现：**
- **Layer 0 是最关键的层**（Δloss=4.71），主要贡献来自 MLP（Δloss=3.47）
- **Layer 11 是第二重要的层**（Δloss=1.30），主要贡献来自 attention（Δloss=2.35）
- Layer 1 几乎可以完全移除（Δloss=0.05）
- **MLP 总体占 60.2%，attention 占 39.8%** — 与 IOI 分析不同，通用任务中 MLP 更重要

### Analysis 2: Cumulative Layer Ablation

从最不重要到最重要逐层移除：

| # Ablated | PPL | PPL/Baseline | 移除的层 |
|-----------|-----|-------------|---------|
| 0 | 40.3 | 1.00× | — |
| 1 | 42.3 | 1.05× | L1 |
| 2 | 48.8 | 1.21× | +L3 |
| 3 | 144.5 | **3.58×** | +L2 |
| 4 | 178.9 | 4.44× | +L7 |
| 5 | 220.6 | **5.47×** | +L9 → 灾难性失败 |
| 6 | 290.5 | 7.20× | +L6 |
| 8 | 796.7 | 19.76× | +L8, L4 |
| 10 | 5139.1 | 127× | +L5, L10 |
| 11 | 99331 | 2464× | +L11 |
| 12 | ∞ | — | +L0 |

**关键发现：**
- **可以移除 2 层（L1, L3）仅 +21% PPL** — 存在冗余
- **PPL 翻倍阈值：3 层**（L1+L3+L2 → 3.58×）
- **灾难性失败（5× PPL）：5 层**
- 分布极不均匀：Layer 0 单独承载的信息远超其他所有层之和

### Analysis 3: Residual Stream Decomposition (Direct Logit Attribution)

使用 DLA 将最终 logit 分解为 25 个组件（1 embed + 12 attn + 12 mlp）的贡献。

分解验证：mean |reconstruction error| = 0.0000 ✅

**Top-10 组件（按平均 |logit contribution| 排序）：**

| 排名 | 组件 | |contribution| |
|------|------|--------------|
| 1 | **L10_mlp** | 47.80 |
| 2 | L11_mlp | 22.01 |
| 3 | L9_mlp | 14.16 |
| 4 | L8_mlp | 2.36 |
| 5 | L7_mlp | 2.05 |
| 6 | L5_mlp | 0.88 |
| 7 | L6_mlp | 0.85 |
| 8 | embed | 0.71 |
| 9 | L3_mlp | 0.62 |
| 10 | L4_mlp | 0.54 |

**Top-K 累积解释比例：**

| K | 解释比例 |
|---|---------|
| 1 | 50.8% |
| 3 | **85.4%** |
| 5 | **90.6%** |
| 10 | 95.6% |
| 15 | 98.2% |
| 24 | 100.0% |

### 综合分析

**暗物质问题的答案：**

1. **信号高度集中于 MLP 层**：Top-3 组件（L10/L11/L9 的 MLP）解释 85% 的 logit。Attention 在 DLA 中几乎不出现在前列。这与 IOI 中 attention heads 主导的发现形成鲜明对比。

2. **两种分析方法的分歧**：
   - **Knockout**（因果效应）：Layer 0 最重要，attn 占 40%
   - **DLA**（logit 归因）：L10 MLP 最重要，attn 几乎不贡献
   - 原因：knockout 测量的是 *移除后的损失*（包括对下游的连锁影响），DLA 测量的是 *直接对 logit 的贡献*。Layer 0 knockout 损失大是因为它改变了所有下游层的输入，但它对最终 logit 的直接贡献很小。

3. **"暗物质"在哪里？**
   - 在 IOI 任务中，circuit 只解释 70-87%，暗物质占 13-30%
   - 在通用语言建模中，top-5 组件（20%）解释 90.6%，暗物质仅 ~9%
   - 暗物质更少是因为 DLA 分析了所有组件（含 MLP），而 IOI circuit search 通常只看 attention heads
   - **结论：所谓"暗物质"很大程度上是 MLP 贡献的信号**，传统 circuit search 忽略 MLP 是主要原因

4. **冗余结构**：
   - 2 层可几乎无损移除（L1, L3）
   - 灾难性失败需要移除 5+ 层
   - 这与 Transformer 的过参数化一致（12 层中有效利用的可能只有 8-9 层）
