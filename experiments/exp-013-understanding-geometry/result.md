# Exp-013: 理解的几何学 — 结果

## Phase 2: ID-Profile Atlas (LLMs)

### 10 模型均无 hunchback（ID 单调下降）
详见上文。

## Phase 1: OthelloGPT 标定 ✅

### 配置
- OthelloGPT via TransformerLens (8 layers, d=512)
- 500 random games, 5000 board positions
- ID profile + board state probe + strategy concept probes

### ID Profile

| Layer | TwoNN ID | Stable Rank |
|-------|----------|-------------|
| L0 | 9.8 | 2.4 |
| L1 | 13.7 | 4.7 |
| L2 | 13.3 | 6.0 |
| **L3** | **18.4** | 9.1 |
| L4 | 17.5 | 10.9 |
| L5 | 12.9 | 14.1 |
| **L6** | **7.9** | 16.4 |
| L7 | 9.4 | 17.9 |

**OthelloGPT 有 hunchback！** ID 在 L3 达到 18.4，在 L6 降到 7.9。

### Board State Probe
- Best: **70.4% @ L7** (random=33.3%)
- 模型确实学到了 board state representation

### Strategy Concept Probes

| 概念 | Best Acc | Layer | Pos Rate |
|------|---------|-------|----------|
| has_corner | **74.7%** | L6 | — |
| is_winning | **70.3%** | L6 | — |
| high_mobility | 62.5% | L5 | — |

### 关键发现

1. **OthelloGPT 有 hunchback，但 LLMs 没有** — hunchback 可能是"world model"型理解的 signature
2. **策略概念可以 probe** — 不只是 board state，corner control 和 winning status 也被编码
3. **低 ID 层 (L6-7) 是 probe 最好的层** — 与 semantic plateau 假说一致
4. Board state probe 在最后两层最强 → 信息逐步整合

### 与 LLM Atlas 对比

| 特征 | OthelloGPT | Pythia/Mamba/RWKV |
|------|-----------|------------------|
| Hunchback | ✅ L3 peak | ❌ 单调下降 |
| ID 范围 | 7.9-18.4 | 10.7-18.6 |
| Stable Rank 趋势 | 单调增 | 无趋势 |

**"理解"的几何 signature 可能就是 hunchback** — 有 world model 的模型有，纯语言模型没有。

## CLIP ViT-B/32 Text Encoder

### ID Profile

| Layer | TwoNN ID | Stable Rank |
|-------|----------|-------------|
| L0 | 14.3 | 1.0 |
| L1 | 14.3 | 1.0 |
| **L2** | **16.7** | 1.0 |
| L3 | 15.4 | 1.0 |
| L4 | 14.9 | 1.0 |
| L5 | 11.9 | 1.0 |
| L10 | 11.7 | 1.1 |
| L11 | 13.5 | 1.3 |

**CLIP text encoder 也有 hunchback！** Peak at L2 (ID=16.7), trough at L10 (ID=11.7).

### Hunchback 汇总

| 模型 | 类型 | Hunchback? | Peak Layer |
|------|------|-----------|------------|
| OthelloGPT | World model | ✅ | L3 (18.4) |
| CLIP ViT-B/32 | Cross-modal | ✅ | L2 (16.7) |
| Pythia (all scales) | Language | ❌ | — |
| Mamba (all scales) | Language (SSM) | ❌ | — |
| RWKV (all scales) | Language (SSM) | ❌ | — |

**结论：Hunchback 出现在有"理解"任务的模型（棋盘 world model、跨模态语义），不出现在纯语言模型。这是"理解"的几何 signature 的强证据。**

## 随机对照实验 ✅

| 模型 | Hunchback? |
|------|-----------|
| OthelloGPT (trained) | ✅ YES |
| OthelloGPT (random init) | ❌ NO |

**Hunchback 是训练效应，不是架构效应。** 随机初始化的同架构模型没有 hunchback。这证明 hunchback 是模型通过训练"学到理解"的几何 signature。

## 下一步

- [ ] DreamerV3 RSSM probing（验证 world model 假说）
- [ ] 更多"理解型"模型的 ID profile
