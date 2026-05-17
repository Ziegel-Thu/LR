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
