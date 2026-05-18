# Exp-013: 理解的几何学 — ID-Profile Atlas 结果

## 配置

10 个模型（Pythia 4 + Mamba 3 + RWKV 3），复用 exp-003/006 的 reps。
每层计算 TwoNN ID + MLE ID + stable rank。

## Hunchback 分析

| 模型 | Early ID | Mid ID | Late ID | Hunchback? |
|------|----------|--------|---------|------------|
| Pythia-70M | 18.6 | 15.6 | 13.9 | ❌ |
| Pythia-410M | 17.5 | 13.8 | 10.9 | ❌ |
| Pythia-1.4B | 17.4 | 15.0 | 11.2 | ❌ |
| Pythia-6.9B | 17.9 | 13.6 | 11.0 | ❌ |
| Mamba-370M | 15.8 | 15.3 | 11.2 | ❌ |
| Mamba-1.4B | 17.7 | 15.3 | 10.9 | ❌ |
| Mamba-2.8B | 15.8 | 15.8 | 11.0 | ❌ |
| RWKV-430M | 17.0 | 13.7 | 10.7 | ❌ |
| RWKV-1.5B | 16.5 | 13.9 | 10.6 | ❌ |
| RWKV-4-3B | 16.6 | 12.8 | 10.7 | ❌ |

## 关键发现

**没有一个模型呈现经典 hunchback（中间层 ID 高于首尾）。**

所有 10 个模型的 ID 都**单调下降**（early > mid > late）。这与 Valeriani 2023 在蛋白质 LM 上发现的 semantic plateau 不同。

### 可能原因
1. **数据类型**：WikiText 文本 vs 蛋白质序列 — hunchback 可能是蛋白质特有的
2. **度量选择**：TwoNN ID 在高维(>1000)文本表征上可能不如 MLE 敏感
3. **Mean pooling**：我们用的是 mean-pooled 表征，不是 per-token — pooling 可能抹平了 hunchback
4. **所有架构一致**：Transformer/Mamba/RWKV 都单调下降 — 这是 universal pattern

### 跨架构比较
- Pythia 和 RWKV 的 ID profile 形状相似（单调下降）
- Mamba 中间层 ID 相对平坦（early ≈ mid），late 才下降 — 可能反映递推结构的不同

## 生成图表
- `id_atlas.png` — 10 模型的 TwoNN/MLE/stable rank 深度曲线

## 下一步
- [ ] Phase 1: OthelloGPT 标定（ground truth 已知）
- [ ] 用 per-token 表征（不做 mean pool）重新分析
- [ ] 加入 manifold capacity / curvature（DR-016 建议）
