# Exp-006: 表征质量 Scaling Law

## 研究问题

表征的"质量"（跨架构对齐度、几何结构）是否随模型规模呈 power law？有没有 phase transition（"表征涌现"）？

## 动机

- 所有人做 loss scaling law（Chinchilla），**没人做过表征 scaling law**
- Huh 2024 定性展示"bigger → more aligned"但从未拟合幂律
- 我们 exp-003 pilot 有 ~160M 一个 scale 点，需要更多点

## 假设

1. 跨架构对齐度 s(N) 随规模 N 呈 power law: s(N) = s∞ - a·N^{-β}
2. 不同度量（CKA, shape, kNN）可能给出不同的 β
3. 架构族内对齐（SSM↔SSM）和族间对齐（Transformer↔SSM）可能有不同 β

## Phase 1: Pythia 单架构 scaling

### 配置
| 参数 | 值 |
|------|-----|
| 模型 | Pythia 70M / 160M / 410M / 1B / 1.4B / 2.8B / 6.9B |
| 数据 | Pile validation, 2000 sentences, len>50 |
| 序列长度 | 256 tokens |
| 表征 | 每层 residual stream, mean-pooled |
| 度量 | CKA, shape metric, mutual kNN (k=10), TwoNN ID, MLE ID (K=100), stable rank, effective rank |

### 输出
- 每个模型每层的所有度量值
- 度量 vs log(params) 的 scaling 曲线
- Power law 拟合 + R² 评估

## Phase 2: 多架构收敛 scaling

### 配置
| 规模档 | Transformer | SSM (Mamba) | SSM (RWKV) |
|--------|-------------|-------------|------------|
| ~160M | Pythia-160M | Mamba-130M | RWKV-169M |
| ~400M | Pythia-410M | Mamba-370M | RWKV-430M |
| ~1.4B | Pythia-1.4B | Mamba-1.4B | RWKV-1.5B |
| ~2.8B | Pythia-2.8B | Mamba-2.8B | RWKV-3B |

每个规模档：
- 3 模型两两 shape distance / CKA / kNN → 3 对
- 区分族内（Mamba↔RWKV）和族间（Pythia↔Mamba, Pythia↔RWKV）

### 输出
- 跨架构 shape distance vs model scale 曲线
- 族内 vs 族间的 β 对比
- 距离→0？→常数？→发散？

## Phase 3: 与 loss scaling 的关系

- Pythia 各规模的 validation loss 已知
- 画 representation alignment vs loss 的关系
- 是否线性？有没有 representation 领先/落后于 loss？

## 成功标准

- 至少一个度量呈 clean power law (R² > 0.95)
- 族内 vs 族间 β 有 significant 差异
- 或：没有 power law 本身也是发现（表征 scaling 和 loss scaling 质的不同）

## 算力估计

- Phase 1: 7 个模型 × inference → ~8 GPU-hours (A40)
- Phase 2: 12 个模型 × inference → ~16 GPU-hours
- 分析: CPU, 几小时
