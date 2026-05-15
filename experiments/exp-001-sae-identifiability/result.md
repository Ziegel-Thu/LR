# Exp-001: SAE 可识别性 Pilot 实验结果

## Run 1: Baseline (Pythia-160M, 4K latents, 2M tokens)

### 配置
| 参数 | 值 |
|------|-----|
| 模型 | EleutherAI/pythia-160m-deduped |
| 层 | Layer 6 MLP output |
| d_model | 768 |
| d_sae | 4096 (expansion 5.3×) |
| SAE 架构 | TopK |
| K | 32 |
| 训练 tokens | 2,000,000 |
| batch size | 4096 |
| epochs | 3 |
| lr | 3e-4 |
| optimizer | Adam |
| 种子数 | 5 (seed 0-4) |
| 设备 | MPS (M4 Max) |
| dead feature resampling | 无 |

### 训练质量
| 种子 | NMSE | Mean L0 | Final Loss |
|------|------|---------|------------|
| 0 | 0.1184 | 32.0 | 0.01595 |
| 1 | 0.1191 | 32.0 | 0.01604 |
| 2 | 0.1179 | 32.0 | 0.01592 |
| 3 | 0.1194 | 32.0 | 0.01607 |
| 4 | 0.1183 | 32.0 | 0.01597 |

5 个 SAE 重建质量几乎相同（NMSE ~0.119, L0 = 32.0），训练本身是稳定的。

### 跨种子特征共享率（10 对比较）
| 指标 | 均值 | 标准差 |
|------|------|--------|
| MMCS | 0.170 | ±0.001 |
| Hungarian matching (cos>0.7) | 1.36% | ±0.16% |
| Mutual NN fraction | 40.0% | ±0.5% |

### 诊断

**Dead feature 分析：**
- 4096 个 latents 中只有 ~970 个激活过（**dead rate ~75%**）
- Dead features 的字典方向是随机的，严重拉低 MMCS

**Alive-only 共享率（seed 0 vs seed 1）：**
| 指标 | 值 |
|------|-----|
| Alive features | seed0: 971, seed1: 992 |
| Alive MCS mean | 0.279 |
| Alive cos>0.7 | 45/971 (4.6%) |
| Alive cos>0.5 | 174/971 (17.9%) |
| Alive cos>0.3 | 347/971 (35.7%) |

**MCS 分布：**
- 长尾分布，非双峰
- 大部分集中在 ~0.1（接近随机水平）
- 少量高共享特征（max=0.98）
- 见 `figures/mcs_distribution.png`

### 结论

共享率极低，主要原因是 **dead features 占 75%** 和 **训练数据不足**（2M tokens）。Alive features 的共享率稍好（cos>0.5 约 18%），但仍远低于 Paulo & Belrose 的 30%。

### 下一步 → Run 2
- 减小字典：d_sae=1536（expansion 2×），减少 dead features
- 增加数据：5M tokens（从 2M 增加到 5M）
- 增加 epochs：5（从 3 增加到 5）

---

## Run 2: 更小字典 + 更多数据 (Pythia-160M, 1536 latents, 5M tokens)

### 配置
| 参数 | 值 | 与 Run 1 的变化 |
|------|-----|----------------|
| 模型 | EleutherAI/pythia-160m-deduped | 不变 |
| 层 | Layer 6 MLP output | 不变 |
| d_model | 768 | 不变 |
| d_sae | **1536** (expansion 2×) | ↓ 从 4096 |
| SAE 架构 | TopK | 不变 |
| K | 32 | 不变 |
| 训练 tokens | **5,000,000** | ↑ 从 2M |
| batch size | 4096 | 不变 |
| epochs | **5** | ↑ 从 3 |
| lr | 3e-4 | 不变 |
| optimizer | Adam | 不变 |
| 种子数 | 5 (seed 0-4) | 不变 |
| 设备 | MPS (M4 Max) | 不变 |
| 训练方式 | 串行（内存安全） | 从并行改串行 |

### 跨种子特征共享率（10 对比较）
| 指标 | Run 2 均值 | Run 1 均值 | 变化 |
|------|-----------|-----------|------|
| MMCS | **0.238** ±0.002 | 0.170 ±0.001 | ↑ +40% |
| Hungarian (cos>0.7) | **6.62%** ±0.43% | 1.36% ±0.16% | ↑ ×4.9 |
| Mutual NN | 36.3% ±1.0% | 40.0% ±0.5% | ↓ 略降 |

### 分析

**共享率有显著提升**，但仍然很低：

1. **Hungarian 从 1.4% → 6.6%**：减小字典 + 增加数据确实降低了 dead features 的干扰，但 1536 个特征中仍然只有约 100 个是跨种子高度共享的
2. **MMCS 从 0.17 → 0.24**：有改善但远未饱和
3. **Mutual NN 略降（40% → 36%）**：可能因为字典更小，每个方向更"挤"，互为最近邻的阈值更严

**与 Paulo & Belrose (~30% at cos>0.7) 的差距仍然很大**，说明 Pythia-160M 可能模型太小、或 5M tokens 仍然不足。

### 下一步方向

1. **继续增大数据**：50M tokens（需要分块加载避免内存溢出）
2. **换更大模型**：Pythia-1.4B（需要服务器）
3. **尝试 ReLU+L1 架构**：Paulo & Belrose 发现 ReLU+L1 比 TopK 更稳定
4. **加 dead feature resampling**：训练中定期重采样不活跃特征
