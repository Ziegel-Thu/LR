# Exp-002 Phase 2: Pythia Scaling Ladder — 有限尺寸标度分析

## 目的

用 4 个不同规模的 Pythia 模型做同样的量化 × SAE 实验，判断 Phase 1 观察到的 3→2 bit 陡降是**真相变**还是**smooth crossover**。

## 背景

Phase 1（Gemma-2 2B）发现：
- 4-bit 几乎无损，3→2 bit 剧烈崩溃（L12: 0.843 → 0.467）
- 但只有一个模型规模，无法做有限尺寸标度分析（Finite-Size Scaling, FSS）
- FSS 是统计物理判断相变的标准方法：如果不同"系统大小"的曲线经 rescale 后 collapse 到一条 master curve，就是真相变

## 假设

1. 如果是真相变：不同 Pythia 规模的 Φ(b) 曲线有共同的临界点 b*，且 susceptibility 峰随模型宽度增长
2. 如果是 smooth crossover：曲线只是平移/缩放，无 collapse，无 susceptibility 峰

## 配置

### 模型
| 模型 | d_model | Layers | 参数量 |
|------|---------|--------|--------|
| Pythia-410M | 1024 | 24 | 410M |
| Pythia-1.4B | 2048 | 24 | 1.4B |
| Pythia-2.8B | 2560 | 32 | 2.8B |
| Pythia-6.9B | 4096 | 32 | 6.9B |

### SAE
| 模型 | SAE 来源 | d_sae |
|------|---------|-------|
| Pythia-410M | EleutherAI/sparsify (如有) 或自训 TopK 32× | 32K |
| Pythia-1.4B | 同上 | 64K |
| Pythia-2.8B | 同上 | 80K |
| Pythia-6.9B | 同上 | 128K |

**注意**：需要先确认 EleutherAI/sparsify 是否提供这些模型的预训练 SAE。如果没有，需要自训或调整为 Gemma Scope 系列（Gemma-2 2B/9B 都有 SAE）。

### 量化
| 参数 | 值 |
|------|-----|
| 方法 | HQQ 权重量化 |
| Group size | 64 |
| Bit-widths | FP16 (baseline), 8, 6, 4, 3, 2 |
| 需要校准数据 | 否 |

### 推理
| 参数 | 值 |
|------|-----|
| Tokens | 2,000,000 per model per bit-width |
| 序列长度 | 1024 |
| 数据集 | The Pile validation |
| Batch size | 根据显存调整 |

### 层选择
每个模型选 5 层（normalized depth α = 0.0, 0.25, 0.5, 0.75, 1.0）：
- 410M (24层): L0, L6, L12, L18, L23
- 1.4B (24层): L0, L6, L12, L18, L23
- 2.8B (32层): L0, L8, L16, L24, L31
- 6.9B (32层): L0, L8, L16, L24, L31

### 度量
1. **Per-feature Pearson r**（FP16 SAE 特征 vs 量化后的 SAE 特征）
2. **按 firing rate 五分位分层**（Q1 最稀有 → Q5 最常见）
3. **Composite order parameter** Φ(b) = mean of (mean_r, frac_above_0.8, CKA) normalized to FP16=1

### 分析
1. **Phase diagram**：4 个模型 × 5 bit-widths × 5 层 的 Φ(b) 曲面
2. **Hill 系数拟合**：Φ(b) = (b/b*)^n / (1+(b/b*)^n)，bootstrap 95% CI on n
   - n > 3：sharp transition
   - n < 3：smooth crossover
3. **Susceptibility**：χ(b) = Var_layers[Φ(b)]，看是否在 b* 处出现峰值
4. **Finite-size scaling collapse**：将 Φ(b, d_model) rescale 为 d^(-α/ν) · F(d^(1/ν)(b - b_c))，拟合 (α, ν, b_c)
   - Bhattacharjee-Seno residual < 0.1 = 成功 collapse
5. **Michaud 频率排序**：按 firing rate 分层的恢复率 vs bit-width，看稀有特征是否一致先崩溃

## 计算量估算

| 项目 | 每 model | 4 models |
|------|---------|----------|
| FP16 推理 + SAE encode (2M tokens) | ~2h on 1×A40 | 8h |
| 每个 bit-width 量化推理 | ~2h | 8h × 5 bits = 40h |
| SAE 分析 | ~0.5h | 2h |
| **总计** | | **~50 A40-hours** |

可以在 jiagpu4-8 上并行：每个模型占 1-2 卡，4 个模型 4 台机器同时跑，1 天完成。

## 硬件分配

| 节点 | 任务 |
|------|------|
| jiagpu4 | Pythia-6.9B (需要 ~28GB 显存，1×A40 够) |
| jiagpu5 | Pythia-2.8B |
| jiagpu6 | Pythia-1.4B |
| jiagpu7 | Pythia-410M |

每台内用 CUDA_VISIBLE_DEVICES 并行跑不同 bit-width。

## 成功标准

**支持相变假说**（至少 3 条满足）：
1. Hill 系数 n 的 95% CI 排除 n < 3
2. Susceptibility χ(b) 在 b* ≈ 2.5-3 bit 处有峰，且峰高随 d_model 增长
3. FSS collapse 的 Bhattacharjee-Seno residual < 0.1
4. b* 在不同层间一致（±0.5 bit）

**反驳相变假说**：
1. Hill n < 3，smooth power-law 拟合优于 step/sigmoid (BIC)
2. 无 susceptibility 峰
3. FSS collapse 失败

**独立于相变假说的有价值发现**：
- Michaud 频率排序是否在所有 4 个模型规模上一致
- b*(d_model) 的 scaling 关系（常数 / log(d) / 饱和）

## 前置条件

1. 确认 Pythia 系列的预训练 SAE 是否可用
2. 如果不可用，需要先训练 SAE（额外 ~200 GPU-hours per model）或改用 Gemma-2 {2B, 9B} + {Gemma Scope SAE} 作为 2-point scaling
