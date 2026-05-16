# Exp-002: 量化对 SAE 特征的选择性破坏

## 研究问题

量化（降低模型数值精度）是否会选择性地杀死某些 SAE 特征？如果是，哪些特征先死？退化曲线是 smooth 还是存在相变？

## 背景

- 大模型部署需要量化（FP16→INT4/INT2），业界关注整体性能下降，但没人研究过量化对模型内部**可解释特征**的影响
- Michaud et al. (arXiv:2303.13506) 的 quantization model 预测：模型按使用频率给特征分配表征预算，**稀有特征先崩溃**
- Elhage et al. (arXiv:2209.10652) 展示了 superposition 中特征按几何结构排列，量化降低精度可能破坏这种精细结构
- Liu et al. (arXiv:2604.19884) 描述了 4→2 bit 的定性"悬崖"，但没有提取临界指数或做有限尺寸标度分析

## 假设

1. 随着 bit-width 降低，SAE 特征恢复率单调下降
2. **稀有特征先崩溃**（Michaud 预测）——低 firing rate 的特征恢复率下降更快
3. 可能存在临界 bit-width b*，在其附近退化加速（sharp transition vs smooth crossover 待定）

## 已完成：Pilot（本地，激活量化模拟）

### 配置
| 参数 | 值 |
|------|-----|
| 模型 | Pythia-160M, Layer 6 MLP, d_model=768 |
| SAE | exp-001 seed 0, TopK, d_sae=4096, K=32 |
| 方法 | 对激活做 per-channel 均匀量化 + 匹配 MSE 高斯噪声对照 |
| Bit-widths | 16, 8, 6, 4, 3, 2, 1 |
| 数据 | 2M tokens 已缓存激活 |

### Pilot 结果
- **频率分层效应非常强**：4-bit 时 low=42% vs high=95%；2-bit 时 low=4.5% vs high=71.5%
- 退化曲线 smooth，无 sharp transition（在激活量化下）
- 量化 ≈ 高斯噪声（中等 bit-width），1-bit 处分离
- 见 `result.md` 和 `figures/bitwidth_recovery.png`

### Pilot 局限
- 是激活量化（直接量化中间表征），不是权重量化（量化模型参数）
- Pythia-160M 太小
- 需要更大模型 + 权重量化 + 有限尺寸标度分析

## 下一步计划：服务器实验

### Phase 1: Gemma-2 2B + Gemma Scope SAE（~100 A100-hours，~2 天）

**为什么选 Gemma-2 + Gemma Scope**：
- Gemma Scope 提供每一层的预训练 JumpReLU SAE（width 16K/65K），直接下载使用
- 不需要自己训练 SAE，避免 exp-001 遇到的 dead feature / 训练不足问题
- CC-BY-4.0 开源，SAELens 一行代码加载

| 参数 | 值 |
|------|-----|
| 模型 | Gemma-2 2B |
| 层 | {0, 6, 12, 19, 25}（5 层，覆盖早/中/晚） |
| SAE | Gemma Scope JumpReLU, width 16K |
| 量化方法 | HQQ（1-8 bit，不需要校准数据） |
| Bit-widths | FP16, INT8, INT6, INT4, INT3, INT2 |
| 对照 | 匹配 MSE 高斯噪声 |
| SAE 种子 | 3（用于误差估计） |
| 推理 tokens | 100M |

**度量**：
1. 共享特征比例（FP16-SAE 在量化激活上的特征 vs FP16 上的特征，Hungarian matching cos>0.7）
2. 特征激活 Pearson r（按 firing rate 五分位分层）
3. Loss recovered（SAE 重建插入量化前向传播后的 CE 恢复率）
4. Debiased CKA

### Phase 2: Pythia scaling ladder（~150 A100-hours，~1.5 天）

| 参数 | 值 |
|------|-----|
| 模型 | Pythia 410M, 1.4B, 2.8B, 6.9B |
| SAE | EleutherAI/sparsify TopK SAE (32× expansion) |
| 量化 | HQQ, 同上 |

**目的**：用 4 个不同宽度的模型做**有限尺寸标度分析**——
- 如果是真相变：不同宽度的 Φ(b) 曲线经 rescale 后 collapse 到一条 master curve
- 如果是 smooth crossover：无法 collapse

### Phase 3: 对照与交叉验证（~120 A100-hours，~1 天）

1. GPTQ 量化作为第二种方法——b* 在不同方法下是否一致
2. 随机 codebook 量化——隔离"校准信号 vs 纯精度"的贡献
3. 通道仿射校正——排除简单分布漂移
4. 剪枝 at matched compression——量化和剪枝对特征谱的影响是否对称

### 总计算量
约 370-500 A100-hours，4×A100 约 5 天

### 成功标准

**正面结果（支持假说）**：
- 频率分层效应在权重量化 + 大模型上复现（低频特征先崩溃）
- Hill 系数 n 的 95% CI 排除 n<3（sharp transition）
- 有限尺寸标度 collapse 成功（Bhattacharjee-Seno 残差 < 0.1）
- b* 在 HQQ 和 GPTQ 之间一致（±0.5 bit）

**负面结果（反驳假说）**：
- 所有 bit-width 下退化都是 smooth power-law
- 匹配 MSE 高斯噪声完全复现量化的效果（量化无特殊性）
- b* 随量化方法大幅漂移（>1 bit）

**独立于相变假说的有价值发现**：
- 频率分层效应本身（不管是否有相变）
- 量化 vs 剪枝对特征谱的对称/不对称性（Michaud vs Borobia 预测冲突）
