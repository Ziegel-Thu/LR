# Exp-009: LRH 系统性测试

## 研究问题

线性表征假说（LRH）在多大范围内成立？概念方向在不同层、不同架构、不同训练阶段如何变化？概念之间的全局几何结构是什么？

## 动机

LRH 是表征分析领域最广泛的隐含假设（steering、probing、SAE、concept erasure 都依赖它），但系统性验证严重不足：
- Park 2024 只测了 unembedding 层、2 个模型、27 个概念
- 没人测过中间层的概念方向跨层一致性
- 没人在 SSM 上验证
- 没人追踪训练过程中方向的演化
- 概念-概念角度矩阵的全局结构未知

本实验用统一框架填充这些空白。

## Phase 1: 跨层一致性

**问题**：同一概念在不同层的方向是否一致？

### 配置
| 参数 | 值 |
|------|-----|
| 模型 | Pythia-1.4B |
| 概念 | 10 个二值概念（见下） |
| 方法 | 每层用 difference-in-means 提取方向 |
| 度量 | 层间余弦相似度矩阵（10 concepts × 32 layers × 32 layers） |

### 概念列表
1. 情感 (positive/negative)
2. 时态 (past/present)
3. 数 (singular/plural)
4. 大小写 (capitalized/not)
5. 形式程度 (formal/informal)
6. 真实性 (factual/fictional)
7. 语言 (English/non-English)
8. 否定 (negated/affirmed)
9. 具体性 (concrete/abstract)
10. 主动/被动 (active/passive voice)

每个概念需要 ~200 对对比样本。

### 输出
- 10 个 32×32 余弦相似度热力图
- 每个概念的"跨层一致性分数"（相邻层余弦相似度的平均）
- 关键问题：是否存在"方向突变层"——某一层概念方向突然改变？

## Phase 2: 概念-概念角度矩阵

**问题**：所有概念方向之间的几何关系是什么？

### 方法
- 选定一层（如 layer 16），提取 10 个概念的方向
- 计算 10×10 余弦相似度矩阵
- 分析：近似正交？有语义簇？和 superposition 理论的 packing prediction 一致吗？

### 扩展
- 在所有 32 层上重复 → 概念间角度如何随深度变化？
- 加入更多概念（扩展到 30+）→ 角度矩阵是否呈现有意义的结构？

## Phase 3: SSM 上的 LRH

**问题**：LRH 在 Mamba 上是否成立？

### 配置
| 参数 | 值 |
|------|-----|
| 模型 | Mamba-130M, Pythia-160M（对照） |
| 概念 | Phase 1 的 10 个（相同对比样本） |
| 方法 | difference-in-means 在每层提取方向 |

### 分析
- Mamba 概念方向的余弦相似度跨层曲线 vs Pythia
- 同一概念在 Mamba vs Pythia 中的方向是否 align？（跨架构因果内积？）
- Steering 效果对比（如果方向存在）

## Phase 4: 训练过程演化

**问题**：概念方向什么时候涌现？

### 配置
| 参数 | 值 |
|------|-----|
| 模型 | Pythia-1.4B 的 ~15 个 checkpoint（log-spaced: 1K, 2K, 5K, 10K, 20K, 50K, 100K, 143K steps）|
| 概念 | 选 5 个（Phase 1 中跨层最一致的 + 最不一致的）|
| 方法 | 每个 checkpoint 每层提取方向 |

### 输出
- 方向-方向余弦相似度 vs training step 曲线（和最终方向的相似度）
- 有没有 phase transition？（突然从随机变成有方向）
- 与 loss 曲线叠加：方向涌现先于/同步于/落后于 loss 下降？

## 成功标准

- Phase 1：发现跨层一致性的 pattern（一致/不一致/分层）
- Phase 2：角度矩阵呈现有意义的结构（不是纯随机）
- Phase 3：SSM 上的 LRH 有定性结论（成立/不成立/部分成立）
- Phase 4：至少一个概念的方向涌现有 clean phase transition

## 算力估计

- Phase 1: inference × 32 layers × 10 concepts → ~4 GPU-hours
- Phase 2: 同 Phase 1 的数据，分析 ~1h
- Phase 3: 2 个小模型 inference → ~2 GPU-hours
- Phase 4: 15 checkpoints × inference → ~10 GPU-hours
