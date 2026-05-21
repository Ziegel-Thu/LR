# 进展记录

## 2026-05-22

- **exp-021 SAE as empirical ICA 完成** [jiagpu4 GPU 1]
  - Pythia-160M layer 6, TopK SAE (d_sae=4096, K=32), 5 个 L1 系数
  - 基线: PCA (128), FastICA (128), Random (128)
  - **核心发现: SAE 特征 MI ≈ 0.0002, 比 ICA (0.019) 低 ~80 倍**
  - 独立性来自 TopK 稀疏约束（99.2% 零值），L1 系数几乎无影响
  - SAE variance explained 83.6%, 0 dead features
  - 结论: SAE 是一种比 ICA 更强的独立分解（MI 意义下），但主要因稀疏性而非直接优化独立性

## 2026-05-21

- **exp-018 SSM LRH 完成** [jiagpu4 + jiagpu5]
  - 首次在非 Transformer 上测试线性表征假设
  - 跨层一致性: Pythia 0.77, Mamba 0.79, RWKV 0.78（一致）
  - 概念正交性: |cos|≈0.17（三架构相同，近正交）
  - 方向稳定性: Mamba 0.82 > Pythia 0.76 > RWKV 0.75
  - **核心发现: same geometry, different coordinate system — LRH 是架构无关的**

- **exp-017 跨模型 lens 迁移完成** [jiagpu4]
  - Pythia-2.8B ↔ RWKV-3B probe 迁移率 = 55%
  - 表面特征高迁移 (空格 79%, 标点 77%)，语义特征低迁移 (稀有词 15%)
  - 结论: 部分 Platonic 收敛，但不完全

- **exp-015 OthelloGPT 策略 probing Phase 1 完成** [jiagpu4]
  - Frontier discs 96.7% @ L4, Mobility R²=0.73 @ L6
  - Board state 线性 probe 72%（MLP 可达 88%+, Phase 2 进行中）
  - 战略概念可线性解码

- **exp-015 Phase 2 运行中** [jiagpu4 GPU 0]
  - MLP probe: Board state 87.7% (vs linear 72%) — 显著提升
  - 新增: Corner occupancy 97.1%, Edge control R²=0.80

- **exp-007 五模型对比完成** [jiagpu4 + jiagpu5]
  - Mamba-2.8B ghost ratio = 89.8%（所有模型中最高）
  - RWKV-3B ghost ratio = 28.4%（与 Gemma 接近）
  - 三个 ghost ratio 群体: ~90% (Mamba), ~70% (Pythia), ~30% (Gemma/RWKV)
  - 修复 dtype mismatch bug（Mamba 输出 fp32 vs probe fp16）

## 2026-05-20

- **exp-007 Pythia-2.8B probing 完成** [jiagpu4]
  - Ghost ratio = 70.0%（vs Pythia-1.4B 70.8% — scale-invariant）
  - 9/11 features 因果效应集中在最后一层 L31
  - is_high_freq 100% ghost（完美编码但完全不用）

- **exp-007 Gemma-2-2B probing 完成** [jiagpu4]
  - Ghost ratio ≈ 30%（vs Pythia 70% — 架构差异 >> 规模差异）
  - 修复架构兼容 bug（`_get_model_layers()` 支持 Pythia/GPT-2/Gemma）
  - Gemma 特有：大量负 Δloss（删除信息反而帮助预测）
  - 因果效应分散在多层（vs Pythia 集中在最后层）

- **三模型对比分析** [jiagpu4]
  - 核心发现：ghost info 是结构性现象，但强度受架构调控
  - 架构差异（Pythia vs Gemma）对 ghost ratio 的影响 > 规模差异（1.4B vs 2.8B）

## 2026-05-19

- **exp-013 hunchback profiling 完成** [jiagpu4]
  - 22+ 模型测试，6/6 understanding 模型有 hunchback，0/17 LLM 有
  - 随机权重对照：无 hunchback（排除架构 bias）
  - V-JEPA 2 是最新 understanding 模型，确认有 hunchback

## 2026-05-17 (晚)

- **exp-009 Phase 3 完成** [jiagpu4] — Mamba LRH
  - Mamba↔Pythia 概念方向 mean cos ≈ 0（完全不对齐）
  - 同信息，不同方向 — 与 exp-008 MMCS=0.13 一致

- **exp-010 完成** [jiagpu4] — 因果抽象方法论审计
  - Zero vs mean ablation ranking ρ=0.27（几乎无关！）
  - Trained vs random ρ=-0.17（circuit 不是 trivial）

- **exp-011 完成** [jiagpu4] — MI vs 因果效应
  - Spearman ρ=0.36 (p=0.01)：信息弱预测因果性

- **exp-012 完成** [jiagpu4] — 几何量 benchmark
  - ID 估计器严重不一致（TwoNN vs MLE-500 ρ=0.11）
  - 无几何量与 loss 显著相关

- **exp-013 ID Atlas 完成** [jiagpu4]
  - 10 模型均无 hunchback（ID 单调下降）

- **exp-014 Phase 1+2+4 完成** [jiagpu4]
  - Phase 1: cos(probe, gradient) ≈ 0.05（梯度不是因果 proxy）
  - Phase 2: IOI lifecycle — probe@L7, ablation@L9（encoding leads use）
  - Phase 4: 88-97% probe 信号来自 input bleed-through

- **exp-008 Phase 3 完成** [jiagpu4] — Mamba-370M SAE
  - 80.5% var, 0/8192 dead — scales smoothly

- **exp-006 Phase 2 完成** [jiagpu4] — 跨架构 Scaling
  - 4 规模档 × 3 架构 = 12 pairs
  - Inter-family β=0.056 vs intra-family β=0.025（2.2x faster）
  - Transformer↔SSM 对齐比 SSM↔SSM 收敛更快，支持 PRH

- **exp-007 Full Run 完成** [jiagpu4] — Encoding ≠ Use
  - 11 features × 24 layers, Pythia-1.4B, GPU parallel probe
  - **Ghost ratio = 70.8%**: 71% 的高准确率 probe 无因果效应
  - 所有 feature probe acc > 0.95，但 Δloss 大多 < 0.01
  - 并行改写：20min（旧 sklearn 版本需 3h+）
  - 首次在真实 LLM 上系统量化 encoding-use gap

- **exp-008 Phase 2 完成** [jiagpu4] — Mamba vs Pythia SAE
  - MMCS = 0.13（接近随机），0 个特征 overlap > 0.9
  - 结论：两架构都有 superposition 但特征方向完全不同

- **exp-006 Phase 1 完成** [jiagpu4] — 表征 Scaling Law
  - Pythia 7-scale ladder (70M→6.9B), 7 metrics per model per layer
  - **kNN overlap 是唯一呈 power law 的度量**: β=0.03, R²=0.90
  - CKA 饱和（>0.99），shape distance 非单调，ID/rank 无 scaling
  - 首次给出表征对齐度的**定量幂律**
  - Phase 2（多架构 scaling）运行中

- **exp-008 Phase 1 完成** [jiagpu4] — SAE on Mamba
  - Mamba-130M, layer 12, TopK SAE (d_sae=4096, K=32), 30K steps
  - **80.3% variance explained, 0% dead features**
  - 首次在 SSM 上训练 SAE：superposition 不是 Transformer 特有的
  - Phase 2（Pythia SAE + MMCS 对比）运行中

- **exp-007 Pilot 进行中** [jiagpu4]
  - 4 features × 24 layers probing on Pythia-1.4B
  - sklearn CPU probe 很慢，下次改用 GPU + 并行

- **模型资产盘点 + 下载** [jiagpu4]
  - 扫描 jiagpu4-8，登记 18 个模型到 CLAUDE.md
  - 下载 5 个缺失模型: Pythia-70M/160M/1B, Mamba-130M, RWKV-169M
  - 安装 netrep + scikit-learn
  - 提取 Pythia-70M/160M/1B + Mamba-130M + RWKV-169M reps

## 2026-05-17

- **exp-003 Phase 1+2 scaling 分析完成** [jiagpu4]
  - 4 个 scale (410M/1.4B/2.8B/6.9B) + pilot (160M)
  - kNN z-score: 160M~350 → 410M~1300 → 2.8B~2642，支持 PRH
  - 6.9B Pythia↔RWKV: z=1562, raw=0.728（Mamba 无 7B 版本）
  - CKA 在 6.9B 下降（维度敏感），kNN 更稳定
  - Mamba↔RWKV 在中等 scale 最相似（1.4B z=1696）
- **exp-005 Additivity Tax 全 5 phases** [本地]
  - Phase 1-5 完成，结论：worst-case only，搁置
- **impossibility theorem formalization** [本地] — v1→v3 + 审阅修正
- **exp-003 shape metric** [本地] — CKA/kNN/shape metric 排序不一致
- **DR-008/009/010/011/013** — 回收并分析
- **OPEN-PROBLEMS.md** — 10 个分支全角度深入分析（376 行）
- **exp-006/007/008 plans** — 三个新实验计划写好
- **7 篇优先论文下载** — Huh/Ziyin/Cloos/Park/Yu/Harvey/Wu
- **jiagpu 分支合并** — exp-003 scaling 结果合入 main

## 2026-05-16

- **exp-002 Phase 1 完成** [jiagpu4]：Gemma-2 2B 量化，4-bit 无损，3→2 bit 崩溃
- 搭建 jiagpu4 Python 环境
- 创建 jiagpu 分支

## 2026-05-10

- 初始化项目结构
- 建立项目约定
