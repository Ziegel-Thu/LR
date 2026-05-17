# 进展记录

## 2026-05-17 (晚)

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
