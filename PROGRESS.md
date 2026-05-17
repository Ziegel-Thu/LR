# 进展记录

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
