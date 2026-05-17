# 进展记录

## 2026-05-17

- **exp-005 Additivity Tax 全 5 phases** [本地]
  - Phase 1: 自然网络 IGA≈2.0（所有 α）→ Bilodeau 不可能性是 worst-case only
  - Phase 2: Adversarial 构造 IGA≈1.1 → worst-case 可触发
  - Phase 3: Smoothed analysis → 方向错误，扰动让 IGA 下降
  - Phase 4: Natural→adversarial 插值 → β=1.0 处 sharp transition
  - Phase 5: GPT-2 small 验证 → 不 conclusive
  - **结论**：方向 surprise 不够，搁置
- **impossibility theorem formalization** [本地] — v1→v3，四篇论文定理提取 + 统一公理 + mini proof
  - 审阅修正 5 个问题；v1/v2 归档至 archive/
- **exp-003 shape metric** [本地] — pilot 数据上加 Williams shape metric
  - 发现：shape metric 与 CKA/kNN 排序不一致（position 2/3 reversal）
- **DR-008 response** 已读 — 五大表征视角 + 六个部分统一框架 + 不可能性定理方向
- **DR-009/010 prompt** 已写 — Squeeze Conjecture 数学化 / 方法互补性理论
- **DR-011 prompt + response** — 四篇不可能性论文的历史定位分析
  - 关键洞察：Ackerman-Ben-David "productive rebuttal" 模式
- **DR-012/013 prompt** 已写 — 跨领域不可能性类比 / 论文地图
- **jiagpu 分支合并** — exp-004 server 结果 + exp-003 Phase 1 scripts 合入 main
- **exp-003 表征提取完成** [jiagpu5]：Pythia/Mamba/RWKV 在 410M/1.4B/2.8B 的 reps 已提取
- **exp-004 Stage 1 完成** [jiagpu5]：ResNet-18/CIFAR-10 weight decay × seed baseline
  - ID vs accuracy: r=0.365（不显著）；Stable rank vs accuracy: r=0.935

## 2026-05-16

- **exp-002 Phase 1 完成** [jiagpu4]：Gemma-2 2B + Gemma Scope SAE 权重量化
  - 关键发现：4-bit 几乎无损，3→2 bit 剧烈崩溃；中间层最脆弱
- 搭建 jiagpu4 Python 环境（venv on NVMe SSD）
- 创建 jiagpu 分支

## 2026-05-10

- 初始化项目结构
- 建立项目约定
