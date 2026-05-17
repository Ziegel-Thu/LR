# 进展记录

## 🔄 正在进行

| 任务 | 节点 | tmux session | 启动时间 | 预计耗时 | 备注 |
|------|------|-------------|---------|---------|------|
| exp-003 Phase 1 分析（CKA/kNN） | jiagpu | — | 待启动 | ~1h | 代码已 push |

## ✅ 已完成

### 2026-05-17

- **exp-005 Additivity Tax** [本地] — IG vs SHAP identifiability tradeoff
  - 意外发现：SHAP 在自然训练网络上 IGA≈2.0（所有 α），Bilodeau 不可能性是 worst-case only
- **impossibility theorem formalization** [本地] — v1→v3，四篇论文定理提取 + 统一公理系统 + mini proof
  - formalization_v3.md 已审阅并修正 5 个问题
- **DR-008 response** 已读 — 五大表征视角 + 六个部分统一框架 + 不可能性定理方向
- **DR-009/010 prompt** 已写 — Squeeze Conjecture 数学化 / 方法互补性理论
- **exp-003 表征提取完成** [jiagpu5]：Pythia/Mamba/RWKV 在 410M、1.4B、2.8B 三个规模的 reps 已提取完成
  - 共 9 个 `.pt` artifact，保存在 `experiments/exp-003-cross-arch-platonic/phase1/data/`
  - 注意：reps 文件较大且已被 git ignore，不直接推到仓库
- **exp-004 Stage 1 完成** [jiagpu5]：ResNet-18/CIFAR-10 weight decay × seed baseline
  - 21 runs 分 4 shards 在 jiagpu5 并行完成
  - TwoNN ID vs test accuracy: r=0.365, p=0.104；不支持“ID 越低泛化越好”的简单负相关
  - Stable rank vs test accuracy: r=0.935, p=5.67e-10；后续干预需要加入 rank/norm 对照

### 2026-05-16

- **exp-002 Phase 1 完成**：Gemma-2 2B + Gemma Scope SAE 权重量化实验
  - HQQ 量化（8/6/4/3/2 bit），5 层（L0, L5, L12, L17, L23）
  - 关键发现：4-bit 几乎无损，3→2 bit 剧烈崩溃；中间层最脆弱；稀有特征先崩溃
  - 硬件：jiagpu4, 8× A40 并行（3 个 bit-width 同时跑）
- 搭建 jiagpu4 Python 环境（venv on NVMe SSD）
- 创建 `jiagpu` 分支
- 更新 CLAUDE.md 服务器信息

### 2026-05-10

- 初始化项目结构
- 建立项目约定
