# 进展记录

## 🔄 正在进行

无当前活跃任务。

## ✅ 已完成

### 2026-05-17

- **exp-003 Phase 1+2 跨架构 scaling 分析完成** [jiagpu4]
  - 4 个 scale（410M / 1.4B / 2.8B / 6.9B）× 3 架构（Pythia / Mamba / RWKV）
  - 6.9B scale 补充提取了 Pythia-6.9B 和 RWKV-7B（Mamba-1 无 7B 版本，只有 Pythia↔RWKV 1 个 pair）
  - 共 11 个 reps 文件，保存在 jiagpu4 SSD `/nvmessd/lifanhong/LR-env/exp003_reps/`
  - **核心发现：kNN z-score 随 scale 持续增强（pilot 379 → 2.8B 2642），强力支持 Platonic Representation Hypothesis**
  - kNN raw overlap 在 2.8B 达 0.65-0.75，6.9B 维持 0.73
  - Mamba↔RWKV 在中等 scale 最相似（共享 SSM/线性注意力偏置）
  - CKA 受 d_model 维度影响，趋势不如 kNN 清晰
  - 详细结果见 `experiments/exp-003-cross-arch-platonic/phase1/result.md`
- **exp-005 Additivity Tax** [本地] — IG vs SHAP identifiability tradeoff
  - 意外发现：SHAP 在自然训练网络上 IGA≈2.0（所有 α），Bilodeau 不可能性是 worst-case only
- **impossibility theorem formalization** [本地] — v1→v3，四篇论文定理提取 + 统一公理系统 + mini proof
  - formalization_v3.md 已审阅并修正 5 个问题
- **DR-008 response** 已读 — 五大表征视角 + 六个部分统一框架 + 不可能性定理方向
- **DR-009/010 prompt** 已写 — Squeeze Conjecture 数学化 / 方法互补性理论
- **exp-004 Stage 1 完成** [jiagpu5]：ResNet-18/CIFAR-10 weight decay × seed baseline
  - 21 runs 分 4 shards 在 jiagpu5 并行完成
  - TwoNN ID vs test accuracy: r=0.365, p=0.104；不支持"ID 越低泛化越好"的简单负相关
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
