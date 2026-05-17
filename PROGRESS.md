# 进展记录

### 2026-05-17

- **exp-003 表征提取完成**：Pythia/Mamba/RWKV 在 410M、1.4B、2.8B 三个规模的 reps 已提取完成
  - 共 9 个 `.pt` artifact，保存在 `experiments/exp-003-cross-arch-platonic/phase1/data/`
  - 注意：reps 文件较大且已被 git ignore，不直接推到仓库
- **exp-004 Stage 1 完成**：ResNet-18/CIFAR-10 weight decay × seed baseline
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
