# 进展记录

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
