# 研究计划

## 研究方向

神经网络表征的数学结构——度量、几何、信息论、可解释性

## 当前阶段：寻找创新点

从方向探索转向锁定问题。之前试过多个方向（不可能性定理、additivity tax、ID→泛化等），获得了领域直觉，但还没找到足够有新意的研究问题。

**当前策略**：先系统搜索领域主体文献，建立全面 overview，再从中寻找：
- 未被解释好的实验现象
- 不同论文结论之间的矛盾
- 可能的统一 / 连接 / 发现 / 泛化
- 有待发现的新现象

## 📋 布置给 jiagpu

| 任务 | 节点 | 状态 | 备注 |
|------|------|------|------|
| exp-003 Phase 1 分析: CKA/kNN/shape metric on 410M/1.4B/2.8B | jiagpu5 | 待执行 | 表征已提取（9 个 .pt），分析脚本待写 |

## 💻 本地进行中

（当前无）

## 下一步

### 最高优先：文献地图
- **DR-013**（论文地图）已回收，10 个分支 overview 完成
- 7 篇优先论文已下载到 literature/
- 详细 open problems 整理见 `OPEN-PROBLEMS.md`

### 待回收 DR

| DR | 主题 | 状态 |
|----|------|------|
| DR-009 | Squeeze Conjecture 数学化 | ✅ 已回收 |
| DR-010 | 方法互补性 + worst-case gap 量化 | ✅ 已回收 |
| DR-012 | 跨领域不可能性类比 | prompt 已写，未发 |
| DR-013 | 论文地图（10 分支 overview） | ✅ 已回收 |

---

## 候选实验方向（17 个）

> 来源：DR-013 论文地图 + 已有实验线索，按 open problem 编号对应 `OPEN-PROBLEMS.md`

### Tier 1：最可能出重要发现

**① 表征质量 Scaling Law (9d)**
- 所有人做 loss scaling law，没人做过表征 scaling law
- Pythia 70M→6.9B 全系列，测 CKA/shape/ID/stable rank/kNN
- 问：度量是否呈 power law？有没有 phase transition（"表征涌现"）？

**② 为什么 LRH 成立 (4c)**
- 全场最深的未解释现象，零理论解释
- 控制变量实验：换 loss（CE vs MSE vs contrastive）、有/无 softmax、训练过程追踪
- 测试 Jiang 2024 的 log-linear 猜想

**③ 收敛的唯一性 (9a)**
- Platonic 说收敛，但收敛到唯一解还是等价族？
- 多架构同数据训练，shape metric 测两两距离随规模变化
- 距离→0 = 唯一；→c>0 = 族；部分收敛 = 边界条件

**⑯ 收敛的选择压力分解 (9b)**
- 四因素实验：数据 × loss × 优化器 × 架构
- 哪个因素对收敛贡献最大？
- 数据驱动 → Platonic "reality"；架构驱动 → 需重新理解

### Tier 2：新颖方法论

**④ ID 因果干预 (7b)**
- 训练时加 ID 正则化器，显式压低 ID，看泛化是否跟着变
- 因果的 → ID regularization 有实用价值；不是 → ID 只是 proxy

**⑤ 多维特征的系统性发现 (4a + 2b)**
- Engels 发现了圆环，但没系统搜索过还有多少概念是多维的
- PCA/ICA 系统扫描 LLM，刻画线性 vs 多维的边界

**⑬ PID 预测因果效果 (6c)**
- PID 分解表征信息，再用因果干预验证
- 如果 unique information 预测干预效果 → 建立 info→causation 桥

**⑮ 预训练表征的因果结构 (8c)**
- LLM 编码的是 correlation 还是 causation？
- 找"相关但非因果"案例，看表征空间能否区分

### Tier 3：实用 Contribution

**⑦ SAE 特征稳定性 (2a)**
- 同模型不同 SAE 配置，哪些特征稳定出现？
- 稳定 → 存在"真"特征；不稳定 → SAE 方法论问题

**⑧ Faithfulness-Completeness Pareto 前沿 (3c)**
- Wu 2025 证了 trade-off，但没画 Pareto 曲线
- 多种方法在同一任务上量化 F-C 平面位置

**⑨ 概念方向组合性 (4d)**
- Steering vector 能否线性组合？"快乐"+"正式"=？
- 组合 work → LRH 强版本；崩溃 → superposition 限制

**⑩ 正则内积寻找 (4b)**
- 因果内积 vs 余弦 vs whitened vs CKA-induced
- 哪个下概念最"线性"？

**⑪ 编码 ≠ 使用的量化 (5a)**
- Probe 高准确率但 ablation 无影响的比例有多大？
- 高 → probing 方法论需根本修正

**⑫ 跨模型 Lens (5b)**
- Pythia-1.4B 训练 lens → Pythia-6.9B / Mamba / RWKV 测试
- 迁移成功 → Platonic + 实用价值

**⑭ ID 估计器系统对比 (7a)**
- TwoNN/MLE/GeoMLE/PHD 在同一组表征上对比
- 找一致/分歧的条件，实用 methods paper

### Tier 4：理论型（field-defining 如果做出来）

**⑥ 度量一致性 Phase Transition (1a)**
- 连续变参数，追踪 CKA/shape/kNN 排序一致性的 transition point
- 存在 sharp transition → 度量捕获的是不同 regime

**⑰ 表征比较的 Kleinberg 式公理化 (10a)**
- 提出 3-5 个合理公理，证明不可能 / 放松后的唯一最优
- 纯理论但 field-defining

## 已有探索线索

### 信号最强：跨架构柏拉图收敛 (exp-003)
- Pilot: z>350，U 形深度曲线，shape metric 与 CKA 排序不一致
- 需要更多数据和分析来判断是否值得深挖

### 有结论但方向有限：不可能性定理
- 四篇论文统一公理化（formalization.md），核心：additivity 是不可能性根源
- exp-005: Bilodeau 不可能性是 worst-case only → 方向 surprise 不够
- DR-009/010 回复可能带来新思路

### 其他实验结论
- exp-002: 4-bit 量化无损，3→2 bit 崩溃（可能有趣但需更多 scale）
- exp-004: stable rank 比 ID 更强的泛化预测器（r=0.935 vs r=0.365）
- exp-001: SAE 可识别性，信号弱，搁置
