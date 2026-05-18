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

### 任务 1: 填写模型资产表 ✅
- 18 个模型已登记到 CLAUDE.md
- 5 个缺失模型已下载（Pythia-70M/160M/1B, Mamba-130M, RWKV-169M）

### 任务 2: exp-006 表征 Scaling Law ✅
- Phase 1 ✅: kNN power law R²=0.90, β=0.03
- Phase 2 ✅: inter-family β=0.056 > intra β=0.025 (2.2x)
- Phase 3 ✅: kNN vs loss r=-0.952

### 任务 3: exp-007 Encoding ≠ Use 量化
- Full Run ✅: Ghost ratio = 70.8%（11 features × 24 layers, Pythia-1.4B）
- 下一步：加语义特征（需 NLP 标注）、DAS 替代单方向 ablation

### 任务 4: exp-008 SSM 上的 SAE ✅
- Phase 1 ✅: Mamba-130M SAE, 80.3% var, 0% dead
- Phase 2 ✅: MMCS=0.13, 特征完全不同
- Phase 3 ✅: Mamba-370M SAE, 80.5% var, 0/8192 dead

### 任务 5: exp-009 LRH 系统性测试 ✅
- Phase 1-2 ✅: 跨层一致(cos=0.93), 概念不正交(cos=0.83)
- Phase 3 ✅: Mamba↔Pythia 概念方向 cos≈0
- Phase 4 ✅: 方向渐进涌现, step1K cos=0.37-0.63

### 任务 6: exp-010 因果抽象方法论审计 ✅
- Ablation baseline ρ=0.27（zero vs mean 几乎无关）
- Trained vs random ρ=-0.17（非 trivial）
- Phase 4 暗物质探索未做

### 任务 7: exp-011 PID 预测因果效果 ✅
- MI↔causation Spearman ρ=0.36 (p=0.01)
- 信息弱预测因果性

### 任务 8: exp-012 几何量对比 + 估计器对比 ✅
- ID estimators 严重不一致（TwoNN vs MLE-500 ρ=0.11）
- 无几何量与 loss 显著相关
- Phase 3 训练动态 ✅: 平滑演化, 无 phase transition

### 任务 9: 不可能性定理统一 + 正面结果
- 理论工作，未启动

### 任务 10: Story A —— "理解的几何学"
- exp-013 ID Atlas ✅: 10 模型均无 hunchback
- exp-014 Phase 1+2+4 ✅: gradient≠probe, IOI lifecycle, 88-97% bleed-through
- DR-018 新发现：DreamerV3 从未被 probe（最大低垂果实）
- 下一步：OthelloGPT 策略 probing, DreamerV3 RSSM probing

### Probing 深层问题（待单独 brainstorm）

**Meta Q1: 什么条件下 encoding = use？**
- Braun 2025 证明了 encoding ≠ use，Wu 2025 实证了 probe≠steerer
- 核心问题：找到条件 C 使得"满足 C 时 probe accuracy 可靠推出因果使用"
- 候选条件：参数噪声鲁棒性、probe 方向和 causal 方向夹角、层深度、概念类型、架构类型、过参数化程度
- 待 brainstorm agent 回收后整理

**Meta Q2: 信息在神经网络中的表示形式如何随计算深度变换？**
- 不只是"有没有"，而是信息从哪来、往哪去、怎么变
- 信息生命周期（逐层 probe+ablation 双曲线）是一个具体实例
- 信息形式变换（同一信息在不同层可能需要不同 probe 才能检测）
- 信息的创造（哪层产生了输入中不存在的新信息）和销毁
- 连接 probing 和 circuit analysis 的桥

**更多 Meta Questions（brainstorm 结果）：**

| MQ | 问题 | 一句话 |
|----|------|--------|
| Q3 | 什么时候两个表征"相同"？ | 表征空间上的等价关系应该是什么？能否只有一个？ |
| Q4 | 表征的自然单元是什么？ | 有没有特权的分解（神经元/方向/字典原子/流形/电路）？ |
| Q5 | 为什么线性无处不在？哪里打破？ | LRH 的理论解释 + 边界条件 |
| Q6 | 表征分析什么条件下能恢复 ground truth？ | 外部分析方法的根本极限 |
| Q7 | 表征空间有没有 universal attractor？ | Platonic hypothesis 的形式化 |
| Q8 | 哪些结构性质因果决定计算行为？ | ID/stable rank/curvature 中谁是因、谁是果？ |
| Q9 | 多少计算是可定位的、多少本质 distributed？ | 稀疏 circuit 假设的适用范围（暗物质问题的推广）|
| Q10 | 表征结构由什么决定？ | 架构 / 数据 / 目标 / 优化器的因果贡献 |
| Q11 | 怎么形式化"表征知道多少"？ | probe accuracy / MI / MDL / PID 哪个是对的度量？ |
| Q12 | 部分的表征怎么组合成整体的？ | 概念/token/特征如何 compose 成结构化语义？ |

**方向 P1: 三角验证（probe / causal / DiffMeans 方向对比）**
- 待单独细化

**方向 P2: "谁的信息"——三来源分解（需深入调研）**
- 待 DR 调研

**方向 P4: 理解模型上的 probing（需深入调研）**
- 待 DR 调研

（当前无——实验走服务器）

## 下一批实验（3 个并行）

### exp-006: 表征质量 Scaling Law
- **问题**：表征对齐度是否随规模呈 power law？
- Phase 1 结果：**kNN 是唯一 power law 度量 (R²=0.90, β=0.03)**
- Phase 2 运行中：多架构 intra vs inter family β 对比
- Phase 3 待做：repr alignment vs validation loss

### exp-007: Encoding ≠ Use 量化
- **问题**：probe 能解码的信息有多大比例模型实际在用？
- Pilot 运行中（4 features, Pythia-1.4B）
- Full run 待做：15 features × 24 layers，改用 GPU probe + 并行

### exp-008: SSM 上的 SAE
- **问题**：superposition 是 Transformer 特有的还是普遍的？
- Phase 1 结果：**SAE works on Mamba (80.3% var, 0% dead)**
- Phase 2 运行中：Pythia-160M SAE + MMCS 特征对比
- Phase 3 待做：规模扩展（Mamba-370M, 1.4B）

## 下一步

### 待回收 DR

| DR | 主题 | 状态 |
|----|------|------|
| DR-009 | Squeeze Conjecture 数学化 | ✅ 已回收 |
| DR-010 | 方法互补性 + worst-case gap 量化 | ✅ 已回收 |
| DR-012 | 跨领域不可能性类比 | prompt 已写，未发 |
| DR-013 | 论文地图（10 分支 overview） | ✅ 已回收 |

## 已有探索线索

### 信号最强：跨架构柏拉图收敛 (exp-003)
- Pilot z>350 → Phase 1+2: z>1000~2642，4 个 scale 支持 PRH
- Mamba↔RWKV（SSM 族内）始终最相似
- CKA 在 6.9B 下降（维度敏感），kNN 更稳定
- U 形深度曲线在所有 scale 持续

### 有结论但方向有限：不可能性定理
- 四篇论文统一公理化（formalization.md），核心：additivity 是不可能性根源
- exp-005: Bilodeau 不可能性是 worst-case only → 方向 surprise 不够
- Smoothed impossibility 是潜在的新方向

### 其他实验结论
- exp-002: 4-bit 量化无损，3→2 bit 崩溃
- exp-004: stable rank 比 ID 更强的泛化预测器（r=0.935 vs r=0.365）
- exp-001: SAE 可识别性，信号弱，搁置

## 候选研究方向（17 个，深入分析后）

> 详见 OPEN-PROBLEMS.md（10 个分支 × 全角度分析，376 行）

### 🚀 已在服务器跑

| # | 方向 | 分支 | 实验 |
|---|------|------|------|
| 1 | 表征 scaling law | 9 | exp-006 |
| 2 | Encoding ≠ use 量化 | 5 | exp-007 |
| 3 | SSM 上的 SAE | 2 | exp-008 |

### 📝 值得做但尚未启动

| # | 方向 | 分支 | 核心卖点 |
|---|------|------|---------|
| 4 | Smoothed impossibility | 10 | 已有 exp-005 数据 + 公理化框架，全新理论方向 |
| 5 | SSM 上的 LRH | 4 | 零论文，和 exp-008 共享基础设施 |
| 6 | Steering OOD 边界 | 4 | Tan 2024 发现了但没系统研究 |
| 7 | Off-distribution intervention 量化 | 3 | 定义 intervention validity score |
| 8 | 跨模型 lens 迁移 | 5 | 快速实验，如果 work 很亮眼 |
| 9 | ID 因果干预 | 7 | exp-004 暗示 ID 不是因，可做 debunking |
| 10 | SAE 作为经验 ICA | 8 | 桥接 SAE 实践和 identifiability 理论 |
| 11 | PID 预测因果效果 | 6 | 桥接信息论和因果抽象 |
| 12 | U 形深度曲线机制 | 1 | 所有人观察到但无解释 |
| 13 | CKA calibration 理论 | 1 | 中间值无可解释含义 |
| 14 | Ziyin break conditions 验证 | 9 | 论文自己说未验证 |
| 15 | 多维特征系统扫描 | 4 | Engels 开了头但没系统化 |
| 16 | Circuit "暗物质" | 3 | IOI 只解释 70%，30% 去哪了 |
| 17 | Kleinberg 式公理化 | 10 | field-defining 如果做出来 |
