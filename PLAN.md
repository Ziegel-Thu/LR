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

### 任务 1: 填写模型资产表
- 检查 HF 缓存目录，列出所有已下载的模型
- 更新 CLAUDE.md 的模型资产表

### 任务 2: exp-006 表征 Scaling Law
- 详见 `experiments/exp-006-repr-scaling-law/plan.md`
- Phase 1: Pythia 70M~6.9B 全系列，提取每层表征，计算 CKA/shape/kNN/ID/stable rank
- Phase 2: 多架构（加 Mamba/RWKV 同规模），计算跨架构距离
- **注意**：exp-003 Phase 1+2 已有 kNN/CKA 数据，补 shape metric / ID / stable rank 即可

### 任务 3: exp-007 Encoding ≠ Use 量化
- 详见 `experiments/exp-007-encoding-use-gap/plan.md`
- 在 Pythia-1.4B 上跑：15 特征 × 32 层 probe + ablation
- 核心输出：probe accuracy vs Δloss 散点图

### 任务 4: exp-008 SSM 上的 SAE
- 详见 `experiments/exp-008-sae-on-ssm/plan.md`
- Phase 1: Mamba-130M 上训练 TopK SAE
- Phase 2: 与 Pythia-160M SAE 特征对比

### 任务 5: exp-009 LRH 系统性测试
- 详见 `experiments/exp-009-lrh-systematic/plan.md`
- Phase 1: 概念方向跨层一致性（Pythia-1.4B, 10 概念, 32 层）
- Phase 2: 概念-概念角度矩阵（全局几何结构）
- Phase 3: LRH 在 Mamba 上是否成立
- Phase 4: 概念方向随训练演化（Pythia checkpoints）

### 任务 6: exp-010 因果抽象方法论审计
- Off-distribution intervention 量化：patching 后算 Mahalanobis distance，画 distance vs IIA 可靠性
- Ablation baseline 对比：mean vs zero vs resample 三种 baseline 在 IOI 上的 circuit 差异
- Trivialization baseline：随机 GPT-2 small 在 IOI 上做 circuit discovery，作为 Wang 2022 的对照
- Circuit "暗物质"：把 MLP 纳入 circuit search，看 87% → ？如果仍有 gap → 量化 distributed computation 的比例

### 任务 7: exp-011 PID 预测因果效果（小规模）
- GPT-2 small 上选 5 个关键 neurons，PID 分解 unique/redundant/synergistic
- 逐个 ablation，比较 unique info rank vs ablation effect rank
- 正相关 → info 预测 causation；不相关 → PID 工具价值存疑

### 任务 8: exp-012 几何量对比 + 估计器对比
- ID vs stable rank vs effective rank vs participation ratio 在多架构多数据集上的泛化预测力对比
- 同一组表征上跑 TwoNN / MLE(K=10/50/100/500) / GeoMLE 估计器对比
- 训练过程中 hunchback 演化追踪（Pythia checkpoints）

### 任务 9: 不可能性定理统一 + 正面结果
- 将 Locatello 2019 纳入五公理框架（目前只有 Bilodeau/Han/Sutter/Kleinberg）
- 对每个不可能性，找最小公理放松下的最优正面结果
- 推广 exp-005 的 smoothed analysis 到 Locatello 和 Sutter（三个不可能性的统一 average-case 处理）

### 任务 10: Story A —— "理解的几何学"
- 核心问题：声称"理解"的模型（V-JEPA 2=物理、DreamerV3=世界、OthelloGPT=棋局、Coconut=推理、CLIP=跨模态），其表征几何有没有共同 signature？
- 工具：ID profile、stable rank、hunchback 形状、probe accuracy vs ablation effect
- 预期：如果有共同 signature → "理解"的几何定义；如果没有 → "理解"需要细分
- 待 DR-016 回收后细化实验设计

（当前无——实验走服务器）

## 下一批实验（3 个并行）

### exp-006: 表征质量 Scaling Law
- **问题**：表征对齐度是否随规模呈 power law？
- 无人做过，Pythia ladder + 多架构（Mamba/RWKV）
- **注意**：exp-003 Phase 1+2 已有 kNN/CKA 数据（160M~6.9B），exp-006 在此基础上加 shape metric/ID/stable rank 并拟合幂律

### exp-007: Encoding ≠ Use 量化
- **问题**：probe 能解码的信息有多大比例模型实际在用？
- Braun 2025 给了理论但没人在真实 LLM 上验证
- 15 特征 × 32 层的 probe accuracy vs ablation effect 散点图

### exp-008: SSM 上的 SAE
- **问题**：superposition 是 Transformer 特有的还是普遍的？
- Mamba 上训练 SAE——**零先例**
- Phase 1 pilot (Mamba-130M) → Phase 2 与 Pythia SAE 对比

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
