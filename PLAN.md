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

## 💻 本地进行中

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

## 候选实验方向（17 个）

> 详见 OPEN-PROBLEMS.md（10 个分支 × 全角度分析）

### Tier 1：最可能出重要发现
- ① 表征质量 Scaling Law (9d) → **exp-006**
- ② 为什么 LRH 成立 (4c)
- ③ 收敛的唯一性 (9a)
- ⑯ 收敛的选择压力分解 (9b)

### Tier 2：新颖方法论
- ④ ID 因果干预 (7b)
- ⑤ 多维特征系统发现 (4a+2b)
- ⑬ PID 预测因果效果 (6c)
- ⑮ 预训练表征因果结构 (8c)

### Tier 3：实用 Contribution
- ⑦ SAE 特征稳定性 (2a)
- ⑧ Faithfulness-Completeness Pareto (3c)
- ⑨ 概念组合性 (4d)
- ⑩ 正则内积寻找 (4b)
- ⑪ 编码≠使用量化 (5a) → **exp-007**
- ⑫ 跨模型 Lens (5b)
- ⑭ ID 估计器系统对比 (7a)

### Tier 4：理论型
- ⑥ 度量一致性 Phase Transition (1a)
- ⑰ 表征比较 Kleinberg 式公理化 (10a)

### 新增：低垂果实
- SSM 上的 SAE → **exp-008**
