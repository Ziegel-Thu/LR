# 研究计划

## 研究方向

神经网络表征的数学结构——度量、几何、信息论、可解释性

## 当前阶段

从方向探索转入两条主线：
1. **实证线**：跨架构收敛 (exp-003) 和量化相变 (exp-002) 的 scaling
2. **理论线**：表征分析不可能性定理——worst-case vs average-case gap

## 已完成

### 文献与调研
- DR-001~008: 全景地图、开放问题、实验设计、统一框架
- 精读: Klabunde (similarity survey), Ehrlich (PID), Braun (function-representation dissociation)
- DR-008: 五大表征视角 + 六个部分统一框架 + 不可能性定理方向

### 实验
- **exp-001 SAE 可识别性** [本地]: 信号弱，搁置
- **exp-002 量化 × SAE 特征** [本地+jiagpu4]: 4-bit 无损，3→2 bit 剧烈崩溃
- **exp-003 跨架构柏拉图收敛** [本地 pilot + jiagpu4/5 Phase 1+2]: z>350→2642，4 个 scale 支持 PRH
- **exp-004 ID→泛化** [本地+jiagpu5]: ID 和 acc 非简单负相关（r=0.365），stable rank 更强（r=0.935）
- **exp-005 Additivity Tax** [本地]: **核心发现——Bilodeau 不可能性是 worst-case only**
  - Phase 1: 自然训练网络 IGA≈2.0（SHAP works）
  - Phase 2: Adversarial 构造 IGA≈1.1（Bilodeau 成立）

### 理论
- **impossibility formalization** v1→v3: 四篇论文定理提取 + 统一公理 + mini proof + 审阅修正
- 核心诊断：additivity（而非 completeness）是不可能性的根源

### 待回收 DR
- DR-009: Squeeze Conjecture 数学化（已发）
- DR-010: 方法互补性 + worst-case gap 量化（已发）

---

## 下一步计划

### 理论线：worst-case vs average-case gap（最有论文价值）

| 编号 | 任务 | 依赖 | 节点 | 状态 |
|------|------|------|------|------|
| T1 | exp-005 Phase 3: Smoothed analysis — adversarial 模型加扰动，找 IGA 的 phase transition | exp-005 P2 ✅ | 本地 | **待做** |
| T2 | exp-005 Phase 4: 实际 failure 场景 — 对抗训练 / model editing 后 SHAP 是否失败 | T1 | 服务器 | 待做 |
| T3 | 定义 "attribution condition number"，量化 avg→worst 距离 | DR-010 | 理论 | 待 DR 回复 |
| T4 | Squeeze Conjecture: 统一 Bilodeau/Sutter 的复杂度参数 | DR-009 | 理论 | 待 DR 回复 |

### 实证线：跨架构收敛（信号最强）

| 编号 | 任务 | 依赖 | 节点 | 状态 |
|------|------|------|------|------|
| E1 | exp-003 Phase 1 分析: CKA/kNN on 410M/1.4B/2.8B reps | 表征已提取 ✅ | jiagpu | ✅ 完成 |
| E2 | exp-003 加 Williams shape metric + PID | E1 ✅ | jiagpu | 待做 |
| E3 | exp-003 Phase 2: 6.9B/7B scale + scaling law 拟合 | E1 ✅ | jiagpu | ✅ 完成（Pythia-6.9B + RWKV-7B，Mamba 无 7B） |

### 实证线：量化相变

| 编号 | 任务 | 依赖 | 节点 | 状态 |
|------|------|------|------|------|
| Q1 | exp-002 Phase 2: Pythia scaling ladder (410M→6.9B) | — | jiagpu | 待做 |
| Q2 | exp-002 Phase 3: GPTQ 对照 + 剪枝对比 | Q1 | jiagpu | 待做 |

### 低优先级

| 编号 | 任务 | 备注 |
|------|------|------|
| L1 | RLHF 表征漂移 (DR-007) | 待 exp-003/005 收敛后启动 |
| L2 | SAE 非唯一性 × Bilodeau 理论 (formalization §6.4) | 待 DR-010 回复 |
| L3 | 范畴论框架探索 | 长期，待理论线明确后 |
