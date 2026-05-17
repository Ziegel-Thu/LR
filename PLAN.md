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

### 任务 2: exp-006 表征 Scaling Law
- Phase 1 ✅: Pythia 7-scale ladder, kNN power law R²=0.90, β=0.03
- Phase 2 ✅: inter-family β=0.056 > intra-family β=0.025 (2.2x faster)
- Phase 3 待做: representation alignment vs validation loss

### 任务 3: exp-007 Encoding ≠ Use 量化
- Full Run ✅: Ghost ratio = 70.8%（11 features × 24 layers, Pythia-1.4B）
- 下一步：加语义特征（需 NLP 标注）、DAS 替代单方向 ablation

### 任务 4: exp-008 SSM 上的 SAE
- Phase 1 ✅: Mamba-130M SAE, 80.3% var explained, 0% dead features
- Phase 2 ✅: MMCS=0.13, Mamba/Pythia 特征完全不同
- Phase 3 待做: 规模扩展（Mamba-370M, 1.4B）

## 💻 本地进行中

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
