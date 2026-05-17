# 表征学习/分析领域 Open Problems

> 来源：DR-013 论文地图 + 论文精读（2026-05-17）

## 分支 1：相似度度量

### 未解释现象
- **(1-P1)** U 形深度曲线：早晚层跨架构相似，中间层分化——所有人观察到但无机制解释
- **(1-P2)** CKA 高不等于编码了任务信息（Cloos：CKA≈1 但任务变量丢失）
- **(1-P3)** 随机网络也能高 CKA/RSA（Cui 2022：输入相似性 confound）

### 未验证假设
- **(1-A1)** "CKA 是好默认"——Kornblith 2019 确立，Cloos/Ding 质疑，但大家还在用
- **(1-A2)** 均值池化不丢信息——所有人用但没人验证
- **(1-A3)** CKA=0.7 意味什么？中间值无可解释含义，无 calibration 理论（Klabunde §6）
- **(1-A4)** CKA 高 → 迁移/stitching 效果好？从未验证

### 已知局限未系统研究
- **(1-L1)** Procrustes 分数随维度升高（Klabunde）——比较不同 d_model 的模型时 confounded
- **(1-L2)** 样本量效应：shape metric 需要大 N/D 才稳定（Williams 2021），我们 exp-003 N/D≈0.65 可能不稳定
- **(1-L3)** 所有度量假设 IID stimuli（Harvey §2.1），但语言是序列的
- **(1-L4)** 12+ 种 CKA 变体没人系统比较过（Cloos future work）

### 新场景未探索
- **(1-N1)** SSM（Mamba/RWKV）上的度量行为——我们 exp-003 可能是首个
- **(1-N2)** 度量作为训练目标（知识蒸馏用 Procrustes loss）
- **(1-N3)** RLHF 前后表征变化的度量
- **(1-N4)** 动态/token-by-token 表征的比较

### 旧结论可能不成立
- **(1-O1)** 所有 CKA 结论在 CNN 上验证，Transformer/SSM 谱结构可能不同
- **(1-O2)** Harvey "几何≈功能" 的 bound 在高维下松动（participation ratio 因子）
- **(1-O3)** 有限样本 regime 下理论结果的适用性（Harvey 自己提出）

**关键参考**：Cloos+ 2024, Harvey+ 2024, Klabunde+ 2023, Williams+ 2021, Kornblith+ 2019

## 分支 2：SAE

- **(2a)** Feature absorption/splitting 说明字典没有收敛到"真"特征集——有没有"真"的？
- **(2b)** 多维特征（圆环、曲面）挑战"每特征一方向"假设
- **(2c)** SAE 在因果基准上不如 DAS——捕获的是计算还是相关几何？
- **(2d)** 评估标准不统一——SAEBench、RAVEL、feature-circuit 排名互相矛盾

**关键参考**：Engels+ 2024（多维特征）、Chaudhary+ 2024（SAE vs DAS on RAVEL）、Friedman+ 2024（SAE 理论限制）

## 分支 3：因果抽象 / Mech Interp

- **(3a)** 干预产生 off-distribution 激活——怎么处理？（Sun+ 2025 刚开始）
- **(3b)** 电路只解释小任务，不 scale 到 CoT/reasoning/agent
- **(3c)** Faithfulness vs completeness **不可同时满足**（Wu+ 2025）
- **(3d)** SAE 特征图如何变成真正的因果解释？

**关键参考**：Geiger+ 2025（统一框架 JMLR）、Wu+ 2025（faithfulness-completeness trade-off）

## 分支 4：线性表征假说 (LRH)

- **(4a)** LRH 什么时候成立什么时候失败？边界在哪里？
- **(4b)** 有没有"正则内积"使线性性最 sharp？（因果的？白化的？）
- **(4c)** **为什么训练会产生线性概念？**——实验证据很多但零理论解释
- **(4d)** 多个概念方向如何交互？叠加、组合、还是干扰？

**关键参考**：Park+ 2024（因果形式化）、Engels+ 2024（反例）、Jiang+ 2024（log-linear 猜想）

## 分支 5：Probing / Lens

- **(5a)** Probe 成功意味着什么？编码 ≠ 使用（Braun 2025）
- **(5b)** Lens 不能跨模型迁移——有没有 model-agnostic lens？
- **(5c)** 怎么扩展到多模态、视觉、SSM（无 clean unembedding）

**关键参考**：Ghandeharioun+ 2024（Patchscopes）、Belrose+ 2023（tuned lens）

## 分支 6：信息论

- **(6a)** MI 估计在高维连续表征中不可靠——所有估计器都有已知 failure mode
- **(6b)** PID 的"正确定义"没有共识（Williams-Beer vs BROJA vs Kolchinsky）
- **(6c)** **信息 → 因果的桥还没建好**（PID 不蕴含因果结构）
- **(6d)** IB/PID 能否从描述工具变成设计原则？

**关键参考**：Ehrlich+ 2024（PID-based complexity）、Murphy+ 2025（causal info decomposition）

## 分支 7：拓扑/几何

- **(7a)** ID 估计器之间差 2-3 倍（TwoNN vs MLE vs GeoMLE）——哪个对？
- **(7b)** **低 ID 是泛化的原因还是结果？** 因果方向不清楚
- **(7c)** 拓扑不变量更多是描述性的，不是预测性的
- **(7d)** 怎么处理 RNN/SSM 和 in-context learning 的动态表征？

**关键参考**：Ansuini+ 2019（hunchback）、Cheng+ 2025（local ID）、Valeriani+ 2023（语义高原）

## 分支 8：解纠缠/可识别性

- **(8a)** 可识别性理论假设太强（需干预/环境标签）
- **(8b)** LLM 表征看起来解纠缠但没用 CRL 推导——两者怎么连？
- **(8c)** 预训练模型是否在识别底层因果结构？
- **(8d)** 评估指标继承了 Locatello 指出的同样问题

**关键参考**：Locatello+ 2019（不可能性）、Khemakhem+ 2020（iVAE）、Reizinger+ 2024（LLM 可识别性）

## 分支 9：Scaling / 收敛

- **(9a)** 收敛到唯一解还是低维族？（Anna Karenina vs 多解）
- **(9b)** 选择压力来自哪里？——数据、loss、优化器、还是正则化？
- **(9c)** 收敛失败的边界条件——Ziyin 2025 开了头但远没定论
- **(9d)** 表征质量的 scaling law 该怎么量化？（不是 loss 的 scaling law）

**关键参考**：Huh+ 2024（Platonic）、Ziyin+ 2025（收敛 breaks）、Bansal+ 2021（model stitching）

## 分支 10：不可能性/理论基础

- **(10a)** 有没有统一的公理化框架用于表征比较？（类似 Kleinberg 对聚类做的）
- **(10b)** SAE/probe/DAS 什么条件下保证恢复真实结构？
- **(10c)** Faithfulness/completeness/simplicity/minimality 能否同时达到？
- **(10d)** 黑箱 vs 权重 vs 训练过程——不同访问级别下可解释性的信息论上限？

**关键参考**：Kleinberg 2002（聚类不可能性模板）、Wu+ 2025、Friedman+ 2024

---

## 最有空间的方向（主观判断）

1. **为什么 LRH 成立** (4c) — 实验证据很多但零理论解释
2. **收敛的边界条件** (9c) — Ziyin 开了头，但远没定论
3. **ID 的因果角色** (7b) — 低 ID 是因还是果？
4. **度量选择的公理化** (1a) — Cloos 指出了问题，但没给解
5. **表征质量 scaling law** (9d) — 大家只做 loss scaling，没做表征 scaling
