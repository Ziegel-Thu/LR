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

## 分支 2：SAE / Dictionary Learning

### 未解释现象
- **(2-P1)** Feature absorption：通用特征被更具体的共现特征吸收——是优化假象（Tang 2024 biconvex 理论的 spurious minima）还是特征层级的自然反映？
- **(2-P2)** SAE 在因果基准（RAVEL）上不如 DAS——SAE 捕获的是几何相关性还是计算因果结构？没人跟进 Chaudhary 2024
- **(2-P3)** 4→2-bit 量化悬崖（我们 exp-002）——中间层最脆弱，稀有特征先崩溃，但无机制解释
- **(2-P4)** 不同种子的 SAE 只共享 30% 特征（Paulo & Belrose）——远低于预期

### 未验证假设
- **(2-A1)** "SAE 特征是正确的分析单元"——多维特征（Engels）、因果失败（Chaudhary）、不可识别性（Friedman）都在挑战
- **(2-A2)** "更大字典 = 更好特征"——Gao 2024 scaling law 只测重建质量，因果指标可能相反
- **(2-A3)** "Feature splitting 是问题"——也许 split 后的子特征才是真实粒度

### 已知局限未系统研究
- **(2-L1)** 所有 SAE 理论基于 1-layer toy model——deep model 跨层 superposition 完全不同
- **(2-L2)** SAE vs DAS 只在一个 benchmark（RAVEL）上比过——需要系统对比
- **(2-L3)** Dead feature 的理论含义不明——是字典太大？训练不足？优化陷阱？
- **(2-L4)** 评估指标（MMCS）和因果有效性相关性极弱（Karvonen 2025）

### 新场景未探索
- **(2-N1)** **SSM（Mamba/RWKV）上的 SAE——零论文**，低垂果实
- **(2-N2)** SAE 特征在训练过程中的涌现/死亡（纵向追踪 Pythia checkpoints）
- **(2-N3)** 量化精度 × SAE 特征的 scaling（我们 exp-002 独有方向）
- **(2-N4)** 跨模型家族的特征 universality（Llama vs Gemma vs Mistral 的 SAE 特征是否对应？）

### 旧结论可能不成立
- **(2-O1)** 1-layer toy model 结论 → deep model：残差流 skip connection 使跨层 superposition 完全不同
- **(2-O2)** Pythia 结论 → frontier model：更大 d_model 可能使 superposition 更松散或更紧密，未知
- **(2-O3)** 重建质量 scaling law → 因果有效性 scaling law：可能方向相反

### 未形式化直觉
- **(2-I1)** Superposition 的 packing density 没有估计方法——N 个特征在 d 维空间，N 是多少？
- **(2-I2)** Dead feature rate 可能是特征总数 N 的估计器：dead% ≈ max(0, d_sae - N) / d_sae

**关键参考**：Elhage+ 2022, Bricken+ 2023, Gao+ 2024, Engels+ 2024, Chaudhary+ 2024, Chanin+ 2024, Tang+ 2024, Paulo & Belrose 2024

## 分支 3：因果抽象 / Mech Interp

### 未解释现象
- **(3-P1)** Sutter trivialization 的程度梯度：unrestricted alignment → 100% IIA（随机网络也行），但 linear alignment → 训练网络 > 随机网络。这个连续过渡无理论刻画
- **(3-P2)** IOI circuit 只解释 ~70% 行为——剩余 30% "暗物质"的结构是什么？冗余 backup？其他任务的计算？方法论 artifact？
- **(3-P3)** SAE features 在 circuit-level 有用（Marks 2024 sparse feature circuits）但 individual feature-level 不如 DAS/prompting（Wu 2025 AxBench）——矛盾
- **(3-P4)** Denoising patching ≠ noising patching（Zhang & Nanda 2024）——方向性不对称无理论解释

### 未验证假设
- **(3-A1)** "Circuit = Algorithm" 隐喻——Sutter 表明在足够灵活的 alignment 下任何算法都能 align 到任何网络
- **(3-A2)** "线性 alignment 就一定有意义"——为什么线性是 sweet spot？本身需要论证
- **(3-A3)** Interchange intervention = 因果测试——off-distribution intervention 使结论可靠性存疑（Sun 2025）
- **(3-A4)** "Small circuits exist"——从未系统验证适用范围，复杂行为可能本质 distributed

### 已知局限未系统研究
- **(3-L1)** Off-distribution intervention 的量化——patched activation 离 on-distribution manifold 多远？无人测量
- **(3-L2)** Faithfulness-Completeness Pareto frontier 的理论基础——可能是 Bilodeau-Sutter squeeze 的实证表现
- **(3-L3)** Circuit 跨任务一致性——IOI/greater-than/factual recall 的 circuit 是否共享 heads？角色兼容吗？
- **(3-L4)** 模型规模对 circuit 结构的影响——几乎全在 GPT-2 small 上，无跨规模研究

### 新场景未探索
- **(3-N1)** 量化网络的因果结构——INT4 量化后 circuit 是否保持？
- **(3-N2)** Multi-step reasoning / CoT 的 across-token 因果结构
- **(3-N3)** Vision Transformers 的 circuit structure
- **(3-N4)** SSM/Mamba 的因果结构——无 attention heads，intervention 定义需适配

### 未形式化直觉
- **(3-I1)** "正确的分析单元"（heads vs features vs circuits）——什么使一个单位"更好"未定义
- **(3-I2)** "Circuit residual = 噪声"——70% explained 30% dark matter 的四种可能解读都未被测试
- **(3-I3)** Prompting > representation steering（Wu 2025）——如果 external API 更好用，internal API 的价值在哪？

### 旧结论可能不成立
- **(3-O1)** GPT-2 small circuits → 大模型：冗余路径增多、superposition 更密
- **(3-O2)** Base model patching → RLHF 模型：activation 分布被约束到更窄流形
- **(3-O3)** Transformer causal abstraction → SSM：框架需要根本适配

### 与我们 impossibility 工作的联系
- 我们的五公理框架直接适用于 causal abstraction
- Sutter trivialization = Incompatibility III: {Identifiability + Generality + ¬Finiteness} → ¬Non-triviality
- Squeeze Conjecture 的实验验证可以定量化 Sutter dilemma

**关键参考**：Geiger+ 2025 JMLR, Sutter+ 2025, Wu+ 2025, Wang+ 2022, Sun+ 2025, Zhang & Nanda 2024, Marks+ 2024

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
