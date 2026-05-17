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

### A. 观察到但未解释的现象

- **(5-P1)** **Encoding ≠ Use 的解离**：Braun+ 2025 (ICML) 在 deep linear networks 中解析证明——表征相似性高不等于功能相似性高，反之亦然。唯一使两者对齐的条件是 parameter noise robustness，而非泛化误差或 input noise robustness。这一解析结果对整个 probing 范式构成根本挑战：probe 测的是 encoding，但模型的 downstream head 用的是 use，二者可以完全解耦。**非线性网络是否也成立？** Braun 只有数值实验暗示，无解析证明。
- **(5-P2)** **Probe 在随机标签上也能成功**：Hewitt & Liang 2019 的 control task 显示，高表达力 probe（MLP）能在随机标签上达到不错准确率——说明 probe accuracy 部分反映 probe 自身能力而非表征质量。但 control task 只给了二值判断（好/坏），**没有给出 "表征质量 vs probe 表达力" 的连续量化**。
- **(5-P3)** **Logit lens 在早期层完全失效但 tuned lens 有效**：logit lens 在 Transformer 早期层输出 garbage（因为残差流尚未被 unembedding 对齐），tuned lens 用 per-layer affine probe 修正后有效。但 **为什么一个简单仿射变换就够了？** 这暗示残差流的早期-晚期关系比预想的更结构化，但无理论解释。
- **(5-P4)** **Edge probing 的 "NLP pipeline" 模式**：Tenney+ 2019 发现 BERT 各层依次编码 POS→constituent→dependency→semantic role，形成经典 NLP pipeline 的镜像。这个 layer-wise 结构为何出现？是训练目标的必然结果还是架构偏好？无人给出因果解释。

### B. 相信但未验证的假设

- **(5-A1)** **"Probe 准确率 = 模型使用了该信息"**——这是全场最大的隐含假设，Braun 2025 直接反驳。Probe 能解码 ≠ 模型在推理时用了它。AMNESIC probing (Elazar+ 2021) 和 AlterRep 试图用因果干预弥补，但只在个别任务上验证，**没有系统性的 "encoding vs use" benchmark**。
- **(5-A2)** **"线性 probe 是 right complexity"**——Alain & Bengio 2016 建议用线性 probe 因为"如果需要非线性 probe 才能解码，信息可能不是 readily accessible"。但这个论证循环：什么叫 readily accessible？Park+ 2024 的 causal inner product 暗示应该用因果度量而非线性度量，但两者关系不清楚。
- **(5-A3)** **"MDL probing 解决了 probe complexity 问题"**——Voita & Titov 2020 用 description length 代替 accuracy，理论上优雅，但 MDL 计算依赖 online coding 的 block size 选择，实际中不同 block size 给出不同排序。**sensitivity analysis 没人做过**。
- **(5-A4)** **"Tuned lens 的仿射 probe 足以捕获 layer-to-output 映射"**——为什么不是 2-layer MLP？为什么不是 kernel method？仿射的选择似乎来自 Occam's razor，但没有系统消融实验验证 affine 是否是 sweet spot。
- **(5-A5)** **"Patchscopes 的 self-explanation 是 faithful"**——Patchscopes 让模型自己 "描述" hidden state，但模型可能 confabulate（和 CoT faithfulness 问题一样）。没有 ground-truth 验证 Patchscopes 的 verbalization 是否真正反映了 hidden state 的内容。

### C. 已知局限未系统研究

- **(5-L1)** **Instruction-tuned / RLHF 模型的 probe leakage**：RLHF 可能压缩表征到更窄的流形上，同时引入 safety-relevant 的隐性表征（如 refusal direction, Sharma+ 2024）。Probe 在 base model 上的结论是否迁移到 chat model？**零系统研究**。
- **(5-L2)** **Probe 的 training data 偏差**：probe 通常在 probe dataset 上训练（如 SentEval），但 probe dataset 的分布可能和模型预训练分布不匹配。这种 distribution shift 对 probe 结论的影响无人量化。
- **(5-L3)** **Layer 选择问题**：probing 需要选择 "哪一层"，不同论文用不同启发式（最佳层、所有层平均、learned 加权）。**没有理论指导什么时候该看哪一层**。
- **(5-L4)** **Token 选择问题**：对 autoregressive LLM，probe 通常取最后一个 token 或 [CLS] token 的表征。但信息可能分布在多个 token 上（尤其长序列）。**token aggregation strategy 的影响未系统研究**。
- **(5-L5)** **Probe 跨模型的可比性**：用同一个 probe 架构在不同模型上训练，然后比较准确率——但不同模型的 representation space 维度、分布、几何结构不同，probe accuracy 不直接可比。MDL 理论上更 principled，但实际中也没做过严格的跨模型比较。

### D. 新场景未探索

- **(5-N1)** **SSM（Mamba/RWKV）上的 lens 方法**——SSM 没有 Transformer 的 clean unembedding matrix，logit lens 不能直接套用。Belrose+ 2024-2025 声称 tuned lens 可部分迁移，但 Balsells Rodas+ 2024 (arXiv:2404.05971 "Does Transformer Interpretability Transfer to RNNs?") 发现信号比 Transformer 弱得多、更 diffuse。**这是一个明确的方法论 gap**——SSM 需要原生的 lens 方法而非 Transformer lens 的适配。
- **(5-N2)** **Multimodal model 的 probing**——VLM（如 GPT-4V, Gemini）在哪一层融合视觉和文本信息？Patchscopes 理论上可以做，但还没有系统的 multimodal probing 研究。视觉 token 和文本 token 的 representation 在什么层开始对齐？
- **(5-N3)** **Cross-model lens transfer**——tuned lens 在 model A 上训练能否用于 model B？如果可以，意味着不同模型共享 "iterative refinement" 的结构；如果不行，每个模型需要单独训练 lens，scalability 差。**零实证研究**。
- **(5-N4)** **训练过程中的 probing 动态**——Pythia 有 154 个 checkpoint，可以追踪 probe accuracy 在训练过程中如何演变。什么时候语法信息出现？什么时候语义信息出现？有没有 phase transition？（类似 Grokking 的 probing 版本）
- **(5-N5)** **Probe × 量化**——INT4/INT8 量化后 probe accuracy 如何变化？哪些语言特征最先被量化损害？与我们 exp-002 的量化悬崖现象直接相关。
- **(5-N6)** **Lens 用于 SAE 特征解释**——用 tuned lens 或 Patchscopes 来 interpret SAE 学到的特征。Auto-interpretation pipeline (Paulo & Belrose 2024) 已开始，但 **没有结合 probing 的 statistical rigor**（control task、MDL）。

### E. 未形式化的直觉

- **(5-I1)** **"Probe 是表征质量的度量"**——直觉上 probe accuracy 应该衡量 "表征多好"，但 Braun 2025 表明这只衡量 encoding 而非 use。需要一个形式化框架：什么条件下 probe accuracy ≈ downstream utility？（可能需要 Braun 的 parameter robustness 条件）。
- **(5-I2)** **"Lens 揭示 iterative inference"**——tuned lens 被解读为"每一层 refine 一个概率分布"，但这只是描述性的。能否形式化为某种 fixed-point iteration 或 variational inference？Geva+ 2021 的 "MLP as key-value memory" 是一步，但离完整理论远。
- **(5-I3)** **"Probe 和 circuit 的关系"**——probe 找到 "哪里有信息"，circuit 找到 "信息怎么被计算和传递"。直觉上 probe 应该是 circuit analysis 的初筛工具，但没有形式化：什么时候 probe positive → circuit positive？反过来呢？
- **(5-I4)** **Probe expressivity 和 representation quality 的 Pareto trade-off**——Hewitt & Liang 2019 指出存在 trade-off 但没有给出 Pareto frontier 的形式。类比 bias-variance trade-off，能否有一个 "probe complexity vs. representation quality" 的解析框架？

### F. 可能不成立的旧结论

- **(5-O1)** **"BERT 编码 NLP pipeline"（Tenney 2019）→ decoder-only LLM**：BERT 是 encoder-only，双向注意力。GPT-style decoder-only 模型的 layer-wise 信息分布可能完全不同。Tenney 的结论是否在 Llama/GPT-4 上成立？只有零星验证。
- **(5-O2)** **"线性 probe 足够"（Alain & Bengio 2016）→ 多模态模型**：视觉-语言融合可能涉及非线性交互（cross-attention、gating），线性 probe 可能根本捕获不到跨模态信息。
- **(5-O3)** **SentEval probing suite（Conneau+ 2018）→ 现代 LLM**：SentEval 的 10 个任务是为 sentence embedding 设计的，在 autoregressive LLM 上意义不大（最后 token 表征 ≠ sentence embedding）。但很多论文还在用。
- **(5-O4)** **Logit lens 的 "iterative refinement" 叙事 → 非 residual-stream 架构**：logit lens 假设残差流从早到晚线性累加，但 SSM 的 state 是 recurrently overwritten 的，不是 residual additive。整个叙事框架需要重建。

### G. 惊人发现 / 反例

- **(5-G1)** **Braun 2025 的核心反例**：parameter-noise-robust 的网络才会使 representational similarity ↔ functional similarity 对齐。大部分正常训练的网络不满足这个条件——意味着 **大量现有 probing 研究的因果推论可能不成立**。
- **(5-G2)** **Patchscopes 比 trained probe 更有效**（Ghandeharioun+ 2024）：用模型自身做 probe 居然比外部训练的 probe 更好——这暗示模型 "知道" 更多关于自己 hidden state 的信息，比外部分析者能恢复的更多。但这也可能是 confabulation。
- **(5-G3)** **SAE features 不如 linear probe**（Chaudhary+ 2024 RAVEL）：SAE 应该给出 "更真实" 的特征分解，但在因果基准上不如简单线性 probe/DAS。这进一步说明 encoding（SAE 找到的方向）≠ causal use。
- **(5-G4)** **Probing "refusal direction" 可以做 inference-time intervention**（Sharma+ 2024）：probe 不仅是分析工具，还能直接用于 steering——这模糊了 probing 和 intervention 的边界。如果 probe 方向 = 可操控方向，encoding 和 use 在这种情况下 *是* 对齐的，这与 Braun 的一般性解离结论矛盾——说明存在 encoding=use 的 special case 值得研究。

### H. 作者明确提出的 Future Work / Open Questions

- **(5-F1)** **DR-013 列出的 5 个 open questions**：(i) probe vs circuit 的关系；(ii) lens transfer；(iii) 扩展到多模态/SSM；(iv) RLHF 模型的 probe leakage；(v) 统一 probe/SAE/DAS/concept erasure。
- **(5-F2)** **Braun+ 2025**：将 dissociation 分析扩展到 nonlinear networks 和 realistic architectures。
- **(5-F3)** **Belrose+ 2023**：tuned lens 的 model-agnostic 版本，以及 SSM 适配。
- **(5-F4)** **Ghandeharioun+ 2024**：Patchscopes 用于 multimodal 和 cross-model interpretation。
- **(5-F5)** **Voita & Titov 2020**：MDL probing 扩展到更复杂的结构（如 tree/graph structured prediction）。

### 最有潜力的研究方向（结合我们的资源）

1. **Encoding ≠ Use 的量化与 benchmark**：以 Braun 2025 为起点，在真实 LLM（Pythia 系列）上系统测量 probe accuracy vs. causal effect（通过 AMNESIC/AlterRep intervention），建立 "encoding-use gap" 的量化指标。可用 jiagpu4-8 跑 Pythia 70M-6.9B 全系列。
2. **SSM 原生 lens 方法**：Mamba 无 unembedding，需要从头设计 lens——可能基于 state-space 的 observation matrix，不是 Transformer 的 residual stream 思路。
3. **Probe × 量化交互**：结合 exp-002 已有的量化实验基础设施，测 INT4/INT8 量化对 probe accuracy 的影响，尤其关注中间层的 "量化悬崖" 是否对应 probe accuracy 的断崖。

**关键参考**：Ghandeharioun+ 2024（Patchscopes）、Belrose+ 2023（tuned lens）、Braun+ 2025（encoding≠use, ICML）、Hewitt & Liang 2019（control task）、Voita & Titov 2020（MDL probing）、Elazar+ 2021（AMNESIC）、Balsells Rodas+ 2024（lens→RNN transfer）、Sharma+ 2024（refusal direction）

## 分支 6：信息论

- **(6a)** MI 估计在高维连续表征中不可靠——所有估计器都有已知 failure mode
- **(6b)** PID 的"正确定义"没有共识（Williams-Beer vs BROJA vs Kolchinsky）
- **(6c)** **信息 → 因果的桥还没建好**（PID 不蕴含因果结构）
- **(6d)** IB/PID 能否从描述工具变成设计原则？

**关键参考**：Ehrlich+ 2024（PID-based complexity）、Murphy+ 2025（causal info decomposition）

## 分支 7：拓扑/几何

### 未解释现象
- **(7-P1)** Hunchback ID 曲线（中间层 ID 高）——所有架构都有但无机制解释。是 manifold unfolding 还是冗余？
- **(7-P2)** LLM 中间层 "语义高原"（Valeriani 2023）——ID 低谷和语义处理的关系是什么？

### 未验证假设
- **(7-A1)** **低 ID 导致好泛化**——所有证据都是 correlational。我们 exp-004: stable rank (r=0.935) 碾压 ID (r=0.365)，暗示 ID 可能只是 epiphenomenon
- **(7-A2)** Local ID (K=50~100) 比 global ID 更有预测力（Yu 2025）——但最优 K 是 task-dependent 的吗？我们用 TwoNN (K=2) 效果很差
- **(7-A3)** TDA/persistent homology 对泛化有预测力——目前全是 post-hoc 描述性分析

### 已知局限未系统研究
- **(7-L1)** ID 估计器之间差 2-3 倍（TwoNN vs MLE vs GeoMLE）——绝对值不可比，但用作度量指标时该瞄准什么？
- **(7-L2)** ID 估计的 sample complexity：TwoNN 需要 N >> e^d，高维最后层 4096 个 probe 可能不够
- **(7-L3)** Confounders：norm、rank、sharpness、neural collapse 都与 ID 和 generalization 共变，因果声明需要 comprehensive matching
- **(7-L4)** 大模型全层 ID profile 计算昂贵，需子采样策略

### 新场景未探索
- **(7-N1)** ID 因果干预实验——训练时加 ID regularizer 看泛化是否跟变。DR-006 已有 7-arm 设计
- **(7-N2)** 多尺度 ID "最佳 K" 的系统研究——K∈{2,10,50,100,500,1000} × 多任务
- **(7-N3)** ID vs stable rank vs effective rank vs shape metric 的 Rosetta Stone——在多架构多数据集上建立关系图谱
- **(7-N4)** TDA 从描述到预测——online persistent entropy regularizer

### 旧结论可能不成立
- **(7-O1)** Ansuini 2019 的 "last-layer ID predicts accuracy" 在 CNN 上——Transformer/SSM 上是否成立？
- **(7-O2)** Hunchback 在不同训练 recipe（RLHF、contrastive、masked LM）下是否保持？

### 与 exp-004 的关键矛盾
- exp-004: stable rank vs accuracy r=0.935, ID vs accuracy r=0.365
- Yu/Cheng 2025: local ID 是强预测器
- 差异来源：(1) TwoNN vs MLE estimator, (2) ResNet vs ViT/ConvNeXt, (3) CIFAR-10 vs ImageNet
- **决定性实验**：同一 setup 上同时测 TwoNN/MLE/stable rank，消除 estimator 差异

**关键参考**：Ansuini+ 2019, Cohen+ 2020, Valeriani+ 2023, Yu/Cheng+ 2025, Tulchinskii+ 2024

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
