### Understanding models research brief

用户:


Claude:
his is a thorough and well-structured research brief — covering probing work across a wide spectrum of "understanding" models, from V-JEPA to OthelloGPT to latent reasoning models like Coconut/Huginn. The request is detailed enough that I can dive straight in. Let me kick off an in-depth research process now.
our research synthesis on probing "understanding-type models" is ready. It surveys 100+ papers from 2018-2026 across model families including JEPA, DreamerV3, OthelloGPT, Chess-GPT/Leela, CLIP, LLMs, and latent reasoning models like Coconut/Huginn. The report maps probed concepts across four abstraction levels (low/mid/high/abstract), reviews methodologies for distinguishing genuine understanding from statistical correlation (selectivity, MDL, amnesic probing, causal abstraction), and identifies a critical asymmetry: low/mid-level concepts are well-probed while conservation laws, symmetries, counterfactuals, and planning remain largely unexplored. It concludes with 5 high-value proposed experiments, including conservation-law probing in V-JEPA 2, systematic RSSM probing in DreamerV3, and a unified cross-model abstract concept benchmark.
# 「理解型模型」探测研究综述:从棋盘到世界模型的认知考古

## 核心结论(BLUF)

**当前对「理解型模型」的探测研究呈现一个清晰的不对称格局:低中层概念(棋盘状态、空间坐标、运动方向)已被广泛、可靠地线性解码,而高阶抽象概念(守恒律、对称性、反事实、规划)在绝大多数模型家族中几乎完全未被探测。** 这一格局意味着,关于「模型是否真正理解」的争论目前主要建立在低阶证据之上,而最关键的证据——抽象概念探测——仍是巨大的研究空白。在过去三年(2023–2026)中,OthelloGPT、Chess-GPT、Leela 国际象棋网络、V-JEPA 2 等少数几个模型获得了较深入的因果干预级探测,而 DreamerV3 的 RSSM、Genie 的潜动作码、UniSim、GAIA、Coconut 的连续思维等被广泛宣称「学到了理解」的模型,其内部表征几乎从未被严肃地探测过。**第三方独立复现实验普遍弱化了原作者关于「潜在推理」「世界模型」的乐观主张**(Vafa et al. 2024 关于 LLM 出租车地图的不连贯结果;Rizvi-Martel et al. 2026 表明 Coconut 的"叠加态"在前几层即坍缩为快捷方式)。本报告系统梳理了 100+ 篇 2018–2026 年文献,按模型家族整理探测工作,提出概念层级图,综述区分理解与统计相关性的方法论,并据此推荐 5 项最具价值的待做探测实验。

---

## 一、按模型家族组织的探测工作

### 1.1 JEPA 家族(I-JEPA / V-JEPA / V-JEPA 2 / V-JEPA 2.1)

**已探测的属性涵盖语义类别、深度、计数、动作识别、运动学量与直觉物理违反期望(VoE)。** I-JEPA(Assran 等,CVPR 2023, arXiv:2301.08243)用线性探测在 ImageNet、CIFAR100、Clevr/Count、Clevr/Dist 上展示其表征同时编码语义类别与几何信息;V-JEPA(Bardes 等, arXiv:2404.08471, 2024)引入"注意力探测头"(attentive probe)作为标准评估方法,在 SSv2 等运动密集数据集上显著优于像素级 MAE。V-JEPA 2(Assran 等, arXiv:2506.09985, 2025)进一步将注意力探测扩展到动作预期(Epic-Kitchens-100)、视频问答(PerceptionTest 84.0)与零样本机器人规划。V-JEPA 2.1(Mur-Labadia 等, arXiv:2603.14482, 2026)展示了线性深度探测在 NYUv2 上达到 **0.307 RMSE**,与 DINOv3-7B 持平。

**直觉物理探测是 JEPA 研究的高峰。** Garrido 等(arXiv:2502.11831, 2025)使用**违反期望(VoE)零样本探测**:用 JEPA 自身的预测器计算掩码未来帧的 L1 距离作为"惊讶值"。V-JEPA 在 IntPhys 上达到 **98%**,GRASP 66%,InfLevel-lab 62%——而像素空间的 VideoMAEv2 与文本推理的 Qwen2-VL、Gemini 1.5 Pro 接近随机水平。**对照实验关键:同架构随机初始化 V-JEPA 也接近随机**,排除了架构本身的功劳。物体永恒、形状恒常、支撑、惯性显著;但**颜色恒常、固体性、碰撞、(部分)重力不显著**——说明 JEPA 学到的是物体追踪类静态物理,而非动力相互作用。Bordes 等(IntPhys 2, arXiv:2506.09849, 2025)更难的基准上,所有 V-JEPA 变体崩塌至接近随机,**与 85–95% 的人类水平存在巨大鸿沟**。

**机制可解释性方面最重要的工作**是 Joseph 等(arXiv:2602.07050, 2026)的《Interpreting Physics in Video World Models》。他们在 V-JEPA 2(L/H/G)上发现:(1) 一个尖锐的"**物理涌现区**"在编码器约 1/3 深度处出现,IntPhys 准确率从随机跳到 85–95%,**且在中层达到峰值,末层反而下降**;(2) **速度、加速度可从早期层线性读出,但方向只在物理涌现区涌现**;(3) 加速度不依赖于显式速度表征——反驳了"物理引擎式阶段化派生"假设;(4) 方向被编码为**高维分布式环形群体码**(类似 MT 区方向选择性细胞),单向量引导失败,需协同调整 40–80 维;(5) **关闭关键局部注意力头使 IntPhys 从 78.3% 降至 52%,而 ImageNet 几乎不受影响**——证明因果使用,而非仅相关。

**抽象概念探测的明确空白:** 截至 2026 年中,**没有任何研究在 JEPA 表征上探测能量、动量、质量守恒,也没有研究测试表征对平移/旋转/时间反演的对称不变性。** 因果性仅通过 CausalVQA 行为基准与 C-JEPA(arXiv:2602.11389,2026)的架构层干预间接探测,但**未有人在冻结 V-JEPA 潜空间内做反事实中介分析**。Liu(arXiv:2603.20327,2026)的"AI Mother Tongue"被动量化探测发现 JEPA 潜空间含可分离的物理范畴信号(χ² p < 10⁻⁴),但互信息仅 0.036–0.117 bits——存在但微弱。

### 1.2 世界模型家族(DreamerV3 / IRIS / DIAMOND / Genie / UniSim)

**令人惊讶的是,DreamerV3 的 RSSM 从未被正式探测过。** 截至 2026 年 5 月,没有发表论文系统地用线性/MLP 探测从 DreamerV3 的 h_t(确定状态)或 z_t(分类状态)中恢复环境变量。Wenninghoff 与 Schwammberger(Springer LNCS, 2025)在五个 Minigrid 任务上比较真实与"想象"轨迹,但未做状态解码;Okada 与 Taniguchi 的 DreamingV2 论文做了 lift 任务上的定性可视化。Peper 等(arXiv:2503.02143, 2025)的《Four Principles for Physically Interpretable World Models》明确诊断了这一空白,指出 RSSM 风格世界模型"缺乏物理可解释性"。

**Zhang(arXiv:2603.21546, 2026)的《What Do World Models Learn in RL?》是目前唯一**对 IRIS(VQ-VAE+GPT-2)与 DIAMOND(扩散去噪 UNet)做正式探测的研究。在 Atari Breakout/Pong 上用 AtariARI 标签训练 Ridge 线性与 2 层 MLP 探测,加上沿探测方向的因果干预。关键发现:**IRIS 中 ball_x 线性 R² ≈ 0.84,paddle_x 与 score > 0.99**;VQ-VAE 编码器自身已达 R²=0.83,Transformer 仅再加 0.02。DIAMOND 出现**倒 V 字模式**——早期编码器层线性 R² 为负(最低 −1.45),瓶颈层升至 0.81,解码器线性下降但 MLP 探测恢复至 0.91(跳跃连接的非线性编码)。沿探测方向的干预的 KL 散度与扰动幅度相关系数 r ≥ 0.96,且 paddle 干预效应是 ball 的约 16 倍——**世界模型对策略相关状态分配更多容量**。

**Genie、Genie 2、UniSim、GAIA-1/2 完全没有发表的探测研究。** Genie(Bruce 等,DeepMind 2024)选择 8 个离散潜动作码"以鼓励可解释性",但没有人定量探测每个码对应的物理/动作语义。UniSim 与 GAIA 的内部表征同样从未被解码。OpenVLA 是少数被探测的机器人基础模型(Molinari, arXiv:2509.24559, 2025):**线性/非线性探测显示状态转移可通过嵌入算术(state_{t+1} ≈ state_t + Δ)预测,且世界模型属性随训练递增**。

**反事实推理在世界模型中几乎未被探测。** Shen 等(CWMDT, arXiv:2511.17481, 2025)明确指出"当前世界模型基于事实观察预测,无法回答'如果移除此物体会怎样'"——他们提出添加结构化"数字孪生"来注入反事实能力,而**不是探测现有模型是否内部表征了反事实**。这是 DreamerV3、IRIS、DIAMOND、Genie、UniSim 集体的巨大空白。

### 1.3 OthelloGPT 与棋类 Transformer

**OthelloGPT 是世界模型探测文献的奠基性案例。** Li 等(ICLR 2023, arXiv:2210.13382)用 64 个两层 MLP(非线性)探测在 8 层 GPT 的残差流上恢复 8×8 棋盘,误差仅 1.7%(随机模型 26.2%);干预实验通过编辑激活改变模型的合法下一步预测,证明**因果使用**。Nanda、Lee、Wattenberg(arXiv:2309.00941, 2023)以"我方/对方"相对坐标重参数化后,**线性探测在第 4 层后准确率 > 99%**——确立"线性表征假设"。Hazineh 等(arXiv:2310.07582, 2023)进一步识别"Mine-Heads"与"Yours-Heads"等注意力头。

**关于策略与规划的探测仍极度稀缺。** Du 等(arXiv:2501.07108, 2025)发现 SAE 在深层抽取出"**tile stability(棋子稳定性)**"特征,这是首次在 OthelloGPT 中发现的高级策略概念(对应长期棋盘控制)。Yuan 与 Søgaard(arXiv:2503.04421, 2025)在 7 种 LLM 架构上**行为性测试两跳预测**,准确率降至 < 70%——直接证据表明**当前 OthelloGPT 类模型不可靠地表征超过一步的未来**。jylin04、Karvonen、Rager 的 LessWrong 帖子《OthelloGPT learned a bag of heuristics》(2024)显示中后期 MLP 神经元充当位置特定的逻辑规则探测器(F1 > 0.99),但**规则不跨棋盘平移泛化**——支持"启发式集合"而非"清晰算法"的图景。Singh 等(arXiv:2511.00059, 2025)用决策树自动提取规则神经元。SAE 在 OthelloGPT 上的覆盖率有限:Robert_AIZI(2024)报告 SAE 只恢复 9/180 个棋盘状态特征(后修订为 33/180);Karvonen 等(NeurIPS 2024, arXiv:2408.00113)的板状态属性覆盖率达 F1 ≈ 0.95,但仍落后于监督线性探测。

**国际象棋 Transformer 的探测远比 OthelloGPT 深入。** Karvonen(arXiv:2403.15498, 2024)的 50M Chess-GPT 用线性探测恢复完整棋盘**并恢复玩家 Elo**(分类 <1550 与 >2050 玩家的准确率 90.5%);沿 Elo 方向的对比激活干预可改变模型的下棋强度。McGrath 等(PNAS 2022, arXiv:2111.09259)在 AlphaZero 内部线性探测物质、机动性、王安全、威胁等人类概念,发现概念学习的"起飞点"现象。最重要的是 **Jenner、Kapur、Georgiev、Allen、Emmons、Russell(NeurIPS 2024, arXiv:2406.00877)的《Evidence of Learned Look-Ahead in a Chess-Playing Neural Network》**:在 Leela Chess Zero 上,(1) 对未来最优着法目标格的激活补丁因果重要;(2) 识别出 L12H12 注意力头将信息从未来格回流到当前格;(3) **双线性探测预测 2 步后最优着法准确率 92%**。Cruz 等(arXiv:2505.21552, 2025)将探测延伸至 7 步,显示前瞻高度依赖位置类型;arXiv:2604.10158(2026)用稀疏替换层在 BT4 LC0 上发现并行分布式推理路径与"过度防御"病理状态。这是迄今**最强的 Transformer 内部搜索/规划证据**。Du 等(LessWrong, 2023)在 Leela Zero(围棋)中通过 logit lens 识别出"梯子"战术的因果电路。

### 1.4 多模态模型(CLIP / SigLIP / ImageBind / LLaVA)

**CLIP 类模型的核心探测结论:对象、属性、关系的"袋子化"行为普遍存在,组合性与空间推理几乎全面失败。** Winoground(Thrush 等,CVPR 2022, arXiv:2204.03162)的 400 个最小词序对让 CLIP、FLAVA、ViLT、UNITER 接近随机(组得分 ~10%,人类 85.5%)。Yuksekgonul 等(ICLR 2023, arXiv:2210.01936)的 ARO 基准是关键论文——CLIP、BLIP、X-VLM **对词序完全不敏感**:打乱 COCO 字幕几乎不影响匹配分数,因为对比检索目标无需组合信息即可解决。VL-CheckList、VALSE、CREPE 进一步细化,关系与计数最弱。

**关键的方法论自我修正**来自 SugarCrepe(Hsieh 等,NeurIPS 2023 D&B, arXiv:2306.14610):许多组合性基准是"可破解的"——纯文本盲模型(语法分数、Vera 合理性)在 ARO 上超过 SOTA CLIP。SugarCrepe 通过对抗过滤去除文本偏差后,NegCLIP 的收益急剧缩小。SugarCrepe++ 与《Hard Positive Truth》(arXiv:2409.17958, ECCV 2024)显示硬负样本微调的改进多为虚假——模型对语义保持的释义变得脆弱。

**空间推理探测全面失败。** Liu 等(VSR, TACL 2023)、Kamath 等(What'sUp, EMNLP 2023, arXiv:2310.19785)显示 18 个 VLM 在受控的左/右/上/下任务上表现不佳——BLIP 微调后仅 56%,人类 99%。**根本原因被定位为文本编码器瓶颈**:Kamath 等(EMNLP 2023, arXiv:2305.14897)的恢复探测证明 CLIP 的单向量文本表征本身就丢失了对象关系、属性绑定、计数、否定、空间/时间关系。

**Khajuria 等(arXiv:2502.03566, 2025)做了重要细化**:CLIP 在**单模态内**属性-对象绑定**是**线性可解码的,失败仅发生在**跨模态对齐**阶段——简单的线性变换即可恢复跨模态绑定。这意味着"袋子化"是对齐瑕疵而非信息缺失。

**机制可解释性:** Goh 等(Distill 2021)发现 CLIP RN50x4 中的**多模态神经元**(蜘蛛侠神经元、特朗普神经元),并记录了"印刷攻击"(在苹果上写 iPod 致误分类)。Gandelsman 等(arXiv:2406.04341, 2024)解释 CLIP ViT 神经元的二阶效应。PatchSAE(Lim 等,ICLR 2025)与 MSAE(Zaigrajew 等,ICML 2025)用稀疏自编码器抽取出 120+ 可解释概念。Tong、Liu、LeCun 等(《Eyes Wide Shut》/MMVP, CVPR 2024, arXiv:2401.06209)的**关键发现是 CLIP 失败与下游 LLaVA/InstructBLIP 失败之间 Pearson 相关 > 0.7**——多模态 LLM 的视觉缺陷直接继承自 CLIP。

**ImageBind 几乎无探测研究**——除了原论文的线性探测评估,没有针对其六模态绑定的组合性、空间、否定探测;这是公开的研究空白。

### 1.5 大语言模型(GPT、LLaMA 等)

**LLM 中"世界模型"证据呈现深度的两极分化。** 正面证据:Gurnee 与 Tegmark(ICLR 2024, arXiv:2310.02207)在 Llama-2(7B/13B/70B)中用 ridge 线性探测**恢复出全球地点、美国地点、纽约地点的纬经度,以及历史人物/艺术品/新闻的年份**——多尺度的"空间方向"与"时间方向"在中后层呈线性。Karvonen 的国际象棋 Elo 探测、Toshniwal 等(2021)的 PGN 状态追踪、Hao 等的 TextWorld 涌现状态表征,均显示 LLM 在结构化领域形成可探测的内部表征。Park、Choe、Veitch(ICML 2024, arXiv:2311.03658)形式化了"线性表征假说"。

**负面证据同样有力且系统。** **Vafa 等(NeurIPS 2024 Spotlight, arXiv:2406.03689)的《Evaluating the World Model Implicit in a Generative Model》**用 Myhill–Nerode 定理构造序列压缩与序列区分度量。在 NYC 出租车导航 Transformer 上:**最短路径预测近乎完美,但重构的"地图"包含不存在的街道**,关闭少数街道即导致灾难性失败。这是迄今**最强反对天真世界模型主张**的负面结果。Berglund 等(《反转诅咒》, ICLR 2024, arXiv:2309.12288)证明在"A is B"上训练的 LLM 回答"B is A"不优于随机——逻辑对称性的根本失败。Dziri 等(《Faith and Fate》, NeurIPS 2023, arXiv:2305.18654)显示多步多位数乘法、逻辑谜题、动态规划的准确率随组合深度从近完美崩塌至零——**Transformer 将多步推理归约为对训练中见过的子图的线性化模式匹配**。McCoy 等(PNAS 2024, arXiv:2309.13638)的《Embers of Autoregression》提供决定性例证:**GPT-4 解码 rot-13 密码在明文为高概率英语时准确率 51%,在低概率英语时降至 13%**,而任务本身完全确定性——LLM 的"准确率"由任务概率、目标概率、输入概率主导。

**Wu 等(NAACL 2024, arXiv:2307.02477)在 11 个反事实任务**(非十进制算术、棋子互换的国际象棋、1-索引 Python)上证明 GPT-3.5/4 在反事实变体上始终显著差于默认变体——记忆与推理在连续谱上共存。**Kıcıman 等(arXiv:2305.00050)**报告 GPT-4 在 Tübingen 因果对数据集上 ~97% 准确率,但 **Jin 等的 CLadder(NeurIPS 2023, arXiv:2312.04350)**显示即使 GPT-4 在形式化反事实/干预推理上仍困难重重;**CounterBench(Wang 等, arXiv:2502.11008, 2025)**显示 GPT-4o、Claude 3.5、DeepSeek-V3 在严格反事实题上接近随机猜测。

**心智理论辩论:** Kosinski(PNAS 2024)报告 GPT-4 通过 95% 假信念任务,但 Ullman(arXiv:2302.08399)显示对 ToM 无关的小扰动(透明容器、"on" vs "in")令 GPT-3.5 性能崩塌;Shapira 等(2023)与 Sclar 等(ExploreToM, arXiv:2412.12175, 2024)的**对抗式 A* 生成 ToM 故事让 Llama-3.1-70B 得分 0%、GPT-4o 仅 9%**。

**数字理解:** Zhu 等与 arXiv:2506.08966(2025)显示 Llama-3、Phi-4、OLMo-2 用正弦归纳偏置探测可**近乎完美地恢复数值**;arXiv:2502.16147 发现亚线性(对数)压缩,符合 Weber–Fechner 律。Ye 等(《Physics of Language Models 2.1》, ICLR 2025, arXiv:2407.20311)在受控合成 GSM 类数据上证明 GPT-2 规模模型发展出**真正的多步"心智"推理**(在计算前预测所需变量)。

**机制可解释性指向"理解"的最强证据**来自 Anthropic 的 Scaling Monosemanticity(Templeton 等,2024)与 Lindsey 等的《On the Biology of a Large Language Model》(2025):SAE 从 Claude 3 Sonnet 抽取数千万个单义特征,包括代码缺陷、内心冲突、谄媚等抽象概念;**电路追踪图揭示了诗歌中的前瞻规划**(模型在写诗行前选定押韵目标)、多语言概念回路与独立的"我知道这个实体吗"元认知回路。Gurnee 等(《Finding Neurons in a Haystack》, TMLR, arXiv:2305.01610)用 k-稀疏线性探测在 7 个模型(70M–6.9B)上验证早层叠加 → 中层专用神经元的进展。

### 1.6 潜在推理模型(Coconut / Huginn / CODI / 暂停令牌)

**这是 2024–2026 年最具争议的领域,第三方探测系统性地弱化了原作者的乐观主张。** Hao 等的 Coconut(Meta, arXiv:2412.06769, 2024)将思维放入连续潜空间,用 LM 头对第一个连续思维做 logit lens,声称其在 ProsQA 上**编码了多个候选路径的潜在广度优先搜索**。Geiping 等的 Huginn-3.5B(arXiv:2502.05171, 2025)在递归深度块上展示了非平凡的轨道式潜状态轨迹。

**但独立复现普遍负面。** Lu 等(arXiv:2507.02199, COLM 2025)在 Huginn 上用 Logit Lens 与新颖的 Coda Lens 发现:递归块间存在**显著的探测不一致性**;token-rank 轨迹**未显示中间结果与最终答案的时序分离**;**递归深度的增加仅带来边际收益**,远不及显式 CoT。Rizvi-Martel 等(arXiv:2604.06374, ICLR 2026 Workshop)的《The Illusion of Superposition?》——对 Soft Thinking,叠加在前几层即坍缩(熵剖面与 KL 散度几乎与离散 CoT 一致);**对微调的 Coconut,实体级探测显示模型直接从问题表征提取答案,零潜在令牌时仍达 96.6% 准确率**——证明使用快捷方式而非真正的 BFS。仅"从零训练含潜在思维"的模型显示叠加迹象。Zhang 等(arXiv:2512.21711, 2025)的《Do Latent Tokens Think?》在 SJTU 通过激活引导与 OOD 测试得出"**潜在令牌作为不可解释占位符**"、"Coconut 比显式 CoT 更依赖快捷方式"。Liang 与 Pan(arXiv:2602.00449, 2026)在 CODI 上发现 2–3 跳任务有忠实的中间桥接,但 ≥4 跳时退化为**最终边界处的晚期融合**而非完整滚动。Du 等(《Latent Thinking Optimization》, arXiv:2509.26314, 2025/2026)证明**Huginn 潜在轨迹上正确性信号线性可解码**,即使具体步骤不可解释——可作为推理时搜索的潜在奖励信号。Blayney 等(arXiv:2604.11791, 2026)从不动点动力学角度证明**循环 LM 的"推理阶段"在每次迭代中重复出现而非编排为顺序步骤**。

**暂停/填充令牌:** Pfau、Merrill、Bowman(《Let's Think Dot by Dot》, COLM 2024, arXiv:2404.15758)证明填充 token 在 3SUM/2SUM 上启用了立即回答无法解决的问题,但仅在密集可并行监督下有效;Bharadwaj(arXiv:2412.04537, 2024)用 token-rank 解码发现**填充 token 模型的 rank-2 token 揭示了连贯的隐藏 CoT 痕迹**。最具颠覆性的是 Stechly、Valmeekam 等(arXiv:2505.13775, 2025)的《Beyond Semantics》:**交换中间 token 的语义内容不损害性能,只有结构长度重要**——警告不要拟人化 CoT/潜在 CoT 内容。Quiet-STaR 至今**没有系统性探测研究**(空白)。

---

## 二、被探测概念的层级图谱

下表汇总各模型家族在四个抽象层级的探测覆盖度,**✓** 表示有发表研究,**部分** 表示间接或弱证据,**空白** 表示截至 2026 年 5 月无公开探测研究。

| 模型家族 | 低阶(像素/token 属性) | 中阶(对象类别/句法) | 高阶(语义/因果关系) | 抽象(守恒/对称/反事实/规划) |
|---|---|---|---|---|
| **I-JEPA / V-JEPA / V-JEPA 2** | ✓(深度、计数、运动)| ✓(语义类、动作识别)| 部分(物体永恒、惯性、支撑)| 空白(守恒/对称);部分(因果 via CausalVQA 行为);空白(JEPA 内部反事实)|
| **DreamerV3 RSSM** | 空白 | 空白 | 空白 | 空白 |
| **IRIS / DIAMOND** | ✓(球/拍位置、分数)| ✓(对象、玩家状态)| 空白 | 空白 |
| **Genie / UniSim / GAIA** | 空白 | 空白 | 空白 | 空白 |
| **OthelloGPT** | ✓(棋盘格状态)| ✓(我方/对方)| 部分(翻转检测、规则神经元)| 极少(仅 tile stability;无明确规划/前瞻探测)|
| **Chess-GPT / Leela** | ✓(棋盘)| ✓(棋子状态、机动性)| ✓(王安全、威胁、Elo)| ✓(**2–7 步前瞻、规划证据 — Jenner 2024**)|
| **CLIP / SigLIP / BLIP** | ✓(颜色、纹理)| ✓(对象、属性)| 部分(单模态绑定 ✓,跨模态空间/否定/计数失败)| 空白(组合性广泛证伪)|
| **ImageBind** | ✓(基础线性探测)| 空白(其他模态)| 空白 | 空白 |
| **LLaVA / 多模态 LLM** | ✓(视觉 token 定位)| ✓(物体)| 部分(视觉-语言对齐瓶颈)| 空白(直觉物理探测失败)|
| **LLM(GPT、LLaMA)** | ✓(数字、token 频率)| ✓(句法、词性)| ✓(空间、时间、ToM、Elo、Othello/棋盘)| 部分(诗歌前瞻规划 — Anthropic 2025);空白(守恒/对称);广泛证伪(反事实、反转)|
| **Coconut / Huginn / CODI** | ✓(正确性信号)| 部分(短跳推理桥接)| 空白(声称的"BFS 叠加"被证伪)| 空白 |
| **OpenVLA(机器人)** | ✓(状态转移)| 部分 | 空白 | 空白 |

**关键观察:** **抽象层(守恒律、对称性、形式化反事实、长程规划)几乎完全是空白的。** 唯一例外是 Leela 国际象棋中的 2–7 步前瞻(Jenner 等 2024)与 Anthropic 在 Claude 3.5 Haiku 中的诗歌押韵规划(Lindsey 等 2025)。这意味着,**"模型是否真正理解"的争论几乎完全建立在中低阶证据之上**——一个深刻的认识论失衡。

---

## 三、区分"理解"与"统计相关"的方法论

**方法论文献揭示了从最弱(原始探测准确率)到最强(基于反事实标注的因果效应)的证据等级。** Belinkov(《Probing Classifiers: Promises, Shortcomings, and Advances》, Computational Linguistics 2022, arXiv:2102.12452)是入口性的综述,提炼出"编码 ≠ 使用"的核心区分。

**控制任务与选择性度量**:Hewitt 与 Liang(EMNLP 2019, arXiv:1909.03368)的《Designing and Interpreting Probes with Control Tasks》将每个语言任务配对一个随机标签的控制任务,定义**选择性 = 真任务准确率 − 控制任务准确率**。他们的关键发现是复杂 MLP 探测会记忆词类型→标签映射,导致真实与随机标签上的高准确率均存在——**单凭高探测准确率不构成表征含结构的证据**。

**信息论探测**:Voita 与 Titov(EMNLP 2020, arXiv:2003.12298)的《MDL Probing》以最小描述长度替代准确率——MDL 同时编码表征质量与解码努力,且**鲁棒地区分预训练 ELMo 与随机初始化 ELMo,而准确率做不到**。Pimentel 等(ACL 2020)用互信息形式化探测。Pimentel 与 Cotterell(EMNLP 2021)的贝叶斯框架进一步算入有限背景知识。

**反事实测试**:Goyal 等(ICML 2019)的《Counterfactual Visual Explanations》。Abraham 等的 CEBaB(NeurIPS 2022, arXiv:2205.14140)用人类撰写的反事实对(餐厅评论沿食物/噪音/氛围/服务四方面修改)估计概念对模型输出的**平均处理效应**——区分因果使用与单纯相关。

**因果干预方法**:Vig 等(NeurIPS 2020)用 Pearl 因果中介分析在 GPT-2 上分解性别偏置的直接/间接效应。Meng 等(ROME, NeurIPS 2022, arXiv:2202.05262)的"因果追踪"逐个恢复隐藏状态以定位事实回忆的中介——中层 MLP 在主语 token 位置;Rank-One Model Editing 改写权重并跨释义泛化(特异性 + 泛化性)。**Geiger 等的因果抽象框架**(NeurIPS 2021, arXiv:2106.02997)与 IIT(ICML 2022)将机制可解释性形式化为通过**互换干预**对齐变量;DAS(CLeaR 2024, arXiv:2303.02536)进一步用梯度下降学习对齐非神经元对齐子空间。Wang 等(IOI 电路, ICLR 2023, arXiv:2211.00593)用激活补丁与路径补丁定位 GPT-2 small 中 ~10 个头实现的间接宾语识别算法——**因果干预揭示了探测无法发现的机制结构**。

**消融性概念擦除**:Elazar 等的**失忆探测**(TACL 2021, arXiv:2006.00995)用 INLP 删除目标属性后测量行为性下降,与随机投影对照——"属性被使用"当且仅当其删除使行为退化超过随机消融。**关键发现:常规探测准确率与下游使用相关性弱。** Ravfogel 等的 INLP(ACL 2020)与 Belrose 等的 LEACE(NeurIPS 2023)提供线性可证擦除。

**组合性与 OOD 测试**:Lake 与 Baroni 的 SCAN(ICML 2018)、Kim 与 Linzen 的 COGS(EMNLP 2020,IID 准确率 96–99% 但泛化仅 16–35%)、McCoy 等的 HANS(ACL 2019)系统性地暴露 NLI 模型的句法启发式快捷方式。Hupkes 等(JAIR 2020)将组合性分解为系统性、生产性、可替换性、局部性、过度泛化五要素。Geirhos 等的《Shortcut Learning in Deep Neural Networks》(Nature Machine Intelligence 2020)是该领域的范式性框架。

**控制实验对照(已训练 vs 随机初始化)**:Zhang 与 Bowman(BlackboxNLP 2018)、Wieting 与 Kiela(ICLR 2019, arXiv:1901.10444)证明**随机投影 / 回声状态网络 / 随机初始化 LSTM** 在 SentEval 上可媲美 InferSent/SkipThought——**没有"随机初始化同架构"基线就不能声称已训练编码器"理解"任何东西**。Ravichander 等(EACL 2021)在 NLI 模型中故意将属性设为常数或随机,探测仍以远超随机的准确率检出该属性——**证明探测准确率与任务使用相关性的根本断裂**。

**重要警告**:Makelov 等(ICLR 2024)、Méloux 等(2025)、Mueller(2024)显示**因果抽象方法也可能在随机初始化模型上产生高于随机的结果**,且多种算法可抽象同一网络——所以这些方法本身也需要控制。Zhang 与 Nanda(ICLR 2024)记录了激活补丁对腐败类型(高斯噪声 vs 对称 token 替换)与度量(logit 差 vs 概率)的敏感性。Heimersheim 与 Nanda(arXiv:2404.15255, 2024)的最佳实践指南区分了**去噪(找 AND 门组件)与加噪(找 OR 门组件)**。

**证据强度等级(从弱到强)**:相关性探测 → 选择性调整探测 → MDL/信息论探测 → 失忆/INLP 行为测试 → 互换干预/因果抽象/IIT → 反事实标注基准(CEBaB 风格)。当代最佳实践组合(i) MDL 或选择性调整探测用于**编码**主张;(ii) 因果干预(失忆、互换、补丁)用于**使用**主张;(iii) OOD/组合性/反事实基准用于"**泛化即理解**"主张。

---

## 四、控制实验的现状

**目前仅 OthelloGPT、V-JEPA、Leela 等少数模型有规范的"理解 vs 不理解"对照。** Garrido 等(2025)在 V-JEPA 直觉物理研究中显式纳入**同架构未训练 V-JEPA** 作为对照——直觉物理性能近随机,证明涌现来自预训练而非架构。Li 等(2023)的 OthelloGPT 论文用**随机初始化 8 层 GPT** 作对照,误差 26.2% vs 已训练 1.7%。Zhang(2026)在 IRIS/DIAMOND 上以**原始像素、随机模型、打乱标签**作三重对照。

**训练良好 vs 训练不足**:McGrath 等(2022)在 AlphaZero 不同训练检查点上探测人类国际象棋概念,发现"概念起飞点"。Chiang 等(EMNLP 2020,《Pretrained Language Model Embryology》)与 Liu 等(NAACL 2019)在 BERT 不同检查点上探测,**句法在早期饱和,语义/任务信息持续累积**——一种"理解轨迹"对照。Molinari(2025)在 OpenVLA 训练检查点上展示世界模型属性的进展。

**已学到 vs 未学到概念的对照**:Ravichander 等(2021)在受控环境中训练 NLI 模型,使目标属性(a) 任务需要、(b) 任务无关、(c) 随机噪声——**探测在 (c) 中仍以远超随机的准确率检出该属性**,证明探测准确率 ≠ 使用。CEBaB 通过对比因果使用与单纯相关的模型完成同样对照。Geiger 等的 IIT 比较以 IIT 训练 vs 多任务 vs 数据增强模型——**只有 IIT 模型可靠地实现目标因果模型**。

**探测复杂度消融**:Hewitt 与 Liang(2019)消融了 MLP 隐藏维度 {0, 10, 100, 1000} 与 5 种正则化方法;Voita 与 Titov(2020)证明 MDL 对探测大小与训练数据比例不敏感而准确率敏感。

**当前重大对照空白**:DreamerV3、Genie、UniSim、Coconut 等模型**完全没有训练良好 vs 训练不足、已训练 vs 随机初始化的标准探测对照**。这是文献中可立即补足的低悬果实。

---

## 五、最有价值的待做探测实验

基于上述空白图谱,以下 5 项实验对回答"模型是否真正理解"问题具有最高边际价值。**核心原则:聚焦抽象概念(守恒、对称、反事实、规划)在尚未被探测的模型家族中的探测**。

**实验 1:V-JEPA 2 中物理守恒律的反事实探测。** Joseph 等(2026)已证明速度与加速度可线性解码,但**没有人测试碰撞中能量(½mv²)、动量(mv)、质量是否守恒**。设计:用 CLEVRER 风格的两体弹性/非弹性碰撞数据,在碰撞前/后帧上分别线性探测每个物体的动量与动能,测试探测向量是否在碰撞前后相等(守恒)。**控制**:同架构随机初始化 V-JEPA、VideoMAE-v2、像素空间基线。**关键判据**:若动量在弹性碰撞中守恒(向量和不变),但 V-JEPA 表征中读出的"动量向量"在碰撞后不守恒,则证明表征是表观运动学的而非动力学的。沿守恒方向的因果干预(强制守恒)是否改变下游 VoE 判断?——这将首次区分"V-JEPA 内含物理引擎"与"V-JEPA 内含运动学统计相关性"。

**实验 2:DreamerV3 的 RSSM 系统性探测。** 当前 DreamerV3 几乎没有正式探测——空白巨大且本应是 World Model 文献的标杆。设计:在 DMC、Atari、Crafter 上分别在 h_t(GRU 状态)与 z_t(分类潜变量)上做线性 / MLP 探测,目标变量包括位置、速度、奖励预测、对象身份、智能体状态。引入 Voita-Titov MDL 探测以区分编码与随机基线。**关键扩展**:用因果追踪(ROME 风格)定位哪些 RSSM 单元因果中介奖励/值的预测;用失忆探测删除位置维度后测量想象推理的退化。**预期影响**:若 RSSM 的 z_t 真为"环境的因子化压缩"则线性可解码;若是分布式压缩则 MDL 显示弱信号——这将决定 DreamerV3 应被理解为"世界模型"还是"无监督的状态压缩器"。

**实验 3:OthelloGPT/Chess-GPT 中战略概念与多步规划的探测。** Du 等(2025)发现 tile stability SAE 特征,但**没有人系统探测机动性、frontier discs、奇偶性、X-square 回避、中心控制、稳定边角**——这些是 Othello 的经典战略概念。设计:为每个概念定义可计算的真值标签(给定棋盘可机械计算),训练线性/MLP 探测并计算 Hewitt-Liang 选择性。**规划扩展**:复制 Jenner 等(2024)的双线性 2 步前瞻探测到 OthelloGPT,测试模型是否表征 1–3 步后的最优着法。**控制**:OthelloGPT(随机走子训练)vs Othello-Synthetic-GPT(冠军级走子训练)——比较两者在战略概念探测上的差距,这是"训练良好 vs 训练不足"的精确对照。**判据**:若仅强模型显示战略表征,则证明这些概念可从棋类经验中涌现;若两者均无,则证明 OthelloGPT 仅学到局部启发式。

**实验 4:Coconut/Huginn 的因果反事实探测——是真潜在推理还是叠加快捷方式?** 跟进 Rizvi-Martel 等(2026)与 Liang-Pan(2026)的负面结果。设计:在多跳问答任务上,**(a) 在每个连续思维位置训练对中间实体/桥接事实的线性探测,确认其编码声称的中间步骤**;(b) **沿探测向量做反事实干预**——将思维 k 修改为对应不同中间事实的方向,测量最终答案是否相应改变;(c) **对照实验:仅插入零向量"占位符思维"或随机思维**,若模型仍能正确回答多跳问题,则证明潜在思维中无关键计算。**关键扩展到 Huginn**:对每次递归迭代做相同测试,确定是否存在迭代级语义阶段(对应人类的"先识别 → 再检索 → 再合成")或仅有不动点收敛(Blayney 等 2026)。**判据**:若反事实干预改变答案且占位符基线失败,则潜在推理是真实的;否则 Coconut/Huginn 是结构性而非语义性的。

**实验 5:跨模型家族的统一抽象概念基准——「守恒/对称/反事实/规划」探测套件。** 当前文献最大的弱点是**抽象概念探测在不同模型家族中使用不同协议**,无法横向比较。提案:构建一个包含四类抽象任务的统一探测基准:(i) **守恒**:在合成碰撞/反弹/相变数据上探测能量/动量/质量的内部表征;(ii) **对称**:测试表征对平移/旋转/反射/时间反演的不变性(via 等变性度量);(iii) **反事实**:用 CEBaB 风格的最小成对反事实数据(物理与文本)测量因果效应;(iv) **规划/前瞻**:复制 Jenner 等 2024 的协议至 Othello、Atari、文本规划任务。应用于:V-JEPA 2、DreamerV3、IRIS、DIAMOND、OthelloGPT、Chess-GPT、Leela、LLaMA-3、GPT-4o、Claude 3.5、Coconut、Huginn。**所有探测使用统一的 MDL 与选择性度量,纳入同架构随机初始化对照**。这将首次产生跨模型家族的、可比较的"抽象理解地图"——直接回答用户提出的根本问题:**哪些模型在哪些抽象层级真正理解?**

---

## 结论:理解的考古学正在加速,但仍处早期

**三年的探测研究已在低中阶建立了相当可靠的证据,但抽象层级的探测才刚刚开始。** 最强结果集中在棋类(Jenner 等 2024 的 Leela 前瞻、Karvonen 2024 的 Elo 探测)与视频物理(Joseph 等 2026 的物理涌现区)——这两个领域已具备因果干预级证据,可声称模型"理解"特定结构。最弱的结果集中在 LLM 的反事实推理(CounterBench)、组合性(Faith and Fate)、世界一致性(Vafa 等 NYC 出租车地图)——这些反复证明大规模训练**不会**自动产生抽象理解。Coconut/Huginn 的争论充分体现了方法论的关键作用:原作者的乐观主张(BFS 叠加)在更严格的第三方探测下系统性地崩塌为"使用快捷方式的不可解释占位符"——**未经独立复现实验,关于"潜在推理"的主张应被视为初步假设而非确证**。

**最具洞察力的方法论遗产是 Hewitt-Liang 选择性、Voita-Titov MDL、Elazar 失忆探测与 Geiger 因果抽象的组合**——它们共同提供了一个跨越"编码 → 使用 → 因果实现"的证据等级,远远优于单一探测准确率。**未来三年最关键的转向**应是:(1) 将这些方法系统应用于尚未被探测的"理解型"模型(DreamerV3、Genie、UniSim);(2) 将抽象概念(守恒、对称、反事实)纳入标准探测套件;(3) 用因果干预证据替代相关性证据。**只有当我们对 100 个不同模型在 10 个抽象概念上有可比较的因果级探测数据时,关于"AI 是否理解"的争论才能从哲学辩论转化为经验科学**。当前的文献离这一目标尚远,但本综述识别出的 5 项高价值实验是通向此目标的具体路径。

