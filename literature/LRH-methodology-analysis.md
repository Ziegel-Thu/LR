# LRH 分支：方法论综合分析

> 5 篇论文的精读 + 跨论文综合。按论文顺序先逐篇报告，再做跨论文分析。

---

## 1. Park et al. 2024 — Causal Inner Product Formalization

### A. 方法

**模型**: LLaMA-2-7B (4096-d, 32k vocab)；附录补充 Gemma-2B。

**核心形式化**:
- 定义 concept variable $W$（latent cause），context $X$，output $Y$。
- **Causally separable** concepts：$Y(W=w, Z=z)$ 对所有 $w,z$ 良定义。
- 证明 unembedding representation 定理：counterfactual 差异 $\gamma(Y(1))-\gamma(Y(0)) \in \text{Cone}(\bar{\gamma}_W)$。
- 证明 measurement 定理：$\text{logit}\; P(Y=Y(1) \mid \lambda) = \alpha\, \lambda^\top \bar{\gamma}_W$。
- 证明 intervention 定理：adding $c\bar{\lambda}_W$ 改变目标概念概率而保持 separable concepts 不变。
- **Causal inner product**：使 causally separable concepts 正交的内积；Riesz 同构连接 $\bar{\gamma}_W \mapsto \bar{\lambda}_W^\top$。
- 实际选择 $M = \text{Cov}(\gamma)^{-1}$。

**数据**: 27 个 binary concepts（22 BATS analogy + 4 language pairs + 1 frequency）。Word pairs 来自 BATS 3.0 + ChatGPT-4 生成。Measurement contexts 来自 Wikipedia 随机句。Intervention contexts 来自 ChatGPT-4 生成（15 个 prompt）。

**实验步骤**:
1. Estimate concept directions：counterfactual word pair differences，leave-one-out 估计。
2. Test LRH：project differences onto concept direction，vs 100K random word-pair differences。
3. Estimate causal inner product：$M = \text{Cov}(\gamma)^{-1}$，比较 causal vs Euclidean。
4. Test measurement：French→Spanish concept，Wikipedia contexts，检验 $\bar{\gamma}_W^\top \lambda(x)$ 是否分离。
5. Test intervention：$\bar{\lambda}_W = \text{Cov}(\gamma)^{-1} \bar{\gamma}_W$，$\alpha \in \{0, 0.1, 0.2, 0.3, 0.4\}$。

### B. 关键结果

- Counterfactual pair projections 对所有 concepts（除 thing→part）呈 **strong right skew** vs random pairs。
- Causal inner product 使 causally separable concepts 近正交；Euclidean 在 LLaMA-2 "somewhat works"，在 Gemma-2B 失败更明显。
- Intervention "Long live the …"：$\alpha=0$ → top-1 king；$\alpha=0.2$ → top-1 queen。
- **没有聚合定量指标**（无 AUC/accuracy 表），主要靠 histogram/heatmap。

### C. 明确未做

- $D$ 的自由度（$d$ 个参数）无原则选择方法。
- 不涉及 intermediate layers / model internals 的可解释性。
- 直接找 embedding representation pairs 在实践中困难，故回避。

### D. 方法论缺口

- 无 aggregate quantitative metrics（只有 histograms）。
- 无 $D$ 选择的 ablation。
- 无 layerwise analysis。
- 无正式统计检验（只有 visual inspection）。
- 仅 2 个模型（LLaMA-2-7B + brief Gemma-2B）。

---

## 2. Engels et al. 2024 — Circular (Multi-dimensional) Features

### A. 方法

**模型**: GPT-2, Mistral 7B, Llama 3 8B。

**核心概念**:
- Feature = function from input subset to $\mathbb{R}^{d_f}$。
- **Irreducible** feature：经 affine 变换后不能分解为独立子特征。
- **Separability index** $S(f) = \min I(a;b)$（grid 1000 rotation angles, 40×40 bins, compute MI）。
- **$\varepsilon$-mixture index** $M_\varepsilon(f)$（gradient descent, 10k steps, lr 0.1）。

**Feature discovery pipeline**:
1. 训练/使用 SAE（sparse autoencoder），dictionary learning loss with $L_p$ sparsity ($p=1/2$)。
   - Mistral SAE: layers 8/16/24, 16× expansion = 65,536 elements, >1B tokens from Pile+Alpaca。
2. Cluster SAE dictionary elements by cosine similarity。
   - GPT-2: spectral clustering 25k features → 1000 clusters, manually inspect ~500。
   - Mistral: graph clustering k=2 NN, prune cosine < 0.5, connected components。
3. For each cluster: reconstruct activations using only cluster features → PCA → test 2D structure。

**Circular tasks**: Weekdays ("Two days from Monday is …", 49 prompts) + Months (144 prompts)。

**Intervention**:
- PCA top-5 PCs on hidden states。
- Linear probe from PCs → unit circle encoding。
- Intervene by swapping circular subspace + average-ablating remainder。
- Baselines: patch top-5 PCs, patch entire layer, replace with task mean。
- Metric: average logit difference (correct vs target token)。
- Off-distribution circle sweep: $r \in [0,2]$, $\theta \in [0, 198\pi/100]$。

### B. 关键结果

- Task accuracy:
  - Llama 3 8B: **29/49** weekdays, **143/144** months。
  - Mistral 7B: **31/49** weekdays, **125/144** months。
  - GPT-2: **8/49** weekdays, **10/144** months（但 circular representations 仍存在）。
- Circular subspace intervention 近似 full-layer patching，优于 top-5 PC patching。
- SAE probe vs PCA probe (layer 8): **-2.58 → -2.01** avg logit diff。
- Attention head analysis (Llama weekdays): top head (L17,H0) EVR $R^2$ = **0.98/0.99**。
- GPT-2 clusters 中 weekdays 排名 **9/1000**（by reducibility metric）。

### C. 明确未做

- 未找到除 weekdays/months 外的 interpretable multi-dimensional features。
- 仅手动检查 ~500/1000 GPT-2 clusters。
- 未比较更丰富的 clustering 方案。
- 未使用 intervention-based irreducibility 定义。

### D. 方法论缺口

- 无发现方法的 precision/recall 定量评估。
- 无 SAE size / sparsity / layer 的系统 ablation。
- 无 >2D 结构的搜索。
- 无与 manifold learning methods (UMAP, t-SNE, etc.) 的比较。
- 无广泛 model scaling study。

---

## 3. Tan et al. 2024 — OOD Failure of Steering Vectors

### A. 方法

**模型**: Llama-2-7b-Chat, Qwen-1.5-14b-Chat（主要）；Llama-2-70B, Gemma-2-2B-IT（补充）。

**数据**: Model-Written Evaluations (MWE) 1000 samples/类, 100+ 类, split 40-10-50。加 TruthfulQA + sycophancy + AI risk, 共 **40 datasets**。

**Steering vector 提取**: Contrastive Activation Addition (CAA)。
$$v_{MD} = \frac{1}{|\mathcal{D}|} \sum [a_L(x, y_+) - a_L(x, y_-)]$$
在 option token 位置取 residual stream activation。

**Application**: 在 last token 位置加 $\lambda \cdot v_L$, $\lambda \in \{-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5\}$。

**OOD 设置**: 向 prompt/system message 注入额外文本 (BASE → USER_POS/NEG, SYS_POS/NEG)。

**核心 metrics**:
- **Logit-difference propensity**: $m_{LD} = \text{Logit}(y_+) - \text{Logit}(y_-)$。
- **Steerability**: slope of mean LD vs $\lambda$ 的 OLS 拟合。
- **Relative steerability**: $s_{rel}(v_A, \mathcal{D}_B) = s(v_A, \mathcal{D}_B) / s(v_B, \mathcal{D}_B)$。

**Optimal layers**: Llama layer 13, Qwen layer 21, Llama-70B layer 30, Gemma-2B layer 14。

### B. 关键结果

- 部分 dataset **接近一半样本** anti-steerable（与预期方向相反）。
- OOD vs ID steerability 相关性:
  - Llama: $\rho = 0.891$
  - Qwen: $\rho = 0.694$
- OOD vs ID steerability variance:
  - Llama: $\rho = 0.535$
  - Qwen: $\rho = 0.341$
- Cross-model steerability:
  - ID: $\rho = 0.769$
  - OOD: $\rho = 0.586$
- Propensity similarity vs relative steerability:
  - Llama: $\rho = -0.26$
  - Qwen: $\rho = -0.46$
- SV cosine similarity vs unsteered logit-diff:
  - Llama: $\rho = -0.63$
  - Qwen: $\rho = -0.86$

### C. 明确未做

- 仅 MCQ 格式；open-ended generation 评估未做。
- 未解释 steerability bias 的成因。
- 仅 2 个主模型。
- 未做 mitigation experiments。

### D. 方法论缺口

- 无 generative/judge-based metrics。
- 无 cross-concept steering tradeoff analysis。
- 无 causal dissection of WHY specific datasets are more steerable。
- 仅 correlation，无 causal explanation。

---

## 4. Zou et al. 2023 — Representation Engineering (RepE)

### A. 方法

**模型**: LLaMA-2 (7B/13B/70B, base + Chat), Vicuna-13B, CLIP (ViT-base-patch32), DeBERTa-xlarge。

**核心框架: Linear Artificial Tomography (LAT)**:
1. Design stimulus（paired experimental vs reference prompts, or self-generated stimuli）。
2. Collect neural activity（decoder last token / encoder concept token / response tokens for functions）。
3. Fit linear model（PCA on pairwise difference vectors; first PC = reading vector）。

**概念范围**: truthfulness, utility, probability, morality, emotion, power-seeking, memorization, harmless instruction-following, fairness。

**四类实验**: Correlation（classification accuracy）, Manipulation（add/subtract directions）, Termination（projection removal）, Recovery（remove then reintroduce）。

**Reading**: dot product with hidden state。
**Control methods**: ActAdd, Reading Vector, Contrast Vector (>3× inference cost), LoRRA。

### B. 关键结果

- TruthfulQA MC1:
  - LLaMA-2-7B: standard **31.0** → LAT best **58.9** (+27.9 pp)
  - LLaMA-2-70B: standard **29.9** → LAT best **69.8** (+39.9 pp)
- QA benchmarks（OBQA/CSQA/ARC-e/ARC-c/RACE）LAT 均优于 few-shot。
- CCS comparison: avg CCS **69** vs LAT **82**。
- Concept reading accuracies: Morality **85.0**, Probability **92.6**, Risk **90.7**, Utility **81.0**。
- CLIP emotion: Happiness **74.2**, Sadness **61.7**, Anger **72.7**。
- Jailbreak defense: Piece-wise operator: **93.8 / 90.2 / 87.2** (Prompt-only / Manual jailbreak / GCG)。
- Memorization control: negative direction EM from **89.3 → 37.9**。
- Sarcoidosis fairness: Female mentions **97% → 55%**, Black female **60% → 13%**。

### C. 明确未做

- 默认无标签（unsupervised LAT），未充分探索 supervised variants。
- 承认 random stimuli 可能 latch onto superficial features。
- 未做 dishonest behavior 的细粒度评估。
- 未分析 representation trajectories / manifolds / state-spaces。

### D. 方法论缺口

- Termination + Recovery 未在所有 concepts 上系统执行。
- 模型覆盖以 LLaMA-2 系列为主。
- 无 adversarial prompt 的大规模鲁棒性测试。
- 无 dynamics / temporal representation 分析。

---

## 5. Nanda et al. 2023 — OthelloGPT Linear World Model

### A. 方法

**模型**: OthelloGPT — 8-layer GPT, 8 attention heads, 512-d hidden space。Synthetic model weights from Li et al. 2023。

**数据**: 3.5M game sequences train, 512 games validation, 1000 games test。

**核心发现**: Board state 用 {Mine, Yours, Empty}（相对于当前玩家）编码时线性可恢复，而 {Black, White, Empty} 编码需要非线性 probe。

**Probes**:
- Linear: $p^\lambda(x_t^l) = \text{softmax}(W x_t^l)$, $W \in \mathbb{R}^{D \times 3}$, predict all 64 tiles。
- Non-linear: $p^\nu(x_t^l) = \text{softmax}(W_1 \text{ReLU}(W_2 x_t^l))$。
- Training: AdamW, lr=1e-2, wd=1e-2, early stopping patience 10, converge ~100k samples。

**Intervention**: Adding probe vector to residual stream:
$$x' \leftarrow x + \alpha\, p^\lambda_d(x)$$
（非 gradient descent editing）。

### B. 关键结果

- Linear probe accuracy {Mine, Yours, Empty}:
  - Layer 0→7: **90.9, 94.8, 97.2, 98.3, 99.0, 99.4, 99.6, 99.5**
- Linear probe {Black, White, Empty}:
  - Layer 0→7: **62.2, 74.8, 74.9, 75.0, 75.0, 74.9, 74.8, 74.4**
- Non-linear probe {Black, White, Empty}:
  - Layer 0→7: **63.4, 88.6, 93.3, 96.3, 97.5, 98.3, 98.7, 98.3**
- Unembedding top-1 error: Layer 7 = **0.004**, Layer 8 = **0.001**。
- Probabilistic baseline: **61.8**。

### C. 明确未做

- Probing 是 correlational 非 causal（明确声明）。
- 未声称结果推广到 larger LMs（仅 "anticipate"）。
- Endgame behavior 不明确（alternative circuits 还是 struggle）。
- Layer intervention 的 mechanism 仅假设（Hydra effect），未解决。

### D. 方法论缺口

- 仅 1 个 synthetic model checkpoint，无 cross-model validation。
- 未系统搜索 alternative linear bases（仅 Mine/Yours vs Black/White）。
- 无 probe capacity ablation（depth/width sweep）。
- 无 OOD board state benchmark。
- 无 statistical significance / confidence intervals。
- 无 mechanistic circuit decomposition。

---

# 跨论文综合分析

## E. Cross-paper Gaps

### E1. Park 的 causal inner product + Nanda 的 OthelloGPT
Park 的理论可直接应用于 OthelloGPT：用 causal inner product 检验 {Mine, Yours, Empty} 方向是否在 causal sense 正交。Nanda 只做了 probing accuracy，未做 causal formalization。**这是一个自然且可执行的实验**：估计 OthelloGPT unembedding 空间的 $\text{Cov}(\gamma)^{-1}$，检验 concept directions 是否在 causal inner product 下正交。

### E2. Engels 的 circular features + Park 的 causal formalization
Park 的理论假设 1D binary concepts。Engels 发现 2D circular features。**Park 的框架能否推广到 multi-dimensional concepts？** Causal inner product 在 irreducible 2D features 上的行为是什么？如果 weekday 是 2D，那么 Park 的 orthogonality 测试在投影到 1D 时会怎样？**无人研究**。

### E3. Tan 的 OOD failure + Zou 的 RepE control
Zou 声称 RepE 可以做 jailbreak defense、bias control 等。Tan 发现 steering vectors 在 OOD 下严重退化。**Zou 的所有 control 结果是否在 OOD 下复现？** 特别是 jailbreak defense（Zou Table 中 GCG 87.2%）在 distribution shift 下是否仍然有效？**无人检验**。

### E4. Engels 的 circular features + Tan 的 steering 失败
如果某些概念本质上是 multi-dimensional（circular），那么 1D steering vector 必然失败。**Tan 观察到的 anti-steerable samples 是否因为目标概念本身是 multi-dimensional？** 将 Engels 的 irreducibility test 应用于 Tan 的 40 个 datasets，可以区分 "steering 失败因为 concept 不是线性的" vs "steering 失败因为 OOD"。**无人做此诊断**。

### E5. Nanda 的 relative encoding + Zou 的 RepE stimuli
Nanda 发现 {Mine, Yours, Empty}（相对编码）远优于 {Black, White, Empty}（绝对编码）。**Zou 的 LAT 是否也应使用 context-relative stimuli 而非 absolute stimuli？** 例如，truthfulness 的 "true vs false" 可能不如 "more true than X vs less true than X" 的 relative framing。**无人系统比较 absolute vs relative stimuli design for RepE**。

### E6. Park 的 causal inner product + Zou 的 multiple reading methods
Park 理论预测 reading vector 应该是 $\bar{\gamma}_W$（在 causal inner product 下）。Zou 比较了 PCA/K-Means/Mean Difference/Logistic Regression 但无理论指导选择。**用 Park 的 causal inner product 能否统一并改进 Zou 的 reading vector estimation？**

## F. Contradictions Between Papers

### F1. 线性 vs 非线性：根本张力
- **Park**: 证明（在假设下）概念在 unembedding/embedding 空间中必然线性表征。
- **Engels**: 发现 weekdays/months 是 **irreducible 2D circular** features，不能降为 1D 线性方向。
- **Nanda**: 发现 {Black, White, Empty} 需要非线性 probe（74.8% linear vs 98.7% nonlinear），但换编码后（{Mine, Yours, Empty}）线性就够了（99.6%）。

**解读**: 表面矛盾。Park 的理论仅针对 binary concepts 的 unembedding 空间，不直接适用于 cyclic/multi-valued concepts。Nanda 提示"线性"可能取决于 encoding choice。Engels 提示某些 concepts 本质上是 multi-dimensional，不能 reduce 到 binary。三者合起来说明：**LRH 的成立范围严格依赖于 concept 的数学结构和 representation basis 的选择**。

### F2. Steering 的可靠性
- **Zou**: RepE control "works"——jailbreak defense 87.2%，memorization control EM 89.3→37.9，fairness 97%→55%。
- **Tan**: Steering vectors 在 OOD 下严重退化（cross-model OOD $\rho=0.586$），且 ~50% samples anti-steerable。

**解读**: 不完全矛盾——Zou 主要在 ID 设置下评估，Tan 专门研究 OOD。但 Zou 的 jailbreak 实验（manual jailbreak = distribution shift）声称有效，而 Tan 的结论暗示这类泛化不可靠。**Zou 的 jailbreak 结果可能在更严格的 OOD 测试下退化**。

### F3. Probe 的 causal 地位
- **Park**: 建立了严格的 causal framework，区分 measurement 和 intervention。
- **Nanda**: 明确承认 "probing is correlational, not causal"。
- **Zou**: 做了 Correlation + Manipulation + Termination + Recovery 四类实验，但 causal claims 未形式化。

**解读**: Park 提供了理论工具来弥合这个 gap，但只在 word-level unembedding 空间验证过。**应用 Park 的 causal framework 来评估 Nanda 和 Zou 的 probes 是否满足 causal criteria** 是一个明显的 next step。

## G. "Everyone Does X but Nobody Does Y" Patterns

### G1. Everyone 做 single layer，nobody 做 cross-layer dynamics
- Park: 只看 final unembedding space。
- Engels: per-layer PCA，但不分析 layer 间 feature 的 evolution。
- Tan: 固定 single optimal layer (13/21/30/14)。
- Zou: 也是 per-layer analysis。
- Nanda: per-layer probe accuracy，但无 cross-layer circuit analysis。

**Nobody** 研究 concept direction 如何在 layers 之间 transform、哪些 attention heads / MLPs 创建或修改这些方向。

### G2. Everyone 用 mean difference，nobody 验证这个选择
- Park: $\bar{\gamma}_W = \text{mean}(\gamma(y(1)) - \gamma(y(0)))$。
- Tan: $v_{MD} = \text{mean}(a_L(x,y_+) - a_L(x,y_-))$。
- Zou: PCA on pairwise differences（本质也是 mean-centered）。
- Nanda: linear probe weights（learned version of mean difference）。

**Nobody** 系统比较 mean difference vs median difference vs robust estimators vs learned directions。特别是当 distribution 有 heavy tails 或 outliers 时，mean 可能不是最优选择。Tan 发现 ~50% anti-steerable samples 直接暗示 mean 可能被 bimodal distribution 污染。

### G3. Everyone 用 binary/paired stimuli，nobody 考虑连续概念
- Park: binary concepts (verb→3pSg, male→female)。
- Tan: binary MCQ (A/B, Yes/No)。
- Zou: paired experimental vs reference prompts。
- Nanda: ternary (Mine/Yours/Empty)。
- Engels: 虽然发现了连续的 circular structure，但 tasks 仍然是 discrete (7 weekdays, 12 months)。

**Nobody** 系统研究 continuous-valued concepts（如 temperature, probability, degree of truthfulness）的表征几何。Park 的理论是否推广到连续变量？

### G4. Everyone claim "linear representation"，nobody 量化 "how linear"
- Park: histograms showing right skew（无 R² / effect size）。
- Nanda: probe accuracy 作为 proxy（但高 accuracy 不意味着 representation 本身是线性的——可能是 probe 的 generalization）。
- Zou: classification accuracy。
- Engels: 用 irreducibility metrics，但这测的是 dimensionality，不是 linearity。
- Tan: 用 steerability slope，但这测的是 intervention effect，不是 representation linearity。

**Nobody** 提供一个统一的 "linearity score"，量化 representation 偏离 linear subspace 的程度。例如：nonlinear probe accuracy - linear probe accuracy 作为 nonlinearity gap（Nanda 做了但没 formalize）。

### G5. Everyone 假设 concepts 是 independent，nobody 研究 concept interactions
- Park: causally separable = independent in causal sense。
- Zou: 逐个 concept 提取 reading vector。
- Tan: 逐个 dataset 提取 steering vector。
- Engels: 分析 individual clusters。

**Nobody** 系统研究 concept 之间的 interaction：两个 concepts 的方向 interact 时会发生什么？Steering 一个 concept 对另一个的影响？Park 的 orthogonality 测试 touch on this，但只在 measurement 层面，未在 intervention 层面验证。

### G6. Everyone 用固定模型权重，nobody 研究 training dynamics
所有 5 篇论文都分析 fixed pretrained models。**Nobody** 研究 linear representations 何时在 training 中出现、是否在 fine-tuning 中改变、不同 training objectives 如何影响 representation geometry。

### G7. Statistical rigor 普遍不足
- Park: 无 formal statistical tests，只有 visual inspection of histograms。
- Nanda: point estimates only，无 confidence intervals / bootstrap。
- Engels: 手动检查 clusters，无 precision/recall。
- Zou: 报告单次结果，无 variance across runs。
- Tan: 做了 correlation $\rho$ 但无 significance tests (p-values)。

**Nobody** 做 proper statistical validation with multiple runs, confidence intervals, significance tests, or correction for multiple comparisons。

---

## 总结表

| 维度 | Park 2024 | Engels 2024 | Tan 2024 | Zou 2023 | Nanda 2023 |
|------|-----------|-------------|----------|----------|------------|
| 理论贡献 | ★★★ causal formalization | ★★ irreducibility definition | ★ empirical framework | ★ LAT framework | ★ relative encoding insight |
| 模型数量 | 2 (LLaMA-7B, Gemma-2B) | 3 (GPT-2, Mistral-7B, Llama3-8B) | 4 (Llama-7B/70B, Qwen-14B, Gemma-2B) | 5+ (LLaMA-2 family, Vicuna, CLIP, DeBERTa) | 1 (OthelloGPT) |
| 概念数量 | 27 binary | 2 circular (weekdays, months) | 40 behavioral | 10+ (truth, emotion, safety...) | 1 (board state) |
| Quantitative rigor | Low (no aggregate metrics) | Medium (accuracy counts) | Medium (correlations) | Medium (accuracy tables) | Medium (accuracy tables) |
| Causal validation | Theoretical + partial | Intervention-based | Correlation only | Partial (4 experiment types) | Partial (intervention) |
| OOD 测试 | None | Off-distribution circle sweep | Core contribution | None | None |
| Statistical tests | None | None | Correlation $\rho$ | None | None |
