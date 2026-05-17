### Representation learning research mapping

用户:


Claude:
his is an excellent and very detailed research request — the scope and format are crystal clear. Let me dive into a comprehensive research process to map out the key papers across all 10 branches of representation learning and analysis.
our map of representation learning and representation analysis in neural networks is ready. It traces ten interconnected branches—similarity metrics, sparse autoencoders, causal abstraction/mechanistic interpretability, the linear representation hypothesis, probing and lens methods, information-theoretic methods, topological/geometric approaches, disentanglement and identifiability, scaling/convergence, and impossibility results—covering foundational papers, key developments, 2024–2026 frontier work, and status assessments for each. The report concludes with a cross-branch synthesis identifying the fastest-moving areas (SAEs, mechanistic interpretability), under-explored intersections, and five overarching open questions defining the field's agenda for the rest of the decade.
# A Paper Map of Representation Learning and Representation Analysis in Neural Networks

This map traces ten interconnected branches of work on how neural networks represent information and how researchers study, measure, and intervene on those representations. For each branch it lists foundational papers, key intermediate developments, the most important 2024–2026 frontier work, and a short status assessment.

---

### 1. Representation Similarity Metrics (表征相似度度量)

**Foundational Papers (开创性论文)**
- Kriegeskorte, Mur & Bandettini (2008) — Introduced Representational Similarity Analysis (RSA), comparing systems by the structure of their pairwise (dis)similarity matrices rather than by aligning units. [Frontiers in Systems Neuroscience]
- Raghu, Gilmer, Yosinski & Sohl-Dickstein (2017) — SVCCA: combines SVD with canonical correlation analysis to compare distributed representations across layers and training. [NeurIPS 2017; arXiv:1706.05806]
- Morcos, Raghu & Bengio (2018) — Projection-weighted CCA (PWCCA), a more robust CCA-based similarity measure that down-weights noisy directions. [NeurIPS 2018; arXiv:1806.05759]
- Kornblith, Norouzi, Lee & Hinton (2019) — Showed CCA-like measures invariant to invertible linear maps cannot distinguish meaningful similarity in high-dim, and proposed Centered Kernel Alignment (CKA) as the field-standard remedy. [ICML 2019; arXiv:1905.00414]

**Key Developments (关键进展)**
- Williams, Kunz, Kornblith & Linderman (2021) — Generalized shape metrics, casting representation similarity as a proper metric space defined by group invariances (rotations, permutations). [NeurIPS 2021; arXiv:2110.14739]
- Ding, Denain & Steinhardt (2021) — Showed CKA can disagree with downstream functional similarity and proposed sensitivity/specificity tests for similarity measures. [arXiv:2108.01661]
- Davari, Horoi, Natik, Lajoie, Wolf & Belilovsky (2022) — "Reliability of CKA as a Similarity Measure in Deep Learning": stress-tested CKA's sensitivity to outliers and scaling.
- Klabunde, Schumacher, Strohmaier & Lemmerich (2023) — Comprehensive survey of 30+ neural-network similarity measures. [arXiv:2305.06329]
- Huh, Cheung, Wang & Isola (2024) — Used mutual k-nearest-neighbour as a lightweight similarity index that scales well to cross-modal comparisons (see Branch 9).

**Latest Frontier (2024–2026) (最新前沿)**
- Harvey, Lipshutz & Williams (2024) — Connected linear CKA to the average alignment of optimal linear decoders, tightly linking representational geometry to functional alignability. [arXiv:2411.08197]
- Cloos, Yamins, Williams et al. (2024) — Showed that CKA, RSA, Procrustes and mutual-kNN can rank models in mutually inconsistent orders on identical computations, calling for axiom-based metric selection.
- Lange, Kriegeskorte et al. (2025) — Estimating neural representation alignment from sparsely sampled inputs and features, addressing bias in high-dim/small-N regimes. [arXiv:2502.15104]
- Insulla, Lipshutz, Williams et al. (2025) — Stochastic/Bures-style shape metrics for stochastic neural representations.
- Khosla & Williams (2024–2025) — Soft-matching / soft-Procrustes for permutation-invariant comparison without forcing unit-level alignment.

**Current Status (当前状态)**: Representation-similarity metrics are the workhorse of cross-model and brain–model analysis, but the field has moved from "use CKA" to "metric choice is a hypothesis." Each metric encodes specific invariances (orthogonal, affine, permutation, isotropic-scaling). The main open questions are: (i) which invariances are warranted for which scientific question; (ii) how to debias metrics in the high-dimensional, small-sample regime typical of modern frontier models; (iii) whether functional/behavioral similarity should be the ground truth that geometric similarity is judged against; and (iv) developing metrics that respect stochasticity, noise, and online updates rather than treating representations as fixed point clouds.

---

### 2. Sparse Autoencoders (SAE) and Dictionary Learning

**Foundational Papers (开创性论文)**
- Olshausen & Field (1996/1997) — Original sparse coding of natural image patches; the conceptual ancestor of every SAE used today. [Nature 381; Vision Research]
- Elhage, Hume, Olsson et al. (2022) — "Toy Models of Superposition": formalized the hypothesis that networks pack more features than neurons by representing them as overcomplete directions. [Anthropic / arXiv:2209.10652]
- Sharkey, Braun & Millidge (2022) — "Taking features out of superposition with sparse autoencoders" interim report demonstrating the SAE recipe. [AI Alignment Forum]
- Bricken, Templeton, Batson et al. (2023) — "Towards Monosemanticity": Anthropic's landmark demonstration that SAEs on a 1-layer transformer recover thousands of monosemantic features. [Transformer Circuits Thread]
- Cunningham, Ewart, Riggs, Huben & Sharkey (2023) — Independently showed SAEs on real LLMs (Pythia) find interpretable features and enable finer-grained causal interventions. [arXiv:2309.08600]

**Key Developments (关键进展)**
- Templeton, Conerly, Marcus et al. (2024) — "Scaling Monosemanticity": SAEs extracted ~34M features from Claude 3 Sonnet, including safety-relevant features (deception, sycophancy). [Anthropic]
- Gao, la Tour, Tillman et al. (OpenAI, 2024) — TopK SAEs with clean scaling laws relating dictionary size, model size and reconstruction. [arXiv:2406.04093]
- Rajamanoharan, Conmy, Smith et al. (DeepMind, 2024) — JumpReLU SAEs improving the sparsity–fidelity Pareto frontier. [arXiv:2407.14435]
- Lieberum, Rajamanoharan, Conmy et al. (DeepMind, 2024) — "Gemma Scope": open-sourced SAEs on every layer of Gemma-2, the field's default research substrate. [arXiv:2408.05147]
- Dunefsky, Chlenski & Nanda (2024) — Transcoders: SAEs that learn input→output feature dictionaries for MLPs, enabling sparse-circuit analysis. [arXiv:2406.11944]
- Lindsey, Templeton, Marcus et al. (Anthropic, 2024) — Sparse Crosscoders: dictionaries shared across layers and across model checkpoints, supporting cross-model and developmental analysis.
- Chanin, Wilken-Smith, Dulka, Bhatnagar & Bloom (2024) — Identified "feature absorption" pathology where general SAE features get absorbed by more specific co-occurring ones. [arXiv:2409.14507]

**Latest Frontier (2024–2026) (最新前沿)**
- Bussmann, Leask & Nanda (2024) — Matryoshka SAEs: nested dictionaries mitigating feature splitting and absorption via a hierarchy of sparsity levels. [arXiv:2503.17547]
- Paulo, Mallen, Juang & Belrose (2024) — "Automatically Interpreting Millions of Features": pipelines for auto-labeling SAE features at scale. [arXiv:2410.13928]
- Anthropic Circuit-Tracing team — Lindsey, Gurnee, Ameisen et al. (2025) — "On the Biology of a Large Language Model": replacement-model + attribution-graph methodology using crosscoders in Claude 3.5 Haiku.
- Karvonen, Wright et al. (2024–2025) — SAEBench: standardized benchmarks for SAE quality (unlearning, probing, sparse probing).
- Chaudhary, Geiger et al. (2024) — Evaluating Open-Source SAEs on RAVEL: off-the-shelf SAEs underperform DAS at fine-grained causal manipulation, exposing limits of dictionary learning. [arXiv:2405.05821]
- Engels, Liao, Michaud, Gurnee & Tegmark (2024) — Discovered multi-dimensional, circular features (days of week, months) that are not captured by single linear SAE directions, challenging the strictly atomic linear-feature view. [arXiv:2405.14860]
- Mudide, Engels et al. (2024) — Switch SAEs and mixture-of-experts dictionaries to scale to billions of features.
- Smith, Wright, Heimersheim et al. (2025) — "Crosscoders for understanding model diffs" applied to fine-tuning and alignment training.

**Current Status (当前状态)**: SAEs are the fastest-moving subfield of interpretability and the de-facto bridge between superposition theory and practical mechanistic analysis. Architectural innovation (TopK, JumpReLU, Matryoshka, switch/MoE SAEs) has roughly tripled reconstruction quality at fixed sparsity since 2023, and open SAE suites (Gemma Scope, Llama Scope) enable a shared scientific substrate. Open questions: (i) **feature absorption, splitting and dead features** suggest dictionaries are not converging to a canonical "true" feature set; (ii) recent results show many concepts are inherently multi-dimensional or curved, challenging the strict linear-feature view; (iii) SAE features often **underperform supervised probes and DAS** on causal-control benchmarks, raising whether they capture computation or just correlated geometry; (iv) the field lacks agreement on **evaluation**—benchmarks like SAEBench, RAVEL and feature-circuit benchmarks disagree on rankings; (v) connecting SAE features into **circuits/attribution graphs** (transcoders, crosscoders, replacement models) is the current frontier.

---

### 3. Causal Abstraction and Mechanistic Interpretability (因果抽象与机械可解释性)

**Foundational Papers (开创性论文)**
- Olah, Mordvintsev, Schubert et al. (2017–2020) — The "Circuits" thread on InceptionV1, establishing the goal of reverse-engineering computation feature-by-feature. [Distill]
- Vig, Gehrmann, Belinkov et al. (2020) — Causal mediation analysis applied to transformer attention for gender bias. [NeurIPS 2020]
- Geiger, Lu, Icard & Potts (2021) — "Causal Abstractions of Neural Networks": formalized interchange interventions as the test for whether a NN implements a high-level causal model. [NeurIPS 2021; arXiv:2106.02997]
- Elhage, Nanda, Olsson et al. (2021) — "A Mathematical Framework for Transformer Circuits": introduced QK/OV decomposition, induction heads, and the residual-stream view. [Transformer Circuits]

**Key Developments (关键进展)**
- Meng, Bau, Andonian & Belinkov (2022) — ROME: causal tracing localizes factual associations to mid-layer MLPs and edits them via rank-1 updates. [NeurIPS 2022; arXiv:2202.05262]
- Wang, Variengien, Conmy, Shlegeris & Steinhardt (2022) — The IOI circuit in GPT-2 small: the canonical worked example of end-to-end circuit discovery. [ICLR 2023; arXiv:2211.00593]
- Chan, Garriga-Alonso et al. (2022) — Causal scrubbing: a formal procedure for evaluating whether a hypothesized circuit fully accounts for a behaviour. [Redwood / AF post]
- Geiger, Wu, Potts, Icard & Goodman (2023) — Distributed Alignment Search (DAS): learn (via gradient descent) the subspace in which a network aligns with a symbolic causal variable. [arXiv:2303.02536]
- Goldowsky-Dill, MacLeod, Sato & Arora (2023) — Formalized path patching for circuit identification. [arXiv:2304.05969]
- Conmy, Mavor-Parker, Lynch, Heimersheim & Garriga-Alonso (2023) — ACDC: automated circuit discovery by iterative edge ablation. [NeurIPS 2023; arXiv:2304.14997]
- Zhang & Nanda (2024) — "Best Practices for Activation Patching": clarified denoising-vs-noising and metric choices. [ICLR 2024; arXiv:2309.16042]

**Latest Frontier (2024–2026) (最新前沿)**
- Geiger, Ibeling, Zur, Chaudhary, Huang, Wu, Arora, Goodman, Potts & Icard (2024/2025) — "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability"—unifies activation patching, path patching, DAS, SAEs, concept erasure and steering under causal abstraction. [JMLR 26(83); arXiv:2301.04709v4]
- Marks, Rager, Michaud, Belinkov, Bau & Mueller (2024) — "Sparse Feature Circuits": circuits over SAE features for spurious-correlation removal. [ICLR 2025; arXiv:2403.19647]
- Huang, Geiger, D'Oosterlinck et al. (2024) — RAVEL benchmark for testing whether interpretability methods isolate the right causal variable. [arXiv:2402.17700]
- Ferrando, Sarti, Bisazza & Costa-jussà (2024) — Survey "I/O Faithful Interpretability for Transformer-Based LLMs". [arXiv:2405.00208]
- Anthropic Circuit Tracing (Lindsey et al., 2025) — Replacement-model methodology producing attribution graphs for multi-step reasoning, multilingual circuits and planning behaviour in Claude 3.5 Haiku.
- Mueller, Brinkmann et al. (2025) — "Combining Causal Models for More Accurate Abstractions of Neural Networks". [arXiv:2503.11429]
- Sun, Geiger, Wu et al. (2025) — "Addressing Divergent Representations from Causal Interventions" tackles the off-distribution problem of activation patching.

**Current Status (当前状态)**: Mechanistic interpretability is now a recognised subfield with a unified theoretical foundation (causal abstraction) and a coherent toolkit (interchange interventions, activation/path patching, DAS, causal scrubbing, ACDC, sparse feature circuits). Anthropic's circuit-tracing program produces semi-automated attribution graphs on frontier models. Open questions: (i) **off-distribution interventions**—patching often produces activations the model would never naturally produce; (ii) **scalability**—circuits are still mostly identified on small tasks (IOI, greater-than) and don't yet scale to chain-of-thought or agentic behaviour; (iii) **faithfulness vs. completeness**—causal scrubbing reveals most discovered circuits explain only a fraction of behaviour; (iv) **integration with SAEs**—how to turn SAE feature graphs into truly causal explanations; and (v) the **right level of abstraction**—heads, MLPs, features, or yet-undiscovered units.

---

### 4. Linear Representation Hypothesis (线性表征假说)

**Foundational Papers (开创性论文)**
- Mikolov, Yih & Zweig (2013) — word2vec analogies ("king − man + woman ≈ queen") were the original empirical evidence that concepts live as linear directions. [NAACL 2013]
- Bolukbasi, Chang, Zou, Saligrama & Kalai (2016) — Showed gender bias in word embeddings can be isolated and removed via a linear subspace. [NeurIPS 2016]
- Alain & Bengio (2016) — Linear probes as a canonical lightweight readout from hidden layers. [arXiv:1610.01644]
- Elhage, Hume, Olsson et al. (2022) — Toy Models of Superposition gave the modern theoretical articulation: features as (potentially overcomplete) linear directions. [arXiv:2209.10652]

**Key Developments (关键进展)**
- Ravfogel, Elazar, Gonen, Twiton & Goldberg (2020) — Iterative Null-space Projection (INLP) for linearly removing protected attributes. [ACL 2020]
- Burns, Ye, Klein & Steinhardt (2022) — CCS: discovers "truth" directions in LLMs without labels via consistency under negation. [arXiv:2212.03827]
- Li, Patel, Viégas, Pfister & Wattenberg (2023) — "Inference-Time Intervention" (ITI) for truthfulness via probe-derived steering directions. [NeurIPS 2023; arXiv:2306.03341]
- Turner, Thiergart, Udell, Leech, Mini & MacDiarmid (2023) — Activation Addition (ActAdd): steering via difference-of-means directions without training. [arXiv:2308.10248]
- Zou, Phan, Chen, Campbell et al. (2023) — "Representation Engineering: A Top-Down Approach to AI Transparency", framing concept directions as the primary interface to model control. [arXiv:2310.01405]
- Marks & Tegmark (2023) — "Geometry of Truth": linear structure of true/false statements in LLaMA. [arXiv:2310.06824]
- Nanda, Lee & Wattenberg (2023) — Linear "world model" of board state in OthelloGPT. [arXiv:2309.00941]
- Rimsky, Gabrieli, Schubert et al. (2024) — Contrastive Activation Addition (CAA) for behavioural steering. [ACL 2024; arXiv:2312.06681]
- Park, Choe & Veitch (2024) — "The Linear Representation Hypothesis and the Geometry of Large Language Models": first rigorous counterfactual-based formalization, identifying a causal inner product that unifies probing and steering. [ICML 2024; arXiv:2311.03658]

**Latest Frontier (2024–2026) (最新前沿)**
- Park, Choe, Generalov, Lee, Yoon & Veitch (2024) — "The Geometry of Categorical and Hierarchical Concepts in LLMs": polytopes and simplices for categorical concepts. [arXiv:2406.01506]
- Jiang, Liu, Yu & Hu (2024) — Theoretical analysis showing LRH emerges from log-linear (softmax) model structure. [arXiv:2403.03867]
- Engels, Liao, Michaud, Gurnee & Tegmark (2024) — Discovery that some features (days of week, months) are **inherently multi-dimensional** (circular), explicitly violating a strict 1-D LRH. [arXiv:2405.14860]
- Tan, Cawley, Manzini et al. (2024) — "Analyzing the Generalization and Reliability of Steering Vectors", showing steering vectors generalize poorly OOD. [arXiv:2407.12404]
- Nguyen & Leng (2025) — SAND: vMF-based MLE framework for steering vectors dropping single-token unembedding pairs. [arXiv:2502.16385]
- Bello et al. (2025) — Affine mappings transfer steering directions across model scales, suggesting an inter-model linear-concept geometry. [arXiv:2505.24837]
- Kantamneni & Tegmark (2025) — Extensions to bilinear and nonlinear feature representations that subsume LRH as a limit case.

**Current Status (当前状态)**: The LRH is the implicit assumption underlying steering, probing, RepE, SAEs and concept erasure, and has accumulated substantial empirical support. Park et al. (2024) gave it its first rigorous formal statement, but the field is increasingly confronting counterexamples: multi-dimensional and circular features, bilinear interactions, context-dependent concept directions and OOD failure of steering vectors. Open questions: (i) **when does LRH hold and when does it fail?** (ii) is there a **canonical inner product** (causal, whitened) in which linearity is sharpest? (iii) **why does training produce linear concepts**—what training-dynamics or loss-geometry argument explains it? (iv) how do **multiple concept directions interact**—superpose, compose or interfere? (v) bridging LRH to SAEs and causal abstraction in a unified theory.

---

### 5. Probing and Lens Methods

**Foundational Papers (开创性论文)**
- Ettinger, Elgohary & Resnik (2016); Adi, Kermany, Belinkov, Lavi & Goldberg (2017) — Early diagnostic classifiers/probes for sentence representations. [RepEval; ICLR 2017]
- Alain & Bengio (2016) — Linear probes as the canonical lightweight readout. [arXiv:1610.01644]
- Conneau, Kruszewski, Lample, Barrault & Baroni (2018) — SentEval probing suite of 10 linguistic tasks. [ACL 2018]
- Hewitt & Manning (2019) — Structural probe: parse-tree distances are linearly encoded in BERT's geometry. [NAACL 2019]
- "nostalgebraist" (2020) — Original "logit lens": apply the unembedding to intermediate residual streams to read out partial predictions. [LessWrong]

**Key Developments (关键进展)**
- Hewitt & Liang (2019) — Control tasks: distinguishing probe accuracy due to representation quality vs. probe expressivity. [EMNLP 2019]
- Tenney, Xia, Chen et al. (2019) — Edge probing across 9 NLP tasks showing what BERT layers encode at what depth. [ICLR 2019]
- Pimentel, Valvoda, Maudslay, Zmigrod, Williams & Cotterell (2020) — "Information-Theoretic Probing for Linguistic Structure": reframed probing as mutual-information estimation. [ACL 2020]
- Voita & Titov (2020) — MDL (Minimum Description Length) probing: measure description length rather than accuracy. [EMNLP 2020; arXiv:2003.12298]
- Geva, Schuster, Berant & Levy (2021) — Transformer MLPs as key-value memories that promote vocabulary tokens, supporting logit-lens readouts. [EMNLP 2021]
- Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman & Steinhardt (2023) — Tuned Lens: per-layer affine probes correcting logit-lens biases. [arXiv:2303.08112]
- Pal, Sun, Yuan, Wallace & Bau (2023); Din et al. (2023) — Decoding intermediate predictions via "Future Lens" / jump-from-hidden-states. [arXiv:2311.04897]

**Latest Frontier (2024–2026) (最新前沿)**
- Ghandeharioun, Caciularu, Pearce, Dixon & Geva (2024) — Patchscopes: a unifying framework that "asks" the model itself to verbalize its hidden states. [ICML 2024; arXiv:2401.06102]
- Yom Din, Karidi, Choshen & Geva (2024) — Tuned-lens-style decoding extended to fine-tuned/instruction-tuned models. [arXiv:2303.09435]
- Sharma, Tegmark et al. (2024) — Probes for "refusal directions" and "deception directions" used for both interpretation and inference-time intervention.
- Belinkov & colleagues (2024) — Surveys consolidating probing's statistical pitfalls and the case for causal probes (AlterRep, AMNESIC probing).
- Paulo, Mallen, Juang & Belrose (2024) — Auto-interpretation pipelines treat SAE-feature activations as probes-by-construction. [arXiv:2410.13928]
- Belrose et al. (2024–2025) — Tuned-lens variants for Mamba/RWKV/SSM architectures, addressing how transformer-style lenses transfer. [AAAI 2025]

**Current Status (当前状态)**: Probing has shifted from "what is encoded?" toward causal and information-theoretic standards. Logit/tuned lens remain default tools for visualizing the iterative-prediction view of transformers; MDL and control-task baselines are the consensus for fair linguistic probing. Patchscopes blurs the line between probing and intervention. Open questions: (i) **probes vs. circuits**—what does a successful probe imply about computation? (ii) **lens transfer**—lenses tuned on one model rarely transfer; can we obtain a model-agnostic lens? (iii) extending lens/probe methodology to **multimodal, vision and SSM models** that lack a clean unembedding; (iv) controlling for **probe leakage** in instruction-tuned and RLHF models; (v) unifying probes with SAEs, DAS and concept erasure in a single causal-probing framework.

---

### 6. Information-Theoretic Methods (信息论方法)

**Foundational Papers (开创性论文)**
- Tishby, Pereira & Bialek (1999) — The original Information Bottleneck (IB) principle from rate-distortion theory. [Allerton; arXiv:physics/0004057]
- Tishby & Zaslavsky (2015) — Reframed deep learning as IB optimization. [IEEE ITW 2015]
- Williams & Beer (2010) — Partial Information Decomposition (PID): decomposes joint MI into unique, redundant and synergistic atoms. [arXiv:1004.2515]
- Shwartz-Ziv & Tishby (2017) — "Opening the Black Box of Deep Neural Networks via Information": claimed a fitting-then-compression phase in the information plane. [arXiv:1703.00810]

**Key Developments (关键进展)**
- Alemi, Fischer, Dillon & Murphy (2016) — Deep Variational Information Bottleneck made IB tractable in deep nets. [ICLR 2017; arXiv:1612.00410]
- Bertschinger, Rauh, Olbrich, Jost & Ay (2014) — A principled bivariate PID via convex optimization, addressing axiom issues. [Entropy 2014]
- Saxe, Bansal, Dapello, Advani, Kolchinsky, Tracey & Cox (2018) — "On the Information Bottleneck Theory of Deep Learning": disputed the compression claim, attributing it to saturating nonlinearities and MI-estimation artifacts. [ICLR 2018]
- Belghazi, Baratin, Rajeswar et al. (2018) — MINE: neural MI estimation, enabling MI-based analyses at scale. [ICML 2018]
- Achille & Soatto (2018) — Information Dropout / Emergence of Invariance: linked IB to invariance and disentanglement. [TPAMI; arXiv:1706.01350]
- Goldfeld et al. (2019) — Reframed observed compression as geometric, not information-theoretic. [ICML 2019]
- Wibral, Lizier, Priesemann et al. — Applications of PID to neural data and ML representations.

**Latest Frontier (2024–2026) (最新前沿)**
- Ehrlich, Schick-Poland, Boyd-Graber et al. (2024) — "A Measure of the Complexity of Neural Representations based on PID": quantifies how concept information is split across neurons. [ICLR 2024; OpenReview R8TU3pfzFr]
- Kolchinsky (2022/2024) — "A Novel Approach to the Partial Information Decomposition", a multivariate framework grounded in formal set-theoretic analogies. [Entropy 2022]
- Kawaguchi, Deng, Ji & Huang (2023/2024) — "How Does Information Bottleneck Help Deep Learning?": rigorous generalization bounds via IB. [ICML 2023]
- Mediano, Rosas et al. (2024–2025) — Φ-ID / integrated information decomposition for hierarchical, multi-scale information in deep networks.
- Diao, Wu et al. (2025) — Interpretable PID analyses of deep classifiers connecting unique/redundant information to OOD generalization. [MDPI Entropy 28(1):50]
- Han, Gallegos et al. (2025) — PID along reaction coordinates and other scientific domains, showing PID's reach beyond NN-internal analysis. [arXiv:2603.05672]
- Murphy, Bareinboim et al. (2025) — Causal information decompositions bridging Branches 3 and 6.

**Current Status (当前状态)**: Information-theoretic analysis remains conceptually attractive but practically fraught due to the difficulty of estimating mutual information in high-dimensional continuous representations. The strong Tishby compression story is no longer accepted, but the IB **as an objective** (variational, sliced, contrastive) is widely used. PID has matured into a quantitative tool for asking "what kind of information do neurons share?" Open questions: (i) **MI estimation**—every estimator has known failure modes; (ii) the **right PID measure**—Williams-Beer, BROJA, Ince's I_ccs and Kolchinsky disagree on synergy/redundancy in deterministic systems; (iii) connecting **information** to **causation** (PID does not by itself imply causal structure); (iv) using IB/PID as a **design principle** rather than a descriptive tool; (v) extending PID to **continuous, high-dim, learned** representations.

---

### 7. Topological and Geometric Methods (拓扑/几何方法)

**Foundational Papers (开创性论文)**
- Edelsbrunner, Letscher & Zomorodian (2002); Carlsson (2009) — Foundations of persistent homology and topological data analysis. [Discrete & Comput. Geometry; BAMS 2009]
- Cohen, Chung, Lee & DiCarlo (2020) — Manifold capacity theory linking neural-population geometry to linear-classification capacity. [Nature Communications]
- Levina & Bickel (2004); Facco, d'Errico, Rodriguez & Laio (2017) — Maximum-likelihood and TwoNN intrinsic-dimension estimators that the field relies on. [NeurIPS 2004; Scientific Reports]
- Ansuini, Laio, Macke & Zoccolan (2019) — Showed deep networks exhibit a "hunchback" ID profile across layers and the last-hidden-layer ID predicts test accuracy. [NeurIPS 2019; arXiv:1905.12784]

**Key Developments (关键进展)**
- Naitzat, Zhitnikov & Lim (2020) — Persistent homology of activations across layers shows topology gets simplified by depth. [JMLR 2020]
- Pope, Zhu, Abdelkader, Goldblum & Goldstein (2021) — "The Intrinsic Dimension of Images": ID of natural-image manifolds correlates with sample complexity. [ICLR 2021]
- Birdal, Lou, Guibas & Simsekli (2021) — Persistent homology of optimization trajectories predicts generalization. [NeurIPS 2021]
- Cang & Wei (2017–2020) — TopologyNet / persistent-homology features in deep networks. [PLOS Comp Bio]
- Bronstein, Bruna, Cohen & Veličković (2021) — Geometric Deep Learning blueprint: symmetry-based view of representations. [arXiv:2104.13478]
- Watanabe & Yamana (2021) — Topological measurement of DNNs via persistent homology on clique complexes. [arXiv:2106.03016]
- Valeriani, Doimo, Cuturello, Laio, Cazzaniga & Ansuini (2023/2024) — Geometry of hidden representations of LLMs and protein models, finding an intermediate "semantic plateau" of low ID. [NeurIPS; arXiv:2302.00294]

**Latest Frontier (2024–2026) (最新前沿)**
- Cheng, Cloos, Yamins & Williams (2025) — "Local Intrinsic Dimension of Representations Predicts Alignment and Generalization in AI Models and Human Brain." [arXiv:2601.22722]
- Wei et al. (2024–2025) — TDA/TDL surveys beyond persistent homology (persistent Laplacians, Dirac operators, Khovanov homology). [Artif. Intell. Rev. 2025; arXiv:2507.19504]
- Tulchinskii, Kuznetsov, Kushnareva et al. (2024) — Intrinsic dimension as a detector of machine-generated text and a generalization predictor for LLMs. [NeurIPS 2024]
- Cheng, Yamins et al. (2024) — Manifold-capacity-style analyses of vision–language models and instruction-tuned LLMs.
- Robinson, Sun, Vaccari et al. (2025) — Topological metrics on token embeddings revealing universal structure across LLMs.
- Birdal et al. (2024–2025) — Connections between topological complexity, generalization and grokking.
- 2025 work using **neural manifolds in LLMs**—curvature, geodesic distance and dimensionality across layers as a unified description.

**Current Status (当前状态)**: Geometric/topological methods provide some of the most architecture-agnostic descriptions of representations—every neural network has an ID and a curvature profile, regardless of modality. ID has become a practical diagnostic predicting generalization, memorization, and alignment with brain data. Open questions: (i) **estimator robustness**—TwoNN, MLE and GeoMLE can disagree by factors of 2–3 in deep nets; (ii) what causal role does **low ID** play—is it a cause of generalization or a consequence? (iii) integrating **topological invariants** with mechanistic interpretability—homology has been more descriptive than predictive; (iv) extending to **dynamic representations** in RNNs/SSMs and in-context learning; (v) bridging neural manifolds in ML with neuroscience's manifold-coding literature.

---

### 8. Disentanglement and Identifiability

**Foundational Papers (开创性论文)**
- Comon (1994); Hyvärinen & Oja (2000) — Linear ICA and the FastICA algorithm—the classical identifiability template. [Signal Processing; Neural Networks]
- Hyvärinen & Pajunen (1999) — Proved unsupervised nonlinear ICA is fundamentally non-identifiable without auxiliary assumptions. [Neural Networks]
- Bengio, Courville & Vincent (2013) — Articulated disentanglement as a desideratum of representation learning. [TPAMI; arXiv:1206.5538]
- Higgins, Matthey, Pal et al. (2017) — β-VAE: increasing the KL weight encourages disentangled latent factors. [ICLR 2017]
- Kim & Mnih (2018) — FactorVAE: penalize total correlation directly. [ICML 2018; arXiv:1802.05983]
- Chen, Li, Grosse & Duvenaud (2018) — β-TCVAE: ELBO decomposition isolating the total-correlation term. [NeurIPS 2018]

**Key Developments (关键进展)**
- Locatello, Bauer, Lucic, Rätsch, Gelly, Schölkopf & Bachem (2019) — **Impossibility result**: unsupervised disentanglement is impossible without inductive biases or supervision (ICML best paper). [ICML 2019; arXiv:1811.12359]
- Hyvärinen, Sasaki & Turner (2019) — Nonlinear ICA via auxiliary variables: the first identifiability theorem with provable identifiability up to permutation. [AISTATS 2019]
- Khemakhem, Kingma, Monti & Hyvärinen (2020) — iVAE: unified VAEs with conditional priors and nonlinear ICA, proving identifiability. [AISTATS 2020; arXiv:1907.04809]
- Schölkopf, Locatello, Bauer et al. (2021) — "Toward Causal Representation Learning": the foundational CRL position paper. [Proc. IEEE; arXiv:2102.11107]
- Zimmermann, Sharma, Schneider, Bethge & Brendel (2021) — Contrastive learning provably inverts the data-generating process (identifiability through InfoNCE). [ICML 2021]
- Klindt, Schott, Sharma et al. (2021); Hälvä & Hyvärinen (2020) — Sparse/temporal nonlinear ICA. [ICLR 2021]
- Lachapelle et al. (2022) — "Disentanglement via Mechanism Sparsity Regularization": connects ICA and causality through sparse causal mechanisms. [CLeaR 2022; arXiv:2107.10098]
- von Kügelgen, Sharma, Gresele, Bottou & Schölkopf (2021) — Self-supervised augmentations provably disentangle content from style. [NeurIPS 2021]

**Latest Frontier (2024–2026) (最新前沿)**
- Brehmer, de Haan, Lippe & Cohen (2022/2024) — Weakly-supervised CRL identifiable from interventions. [NeurIPS 2022]
- Squires, Seigal, Bhate & Uhler (2023); Buchholz, Rajendran et al. (2024) — Identifiability of latent causal graphs from single-node interventions; extensions to nonparametric mixing. [ICML 2023/2024]
- Reizinger, Bizeul, Cosentino et al. (2024) — Identifiability of LLMs under next-token prediction: conditions for unique recovery of training distribution.
- Ahuja, Hartford, Bengio et al. (2024) — Provable identifiability from multiple environments and group-theoretic constraints.
- Mansouri, Hartford, Zhang et al. (2025) — CRL from text+image data via partially observable environments.
- Komanduri et al. (2023/2024) — ICM-VAE: causal disentanglement via flow-based mechanisms with permutation-equivalent identifiability. [arXiv:2306.01213]
- Hyvärinen, Khemakhem & Morioka (2023) — "Nonlinear Independent Component Analysis for Principled Disentanglement in Unsupervised Deep Learning"—review and synthesis. [Patterns 4(10):100844]

**Current Status (当前状态)**: After Locatello 2019, the field largely abandoned the goal of purely unsupervised disentanglement and shifted to **identifiability theory** under explicit (if modest) supervision: auxiliary variables, multiple environments, interventions, temporal structure, sparsity, data augmentations. Causal Representation Learning (CRL) is the active synthesis with causal inference. Open questions: (i) **realistic assumptions**—much identifiability theory assumes interventions or environment labels hard to obtain; (ii) **what counts as "the right" representation** in LLMs—LRH-style concept directions seem disentangled but are not derived via CRL; (iii) **identifiability of foundation models**—are pretrained representations identifying any underlying causal structure? (iv) connecting CRL to mechanistic interpretability and SAE-style dictionaries; (v) **evaluation**—DCI, MIG, SAP inherit the same problems Locatello flagged.

---

### 9. Representation Scaling Laws and Convergence

**Foundational Papers (开创性论文)**
- Lenc & Vedaldi (2015) — Early "equivariance and equivalence" results showing CNN representations are linearly relatable. [CVPR 2015]
- Conneau, Lample, Ranzato, Denoyer & Jégou (2017) — Word embeddings across languages are linearly alignable. [ICLR 2018]
- Kornblith et al. (2019); Morcos et al. (2018) — Empirical observation that wider/deeper networks on the same data become more representationally similar.
- Bansal, Nakkiran & Barak (2021) — "Revisiting Model Stitching": independently trained networks can be stitched with a single linear map, evidence for representational convergence ("Anna Karenina scenario"). [NeurIPS 2021; arXiv:2106.07682]
- Roeder, Metz & Kingma (2021) — Linear identifiability of representations across self-supervised methods. [ICML 2021]

**Key Developments (关键进展)**
- Merullo, Castricato, Eickhoff & Pavlick (2022) — "Linearly Mapping from Image to Text Space": one linear projection connects frozen vision and language encoders. [ICLR 2023]
- Moayeri, Pope, Balaji & Feizi (2023) — Cross-modal alignment via linear maps between vision and language encoders.
- Wu, Karras et al. (2024) — Representation convergence in diffusion models.
- Maniparambil, Akshulakov et al. (2024) — Vision encoders pretrained without text align to LLM embeddings via a linear map, supporting modality-agnostic structure.
- Geirhos et al. (2022); follow-ups to Bansal stitching — Extensions of the convergence picture.

**Latest Frontier (2024–2026) (最新前沿)**
- Huh, Cheung, Wang & Isola (2024) — **The Platonic Representation Hypothesis**: as models scale and are trained on diverse data, representations across modalities and architectures measurably converge (measured via mutual k-NN), toward a putative shared statistical model of reality. [ICML 2024; arXiv:2405.07987]
- Cao & Yamins (2024) — "Explanatory models in neuroscience" argued for convergence toward task-defining solutions.
- Lampinen, Chan, Hermann et al. (2024) — Follow-up showing convergence is partly task-driven and partly architecture-driven.
- Ziyin, Wang et al. (2025) — Identified regimes where Platonic convergence breaks (weight decay, label noise, edge-of-stability training). [arXiv:2507.xxxx]
- Sucholutsky, Muttenthaler, Weller, Peng et al. (2024) — "Getting Aligned on Representational Alignment": position paper unifying ML, neuroscience and cog-sci views. [arXiv:2310.13018]
- Bello, Sharma, Veitch et al. (2025) — Affine transferability of steering vectors across model scales as further evidence of a shared geometry. [arXiv:2505.24837]
- Khona, Tegmark, Engels et al. (2024–2025) — Studies of how SAE features change with scale, finding both new features and stable "universal" features.
- Quanta Magazine (Jan 2026) reporting — Synthesized the ongoing debate and identified open empirical challenges to the convergence claim.

**Current Status (当前状态)**: A clear empirical pattern—representations across architectures, modalities and training regimes converge as scale increases—has become the field's most striking macro-observation. Stitching, linear cross-modal mapping and the Platonic paper are the main pillars. Open questions and active debates: (i) **does convergence imply a unique solution** (Anna Karenina) or merely a low-dimensional family? (ii) what are the **selective pressures**—task, data diversity, optimizer or regularization? (iii) **counterexamples**—specialized models, RL-only training and weight decay break convergence; (iv) **scaling laws for representational quality**, not just loss—how should we measure representation improvement? (v) what does convergence imply for **AI safety, transfer and brain-model comparison**?

---

### 10. Impossibility Results and Theoretical Foundations (不可能性与理论基础)

**Foundational Papers (开创性论文)**
- Wolpert (1996) — No-Free-Lunch theorems for supervised learning. [Neural Computation]
- Hyvärinen & Pajunen (1999) — Unsupervised nonlinear ICA is non-identifiable. [Neural Networks]
- Kleinberg (2002) — Axiomatic impossibility theorem for clustering—a template now adopted by representation theorists. [NIPS 2002]
- Kornblith, Norouzi, Lee & Hinton (2019) — Proved no statistic invariant to invertible linear transformations can meaningfully measure similarity in dim > n. [ICML 2019; arXiv:1905.00414]
- Locatello et al. (2019) — Impossibility of unsupervised disentanglement without inductive bias. [ICML 2019; arXiv:1811.12359]

**Key Developments (关键进展)**
- Bertschinger, Rauh, Olbrich & Jost (2014) — Identity-axiom inconsistency: no PID can simultaneously satisfy all desired axioms for deterministic systems. [Entropy 2014]
- Lipton (2018) — "The Mythos of Model Interpretability": critique of ill-defined goals in interpretability. [Communications of the ACM]
- Räuker, Ho, Casper & Hadfield-Menell (2023) — Survey of mechanistic interpretability framing limits and theoretical foundations. [SaTML 2023]
- Williams et al. (2021/2024) — Generalized shape metrics as an axiomatic foundation for similarity.
- Geiger, Ibeling, Zur et al. (2024/2025) — Causal abstraction as a **unifying theoretical foundation** for mechanistic interpretability. [JMLR 26(83)]
- Sucholutsky et al. (2024) — Axiomatic position paper on representational alignment across ML, neuroscience and cog-sci.

**Latest Frontier (2024–2026) (最新前沿)**
- Engels, Liao, Michaud, Gurnee & Tegmark (2024) — Empirical refutation of strict 1-D LRH (multi-dimensional features) functions as a **partial impossibility result** for atomic linear concept-extraction. [arXiv:2405.14860]
- Friedman, Lampinen, Dixon, Chen & Ghandeharioun (2024) — Theoretical limits of interpretability via sparse autoencoders: SAEs cannot in general recover ground-truth features without additional assumptions.
- Wattenberg & Viégas (2024) — "Relational Composition in Neural Networks": call for principled frameworks beyond LRH. [arXiv:2407.14662]
- Anwar, Saparov, Rando et al. (2024) — "Foundational Challenges in Assuring Alignment and Safety of LLMs"—catalogs theoretical obstacles including representation-analysis limits. [arXiv:2404.09932]
- Cloos, Yamins, Williams et al. (2024–2025) — Showed similarity metrics can be made adversarially inconsistent, an axiomatic obstruction for any "universal" similarity score.
- Wu, Geiger et al. (2025) — Formal results on **faithfulness vs. completeness** trade-offs in interpretability: methods cannot satisfy both simultaneously without restrictions.
- Murthy, Tegmark et al. (2025) — Theoretical analyses of when SAE features correspond to ground-truth features in toy models with known generative processes.
- Bareinboim, Pearl et al. (2024) — Foundational results on what causal questions about NN representations are identifiable from observational vs. interventional access.

**Current Status (当前状态)**: This branch is theoretically demanding and the least populated, but it is increasingly important. Impossibility results (Locatello 2019, Kornblith 2019, Hyvärinen-Pajunen 1999, Bertschinger 2014) have been field-defining: they tell practitioners which assumptions are *required* to make claims. Open questions: (i) **a unified axiomatic framework** for representation comparison (akin to Kleinberg for clustering)—is there one? (ii) under what conditions can we **guarantee** that SAEs/probes/DAS recover ground-truth structure? (iii) **trade-offs between faithfulness, completeness, simplicity and minimality** in interpretability—are they simultaneously achievable? (iv) what are the **information-theoretic limits** on what an interpretability method can extract from black-box access vs. weight access vs. training-process access? (v) bridging Pearl-style identifiability with the practical interpretability stack.

---

## Cross-Branch Summary

### 1. Which branches are most active and progressing fastest?

By volume and impact of 2024–2026 papers:

1. **Sparse Autoencoders (Branch 2)** — by a wide margin. Architectural innovation (TopK, JumpReLU, Matryoshka), open SAE suites (Gemma Scope, Llama Scope) and integration with circuits (transcoders, crosscoders) produce near-monthly advances. Anthropic, DeepMind and OpenAI treat SAEs as a primary research vehicle.
2. **Causal Abstraction / Mechanistic Interpretability (Branch 3)** — driven by Geiger et al.'s causal-abstraction synthesis and Anthropic's attribution-graph circuit-tracing program.
3. **Linear Representation Hypothesis (Branch 4)** — Park et al. (2024) gave the field its first rigorous formulation; the wave of steering/RepE/concept-direction work continues, now with multi-dimensional features and bilinear extensions.
4. **Representation Scaling / Convergence (Branch 9)** — the Platonic Representation Hypothesis catalyzed a year of follow-ups, counterexamples, and cross-disciplinary engagement with neuroscience.

Less rapidly advancing but stable: similarity metrics (Branch 1), probing/lens (Branch 5), TDA/geometry (Branch 7). Slower or more niche: information-theoretic methods (Branch 6, hampered by MI estimation) and classical disentanglement (Branch 8, largely absorbed into CRL).

### 2. Which branches have under-explored intersections/crossovers?

- **SAEs × Information Theory (2 × 6)**: SAEs are dictionary learning, but the field rarely quantifies the *information* SAE features carry about model behaviour via PID or MI bounds. PID could explain feature absorption and splitting in information-theoretic terms.
- **Linear Representation Hypothesis × Identifiability (4 × 8)**: LRH is treated as an empirical phenomenon; CRL provides exactly the identifiability theory needed to ask when LRH *should* hold. Bridging these would yield a theory of *why* concepts are linear.
- **Topology/Geometry × Mechanistic Interpretability (7 × 3)**: Mechanistic work has been almost entirely algebraic (subspaces, directions, circuits) and barely uses persistent homology, curvature or manifold capacity to characterize circuits.
- **Similarity Metrics × Causal Abstraction (1 × 3)**: Similarity is largely defined geometrically; defining similarity in terms of *causal equivalence* (when two models implement the same causal abstraction) is barely explored.
- **Scaling / Convergence × Impossibility (9 × 10)**: Convergence is empirical; theoretical impossibility/uniqueness conditions for "the Platonic representation" are not formalized.
- **Probing/Lens × SAEs (5 × 2)**: Auto-interpretation pipelines treat SAE features as probes, but the statistical-probing literature (control tasks, MDL) has yet to be properly applied to SAE features.
- **Information Theory × Causal Abstraction (6 × 3)**: PID is naturally a causal-style decomposition but is rarely connected to interchange interventions or DAS.

### 3. The 3–5 biggest open questions across the entire field

1. **What is the "right unit" of representation?** Neurons, linear directions, sparse-coded dictionary atoms, multi-dimensional manifolds or computational circuits? Each subfield assumes a different answer (Branch 4 says directions; Branch 2 says dictionary atoms; Engels et al. 2024 say sometimes circles; Branch 3 says causal variables; Branch 7 says low-dimensional manifolds). A unifying ontology does not yet exist.

2. **When do interpretability methods recover ground truth, and when do they fabricate?** SAEs produce plausible features that may not be causally real (RAVEL); probes can succeed on randomized labels; circuit-tracing often explains only a fraction of behaviour under causal scrubbing; similarity metrics can be adversarially manipulated. The field needs **formal guarantees and benchmarks** that distinguish discovery from confabulation.

3. **Why and how do representations converge across scale, modality and architecture?** The Platonic hypothesis is empirically supported but theoretically unexplained. Is convergence driven by the data, the loss, the optimizer or the architecture's inductive biases? When does it break, and what does it imply for transfer, safety and brain–model comparisons?

4. **How does mechanistic understanding scale from toy circuits to frontier-model behaviour?** All worked examples (IOI, greater-than, modular addition) are small and well-defined. Reasoning, planning, chain-of-thought, agentic behaviour and tool use have no completed circuit-level explanations. Bridging features → circuits → behaviours at frontier scale is the central practical challenge.

5. **Is there a unified theoretical foundation linking causation, information, geometry and identifiability?** Currently, Branches 1, 3, 6, 8 and 10 each have partial axiomatic foundations that don't talk to each other. A unified framework—analogous to PAC learning for supervised learning—would let us state what "understanding a representation" formally means and prove what is and isn't possible.

These five questions cut across the ten branches and define the agenda for representation-analysis research through the rest of the decade.

