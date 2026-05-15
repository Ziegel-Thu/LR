### 神经网络表征数学结构的深层研究问题

用户:
基于 DR-001 的追问：更多开放问题  附件是我上一次 deep research 的完整输出（DR-001：神经网络表征的数学结构）。它梳理了 8 个子方向并给出了 10 个开放问题。  请在此基础上再给出 10-15 个值得探索的研究问题。要求：   - 不要重复 DR-001 已列出的 10 个   - 可以是某个子方向内部更细的问题，也可以是跨方向的，也可以是 DR-001 未覆盖的新方向   - 每个问题同样给出：问题陈述、为什么重要、当前进展与 gap、难度和风险评估   - 给出相关论文的 arXiv ID

Claude:

Let me first read the uploaded file to understand DR-001's content before launching the research.
et me see the truncated portion too:
ow I have the full picture of DR-001. Let me launch the research task to find 10-15 additional open problems.
our research report on open problems in neural network representation theory is ready. It proposes 15 new research problems for 2024-2026 that complement (without duplicating) the prior DR-001 problem set. The problems span score-field Riemannian geometry in diffusion models, Mamba vs. Transformer representational capacity, KAN geometry, RLHF-induced representation drift, in-context learning Bayes gaps, Platonic representation refinements, quantization phase transitions, equivariant network capacity, μP for non-standard architectures, transcoders vs. SAEs, manifold dictionary learning, continuous chain-of-thought superposition, linear steering theory, cross-architecture convergence, and symmetry discovery from activations. Each problem includes a formal statement, motivation, progress/gap analysis, difficulty rating, and anchor arXiv IDs, organized across four categories (sub-area refinements, cross-cuts, undeveloped DR-001 threads, and new directions).
# DR-002: Fifteen New Open Problems on the Mathematical Structure of Neural Network Representations (2024–2026)

This document proposes **15 new open research problems** that complement, but do **not duplicate**, the ten problems listed in DR-001. Each problem is stated specifically, motivated against current literature (with arXiv anchors from 2024–2026), and tagged with a difficulty/risk profile. Problems span DR-001's eight sub-areas at finer resolution (Category A), cross-cuts between them (B), directions DR-001 mentioned but did not develop (C), and brand-new directions (D).

---

## Q1. Score-Field Riemannian Geometry: Curvature–Generalization Coupling in Diffusion Models  (Category C)

**Problem.** Let $s_\theta(x,t) \approx \nabla_x \log p_t(x)$ be the learned score of a diffusion model. Define the score-induced Riemannian metric $g(x) = \alpha I + \beta\, s_\theta s_\theta^\top + \gamma\, \nabla s_\theta$. (i) Prove or refute that *generalization error scales with a specific functional of sectional curvature* of the learned manifold $\mathcal{M}_t$ rather than its intrinsic dimension alone. (ii) Identify the precise t-dependent phase at which curvature explodes (the "geometric phase transition") and tie it to the memorization–generalization transition.

**Importance.** Connects three threads: manifold hypothesis, generalization theory, and the empirical observation that diffusion models generalize from very few samples on highly curved natural-image manifolds.

**Progress/gap.** Stanczuk et al. show diffusion encodes intrinsic dimension; Azeglio et al. (arXiv 2505.11128) define a score-based Riemannian metric; arXiv 2510.02305 shows log-domain smoothing is geometry-adaptive; arXiv 2603.20645 decomposes score into tangential/normal parts and gives statistical complexity bounds. None tie *curvature* (not just dimension) causally to generalization or memorization phase transitions (cf. Ventura et al., ICLR 2025).

**Difficulty/risk.** Hard. Risk: curvature may be ill-defined globally on real image manifolds; surrogate notions (Ricci on $\epsilon$-tubes) may be needed.

**Key arXiv IDs.** 2505.11128, 2510.02305, 2603.20645, 2504.20288, 2505.09922.

---

## Q2. Representation Geometry of Selective State-Space Models vs. Transformers  (Category D)

**Problem.** Mamba and Transformer match on language modeling loss but diverge on retrieval. Formalize this: prove that for a class of in-context multimodal retrieval tasks the *hidden-state manifold* of an SSM of width $d$ has bounded intrinsic capacity $O(d)$ for storing key–value pairs, whereas a Transformer with attention has $\Omega(L)$ capacity in sequence length $L$. Identify the geometric invariant (e.g., effective rank of the controllability Gramian of the input-dependent $A(x)$ matrix) that controls this gap.

**Importance.** Differentiates the two dominant LLM architectures at the level of representational *capacity*, not just FLOPs.

**Progress/gap.** Empirical: arXiv 2409.05395 shows Mamba-VLM lags Transformer at fine-grained retrieval but matches on coarse tasks. arXiv 2406.16722 surveys the architectural transition. No paper to date proves a capacity separation in terms of an explicit geometric invariant of the selective SSM recurrence.

**Difficulty/risk.** Hard but tractable. Control-theoretic Gramian framework is well-developed; mapping to in-context retrieval needs care.

**Key arXiv IDs.** 2312.00752, 2409.05395, 2406.16722.

---

## Q3. Geometric Theory of KAN Representations  (Category D)

**Problem.** KANs replace linear weights with learnable splines on edges. (i) Characterize the *feature manifold* of a trained KAN: is the residual/activation geometry fundamentally different (e.g. piecewise-polynomial cell complex) from an MLP's piecewise-linear cell complex? (ii) Do the eight DR-001 subareas (CKA similarity, intrinsic dimension, superposition, mode connectivity, etc.) transfer qualitatively, or do KAN representations require a new toolkit (e.g. B-spline–coefficient lattices instead of activation vectors)?

**Importance.** KANs are an alternative to the universal-approximation paradigm. If their representational geometry differs fundamentally, much of mechanistic interpretability must be redone.

**Progress/gap.** arXiv 2404.19756 introduced KANs; arXiv 2410.08026 proves generalization bounds; arXiv 2508.00247 gives a sinusoidal approximation theorem; arXiv 2509.03417 studies initialization and NTK spectra. None has answered whether superposition, polysemanticity, or LMC arise in KANs.

**Difficulty/risk.** Medium-hard. Risk: KANs may not scale to LLMs, limiting generality of conclusions.

**Key arXiv IDs.** 2404.19756, 2410.08026, 2508.00247, 2509.03417.

---

## Q4. RLHF/DPO-Induced Representation Drift: A Quantitative Theory  (Category C)

**Problem.** Define a drift functional $D(\theta_{\rm SFT}, \theta_{\rm RLHF})$ measuring how RLHF deforms internal representations beyond the KL-regularized policy distance. (i) Prove that for a reward model $r$ with bounded gradient w.r.t. residual stream, the induced representation shift concentrates on a low-rank subspace whose dimension is $O(\text{rank}(\nabla r))$. (ii) Test the "superficial alignment hypothesis" formally: does the spectrum of $\theta_{\rm RLHF}-\theta_{\rm SFT}$ have a heavy tail concentrated on later layers? (iii) Predict catastrophic-forgetting safety loss from this geometric measure.

**Importance.** Practical: enables principled monitoring of which capabilities RLHF distorts. Theoretical: connects RL post-training to feature-learning theory.

**Progress/gap.** arXiv 2312.15997 (RLHF via representation engineering) and arXiv 2410.15610 (online RLHF convergence) give algorithmic results, not geometric ones. arXiv 2602.08813 (FRPO) addresses safety-preservation empirically. The geometric drift question is open.

**Difficulty/risk.** Medium. Mostly empirical at first; theoretical lower bounds harder.

**Key arXiv IDs.** 2312.15997, 2410.15610, 2602.08813, 2504.12501.

---

## Q5. The Bayes Gap of In-Context Learning: Provable Misspecification Limits  (Category C)

**Problem.** Wakayama et al. (arXiv 2510.10981) decompose ICL risk into Bayes Gap + Posterior Variance and show the Bayes Gap is controlled by a Wasserstein OOD penalty. Open: derive a *tight matching lower bound* showing that no Transformer of width $w$ and depth $L$ trained on prompt distribution $\mathcal{P}_{\rm pre}$ can shrink the Bayes Gap below $c\cdot W_2(\mathcal{P}_{\rm pre},\mathcal{P}_{\rm test})/\sqrt{wL}$. Identify the architectural mechanism (induction heads? attention sinks?) that saturates this bound.

**Importance.** Closes the explanatory loop on "is ICL Bayesian?" by quantifying *when it provably is not*.

**Progress/gap.** Upper bounds exist (arXiv 2510.10981, 2306.04891, 2501.16825). Matching lower bounds, especially ones tied to specific circuits, are missing.

**Difficulty/risk.** Hard. Lower bounds for learned algorithms inside Transformers are notoriously difficult; risk of producing only loose bounds.

**Key arXiv IDs.** 2510.10981, 2306.04891, 2501.16825.

---

## Q6. Heterogeneity-Dependent Failure of the Platonic Representation Hypothesis  (Category A/B)

**Problem.** Tjandrasuwita et al. (arXiv 2502.16282) empirically show Platonic alignment correlates with downstream performance only at low modality heterogeneity and high redundancy (PID decomposition). Formal open problem: derive a *no-free-lunch theorem* stating that for any pair of modalities with unique-information component $U>0$ in the PID decomposition, the optimal joint predictor *cannot* lie in the intersection of unimodal-aligned subspaces. This would supply a *falsifiable refinement* of the Platonic hypothesis distinct from DR-001's open problem #3.

**Importance.** Resolves a fast-growing conflict between Huh et al.'s Platonic claim and accumulating counterexamples.

**Progress/gap.** arXiv 2502.16282, 2511.12121, 2503.15352, 2502.17028 explore the empirical phenomenon and propose distributional alignment objectives, but none gives a clean impossibility result indexed by PID quantities.

**Difficulty/risk.** Medium. Information-theoretic; risk of being trivial unless coupled with finite-sample rates.

**Key arXiv IDs.** 2502.16282, 2511.12121, 2503.15352, 2502.17028, 2502.14047.

---

## Q7. Quantization-Induced Representation Deformation and a Critical Bit-Width  (Category D)

**Problem.** Let $f_b$ be the b-bit quantized version of an LLM $f$. Define the representation distortion $\Delta_b(x)=\|h(x)-h_b(x)\|/\|h(x)\|$ in the residual stream. (i) Conjecture and prove a *critical bit-width* $b^\star$ below which superposition collapses: features that were geometrically separable in full precision become indistinguishable, and SAE-discovered features cease to be linearly recoverable. (ii) Tie $b^\star$ to the intrinsic dimension and the eigenvalue gap of the activation covariance.

**Importance.** Practically: predicts quantization recipes that preserve interpretability, not just loss. Theoretically: places "compression vs. representation" on a rigorous footing.

**Progress/gap.** arXiv 2307.02973 and arXiv 2510.22058 (GNNs) compare pruning vs. quantization empirically; no published work that I located connects quantization bit-width to a phase transition in superposition or SAE feature recoverability.

**Difficulty/risk.** Medium. Risk: the "transition" may be smooth rather than sharp, requiring soft-threshold analysis.

**Key arXiv IDs.** 2307.02973, 2510.22058, 2407.04803.

---

## Q8. Representation Theory of Equivariant Networks Beyond Linear Decomposition  (Category D)

**Problem.** Gibson–Tubbenhauer–Williamson (arXiv 2408.00949) decompose equivariant layers into simple group representations and show ReLU yields piecewise-linear equivariant maps governed by normal subgroups (Galois-like). Open problems: (i) Quantify the *expressive capacity* of these filtered nonlinear equivariant maps, e.g. the number of irreducible components reachable by depth-$L$ ReLU equivariant networks. (ii) Relate the filtration to *learned* symmetry — predict which non-trivial irreps a SymPE-style symmetry-broken network must populate.

**Importance.** Builds a true representation-theoretic foundation for equivariant networks (the field currently relies on universal approximation arguments).

**Progress/gap.** arXiv 2408.00949 is the seed result; arXiv 2503.21985 (SymPE), 2505.13631 (relaxed equivariance), 2410.09841, 2302.00236 (symmetry discovery) provide complementary directions but the capacity/learnability questions are open.

**Difficulty/risk.** Hard. Requires fluency in modular representation theory and learning theory; risk of being beautiful but inapplicable to large-scale practice.

**Key arXiv IDs.** 2408.00949, 2503.21985, 2410.09841, 2302.00236, 2505.13631.

---

## Q9. μP for Non-Standard Architectures: Mamba, MoE, and KAN Feature Learning Limits  (Category C)

**Problem.** Yang's Tensor Programs derive maximal-update parametrization (μP) for MLPs and Transformers. The infinite-width feature-learning limit is unknown for: (a) selective SSMs whose $A(x),B(x),C(x)$ depend on input; (b) Mixture-of-Experts where the router introduces discrete switching; (c) KANs whose weights are functions, not scalars. Derive μP-style parametrizations for each and prove zero-shot HP-transfer, or exhibit a no-go theorem. This is distinct from DR-001 open problem #4 (which concerns scaling exponents *from* feature learning, not the parametrization itself).

**Importance.** Hyperparameter transfer is critical for training $>$100B models; current practice for Mamba/MoE is ad hoc.

**Progress/gap.** arXiv 2011.14522 (TP IV), 2310.02244 (TP VI, depth), 2503.09565 (L-layer global convergence under μP), 2508.09752 (μP for MoE — partial), 2502.02531 (Bordelon–Pehlevan deep linear). MoE μP is only initial; SSM/KAN μP not addressed.

**Difficulty/risk.** Medium-hard. Risk: input-dependent SSM dynamics may lack a clean width-limit DMFT description.

**Key arXiv IDs.** 2310.02244, 2011.14522, 2503.09565, 2508.09752, 2502.02531, 2402.17457.

---

## Q10. Transcoders vs. SAEs: A Theoretical Account of When Function Sparsity Beats Activation Sparsity  (Category C)

**Problem.** Empirically, transcoders (Dunefsky et al., arXiv 2406.11944; Paulo et al., arXiv 2501.18823) outperform SAEs on circuit-interpretability metrics. Open: prove a separation theorem. Specifically, for the class of MLPs implementing $f(x)=\sum_i \sigma(w_i^\top x)\,v_i$ with $k$-sparse activations, a transcoder of width $m$ can recover the input–output map with error $\epsilon$ whenever $m=\tilde O(k)$, while an SAE on the post-MLP activations requires $m=\tilde\Omega(k^2)$ to disentangle superposed function features. This is distinct from DR-001 problem #2 (SAE identifiability across seeds).

**Importance.** Tells practitioners which decomposition primitive (function-level vs. activation-level) is fundamental.

**Progress/gap.** Skip transcoders, crosscoders, KronSAE, route-SAE, binary sparse coding (arXiv 2501.18823, 2505.22255, 2503.08200, 2509.25596) proliferate without underlying theory. SAEBench (arXiv 2503.09532) measures empirically.

**Difficulty/risk.** Medium. Sparse-recovery theory exists; adaptation to nonlinear dictionaries is the novel piece.

**Key arXiv IDs.** 2406.11944, 2501.18823, 2505.22255, 2503.08200, 2509.25596, 2503.09532.

---

## Q11. Manifold Dictionary Learning for Non-Linear Features  (Category B; bridges geometry + mech interp)

**Problem.** Engels et al. (arXiv 2405.14860) show some LLM features are inherently *circular* or multi-dimensional manifolds, not linear directions. Open: develop a *manifold-dictionary* learning theory that decomposes residual streams as $h = \sum_i \phi_i(z_i)$ where each $\phi_i:\mathbb{S}^{d_i}\to\mathbb{R}^D$ is a learned low-dim immersion. Prove identifiability conditions (analogous to ICA) when the latent $z_i$ are independent, and bound sample complexity. Connect to the "non-linear dictionary learning as SAE alternative" direction listed in DR-001's preamble.

**Importance.** SAE-style linear decomposition is provably wrong for some features. A geometric replacement is needed to extend mech interp beyond toy linear cases.

**Progress/gap.** arXiv 2405.14860 identifies the phenomenon; no replacement dictionary theory exists. Connects to Q10 but at a *geometric* rather than functional level.

**Difficulty/risk.** Hard. Identifiability for nonlinear ICA is famously delicate (Hyvärinen–Pajunen impossibility).

**Key arXiv IDs.** 2405.14860, 2501.18823, 2503.09532.

---

## Q12. Representation Dynamics of Continuous Chain-of-Thought ("Reasoning in Superposition")  (Category D)

**Problem.** Zhu et al. (arXiv 2505.12514, 2509.23365) argue continuous-CoT reasoning operates in a *superposition* of discrete reasoning paths inside the residual stream. Formalize: (i) Define a trajectory $h_1,\dots,h_T$ of latent thoughts and prove that for graph-reachability tasks, the optimal continuous-CoT policy maintains a distribution over states whose dimension grows as $O(\log N)$ (vs. $N$ for explicit CoT). (ii) Characterize the geometric phase where a model transitions from "single-trajectory" to "superposed" reasoning during training. Distinct from DR-001 problem #5 (Grokking mechanism).

**Importance.** Test-time compute is the new scaling frontier; understanding its representational substrate is the key open theoretical question.

**Progress/gap.** arXiv 2505.12514, 2509.23365, 2412.13171 (compressed CoT), 2501.19393 (S1), 2509.00190 (state-aware CoT), 2510.09312 (CRV verification), 2508.17627 (overthinking) — all empirical or partial-theoretical.

**Difficulty/risk.** Hard. Risk: superposition framework may not generalize beyond synthetic graph tasks.

**Key arXiv IDs.** 2505.12514, 2509.23365, 2412.13171, 2501.19393, 2509.00190, 2510.09312, 2508.17627, 2503.04697.

---

## Q13. Theoretical Foundations of Representation Engineering: A Sufficiency Theorem for Linear Steering  (Category D)

**Problem.** Activation-steering / representation engineering works empirically but its success is uneven (arXiv 2407.12404). Open: prove a *sufficient condition* for linear steering effectiveness in terms of the conditional geometry of activations. Specifically: if the within-concept activation distribution is approximately log-concave and the across-concept direction has high logit-lens projection, then a difference-of-means steering vector achieves bounded behavioral change with $1-\delta$ probability. Conversely, exhibit a class of concepts (e.g. compositional or non-monotonic) provably *non*-linearly-steerable.

**Importance.** Representation engineering is now widely deployed for safety; it lacks any provable correctness guarantees.

**Progress/gap.** Park et al. (linear representation hypothesis, causal inner product) and arXiv 2502.16385, 2502.17601 (survey), 2407.12404 (generalization analysis), 2604.15557 (predicting steering success) provide empirical scaffolding. The "three-regime" framework of 2604.15557 is a starting point but lacks formal proof.

**Difficulty/risk.** Medium. Risk: the linear representation hypothesis may itself be only approximately true (cf. Q11), limiting applicability.

**Key arXiv IDs.** 2502.17601, 2502.16385, 2407.12404, 2502.19649, 2604.15557, 2308.10248.

---

## Q14. Cross-Architecture Platonic Convergence: Do Mamba and Transformer Converge to the Same Representation?  (Category B)

**Problem.** The Platonic representation hypothesis was articulated for Transformers (and image encoders). Test on architectures with structurally different inductive biases: do Mamba, RWKV, KAN-based, and Transformer LLMs trained on the same corpus converge under CKA / mutual nearest-neighbors as size grows? Provide a *quantitative scaling law* for cross-architecture alignment as a function of parameter count, and identify the inductive-bias gap (e.g. attention sinks, recurrent state bottleneck) that persists asymptotically. Distinct from DR-001's Platonic problem #3, which targets falsifiability *within* a class.

**Importance.** Strong cross-architecture convergence would constitute the strongest evidence for a universal representation theory; persistent divergence would refute strong forms of the hypothesis.

**Progress/gap.** Platonic hypothesis tested almost exclusively across Transformer/ViT variants. Mamba is now mature enough (arXiv 2312.00752, 2406.16722, 2409.05395) to be included; the comparison has not been performed at scale.

**Difficulty/risk.** Medium. Mostly compute-bound; theoretical contribution comes from explaining the gap.

**Key arXiv IDs.** 2312.00752, 2409.05395, 2502.16282, 2502.14047, 2602.06205.

---

## Q15. Symmetry Discovery from Internal Representations: An Inverse Problem  (Category D)

**Problem.** Given the activations of a trained network on a dataset, can we *recover* the (Lie) symmetry group of the data-generating process? Formally: design an estimator that maps activation tensors to a Lie algebra basis with provable consistency under sample size and width. Connect to learned equivariance: does a non-equivariant Transformer trained on rotation-invariant data develop *implicit* representations of $SO(d)$ in a subset of its heads, and can this be detected from activations alone?

**Importance.** Provides a bridge between mechanistic interpretability (head-level analysis) and equivariant theory (Q8); could explain emergent symmetry use in foundation models.

**Progress/gap.** Symmetry discovery from *data* exists (LieGAN, arXiv 2302.00236; arXiv 2410.09841). Discovery from *internal activations* of standard architectures is nascent. SymPE (2503.21985) and arXiv 2505.13631 study symmetry breaking but not symmetry recovery.

**Difficulty/risk.** Hard. Lie-algebraic structure may be entangled with task structure; identifiability could fail.

**Key arXiv IDs.** 2302.00236, 2410.09841, 2408.00949, 2503.21985.

---

## Completion Table

| # | Problem | E1 Statement | E2 Importance | E3 Progress/Gap | E4 Difficulty | E5 arXiv IDs | Category |
|---|---|---|---|---|---|---|---|
| 1 | Score curvature & generalization | ✔ | ✔ | ✔ | Hard | 2505.11128, 2510.02305, 2603.20645 | C |
| 2 | Mamba vs Transformer capacity | ✔ | ✔ | ✔ | Hard | 2312.00752, 2409.05395 | D |
| 3 | KAN representation geometry | ✔ | ✔ | ✔ | Med-Hard | 2404.19756, 2410.08026 | D |
| 4 | RLHF representation drift | ✔ | ✔ | ✔ | Medium | 2312.15997, 2410.15610 | C |
| 5 | Bayesian ICL lower bound | ✔ | ✔ | ✔ | Hard | 2510.10981, 2306.04891 | C |
| 6 | Platonic heterogeneity no-free-lunch | ✔ | ✔ | ✔ | Medium | 2502.16282, 2511.12121 | A/B |
| 7 | Quantization critical bit-width | ✔ | ✔ | ✔ | Medium | 2307.02973, 2510.22058 | D |
| 8 | Equivariant capacity filtration | ✔ | ✔ | ✔ | Hard | 2408.00949, 2503.21985 | D |
| 9 | μP for Mamba/MoE/KAN | ✔ | ✔ | ✔ | Med-Hard | 2310.02244, 2508.09752 | C |
| 10 | Transcoder vs SAE separation | ✔ | ✔ | ✔ | Medium | 2406.11944, 2501.18823 | C |
| 11 | Manifold dictionary learning | ✔ | ✔ | ✔ | Hard | 2405.14860 | B |
| 12 | Continuous CoT superposition | ✔ | ✔ | ✔ | Hard | 2505.12514, 2509.23365 | D |
| 13 | Linear steering sufficiency | ✔ | ✔ | ✔ | Medium | 2502.17601, 2604.15557 | D |
| 14 | Cross-architecture Platonic | ✔ | ✔ | ✔ | Medium | 2312.00752, 2502.16282 | B |
| 15 | Symmetry discovery from activations | ✔ | ✔ | ✔ | Hard | 2302.00236, 2408.00949 | D |

**Non-duplication check.** Every problem above is distinct from the ten DR-001 open problems: Q3/Q9 concern KAN/Mamba/MoE which differ from DR-001 #4's scaling exponents and #1's superposition capacity; Q6/Q14 refine the Platonic hypothesis along different axes than #3; Q10/Q11 address SAE *replacements* rather than identifiability (#2); Q12 concerns continuous CoT, not Grokking on non-commutative groups (#5); Q1 concerns generalization via curvature, not intrinsic dimension causal chain (#9); Q5 concerns ICL Bayes-gap lower bounds, not functional-equivalence similarity theorems (#6); Q4/Q7/Q8/Q13/Q15 address topics DR-001 does not touch.

**Category coverage.** A: Q6. B: Q6, Q11, Q14. C: Q1, Q4, Q5, Q9, Q10. D: Q2, Q3, Q7, Q8, Q12, Q13, Q15. All four categories covered as required.

