### Impossibility theorems in neural networks

用户:


Claude:
his is a beautifully structured research agenda. The prompt is detailed enough that I can dive straight into research. Let me launch a comprehensive search across all six domains to find impossibility theorems and their potential analogs in neural network representation analysis.
our cross-domain impossibility theorem analysis for neural network representation analysis is ready. It systematically maps classical impossibility results from six fields—information theory, statistics, economics/social choice, computational complexity, topology/differential geometry, and measurement theory/physics—onto the interpretability setting. The report evaluates each theorem on novelty, feasibility, impact, synergy with worst-case/average-case framing, positivity, and testability, while also surveying known partial results (Bilodeau, Han, Sutter, Kleinberg). It concludes with five top recommended research directions, including Blackwell sufficiency ordering of interpretability methods, a Myerson-Satterthwaite four-axiom impossibility for mechanistic interpretability, Cramér-Rao bounds with James-Stein shrinkage for SAE estimators, communication complexity lower bounds for method ensembles, and a Bell-inequality-style test for the linear representation hypothesis.
# Cross-Domain Impossibility Theorem Analogs for Neural Network Representation Analysis: A Systematic Search

## Executive Summary and Direct Answer

This report systematically maps classical impossibility theorems from six fields onto the representation-analysis setting in which a map Φ goes from neural representations to interpretable descriptions (probing, sparse autoencoders, activation patching, CKA, causal abstraction, etc.). For each theorem we (a) restate it precisely, (b) check whether a representation-analysis analog has been formally proved (or even named), (c) cite the closest published work, and (d) score it on novelty, feasibility, impact, synergy with the existing worst-case-vs-average-case gap (exp-005), positivity (i.e., does relaxation give a constructive method?), and experimental testability.

The four "known partial results" are well-documented: **Bilodeau et al. (2024, PNAS)** proves that any complete-and-linear feature-attribution method (SHAP, IG) cannot beat random guessing on counterfactual end-tasks; **Han et al. (2022, NeurIPS)** proves a no-free-lunch theorem stating that any local-function-approximation explainer is optimal only on the neighborhood it was designed for; **Sutter, Minder, Hofmann & Pimentel (2025, NeurIPS)** prove the "non-linear representation dilemma": without a complexity constraint on the alignment map, any neural network is causally abstractable into any algorithm, trivializing mechanistic interpretability via causal abstraction; **Kleinberg (NeurIPS 2002)** proves no clustering function satisfies Scale-Invariance, Richness, and Consistency simultaneously. All four share an algebraic flavor — an axiom (additivity, completeness, linearity, refinement) that, when combined with another, collapses the method to triviality.

The most promising under-studied analogs, by my assessment, are:

1. **Blackwell-sufficiency ordering of representation-analysis methods** (Domain 1) — currently *no* paper formalizes probing ⪯ SAE ⪯ patching as a Blackwell order, despite all the surrounding machinery existing.
2. **A Cramér–Rao-style lower bound on representation-analysis estimators**, with Fisher information of the activation distribution as the sharpness limit (Domain 2) — unexplored in mechanistic interpretability.
3. **A Myerson–Satterthwaite four-axiom incompatibility** for {faithfulness, completeness, sparsity, causal-sufficiency} in SAEs / patching (Domain 3) — the natural next move after the additivity-based mini-impossibility you proved.
4. **A Borsuk–Ulam topological obstruction to dimensionality reduction of polysemantic features** (Domain 5) — Batson et al. proved an analog for autoencoders but not for SAEs/probes.
5. **A communication-complexity lower bound for combining two interpretability methods** to recover a full circuit (Domain 4) — directly motivated by your exp-005 worst-case-vs-average-case gap.

The rest of the report develops each domain in detail.

---

## DOMAIN 1: Information Theory

### 1.1 Rate–Distortion Theory

**Original theorem (Shannon, 1959).** For source X with distortion measure d, the minimum rate at which X can be coded so that expected distortion does not exceed D is R(D) = inf_{p(x̂|x): E[d(X,X̂)]≤D} I(X;X̂). The function R(D) is convex, non-increasing, and characterizes a fundamental tradeoff between code length and reconstruction error.

**Existing representation-analysis work.** The information bottleneck (Tishby–Pereira–Bialek 1999; Tishby–Zaslavsky 2015) is exactly rate–distortion with the distortion replaced by an "irrelevance" term I(X;T) - β I(T;Y). Subsequent work (Nguyen & Choi, "Markov Information Bottleneck", PMC 2020; "Bitwise Information Bottleneck" 2020) extends it through multi-bottleneck Markov chains; bifurcations of R(D) and successive-refinement variants are studied in the MDPI special issue *Theory and Application of the Information Bottleneck*. None of these, however, treat Φ (the interpretability map) as the encoder and "faithfulness loss" as distortion.

**Status of analog: PARTIALLY OPEN.** A rate–distortion view of Φ would say: any interpretability map producing a description of length R bits must incur a faithfulness distortion at least R⁻¹(R). This has not been formally stated for SAEs, probes, or patching, although Choi, Jung & Watanabe ("Understanding Probe Behaviors through Variational Bounds of Mutual Information", 2023) come close — they bound probe accuracy by I(activation; concept).

- **Novelty:** Moderate; latent in the IB literature but not crystallized for interpretability.
- **Feasibility:** High; just identify the right distortion measure (e.g., causal-effect-mismatch).
- **Impact:** High — would give an a priori limit on SAE width vs reconstruction fidelity.
- **Synergy with exp-005:** Yes — the worst-case/average-case gap is exactly the gap between R(D) and the operational rate.
- **Positive result?** Yes — R(D) is achievable, so the relaxation tells you the optimal SAE width.
- **Testable?** Yes — measure achieved-rate vs distortion on toy circuits where ground truth is known.

### 1.2 Data Processing Inequality (DPI)

**Original theorem.** If X → Y → Z is a Markov chain, then I(X;Z) ≤ I(X;Y). Post-processing cannot increase information.

**Existing representation-analysis work.** Alain & Bengio's classic probing paper (2016) explicitly invokes DPI to argue that deeper layers cannot have more label-information than earlier ones, yet linear probes do better deeper — they explain this apparent contradiction by computational accessibility. Ivanova, Hewitt & Zaslavsky ("Probing artificial neural networks: insights from neuroscience", 2021) discuss the "decoding from where it isn't" problem as a DPI consequence. Choi et al. (2023) develop variational bounds and analyze a "seemingly DPI-violating" convex layer-wise curve.

**Status of analog: WELL-COVERED but UNDER-FORMALIZED for Φ.** A direct DPI for the interpretability pipeline X → activations → Φ(activations) → behavioral prediction is not stated as a theorem; the conclusion would be that Φ's downstream-behavior-prediction performance is upper-bounded by I(activations; behavior).

- **Novelty:** Low; essentially folklore.
- **Feasibility:** Trivial.
- **Impact:** Low alone, but enables sharper statements (e.g., usable in proofs of other analogs).
- **Positive result?** No.

### 1.3 Channel Capacity Theorem

**Original theorem (Shannon).** A discrete memoryless channel with transition p(y|x) admits arbitrarily reliable transmission iff the rate R is below C = max_{p(x)} I(X;Y).

**Status: OPEN AND NOVEL.** Treating the model-to-interpretation pipeline as a noisy channel — input distribution = activations, channel = Φ, output = interpretable description — and asking the maximum rate at which interpretable hypotheses about the model can be reliably distinguished is, to my knowledge, not done. Nokleby–Rodrigues–Calderbank ("Discrimination on the Grassmann Manifold: Fundamental Limits of Subspace Classifiers", 2014) define a "classification capacity" for subspace discrimination using exactly this analogy but apply it to feature extraction, not interpretability.

- **Novelty:** High.
- **Feasibility:** Medium; the channel model must be carefully defined to be non-vacuous.
- **Impact:** Could quantify "how many distinct hypotheses about a layer's role can be reliably distinguished by patching".
- **Testable?** Yes — count distinguishable circuit hypotheses vs noise level introduced by activation patching.

### 1.4 Blackwell's Theorem / Blackwell Sufficiency

**Original theorem (Blackwell 1953, *Annals of Math. Stat.* 24:265–272).** Experiment A is more informative than experiment B (in every Bayes decision problem) iff B is obtainable from A by post-processing through a stochastic kernel (a "garbling"). Equivalent characterizations include convex dominance and Bayesian preference (Sufficiency in Blackwell's Theorem, *J. Math. Econ.*, 2003).

**Existing representation-analysis work.** Paxton's 2026 survey "The Theorems of Dr. David Blackwell and Their Contributions to Artificial Intelligence" (arXiv 2604.06621) reviews modern uses but does *not* apply Blackwell ordering to interpretability methods. No paper currently asks "is probing a garbling of activation patching?" or "is CKA a Blackwell-coarsening of SAE features?".

**Status: OPEN — and unusually clean target.** A Blackwell ordering on the family Φ would *uniquely* identify whether one method dominates another for *every* downstream interpretability task (concept localization, recourse, mechanism discovery), rather than for one particular benchmark.

- **Novelty:** Very high.
- **Feasibility:** Medium — the right "experiment" model needs to be cast (parameter = some property of the model/circuit; observation = the method's output).
- **Impact:** Very high — provides a principled answer to the perennial "which method should I use?" question (the question Lakkaraju et al. and Bilodeau et al. raise).
- **Synergy with exp-005:** Yes — Blackwell dominance is by definition worst-case across decision problems; the gap between it and average dominance maps directly onto worst-case vs typical-case interpretability gaps.
- **Positive result?** Yes — finds optimal (Blackwell-maximal) methods.
- **Testable?** Yes — empirically check whether outputs of method B can be reconstructed from outputs of method A via a learned stochastic kernel.

---

## DOMAIN 2: Statistics

### 2.1 Cramér–Rao Lower Bound

**Original theorem.** For an unbiased estimator θ̂ of a parameter θ with regular likelihood f(x;θ), Var(θ̂) ≥ I(θ)⁻¹, where I(θ) is the Fisher information.

**Existing representation-analysis work.** Generative CRB approaches (Habi, Messer et al., "Learning to Bound: A Generative Cramér-Rao Bound", arXiv 2203.03695; Starshynov et al., 2025, "Model-free estimation of the Cramér-Rao bound for deep-learning microscopy") apply CRB *outside* the interpretability setting, where the parameter is a physical quantity. Ben-Haim & Eldar's sparse-estimation CRB (arXiv 0905.4378) is the closest formal tool: the CRB equals the oracle estimator's MSE almost everywhere. None applies CRB to "parameters" that are circuit identities, attribution scores, or feature-direction angles in an SAE.

**Status: OPEN, HIGHLY NOVEL.** Treat the interpretability output (e.g., the SAE feature directions, the probing weight vector, the patching effect estimate) as an unbiased estimator of a latent "true" representation parameter, then derive CRB from the Fisher information of the activation distribution. James–Stein then implies these estimators are *inadmissible* in d ≥ 3 — a striking claim for high-dimensional SAEs.

- **Novelty:** Very high.
- **Feasibility:** Medium — requires a probabilistic model of activations, which exists (e.g., Gaussian linear-superposition models).
- **Impact:** Could yield concrete variance lower bounds for probing and SAE estimates, plus a principled shrinkage estimator.
- **Synergy with exp-005:** Strong — CRB is achieved asymptotically by ML but typically not finite-sample; this is exactly the worst-case/typical-case framing.
- **Positive result?** Yes — efficient estimators exist and shrinkage improves them.
- **Testable?** Yes — bootstrapped variance of SAE feature directions can be compared to CRB.

### 2.2 Rao–Blackwell Theorem

**Original theorem.** If T is sufficient for θ and δ is any unbiased estimator, then E[δ|T] is unbiased with variance ≤ Var(δ).

**Existing representation-analysis work.** No paper, to my knowledge, formalizes a "minimal sufficient activation" — i.e., a coarsening of the residual stream that preserves all behaviorally relevant information and on which conditioning improves every interpretability estimator. This is the natural formal correlate of the "minimal description" goal of mechanistic interpretability. Achille & Soatto's "Information Dropout" / "Emergence of Invariance" work is the closest in spirit but does not call out Rao–Blackwellization explicitly.

**Status: OPEN, HIGH NOVELTY.**

- **Novelty:** High.
- **Feasibility:** Medium — define a representation σ-algebra and ask for sufficiency w.r.t. behavior distribution.
- **Impact:** Could yield a canonical interpretable representation (Blackwell-Rao-optimal SAE).
- **Positive result?** Yes — produces an improved estimator.
- **Testable?** Yes — compare variance of patching effects before/after conditioning on candidate sufficient statistics.

### 2.3 Lehmann–Scheffé / Completeness

**Original theorem.** If T is complete sufficient for θ, then any unbiased estimator that is a function of T is the unique UMVUE.

**Status: OPEN.** No paper formalizes completeness of an internal representation. If completeness can be shown, this would give a *uniqueness* result for the optimal interpretability method — a strong positive complement to Sutter et al.'s impossibility.

### 2.4 James–Stein Paradox

**Original result.** For estimating the mean μ ∈ ℝ^d of N(μ,I) with d ≥ 3, the MLE X is inadmissible: the shrinkage estimator (1 − (d−2)/‖X‖²)X dominates it in MSE.

**Existing representation-analysis work.** None applies this directly to interpretability methods.

**Status: OPEN, COMPELLING.** Claim: every naïve interpretability estimator (per-feature attribution, per-neuron probing, per-direction SAE coefficient) in high dimension is *inadmissible*, and shrinkage / hierarchical pooling strictly improves it. This is unusual because it predicts a *positive result*: a concrete improved estimator exists and should be derived.

- **Novelty:** Very high.
- **Feasibility:** Medium — requires a Gaussian (or sub-Gaussian) noise model.
- **Impact:** Practical — gives the "denoised SAE" or "shrinkage probe" with provably lower MSE.
- **Testable?** Yes, immediately.

---

## DOMAIN 3: Economics / Social Choice

### 3.1 Gibbard–Satterthwaite

**Original theorem.** Any non-dictatorial deterministic voting rule over ≥3 alternatives is manipulable: some voter can misreport preferences to obtain a better outcome.

**Existing representation-analysis work.** Bilodeau et al.'s adversarial constructions (uncountably many models share the same attribution but differ in counterfactual behavior) are *de facto* manipulation: the "model" plays the role of the manipulating voter. This connection is, however, not framed as a Gibbard–Satterthwaite analog. Qiu's "Representative Social Choice: From Learning Theory to AI Alignment" (arXiv 2410.23953) proves Arrow- and Gibbard-Satterthwaite-like results for *AI alignment* preference aggregation, not for representation analysis.

**Status: OPEN.** A formal analog would say: any non-trivial interpretability method Φ admits an adversarial model that "mis-reports" its internals (by reparameterization preserving function) to change Φ's output. Bilodeau's result is essentially a proof of this in the complete-and-linear case.

- **Novelty:** Medium (Bilodeau covers a special case, but the general theorem is unproved).
- **Feasibility:** Medium-high.
- **Impact:** Would unify Bilodeau, Han, and Sutter under one framework.
- **Synergy with exp-005:** Strong — gives a sharper worst-case lower bound.
- **Testable?** Yes — construct adversarial reparameterizations and measure Φ shift.

### 3.2 Myerson–Satterthwaite

**Original theorem.** In bilateral trade with independent private valuations, no mechanism simultaneously achieves efficiency, budget balance, individual rationality, and (Bayesian) incentive compatibility.

**Status: OPEN — TOP RECOMMENDATION.** A four-axiom analog for representation analysis: {faithfulness (Φ(model) reflects model behavior), completeness (every relevant feature is reported), sparsity (a small set of features suffices), causal sufficiency (intervening on a reported feature changes behavior in the claimed way)} are jointly inconsistent. This is the natural generalization of your additivity-based mini-impossibility. The strength of M–S is that the four axioms feel individually unobjectionable, mirroring the goal of SAE practitioners.

- **Novelty:** Very high.
- **Feasibility:** Medium — requires careful axiomatization.
- **Impact:** Very high — would supersede Bilodeau in scope.
- **Positive result?** Relaxing any one axiom gives a constructive class of methods (analogous to second-best mechanisms).
- **Testable?** Indirectly — check whether real SAEs trade off the four axioms predictably.

### 3.3 Sen's Impossibility of a Paretian Liberal

**Original theorem.** No social choice function satisfies Unrestricted Domain, Weak Pareto, and Minimal Liberalism (each of ≥2 individuals is decisive over at least one pair of alternatives) simultaneously.

**Status: OPEN, INTERESTING.** Analog: no Φ simultaneously allows each interpretability method "decisive power" over its own preferred axis (e.g., probes own concept directions, SAEs own feature directions, patching owns causal effects) while also remaining Pareto-improving across global benchmarks. Could explain the "method-dependence" of mechanistic interpretability results.

- **Novelty:** High.
- **Feasibility:** Medium.
- **Impact:** Moderate; would explain ecosystem fragmentation.

### 3.4 VCG / Positive Mechanism Design

**Original result.** The Vickrey–Clarke–Groves mechanism is efficient and truth-revealing if utilities are quasi-linear (i.e., relaxing ordinal preferences to cardinal money-equivalents restores possibility from Arrow).

**Status: OPEN, IMPORTANT POSITIVE DIRECTION.** Identify which axiom relaxation in interpretability restores possibility. Sutter et al.'s answer is "constrain map complexity"; Bilodeau's implicit answer is "drop linearity". A VCG-style answer might be "allow continuous-valued faithfulness scores and use a quasi-linear utility": the optimal Φ exists and is essentially unique.

- **Novelty:** High.
- **Feasibility:** Medium-high (constructive).
- **Impact:** Very high — first positive theorem in the family.

---

## DOMAIN 4: Computational Complexity

### 4.1 PCP Theorem / Inapproximability

**Original theorem (Arora–Safra; Arora–Lund–Motwani–Sudan–Szegedy, 1992–98).** NP = PCP(O(log n), O(1)); equivalently, MAX-3SAT is NP-hard to approximate within 7/8 + ε.

**Existing representation-analysis work.** Dictionary learning is *known* to be NP-hard (Tillmann's "Computational Intractability of Dictionary Learning for Sparse Representation", arXiv 1511.01776; Rambhatla, Tropp et al.'s "Sparse Coding and Autoencoders", arXiv 1708.03735, gives partial recovery guarantees under genericity). Anthropic's *Towards Monosemanticity* explicitly notes that "Exact compressed sensing is NP-hard". Gated-SAE work (DeepMind, arXiv 2404.16014) observes the L0-sparsity problem "exactly is equivalent to integer linear programming, and is NP hard". Switch-SAE (Mudide et al., arXiv 2410.08201) addresses this empirically.

**Status: PARTIALLY OPEN.** NP-hardness is established; *inapproximability* — i.e., a PCP-style claim that no polynomial algorithm can approximate the optimal SAE reconstruction to within a (1+ε) factor — is not. Establishing this would give a tight worst-case lower bound for SAE quality.

- **Novelty:** High.
- **Feasibility:** Medium — reductions from known hard problems (label cover, 3-SAT) are conceivable.
- **Impact:** Very high — explains why SAE benchmark gaps persist.
- **Synergy with exp-005:** Strong — worst-case hardness vs average-case (smoothed-analysis) tractability is exactly this kind of gap.
- **Positive result?** Approximation algorithms with explicit gaps.
- **Testable?** Algorithmically.

### 4.2 Communication Complexity Lower Bounds

**Original theorem (Yao 1979).** For functions like EQUALITY, DISJOINTNESS, INNER-PRODUCT-MOD-2, two parties holding x and y need Ω(n) bits of communication to compute f(x,y).

**Status: OPEN, HIGHLY PROMISING.** Treat each interpretability method as a party holding a partial view of the model. How many bits must they exchange to recover a complete circuit description? Lower bounds would give a fundamental limit on "interpretability ensembles" — and the analog is unusually well-motivated by the empirical observation that no single method (probing, SAE, patching) captures everything.

- **Novelty:** Very high.
- **Feasibility:** Medium — requires modelling the partial-view assumption precisely.
- **Impact:** High — could justify (or refute) the multi-method ensemble paradigm of mechanistic interpretability.
- **Synergy with exp-005:** Yes — distinguishes worst-case (adversarial f) from typical-case (structured f).
- **Testable?** Yes — empirically measure mutual information between method outputs.

### 4.3 Property Testing

**Original framework (Goldreich–Goldwasser–Ron 1998).** Distinguish, with sublinear queries, between functions satisfying property P and functions ε-far from P.

**Existing representation-analysis work.** Czumaj & Sohler ("Property Testing of Computational Networks", arXiv 2512.07577) recently initiated property testing of neural networks under a weight-perturbation distance. They show that *near-constant functions are testable in query complexity independent of network size*, and prove negative results in distribution-free models. This work does *not* yet treat properties like "this layer linearly represents the concept C" or "this circuit implements the IOI algorithm".

**Status: OPEN, EMERGING.** Defining mechanistic-interpretability properties (linear-representability of a concept, causal-mediation by a layer, monosemanticity of an SAE feature) as testable predicates is novel.

- **Novelty:** High; aligned with brand-new work.
- **Feasibility:** Medium-high; the Czumaj–Sohler framework is reusable.
- **Impact:** Could yield extremely cheap "is this hypothesis worth investigating?" pre-tests for circuits.
- **Testable?** Yes by construction.

---

## DOMAIN 5: Topology / Differential Geometry

### 5.1 Borsuk–Ulam

**Original theorem.** Every continuous map f: S^n → ℝ^n identifies some pair of antipodal points: f(x) = f(−x).

**Existing representation-analysis work.** Batson, Haaf, Kahn & Roberts' "Topological obstructions to autoencoding" (*JHEP* 2021/4) is the foundational paper: they prove that some manifolds *cannot* be autoencoded into ℝ^k for any k below an obstruction dimension. Diepeveen et al.'s "Topological Obstructions and How to Avoid Them" (arXiv 2312.07529) and Barannikov et al.'s "Representation Topology Divergence" (ICML 2022) extend this. Dimensionality-reduction width-of-deep-networks work (arXiv 2511.06821) gives Borsuk-Ulam-style obstructions to dimensionality reduction maps.

**Status: PARTIALLY OPEN, RIPE.** The Borsuk–Ulam consequence specifically for *polysemantic feature decomposition* (i.e., SAE features as a map S^d → ℝ^k must collapse antipodal — interpretively opposite — directions whenever k < d) is not stated. This would directly explain SAE feature splitting and dead features.

- **Novelty:** High; partial results exist but not for SAEs/probes.
- **Feasibility:** Medium-high; tools from algebraic topology are standard.
- **Impact:** High — predicts feature-collapse failure modes.
- **Synergy with exp-005:** Topological obstructions are inherently worst-case; quantifying their typical-case prevalence (probabilistic Borsuk–Ulam) is open.
- **Testable?** Yes — search for "antipodal" feature pairs that collapse under SAE.

### 5.2 Nash Embedding

**Original theorem.** Every Riemannian k-manifold embeds isometrically in ℝ^N for N = O(k²) (C^∞ Nash 1956) or N = k + 1 + …(C¹ version).

**Status: OPEN.** The corresponding interpretability statement — "to faithfully represent the d-dimensional task manifold, the SAE must use at least f(d) features" — has not been proved. Park, Choe & Veitch's "linear representation hypothesis" gives empirical evidence; an isometric-embedding bound would be theoretical.

- **Novelty:** High.
- **Feasibility:** Medium.
- **Impact:** Direct guidance for SAE width selection.

### 5.3 Fixed-Point Theorems

**Original.** Brouwer: every continuous self-map of a compact convex set has a fixed point. Kakutani: extension to set-valued correspondences.

**Status: OPEN.** Iterative interpretability pipelines (interpret → modify → re-interpret, or SAE-train → discover-feature → improve-SAE) are naturally cast as self-maps; Brouwer guarantees a fixed point — a "stable interpretation". The unstudied questions are uniqueness (Banach), stability (hyperbolicity), and *rate* of convergence (analog of policy-iteration convergence).

- **Novelty:** Medium.
- **Feasibility:** Medium.
- **Impact:** Moderate; pedagogically clean but practical implications unclear.

---

## DOMAIN 6: Measurement Theory / Physics

### 6.1 Heisenberg Uncertainty

**Original.** For conjugate observables Â, B̂, ΔA·ΔB ≥ ½|⟨[Â,B̂]⟩|. Classically, for f and its Fourier transform, σ_x σ_p ≥ ½.

**Existing representation-analysis work.** Zhang et al. ("On the uncertainty principle of neural networks", arXiv 2205.01493; *iScience* 2025) prove an uncertainty principle relating a network's feature precision to gradient precision and tie it to the accuracy-vs-robustness tradeoff. Zhang et al. (2024, *Sci. Reports*) extend this to binary classification with conjugate variables (x, p = gradient). These results apply to *trained models*, not to interpretability methods.

**Status: PARTIALLY OPEN.** A direct analog for representation analysis — that "local feature precision" (e.g., per-token activation patching effect) and "global structural precision" (e.g., circuit-level identification via probing) cannot both be sharp — has not been written down. The conjugate observables would be circuit-locality and circuit-completeness.

- **Novelty:** High.
- **Feasibility:** Medium; needs careful identification of the commuting/non-commuting operator pair.
- **Impact:** High — would explain why patching gives crisp local answers but vague global ones (and vice versa for SAEs).
- **Synergy with exp-005:** Yes — uncertainty bounds are tight worst-case but not typical-case.
- **Testable?** Yes — explicitly measure tradeoff curve.

### 6.2 Observer Effect

**Original (operational).** Measurement disturbs the system; quantum non-demolition or weak-measurement frameworks formalize the disturbance.

**Existing representation-analysis work.** Activation patching is *literally* an interventionist measurement and is known to disturb downstream computation via self-repair (Heimersheim & Nanda, arXiv 2404.15255; "Towards Best Practices of Activation Patching", arXiv 2309.16042). Makelov, Lange & Nanda's "Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching" (arXiv 2311.17030) is essentially an observer-effect result: subspace interventions appear to change the feature but actually activate a dormant parallel pathway. This is a known but unquantified phenomenon.

**Status: PARTIALLY OPEN.** What is missing is a *quantitative* observer-effect inequality: ΔBehavior ≥ g(ΔInternalState), giving a fundamental floor on how much an interpretability intervention must perturb the model.

- **Novelty:** Medium (illustrated, not quantified).
- **Feasibility:** High.
- **Impact:** High — would give a principled "minimum-intervention" patching protocol.
- **Testable?** Yes.

### 6.3 Bell Inequalities

**Original.** Local hidden-variable theories must satisfy CHSH ≤ 2; quantum mechanics violates this, achieving 2√2.

**Existing representation-analysis work.** Sutter et al.'s trivialization result is exactly Bell-like in spirit: linear (local) alignment maps cannot explain certain causal-abstraction correlations, but non-linear maps can — and unrestrictedly so they explain *too much*. No paper explicitly imports Bell's apparatus.

**Status: OPEN.** A formal Bell-style inequality would say: if a representation Φ admits a *local linear hidden-variable model* (i.e., each feature acts independently and linearly), then certain measurable correlation statistics across activations are bounded by 2; if non-linear interactions exist, the bound is violated. This would make the linear-representation hypothesis *empirically falsifiable* via a CHSH-style experiment.

- **Novelty:** Very high.
- **Feasibility:** Medium.
- **Impact:** Could decisively settle linear-representation-hypothesis debates.
- **Synergy with exp-005:** Yes — Bell violations are worst-case results, but real systems may sit close to the bound on average.
- **Testable?** Yes — and this is the unique theorem on this list with a *direct empirical test* analogous to CHSH.

---

## Cross-Cutting Observations and Related Literature

Several lines of work cross multiple domains and deserve flagging as background for any new paper:

1. **Probing classifiers are unreliable** (Zhou et al., NeurIPS 2022, "Probing Classifiers are Unreliable for Concept Removal and Detection"; Belinkov 2022 squib in *Computational Linguistics*): formal limitations of probes that overlap Domains 1, 2, and 6.

2. **Function-approximation unification** (Han, Srinivas & Lakkaraju, "Which Explanation Should I Choose?", NeurIPS 2022): the no-free-lunch theorem that gave the "Han impossibility" referenced in your context. Their statement: "no single method can perform optimally across all neighborhoods" — a Domain-2/3 hybrid.

3. **Biologically informed neural networks (BINNs)** (bioRxiv 2026, "The illusion of interpretability in biologically informed neural networks"): nonidentifiability result showing that even teacher–student matched architectures fail to recover the right gene-to-pathway weights. Domain 2 (statistical identifiability).

4. **"On the Mathematical Impossibility of Safe Universal Approximators"** (arXiv 2507.03031): catastrophe-density results that *automatically inherit* via universal approximation. Useful precedent for the "automatic inheritance" structure your additivity-based proof exploits.

5. **CKA criticisms** (Davari et al. 2022; Murphy et al. 2024; Chun et al. 2025): targeted attacks can arbitrarily modulate CKA without changing network behavior — a Gibbard–Satterthwaite-style manipulability result in disguise.

6. **Interpretability illusions in subspace patching** (Makelov, Lange, Nanda, ICLR 2024): dormant parallel pathways are activated by interventions, a concrete Domain-6 observer-effect phenomenon.

---

## Top 5 Recommended Research Directions

Based on the criteria of novelty, feasibility, impact, synergy with worst-case/average-case framing, positive-result potential, and experimental testability, the five most promising open problems are:

### Recommendation 1 — Blackwell Sufficiency Ordering of Representation-Analysis Methods (Domain 1.4)

Prove (or disprove) that {linear probe} ⪯_Blackwell {sparse autoencoder} ⪯_Blackwell {activation patching} as experiments parametrized by the latent "circuit". A positive Blackwell dominance gives an objective ranking; a non-dominance result identifies fundamentally complementary methods and gives a basis for principled ensembling. This addresses the deepest open question in the field: which method should I use? Combine with exp-005 to study how the Blackwell partial order degrades from worst-case to typical-case input distributions.

### Recommendation 2 — Myerson–Satterthwaite Four-Axiom Impossibility for Mechanistic Interpretability (Domain 3.2)

Identify four individually compelling axioms — e.g., {faithfulness, completeness, sparsity, causal sufficiency} — and prove they are jointly unsatisfiable. This is the natural generalization of your additivity-based mini-impossibility and directly extends Bilodeau (which used two axioms: complete + linear). The proof technique would likely be a measure-theoretic separation argument similar to M–S's. Positive corollary: a "second-best" interpretability method emerges from relaxing any one axiom.

### Recommendation 3 — Cramér–Rao + James–Stein for Interpretability Estimators (Domain 2.1, 2.4)

Model SAE feature coefficients and probe weights as estimators of latent population quantities under a Gaussian (or sub-Gaussian) activation model; derive Fisher-information lower bounds; prove James–Stein inadmissibility in d ≥ 3 (always true in practice); construct an explicit shrinkage estimator that strictly dominates. This is unique on the list in delivering a *constructive method improvement*. Trivially testable: bootstrap variance vs CRB on toy models with known circuits.

### Recommendation 4 — Communication Complexity of Method Ensembles (Domain 4.2)

Cast each interpretability method as a party with a partial "view" of the network and ask how many bits two methods must exchange to certify a circuit hypothesis. Lower bounds (e.g., via DISJOINTNESS-style reductions) would prove that no pair of common methods suffices for full mechanistic explanation, justifying or refuting multi-method paradigms. Strong synergy with exp-005: worst-case (adversarial models) requires Ω(n) bits, while average-case (structured models) admits sublinear protocols.

### Recommendation 5 — Bell-Inequality-Style Test for the Linear Representation Hypothesis (Domain 6.3)

Construct a CHSH-style protocol over four interpretability "measurements" on activation pairs; prove that any model whose features are encoded by a local linear hidden-variable structure must satisfy a 2-bound; identify activations achieving 2√2-style violations. This is the only proposal yielding an experimentally falsifiable test of the linear representation hypothesis that is structurally distinct from existing benchmarks. Directly engages Sutter et al.'s non-linear representation dilemma by giving a principled way to declare a representation "non-linear enough to require non-linear alignment maps."

### Secondary Picks (Worth Pursuing After the Top 5)

- **Borsuk-Ulam topological obstruction for SAE feature decomposition** (Domain 5.1) — natural extension of Batson et al. 2021.
- **PCP-style inapproximability for SAEs** (Domain 4.1) — completes the NP-hardness picture.
- **Rao–Blackwell minimal sufficient activation** (Domain 2.2) — would justify the search for a canonical SAE.
- **Property testing of mechanistic-interpretability properties** (Domain 4.3) — leverages the brand-new Czumaj–Sohler 2025 framework.

### Common Synergy Pattern

Four of the top five recommendations share a key structural property aligned with your exp-005 finding: the impossibility holds in the *worst-case* sense, but admits a *typical-case* relaxation when the underlying distribution of activations is structured (low-rank, sparse, smooth, or generated by a known prior). This is exactly the worst-case-vs-average-case gap; making it quantitative would convert each impossibility theorem into a *spectrum* of impossibility-to-tractability, which is a far richer object than the binary "possible/impossible" framing of the existing four results (Bilodeau, Han, Sutter, Kleinberg). I believe this hybrid worst-case/typical-case impossibility is the largest unclaimed conceptual territory in the field, and any of the five recommendations above could serve as the vehicle for staking it out.

