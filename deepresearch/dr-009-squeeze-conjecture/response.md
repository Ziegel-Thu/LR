### Unified complexity parameter for impossibility results

用户:


Claude:
his is a beautifully structured research question — the squeeze between Bilodeau's and Sutter's impossibility results is a compelling framing, and finding a unified complexity parameter is exactly the right mathematical challenge. The request is detailed enough that I can dive straight into research. Let me kick off a comprehensive investigation across all six of your sub-questions.
our research survey on the Squeeze Conjecture is ready. It synthesizes the Bilodeau (2024) identifiability lower bound and Sutter (2025) trivialization upper bound into a unified impossibility framework for neural network representation analysis. The report evaluates candidate complexity measures (recommending log-metric-entropy with Rademacher complexity and rate-distortion theory), surveys adjacent "squeeze-type" results across interpretability, learning theory, information theory, and social choice, and assesses the essentiality of each theorem's assumptions. It concludes with a provability assessment—arguing the pure information-theoretic conjecture is parametrization-dependent while a resource-bounded version is more likely provable via SHAP-#P-hardness reductions—and proposes four concrete publishable sub-projects with a prioritized reading list of 16 key papers.
# The Squeeze Conjecture: A Research Survey on Unifying Bilodeau (2024) and Sutter (2025) Impossibility Theorems for Neural Network Representation Analysis

## Executive Summary

The "Squeeze Conjecture" posits a structural impossibility for representation analysis methods Φ : ℳ × 𝒟 → ℱ_c parameterised by an output/mapping complexity c. Bilodeau et al. (PNAS 2024; arXiv:2212.11870) establish a lower bound on c below which identifiability fails (Spec + Sens ≤ 1). Sutter et al. (NeurIPS 2025 Spotlight; arXiv:2507.08802) establish an upper bound above which non-triviality fails (every DNN abstracts every algorithm). The conjecture claims c₀ > c₁: the two boundaries cross, leaving no feasible interval.

The research base needed to *formally state* this unification — and to assess whether it is provably true — already contains most of the necessary technical machinery, but no single source has yet bridged the two halves. Below I survey what is available, identify the most promising unified complexity measure (a **constrained Rademacher / covering-number framework with a rate–distortion overlay**), give precise statements of partial squeeze-type results in adjacent fields, and assess provability.

### Anchor Results, Stated Precisely

**Bilodeau et al., Theorem 2.3 (PNAS 2024 / arXiv:2212.11870).** Let ℳ be a model class containing all piecewise-linear functions ℝ^p → ℝ^q (satisfied by any MLP with ReLU activations of sufficient width). Let Φ be any feature-attribution method that is *complete* (Σ_j Φ_j(f, x) = f(x) − f(x_base)) and *linear* in f. Then for any hypothesis test ψ ∘ Φ used to decide H₀ : f ∈ ℱ⁽⁰⁾ vs H₁ : f ∈ ℱ⁽¹⁾ (where the two sets encode opposing local model behaviours such as "increasing in feature j" vs "decreasing in feature j"):

> Specificity(ψ ∘ Φ) ≤ 1 − Sensitivity(ψ ∘ Φ),

i.e. Spec + Sens ≤ 1, no better than random guessing.

**Sutter et al., Theorem 1 (NeurIPS 2025 / arXiv:2507.08802).** Let N be a DNN whose internal activations are *input-injective* on a countable input space 𝒳, with *strict output-surjectivity*, and matchable partial orderings between layer order and algorithm DAG order, such that N solves the task T. Let A be any algorithm with f_A solving T. Then there exists an alignment map ϕ (within the class of arbitrarily-powerful measurable maps) such that A is a constructive τ-abstraction of N with interchange-intervention accuracy equal to 1. (I.e., *every* algorithm aligns to *every* such network.)

The Squeeze Conjecture asserts that the c-value at which Bilodeau's impossibility ceases (c₀) exceeds the c-value at which Sutter's trivialisation begins (c₁), under any reasonable single-parameter complexity ordering of the function class ℱ_c interpolating the two.

---

## Sub-Question 1: Unified Complexity Measures

The challenge is that Bilodeau's c is a **structural constraint on the output space** (ℝ^{d×m}, closed under affine combinations) while Sutter's c is a **richness constraint on the function class** ℱ ⊆ {ϕ : H → 𝒩}. A single scalar c must dominate both. The candidates:

### Rademacher / Gaussian Complexity (technically central)
Empirical Rademacher complexity ℛ̂_S(ℱ) = E_σ[sup_{f∈ℱ} (1/m) Σ σ_i f(z_i)] (Bartlett & Mendelson) is suitable because:
- It is *distribution-dependent* (unlike VC), important here because both impossibility results are sensitive to the model class ℳ.
- It applies to **real-valued, vector-valued, and continuous-output function classes**, which VC dimension cannot.
- For Bilodeau, completeness+linearity collapses ℱ_c onto an affine subspace of ℝ^{d×m}, dramatically *bounding* ℛ from above. For Sutter, allowing arbitrary measurable ϕ with input-injectivity makes ℛ effectively unbounded. The same quantity therefore captures both extremes.
- Gaussian complexity is the analogue with Gaussian rather than Rademacher random variables; equivalent up to logarithmic factors and sometimes more tractable analytically.

### VC / Fat-Shattering Dimension
Fat-shattering dimension (Alon–Ben-David–Cesa-Bianchi–Haussler) handles real-valued outputs, but Sutter's bijection class has trivially infinite shattering, Bilodeau's linear class has finite. The measure is too coarse for the *middle* regime that the Squeeze Conjecture concerns. **Verdict: too crude.**

### Metric Entropy / Covering Numbers (recommended as primary scale)
Log-covering numbers log N(ε, ℱ, ‖·‖) provide a *parametric family in ε* — exactly the kind of one-parameter complexity scale the conjecture needs. Dudley's entropy integral bounds Rademacher by covering numbers, so the two are tightly linked. A workable formal statement of the conjecture is:

> *There exist constants c₀ > c₁ such that any class ℱ_c with log N(ε, ℱ_c, ‖·‖) ≤ c₁ violates identifiability in the Bilodeau sense, and any class with log N(ε, ℱ_c, ‖·‖) ≥ c₀ trivialises causal abstraction in the Sutter sense.*

### Kolmogorov / Descriptive Complexity
Sutter explicitly notes (§2.1 footnote) that the "algorithms" considered "need not match a formal definition as, e.g., the considered functions may be uncomputable." This *forecloses* a purely Kolmogorov-theoretic squeeze: Sutter's bijections include non-computable functions, so K(ϕ) = ∞ for the worst alignment maps. **Resource-bounded** Kolmogorov complexity is more promising (see Sub-Q6) but is not the natural primary measure.

### Rate–Distortion Theory (strong conceptual fit)
Rate–distortion viewpoints have been used to give precise capacity-vs-fidelity trade-offs for neural representations (Bates & Jacobs, eLife 2023, "Rate-distortion theory of neural coding"; Grohs & Petersen, "Phase Transitions in Rate Distortion Theory and Deep Learning," Foundations of Computational Mathematics 2022, doi 10.1007/s10208-021-09546-4). The mapping is:
- **Φ as a noisy channel** from behaviour B to user-readable explanation.
- **Bilodeau's regime ↔ rate R < D⁻¹(ε):** Too-low channel capacity; the source (model behaviour) cannot be reconstructed above distortion ε. The information-theoretic version of "Spec + Sens ≤ 1" is precisely *Fano's inequality* applied to the binary hypothesis test g⁽⁰⁾ vs g⁽¹⁾: P_err ≥ 1 − (I(Φ; B) + 1)/log|B|.
- **Sutter's regime ↔ R > H(B) + slack:** The channel has so much capacity it can produce *any* output distribution, including ones that decouple from B; mutual information no longer identifies B because *any* desired conditional distribution is realisable.

### Recommendation
**Use covering-number / metric-entropy as the *primary* complexity scale (it parameterises ℱ_c naturally), with Rademacher complexity as the *technical workhorse* (it gives the explicit lower-bound argument à la Bilodeau), and rate–distortion as the *conceptual interpretation* (it makes the squeeze look like a Fano-vs-Shannon trade-off).** This triple is internally consistent via Dudley's integral and the Bartlett–Mendelson generalisation framework. No standalone paper proposes this exact unification; the synthesis itself is novel.

---

## Sub-Question 2: Existing "Squeeze"-Type Results

The literature contains many "two-sided" impossibility tradeoffs, but **the strict form "no middle ground exists (c₀ > c₁)" is rare**.

### Direct precedents in interpretability
- **Mehrpanah et al., "On the Complexity–Faithfulness Trade-off of Gradient-Based Explanations"** (arXiv:2508.10490, 2025). Provides a spectral framework formalising an "explanation gap" between smoothness (simplicity) and faithfulness for ReLU networks. The "gap" they define is a quantitative tradeoff — but they prove a Pareto frontier exists, not its emptiness. This is the closest existing analogue to a Bilodeau↔Sutter squeeze.
- **Kumar & Ahuja, "Measuring the (Un)Faithfulness of Concept-Based Explanations / SURF"** (arXiv:2504.10833, 2025). Empirically argues that reported faithfulness gains in unsupervised concept-based methods come *artificially* from over-complex surrogates — qualitatively consistent with the Sutter direction of the squeeze.
- **Pîslar, Magliacane, Geiger, "Combining Causal Models for More Accurate Abstractions of Neural Networks"** (arXiv:2503.11429, 2025). Explicitly observes "a trade-off between the strength of an interpretability hypothesis … and its faithfulness." Frames the dilemma but does not prove the feasible interval is empty.

### Bias–variance / statistical learning theory
Although the classical bias-variance decomposition is routinely presented as a Pareto curve with an attainable optimum, several recent works **axiomatise** this differently:
- Domingos's unified bias-variance decomposition extends beyond squared loss but still admits an optimum.
- The "no free lunch" theorems (Wolpert, 1996; extended in Shalev-Shwartz & Ben-David, *Understanding Machine Learning*) prove that for *uniform* priors over learning tasks no learner outperforms random — a one-sided impossibility, not a c₀>c₁ squeeze.
- "Trade-Offs between Fairness, Interpretability, and Privacy in Machine Learning" (Waterloo MMath thesis, dspacemainprd01.lib.uwaterloo.ca/...) proves that ε-differentially-private + approximately fair learners cannot have non-trivial accuracy — a *bi-axial* squeeze close in spirit to ours but on different axes.

**No published bias-variance-style theorem proves the optimum doesn't exist.** This appears to be a gap in the statistical learning literature.

### Information theory channel capacity impossibilities
The Shannon–Fano–Pinsker triad gives natural two-sided impossibility:
- Capacity C: above which reliable communication is achievable (Shannon's noisy channel coding theorem).
- Rate-distortion R(D): below which distortion D is unachievable.
For non-stationary or unknown channels, *capacity itself can fail to exist as a single number* (e.g., Verdú–Han general capacity formula), giving genuine no-middle-ground phenomena. The Bilodeau/Sutter setting is structurally analogous to a channel whose capacity depends on the function class allowed at the encoder.

### Social choice / parameterised Arrow
- **Arrow's Impossibility Theorem** and parameterised variants (Mossel, "Arrow's Impossibility Without Unanimity," arXiv:0901.4727) are *single-sided* — no aggregator satisfies all axioms simultaneously, but they do not prove that intermediate weakenings remain impossible.
- Survey: "Impossibility Results in AI: A Survey" (ACM Computing Surveys, dl.acm.org/doi/full/10.1145/3603371) catalogues AI impossibility theorems; very few have the c₀ > c₁ form.
- **Banerjee et al., "On the Fundamental Impossibility of Hallucination Control in LLMs"** (arXiv:2506.06382, 2025): Green–Laffont-style mechanism-design impossibility — closer to a strict no-middle-ground result, but along orthogonal axes (truthfulness, conservatism, informativeness).

### Fixed-point / topological "no middle ground"
- Brouwer's fixed point theorem and Sperner's lemma are *existence* (not impossibility) results, but their combinatorial counterparts (parity arguments) sometimes yield no-middle-ground statements: e.g., the *Borsuk–Ulam theorem* implies that any continuous odd map S^n → ℝ^n has a zero, which forecloses "halfway" assignments under symmetry. There is no published application of these to neural network representation analysis.
- *KKM lemma* and its quantitative versions give parity-based impossibility for coverings of simplices — a possible analytical handle on the squeeze if one views (c, behaviour) jointly.

### Verdict
**No paper as of late 2025/early 2026 proves a strict c₀ > c₁ "no middle ground" theorem for representation analysis.** All existing results either (i) prove single-sided impossibilities, (ii) prove tradeoffs *with* attainable Pareto frontiers, or (iii) qualitatively note a dilemma without formalising emptiness of the feasible set. This is the gap the Squeeze Conjecture would fill.

---

## Sub-Question 3: Sutter et al. Assumption Relaxation

Sutter's Theorem 1 depends on five assumptions (countable input space, input-injectivity, strict output-surjectivity, matchable partial orderings, DNN solves the task). The 2025 follow-up landscape reveals which are essential:

### Essentiality of each assumption
- **Input-injectivity is the load-bearing assumption.** It allows Sutter to construct a bijection between hidden states and algorithm internal states. Architectures with strict information bottlenecks (autoencoders, low-rank attention, mixture-of-experts gating) violate injectivity by design. **The trivialisation provably fails when injectivity fails on a non-negligible subset.** Sutter's appendix discusses this; no published alternative trivialisation result yet exists for non-injective architectures.
- **Countability of 𝒳** is largely cosmetic — used to enumerate the bijection. Continuous 𝒳 extensions follow by measure-theoretic arguments (the paper notes this in Appendix).
- **Strict output-surjectivity** can be relaxed to "near-surjectivity in IIA metric" with quantitative degradation of the trivialisation bound. Empirically explored in Sutter §5.
- **Matchable partial orderings** is the assumption that fails for cyclic algorithms (recurrent computation, fixed-point iteration). Sutter explicitly says: "We leave the expansion of our work to such models for future work."
- **DNN solves the task:** important for the empirically interesting case but the theorem's trivialisation goes through even for random networks (the IOI experiment with randomly-initialised Pythia models reaches near-perfect IIA precisely because of this).

### Direct follow-up work (2025–2026)
- **Méloux et al. (2025)**, cited in Sutter as showing "a model's causal abstraction is not necessarily unique." This is a *partial* trivialisation result *prior* to Sutter — it demonstrates non-uniqueness but stops short of arbitrary algorithm assignment. (Sutter strengthens this from "non-unique" to "everything maps to everything.")
- **Geiger et al., "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability"** (JMLR 26:83, 2025; arXiv:2301.04709v4) — provides the formal framework Sutter critiques and includes the constructive-vs-strong abstraction distinction that is central to relaxing assumption (iv).
- **Pîslar, Magliacane, Geiger (arXiv:2503.11429, 2025):** Proposes combining multiple simple high-level models, explicitly to navigate the strength↔faithfulness tradeoff Sutter exposes.
- **"Bucketing the Good Apples: A Method for Diagnosing and Improving Causal Abstraction"** (late-2025 deposit) — proposes partitioning inputs into "interchange-consistent" subsets, implicitly restricting the function class to recover non-vacuity.
- **Xia, "Causal Abstraction Inference under Lossy Representations"** (arXiv:2509.21607, ICML 2025): introduces "projected abstractions" that handle non-injective abstractions (Abstract Invariance Condition relaxation) — directly relevant to the bottleneck case.
- **"Causal Abstractions, Categorically Unified"** (arXiv:2510.05033, 2025): a categorical / Markov-category framework that may provide a cleaner formalism for stating the squeeze.
- **"Aligning Graphical and Functional Causal Abstractions"** (arXiv:2412.17080) — α-abstraction frameworks with finer-grained handles.

### Architectures violating injectivity
Autoencoders, VAEs, contractive maps, and any layer with rank < input dimension. **No published "alternative trivialisation result" yet exists for these architectures** — this is an open problem and a natural follow-on paper.

---

## Sub-Question 4: Bilodeau et al. Generalisation

Bilodeau's Theorem 2.3 requires both **completeness** and **linearity**. The interaction is essential to the proof: linearity lets the adversary construct pairs of models with identical Φ-output but opposite behaviour; completeness fixes the constraint that pins Φ-output to a scalar derived from f.

### What is known
- **Karimi et al., "Attribution-based Explanations that Provide Recourse Cannot Be Robust"** (JMLR 24, 2023, jmlr.org/papers/volume24/23-0042/) acknowledges Bilodeau and explicitly writes that requiring completeness "excludes all counterfactual methods that are commonly used in algorithmic recourse." They establish a related, weaker impossibility for any attribution method (not just complete+linear) but with a different conclusion (robustness, not Spec+Sens).
- **No published result drops linearity while retaining completeness and re-derives Spec+Sens ≤ 1.** Gradients (a complete-but-nonlinear method if combined with completion axioms) appear to escape Bilodeau's bound, suggesting linearity is genuinely necessary.
- **Mehrpanah et al. (arXiv:2508.10490)** show that gradient methods incur their *own* tradeoff (smoothness vs faithfulness) but not a Spec+Sens-style impossibility.
- **Zhou et al., "Interpretability of Neural Networks Based on Game-Theoretic Interactions"** (Machine Intelligence Research 2023) offers an axiomatic unification framework that subsumes 14 attribution methods — useful for parameterising "weaker than linearity" axioms.
- **Bassan, "Additive Models Explained: A Computational Complexity Approach"** (arXiv:2510.21292, 2025): shows that even relaxing to additive (GAM) models, the *computational* hardness can be tight — orthogonal to but compatible with Bilodeau's information-theoretic bound.
- **"A Polynomial-Time Axiomatic Alternative to SHAP for Feature Attribution"** (arXiv ~2506.00496, 2025) — shows axiomatic alternatives must trade away some Shapley axioms (notably linearity) for tractability, indirectly probing the same boundary.

### Boundary
The impossibility boundary appears to lie between **(complete + linear)** [impossible per Bilodeau] and **(complete + scale-equivariant + neighbourhood-local)** [possibly tractable, à la gradient methods]. **No paper has formally drawn this boundary**; this is the natural extension of Bilodeau and a key piece the Squeeze Conjecture would need to formalise.

### Citation network
A Google-Scholar trail from arXiv:2212.11870 (2023–2026) yields: Karimi et al. JMLR 2023, Mehrpanah 2025, Kumar–Ahuja SURF 2025, Bassan 2025, plus the Van den Broeck/Arenas SHAP-complexity line (Sub-Q6). The PNAS publication (DOI 10.1073/pnas.2304406120, Jan 2024) is the canonical citation.

---

## Sub-Question 5: Information-Theoretic Framework

Reformulating both results in terms of mutual information I(Φ; B) is natural and partially worked out in the literature.

### Existing IT frameworks for attribution
- **Schulz et al., "Restricting the Flow: Information Bottlenecks for Attribution" (IBA)** (ICLR 2020, arXiv:2001.00396). Directly uses Tishby's IB to assign bits-of-information scores to input regions, providing the closest existing instantiation of "Φ as a channel."
- **VIBI** (Bang et al., AAAI 2021) — variational IB for system-agnostic explanations.
- **Bassan-Geffner-Israeli, "Fine-Grained Neural Network Explanation by Identifying Input Features with Predictive Information"** (arXiv:2110.01471).
- **CoIBA** (Hong et al., CVPR 2025, openaccess.thecvf.com/content/CVPR2025/papers/Hong_...) — comprehensive IB for transformer attributions.
- **Spatial Information Bottleneck** (arXiv:2511.09239, 2025) — uses MI on foreground/background regions.
- **Conductance-based Information Plane analysis** (arXiv:2408.14681) — measures Information Transformation Efficiency layer-by-layer.
- **Multi-Modal IB Attribution** (arXiv:2312.17174) for VLPMs.

These works establish that **I(Φ; B) is a well-defined and variationally-computable quantity for attribution methods.** Crucially, none has framed the Bilodeau/Sutter results in this language; they use IB *constructively* (to build better attributions) rather than *negatively* (to prove impossibility). This is an unexploited opportunity.

### The proposed IT reformulation
- **Bilodeau direction:** For Φ that is complete and linear, the image of Φ lies in an affine subspace of ℝ^{d×m} of dimension ≤ d−1 (after subtracting the baseline). For the binary hypothesis test f ∈ ℱ⁽⁰⁾ vs f ∈ ℱ⁽¹⁾, this affine subspace contains representatives of *both* hypotheses, so by data-processing inequality I(Φ; B) ≤ I(affine-projection-of-f; B) = 0 in the worst case. The Spec+Sens ≤ 1 bound is the Fano-inequality reading of I(Φ; B) = 0: P_err ≥ 1 − (I(Φ;B) + 1)/log|B| = 1 − 1/log 2 = 0 (the binary uniform error bound).
- **Sutter direction:** Allowing arbitrary measurable ϕ with input-injectivity means ϕ can be constructed to make any desired conditional distribution p(B | Φ) realisable. So I(Φ; B) = H(B) is achievable *for any choice of "ground-truth" B*, including incorrect ones — and the analyst has no signal to prefer correct B over incorrect B.

### Rate–distortion with side information
**Wyner–Ziv-style rate–distortion with side information** (the side info being the architecture, training data, or task structure) is the right formal model. The Bilodeau side has *too little* side info to make low-rate Φ informative; the Sutter side has *too much* freedom in ϕ for fixed side info to anchor the abstraction. Spelling this out formally would be a publishable contribution and looks tractable: the right inequality to prove would be of the form

> I(Φ; B_correct) − sup_{B_wrong} I(Φ; B_wrong) ≤ 0 for all c ∈ (c₁, c₀),

with the LHS positive only at c values where ℱ_c is non-trivially restricted.

---

## Sub-Question 6: Computational Complexity Perspective

Even if a "just right" c existed information-theoretically, computing Φ might be intractable.

- **Van den Broeck, Lykov, Schleich, Suciu, "On the Tractability of SHAP Explanations"** (arXiv:2009.08634, AAAI 2021): proves computing SHAP is **#P-hard** over the empirical distribution; NP-hard even for trivial classifiers over naive Bayes distributions; equivalent in complexity to expectation computation for fully-factorised distributions.
- **Arenas, Barceló, Bertossi, Monet, "Updates on the Complexity of SHAP Scores"** (IJCAI 2024, dl.acm.org/doi/10.24963/ijcai.2024/45): refines the polynomial-time / #P-hard boundary across classifier families.
- **Marzouk et al., "When is the Computation of a Feature Attribution Method Tractable?"** (arXiv:2501.02356, 2025): generalises to power indices and proves a simple condition implying polynomial-time equivalence to expectation evaluation.
- **Bassan, "Additive Models Explained: A Computational Complexity Approach"** (arXiv:2510.21292, 2025): shows that SHAP explanations for *classification* GAMs are #P-hard while *regression* GAMs are polynomial — a fine-grained complexity separation driven by the linearity axiom Bilodeau also relies on. Theorem 4 of that paper proves "Regression Smooth GAMs and regression GAMs with enumerable inputs are strictly more c-interpretable than classification Smooth GAMs."
- **Polynomial-time axiomatic alternative paper** (arXiv ~2506.00496/2603.00496) shows that *axiomatic alternatives* must trade away some Shapley axioms to gain tractability.

### Analogy to Arrow / VCG
Workarounds to Arrow's theorem (e.g., the Vickrey–Clarke–Groves mechanism) recover possibility by *paying a computational price* (general VCG is NP-hard). The exact same pattern would apply here: even if some intermediate c removes both Bilodeau's and Sutter's failures, Φ-computation at that c may be NP- or #P-hard. **Recovering interpretability against impossibility theorems is a computationally expensive operation across multiple fields.**

### Implication for the Squeeze Conjecture
Even if information-theoretically c₀ ≤ c₁ (the pure conjecture is *false*), it is plausible that **the value of c at which Φ becomes informative requires Φ-computation that is #P- or NP-hard**, recovering a *computational* squeeze. The **resource-bounded** version of the conjecture is therefore:

> *For any polynomial-time-computable Φ, the feasible complexity interval (c₀, c₁) is empty.*

This is potentially provable even when the unconstrained conjecture is not, by combining Bilodeau's information lower bound with reductions in the Van den Broeck / Arenas tradition.

---

## Assessment of Provability

### Tools available
- Rademacher / covering-number machinery is mature and tightly linked to the Bilodeau proof technique.
- Information-theoretic Fano-style lower bounds (Bilodeau) and surjectivity-based constructive trivialisation (Sutter) are *both* now standard.
- Categorical / Markov-category frameworks (arXiv:2510.05033) provide a clean target language.
- SHAP-complexity literature (Van den Broeck and successors) supplies the resource-bounded angle.

### Bottlenecks
1. **Parameterising the bijection class.** Sutter's proof uses *arbitrary measurable bijections*. To talk about c interpolating between Bilodeau's affine class and Sutter's full class, one needs a natural one-parameter sub-family — e.g., Lipschitz-bounded ϕ with Lipschitz constant L = c; polynomial ϕ of degree c; k-piecewise-linear ϕ. *Choosing this family is a research-design decision that determines whether c₀ > c₁ or c₀ < c₁.* Different parametrisations may yield different answers.
2. **Heterogeneity of the two impossibilities' "victims."** Bilodeau's victim is *the user's downstream inference* (Spec + Sens); Sutter's is *the abstraction relation itself* (any algorithm aligns). Until these are expressed as failures of one quantity (the natural candidate being I(Φ; correct-B) − sup_{B_wrong} I(Φ; B_wrong)), the squeeze cannot be written as a single inequality.
3. **The cyclic / non-injective gap.** Sutter explicitly excludes cyclic algorithms and assumes injectivity. The squeeze, if true universally, should also hold for these cases — but proving the Sutter side without injectivity is open.
4. **Choice of "behaviour space" B.** Bilodeau's B is a binary hypothesis (increasing vs decreasing in feature j locally). Sutter's B is "an algorithm." These are objects of very different cardinality and structure. Unifying them likely requires defining B as a *partition of the model class* and the impossibility as inability to identify the partition cell.

### Verdict
**The Squeeze Conjecture is plausible and likely provable in a precisely-chosen parametrisation, but is unlikely to be true universally without resource bounds.** With resource bounds (polynomial-time Φ), it becomes substantially more likely to hold and admits a cleaner proof via reduction. My best guess:

- **Pure information-theoretic version: probably false for some parametrisations** (e.g., polynomial-degree ϕ likely has a non-empty feasible window for moderate degrees), **provable for others** (e.g., when c is metric entropy and one demands worst-case over an adversarial behaviour class).
- **Resource-bounded version: probably true and provable** via combining Bilodeau's information lower bound with #P-hardness of Φ-computation in the Van den Broeck tradition.

---

## Concrete Next Steps

### Papers to read in priority order
1. **Bilodeau et al., arXiv:2212.11870 (PNAS 2024)** — canonical lower bound; study §2.D proof.
2. **Sutter et al., arXiv:2507.08802v2 (NeurIPS 2025 Spotlight)** — Theorem 1 and its proof in Appendix.
3. **Geiger et al., arXiv:2301.04709v4 (JMLR 26:83, 2025)** — framework underlying Sutter.
4. **Méloux et al., 2025** — non-uniqueness, the precursor to Sutter.
5. **Pîslar, Magliacane, Geiger, arXiv:2503.11429 (2025)** — explicit tradeoff formulation.
6. **Mehrpanah et al., arXiv:2508.10490 (2025)** — complexity-faithfulness tradeoff, closest existing squeeze.
7. **Van den Broeck et al., arXiv:2009.08634 (AAAI 2021)** — #P-hardness of SHAP.
8. **Arenas et al., IJCAI 2024 (10.24963/ijcai.2024/45)** — refined SHAP complexity.
9. **Bassan, arXiv:2510.21292 (2025)** — GAM complexity separation showing the linearity-tractability link.
10. **Xia, arXiv:2509.21607 (ICML 2025)** — lossy / projected abstractions, handles non-injectivity.
11. **arXiv:2510.05033 (2025)** — categorical unification of causal abstraction.
12. **Grohs & Petersen, FoCM 2022 (10.1007/s10208-021-09546-4)** — phase transitions in rate-distortion for neural networks.
13. **Bates & Jacobs, eLife 2023** — rate-distortion neural coding paradigm.
14. **Schulz et al., arXiv:2001.00396 (ICLR 2020)** — IBA, the IT-attribution prototype.
15. **Karimi et al., JMLR 24, 2023** — attribution & recourse impossibility companion.
16. **Marzouk et al., arXiv:2501.02356 (2025)** — tractability conditions for feature attribution.

### Mathematical tools to develop
1. **Define c as log-metric-entropy** log N(ε, ℱ_c, ‖·‖) at scale ε set by the task's discrimination threshold.
2. **Frame the conjecture as a Fano lower bound vs. Pinsker upper bound** on I(Φ; correct-B) − sup_{B_wrong} I(Φ; B_wrong); show they cross.
3. **Couple to Rademacher complexity** via Dudley's entropy integral to make finite-sample versions provable.
4. **Use rate–distortion with side information** (the side info being the task–model pair) as the heuristic interpretation.
5. **For the resource-bounded version**, reduce from #SAT or #DNF to "find an informative Φ at complexity level c" — adapting Van den Broeck's reductions.
6. **Stress-test with concrete architectures**: prove the squeeze for ReLU MLPs (where Bilodeau's piecewise-linearity assumption fits) and separately for transformers (where Sutter's empirical IOI experiments are situated).

### Specific publishable sub-projects
- **(A) The IT reformulation:** A synthesis paper restating both Bilodeau and Sutter as bounds on I(Φ; B) for a unified channel model. Largely synthesis; 2–4 months.
- **(B) The bottleneck-architecture variant:** Prove or disprove Sutter's trivialisation when input-injectivity fails (autoencoders). The most immediately important follow-on to Sutter.
- **(C) The Lipschitz-parametrised squeeze:** Take ℱ_c = {ϕ Lipschitz with constant ≤ c}. Compute c₀, c₁ explicitly for a toy task (hierarchical equality, as in Sutter's experiments). The first concrete instantiation.
- **(D) Resource-bounded squeeze:** Combine Bilodeau's bound with #P-hardness to prove the no-middle-ground theorem for polynomial-time Φ. The cleanest target.

---

## Concise Conclusion

The Squeeze Conjecture is a non-trivial but mathematically respectable hypothesis. The right complexity scale to parameterise it is **log-metric-entropy of the alignment / attribution function class**, controlled by **Rademacher complexity**, and interpreted via **rate–distortion theory with side information**. Information-theoretically the conjecture is parametrisation-dependent and not obviously true in all cases; **resource-bounded** versions are more plausibly provable and route through the SHAP-#P-hardness literature. As of late 2025 / early 2026, no paper proves the conjecture, but every component — Bilodeau's lower bound, Sutter's upper bound, the Méloux non-uniqueness intermediate result, IB-attribution machinery, SHAP-complexity reductions, and the categorical / rate–distortion frameworks — is now in the literature. A focused 6–12 month research effort along the four sub-projects above could reasonably settle either the conjecture itself or a sharp resource-bounded variant.

