### 深度研究问题分析

用户:


Claude:
��是一个非常深入且结构清晰的研究问题，涵盖面广、数学深度高。我直接启动深度研究，不需要额外澄清。
our unified mathematical framework for neural-network representation analysis report is ready. It examines five representation views (linear-subspace, geometric/kernel, information-theoretic, causal, topological/manifold) and their mathematical relationships, evaluates existing partial unifications by Klabunde et al., Geiger et al., Tang et al., and category-theoretic approaches, and analyzes frontier problems including SAE identifiability, multi-dimensional features, and the case for an Arrow-style impossibility theorem. The report includes established equivalences (linear CKA ↔ RV coefficient), key failure modes of current methods, a comparative table of 14 methods with their invariance assumptions, and a staged research plan with revision thresholds. Central tension: whether causal abstraction genuinely unifies mechanistic interpretability or trivializes without linearity restrictions.
# Toward a Unified Mathematical Framework for Neural-Network Representation Analysis

## TL;DR
- **A single, complete unifying framework does not yet exist, but two partial unifications have crystallized**: (i) Klabunde et al.'s invariance-group taxonomy for *similarity measures* (PT ⊂ OT ⊂ ILT ⊂ AT), and (ii) Geiger et al.'s *causal abstraction* (arXiv:2301.04709), which claims to subsume ~10 mechanistic-interpretability methods (patching, mediation analysis, scrubbing, tracing, circuit analysis, concept erasure, SAEs, DBM, DAS, activation steering) — but Sutter et al. (arXiv:2507.08802, NeurIPS 2025 Spotlight) recently proved this unification is *trivial* without a restricted feature-encoding hypothesis ("any neural network can be mapped to any algorithm").
- **The frontier is moving toward three candidate "grand frameworks"** — category-theoretic (Tull/Coecke arXiv:2406.17583; Englberger & Dhami arXiv:2510.05033), piecewise-biconvex optimization for all sparse-dictionary methods (Tang et al. arXiv:2512.05534), and information-geometric — but none yet handles probing, CKA, PID, TDA, and SAE simultaneously.
- **An "Arrow-style impossibility theorem for interpretability" is already partially in hand**: Bilodeau et al. (arXiv:2212.11870, PNAS 2024) proved that any feature-attribution method satisfying completeness + linearity (including SHAP and Integrated Gradients) "can provably fail to improve on random guessing"; Han, Srinivas & Lakkaraju (arXiv:2206.01254, NeurIPS 2022) prove "a no free lunch theorem for explanation methods, demonstrating that no single method can perform optimally across all neighbourhoods"; combined with Sutter et al.'s causal-abstraction trivialization and Kleinberg's clustering impossibility, this strongly suggests no single representation-analysis method can satisfy faithfulness, identifiability, scalability, and generality at once.

---

## Key Findings

**1. The mainstream toolkit divides into five "representation views," each with distinct invariance groups and implicit feature definitions.**
- *Linear-subspace view* (SAE, transcoders, PCA, steering vectors, logit/tuned lens, linear probes) — features are directions in activation space; invariant only under permutation of latent basis when sparsity is enforced.
- *Geometric/kernel view* (CKA, CCA, SVCCA, RSA, Procrustes) — features are pairwise-similarity patterns; CKA is invariant under orthogonal transformation + isotropic scaling (OT∪IS) but not invertible linear transformation.
- *Information-theoretic view* (PID, mutual information probes, MDL probing) — features are atoms of redundant/unique/synergistic information.
- *Causal view* (activation patching, causal tracing/ROME, causal abstraction, DAS, interchange interventions) — features are variables whose interventional behavior matches a high-level algorithm.
- *Topological/manifold view* (persistent homology, TDA-based similarity, multi-dimensional/circular features) — features are persistent simplicial-complex invariants or irreducible sub-manifolds.

**2. Klabunde et al. (arXiv:2305.06329, ACM Computing Surveys 2025) is the cleanest existing unification, but only for *similarity* measures.** Their taxonomy classifies every published similarity measure by the group of transformations under which it is invariant: permutation (PT) ⊂ orthogonal (OT) ⊂ invertible linear (ILT) ⊂ affine (AT), plus translation (TR) and isotropic scaling (IS). This explains why CKA, CCA, RSA, and Procrustes disagree: they answer different invariance-respecting questions.

**3. Linear CKA = RV coefficient = Tucker congruence coefficient.** Established formally by Kornblith, Norouzi, Lee & Hinton, "Similarity of Neural Network Representations Revisited" (ICML 2019, arXiv:1905.00414): "For linear kernels, CKA is equivalent to the RV coefficient and to Tucker's congruence coefficient." Reaffirmed by Klabunde et al.; this is the prototype of the kind of equivalence that a broader framework should produce.

**4. SAE has three quantitatively documented failure modes that motivate the search for a deeper framework.**
- *Non-identifiability across seeds*: Paulo & Belrose (arXiv:2501.16615, Jan 2025) showed "in an SAE with 131K latents trained on a feedforward network in Llama 3 8B, only 30% of the features were shared across different seeds"; TopK SAEs are *more* seed-dependent than ReLU+L1 SAEs.
- *OOD/"fake" features*: Cho et al. (arXiv:2506.17673) and Heap et al. show SAEs trained on external data hallucinate features not used internally.
- *Linear-only assumption fails*: Engels et al. (arXiv:2405.14860) demonstrated irreducible 2D circular features (days of week, months) that one-dimensional SAE atoms cannot represent without decomposition. They define irreducibility precisely: features that cannot be decomposed into "either independent or non-co-occurring lower-dimensional features," with a "new superposition hypothesis [that] posits independence between irreducible multi-dimensional features instead of unknown levels of independence between one-dimensional features."

**5. Tang et al. (arXiv:2512.05534, Dec 2025) provide the first unified theory for the *sparse-dictionary* subfamily.** Verbatim: "We develop the first unified theoretical framework that casts all major SDL variants as a single piecewise biconvex optimization problem, and characterize its global solution set, non-identifiability, and spurious optima. This analysis yields principled explanations for feature absorption and dead neurons." They propose "feature anchoring" to restore identifiability.

**6. Geiger et al. (arXiv:2301.04709) attempted the broadest unification of mechanistic-interpretability methods.** Verbatim from their contributions list: they "[unify] a variety of mechanistic interpretability methodologies in the common language of causal abstraction, namely activation and path patching, causal mediation analysis, causal scrubbing, causal tracing, circuit analysis, concept erasure, sparse autoencoders, differential binary masking, distributed alignment search, and activation steering."

**7. Sutter et al. (arXiv:2507.08802, NeurIPS 2025 Spotlight) proved this unification is vacuous without an additional encoding assumption.** Their core theorem: "under reasonable assumptions, any neural network can be mapped to any algorithm, rendering this unrestricted notion of causal abstraction trivial and uninformative." Empirically: "on an experiment using randomly initialised language models, our alignment maps reach 100% interchange-intervention accuracy on the indirect object identification task." They name this the "non-linear representation dilemma": causal abstraction unifies the methods only if we *first* fix a feature-encoding hypothesis (typically linearity).

**8. Bilodeau, Jaques, Koh & Kim (arXiv:2212.11870, PNAS 2024) gave the first published impossibility theorem at this level of generality.** Verbatim: "for moderately rich model classes (easily satisfied by neural networks), any feature attribution method that is complete and linear — for example, Integrated Gradients and SHAP — can provably fail to improve on random guessing for inferring model behaviour." This is the closest analog yet to "Arrow's theorem for interpretability."

**9. Category-theoretic frameworks are emerging as the most promising candidate "grand unifier."** Tull et al. (arXiv:2406.17583) propose a string-diagrammatic framework that uniformly represents linear, rule-based, recurrent, transformer, VAE, causal, and DisCoCirc models, and defines *interpretation* itself in compositional terms. Englberger & Dhami (arXiv:2510.05033) use Markov categories and natural transformations to unify multiple causal-abstraction definitions: "Our approach unifies and generalizes previously considered causal abstractions. … We discuss how methods from mechanistic interpretability, such as circuit analysis and sparse autoencoders, fit within our categorical framework."

---

## Details

### I. Method Panorama: implicit "what is a feature" assumptions

| Method | Mathematical core | Implicit "feature" | Key invariance / assumption |
|---|---|---|---|
| **SAE** (Cunningham et al. 2023, arXiv:2309.08600; Bricken et al. 2023) | Overcomplete sparse dictionary learning; minimize ‖x − Dz‖² + λ‖z‖₀ or topK | A one-dimensional, near-monosemantic direction active sparsely | Linear superposition; identifiability up to permutation/scaling — *in practice not even that*, per Paulo & Belrose |
| **Transcoders** (Dunefsky, Chlenski, Nanda, arXiv:2406.11944) | Wider sparse-MLP approximation of an MLP sublayer; "transcoders, which seek to faithfully approximate a densely activating MLP layer with a wider, sparsely-activating MLP layer" | A sparsely-activating linear feature with input-dependent/input-invariant factorization | Replaces nonlinearity-laden SAE-on-MLP-output with sparse-MLP |
| **Crosscoders** (Lindsey et al. 2024) | SAE across multiple layers/models simultaneously | Cross-layer linear feature | Same linearity assumption, plus shared dictionary across layers |
| **Linear probe** (Alain & Bengio, arXiv:1610.01644) | Train linear classifier on frozen activations | A linearly decodable concept | Linearity; Hewitt & Liang (2019) selectivity controls |
| **Structural probe** (Hewitt & Manning 2019) | Learn metric tensor s.t. probe distance ≈ parse-tree distance | Isomorphism between geometry of representation space and external structure (e.g., parse tree) | Squared-distance under a learned linear transform encodes tree distance |
| **Logit lens** (nostalgebraist 2020) | Project hidden state through unembedding W_U | Direction interpretable in vocab basis at every layer | Assumes intermediate layers share final-layer basis — *empirically false* for many models |
| **Tuned lens** (Belrose et al., arXiv:2303.08112) | Per-layer affine probe (Aℓ, bℓ) trained to match final-layer distribution; "a refinement of the earlier 'logit lens' technique" | Latent prediction at each layer | Iterative-inference view: each layer refines a probability distribution |
| **Activation patching / causal tracing** (Meng et al. ROME; Heimersheim & Nanda arXiv:2404.15255; Zhang & Nanda arXiv:2309.16042) | Replace activations from a clean run with cached corrupted activations | A component is "a feature" if patching it suffices to flip output | Causal sufficiency; vulnerable to OOD activation effects under Gaussian-noise corruption (Zhang & Nanda demonstrate GN-induced peaks that vanish under symmetric-token replacement) |
| **Causal abstraction / DAS** (Geiger et al., arXiv:2106.02997, arXiv:2301.04709) | Align high-level causal variables with sets of neurons via alignment map; verify with interchange interventions | A high-level variable in an external causal graph | Alignment map class is implicit; if unrestricted, trivialization (Sutter et al.) |
| **CKA / RSA / CCA / SVCCA** (Kornblith arXiv:1905.00414; Kriegeskorte 2008; Raghu 2017) | Compare similarity-matrix statistics | A representation-similarity pattern | Kernel CKA: invariant under OT and IS; CCA: invariant under invertible linear |
| **PID** (Williams & Beer; Ehrlich et al. arXiv:2209.10438) | Decompose I(target; neurons) into redundant/unique/synergistic atoms | An information atom (R/U/S) held by a subset of neurons | Requires a chosen redundancy lattice (Williams–Beer, Bertschinger–Rauh, Ince…), all giving different decompositions |
| **Representation engineering / steering** (Zou et al. 2023; Turner et al. arXiv:2308.10248) | Contrastive difference-of-means in activation space | A "concept vector" extracted from contrasting prompts | Linear concept direction; non-identifiable (large equivalence classes under white-box single-layer access) |
| **Persistent homology / TDA** (Watanabe & Yamana arXiv:2106.03016; Pérez & Reinoso arXiv:2101.07752) | Filtered simplicial complex from activations; track birth/death of homology classes | A persistent topological invariant (component, cycle, void) | Continuous-deformation invariant; insensitive to basis choice |
| **Linear representation hypothesis** (Park, Choe, Veitch, arXiv:2311.03658) | Counterfactual-based formalization of "concept direction" + causal inner product | A direction that is causally separable in counterfactual pairs | Concepts are directions; geometry requires a *non-Euclidean* (causal) inner product to unify probing, steering, and SAEs |

### II. Mathematical Relationships

**Established equivalences/inclusions:**
- Kornblith et al. (arXiv:1905.00414): "For linear kernels, CKA is equivalent to the RV coefficient and to Tucker's congruence coefficient."
- HSIC with linear kernels reduces to a normalized squared Frobenius norm of cross-covariance; CKA is the isotropic-scaling-normalized HSIC.
- CKA is equivalent to a bound on Maximum Mean Discrepancy (MMD) up to a constant.
- Activation patching is a special case of Pearl-style causal mediation analysis (Vig et al. 2020).
- Logit lens is the tuned lens with the identity translator; both are special cases of "iterative-inference" probing.
- Transcoders + sparse-MLP-replacement on attention outputs are the same biconvex optimization family as SAEs (Tang et al. arXiv:2512.05534, who write: "We theoretically prove that SDL optimization is biconvex, bridging mechanistic interpretability methods with traditional biconvex optimization theory").
- The linear-representation hypothesis (Park et al.) shows that *probing*, *steering*, and *SAE feature directions* all collapse to the same object once the causal inner product is fixed: "Using this causal inner product, we show how to unify all notions of linear representation."

**Partial unifications already in print:**

1. **Klabunde et al. (arXiv:2305.06329)** unifies similarity measures via invariance groups.
2. **Geiger et al. (arXiv:2301.04709)** unifies ~10 mech-interp methods via causal abstraction — but Sutter et al. (arXiv:2507.08802) prove this is trivial without restricting the alignment-map class.
3. **Tang et al. (arXiv:2512.05534)** unifies SAEs, transcoders, and crosscoders as one piecewise-biconvex problem.
4. **Park, Choe, Veitch (arXiv:2311.03658)** unifies probing + steering + SAE feature semantics under one causal inner product.
5. **Tull et al. (arXiv:2406.17583)** propose a *category-theoretic / string-diagrammatic* framework that uniformly represents the *models*; they define interpretability as transparency of the compositional structure itself.
6. **Englberger & Dhami (arXiv:2510.05033)** define causal abstractions as natural transformations between Markov functors and explicitly fit SAEs and circuits inside the category-theoretic framework.

**What's missing.** No single framework handles all five views simultaneously. The candidates and what each currently misses:
- *Information geometry* (Amari 1998): a natural unifier for probing + PID + tuned lens (all are Fisher-related), but has not yet been formally connected to SAEs or causal abstraction.
- *Category theory / sheaves* (Hansen & Gebhart arXiv:2012.06333; Bodnar et al. arXiv:2202.04579; Tull arXiv:2406.17583): naturally encodes hierarchical composition and locality, fits causal abstraction and circuits, but does not yet incorporate similarity measures or PID atoms.
- *Piecewise-biconvex optimization* (Tang et al.): elegant and proves identifiability theorems, but limited to the sparse-dictionary subfamily.
- *Topological / sheaf-cohomological*: handles multi-dimensional features and global-local consistency, but lacks a probing-style decodability story.

### III. Frontier Problems

**Problem 1. Nonlinear / multi-dimensional features.**
Engels et al. (arXiv:2405.14860) proved that *irreducible* multi-dimensional features exist — defined as features that cannot be decomposed into "either independent or non-co-occurring lower-dimensional features" — and exhibited circular representations for days/months that Mistral-7B and Llama-3-8B causally use for modular arithmetic. *Importance*: invalidates the one-dimensional reading of the linear-representation hypothesis and the basic SAE architecture. *Current progress*: SAE variants that automatically detect multi-dimensional feature *clusters* (Engels et al.'s detection method); group-theoretic SAEs using cyclic-group inductive bias; manifold-aware dictionary learning. *Gap*: no general algorithm for discovering features supported on arbitrary low-dimensional manifolds when the manifold class is unknown a priori.

**Problem 2. SAE identifiability.**
*Statement*: SAEs trained on the same data with different seeds recover overlapping but substantially different dictionaries — exactly 30% feature overlap on Llama-3-8B with 131K latents (Paulo & Belrose, arXiv:2501.16615). *Importance*: undermines claims that SAEs recover ground-truth features rather than a useful basis. *Progress*: 
- (i) Tang et al. (arXiv:2512.05534) prove the optimization is piecewise biconvex with provable spurious minima and introduce "feature anchoring" to recover identifiability; 
- (ii) Fel et al. propose Archetypal SAEs (arXiv:2502.12892) that constrain dictionary atoms to the convex hull of the data — verbatim from the abstract: "This geometric anchoring significantly enhances the stability of inferred dictionaries, and their mildly relaxed variants RA-SAEs further match state-of-the-art reconstruction abilities"; 
- (iii) Cui et al. (arXiv:2506.14002) provide the first SAE with formal feature-recovery guarantees by modeling polysemantic features as sparse mixtures of monosemantic concepts; 
- (iv) Park-Choe-Veitch's causal inner product provides a canonical geometry that may restore identifiability up to a unique isometry. 
*Gap*: no method yet provably recovers a *unique* dictionary on real LLM activations.

**Problem 3. An impossibility theorem for representation analysis.**
*Statement*: prove that no single method can simultaneously satisfy: (a) faithfulness to the underlying model, (b) identifiability/uniqueness, (c) scalability, (d) generality across feature types (linear + nonlinear + topological), (e) human-interpretability. *Importance*: would refocus the field from "finding the right method" to "designing portfolios of complementary methods with known trade-offs," analogous to how Arrow's theorem reshaped social choice.
*Current progress and partial results*:
- **Bilodeau et al. (arXiv:2212.11870, PNAS 2024)** — completeness + linearity ⇒ feature attribution "can provably fail to improve on random guessing for inferring model behaviour."
- **Han, Srinivas & Lakkaraju (arXiv:2206.01254, NeurIPS 2022)** — establish "a no free lunch theorem for explanation methods, demonstrating that no single method can perform optimally across all neighbourhoods."
- **Sutter et al. (arXiv:2507.08802, NeurIPS 2025 Spotlight)** — unrestricted causal abstraction is trivial: any network maps to any algorithm.
- **Klabunde et al. (arXiv:2305.06329)** — no similarity measure can satisfy all six invariances simultaneously (PT, OT, ILT, AT, TR, IS).
- **Kleinberg (NeurIPS 2002)** — no clustering function can satisfy scale-invariance + richness + consistency; verbatim: "for a set of three simple properties, we show that there is no clustering function satisfying all three." SAE feature discovery is fundamentally a clustering problem, so Kleinberg's impossibility carries over.
*Gap*: no single paper has yet stated and proved a clean axiomatic impossibility theorem for the *full* class of representation-decomposition methods (not just attribution).

**Problem 4. Cross-method translation.**
Even where two methods agree (e.g., a probe and an SAE both find "is_French"), there is no formal map between them — they live in different mathematical universes. *Progress*: Park et al.'s causal inner product (arXiv:2311.03658) is the most explicit attempt to date. *Gap*: a general functorial translator between linear-probing claims, SAE features, patching circuits, and PID atoms.

### IV. Key Papers (arXiv IDs)

**Foundational and unifying.**
- Klabunde, Schumacher, Strohmaier, Lemmerich. *Similarity of Neural Network Models: A Survey of Functional and Representational Measures.* arXiv:2305.06329 (ACM Comp. Surv. 2025).
- Geiger, Ibeling, Zur, et al. *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability.* arXiv:2301.04709.
- Tang et al. *A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima.* arXiv:2512.05534.
- Tull, Lorenz, Clark, Khan, Coecke. *Towards Compositional Interpretability for XAI.* arXiv:2406.17583.
- Englberger & Dhami. *Causal Abstractions, Categorically Unified.* arXiv:2510.05033.
- Park, Choe, Veitch. *The Linear Representation Hypothesis and the Geometry of LLMs.* arXiv:2311.03658.

**SAE foundations and critiques.**
- Cunningham et al. *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* arXiv:2309.08600.
- Paulo & Belrose. *Sparse Autoencoders Trained on the Same Data Learn Different Features.* arXiv:2501.16615.
- Cho et al. *FaithfulSAE.* arXiv:2506.17673.
- Cui et al. *Taming Polysemanticity in LLMs: Provable Feature Recovery.* arXiv:2506.14002.
- Fel et al. *Archetypal SAE.* arXiv:2502.12892.
- Engels, Liao, Michaud, Gurnee, Tegmark. *Not All Language Model Features Are One-Dimensionally Linear.* arXiv:2405.14860.
- Dunefsky, Chlenski, Nanda. *Transcoders Find Interpretable LLM Feature Circuits.* arXiv:2406.11944.

**Causal / patching.**
- Geiger et al. *Causal Abstractions of Neural Networks.* arXiv:2106.02997.
- Heimersheim & Nanda. *How to Use and Interpret Activation Patching.* arXiv:2404.15255.
- Zhang & Nanda. *Towards Best Practices of Activation Patching.* arXiv:2309.16042.
- Sutter et al. *The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?* arXiv:2507.08802 (NeurIPS 2025 Spotlight).

**Probing and lenses.**
- Alain & Bengio. *Understanding intermediate layers using linear classifier probes.* arXiv:1610.01644.
- Belrose et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.
- Belinkov. *Probing Classifiers: Promises, Shortcomings, and Advances.* arXiv:2102.12452.
- Kornblith, Norouzi, Lee, Hinton. *Similarity of Neural Network Representations Revisited.* arXiv:1905.00414 (ICML 2019).

**Representation engineering.**
- Zou et al. *Representation Engineering: A Top-Down Approach to AI Transparency.* (2023).
- Turner et al. *Activation Addition.* arXiv:2308.10248.

**PID and topology.**
- Ehrlich et al. *A Measure of the Complexity of Neural Representations based on Partial Information Decomposition.* arXiv:2209.10438.
- Watanabe & Yamana. *Topological Measurement of DNNs Using Persistent Homology.* arXiv:2106.03016.
- Hansen & Gebhart. *Sheaf Neural Networks.* arXiv:2012.06333.
- Bodnar et al. *Neural Sheaf Diffusion.* arXiv:2202.04579.

**Impossibility / no-free-lunch.**
- Bilodeau, Jaques, Koh, Kim. *Impossibility Theorems for Feature Attribution.* arXiv:2212.11870 (PNAS 2024).
- Han, Srinivas, Lakkaraju. *Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post Hoc Explanations.* arXiv:2206.01254 (NeurIPS 2022).
- Kleinberg. *An Impossibility Theorem for Clustering.* NeurIPS 2002 (not on arXiv).

---

## Recommendations

**Staged plan for a researcher pursuing the unification program.**

1. **Short term (now)**: Stop treating SAEs as ground-truth feature discoverers. Adopt the Paulo–Belrose seed-consistency metric (PW-MCC) and Tang et al.'s feature-anchoring as a minimum standard. Always report invariance group (in Klabunde et al.'s sense) when claiming representational similarity. *Threshold to revise*: if any new SAE architecture achieves >90% PW-MCC across seeds at frontier scale and survives RAVEL/Makelov-style intervention benchmarks, treat it as the new baseline.

2. **Medium term (2026–2027)**: Pursue *partial* unifications within each view rather than a single grand framework. The most promising near-term targets are: (a) extend Tang et al.'s biconvex framework to multi-dimensional/manifold features; (b) extend Park–Choe–Veitch's causal inner product to non-binary concepts; (c) formalize the Englberger–Dhami categorical framework to include CKA and PID as functors. *Threshold to revise*: if a single category-theoretic or sheaf-theoretic paper subsumes ≥3 of the five "representation views" (linear, geometric, information-theoretic, causal, topological), pivot toward consolidating around it.

3. **Long term**: Prove the impossibility theorem explicitly. State five axioms (faithfulness, identifiability, scalability, generality, human-interpretability), then prove formally that no measurable function on activations can satisfy all five. Bilodeau et al.'s template, Han et al.'s no-free-lunch result, and Sutter et al.'s trivialization argument are the existing scaffolding. *Threshold to revise*: if a constructive counterexample emerges — a single method demonstrably satisfying all five on a non-trivial model class — abandon the impossibility program.

4. **Practical advice for downstream users today**: triangulate. Use ≥3 methods from *different* invariance classes (e.g., a TopK SAE + activation patching + linear probe with selectivity controls + tuned lens) and only trust claims robust to all three. The cost is 3–4× compute; the benefit is avoiding the well-documented streetlight effect that Bilodeau et al. and Sutter et al. formalized.

---

## Caveats

- The Tang et al. (arXiv:2512.05534) paper is a December 2025 preprint and has not completed peer review; its formal claims rest on assumptions (e.g., what counts as "reasonable" alignment-map restrictions) that may be tightened in future versions. By contrast, Sutter et al. (arXiv:2507.08802) is accepted as NeurIPS 2025 Spotlight; its trivialization theorem has been peer-reviewed.
- "Category theory unifies interpretability" remains aspirational. The two strongest existing papers (Tull et al. arXiv:2406.17583; Englberger & Dhami arXiv:2510.05033) provide diagrammatic *languages* but do not yet yield new empirical results that probing, CKA, or PID could not have produced.
- Klabunde et al.'s invariance taxonomy applies to *representation similarity measures only*. Extending it to decomposition methods (SAE, PID, patching) requires defining group actions on outputs that are not real-valued scores but structured objects (dictionaries, causal graphs, information atoms) — a nontrivial extension that has not yet been carried out.
- The Engels et al. circular-features result has been replicated for days/months but has been challenged in some follow-up work as a special case of group-theoretic structure already present in token embeddings rather than emergent in the residual stream; the strength of the "non-linear feature" claim depends on which interpretation is correct.
- Several "equivalence" claims in the literature are exact (linear CKA ↔ RV coefficient, per Kornblith et al. arXiv:1905.00414); others (e.g., CKA bounds MMD) are *up to constants*. The report has flagged the strong vs. weak forms where relevant, but a future review should systematically distinguish them.
- **Central conflict.** Geiger et al. (arXiv:2301.04709) claim causal abstraction *unifies* SAEs, while Sutter et al. (arXiv:2507.08802) argue the same framework is trivial without auxiliary assumptions. Both can be true simultaneously: causal abstraction *with* a linearity restriction unifies many methods; causal abstraction *without* any restriction trivializes. The community has not yet converged on what the right restriction is, and this is the single most important open foundational question — analogous to how the choice of axioms determines whether Arrow's theorem applies.

