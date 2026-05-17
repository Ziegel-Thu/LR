### Research brief on ensemble diversity and voting theory

用户:


Claude:

This is a rich and well-structured research brief. The six questions are clear and specific enough that I can launch the research right away. Let me kick off a deep dive into all six areas — from smoothed analysis and ensemble diversity theory to Arrow-theorem analogs and attribution failure modes.

Starting the research now.
our literature review and research plan on worst-case vs. average-case attribution impossibility is ready. It synthesizes findings across attribution impossibility theorems (Bilodeau, Han, Sutter, Kleinberg), worst-case/average-case templates from compressed sensing and smoothed analysis, ensemble theory (Krogh–Vedelsby, boosting, co-training), and post-Arrow social choice escape routes. The report proposes framing the paper as "smoothed analysis of attribution impossibility," introduces an attribution condition number and smoothed identifiability gap, and includes a four-stage research plan plus a prioritized list of 44 key papers across seven tiers.
# Worst-Case vs Average-Case in Attribution Impossibility: A Literature Review and Research Plan

## TL;DR
- Attribution impossibility (Bilodeau PNAS 2024, Han NeurIPS 2022, Sutter NeurIPS 2025, Kleinberg NIPS 2002) is structurally analogous to a long list of "worst-case impossible / average-case tractable" phenomena — compressed-sensing NP-hardness vs. RIP-good random matrices, simplex worst-case exponential vs. Spielman–Teng smoothed polynomial, Arrow's theorem vs. Black's single-peaked possibility — and these analogies directly suggest the four most promising escape routes: smoothed analysis, distributional robustness over Wasserstein balls, domain restriction (Lipschitz / naturally trained networks), and method-portfolio combination.
- The closest existing framework for theorizing "multiple weak attribution methods become jointly strong" is the Krogh–Vedelsby (1995) ambiguity decomposition generalized to non-square losses by Wood, Mu, Webb, Reeve, Luján & Brown (JMLR 24, 2023; arXiv 2301.03962), combined with information-theoretic multi-view bounds (Sridharan & Kakade, COLT 2008); boosting (Schapire 1990; Freund–Schapire JCSS 1997) gives the right *narrative* analogy but the wrong loss structure, because explanations are estimands rather than predictions.
- The decisive research move is to define an **attribution condition number** (smallest model-class or distribution perturbation that collapses identifiability) and a **smoothed identifiability gap** (IGA averaged over Gaussian weight perturbation of variance σ², or over a Wasserstein-ε ball around the natural-data distribution), then prove that natural networks live in a low-condition-number / large-σ_eff regime where common methods like SHAP succeed, while Bilodeau's adversarial polynomial constructions live at σ = 0 — this is the attribution analogue of Spielman–Teng smoothed analysis of the simplex method.

---

## Key Findings

**Finding 1 — The impossibility results form a coherent family but all are worst-case.** Bilodeau, Jaques, Koh & Kim (PNAS 121(2):e2304406120, 2024; arXiv 2212.11870) show that "for moderately rich model classes (easily satisfied by neural networks), any feature attribution method that is complete and linear—for example, Integrated Gradients and SHAP—can provably fail to improve on random guessing for inferring model behaviour." Han, Srinivas & Lakkaraju (NeurIPS 2022; arXiv 2206.01254) prove a no-free-lunch theorem within the Local Function Approximation (LFA) framework that unifies LIME, KernelSHAP, IG, SmoothGrad and Gradients×Input. Kleinberg (NIPS 2002) gave the original three-axiom impossibility theorem (scale-invariance, richness, consistency) for clustering, explicitly modeled on Arrow's theorem. Denis Sutter, Julian Minder, Thomas Hofmann & Tiago Pimentel (NeurIPS 2025 Spotlight; arXiv 2507.08802) extend this to mechanistic interpretability: "under reasonable assumptions, any neural network can be mapped to any algorithm, rendering this unrestricted notion of causal abstraction trivial and uninformative" — empirically, "on an experiment using randomly initialised language models, our alignment maps reach 100% interchange-intervention accuracy on the indirect object identification task."

**Finding 2 — Every one of these results is built around an adversarial construction; none has been given an average-case treatment.** This is the central gap in the literature. By contrast, compressed sensing has the canonical worst-case-impossible / random-case-tractable story (Candès–Romberg–Tao, IEEE TIT 52(2):489–509, 2006; Donoho, IEEE TIT 52(4):1289–1306, 2006): ℓ₀ recovery is NP-hard and RIP-certification is NP-hard (Bandeira, Dobriban, Mixon & Sawin, arXiv 1204.1580), and even *average-case* certification is hard in tight regimes (Wang, Berthet & Plan, arXiv 1605.09646) — yet Gaussian random matrices satisfy RIP with high probability and ℓ₁ recovery succeeds. Smoothed analysis (Spielman & Teng, JACM 51(3):385–463, 2004; arXiv cs/0111050) gives the same lesson for the simplex algorithm: worst-case exponential, but expected number of pivots is polynomial after small Gaussian perturbation. Dadush & Huiberts (SIAM J. Comput. 49(5):STOC18-449–STOC18-499, 2020; arXiv 1711.05667) proved a best-of-both-worlds bound of O(d²σ⁻²√ln n + d³ ln^{3/2} n) pivot steps; Bach & Huiberts (FOCS 2025: 1829–1856; arXiv 2504.04197) prove both an upper bound of O(σ⁻¹ᐟ² d¹¹ᐟ⁴ log(n)⁷ᐟ⁴) and a matching lower bound Ω(σ⁻¹ᐟ² d¹ᐟ² ln(4/σ)⁻¹ᐟ⁴), showing "our algorithm has optimal noise dependence among all simplex methods, up to polylogarithmic factors." The structural analogy to attribution is clean: Bilodeau's construction places two models with identical evaluations on a neighborhood but radically different counterfactual behavior — exactly the kind of adversarial alignment that smoothing destroys.

**Finding 3 — Method combination has rigorous mathematical scaffolding that has not been ported to interpretability.** Boosting (Schapire, "The Strength of Weak Learnability," Machine Learning 5(2):197–227, 1990, doi 10.1007/BF00116037; Freund–Schapire, JCSS 55(1):119–139, 1997, doi 10.1006/jcss.1997.1504) shows weak ≡ strong PAC learnability. Krogh & Vedelsby (NIPS 1995) prove the ambiguity decomposition: ensemble squared error = weighted average member error − diversity (ambiguity). Wood, Mu, Webb, Reeve, Luján & Brown (JMLR 24, 2023; arXiv 2301.03962) generalize this to arbitrary Bregman-divergence losses (bias–variance–diversity decomposition for any twice-differentiable loss). Blum & Mitchell (COLT 1998) co-training proves that two conditionally-independent views can learn from unlabeled data alone; Sridharan & Kakade ("An Information Theoretic Framework for Multi-view Learning," COLT 2008, pp. 403–414) recast this as an information-theoretic redundancy bound. In the XAI literature, several practical ensembles exist — NormEnsembleXAI (arXiv 2401.17200), Meta-Explainers (Bello et al., Int. J. Intell. Syst. 2025; doi 10.1155/int/4841666), ensemble-based RF explanation combinations (arXiv 2402.19025), and the "ensemble interpretation: a unified framework" of Neural Computing and Applications 2025 (doi 10.1007/s00521-025-11130-8) which builds on Han–Srinivas–Lakkaraju's LFA — but none prove the analogue of Krogh–Vedelsby for the identifiability gap of attribution.

**Finding 4 — Post-Arrow escape routes have direct attribution analogues.** Arrow (1951) was followed by: Black's single-peaked-preferences theorem (Duncan Black, "On the Rationale of Group Decision-making," J. Pol. Econ. 56:23–34, 1948) which restricts the preference domain so that majority rule satisfies all axioms; Gibbard–Satterthwaite (Gibbard, Econometrica 41:587–601, 1973; Satterthwaite, JET 10:187–217, 1975) which reproduces the dictatorship trap in social-choice-function form, with three classical escapes — single-peaked domains, the quasilinear / VCG domain, and house-matching — surveyed by Bade (arXiv 1610.04873); Sen's "Impossibility of a Paretian Liberal" (showing Pareto + minimal-liberty are inconsistent unless the preference domain is restricted); randomization (Gibbard 1977); and the large domain-restriction literature surveyed by Barberà ("Arrow on Domain Conditions: A Fruitful Road to Travel," BSE Working Paper 1095). The attribution-side analogues are: (a) restrict the model class (Lipschitz networks, ReLU networks of bounded depth — Bilodeau's Assumption 2 is precisely where the impossibility bites, so restricting it breaks the impossibility); (b) relax linearity (gradient methods retain validity for "local sensitivity" — Bilodeau §D.1 proves the Gradient method is always successful on the local-sensitivity sub-task); (c) randomize (smoothed SHAP, SmoothGrad already exist empirically); (d) restrict to *natural* model distributions, the attribution analogue of single-peaked preferences.

**Finding 5 — Adversarial / edited / aligned models are the predicted failure regime, but the empirical signal is counterintuitive.** Etmann, Lunz, Maass & Schönlieb (ICML 2019; arXiv 1905.04172) show that adversarially-trained networks produce *more* interpretable saliency maps, not less, and hypothesize that "as the distance to the decision boundary grows, so does the alignment" — surprising and a hint that the worst-case failure mode is closer to natural-training pathologies than to adversarial training. Conversely, model editing is *exactly* Bilodeau's adversarial construction in operational form: a rank-1 weight update changes one factual association while leaving the loss-landscape near the original input untouched (Meng et al. 2022 ROME; Meng et al. 2023 MEMIT; Gupta et al. 2024 EMMET unified framework, arXiv 2403.14236). Atanasova et al. (arXiv 2502.18848) use MEMIT explicitly to construct faithful/unfaithful explanation pairs as a "causal diagnosticity" benchmark for faithfulness metrics. Adebayo et al. ("Sanity Checks for Saliency Maps," NeurIPS 2018; arXiv 1810.03292) showed empirically that "some existing saliency methods are independent both of the model and of the data generating process," and (arXiv 2011.05429, NeurIPS 2020) that "a class of methods, that modify the back-propagation algorithm are invariant to the higher layer parameters of a deep network; hence, ineffective for diagnosing model contamination." Krishna, Han, Gu, Pombra, Jabbari, Wu & Lakkaraju (TMLR 2024; arXiv 2202.01602) document "the disagreement problem": IG, SHAP, LIME, and SmoothGrad routinely disagree on top-k features in practice.

**Finding 6 — Quantification tools already exist in adjacent fields.** Wasserstein Distributionally Robust Optimization (Mohajerin Esfahani & Kuhn, Math. Prog. 171(1):115–166, 2018; tutorial Blanchet, Murthy & Si, arXiv 2108.02120) provides the natural framework for an IGA(ε) curve where ε is the radius of the Wasserstein ball around the natural-data distribution. Smoothed analysis itself (Spielman–Teng, JACM 2004) provides the framework for an IGA(σ) curve where σ is the Gaussian-perturbation magnitude on weights. Condition numbers from numerical linear algebra and the Bayes-risk vs. minimax-risk duality (Bayes_risk ≤ minimax_risk, with equality at the least-favorable prior — Wald 1950; cf. statistical decision theory expositions by Yihong Wu, Yale) are the right inspiration for an "attribution condition number" defined as the smallest distribution shift that collapses IGA.

---

## Details

### Q1. Worst-case vs. average-case impossibility — five concrete templates from other fields

1. **Compressed sensing (the canonical analogy).** ℓ₀-sparse recovery is NP-hard, and even certifying RIP is NP-hard (Bandeira, Dobriban, Mixon, Sawin, arXiv 1204.1580); average-case certification is also hard in tight regimes (Wang, Berthet, Plan, arXiv 1605.09646). Yet Candès–Romberg–Tao (IEEE TIT 52(2):489–509, 2006) and Donoho (IEEE TIT 52(4):1289–1306, 2006) prove that random Gaussian / partial-Fourier matrices satisfy RIP with overwhelming probability and ℓ₁ minimization recovers exactly. Structural analogy to attribution: Bilodeau's hard construction is to attribution what an "adversarially designed" measurement matrix is to compressed sensing — natural networks trained with SGD on natural data are to attribution what random Gaussian sensing matrices are to CS.

2. **Smoothed analysis (the perfect formal template).** Spielman & Teng (JACM 51(3):385–463, 2004; arXiv cs/0111050) introduced smoothed complexity as a hybrid between worst-case and average-case: a worst-case adversary chooses the instance, then nature adds Gaussian perturbation of standard deviation σ. The simplex method has smoothed complexity polynomial in d, n, 1/σ. Dadush & Huiberts (SIAM J. Comput. 49(5):STOC18-449–STOC18-499, 2020; arXiv 1711.05667) gave a substantially simplified analysis with a best-of-both-worlds bound O(d²σ⁻²√ln n + d³ ln^{3/2} n); Bach & Huiberts (FOCS 2025; arXiv 2504.04197) prove the optimal noise dependence. This framework applies *directly* to attribution: define smoothed_IGA(σ) = sup over base model f, expectation over weight perturbation w → w+N(0, σ²I) of IGA(f+noise). Bilodeau's adversarial construction collapses IGA at σ = 0; under any positive σ, the worst-case identifiability gap should grow continuously with σ. *This is the single best mathematical template for the paper.*

3. **Minimax vs. Bayes risk.** Wald's framework gives Bayes_risk(π) = inf_δ E_π R(δ,θ) ≤ sup_π Bayes_risk(π) ≤ minimax_risk = inf_δ sup_θ R(δ,θ). When the prior π concentrates on "non-adversarial" models, Bayes risk can be order-of-magnitude smaller than minimax. The minimax risk is achieved by the least-favorable prior; for attribution, Bilodeau's construction is essentially constructing a least-favorable prior over polynomial models with adversarial alignment. The minimax–Bayes duality (cf. Yale stat lecture notes by Y. Wu) gives the exact gap that an average-case theory of attribution should measure.

4. **PAC vs. agnostic learning.** Boosting (Schapire 1990; Freund–Schapire JCSS 1997) shows weak ≡ strong PAC learnability under realizable assumptions, but agnostic boosting requires substantially stronger weak learners. The attribution analogue: "realizable" attribution (the true counterfactual model behavior lies in a known restricted class) is tractable; "agnostic" attribution over Bilodeau's full polynomial class is impossible.

5. **No prior paper does average-case analysis of Bilodeau's impossibility.** This appears to be a clean gap. Fokkema, de Heide & van Erven (arXiv 2205.15834) prove a related impossibility ("Attribution-based Explanations that Provide Recourse Cannot be Robust") but again worst-case. The closest precedent in spirit is Sutter et al.'s NeurIPS 2025 *empirical* observation that random networks also get 100% intervention accuracy. Notably, this *complicates* the optimistic average-case story for attribution in the mechanistic-interpretability sub-field: depending on the construction, mechanistic interpretability may actually be worse on average than on adversarial worst case (because random networks are *typical* under the relevant prior, whereas Bilodeau's polynomials are *atypical*).

### Q2. Mathematical theory of method complementarity

The cleanest scaffolding is the **Krogh–Vedelsby ambiguity decomposition** (NIPS 1995). For squared loss and convex combination weights wᵢ summing to 1:
(f_ens − d)² = Σᵢ wᵢ(fᵢ − d)² − Σᵢ wᵢ(fᵢ − f_ens)².
Generalized to all twice-differentiable Bregman losses by Wood, Mu, Webb, Reeve, Luján & Brown (JMLR 24, 2023; arXiv 2301.03962) into bias–variance–diversity. **This decomposition can be ported directly to attribution if we define a "loss" between an attribution and the ground-truth counterfactual behavior** — e.g., signed-error loss on counterfactual model output. Combined attribution = weighted average of method attributions; ensemble IGA decomposes into average-method IGA plus an ambiguity (diversity) term.

Boosting (Schapire 1990, *Machine Learning* 5(2):197–227) gives the *qualitative* "weak → strong" story but uses re-weighting of training examples, not aggregation of disjoint information channels; for attribution, where each method is a different "view" of the model, the better analogy is **co-training** (Blum & Mitchell, COLT 1998): if two views are conditionally independent given the target (counterfactual model behavior), they are PAC-learnable from unlabeled data alone, with the redundancy bounded by Sridharan & Kakade's information-theoretic ε_info gap (COLT 2008): I(Y; X₂ | X₁) ≤ ε_info means view-2 adds at most ε_info bits beyond view-1.

Existing XAI combination work is purely empirical: NormEnsembleXAI (arXiv 2401.17200) uses normalization + max/min/mean aggregation; Meta-Explainers (Bello et al., Int. J. Intell. Syst. 2025; doi 10.1155/int/4841666) is a five-stage ensemble pipeline; the RF weak-learner explanation combination (arXiv 2402.19025); and "ensemble interpretation: a unified framework" (Neural Computing and Applications 2025; doi 10.1007/s00521-025-11130-8) which builds on Han–Srinivas–Lakkaraju's LFA. None of these prove a Krogh–Vedelsby-style identity for IGA or a co-training-style PAC bound for attribution. **This is the most underexploited gap.**

### Q3. Post-Arrow positive results — which transfer to attribution?

| Arrow escape route | Mechanism | Attribution analogue | Promise |
|---|---|---|---|
| Black (1948) single-peaked preferences | Restrict preference domain | Restrict model class (Lipschitz, depth-bounded, natural-training-distributed) | **High** — Bilodeau's Assumption 2 is the exact axiom to relax |
| Gibbard–Satterthwaite quasilinear / VCG | Allow side payments / cardinal information | Use cardinal model-behavior signal (gradient magnitudes, logit differences) rather than binary counterfactual labels | Medium |
| Sen's Paretian liberal | Relax universality or rights | Relax linearity (drop the Σϕⱼ = f(x) − E[f(X)] constraint) | **High** — gradient methods already trade completeness for local fidelity, and Bilodeau §D.1 proves gradient succeeds on the local-sensitivity sub-task |
| Gibbard (1977) randomization | Allow randomized voting | Randomized attribution (SmoothGrad, smoothed SHAP) | Medium — randomization alone doesn't seem to break worst-case impossibility |
| Approximation (Procaccia–Tennenholtz) | α-approximation truthful mechanisms | Approximate completeness with bounded gap | Medium |

**Most likely to replicate: domain restriction (a + d combined).** The empirical observation that SHAP achieves a meaningful identifiability gap on naturally trained networks (your reported IGA ≈ 2.0) is precisely the kind of "natural domain restriction makes the impossibility moot" story that Black's theorem tells for single-peaked preferences. The Sutter et al. NeurIPS 2025 finding that random networks already achieve perfect causal-abstraction alignment is, by contrast, the *negative* evidence for this strategy in mechanistic interpretability — but Bilodeau's setting is different (counterfactual behavior, not algorithm alignment), and SHAP's average-case success suggests the natural-training prior *does* concentrate away from worst case.

### Q4. Optimal method-portfolio design

The cleanest mathematical home is **robust Markowitz / minimax portfolio theory**:
- Robust Markowitz with ellipsoidal uncertainty sets (Tütüncü & Koenig 2004) and minimax-regret robust portfolios (Xidonas, Mavrotas, Hassapis & Zopounidis, EJOR 262:299–305, 2017; doi 10.1016/j.ejor.2017.04.005).
- Wasserstein DRO portfolio formulations (Blanchet, Chen, Zhou; cf. Mohajerin Esfahani & Kuhn 2018).
- A portfolio of K attribution methods {IG, SHAP, SAE, activation patching, linear probe} has weights w ∈ Δᴷ; objective is to choose w minimizing max over f ∈ ℱ_natural of identifiability gap of the combined attribution.

The right reformulation: under Krogh–Vedelsby decomposition, the optimal weights are those that maximize ambiguity (diversity) while keeping individual IGA bounded — exactly the negative-correlation-learning objective of Brown, Wyatt, Harris & Yao (2005). No published paper in XAI casts portfolio choice as robust optimization in this way; this is a clean contribution.

### Q5. When do attribution methods actually fail?

- **Adversarial training counterintuitively improves attribution faithfulness** (Etmann et al., ICML 2019; arXiv 1905.04172) — saliency-map / image alignment grows with distance to decision boundary. This is evidence *against* the hypothesis that adversarial training triggers Bilodeau-type failure; adversarial training appears to push models *toward* the Bilodeau-friendly regime.
- **Model editing (ROME / MEMIT / EMMET) is the operational realization of Bilodeau's construction**: Meng et al. 2022 ROME (rank-1 update); Meng et al. 2023 MEMIT (batched); Gupta et al. 2024 unified framework (arXiv 2403.14236). Atanasova et al. (arXiv 2502.18848) use MEMIT to construct faithful/unfaithful explanation pairs — a natural benchmark for whether attribution methods detect the edit. This is the strongest existing empirical instantiation of Bilodeau's theoretical concern.
- **Adebayo et al. ("Sanity Checks for Saliency Maps," NeurIPS 2018; arXiv 1810.03292)**: "some existing saliency methods are independent both of the model and of the data generating process" — guided backprop and deconvnet pass essentially unchanged after randomizing weights.
- **Adebayo et al. ("Debugging Tests for Model Explanations," NeurIPS 2020; arXiv 2011.05429)**: "a class of methods, that modify the back-propagation algorithm are invariant to the higher layer parameters of a deep network; hence, ineffective for diagnosing model contamination."
- **Krishna et al. (TMLR 2024; arXiv 2202.01602)**: documents that IG / SHAP / LIME / SmoothGrad routinely disagree on top-k features.
- **RLHF / alignment-induced cancellation**: I find no published study explicitly framing RLHF as a Bilodeau-style adversarial alignment that fools attributions. This is a clean empirical question.

### Q6. Tools for quantifying the worst-case-to-average-case gap

Recommended (in order of fit to attribution impossibility):

1. **Smoothed analysis (Spielman–Teng JACM 2004; arXiv cs/0111050; refined by Dadush–Huiberts 2020, Bach–Huiberts 2025).** Define smoothed_IGA(σ) = sup over base model f of expectation over weight Gaussian perturbation of IGA. At σ = 0 this is the worst-case (Bilodeau); large σ recovers something close to a uniform-over-model-class bound. The slope of this curve quantifies the gap.
2. **Wasserstein DRO (Mohajerin Esfahani & Kuhn 2018; tutorial Blanchet–Murthy–Si arXiv 2108.02120; Chu, Toh & Zhang JMLR 2022; Cranko et al. "Generalised Lipschitz regularisation equals distributional robustness," ICML 2021).** Replace adversarial choice of f with adversarial choice of distribution over (f, x) within an ε-Wasserstein ball of the natural-training distribution. IGA(ε) curve as ε varies from 0 (pure average-case) to ∞ (Bilodeau's worst case).
3. **Attribution condition number** (novel — by analogy with κ(A) = ‖A‖‖A⁻¹‖). Define κ_attr(f) = inf{‖δ‖ : IGA(f + δ) ≤ random-guessing}. Bilodeau's construction shows that within the full polynomial class κ_attr can be 0 (worst case); we conjecture κ_attr(f) is bounded below by a constant depending on training-data complexity and weight norm for natural networks.
4. **Bayes-risk / minimax-risk gap from Wald's statistical decision theory.** Define Bayes_IGA(π) and minimax_IGA = sup_π Bayes_IGA(π); the latter is Bilodeau's bound, the former is what experiments measure with the empirical training distribution.

---

## Recommendations

**Recommendation 1 (highest priority): Frame the paper as "smoothed analysis of attribution impossibility."** This is the cleanest formal template available. The pitch: Bilodeau is to attribution what worst-case simplex complexity is to LP; SHAP-works-on-natural-networks is to attribution what Spielman–Teng's polynomial smoothed complexity is to simplex. Concretely:
  - Define smoothed_IGA(σ).
  - Prove a positive lower bound: there exists C such that for σ > σ₀, smoothed_IGA(σ) ≥ C > random guessing (replicates Spielman–Teng's positive result).
  - Prove an upper bound recovering Bilodeau as σ → 0 (recovers worst case).
  - Verify experimentally that natural training places the model at an effective σ_eff bounded below.

**Recommendation 2: Build the method-portfolio theory on Krogh–Vedelsby + Wood et al.** Adopt the generalized bias–variance–diversity decomposition of Wood, Mu, Webb, Reeve, Luján & Brown (JMLR 24, 2023; arXiv 2301.03962) with the natural attribution loss; prove that for any K methods with individual IGA_i and pairwise ambiguity α_ij, the optimally weighted ensemble achieves IGA ≥ average_IGA + average_ambiguity. This gives the first formal "weak attribution methods become strong jointly" theorem.

**Recommendation 3: Replicate post-Arrow domain-restriction.** The Bilodeau theorem hinges on Assumption 2 (richness of model class). Show that for Lipschitz networks with bounded depth, or networks in the "natural training" distribution (formalized as e.g. NTK or feature-learning regime), the impossibility quantitatively weakens — analogous to Black's theorem rescuing majority rule under single-peaked preferences.

**Recommendation 4: Use ROME/MEMIT-edited models as the canonical empirical adversarial setting.** Atanasova et al. (arXiv 2502.18848) provide the benchmark infrastructure. Show: (a) on natural networks, attribution methods achieve a meaningful identifiability gap; (b) on ROME-edited models, IGA collapses near random guessing — operationally demonstrating Bilodeau's theoretical worst case.

**Benchmarks/Thresholds that would change the recommendation:**
- If smoothed_IGA(σ) collapses for *any* σ > 0, abandon smoothed analysis and pivot to pure Wasserstein DRO.
- If method-portfolio ambiguity is empirically near zero (methods are highly redundant in practice), the Krogh–Vedelsby angle is dead and "no free lunch" is the honest framing.
- If ROME-edited models *don't* break attribution methods (i.e., the disagreement is small), then Bilodeau's worst case is non-physical and the right framing becomes "attribution works on all interesting models — the impossibility is mathematically real but practically vacuous."

---

## Concrete Research Plan (from experiment to paper)

**Stage 1 (1–2 months): Establish the empirical phenomenon precisely.**
- Reproduce a meaningful IGA for SHAP / IG on standard networks (Bilodeau's setup; OpenXAI benchmark — Agarwal, Krishna, Saxena, Pawelczyk, Johnson, Puri, Zitnik, Lakkaraju, NeurIPS 2022 D&B; arXiv 2206.11104).
- Construct an empirical "worst-case to average-case axis" by interpolating between natural networks and Bilodeau's adversarial polynomial construction.
- Apply ROME/MEMIT edits to standard LLMs; measure IGA on the edited fact vs. control facts (Atanasova et al., arXiv 2502.18848 infrastructure).

**Stage 2 (2–4 months): Define and bound the two key quantities.**
- Define smoothed_IGA(σ) formally.
- Prove a smoothed Bilodeau theorem: for σ > σ_critical(model class), smoothed_IGA(σ) ≥ 1 + g(σ) for an explicit g.
- Define the attribution condition number κ_attr(f); prove for Lipschitz networks with constant L, κ_attr ≥ c/L for an explicit c.

**Stage 3 (2–3 months): Method-portfolio theory.**
- Generalize Krogh–Vedelsby / Wood et al. to the attribution loss.
- Prove a co-training-style result (à la Sridharan–Kakade 2008): if two attribution methods provide conditionally-independent information about counterfactual behavior given f, their combination achieves identifiability with sample (or query) complexity that is the sum, not product, of individual complexities.
- Empirically verify on {IG, SHAP, SmoothGrad, gradient×input, linear probe}.

**Stage 4 (1–2 months): Synthesis and paper.**
- Frame as the attribution analogue of Spielman–Teng for simplex.
- Title candidate: "Smoothed Analysis of Feature Attribution: Why Impossibility Theorems Don't Predict Practice."
- Target venue: NeurIPS or ICML (theory track), with extended journal version to TMLR or JMLR.

---

## Caveats

- The most ambiguous claim in the brief was "Sutter NeurIPS 2025." The actual paper is by **Denis Sutter (ETH Zürich), not Thomas Sutter**, and concerns mechanistic interpretability / causal abstraction (NeurIPS 2025 Spotlight; arXiv 2507.08802), not feature attribution per se. Its result that "any neural network can be mapped to any algorithm" under unrestricted alignment maps is an impossibility for a different sub-field, but the same worst-case-vs-average-case framing applies.
- Sutter et al.'s finding that *random* networks achieve perfect causal-abstraction alignment is empirical evidence *against* a naive average-case story in mechanistic interpretability — random networks are arguably the "average case" under a uniform-over-weights prior, and they fail. This should be discussed honestly in any paper.
- Etmann et al.'s (ICML 2019) finding that adversarially-trained networks have *more* interpretable saliency maps is also against the intuition that adversarial training triggers Bilodeau failure. The intuitive story "natural training = good attribution, adversarial training = bad attribution" is not supported; the right axis is probably training-data distribution, not training-method.
- The user's reported empirical figure of IGA ≈ 2.0 is a project-internal number; the published Bilodeau et al. PNAS 2024 results show that "for simple tabular data, [gradient] methods can outperform SHAP and IG, while for image data all methods are comparable to random guessing." The "natural-training prior is non-adversarial" claim needs to be supported by your own experiments — published evidence is mixed.
- The attribution condition number proposed in Recommendation 3 is novel and unproven; the conjecture that it is bounded below for Lipschitz networks is plausible but not established.
- I did not find a confirmed published paper drawing an explicit Arrow → attribution analogy with full formal escape routes worked out. Kleinberg (NIPS 2002) draws the Arrow → clustering analogy explicitly, and Brcic & Yampolskiy (arXiv 2109.00484, "Impossibility Results in AI: A Survey") catalog impossibility results across AI but do not give the specific Arrow → attribution mapping proposed here. This itself is a clean novelty axis for the paper.

---

## Key Paper List (prioritized by relevance)

**Tier 1 — directly on attribution / interpretability impossibility:**
1. Bilodeau, Jaques, Koh, Kim, "Impossibility Theorems for Feature Attribution," PNAS 121(2):e2304406120, 2024 — arXiv 2212.11870.
2. Han, Srinivas, Lakkaraju, "Which Explanation Should I Choose? A Function Approximation Perspective…," NeurIPS 2022 — arXiv 2206.01254.
3. Sutter, Minder, Hofmann, Pimentel, "The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?," NeurIPS 2025 Spotlight — arXiv 2507.08802.
4. Kleinberg, "An Impossibility Theorem for Clustering," NIPS 2002.
5. Fokkema, de Heide, van Erven, "Attribution-based Explanations that Provide Recourse Cannot be Robust" — arXiv 2205.15834.
6. Brcic, Yampolskiy, "Impossibility Results in AI: A Survey" — arXiv 2109.00484.

**Tier 2 — worst-case vs. average-case templates:**
7. Spielman, Teng, "Smoothed Analysis of Algorithms: Why the Simplex Algorithm Usually Takes Polynomial Time," JACM 51(3):385–463, 2004 — arXiv cs/0111050.
8. Candès, Romberg, Tao, "Robust Uncertainty Principles…," IEEE TIT 52(2):489–509, 2006.
9. Donoho, "Compressed Sensing," IEEE TIT 52(4):1289–1306, 2006.
10. Bandeira, Dobriban, Mixon, Sawin, "Certifying the RIP is Hard" — arXiv 1204.1580.
11. Wang, Berthet, Plan, "Average-case Hardness of RIP Certification" — arXiv 1605.09646.
12. Dadush, Huiberts, "A Friendly Smoothed Analysis of the Simplex Method," SIAM J. Comput. 49(5):STOC18-449–STOC18-499, 2020 — arXiv 1711.05667.
13. Bach, Huiberts, "Optimal Smoothed Analysis of the Simplex Method," FOCS 2025: 1829–1856 — arXiv 2504.04197.

**Tier 3 — method combination / ensemble theory:**
14. Schapire, "The Strength of Weak Learnability," Machine Learning 5(2):197–227, 1990 (doi 10.1007/BF00116037).
15. Freund, Schapire, "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting," JCSS 55(1):119–139, 1997 (doi 10.1006/jcss.1997.1504).
16. Krogh, Vedelsby, "Neural Network Ensembles, Cross Validation, and Active Learning," NIPS 1995.
17. Wood, Mu, Webb, Reeve, Luján, Brown, "A Unified Theory of Diversity in Ensemble Learning," JMLR 24, 2023 — arXiv 2301.03962.
18. Blum, Mitchell, "Combining Labeled and Unlabeled Data with Co-Training," COLT 1998.
19. Sridharan, Kakade, "An Information Theoretic Framework for Multi-view Learning," COLT 2008, pp. 403–414.
20. Brown, Wyatt, Harris, Yao, "Diversity Creation Methods: A Survey and Categorisation," 2005.

**Tier 4 — social choice escape routes:**
21. Arrow, *Social Choice and Individual Values*, 1951/1963.
22. Black, "On the Rationale of Group Decision-making," J. Pol. Econ. 56:23–34, 1948.
23. Gibbard, Econometrica 41:587–601, 1973; Satterthwaite, JET 10:187–217, 1975.
24. Bade, "Gibbard–Satterthwaite Success Stories and Obvious Strategyproofness" — arXiv 1610.04873.
25. Barberà, "Arrow on Domain Conditions: A Fruitful Road to Travel," BSE Working Paper 1095.
26. Ma, Meir, Parkes, "Social Choice with Non Quasi-linear Utilities" — arXiv 1804.02268.

**Tier 5 — attribution failure modes empirically:**
27. Adebayo et al., "Sanity Checks for Saliency Maps," NeurIPS 2018 — arXiv 1810.03292.
28. Adebayo et al., "Debugging Tests for Model Explanations," NeurIPS 2020 — arXiv 2011.05429.
29. Krishna et al., "The Disagreement Problem in Explainable Machine Learning," TMLR 2024 — arXiv 2202.01602.
30. Agarwal et al., "OpenXAI: Towards a Transparent Evaluation of Model Explanations," NeurIPS 2022 D&B — arXiv 2206.11104.
31. Etmann, Lunz, Maass, Schönlieb, "On the Connection Between Adversarial Robustness and Saliency Map Interpretability," ICML 2019 — arXiv 1905.04172.
32. Meng et al., "Locating and Editing Factual Associations in GPT" (ROME), NeurIPS 2022.
33. Meng et al., "Mass-Editing Memory in a Transformer" (MEMIT), ICLR 2023.
34. Gupta et al., "A Unified Framework for Model Editing" (EMMET) — arXiv 2403.14236.
35. Atanasova et al., "A Causal Lens for Evaluating Faithfulness Metrics" — arXiv 2502.18848.

**Tier 6 — quantification tools (Wasserstein DRO, condition numbers, statistical decision theory):**
36. Mohajerin Esfahani, Kuhn, "Data-driven distributionally robust optimization using the Wasserstein metric," Math. Prog. 171(1):115–166, 2018.
37. Blanchet, Murthy, Si, "Statistical Analysis of Wasserstein Distributionally Robust Estimators" — arXiv 2108.02120.
38. Cranko, Shi, Zhang, Nock, Kornblith, "Generalised Lipschitz Regularisation Equals Distributional Robustness," ICML 2021.
39. Chu, Toh, Zhang, "On Regularized Square-Root Regression Problems: Distributionally Robust Interpretation and Fast Computations," JMLR 23(308):1–39, 2022.
40. Xidonas, Mavrotas, Hassapis, Zopounidis, "Robust Multiobjective Portfolio Optimization: A Minimax Regret Approach," EJOR 262:299–305, 2017.

**Tier 7 — existing XAI ensemble work:**
41. NormEnsembleXAI — arXiv 2401.17200.
42. Bello et al., "Meta-Explainers," Int. J. Intell. Syst. 2025; doi 10.1155/int/4841666.
43. "Combination of Weak Learners eXplanations to Improve Random Forest Explicability Robustness" — arXiv 2402.19025.
44. "Ensemble interpretation: a unified framework," Neural Computing and Applications 2025; doi 10.1007/s00521-025-11130-8.

