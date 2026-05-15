### SAE identifiability research design

用户:


Claude:
his is a rich and well-scoped research design question. The prompt is detailed enough that I can dive straight into research — let me pull together the latest on SAE identifiability, the Paulo & Belrose findings, and related experimental methodologies to help you design a comprehensive experiment plan.
our systematic experiment plan for studying Sparse Autoencoder (SAE) identifiability in mechanistic interpretability is ready. It builds on the Paulo & Belrose finding that seed-different SAEs share only ~30% of features and proposes a full reproduction across Llama-3 8B, Gemma-2 2B, and Pythia-1.4B, with ablations over sparsity, width, training tokens, layer position, and architecture (ReLU+L1, TopK, Gated, JumpReLU, Matryoshka). The plan evaluates five improvement strategies (weight regularisation, OrtSAE-style orthogonality, Matryoshka nesting, Mutual Feature Regularisation, and contrastive/archetypal objectives), specifies a six-metric panel for feature sharing (MMCS, Hungarian matching, mutual NN, activation correlation, CKA, and SCR/TPP causal metrics), grounds the design in theoretical references (Spielman, Engels, Cunningham, Templeton), and includes M4 Max pre-flights, a ~6,000 A100-hour budget, a 16-week timeline, and explicit positive/negative success criteria with implications for the field.
# A Systematic Experiment Plan for Studying Sparse Autoencoder Identifiability in Mechanistic Interpretability

## 1. Motivation and Scientific Context

Paulo & Belrose (arXiv:2501.16615, Jan 2025) report that two Sparse Autoencoders (SAEs) trained with the *same* model, *same* data, *same* architecture, and *same* sparsity level — but with different random seeds — share only roughly 30 % of their features on a 131 K-latent SAE applied to a Llama-3 8B feedforward layer. The phenomenon replicates across Llama-3 8B, Gemma-2 2B, and Pythia-1.4B, two datasets (RedPajama / FineWeb-style mixes and The Pile), and several architectures (ReLU + L1, TopK, Gated). TopK SAEs were *more* seed-sensitive than ReLU-L1 SAEs at matched L0. The authors' conclusion — that SAE decompositions should be regarded as *pragmatic* rather than canonical recoveries of "true" features — is now widely cited (e.g., O'Neill et al. 2025; Bussmann et al. 2025; Karvonen et al. 2025).

This finding directly challenges the implicit identifiability assumption inherited from classical sparse dictionary learning (Spielman, Wang & Wright 2012; Gribonval & Schnass 2010). Spielman's ER-SpUD result guarantees exact recovery of a square dictionary from O(n log n) samples *provided* coefficients are drawn from an i.i.d. Bernoulli-Gaussian model with sufficient sparsity. LLM activations violate every assumption: dictionaries are massively overcomplete (typically 8×–64× expansion), coefficients are highly correlated (feature co-occurrence drives "absorption"; Chanin et al. 2024), the linear-representation hypothesis itself is incomplete (Engels et al., arXiv:2405.14860, show genuinely multi-dimensional / circular features), and activation distributions are far from i.i.d. The Paulo–Belrose result is therefore *expected* theoretically but had not been measured at scale.

The plan below addresses five questions in order: (i) how to reproduce, (ii) how to ablate, (iii) how to mitigate, (iv) how to measure, and (v) what counts as a finding.

## 2. Reproduction Experiment Design

### 2.1 Models, layers and data
- **Primary model:** Llama-3.1 8B-base (replacing the original Llama-3 to stay aligned with Llama-Scope, arXiv:2410.20526; same MLP/residual structure).
- **Secondary models for robustness:** Pythia-1.4B-deduped and Gemma-2 2B-base.
- **Hookpoints:** to match Paulo & Belrose, train SAEs on the MLP-output of layer 16 (the FFN they highlighted), residual-stream post-attention layer 12, and an early residual layer (layer 4). This brackets early/mid/late behaviour and matches Llama-Scope canonical layers.
- **Datasets:** primary corpus FineWeb-Edu (10 B tokens shuffled); secondary The Pile sample for cross-dataset check. Tokenise to 1024-token sequences. Cache activations to disk (about 30 GB per layer per 200 M tokens at bfloat16).

### 2.2 SAE configurations
Match Llama-Scope and Paulo–Belrose for direct comparability:
- **Architectures:** (a) ReLU + L1 (Bricken/Cunningham style), (b) TopK (Gao et al. 2024) with K∈{32, 64, 128}, (c) Gated SAE (Rajamanoharan et al. 2024a), (d) JumpReLU (Rajamanoharan et al. 2024b), (e) Matryoshka TopK (Bussmann et al. 2025) with nested widths {2¹², 2¹⁴, 2¹⁶}. Optionally BatchTopK (Bussmann et al., arXiv:2412.06410).
- **Latent widths (d_SAE):** primary at 32 K (expansion ≈ 8×), with a single 131 K run to match the headline result. Pre-experiments at 4 K and 16 K.
- **Training:** Adam, lr 3 × 10⁻⁴ → cosine to 3 × 10⁻⁵; batch 8192 tokens; 200 M – 1 B tokens; K-annealing (start K = 4096 → final K) per Llama-Scope; decoder columns unit-normalised; dead-feature resampling every 25 M tokens; bf16 activations / fp32 params.
- **Seeds:** 5 seeds per configuration as the basic unit of analysis (Paulo & Belrose used 3 — 5 yields a usable 10-pair distribution of MMCS).

### 2.3 Evaluation metrics (run every configuration through all)
- **Reconstruction:** NMSE (variance-unexplained), Δ-cross-entropy when SAE is spliced in.
- **Sparsity:** mean L0 over a 50 K-token validation split.
- **Identifiability / feature sharing:** see §5.
- **Quality via SAEBench (arXiv:2503.09532):** the eight evaluations cover Concept Detection, Interpretability (auto-interp with an LLM judge), Reconstruction, and Feature Disentanglement (absorption, SCR, TPP). Use SAEBench v1.0 code and Neuronpedia comparison harness.

### 2.4 Compute budget
A 32 K-latent TopK SAE on Llama-3 8B layer-16 MLP (4096 dims) trained on ~400 M tokens takes ≈ 8–12 GPU-hours on a single A100-80GB once activations are cached. With 4 × A100, one such SAE trains in ~3 hours.
- **Per "cell" (model × layer × architecture × width × K):** 5 seeds × ~3 h = 15 GPU-hours on 4×A100 (or ~30 GPU-h on 8×A40 due to slower interconnect / 48 GB memory).
- **Core reproduction matrix** (3 models × 3 layers × 4 architectures × 2 widths × 5 seeds = 360 SAE trainings, dominated by Llama-3) ≈ 3 000 GPU-hours ≈ 31 wall-clock days on 4×A100 if dedicated. A single 131 K-latent Llama-3 SAE will cost ~80 GPU-hours per seed; budget five seeds at 400 GPU-hours total. Including activation caching (one-pass forward), SAEBench eval (~20 GPU-h per SAE for the heavy auto-interp tasks), and slack, the reproduction phase fits in **≈ 5 000–6 000 A100-hours**, well within a six-week run on a 4×A100 node.

### 2.5 M4 Max (32 GB) pre-experiments
The MPS backend in PyTorch handles SAEs comfortably below ~16 K latents on small models.
- **Pre-flight #1:** Pythia-160M (768 dims), layer 6 MLP, SAE width 4 K, TopK K=32, 50 M tokens. ~4 hours on M4 Max; verifies pipeline, MMCS sharp drop at low expansion, dead-feature behaviour, code correctness.
- **Pre-flight #2:** GPT-2 small (768 dims), layer 6, width 8 K. Replicate the Cunningham et al. (arXiv:2309.08600) / Bricken et al. (2023) setup so that absolute MMCS values can be sanity-checked against published numbers (Bricken's 8 192-feature, 16× expansion run; bimodal MCS distributions observed by aizi).
- **Pre-flight #3:** Toy ground-truth dictionary with planted features (Spielman regime) to confirm metric implementations recover MMCS ≈ 1 when the planted dictionary is known. This is essential to separate metric bugs from genuine identifiability failure.

All M4 Max runs use `mps` with float32 (bf16 unreliable on Apple silicon) and offline activation caching to NVMe.

## 3. Ablation Experiments

### 3.1 Sparsity
Vary K ∈ {16, 32, 64, 128, 256} for TopK and L1-coefficient λ ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³, 10⁻²} for ReLU SAEs, holding L0 matched across architectures via a calibration sweep. Hypothesis (from Paulo & Belrose and Chanin & Garriga-Alonso 2025): sharing rate is non-monotone in K — too sparse forces feature absorption (lower sharing), too dense leaves many low-importance directions free to vary (also lower sharing). Expect a maximum near the "Pareto knee" (Llama-Scope identifies K=50–100 for 32 K latents).

### 3.2 Dictionary size / expansion factor
Run widths {2¹², 2¹³, 2¹⁴, 2¹⁵, 2¹⁶, 2¹⁷} (4 K → 131 K). Paulo & Belrose's headline (~30 % shared at 131 K) suggests sharing degrades monotonically with width. We measure both global sharing rate and the **scale-decomposed** rate: fraction of features whose nearest neighbour at width 2w is also nearest neighbour at width w (a "feature splitting" indicator following Bricken et al.).

### 3.3 Training-data volume
Replicate at {50 M, 200 M, 1 B, 4 B} tokens to test the "undertraining" hypothesis raised by the original LessWrong follow-up to Cunningham et al. — namely that low MMCS may reflect insufficient training. If MMCS continues to climb monotonically to 4 B tokens, the identifiability problem is partly an optimisation artefact; if it plateaus, it is intrinsic.

### 3.4 Model scale and layer position
Across Pythia-1.4B, Gemma-2 2B, Llama-3 8B, run a fixed configuration (32 K width, TopK K=64, 400 M tokens, 5 seeds) at layers {early, middle, late}. Engels et al. show that multi-dimensional / circular features concentrate in middle layers; we expect identifiability to be *worst* there because the SAE must impose a 1-D basis on inherently 2-D manifolds.

### 3.5 Architecture
Pre-registered ranking hypothesis (from SAEBench's feature-disentanglement result and OrtSAE, arXiv:2509.22033): Matryoshka ≈ OrtSAE > Gated ≈ JumpReLU > BatchTopK > TopK > ReLU+L1 in *quality* metrics, but with respect to *seed identifiability*: ReLU+L1 highest, Matryoshka and OrtSAE next, TopK lowest — preserving Paulo & Belrose's L1-is-most-stable observation while testing whether modern explicit-orthogonality / hierarchical losses recover stability.

## 4. Improvement Strategies

Five families of interventions, each implemented and evaluated against the same reproduction matrix:

1. **Weight / activity regularisation.** L2 weight penalty on encoder/decoder (Wang et al. arXiv:2603.04198 report doubled shared-feature fraction and doubled steering success on Pythia-70M TopK SAEs with a small L2). Decorrelation losses (Cogswell et al. 2016). **Masked-token regularisation** (Narayanaswamy et al. arXiv:2604.06495) — randomly replace tokens during activation collection to break co-occurrence — reduces absorption and OOD gap.

2. **Orthogonality / atomicity constraints.** OrtSAE adds a chunk-wise pairwise cosine-similarity penalty on decoder columns; the authors report 9 % more distinct features, 2.7× lower MeanCosSim than BatchTopK, and competitive reconstruction. Implement OrtSAE on TopK and Gated bases; measure whether enforced atomicity translates into seed-stability.

3. **Hierarchical / nested objectives.** Matryoshka SAE (Bussmann et al. 2025) trains nested dictionaries of geometric size; smaller prefixes must reconstruct independently, which prevents large-width specialists from absorbing the small-width generalists. Test whether nesting also stabilises features across seeds (a hypothesis SAEBench supports: Matryoshka's disentanglement advantage *grows* with width, mirroring the regime where identifiability collapses).

4. **Co-training / mutual regularisation.** Mutual Feature Regularisation (Marks, Paren, Krueger, Barez, arXiv:2411.01220) trains N parallel SAEs and adds an auxiliary loss encouraging their feature dictionaries to align (with optional conditional re-initialisation of features that fail to be shared). Theoretically motivated by the observation that features recovered by multiple independent runs are more likely to be "real". Directly addresses the Paulo–Belrose phenomenon by construction; we propose to scale MFR from its original synthetic / GPT-2 small setting to Llama-3 8B and measure whether the gain in shared features is achieved without quality loss.

5. **Initialisation and contrastive approaches.** (a) Initialise the second seed's encoder from the first seed's decoder transpose ("warm-restart" — measures how much variability is purely basin-of-attraction). (b) Non-negative contrastive / shift-based identifiable objectives (Sparse Shift Autoencoders, Hewitt et al. arXiv:2502.12179) which leverage paired data to guarantee identifiability up to permutation/scaling — adaptable to paired activations from minimally-different prompts. (c) Archetypal SAE (arXiv:2502.12892) and "ordered" SAE (arXiv:2512.02194) constrain features to lie in the convex hull of the data or impose an explicit order, both of which provably tighten identifiability.

Each method is judged against TopK and Gated baselines at the same L0 and reconstruction quality.

## 5. Metrics for Feature Sharing

No single metric suffices; we use a panel.

- **Mean Max Cosine Similarity (MMCS).** For dictionaries D₁, D₂ compute, for every column of D₁, max_j cos(d₁ᵢ, d₂ⱼ); average. Paulo & Belrose and Sharkey et al. use this. Invariant to permutation/positive rescaling. Pitfalls: bimodal MCS distributions (aizi 2024) — a single mean hides them; the random baseline at large width is non-trivial (the lesswrong "bimodal" post showed the <0.5 tail matches Gaussian-random dictionaries).
- **Hungarian / bipartite optimal matching.** Solve assignment between D₁ and D₂ maximising total cosine similarity; report the fraction of pairs above a threshold τ (Paulo & Belrose use τ = 0.6 and 0.7; this gives the "30 % shared" headline). Both encoder and decoder matchings should be reported; the dual check (do both directions agree?) is more stringent.
- **Mutual nearest neighbours.** Pair is counted only if i is j's best in D₂ *and* j is i's best in D₁. More conservative; closer to "unique recovery".
- **Activation correlation.** For matched pairs, compute Pearson r of feature activations on a held-out token set. Geometric similarity does not entail functional similarity: a feature pair with cos 0.9 but activation r=0.3 is not the same feature.
- **CKA / representational similarity.** Centered Kernel Alignment between the full activation matrices F₁, F₂ across many tokens measures whether the two SAEs span the *same subspace* even if individual atoms differ. CKA close to 1 with low MMCS would mean "different bases for the same space" — a much weaker form of non-identifiability than orthogonal features.
- **Causal / functional metrics.** From SAEBench: SCR (Spurious Correlation Removal), TPP (Targeted Probe Perturbation), and feature-absorption tests. A feature pair is "truly shared" only if ablating each produces matched downstream effects. We treat this as the gold standard.

The plan is to report all six on every comparison, since recent work (Karvonen et al. 2025) shows proxy metrics correlate poorly with downstream practical metrics.

## 6. Theoretical Reference Points

- **Spielman et al. 2012.** Guarantees apply to *square* dictionaries with Bernoulli–Gaussian coefficients. LLM SAEs are 8–64× overcomplete, coefficients are correlated, and feature directions are not random. Recent work (Cui et al. 2025; arXiv:2512.05534 on piecewise biconvexity) characterises spurious minima in tied-weight SAEs and shows that SAE optimisation admits multiple equally good solutions even under the linear-representation hypothesis. We treat Spielman as a sanity bound on the toy / planted experiments and as the explanation for why scaling alone cannot fix identifiability.
- **Engels et al. (arXiv:2405.14860).** Demonstrates that some features (days of week, months of year, modular-arithmetic computations) are inherently 2-D / circular. Any SAE that imposes a 1-D linear basis must split such a feature inconsistently across seeds — predicting the identifiability problem will be *worst* on tokens involving these features. A targeted sub-experiment uses Engels' circular-feature task as a probe.
- **Cunningham et al. (arXiv:2309.08600) and Bricken et al. (Anthropic, Oct 2023).** Establish the standard SAE-validation pipeline: auto-interpretability (LLM-judged neuron explanations), human spot-checks, and ablation-based causal proxies. Bricken's one-layer transformer + 8192-feature dictionary remains the canonical "ground-truth-ish" testbed and a useful sanity layer between the M4 Max pre-flights and full Llama-3 runs.
- **Templeton et al. 2024 (Scaling Monosemanticity).** Provides scaling-law methodology (joint sweep of width and tokens; choose L0 by Pareto frontier). We adopt their scaling-law-guided width selection and their feature-steering protocol for the causal-functional metric.

## 7. Success Criteria and Implications

**A meaningful positive finding** (improvement strategies work):
- An intervention that *raises* the bipartite shared-feature fraction at τ=0.7 from ~30 % to **≥ 60 %** on Llama-3 8B at 131 K latents, **without** more than a 5 % regression in NMSE or Δ-CE, and **without** SAEBench Concept-Detection / Feature-Disentanglement regression. Bonus: improved auto-interpretability and steering reliability.
- We additionally require that the gain replicate on Gemma-2 2B and Pythia-1.4B and across both datasets — otherwise the intervention is overfit.

**A meaningful negative finding** (identifiability is intrinsic):
- All five intervention families fail to push shared-feature fraction above ~40 % at 131 K width even at matched quality. This would corroborate the theoretical work (Cui et al. 2025; the piecewise-biconvexity result) that the SAE objective is fundamentally non-identifiable in the LLM regime and would motivate the field to move toward (i) committee-style SAEs (always report ensembles), (ii) functionally-defined features (causal interventions rather than dictionary atoms), or (iii) richer geometric primitives (manifold features per Engels et al.).

**Implications for mechanistic interpretability** in either case:
- *Positive:* identifiability-stable SAEs make individual latents legitimate scientific objects, justifying claims like "feature 4327 in SAE X corresponds to Golden Gate Bridge".
- *Negative:* such claims are seed-dependent and must be replaced by claims about *equivalence classes* of features under representational similarity, or by causal-functional definitions. Sparse Feature Circuits, model-steering, and unlearning research would need to recompute on multiple seeds and report only intersections.

## 8. Timeline and Risk Register

**Weeks 1–2 (M4 Max):** pipeline, planted-dictionary sanity, GPT-2-small Bricken reproduction.
**Weeks 3–4 (4×A100):** Pythia-1.4B + Gemma-2 2B full reproduction matrix; verify ~30 % headline.
**Weeks 5–8:** Llama-3 8B at 32 K and 131 K widths, all architectures, 5 seeds.
**Weeks 9–10:** Ablations (sparsity, width, data, layer).
**Weeks 11–14:** Improvement strategies — MFR, OrtSAE, Matryoshka, weight-reg, masked-reg, contrastive.
**Weeks 15–16:** SAEBench evaluation of every saved SAE; write-up.

Key risks: (1) under-training masquerading as non-identifiability — mitigated by the data-volume ablation; (2) metric ambiguity — mitigated by the six-metric panel and planted-dictionary calibration; (3) architectural confounds in TopK at small K (dead features) — mitigated by K-annealing and resampling; (4) the "dead salmon" risk (Heap et al. 2025) that auto-interp scores look fine even for random networks — mitigated by including SCR/TPP causal metrics from SAEBench; (5) compute over-run on the 131 K-latent runs — mitigated by activation caching and the option to fall back to 65 K width if needed.

The combined plan is technically feasible on the stated hardware (≈ 6 000–8 000 A100-hours including ablations and improvements), uses only open models and datasets, and produces unambiguous evidence on whether SAE identifiability is a solvable engineering problem or a fundamental limitation of the dictionary-learning frame for mechanistic interpretability.

