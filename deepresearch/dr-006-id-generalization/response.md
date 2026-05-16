### Causal intervention on representation geometry

用户:


Claude:
his is a rich and well-scoped research design question — you've clearly laid out the gap between correlation and causation for ID → generalization. Let me dive into a deep research session to pull together the latest methods for causal intervention on representation geometry, ID estimation during training, and relevant experimental designs.

Starting my research now.
our causal intervention protocol for studying intrinsic dimension and generalization is ready. It outlines a 7-arm experimental design (4 ID-reducing treatments, 3 confounder-matched sham controls, and 1 ID-increasing anti-treatment) tested on CIFAR-10/100, Tiny-ImageNet, and modular arithmetic tasks with ResNet-18 and ViT-Small. The protocol covers differentiable ID regularizers (TwoNN, spectral entropy, Jacobian nuclear norm, persistent homology), formal causal identification via mediation analysis and instrumental variables, pre-registered success/falsification criteria, compute budgets for both 4×A100 and M4 Max setups, and a 5-stage staged execution plan with explicit go/no-go thresholds. It also situates the work against two recent 2026 preprints (Yadav, Khanh et al.) and the cautionary Saxe et al. 2018 information-bottleneck precedent.
# A Causal Intervention Protocol for Intrinsic Dimension and Generalization

## TL;DR
- **Design the experiment as a 7-arm intervention study** (4 ID-reducing treatments + 3 confounder-matched sham controls + 1 ID-increasing anti-treatment) on CIFAR-10/100 with ResNet-18 and ViT-Small, treating the question as a mediation problem rather than a correlation study. The single most decisive control is a **norm/spectral-norm-matched sham** in the style of Khanh et al. (arXiv:2604.13123) — that contrast is what separates a causal claim from yet another correlation.
- **Prior 2026 "causal" claims are suggestive but not definitive.** Yadav (arXiv:2602.00130) reports r=−0.94 (Gaussian noise) and a *mean −0.03pp accuracy loss at 95% PCA-variance projection across ResNet18, ResNet34, DenseNet121* on 52 ImageNet classifiers; Khanh et al. (arXiv:2604.13123, four authors across Clevix LLC and Banking Academy of Vietnam) report a +5,020-step grokking delay at p=0.044 (Cohen's d=0.70) plus a norm-matched control at n=30, p=5×10⁻⁵, Cohen's d=1.55 — but they explicitly state "we do not claim a fully causal relationship in the Pearl/Rubin sense." A properly-powered (≥10 seeds × 5+ dose levels × 3 sham controls × 4 datasets) protocol on standard vision benchmarks would substantially advance the field.
- **The fastest falsifier**: if a differentiable TwoNN-penalty surgically lowers last-layer ID by ≥30% while a norm-matched, rank-matched control achieves the *same* generalization gain, the ID→generalization causal hypothesis is falsified — ID is then a marker, not a cause.

## Key Findings

1. **Two existing 2026 preprints already attempt causal interventions** on representation geometry: Yadav (arXiv:2602.00130) via Gaussian/Uniform/Dropout/Salt-and-pepper noise and PCA projection (with reported correlations r=−0.94, −0.96, −0.91, −0.99 respectively, all p<10⁻⁹), and Khanh et al. (arXiv:2604.13123) via penultimate-layer "representation mixing" with a norm-matched sham control on a 1-layer Transformer doing modular arithmetic. Both report positive but limited evidence; neither uses formal mediation/IV machinery, and Khanh et al. explicitly note their intervention "controls for parameter norm but does NOT control for changes in training dynamics induced by the intervention itself."

2. **A differentiable ID-regularization toolkit already exists.** Birdal et al. (arXiv:2111.13171, NeurIPS 2021) provide a differentiable persistent-homology ID estimator usable as a training loss (TorchPH, α=1 default). DeMoss et al. (arXiv:2412.09810) provide a spectral-entropy regularizer that "encourages networks toward low-complexity representations by penalizing their intrinsic dimension" and produces clean grokking phase transitions. Tikquuss/intrinsics_dimension provides a tensorized PyTorch TwoNN that is differentiable w.r.t. activations. Scarvelis (NeurIPS 2024, arXiv:2405.14544) provides a tractable denoising-style surrogate for Jacobian nuclear-norm penalty that "scales to high-dimensional deep learning problems." These are the four candidate "treatment" arms.

3. **Manifold capacity (Cohen, Chung, Lee, Sompolinsky, Nature Commun. 2020; Wakhloo & Chung, PRL 2023) factorizes into three orthogonal geometric quantities** — manifold dimension D_M, manifold radius R_M, and inter-manifold correlation ρ. This factorization is what makes the causal question tractable: an intervention that lowers D_M but holds R_M and ρ fixed is the surgical manipulation the task requires. MFTMA reference code (schung039/neural_manifolds_replicaMFT) computes all three from features via `manifold_analysis_corr(X, kappa, n_t)` at kappa=0, n_t=200.

4. **Confounders are well-characterized.** Stable rank (Sanyal et al., arXiv:1906.04659), spectral norm, weight-norm products (Bartlett-Foster-Telgarsky, Neyshabur et al.), neural collapse NC1 within-class variability (Tirer-Bruna VCI, arXiv:2306.03440), sharpness/Hessian trace, and Jacobian Frobenius/nuclear norm all correlate with generalization and partially correlate with ID. A sham control that matches each one at a time is mandatory; Saxe et al. (ICLR 2018) is the cautionary precedent — they showed for the information bottleneck that "there is no evident causal connection between compression and generalization: networks that do not compress are still capable of generalization, and vice versa."

5. **Compute is feasible.** Jordan (arXiv:2404.00498) reports "training methods for CIFAR-10 which reach 94% accuracy in 3.29 seconds, 95% in 10.4 seconds, and 96% in 46.3 seconds, when run on a single NVIDIA A100 GPU" using a custom (CIFAR-tuned, not stock) ResNet-style architecture; a 5-dose × 4-treatment × 10-seed sweep costs ~10 GPU-minutes per cell at this speed and ~3 GPU-hours per cell with a standard ResNet-18 (Lydia Benzemrane, 215 s/epoch × 100 epochs). On 4×A100 the full design completes in ~3 days; on M4 Max 32 GB it is restricted to CIFAR-10/100 with ResNet-18/20 (~1 week).

## Details

### 1. Intervention Methods — Four Treatment Arms Plus Three Shams

**Treatment A (primary): Differentiable TwoNN penalty on penultimate features.** Compute on each mini-batch the TwoNN estimator (Facco et al., Scientific Reports 2017) on penultimate-layer activations Z ∈ ℝ^{B×d}:
```
mu_i = r_2(z_i) / r_1(z_i)
d_hat = - mean_i  log(mu_i) / log(1 - F_emp(mu_i))      # TwoNN slope
```
implemented via `twonn_pytorch` from Tikquuss/intrinsics_dimension. Soft-rank tricks (Cuturi-style soft-sort for nearest-neighbor rankings) make the gradient stable. Add `L_total = L_CE + λ · d_hat` with λ ∈ {0, 0.01, 0.03, 0.1, 0.3, 1.0} (six doses for dose-response).

**Treatment B: Spectral entropy regularizer (DeMoss et al. 2024; Khanh et al. 2026).** Compute the normalized Shannon entropy of the eigenvalue distribution of the penultimate-feature covariance Σ̂ = (1/N) Σᵢ zᵢ zᵢᵀ. Per Khanh et al. (arXiv:2604.13123, Def. 2):
```
p_k = lambda_k / sum_j lambda_j
H_tilde = - (sum_k p_k log p_k) / log d       # in [0,1]
```
Penalize `L = L_CE + λ · H_tilde`. Differentiable via SVD; use truncated SVD (top-k) for speed. Khanh et al. report a task-specific threshold H̃\*=0.609 (95% CI [0.595, 0.624]) on ℤ/97ℤ and H̃\*=0.655 on S₅ that is crossed before generalization "in 100% of runs (mean lead: 1,020 steps)."

**Treatment C: Jacobian nuclear-norm penalty (Scarvelis, NeurIPS 2024, arXiv:2405.14544).** A denoising-style efficient surrogate for ‖J_f[x]‖_\* makes the encoder locally low-rank: "encourages the encoder to locally behave like a low-rank linear map whose image is low-dimensional." Avoids per-batch SVD; scales to deep nets.

**Treatment D: Persistent-homology dim (Birdal et al., NeurIPS 2021, arXiv:2111.13171).** L_total = L_CE + λ · dim_PH({w_k}) over a rolling window of recent SGD weight iterates, using `calculate_ph_dim_gpu` from the PHDimGeneralization repo with α=1 (the paper's stated default: "we will assume that the intrinsic dimension is strictly larger than 1, hence we will set α=1").

**Sham 1 (norm-matched): Weight decay schedule.** Tune weight decay so that ‖θ‖₂ matches the trajectory of Treatment A at each step, but apply no ID penalty. This is the critical control from Khanh et al. (one-sided Mann–Whitney U, n=30, p=5×10⁻⁵, Cohen's d=1.55 in their grokking setup): if Sham 1 produces equal generalization gains, ID is not the causal channel.

**Sham 2 (rank-matched, ID-free): Stable rank normalization (Sanyal et al., arXiv:1906.04659).** Their SRN scheme is "a novel, provably optimal, and computationally efficient weight-normalization scheme which minimizes the stable rank of a linear operator." Forces srank(W_i) to match Treatment A's trajectory without imposing a manifold-dimension penalty. Distinguishes "low parameter rank" from "low representation manifold dimension."

**Sham 3 (alignment-matched): Class-mean alignment penalty (NC2/NC3-style ETF target).** Forces neural-collapse geometry (Papyan-Han-Donoho, PNAS 2020) without forcing low D_M of the within-class manifold. Tests whether ID effects are downstream of neural collapse rather than independent.

**Anti-treatment (ID-increasing): Inverse spectral-entropy penalty** (push H̃ up) and **Gaussian noise on penultimate features** at σ ∈ {0.05, 0.1, 0.2, 0.4} (replicating Yadav 2026's r=−0.94 Gaussian-noise result while controlling). Required for the dose-response curve to extend in both directions and for the bidirectional-causality test of Yadav 2026.

### 2. Causal Identification Strategy

The causal DAG: **Treatment → ID(L) → Generalization(L)**, with confounders **C = {‖θ‖, srank, sharpness Tr(H), NC1 (VCI), Jacobian norm}** affecting both ID and generalization.

**Step A — Front-door + matching design.** For each pair (Treatment k, dose d), run 10 seeds and compute (i) ΔID, (ii) Δ in each confounder C_j, (iii) Δtest accuracy. Run Sham_j at a dose calibrated so it produces ΔC_j matching the Treatment's ΔC_j but ΔID ≈ 0. The contrast (Treatment vs. Sham_j) at matched ΔC_j is the **direct effect of ID on accuracy with C_j blocked**.

**Step B — Mediation analysis (Imai-Keele-Tingley natural direct/indirect effects).** Using DeepMed-style debiased ML (Xu et al., NeurIPS 2022, arXiv:2210.04389) — which "can achieve semiparametric efficiency bound without imposing sparsity constraints on the DNN architecture" — treat ID as mediator M, accuracy as outcome Y, intervention strength λ as treatment T, confounders C as covariates. Estimate the Natural Indirect Effect (NIE) = E[Y(T=1, M(T=1))] − E[Y(T=1, M(T=0))]. If NIE through ID is statistically significant (bootstrap CI excludes 0) and dominates the Natural Direct Effect (NDE), the causal chain is supported.

**Step C — Instrumental variable.** Use random initialization seed as an instrument: it perturbs ID independently of test accuracy through training dynamics (Wald-style two-stage least squares). Required only if confounder set is suspected to be incomplete.

**Step D — Do-operator via projection.** Following Yadav 2026: at inference, project penultimate features onto top-k PCs (do(ID ← k)) while leaving weights untouched. Plot accuracy vs. k. Yadav reports the headline result "improving geometry via PCA projection maintains accuracy across multiple architectures (mean −0.03pp loss at 95% variance across ResNet18, ResNet34, DenseNet121)" — replicate this baseline and extend with more variance thresholds (50%, 75%, 90%, 95%, 99%).

### 3. Models and Datasets

**Vision (primary).** ResNet-18 (~11 M params) and ResNet-20 on CIFAR-10 / CIFAR-100; ViT-Small (~22 M) on Tiny-ImageNet (200 classes, 64×64) and STL-10. Reasoning: ResNet-18 on CIFAR is the standard benchmark for ID studies (Ansuini et al. NeurIPS 2019; Brown et al. 2022; DeMoss et al. 2024); ViT-S adds architecture diversity to test whether causal effects generalize across inductive biases.

**Language (secondary).** A 1-layer Transformer (d_model=128, 4 heads, ff=512, AdamW η=10⁻³, weight decay λ=1.0, batch size 512, up to 50,000 steps) on modular arithmetic (a+b mod 97, train fraction 0.20, n_train=1,881) and S₅ permutation composition (120 classes) — the exact Khanh et al. (arXiv:2604.13123) configuration — for replicating their grokking-regime causal result and extending it with a richer sham battery.

**Compute budget.**
- 4×A100 (40 GB): full design ≈ 1,600 runs (4 datasets × 4 treatments × 3 shams × 6 doses × 10 seeds, with airbench-style optimizations and heavy checkpoint reuse). With CIFAR ResNet at ~5 s/run for fast configurations and CIFAR-100/Tiny-ImageNet runs averaging 30 min, ~3 days wall-clock.
- M4 Max 32 GB: restricted to CIFAR-10/100 + ResNet-18/20, ~1 week wall-clock.

### 4. Metrics

**Representation ID (mediator).**
- **Online (every epoch):** TwoNN on a fixed 4,096-sample probe set, penultimate features. Cost ~0.1 s on A100 per evaluation.
- **Online (every 200 steps for grokking runs):** Normalized spectral entropy H̃, following Khanh et al. (probe every 200 steps on 512 fixed examples; float64 SVD; train- vs test-probe agree at Pearson r=0.998 in their setup).
- **Post-hoc:** MLE (Levina-Bickel 2005), PHD (Bac-Zinovyev via scikit-dimension), and DADApy's Gride (Denti et al. arXiv:2104.13832, "more robust to noisy measurements than the TwoNN") for noise-robustness comparison. Report all four estimators; if conclusions differ across estimators, the causal claim weakens.

**Generalization (outcome).** Test accuracy; OOD generalization on CIFAR-10-C / CIFAR-100-C (Hendrycks corruption benchmark); ECE (calibration); generalization gap (train–test); ImageNet-R, ImageNet-Sketch for transfer (where applicable).

**Confounder control metrics.** ‖W_i‖_2 spectral norm products, stable rank Σsrank(W_i)⁻¹, Jacobian Frobenius norm, NC1 within-class variability via VCI (Tirer-Bruna, arXiv:2306.03440), Hessian trace via Hutchinson estimator, manifold capacity α_M via MFTMA at kappa=0, n_t=200 returning (α_M, R_M, D_M, ρ).

**ID-vs-capacity decomposition.** MFTMA gives D_M independently of TwoNN-based ID — running both lets us check whether classification-relevant ID (D_M) and representational ID (TwoNN) are the same thing. They are not always equal (Mamou et al., ICML 2020 "Emergence of Separable Manifolds in Deep Language Representations" found dissociations), so reporting both is essential.

### 5. Success and Failure Criteria

**Pre-registered hypothesis H1 (causal):** For Treatment A (TwoNN penalty), the natural indirect effect through ID exceeds the natural direct effect at all three sham-matched conditions, and the dose-response curve is monotone with a slope significantly different from zero (p<0.01 after Bonferroni correction across 4 treatments).

**Falsification criteria.**
- F1: A sham control achieves ≥80% of the Treatment's accuracy gain at matched ID change ≤20% of Treatment's. (ID is not the channel.)
- F2: Dose-response curve is non-monotone or has slope CI crossing zero.
- F3: Direction reversal across estimators (e.g., TwoNN says ID dropped, but D_M from MFTMA rose).
- F4: Anti-treatment (ID increase) does NOT produce a symmetric accuracy decrease (would break Yadav's "bidirectional causality" claim).

**Statistics.**
- 10 seeds per cell minimum; 30 for the headline norm-matched contrast (Khanh et al. used n=30; Jordan 2023 reports σ≈0.14% per-run on CIFAR-10 with airbench).
- One-sided Mann–Whitney U for primary effect (matching Khanh et al.'s protocol); bootstrap 95% CIs from 10,000 resamples; Cohen's d ≥ 0.5 required for "practically significant."
- α = 0.01 after Bonferroni across 4×3 treatment-sham contrasts.
- Power analysis: detecting a 1% accuracy difference at σ=0.14% (airbench) requires n≥10 per arm at α=0.01, β=0.8; for CIFAR-100 σ≈0.5%, n≥30.

**Dose-response & scaling-law check.** Five doses spanning ~10× range of λ. The Sharma-Kaplan connection (arXiv:2004.10802, JMLR 2022): "scaling exponents α ≈ 4/d for cross-entropy and mean-squared error losses." The test-loss scaling exponent should change predictably as we move along the ID dose axis. Fitting α(ID) ∝ 1/ID across doses provides a quantitative consistency check with theory.

### Code-level scaffold

```python
# Pseudocode (PyTorch)
import torch, torch.nn.functional as F
from intrinsics_dimension import twonn_pytorch  # Tikquuss

def train_step(model, x, y, lam=0.1, treatment="twonn"):
    z = model.features(x)                    # penultimate activations B×d
    logits = model.classifier(z)
    L = F.cross_entropy(logits, y)
    if treatment == "twonn":
        L = L + lam * twonn_pytorch(z)       # differentiable ID penalty
    elif treatment == "spec_entropy":
        cov = (z.T @ z) / z.size(0)
        eig = torch.linalg.eigvalsh(cov).clamp(min=1e-9)
        p = eig / eig.sum()
        H = -(p * p.log()).sum() / torch.log(torch.tensor(eig.numel(), dtype=z.dtype))
        L = L + lam * H                      # Khanh et al. H̃
    elif treatment == "sham_norm":
        # match ||theta||2 trajectory of treatment via adaptive WD
        L = L + lam_wd(step) * sum(p.pow(2).sum() for p in model.parameters())
    L.backward(); optimizer.step()
    # Probe metrics every K steps:
    if step % K == 0:
        log({
            "twonn_id":     twonn_pytorch(probe_features).item(),
            "spec_entropy": spectral_entropy(probe_features).item(),
            "srank":        stable_rank(model),
            "weight_norm":  total_param_norm(model),
            "test_acc":     evaluate(model, test_loader),
        })
    return L.item()
```

## Recommendations (staged)

**Stage 1 (Week 1, M4 Max feasible).** Replicate Brown et al. 2022 (arXiv:2211.13239) LLID-vs-regularization correlation on CIFAR-10 + ResNet-18 using the act_ids repo (https://github.com/BradleyBrown19/act_ids). This establishes the correlational baseline in your own pipeline. Threshold to proceed: Pearson r between LLID and test accuracy ≤ −0.7 across ≥5 regularizer strengths.

**Stage 2 (Week 2-3, 4×A100).** Run Treatment A (TwoNN penalty) at 6 doses × 10 seeds on CIFAR-10 ResNet-18. Plot dose-response. Threshold to proceed: monotone dose-response with slope significantly ≠ 0 at p<0.01.

**Stage 3 (Week 3-4).** Add Sham 1 (norm-matched weight-decay schedule) at the dose producing maximal ID change. This is the decisive contrast. Threshold to publish positive: NIE through ID > NDE at p<0.01, Cohen's d ≥ 0.5. Threshold to publish negative: F1–F4 met.

**Stage 4 (Week 4-6).** Extend to CIFAR-100, Tiny-ImageNet, ViT-S; add Treatments B–D and Shams 2–3 to test robustness. Run formal DeepMed mediation analysis.

**Stage 5 (Week 6-8).** Modular arithmetic + 1-layer Transformer replication of Khanh et al. with our richer sham battery, to bridge the vision and grokking regimes and test whether the H̃\*≈0.61 threshold result depends only on entropy or whether sham controls equalize the effect.

**Stop-condition for the program.** If after Stage 3 the sham controls match treatment gains, write up as a negative result connecting to Saxe et al.'s 2018 ICLR finding that "there is no evident causal connection between compression and generalization" — this would be high-impact in itself and would complete the conceptual parallel with the IB literature.

## Caveats

1. **Both 2026 "causal" preprints are from non-traditional venues.** arXiv:2602.00130 is a single-author preprint by Sumit Yadav (Institute of Engineering, Pulchowk Campus, Tribhuvan University, Nepal); arXiv:2604.13123 is by four authors at Clevix LLC (Hanoi) and Banking Academy of Vietnam. Their headline numbers (r=−0.99 for salt-and-pepper noise; ΔT=+5,020 steps at p=0.044) should be replicated, not assumed. Both explicitly disclaim Pearl/Rubin causality; Khanh et al. write: "we do not claim a fully causal relationship in the Pearl/Rubin sense."

2. **ID estimators disagree.** TwoNN, MLE, PHD, and DANCo can produce ID estimates differing by factors of 2 on the same activations. scikit-dimension's 19-estimator consensus approach (Bac & Zinovyev 2021, MDPI Entropy) is recommended for robustness; if a finding holds only under one estimator, it should not be reported as causal.

3. **Pope et al.'s ImageNet ID is for input pixels, not internal activations.** Per arXiv:2104.08894: "ImageNet, despite containing 224×224×3 = 150528 pixels per image, only has intrinsic dimension between 26 and 43." Ansuini's "hunchback" curve and Brown et al.'s LLID are for *representations*. The causal chain we test is **representation ID → generalization**, NOT input-data ID → generalization (the latter being effectively unmanipulable without changing the dataset itself, hence non-causal-experimental).

4. **Manifold dimension D_M (MFTMA) and TwoNN ID are not the same quantity.** D_M is a classification-relevant geometric width derived from replica mean-field theory; TwoNN is a topological dimension estimated from nearest-neighbor ratios. Treatments may affect one but not the other. The protocol explicitly tracks both.

5. **Differentiable PH-dim is numerically fragile** at small batch sizes and depends on TorchPH being installed against a specific PyTorch version. Spectral entropy and TwoNN are more robust starting points.

6. **Sharma & Kaplan's α ≈ 4/d scaling law** is asymptotic (large-N regime). Small CIFAR runs may not access this regime; treat the scaling-law check as a sanity test, not a falsification axis.

7. **Saxe et al. (ICLR 2018) provide the cautionary precedent.** They showed that an apparently causal compression-generalization story (Tishby's information bottleneck) failed under proper controls: "there is no evident causal connection between compression and generalization: networks that do not compress are still capable of generalization, and vice versa." The current protocol's 7-arm design (4 treatments + 3 shams + anti-treatment) is explicitly built to avoid the same failure mode.

## Completion Table

| Spec item | Covered |
|---|---|
| Intervention methods (regularizers, projections, bottlenecks) | ✓ Treatments A–D + anti-treatment |
| Surgical "ID without other pathways" requirement | ✓ Three sham arms (norm / rank / alignment) |
| Causal identification strategy (do-calculus, mediation, IV) | ✓ DeepMed mediation, IV-via-seed, do-projection à la Yadav |
| Controls for norm/alignment/rank | ✓ Sham 1/2/3 |
| Model & dataset selection (4×A100 / M4 Max) | ✓ ResNet-18, ViT-S; CIFAR-10/100, STL-10, Tiny-ImageNet; modular arithmetic |
| Online vs post-hoc ID estimators + cost | ✓ TwoNN online (~0.1 s), MLE/PHD/Gride post-hoc |
| Generalization metrics + OOD + calibration | ✓ acc, ECE, gen-gap, CIFAR-10-C/CIFAR-100-C |
| Control metrics (rank, NC, spectral) | ✓ srank, VCI/NC1, MFTMA(D_M,R_M,ρ), Hessian trace |
| Success/failure criteria + statistics + dose-response | ✓ Pre-registered H1, F1–F4, n=10–30, dose curves, Mann–Whitney U |
| Recent 2022–2026 differentiable ID estimators | ✓ Birdal 2021 (PH), DeMoss 2024 (spec entropy), Khanh 2026, Scarvelis 2024 (nuclear), TwoNN Tikquuss |
| Grokking / double descent / phase transitions | ✓ Khanh 2604.13123, DeMoss 2412.09810, Wang 2604.04655, Tan 2024 |
| Existing causal attempts | ✓ Yadav 2602.00130, Khanh 2604.13123, Saxe 2018 negative precedent |
| PyTorch libraries (scikit-dimension, DADApy, intrinsics_dimension) | ✓ All cited with usage; MFTMA replicaMFT for capacity |
| Code-level suggestions | ✓ Pseudocode block with treatment switches |
| Staged actionable recommendations | ✓ Stages 1–5 with thresholds and stop-condition |

