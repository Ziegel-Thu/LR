### Model representation geometry and drift analysis

用户:


Claude:
his is a well-structured and detailed research request — the scope, questions, and constraints are all clearly specified. I'll dive into comprehensive research right away, covering the model landscape, representation geometry metrics, and the relationship between drift and capabilities/safety.
our experimental plan for quantifying RLHF/DPO representation drift in LLMs is ready. It specifies three fully-open model triples (Tulu 3, OLMo-2, Zephyr) as primary subjects, a layered metric stack (CKA, Procrustes, Shesha, mutual k-NN, PWCCA) designed to compensate for CKA's known blind spots, and a low-rank ΔW characterization that tests the LoRA and "intruder dimensions" hypotheses. The plan includes regression models linking per-layer drift to capability benchmarks (MMLU, GSM8K, HumanEval, IFEval, TruthfulQA) and safety outcomes (HarmBench, AdvBench, XSTest), a 12-14 week staged execution schedule on 4×A100/8×A40 compute, explicit success criteria with falsifiable predictions, and a caveats section covering CKA pitfalls, DPO-vs-PPO generalization limits, and citation accuracy notes.
# Quantifying RLHF/DPO Representation Drift in LLMs: An Experimental Plan

## TL;DR
- **Anchor the study on three fully-open triples — Tulu 3 (Llama-3.1-8B base → allenai/Llama-3.1-Tulu-3-8B-SFT → -DPO), OLMo-2-7B (base → SFT → DPO), and Zephyr (Mistral-7B-v0.1 → HuggingFaceH4/mistral-7b-sft-beta → HuggingFaceH4/zephyr-7b-beta) — because they are the only public pipelines that release *all three* stages with documented data and training code, fitting comfortably in inference on 4×A100 (80 GB) or 8×A40 (48 GB).**
- **Use a *layered metric stack* that explicitly compensates for CKA's known blind spot to RLHF-induced spectral-tail reorganization: pair linear/RBF CKA with Procrustes, mutual k-NN/Shesha-style RDM correlation, PWCCA, and per-layer SVD of activation-difference and ΔW matrices; verify the low-rank hypothesis by computing the effective rank (stable rank, 90 %-energy rank) of ΔW = W_DPO − W_SFT layer-by-layer and comparing it to canonical LoRA ranks (r = 1–16 as in Hu et al. 2022 for GPT-3 175B/GPT-2 Medium).**
- **Predict capability loss and safety degradation by regressing per-layer drift descriptors (effective rank of ΔW, principal angle between the drift subspace and the Arditi refusal direction / RepE safety reading vectors) on benchmark deltas across MMLU, GSM8K, HumanEval, IFEval, TruthfulQA and HarmBench/AdvBench/XSTest — with success defined as out-of-sample R² > 0.5 for at least one capability axis and Spearman ρ > 0.6 between drift-safety subspace alignment and HarmBench attack-success rate.**

## Key Findings (literature anchors that shape the plan)

1. **Three-stage public pipelines are now mature.** Tulu 3 (Lambert et al., arXiv:2411.15124) releases base, SFT, DPO, and RLVR variants at 8B/70B; OLMo-2 (Walsh et al., arXiv:2501.00656) releases base, SFT, DPO at 7B/13B/32B with intermediate checkpoints every 1 B tokens; Zephyr-7B-β (Tunstall et al., arXiv:2310.16944) provides Mistral-7B-v0.1 → mistral-7b-sft-beta → zephyr-7b-beta on the alignment-handbook recipe.
2. **CKA systematically *underestimates* RLHF drift.** Raju (arXiv:2604.17698, "The Geometric Canary," 2026) shows CKA "is dominated by the top principal components and invariant to the spectral-tail reorganization that fine-tuning induces," reporting a 5.23× gap between Shesha and CKA on the Llama family of post-training pairs. Procrustes overshoots, "triggering false alarms in 44 % of cases where no functional degradation has occurred." This argues for a *metric ensemble*, not a single number.
3. **The "low-rank ΔW" picture has been complicated.** Aghajanyan, Zettlemoyer & Gupta (ACL 2021, arXiv:2012.13255) measured an intrinsic dimension of ~200 for RoBERTa on MRPC ("by optimizing only 200 trainable parameters randomly projected back into the full space, we can tune a RoBERTa model to achieve 90 % of the full parameter performance levels on MRPC"). The LoRA paper (Hu et al., ICLR 2022, arXiv:2106.09685) found "a very low rank (i.e., r in Figure 1 can be one or two) suffices even" for GPT-3 175B, with "the optimal rank for GPT-2 Medium … between 4 and 16." But Shuttleworth et al. (arXiv:2410.21228, "LoRA vs Full Fine-tuning: An Illusion of Equivalence") show that full fine-tuning's top singular vectors are "nearly orthogonal" to those of the pre-trained weights — i.e. ΔW is dominated by a small number of *new* "intruder" directions, not a rotation of existing ones. We must measure this carefully.
4. **Safety lives in a small, low-rank subspace.** Arditi et al. (arXiv:2406.11717) demonstrate refusal is mediated by a *one-dimensional* residual-stream direction across 13 chat models up to 72 B; Wollschläger et al. (arXiv:2502.17420, "The Geometry of Refusal") generalize this to a "concept cone" of independent refusal directions; Jain et al. (NeurIPS 2024, arXiv:2407.10264) provide "significant evidence demonstrating that these methods minimally transform MLP weights to specifically align unsafe inputs into [the] weights' null space." Together these motivate measuring the *principal angle* between the ΔW row-space and these safety subspaces.
5. **Alignment tax is real but non-uniform.** Lin et al. (arXiv:2309.06256, "Mitigating the Alignment Tax of RLHF") measure drops on ARC Easy/Challenge, RACE, PIQA, SQuAD, DROP, and WMT-2014 Fr→En translation using OpenLLaMA-3B with rejection-sampling fine-tuning. Lu et al. (arXiv:2405.17931, "Online Merging Optimizers") evaluate alignment tax "across 14 benchmarks" for Qwen and LLaMA 1.8B–8B models under DPO and KTO. Lee et al. (arXiv:2401.01967) and the NeurIPS 2024 follow-up "Beyond Toxic Neurons" (arXiv:2411.06424) show DPO suppresses toxicity through *distributed* activation shifts rather than a few neurons — implying drift *geometry*, not weight-norm magnitude, should be the predictor.

## Details

### 1. Model Selection

| Family | Base | SFT | RLHF/DPO | Why chosen |
|---|---|---|---|---|
| **Tulu 3** | meta-llama/Llama-3.1-8B | allenai/Llama-3.1-Tulu-3-8B-SFT | allenai/Llama-3.1-Tulu-3-8B-DPO (and -RLVR ablation) | Same Llama-3.1 backbone across all stages; data + recipe public; 8 B fits on one A100-80 GB at bf16. |
| **OLMo-2** | allenai/OLMo-2-1124-7B | allenai/OLMo-2-1124-7B-SFT | allenai/OLMo-2-1124-7B-DPO | *Fully* open (weights, Dolma-2 data, code, intermediate checkpoints every 1 B tokens — the only family that lets us trace drift *during* training). 13B and 32B available if compute permits. |
| **Zephyr** | mistralai/Mistral-7B-v0.1 | HuggingFaceH4/mistral-7b-sft-beta | HuggingFaceH4/zephyr-7b-beta | Canonical *distilled* DPO pipeline (UltraChat → UltraFeedback); widely replicated; serves as a "different-recipe" cross-check on Tulu/OLMo. |
| **Optional cross-check** | Qwen2.5-Math-7B | RLHFlow/Qwen2.5-7B-SFT | RLHFlow/Qwen2.5-7B-DPO and -PPO-Zero | Lets us compare DPO vs. PPO from a common SFT under a different base family. |

**Why not Llama-2-Chat, Llama-3-Instruct, Mistral-Instruct, OpenChat, or vanilla Qwen2.5-Instruct?** Meta and Alibaba never released their SFT-only intermediate checkpoints; only the final RLHF/Instruct model is public, so ΔW cannot be cleanly decomposed into SFT and RLHF components. OpenChat releases only SFT-style models. These can serve as out-of-distribution generalization tests for the prediction model but not as primary triples.

**Compute fit.** On 4×A100-80 GB the three 7–8 B triples (≈ 14 GB bf16 each) all fit; we can hold base + SFT + DPO simultaneously in VRAM for paired-activation extraction. 70 B Tulu-3 and 32 B OLMo-2 are reachable with FSDP-offload at moderate throughput; we treat them as scaling-sanity-check runs (one full pass each), not primary subjects.

### 2. Representation Drift Metrics (layer-by-layer)

We collect, for each of 5,000 calibration prompts drawn from C4 (factual), Dolly-15K-style instructions, MT-Bench prompts (conversational), AlpacaEval prompts, and a 1,000-sample harmful/harmless paired set from Arditi's release, the residual-stream activations at the *last-token position* of every transformer block ℓ for each checkpoint.

**Metric stack (per layer ℓ):**
- **Linear CKA and RBF-CKA** (Kornblith et al., 2019) — baseline; recognise it will *under-call* drift in late layers (Raju, 2026).
- **Orthogonal Procrustes distance** on mean-centered activations — sensitive to spectral tail; flag values but interpret cautiously given the 44 % false-alarm rate documented by Raju.
- **Mutual k-NN / Aristotelian RDM agreement** (Wolfram & Schein, arXiv:2504.08775; Brbić group, arXiv:2602.14486) — robust local-structure metric that survives global rescaling; k ∈ {10, 50}.
- **Sample-wise cosine similarity** (Sun et al., arXiv:2406.14479) — cheap residual-stream sanity check; aligns with CKA on residual-rich layers.
- **PWCCA / SVCCA** — for spectrum-sensitive comparison.
- **Shesha-style supervised RDM split** (Raju, 2026, arXiv:2604.17698) — the only metric in our stack *not* invariant to orthogonal transforms; gives "early-warning" drift signal (detected drift first in 73 % of 26 models in the canary paper, vs CKA 0 %) with a 6× lower false-alarm rate than Procrustes (Shesha 6.5 % FAR vs. Procrustes 38.7 %).

**Directional drift analysis.** For each layer ℓ, form A_SFT, A_DPO ∈ ℝ^(N×d) (N=5000 prompts). Compute ΔA = A_DPO − P·A_SFT where P is the optimal Procrustes rotation. Run SVD on ΔA; report (i) the singular-value spectrum, (ii) the effective rank r_eff = (Σσ_i)² / Σσ_i², (iii) the rank required to capture 90 % / 95 % / 99 % of ‖ΔA‖_F. Repeat with a row-shuffle null distribution to set significance.

### 3. Low-rank Structure of ΔW

For each weight tensor W ∈ {q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj} at every layer:
- Compute ΔW = W_DPO − W_SFT and W_SFT − W_base; report ‖ΔW‖_F and the relative norm ‖ΔW‖_F / ‖W_SFT‖_F.
- SVD(ΔW): plot singular-value spectrum on log scale; compute stable rank σ_max(ΔW)² / ‖ΔW‖_F² and 90 %-energy rank.
- **Test the LoRA hypothesis.** The LoRA paper found r ∈ {1, 2} sufficed for GPT-3 175B and r ∈ {4–16} for GPT-2 Medium; intrinsic-dimension work (Aghajanyan et al. 2021) showed RoBERTa needed ~200 dimensions to recover 90 % MRPC accuracy. Shuttleworth et al. (2024) show full-FT introduces "intruder dimensions" — singular vectors of ΔW orthogonal to W_base. We will quantify intruder structure by computing principal angles between top-k singular vectors of ΔW and W_SFT. *Falsifiable prediction:* RLHF/DPO ΔW will have *lower* effective rank than SFT ΔW (DPO is a smaller perturbation), but its top singular directions will be *more orthogonal* to W_SFT than SFT's are to W_base — because DPO carves out behavior-specific directions (refusal, format, length) that the base model had no use for.
- **Mapping to LoRA-r.** We will report the smallest r such that a rank-r truncation of ΔW reproduces ≥ 95 % of the DPO–SFT behavioral delta on MT-Bench/IFEval (an inference-only ablation: substitute W_SFT + truncate_r(ΔW) at chosen layers).

### 4. Drift vs Capability Correlation

**Capability benchmarks** (all inference-only via `lm-evaluation-harness` and `evalplus`):
- **General:** MMLU (5-shot), MMLU-Pro, ARC-Challenge, HellaSwag, WinoGrande.
- **Reasoning/math:** GSM8K (CoT, 8-shot), MATH (4-shot).
- **Code:** HumanEval, MBPP, HumanEval+.
- **Instruction following:** IFEval, MT-Bench (GPT-4-judge), AlpacaEval-2 (length-controlled).
- **Calibration/honesty:** TruthfulQA (MC2).

**Statistical model.** For each (model triple) × (layer ℓ) × (benchmark b), fit a multivariate regression:

ΔScore_b = β₀ + β₁·CKA_ℓ + β₂·Procrustes_ℓ + β₃·Shesha_ℓ + β₄·r_eff(ΔW_ℓ) + β₅·sin(θ_safety,ℓ) + ε

with leave-one-checkpoint-out cross-validation. Use partial-correlation analysis to identify which *layers* most predict each capability axis. Liu et al. (Findings of ACL 2025, arXiv:2506.00382) provide a precedent: for LLaMA-2-7B-Chat they identify change-point layers in the range "from the 8th to the 14th"; we will report the analogous range for our triples without pre-committing to specific layer indices.

**Per-layer attribution.** Use *activation patching* from base → DPO at single layers and measure benchmark recovery — an inference-only causal test that scales linearly with #layers. 4×A100 supports ~32 patching runs per benchmark per day.

### 5. Safety Prediction from Drift Geometry

**Safety direction extraction (three complementary methods):**

1. **Arditi refusal direction** (arXiv:2406.11717): difference-in-means between residual-stream activations on 128 harmful (AdvBench) and 128 harmless (Alpaca) prompts at the post-instruction token position, every layer. Use the published `refusal_direction` pipeline; pick the layer with highest Δ-projection.
2. **RepE reading vectors** (Zou et al., arXiv:2310.01405): contrastive prompt pairs from the RepE honesty / harmlessness datasets, then PCA on the differences to obtain a *k*-dimensional safety subspace (k = 5).
3. **Concept-cone / multiple refusal directions** (Wollschläger et al., arXiv:2502.17420; the Hidden Dimensions of LLM Alignment paper, arXiv:2502.09674): gradient-based discovery returning a low-rank "harmfulness" subspace adapted for SFT→DPO comparison.

**The geometric prediction.** Let U_drift be the top-k left singular vectors of ΔA at layer ℓ (from §2) and V_safety the safety subspace from above. Compute principal angles {θ_i} via SVD of U_drift^T V_safety. The **drift-safety alignment score** is cos(θ_1) (largest) and the **subspace overlap** is ‖U_drift^T V_safety‖_F² / k.

**Safety benchmarks:**
- HarmBench (Mazeika et al., arXiv:2402.04249) — 510 standardized behaviors; attack success rate (ASR) is the primary outcome.
- AdvBench (Zou et al., 2023) — 520 harmful instructions; baseline refusal rate.
- ToxiGen (toxicity), BBQ (bias), XSTest (over-refusal).
- StrongREJECT-judge for grading completions.

**Predictive hypothesis.** Models with *higher* drift-safety alignment at the Arditi layer will have *lower* HarmBench ASR after a fixed-budget GCG attack — i.e. safety is "concentrated" along well-defined directions. Conversely, models where ΔW has high alignment with the safety subspace but *low* effective rank are predicted to be the *most* jailbreakable, because a single rank-1 abliteration (Arditi's weight-orthogonalization) suffices. We test this on the three primary triples plus the Qwen DPO/PPO cross-check.

### 6. Success Criteria

A meaningful finding requires at least two of the following:
1. **Reproducible drift profile.** CKA-vs-Shesha gap > 2× on ≥ 2 of 3 triples at late layers, replicating Raju (2026) on DPO-specific pairs.
2. **Low-rank ΔW verified, but with "intruder" structure.** ≥ 80 % of ΔW Frobenius energy in r ≤ 64 directions at > 70 % of layers, *and* mean principal angle between top-32 singular vectors of ΔW and W_SFT > 60° — confirming Shuttleworth et al.'s finding generalises to DPO.
3. **Predictive capability model.** Out-of-sample R² > 0.5 for at least one benchmark (likely IFEval or AlpacaEval, where DPO has clearest gains) and < 0.1 on a placebo benchmark (HellaSwag, expected near-invariant).
4. **Safety-geometry prediction works.** Spearman ρ > 0.6 between drift-safety subspace overlap and (a) HarmBench ASR under GCG; (b) over-refusal rate on XSTest. A *negative* result here is also informative: it would falsify the strong form of the Linear Representation Hypothesis applied to drift.
5. **A failed control.** Drift between two independently-DPO-trained checkpoints from the *same* SFT model (e.g. Tulu-3-DPO vs the WildChat-50M reproduction at nyu-dice-lab/tulu3-dpo-repro) should show much smaller, more isotropic ΔW — confirming the measured "drift directions" are pipeline-specific, not noise.

## Recommendations (staged plan, 12-14 weeks on 4×A100/8×A40)

**Stage 1 — Weeks 1-2 (Infrastructure).** Download all triples, verify identical tokenizers, build the activation-cache pipeline with `nnsight` or `transformer-lens` hooks. Establish reproducibility on a single model with the Arditi refusal-direction code. **Threshold to proceed:** end-to-end refusal-direction extraction reproduced within 2 % of paper-reported ASR on Llama-3-8B-Instruct.

**Stage 2 — Weeks 3-5 (Drift metric panel).** Compute all metrics in §2 layer-by-layer for the three triples. Publish the layer-wise CKA / Procrustes / Shesha / mutual-k-NN tables as the empirical core. **Threshold:** if the CKA–Shesha gap is < 1.2× across all layers and models, treat as a null replication of Raju — pivot to ΔW-only analysis.

**Stage 3 — Weeks 6-8 (Low-rank ΔW characterization).** Full SVD of ΔW for every weight matrix; intruder-direction quantification; rank-truncation behaviour ablations (replace ΔW with rank-r truncation, measure MT-Bench/HarmBench delta).

**Stage 4 — Weeks 9-11 (Capability/safety regression).** Run all benchmarks on all 9 checkpoints (≈ 400-500 GPU-hours total via vLLM batching), fit the regression model, run activation patching for per-layer attribution.

**Stage 5 — Weeks 12-14 (Safety-direction prediction).** Extract refusal/concept-cone subspaces; compute principal angles; test predictive ρ against HarmBench under both static and GCG-attacked conditions. Write up.

**Decision rule.** If Stage 4 R² is below 0.3 across *every* benchmark, the linear-drift-predicts-capability hypothesis is falsified — pivot the writeup toward characterising the *failure* and proposing nonlinear (kernel-regression or SAE-feature-level) extensions.

## Caveats

- **CKA pitfalls.** Beyond the spectral-tail blindness, CKA is dominated by high-norm features and can be biased by sequence-length differences; we mitigate by L2-normalising per-token activations and using fixed token positions.
- **Procrustes false alarms.** The 44 % FAR documented by Raju (2026) means a positive Procrustes signal alone is insufficient evidence of functional change — always corroborate with Shesha and a downstream benchmark delta.
- **DPO ≠ RLHF.** Strictly, DPO is reward-model-free preference optimization (Rafailov et al., arXiv:2305.18290); PPO-RLHF and RLVR may induce different drift geometries. The Qwen PPO-vs-DPO cross-check is the only direct comparison; generalising findings to "RLHF" broadly requires caution.
- **Causation vs correlation.** Subspace overlap predicting ASR is correlational. Activation patching gives layer-localized *causal* evidence, but layers are not independent (the residual stream is additive), so attribution is approximate.
- **Single-direction refusal is contested.** Recent work (Wollschläger et al., 2025; Pan et al., 2025) shows refusal is mediated by a *cone* of directions, not strictly one; our use of k-dimensional safety subspaces (k=5–10) accommodates this.
- **The "Geometric Canary" paper (arXiv:2604.17698) is a single-author 2026 preprint** without explicit SFT-vs-DPO disaggregation; we treat its Shesha metric as a useful tool, not an established standard, and confirm any Shesha-flagged drift with at least one orthogonal-invariant metric.
- **Compute realism.** Full SVD of a 4096×14336 MLP weight at every layer for three 7B models is ≈ 6 hours on one A100; the full plan stays well within a 4×A100 budget. The 70 B Tulu-3 and 32 B OLMo-2 SVDs are the binding constraint and should be scheduled as overnight jobs.
- **Citation accuracy.** Lin et al. (arXiv:2309.06256) measure alignment-tax on ARC, RACE, PIQA, SQuAD, DROP and WMT-translation rather than MMLU; Lu et al. (arXiv:2405.17931) report aggregate alignment tax across 14 benchmarks for Qwen/LLaMA 1.8B–8B under DPO/KTO — readers should consult the per-benchmark tables before quoting specific MMLU deltas.

