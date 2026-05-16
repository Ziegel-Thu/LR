### Cross-architecture representation convergence

用户:


Claude:
his is a beautifully scoped research design question — testing whether the Platonic Representation Hypothesis holds across fundamentally different architectural inductive biases (attention vs. SSM). The prompt is detailed enough that I can dive straight into research. Let me do a deep dive to find the latest open-source checkpoints, representation comparison tooling, and any recent empirical results on cross-architecture convergence.
our experimental protocol for testing the Platonic Representation Hypothesis across Transformer, Mamba, and RWKV architectures is ready. It includes a complete checkpoint roster (Pythia/Mamba-1/RWKV-4 Pile-trained triple, Falcon-Mamba, Bamba, RWKV-7 up to 13B), a comparison of 10+ similarity metrics with cost analyses, layer-alignment strategies for differing depths, scaling-law fitting procedures, mandatory null-calibration controls drawn from the Aristotelian critique, pre-registered success criteria with falsifiable thresholds, a synthesis of 12+ recent related papers (2024–2026), and a phased 6-week compute plan fitting within 2000–3000 A100-hours on 4×A100 or 8×A40 GPUs without any training.
# Testing the Platonic Representation Hypothesis Across Transformer, Mamba, and RWKV Architectures: A Comprehensive Experimental Protocol

## Executive Summary

The Platonic Representation Hypothesis (PRH) of Huh, Cheung, Wang & Isola (arXiv:2405.07987) argues that as neural networks scale in capacity, data, and task diversity, their internal representations converge toward a shared statistical model of reality. The hypothesis was largely formulated and tested using Transformer (vision and language) models. The recent maturation of two fundamentally different sequence-modeling families — **Mamba/Mamba-2 (selective structured state-space models)** and **RWKV (linear-attention RNNs)** — makes it possible, for the first time, to test PRH across **non-attention** language architectures at 7B+ parameter scale using only open weights and inference-only compute.

This protocol specifies (i) the full checkpoint roster currently available, (ii) the representation-comparison metrics with their pros, cons, and costs, and how to apply them across architectures with different depths, widths and tokenizers, (iii) a scaling-law analysis framework, (iv) confound controls drawn from the very recent critique literature (notably the "Aristotelian" null-calibration framework of Gröger et al., arXiv:2602.14486), (v) pre-specified quantitative success criteria, and (vi) a compute budget that fits on 4×A100 80GB or 8×A40 48GB GPUs without any training.

---

## 1. Model Selection & Open-Source Checkpoints

### 1.1 Mamba and Mamba-2 (pure SSM and hybrid)

**Pure Mamba-1** (state-spaces org; trained on The Pile, ~300B tokens, GPT-NeoX tokenizer):
- `state-spaces/mamba-130m-hf`, `mamba-370m-hf`, `mamba-790m-hf`, `mamba-1.4b-hf`, `mamba-2.8b-hf` (the canonical model directly matched to Pythia-2.8B at 300B Pile tokens)

**Pure Mamba-2** (state-space duality, May 2024, Pile-trained at small scales):
- `state-spaces/mamba2-130m`, `…-370m`, `…-780m`, `…-1.3b`, `…-2.7b`

**Pure Mamba at 7B scale:**
- `TRI-ML/mamba-7b-rw` (Toyota Research Institute; ~1.2T tokens of RefinedWeb-equivalent data)
- `tiiuae/falcon-mamba-7b` and `falcon-mamba-7b-instruct` (≈5.5T tokens; RefinedWeb-English + FineWeb-edu + technical/code/math; Falcon-7B/11B tokenizer; trained on 256 H100s)
- `tiiuae/Falcon3-Mamba-7B` (later release; improved benchmark scores)
- `mistralai/Mamba-Codestral-7B-v0.1` (code-specialized Mamba-2; useful as a domain-shifted control)

**Hybrid Mamba/Transformer** (these are *not* pure non-Transformer architectures and should form a separate experimental stratum):
- `Zyphra/Zamba-7B-v1` (Mamba-1 backbone + shared attention block, ~1T tokens; **open intermediate training checkpoints** — valuable for training-dynamics analysis)
- `Zyphra/Zamba2-1.2B`, `Zamba2-2.7B`, `Zamba2-7B` (Mamba-2 + shared attention; Mistral-7B tokenizer)
- `ai21labs/Jamba-v0.1` (12B active / 52B total MoE; Mamba+attention+MoE)
- `ibm-fms/Bamba-9B-v1` (IBM/Princeton/CMU/UIUC; 2.2T tokens; hybrid Mamba-2; **fully open data lineage**, explicitly compared in the Bamba blog against OLMo-7B and LLaMA-2-7B trained on similar token counts)
- `tiiuae/Falcon-H1` family (Mamba-2 based)
- NVIDIA hybrid Mamba-2 family (released with the NVIDIA 8B technical report)

### 1.2 RWKV checkpoints

**RWKV-4** (Pile-trained, directly parallel to Pythia/Mamba-1):
- `RWKV/rwkv-4-{169m,430m,1b5,3b,7b,14b}-pile` — the cleanest controlled-data series for cross-architecture comparison because Pythia, Mamba-1, and RWKV-4 share The Pile training corpus.

**RWKV-5 "Eagle"** (multilingual, ~1.1T tokens, World tokenizer):
- `BlinkDL/rwkv-5-world` (1.5B, 3B, 7B). Eagle 7B reaches performance similar to Pythia-6.9B near 300B tokens, then continues to improve.

**RWKV-6 "Finch":**
- `BlinkDL/rwkv-6-world` (1.5B, 3B, 7B)
- `RWKV/v6-Finch-1B6-HF`, `…-3B-HF`, `…-7B-HF`, `…-14B-HF`

**RWKV-7 "Goose"** (current generation; introduces vector-valued gating, in-context learning rates, and demonstrably exceeds TC⁰ expressivity per the 2025 arXiv companion paper):
- `BlinkDL/rwkv7-g1` / `rwkv7-g0` — sizes **0.1B, 0.4B, 1.5B, 2.9B, 7.2B, 13.3B** (vocab 65,536; head size 64; standard layer/width grid in BlinkDL/RWKV-LM repo)
- `fla-hub/rwkv7-7.2B-g0` (flash-linear-attention port)

**Distilled / converted variants** (ancillary, not primary, because representations are inherited):
- `RWKV-Red-Team/ARWKV-7B-Preview-0.1` and `ARWKV-R1-7B` (distilled from Qwen2.5 / DeepSeek-R1-Distill-Qwen-1.5B; based on RWKV-7 attention)
- RADLADS-converted 7B/32B/72B RWKV-variant models (350–700M conversion tokens from Qwen2.5)

### 1.3 Transformer comparators with comparable training data

| Comparator | Tokens | Data | Tokenizer | Pairs naturally with |
|---|---|---|---|---|
| `EleutherAI/pythia-{160m,410m,1b,1.4b,2.8b,6.9b,12b}` | 300B | The Pile | GPT-NeoX | Mamba-1, RWKV-4 — the **gold-standard controlled triple** |
| `allenai/OLMo-1B/-7B/-2-{7B,13B}` | 2–5T | Dolma / Dolma-2 | OLMo / cl100k-like | Bamba-9B, Falcon-Mamba |
| `meta-llama/Llama-2-7b`, `Llama-3-8B` | 2T / 15T | proprietary | SentencePiece / tiktoken | Zamba2, Falcon-Mamba (size match) |
| `mistralai/Mistral-7B-v0.1` | ~8T est. | proprietary | Mistral SP | Zamba2 (which reuses Mistral tokenizer), Codestral-Mamba |
| `tiiuae/falcon-7b`, `falcon-11b` | 1–5T | RefinedWeb | Falcon BPE | Falcon-Mamba-7B — **best 7B controlled pair** (matched tokenizer + heavily overlapping data) |
| `Qwen/Qwen2.5-{1.5B,3B,7B,32B,72B}` | 18T | proprietary | Qwen BPE | ARWKV / RADLADS-RWKV (distilled from these) |

**Recommended primary axis:** the Pile-trained triple {Pythia, Mamba-1, RWKV-4} at six matched parameter scales (160/410M ↔ 130/370M ↔ 169/430M; 1B/1.4B ↔ 1.4B ↔ 1.5B; 2.8B ↔ 2.8B ↔ 3B; 6.9B ↔ 7B-rw ↔ 7B). **Secondary axis:** Falcon-7B ↔ Falcon-Mamba-7B (same tokenizer, overlapping RefinedWeb-based mix). **Tertiary axis:** OLMo-7B ↔ Bamba-9B (open Dolma-like data, ~2T tokens; Bamba was explicitly designed for this comparison).

---

## 2. Representation Comparison Methods

The goal is to estimate, on identical input stimuli, the geometric/topological similarity of internal hidden states across architectures.

### 2.1 Method-by-method pros, cons, and computational cost

| Metric | Invariances | Pros | Cons | Cost (N stimuli, d-dim) | Suitable for cross-arch? |
|---|---|---|---|---|---|
| **Linear CKA** (Kornblith et al., 1905.00414) | Orthogonal rotations, isotropic scale | Architecture-agnostic; operates on N×N gram; widely used; minibatch-unbiased estimator exists (Nguyen et al.) | **Confounded by width/depth: scales as d/N when d is large** (Gröger et al. 2602.14486); inflated raw scores; orthogonality assumption strong | O(N²d) to form grams, O(N²) for HSIC; **~few minutes per layer-pair on one A100 at N=50k**; minibatch version O(B²d) with B=4096 reduces memory to <2 GB | **Yes**, but null-calibration mandatory |
| **RBF (kernel) CKA** | Non-linear monotone | Captures non-linear similarity structure | Bandwidth choice; same width inflation as linear CKA; harder to interpret | Same as linear CKA + kernel evaluation | Yes, secondary |
| **Mutual k-NN overlap** (Huh et al., 2405.07987; Tjandrasuwita et al.) | Any monotone distance transform | **Robust to width inflation** (Aristotelian-stable); topological; minimal assumptions; cheap | Coarse-grained; choice of k matters; less sensitive to global geometry | O(N²d) for distance matrix, O(Nk) for overlap; FAISS HNSW makes it O(N log N) | **Yes — PRIMARY metric** (recommended by both the original PRH paper and the Aristotelian critique as the most robust) |
| **Procrustes (orthogonal)** (Williams et al., 2110.14739) | Orthogonal rotations | Satisfies triangle inequality (true metric); stricter than CKA (small Procrustes ⇒ small CKA) | Requires equal d (or PCA projection); sensitive to outliers; orthogonality assumption | O(min(N,d)³) for SVD; O(d²N) for projections; **~seconds per layer-pair after PCA to d*=256** | Yes, after PCA or via Soft Matching |
| **Soft Matching Distance** (Khosla & Williams) | Permutation of neurons with D≠D' | Principled handling of unequal hidden dims via optimal transport | More expensive than vanilla Procrustes | O(d² log d) per OT solve | Yes, when Procrustes alignment desired |
| **SVCCA** (Raghu et al. 2017) | Linear invertible | Captures shared subspace structure | SVD-truncation sensitive; less stable at scale; can over-report similarity | O(min(N,d)³) for SVD then CCA | Legacy; secondary |
| **PWCCA** (Morcos et al. 2018) | Linear invertible (projection-weighted) | More robust than SVCCA | Same SVD bottleneck | Same as SVCCA | Legacy; secondary |
| **RSA / RDM correlation** (Kriegeskorte et al.) | Any monotone | Modality-agnostic; standard in neuroscience cross-comparisons | Spearman correlation conflates strong and weak structure | O(N²d) + O(N² log N²) | Yes, tertiary triangulation |
| **GULP / Bures** | Various | Information-theoretic flavors | Less standard | O(N²d) | Tertiary |
| **Linear probing alignment / stitching** (Cloos et al.; arXiv:2411.08197) | Decodability-grounded | Functionally meaningful; ties similarity to downstream-task signal | Requires training a probe; slow | Train probe per layer ~10⁵ steps | Tertiary, for grounding |

**Implementation recommendation: build on the ReSi codebase** (Klabunde et al., arXiv:2408.00531, github.com/mklabunde/resi), which ships validated reference implementations of 23–24 measures including all of the above, unified preprocessing, and grounded tests against ground-truth similarity. Adopt ReSi's preprocessing defaults (centering, optional unit-norm, no whitening) for cross-paper comparability.

### 2.2 Layer alignment across different depths

Mamba-2.8B has 64 layers; Pythia-2.8B has 32; RWKV-4-3B has 32; RWKV-7-7.2B has 32; RWKV-7-13.3B has 61. Use all three of the following and triangulate:

1. **Normalized-depth matching (i/L).** Compare layer ⌊α·L_A⌋ to ⌊α·L_B⌋ for α ∈ {0.05, 0.10, …, 0.95, 1.0}. Standard in Kornblith et al. and in the SSM-vs-TBM literature (Hoang et al., arXiv:2510.06640).
2. **Functional / Hungarian matching.** For each pair (i, j) compute similarity; pick j*(i) = argmax_j s(i,j) (or solve a Hungarian assignment). Produces a layer-correspondence map and is more sensitive to architecture-specific bottlenecks (e.g., Mamba's separation of channel-mixing from sequence-mixing per the MOHAWK decomposition, arXiv:2408.10189).
3. **Tuned-lens decoding alignment** (Differential Mamba, arXiv:2507.06204; Locating-and-Editing-Mamba, arXiv:2404.03646). Project each intermediate representation through the model's own unembedding (after final RMSNorm) and compare implied next-token distributions via KL / TVD. Bypasses raw-feature comparison and tests *functional* layer correspondence.
4. **Block-type matching for hybrids.** For Zamba/Jamba/Bamba, separately compare (a) attention layers vs. attention layers in the Transformer comparator and (b) SSM layers vs. mid-network Transformer layers, using the MOHAWK sequence-mixer/channel-mixer decomposition.

Visualization template: a (L_A × L_B) similarity matrix per (model-pair, metric, α-grid); plot both raw and null-calibrated scores side-by-side.

### 2.3 Handling different hidden dimensions

| Strategy | Pros | Cons | When |
|---|---|---|---|
| Kernel metrics (CKA, RSA, mutual k-NN) | No projection needed; operate on N×N | CKA inflates with d/N | Primary metric |
| PCA project both to d*=256 | Simple; preserves dominant variance | Loses fine structure; arbitrary d* | Sanity check / Procrustes prep |
| Soft Matching Distance | Principled D≠D' Procrustes via OT | More expensive | When geometric (not topological) match is critical |
| Stitching / linear probe | Functionally grounded | Requires probe training | Decodability checks (Cloos et al., 2411.08197) |

---

## 3. Convergence Rate & Scaling Laws

### 3.1 Functional forms

Define convergence score s(N) for a model pair {Transformer-N, SSM-N} at parameter count N as the maximum similarity across all (i,j) layer-pairs for a chosen metric (mutual k-NN at k=10 is recommended primary).

Candidate forms to fit (consistent with prior representation-convergence literature and standard LLM scaling-law practice):

- **Power-law approach to a ceiling**: s(N) = s_∞ − a·N^(−β). Fitting yields s_∞ (asymptotic platonic alignment) and β (convergence rate). The Huh et al. paper shows LLM↔vision alignment curves monotone in vision-model scale that visually fit this form.
- **Log-linear**: s(N) = a + b·log N. Fewer free parameters; appropriate over the limited 0.1B–13B span.
- **Sigmoid / logistic**: s(N) = s_min + (s_max − s_min) / (1 + exp(−k(log N − log N₀))). Captures emergent-style transitions.

Fit via non-linear least squares with leave-one-N-out cross-validation; report AIC/BIC across forms. **Pre-register**: under PRH, s(N) for {Pythia↔Mamba} and {Pythia↔RWKV-4} should both follow approximately the same functional form, with overlapping CIs on β, *and* match the Pythia↔OLMo same-architecture-different-data baseline by 7B scale.

### 3.2 Capability-conditioned scaling

The Tjandrasuwita et al. (arXiv:2502.16282) result is that alignment depends on modality heterogeneity and task-information balance; in our cross-architecture single-modality setting the controlled variables are (i) parameter count N, (ii) training tokens T, and (iii) downstream capability C (e.g., MMLU/HellaSwag — already published for all models above). Fit:

s = α₀ + α_N log N + α_T log T + α_C·C + ε

PRH-consistent prediction: α_C dominates α_N alone, mirroring the Huh et al. finding that *capability*, not raw parameter count, predicts cross-modal alignment.

### 3.3 Layer-wise convergence rate

Following Hoang et al. (arXiv:2510.06640), who report that **Transformers homogenize token representations rapidly in early/middle layers while SSMs preserve token uniqueness early and homogenize deep**, expect convergence rate to vary with depth fraction α. Fit separate s(N; α) curves per α-bin; report the depth at which cross-architecture convergence peaks.

---

## 4. Confound Control

### 4.1 Width/depth bias — the Aristotelian critique

Gröger et al. (arXiv:2602.14486) demonstrate that CKA and other global spectral measures are systematically inflated when d/N is large; their key result is that **most reported PRH convergence largely disappears under permutation-based null calibration, while local mutual-kNN agreement survives.** This protocol therefore mandates:

- **Permutation null calibration for every metric**: for each (A,B) pair, draw R ≥ 1000 random row-permutations of one representation, recompute the metric, and report the **calibrated similarity** = (raw − null mean) / null std (a z-score), plus a permutation p-value.
- Always report raw *and* calibrated scores side-by-side; flag any conclusion that depends on raw-only.
- **Use mutual k-NN as primary, CKA as secondary**, since k-NN is far more robust to width inflation.
- Match N (number of stimuli) ≥ 10·max(d_A, d_B): N ≈ 50,000 covers all models up to d=4096; N ≈ 100,000 is safer.

### 4.2 Training-data confounds

Use the **Pythia ↔ Mamba-1 ↔ RWKV-4 Pile-trained triple at 300B tokens** as the primary analysis cell; this isolates architecture as the independent variable. Use **Falcon-7B ↔ Falcon-Mamba-7B** as a second matched cell (same RefinedWeb-based mix, same tokenizer). Treat **OLMo-7B ↔ Bamba-9B** as a third (Dolma/Dolma-equivalent open data, ~2T tokens; Bamba was explicitly designed for this comparison and the blog reports it outperforms OLMo-7B at identical tokens).

Across cells, compare s(architecture-pair) under matched data vs. unmatched data to estimate the data-confound size.

### 4.3 Tokenizer confounds

Different tokenizers produce different token sequences for the same input string. Mitigations:

1. **Stimulus-level (text-aligned) analysis**: compare *sentence-pooled* representations (mean over non-padding tokens) for each input sentence. Tokenizer-invariant. Default pooling.
2. **First-byte-aligned pooling**: for sub-sentence resolution, align token boundaries to byte spans and pool within shared spans.
3. **Last-token (final-position) embedding**: standard for autoregressive comparison; the protocol Huh et al. used.
4. Report sensitivity of every result to pooling choice (mean, last-token, first-byte-aligned).

### 4.4 Training-duration confounds

Where intermediate checkpoints exist (Pythia: 154 checkpoints; Zamba/Bamba/OLMo also release intermediates), trace s(N, training-step). Aristotelian calibration permits checking whether convergence emerges over training or is present at init (it should not be — random networks give null-calibrated z ≈ 0).

### 4.5 Stimulus design

Stratified evaluation set of N ≈ 50,000 inputs, truncated to 512 tokens, cached once per (model, stimulus, layer):
- 10k sentences from **WikiText-103 validation** (long-form coherent prose)
- 10k from **C4 validation** (diverse web)
- 5k from **The Pile validation** (matches Pythia/Mamba-1/RWKV-4 distribution)
- 5k code from **The Stack v2 dev** (code-specialized probe)
- 5k from **MMLU / HellaSwag / ARC** test prompts (task-relevant)
- 5k **counterfactual-pair** stimuli (minimally edited pairs) for contrastive mutual-kNN that isolates semantic structure
- 5k **syntactic-template** stimuli from **BLiMP / SyntaxGym** (syntax-vs-semantics decomposition)
- 5k from a held-out distribution (multilingual mC4 subset) for OOD probes

---

## 5. Success Criteria

Pre-registered, falsifiable thresholds. All stated on null-calibrated z-scores unless noted.

### 5.1 Primary thresholds

| Outcome | Definition | Interpretation |
|---|---|---|
| **Strong convergence** | Calibrated mutual-kNN z ≥ 5 AND raw mutual-kNN@10 ≥ 0.25 at peak α, with ds/d log N > 0 (p < 0.01) across ≥ 4 scales | PRH supported for this architecture pair |
| **Partial convergence** | Calibrated z ≥ 3 but raw < 0.25, OR slope significant but z < 5 | Some shared structure; PRH partially supported |
| **No convergence** | Calibrated z < 3 across all layers/scales | PRH refuted for this architecture pair |
| **Anti-convergence** | ds/d log N significantly negative | Active divergence — strongly refutes PRH |

### 5.2 Task-specific convergence decomposition

Compute s separately for stimulus subsets probing (a) **syntactic** structure (BLiMP/SyntaxGym), (b) **lexical/semantic** structure (counterfactual pairs, WordNet hypernym hierarchies), (c) **world-knowledge** structure (factual recall stimuli), (d) **reasoning** structure (chain-of-thought prefixes), (e) **state-tracking** (FSA emulation, parity, modular arithmetic — per arXiv:2404.08819). Predictions:

- Syntactic convergence: strongest early-middle for Transformers, middle-late for SSMs (depth-shifted alignment, per Hoang et al. 2510.06640).
- Semantic/world-knowledge convergence: late-layer in both, per arXiv:2404.03646 showing factual recall localizes similarly across Mamba and Transformers.
- Retrieval / in-context structure: expected to **diverge** (Transformers superior), per arXiv:2409.05395 "Shaking Up VLMs" which shows the gap widens with scale.
- State-tracking: theoretically RWKV-7 exceeds TC⁰ while standard Transformers do not, predicting divergence at scale on this subset.

### 5.3 Architecture-specific predictions

- **PRH-supporting outcome**: mutual-kNN(Pythia_N, Mamba_N) and mutual-kNN(Pythia_N, RWKV-4_N) both rise monotonically with N, with overlapping exponents within CI, *and* mutual-kNN(Mamba_N, RWKV_N) rises comparably — i.e., all three converge on a common representation, not pairwise to one anchor.
- **PRH-refuting outcome**: mutual-kNN saturates below the Transformer-Transformer baseline (Pythia↔OLMo at matched N) by a gap that does not close with scale.
- **Aristotelian outcome**: raw CKA increases with scale but null-calibrated CKA stays near zero, while *local* mutual-kNN shows real but small calibrated increases — architectures share local-neighborhood structure but not global geometry.

### 5.4 Capability vs. architecture decomposition

If PRH holds, partial correlation s ~ C | log N should remain significantly positive after controlling for parameter count; β_C > β_N. If PRH fails for SSMs, the architecture-pair indicator stays significantly negative after capability matching.

---

## 6. Recent Related Work Synthesis (2024–2026)

**arXiv:2409.05395 — Shaking Up VLMs (Mamba vs Pythia-VL).** Trains Mamba-VL and Pythia-VL on *identical data in identical order with identical hyperparameters*, the cleanest published controlled comparison. Finds Mamba-VL outperforms Pythia-VL on captioning/QA/reading comprehension but underperforms on visual grounding, with the gap widening at scale. Attributes the gap to in-context multimodal retrieval — a representational weakness that should manifest as architecture-specific divergence in our retrieval-stimulus subset.

**arXiv:2408.10189 — MOHAWK (Transformers-to-SSMs distillation).** Decomposes both architectures into sequence-mixing + channel-mixing blocks and successfully distills attention into Mamba via matrix-mixing alignment. Useful for layer-correspondence and block-level similarity computation. The query mentions arXiv:2406.16722; that exact ID could not be confirmed in this search and likely refers to MOHAWK or a related 2024 distillation paper (Phi-Mamba, Mamba-in-Llama family). Use the MOHAWK formalism as the canonical reference for SSM↔Transformer "transitions"; confirm the exact citation before publication.

**arXiv:2502.16282 — Tjandrasuwita et al., Multimodal Representation Alignment.** Shows alignment is *not always* a reliable indicator of performance, and depends on data heterogeneity and shared-vs-unique modality information balance. Motivates the task-decomposition in Section 5.2.

**arXiv:2602.14486 — Gröger et al., "Revisiting the Platonic Representation Hypothesis: An Aristotelian View."** Demonstrates CKA and other global metrics are confounded by width/depth; proposes permutation null-calibration that **largely eliminates apparent PRH convergence**, leaving only local mutual-kNN agreement. Proposes the Aristotelian hypothesis: networks converge to *local neighborhood relationships*, not global geometry. **This paper defines the most important confound control in this protocol** and reframes the primary success criterion around null-calibrated mutual-kNN.

**arXiv:2502.18710 — Bridging Critical Gaps in Convergent Learning.** Examines alignment across layers, training time, and distribution shifts; provides protocols for layer-wise and training-trajectory analysis.

**arXiv:2510.06640 — Hoang et al., Contextual Representation Flow in SSMs vs Transformers.** Uses CKA, stability metrics, and probing across both architectures and finds that Transformers rapidly homogenize token representations early/middle with diversity re-emerging late, while SSMs preserve token uniqueness early and homogenize deep. Ties Transformer oversmoothing to architecture and SSM oversmoothing to training dynamics. **Directly predicts the depth-fraction profile** of cross-architecture convergence for our experiment.

**arXiv:2404.03646 — Locating and Editing Factual Associations in Mamba.** Factual recall localizes in Mamba at roughly the same relative depth as in Transformers; rank-one editing transfers. Predicts factual-stimulus convergence will be strong.

**arXiv:2403.01590 — The Hidden Attention of Mamba Models.** Reformulates Mamba as data-controlled implicit attention with comparable interpretability metrics to Transformers, supporting functional alignment of representations.

**arXiv:2402.03170 — Is Mamba Capable of In-Context Learning?** Probes both architectures performing ICL; finds both implement iterative-optimization-like solutions — functional evidence of representational convergence for ICL stimuli.

**arXiv:2411.08197 — What Representational Similarity Measures Imply about Decodable Information.** Shows CKA and Procrustes upper-bound and are bounded by linear-decoder alignment, formalizing the link between geometric similarity and functional convergence. Relevant for using linear probes as tertiary metric.

**RWKV-7 paper (Peng et al., 2025).** Proves RWKV-7 expresses regular languages and state-tracking beyond TC⁰, **exceeding standard-conjectured Transformer expressivity**. Predicts RWKV-7 representations may *diverge* from Transformer representations on state-tracking stimuli — a falsifying signature for PRH on this architecture pair. Include FSA-emulation, parity, and modular-arithmetic stimuli (per arXiv:2404.08819 "The Illusion of State in State-Space Models") to test.

**arXiv:2408.00531 — ReSi benchmark (Klabunde et al.).** Provides the comprehensive evaluation framework and codebase for our similarity-metric implementations.

---

## 7. Practical Compute Plan (4×A100 80GB or 8×A40 48GB, inference only)

### 7.1 Memory feasibility at 7B+ scale

A 7B-parameter model in bf16 needs ~14 GB; in 8-bit ~7 GB. On 4×A100 80GB, all 7B models fit on a single GPU with bf16 activations. The 13B RWKV-7 fits in bf16 on one A100 80GB or splits across two A40s. Install the optimized Mamba kernels (`causal-conv1d`, `mamba-ssm`) and Flash-Attention 2 for Transformers for ≥10× throughput.

### 7.2 Computational feasibility of similarity metrics at 7B+ scale

**Hidden-state extraction.** Per model M, per stimulus s ∈ S (|S| ≈ 50k), per layer ℓ: extract one pooled vector v ∈ ℝ^d. A 32-layer 4096-dim model: 50k × 32 × 4096 × 2 bytes ≈ 13 GB bf16 on disk. Full roster of ~30 models × ~32 layers × 50k × ~4096 × 2 bytes ≈ 400 GB — feasible on a single NVMe drive. Forward-pass throughput ~3000 tok/s for a 7B model on one A100; 50k × 512 tokens ≈ 25.6M tokens ≈ 2.5 hours per 7B model. Total extraction across the roster ≈ **3–5 GPU-days on 4×A100**.

**CKA at 7B scale.** Naive dense N×N gram for N=50k is 10 GB in float32. Use the **minibatch unbiased CKA estimator** (Nguyen et al.) with batches B=4096 and R=10 repeats: memory <2 GB, runtime ~few minutes per layer-pair. Linear CKA across all (L_A=32 × L_B=32 = 1024) layer-pairs of one model pair: ~1 GPU-hour.

**Mutual k-NN at 7B scale.** Build a FAISS HNSW index on 50k × 4096 vectors per (model, layer): ~30 seconds; query all 50k against the other model's index: ~30 seconds. Across 1024 layer-pairs per model pair: ~1 GPU-hour.

**Procrustes at 7B scale.** Project both representations to d* = 256 via incremental PCA (one-time cost ~10 minutes per model); then per layer-pair O(N × d*²) = trivial. Across all layer-pairs: ~10 minutes per model pair.

**Null calibration.** R=1000 permutations × 1024 layer-pairs × per-permutation cost (mutual-kNN: ~30s; CKA-minibatch: ~10s). Use the *same* gram-matrix / NN-index across permutations to amortize: only the post-hoc statistic recomputation costs scale with R. Total calibration overhead ≈ 5–10× the raw-metric cost.

**Full pairwise sweep.** ~30 models → ~435 ordered pairs × ~5 metrics × ~1 GPU-hour ≈ ~2000 GPU-hours of metric compute, plus extraction. **Total project compute ≈ 2000–3000 A100-hours**, fitting comfortably in a 1-month allocation on 4×A100 (≈ 2880 GPU-hours/month) or 8×A40 (which delivers ~70% the throughput at ~140% the aggregate memory — equally feasible).

### 7.3 Phased timeline

- **Phase 0 (1 week):** Fork ReSi codebase; verify per-model forward passes; pilot extraction at 1000 stimuli on Pythia-2.8B, Mamba-2.8B, RWKV-4-3B; reproduce a published similarity number (e.g., Pythia-Pythia layer-wise CKA from Klabunde et al.) as a smoke test.
- **Phase 1 (1 week):** Full hidden-state extraction across all models and stimuli; cache to disk.
- **Phase 2 (2 weeks):** Run all metrics (linear & RBF CKA; mutual k-NN at k ∈ {10, 25, 50, 100}; Procrustes via 256-dim PCA; Soft Matching Distance; RSA; SVCCA) on all pairs and all layer-pairs; compute null-calibration permutations.
- **Phase 3 (1 week):** Fit scaling laws s(N), s(N, T, C); decompose by stimulus subset; produce final figures (similarity heatmaps, scaling curves, raw-vs-calibrated comparisons, layer-correspondence maps).
- **Phase 4 (1 week):** Statistics — bootstrap CIs over stimulus subsets, leave-one-model-out cross-validation of scaling fits, sensitivity analyses for pooling, projection, and permutation choices.

### 7.4 Reproducibility deliverables

Release: (i) extracted representation tensors (or extraction scripts with deterministic seeds), (ii) the full similarity tensor (model_A × layer_A × model_B × layer_B × metric × stimulus_subset), (iii) analysis notebooks producing every figure, (iv) a `models.yaml` manifest pinning HuggingFace revisions for every checkpoint.

---

## 8. What Would Decisively Resolve the Question

The protocol is designed so that four outcomes lead to clear conclusions:

1. **PRH holds for non-Transformer architectures**: null-calibrated mutual-kNN between {Pythia, Mamba, RWKV} rises monotonically with scale, with overlapping exponents, and matches the Pythia↔OLMo same-architecture-different-data baseline by 7B. Most informative experiment: the {Pythia-2.8B, Mamba-2.8B, RWKV-4-3B} triple at the Pile-controlled cell.
2. **PRH holds but only locally** (Aristotelian view): null-calibrated *mutual-kNN* increases with scale but null-calibrated CKA/Procrustes do not. Architectures share *who-is-near-whom* but not *how-far-apart*. Currently the most likely outcome given Gröger et al.'s findings on standard PRH benchmarks.
3. **PRH fails for state-tracking / retrieval stimuli**: mutual-kNN converges on syntactic and semantic stimuli but diverges (or plateaus low) on state-tracking, factual-retrieval, and long-context-recall stimuli, consistent with RWKV-7's theoretical super-TC⁰ expressivity and Mamba's empirical retrieval gap (arXiv:2409.05395, Bamba blog).
4. **PRH fails outright for SSM/RWKV**: cross-architecture similarity does not approach within-architecture similarity even at 7–13B; architecture imposes a strong, persistent inductive bias on the learned representation.

Each outcome maps to a distinct theoretical position; the protocol's null-calibration plus pre-registered thresholds ensure the result is interpretable regardless of which obtains.

---

## 9. Caveats and Open Issues

- The arXiv:2406.16722 reference in the original brief could not be verified through standard search. The SSM↔Transformer transition literature most aligned with that description is MOHAWK (2408.10189) and the retrieval-aware distillation work by Bick et al. Confirm the exact reference before citing.
- Hybrid models (Zamba, Jamba, Bamba, FalconH1) are *not* pure non-Transformer architectures; treat them as a separate stratum to avoid contaminating the pure-Mamba ↔ Transformer comparison.
- The Aristotelian critique implies that several previously published cross-architecture convergence results may shrink substantially after calibration; conservative reporting of both raw and calibrated scores is essential.
- RWKV-7's super-TC⁰ expressivity is a recent theoretical claim from the 2025 RWKV-7 paper; the empirical implication for representational convergence with Transformers is, as of writing, untested at scale — this protocol is among the first that can resolve it.
- All training-data-overlap claims for proprietary corpora (LLaMA, Mistral, Qwen) cannot be perfectly verified; for the strongest causal conclusions, restrict to the Pile-trained triple (Pythia/Mamba-1/RWKV-4) and the open-data pairs (Falcon-7B/Falcon-Mamba; OLMo/Bamba).
- The Tjandrasuwita et al. finding that alignment is not always a reliable performance indicator means that even strong cross-architecture convergence does not automatically validate PRH's stronger claim of approach to a "shared statistical model of reality"; functional / decodability tests (Section 2.1, last row) provide the complementary functional grounding.

