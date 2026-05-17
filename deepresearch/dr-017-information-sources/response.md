### Neural network probe signal decomposition

用户:


Claude:
his is a well-scoped research question covering probe signal decomposition in neural network interpretability. The document is detailed and specific enough that I can dive straight into research. Let me kick off a deep dive now.
our probe accuracy decomposition report is ready. It analyzes how to separate three signal sources in neural network probes — input bleed-through, training-objective statistics, and genuine model computation — drawing on conditional V-information (Hewitt-Ethayarajh-Liang 2021), deconfounded CKA (Cui 2022), and the Encoding Probe (Shen 2025). The report covers information-theoretic frameworks (V-information, PID, MDL probing), controlled cross-objective studies (MLM vs LM vs contrastive vs diffusion across NLP, vision, and speech), a four-stage experimental pipeline with decision benchmarks, recommended baselines and controls, tooling guidance (`nninfo`, `dit`, MINE), and caveats about PID measure disagreement, MINE variance, and several preprints with anomalous arXiv identifiers.
# Decomposing Probe Accuracy: Input Bleed-Through, Training Statistics, and Model Computation

## TL;DR
- **The single most actionable finding**: the cleanest existing template for a "probe-signal source decomposition" experiment combines (a) Hewitt, Ethayarajh & Liang's 2021 *conditional V-information probing* (which subtracts the input/baseline contribution), (b) Cui et al.'s 2022 *deconfounded CKA* (which strips the input-similarity confound), and (c) Shen et al.'s 2025 *Encoding Probe* (which reverses probe direction so feature contributions become directly comparable under a shared regression). Together these isolate input bleed-through, training-objective statistics, and genuine model computation in a way no single method does.
- **For statistical machinery**, V-information (Xu et al. 2020) plus conditional MI via MINE (Belghazi et al. 2018) or Donsker–Varadhan/InfoNCE variational bounds is the most defensible decomposition; Partial Information Decomposition (Williams & Beer 2010; Ehrlich et al. TMLR 2023) cleanly maps onto the three target sources as redundant (input), unique-to-model, and synergistic information, and the `nninfo` package (Priesemann group) is the closest off-the-shelf tool.
- **For cross-objective comparison**, the best controlled designs in the literature — Voita, Sennrich & Titov 2019 (MT/LM/MLM on identical Transformers); Park et al. ICLR 2023 (CL vs MIM on identical ViTs); Alajrami & Aletras ACL 2022 (five pretraining objectives, same BERT architecture); DiffMAE (Wei et al. ICCV 2023) — show that probing differences across objectives are large and systematic, but Alajrami & Aletras's null result also shows that probe accuracy alone cannot resolve them, motivating exactly the decomposition this report is about.

## Key Findings

1. **Probing accuracy is structurally confounded.** Hewitt & Liang (EMNLP 2019, arXiv:1909.03368) showed via "control tasks" (random labels assigned to word types) that probes routinely achieve high linguistic accuracy *and* high random-label accuracy. Their reported MLP probe on ELMo achieves 97.3% POS accuracy *and* 92.8% control-task accuracy — yielding a selectivity of only 4.5 — versus a linear probe's 97.2% linguistic / 71.2% control / 26.0 selectivity. *Accuracy alone does not separate "representation encodes property" from "probe learned property."*

2. **Information-theoretic reframings disagree about what probes measure.** Pimentel et al. (ACL 2020) recast probing as mutual-information estimation and proved that any deterministic function of the input contains the same I(Y;·) as the input itself — making the original probing question "nonsensical" without further structure. Voita & Titov (EMNLP 2020) sidestep this by measuring *minimum description length* (MDL) of labels given representations via online/variational coding; MDL reliably distinguishes pretrained from randomly initialized ELMo where accuracy does not. Xu et al. (ICLR 2020) introduce predictive **V-information**, which restricts the decoder class V and *can be created by computation*, violating the data-processing inequality and making it the right quantity for "what the model has made usable."

3. **Conditional probing is the most direct existing decomposition.** Hewitt, Ethayarajh & Liang (EMNLP 2021, "Conditional probing: measuring usable information beyond a baseline," arXiv:2109.09234) train two V-information probes — one on a baseline B (e.g., non-contextual word embeddings) and one on [B; R] — and report the difference. This is operationally equivalent to I_V(R; Y | B) and answers exactly "how much additional usable signal does the representation add over what is trivially in the input?" It is the closest published analog to the conditional-MI decomposition the user asks about.

4. **Random/untrained baselines beat strong intuitions.** Cui, Kumar, Marttinen & Kaski (NeurIPS 2022, arXiv:2202.00095) show that CKA and RSA between *completely random* networks are spuriously high because both inherit the input-space population structure; their covariate-adjusted (deconfounded) CKA/dCKA removes the input confound. Follow-ups (Davari et al. 2022 "Reliability of CKA"; Murphy et al. 2024 "Correcting Biased CKA") corroborate that biased linear CKA is sensitive to feature-sample ratio and can be manipulated without behavioral change. In speech, Gubian et al. (arXiv:2506.10855, Jun 2025) report that even untrained HuBERT reaches relatively high phone–speaker subspace orthogonality in late layers, showing that geometric signatures previously attributed to learning are partly architectural.

5. **The Encoding Probe (Shen et al., arXiv:2605.00607, 2025) is the "reverse probe" the user asks about.** Rather than predicting a property from representations, it *reconstructs* representations from a panel of interpretable features (acoustics, phonetics, syntax, lexicon, speaker) under a shared regression and uses feature-ablation R² drops as the comparison currency. This solves two structural problems of decoding probes: (i) different features had different ceilings under decoding probes and were not comparable, and (ii) correlated features inflate individual probe accuracies. The paper's headline finding — "speaker-related effects vary strongly across different training objectives and datasets, while syntactic and lexical features contribute independently to reconstruction" — is exactly the cross-objective signal the user wants.

6. **PID gives the cleanest theoretical decomposition.** Williams & Beer's Partial Information Decomposition splits I({sources}; target) into redundant, unique, and synergistic atoms. Mapping onto the probe-decomposition problem: redundant(input, representation; concept) = "input bleed-through," unique(representation; concept | input) = "model contribution," and synergistic = information requiring *both* the raw input and the representation. Ehrlich, Schneider, Priesemann, Wibral & Makkeh (**TMLR 2023**, arXiv:2209.10438) apply this directly to DNN layers and observe that "for quantized deep neural networks solving the MNIST *and CIFAR10* tasks, representational complexity decreases both through successive hidden layers and over training." The accompanying `nninfo` package operationalizes the framework.

7. **Cross-objective probing differences are large but not monotonic.** Voita, Sennrich & Titov (EMNLP 2019, arXiv:1909.01380) trained the *same* Transformer with MT, left-to-right LM, and MLM and found that "as you go from bottom to top layers, information about the past in left-to-right language models gets vanished and predictions about the future get formed. In contrast, for MLM, representations initially acquire information about the context around the token, partially forgetting the token identity… then [recreate it] at the top MLM layers." Fayyaz et al. (BlackboxNLP 2021, arXiv:2109.05958) extended this with MDL probing to BERT/ELECTRA/XLNet: "ELECTRA tends to encode linguistic knowledge in the deeper layers, whereas XLNet instead concentrates that in the earlier layers." Park et al. (ICLR 2023) found "CL trains self-attentions to capture longer-range global patterns than MIM… CL utilizes the low-frequency signals of the representations, but MIM utilizes high-frequencies… CL plays a crucial role in the later layers, while MIM mainly focuses on the early layers." DiffMAE (Wei et al., ICCV 2023, arXiv:2304.03283) reports that pure DDPM ViT-L pretraining yields 83.4% ImageNet-1K fine-tune accuracy versus MAE's 85.9% and DiffMAE's 85.8% — with the explicit statement that "directly pre-training with diffusion models does not produce strong representations" unless reformulated as masked denoising.

8. **A 2022 null result motivates exactly the experiment in the prompt.** Alajrami & Aletras (ACL 2022, arXiv:2203.10415) pretrained BERT-Base from scratch with five objectives (two linguistically motivated, three not) and report verbatim: "We find strong evidence that there are only small differences in probing performance between the representations learned by the two different types of objectives. These surprising results question the dominant narrative of linguistically informed pre-training." Either objectives genuinely don't matter for these properties (unlikely given Voita 2019 and Park 2023) or accuracy-based probes are too blunt — the source-decomposition framing should resolve this.

## Details

### §1. Existing methods for information-source analysis in probes

| Method | Year | Core idea | What it isolates |
|---|---|---|---|
| Control tasks / selectivity (Hewitt & Liang) | 2019 | Subtract random-label accuracy | Probe memorization capacity |
| Control function (Pimentel et al., ACL 2020) | 2020 | Restrict probe family | Probe complexity; Zhu & Rudzicz 2020 prove equivalent to control tasks modulo irrelevant terms |
| MDL probing (Voita & Titov, EMNLP 2020) | 2020 | Online/variational code length | Effort to extract label; separates pretrained from random |
| Pareto probing (Pimentel et al., EMNLP 2020) | 2020 | Sweep accuracy×complexity frontier | Operating point along extraction-effort curve |
| Bayesian MI (Pimentel & Cotterell, EMNLP 2021) | 2021 | Bayesian agents with finite data | Restores "processing can help" property |
| Conditional probing (Hewitt, Ethayarajh & Liang, EMNLP 2021) | 2021 | V-info gap between [B] and [B;R] | Usable info beyond an input baseline — most relevant prior |
| Amnesic probing (Elazar et al., TACL 2021) | 2021 | INLP nullspace projection + behavioral test | Whether the model *uses* the property; warns "conventional probing performance is not correlated to task importance" |
| Deconfounded CKA/RSA (Cui et al., NeurIPS 2022) | 2022 | Covariate-adjust for input similarity | Removes spuriously high CKA between random nets |
| Encoding Probe (Shen et al., arXiv:2605.00607) | 2025 | Predict R from interpretable features; feature-ablation R² | Comparable per-feature contributions; controls feature correlation |
| Geometric / orthogonal-subspace (Liu, Tang & Goldwater 2023; Gubian et al. 2025) | 2023–25 | Cumulative residual variance; subspace overlap | Whether two properties live in orthogonal subspaces; calibrates against untrained nets |

Zhu & Rudzicz (arXiv:2009.07364) prove that control tasks and control functions are *equivalent modulo irrelevant terms*: the two dominant pre-2021 approaches to "what does the probe vs the representation contribute" reduce to the same thing operationally.

### §2. Statistical and information-theoretic decomposition tools

**Conditional MI as "model contribution."** If X is input, R = f(X) is representation, and Y is concept, then I(R; Y | X) is exactly zero by data-processing for any deterministic f. So "model contribution" must be defined under a non-Shannon framework — either V-information (restricting decoders to a tractable class V, so I_V(R; Y) > I_V(X; Y) becomes possible and meaningful) or PID atoms.

**PID mapping.** Williams & Beer (2010) and the trivariate refinement of Pica, Piasini, Chicharro & Panzeri (Entropy 2017) give the cleanest framework: I({X, R}; Y) = Red(X, R; Y) + Unq(X; Y | R) + Unq(R; Y | X) + Syn(X, R; Y). The probe-signal sources map as: Red ≈ input bleed-through; Unq(R; Y|X) ≈ model computation; training-objective statistics live partly in Syn and partly in mass-shifting between Red and Unq across training. Ehrlich et al. (TMLR 2023) make this computable for small layers (MNIST, CIFAR10) and propose subsampling/coarse-graining for larger ones.

**Estimators.** MINE (Belghazi et al., ICML 2018) gives a Donsker–Varadhan lower bound trainable by SGD; Poole et al. 2019 ("On Variational Bounds of MI") show MINE-style bounds can have variance growing exponentially with true MI, so they should be paired with InfoNCE / NWJ / SMILE bounds for robustness. For discrete labels on continuous representations, the practical conditional-MI estimator is the "two-probe difference" of Hewitt-Ethayarajh-Liang 2021, since V-information has PAC-style estimation guarantees (Xu et al. 2020) that Shannon MI does not.

**Software.** The Priesemann group's `nninfo` (github.com/Priesemann-Group/nninfo, TMLR 2023) implements layer-wise information measurements on small DNNs; the NIT toolbox (bioRxiv 2022) covers neuroscience-style estimators including bias correction; `dit` (Python) implements most PID definitions (Williams–Beer, Bertschinger et al., Ince's I_ccs, Kolchinsky's I_∩). For LM-scale experiments, the V-info-based conditional probing pipeline from the Hewitt et al. 2021 GitHub release is the only tested option at scale.

**Random-label and shuffled-input tests** remain the practical first-line statistical control: report accuracy under (a) shuffled labels (Hewitt–Liang control task); (b) shuffled inputs paired with true labels; (c) untrained random-initialization model (Voita & Titov 2020 show MDL but not accuracy reliably distinguishes this); and (d) representations from a model trained on a *different* but plausible objective with same architecture.

### §3. Comparative studies across training objectives

The strongest controlled comparisons:

- **Voita, Sennrich & Titov (EMNLP 2019, arXiv:1909.01380)**: same Transformer trained as MT vs left-to-right LM vs MLM. Established that representational dynamics across layers diverge sharply by objective. Used CCA + token-identity MI rather than supervised probes.
- **Fayyaz, Aghazadeh, Modarressi, Mohebbi, Pilehvar (BlackboxNLP 2021, arXiv:2109.05958)**: MDL probing across BERT/ELECTRA/XLNet — the cleanest "same architecture, different objective" study in NLP. ELECTRA pushes linguistic knowledge deeper; XLNet pushes it shallower.
- **Alajrami & Aletras (ACL 2022, arXiv:2203.10415)**: five-objective pretraining of BERT-base from scratch. Surprisingly small probing differences between linguistically and non-linguistically motivated objectives — a finding that *demands* a source-decomposition analysis to interpret. Their abstract gives no quantitative threshold for "small"; per-task numbers are in the body tables.
- **Park, Kim, Heo, Kim & Yun (ICLR 2023, arXiv:2305.00729)**: CL (MoCo v3) vs MIM (SimMIM, MAE) on identical ViTs. Different attention ranges, different frequency content, different shape/texture bias, different layer-wise specialization. Best single CV-side controlled probing study.
- **Wei, Mangalam, Huang, Li et al. (DiffMAE, ICCV 2023, arXiv:2304.03283)**: pure DDPM ViT-L pretraining → 83.4% ImageNet-1K fine-tune, vs MAE 85.9%, DiffMAE 85.8% — vanilla diffusion underperforms MAE/contrastive on identical ViTs unless masking is added.
- **Clark & Jaini (NeurIPS 2023, arXiv:2303.15233)**: Stable Diffusion / Imagen as zero-shot classifiers via denoising likelihood, competitive with CLIP. But the preprint Braunstein et al. (arXiv:2511.08075) cautions via linear-ridge probing that "semantic understanding comes mostly from the pretrained CLIP [text encoder] and not from the reverse diffusion process" — a striking case where probe signal in a diffusion stack is actually input bleed-through from a contrastive encoder.
- **Gubian et al. (arXiv:2506.10855, Jun 2025)** and **Liu, Tang & Goldwater (Interspeech 2023, arXiv:2305.12464)** for speech SSL: phones, tones, and speakers occupy orthogonal subspaces across pretraining languages; later wav2vec2/data2vec layers have *worse* speaker linear-separability than untrained models — an explicit "model destroys input information" finding.

### §4. Experimental design recommendations

**Recommended decomposition pipeline (4 components, in order):**

1. **Input bleed-through floor.** Compute I_V(X_simple; Y) where X_simple is a deliberately weak feature extractor (bag-of-words, raw pixels, MFCCs, or non-contextual embeddings). Report this floor for every property/probe.
2. **Training-statistics floor.** Compute I_V(R_untrained; Y) for the same architecture with random weights. The Cui 2022 result demands this; report deconfounded CKA simultaneously. The gap I_V(R_untrained; Y) − I_V(X_simple; Y) is "what the architecture + data distribution gives you for free."
3. **Model contribution.** Compute conditional V-info I_V(R; Y | X_simple) using the Hewitt-Ethayarajh-Liang 2021 two-probe estimator: train probe on [X_simple], train probe on [X_simple; R], subtract. This is the cleanest scalar for "model adds beyond input."
4. **Amnesic/causal complement.** For a subset of properties, apply Elazar et al.'s amnesic probing (INLP nullspace projection) and measure downstream behavioral change. If conditional V-info is high but amnesic-removal causes no behavior change, the model *encodes but does not use* the property — a fourth category beyond the user's three sources.

**Best model/task pairs to distinguish the three sources:**

- **Speech**: wav2vec2 vs HuBERT vs WavLM probed for phones, tones, speaker. Property-orthogonality already mapped (Gubian 2025); untrained baselines well-characterized.
- **Vision**: ViT under MoCo v3 (CL), MAE (MIM), DiffMAE (diffusion), DINO (self-distillation), supervised — probed for shape/texture/frequency. Park et al. 2023's setup is reproducible.
- **NLP**: BERT-base under five pretraining objectives à la Alajrami & Aletras 2022, probed for POS, dep-edges, NER, SRL, coreference (the edge-probing suite). Use MDL + conditional V-info + PID atoms together. The Alajrami null result will be resolved by decomposition if any of the four sources differ.

**Required baselines and controls:**

| Control | Purpose |
|---|---|
| Random-label control task (Hewitt–Liang) | Bound probe memorization |
| Shuffled-input control | Detect probe leakage from non-content statistics |
| Untrained-architecture control | Quantify architectural prior (Cui 2022) |
| Same architecture, different objective | Isolate training-objective contribution (Voita 2019; Alajrami 2022; Park 2023) |
| Bag-of-words / MFCC / raw-pixel input baseline | Quantify input bleed-through (Hewitt-Ethayarajh-Liang 2021) |
| Amnesic counterfactual (Elazar 2021) | Test whether encoded ≠ used |
| Deconfounded CKA (Cui 2022) | Geometric corroboration of probe results |

**Tooling concretely**: `nninfo` for small-scale MI/PID on toy nets; `dit` for PID atom estimation; the Hewitt-Ethayarajh-Liang conditional-probing repo for V-info at scale; MINE/InfoNCE for continuous-rep continuous-concept MI lower bounds (with Poole 2019 variance caveat); Shen et al.'s Encoding Probe code as the reverse-direction sanity check.

## Recommendations

**Stage 1 (≤2 weeks).** Reproduce conditional V-information (Hewitt-Ethayarajh-Liang 2021) on BERT/RoBERTa with non-contextual baselines for POS, NER, dep-edges. *Decision benchmark:* if conditional V-info is < 0.1 nats for a property, the property is overwhelmingly input bleed-through; do not run downstream PID on it.

**Stage 2 (1–2 months).** Add an untrained-network arm and a same-architecture-different-objective arm (MLM vs causal LM vs ELECTRA discriminator at 110M params, identical tokenizer and data). Compute MDL probes and deconfounded CKA in parallel. *Decision benchmark:* if MDL gap between trained and untrained < 20% but accuracy gap < 5%, the standard probing literature has been measuring accuracy-saturated noise — proceed to Stage 3.

**Stage 3 (3+ months).** Apply PID with `nninfo` / `dit` on a low-dimensional bottleneck of the representation (Ehrlich et al. 2023's coarse-graining trick). Report Red, Unq, Syn explicitly. *Decision benchmark:* if Unq(R; Y | X) dominates Syn, the model is creating new usable features (classical "learned representation"); if Syn dominates, training statistics matter only in combination with input — an underexplored regime worth a focused paper.

**Stage 4 (corroboration).** Run amnesic probing on the same properties; if conditional V-info is high but amnesic-removal does not affect downstream behavior, flag the property as encoded-but-unused and exclude it from causal/mechanistic claims.

**For cross-objective claims specifically**, pair Park-et-al-style identical-architecture probing with the Encoding Probe (Shen et al. 2025) reverse direction — the Encoding Probe tells you which input features account for representation similarity, while conditional V-info tells you how much the model adds beyond them.

## Caveats

- **Several cited preprints have anomalous arXiv identifiers**: Shen et al.'s Encoding Probe (arXiv:2605.00607), Braunstein et al. (arXiv:2511.08075), and Vicentino's AR-vs-MDLM comparison (arXiv:2603.22075) appear with future-dated IDs in the indexes consulted; treat all three as preliminary preprints and verify canonical identifiers at use time. They are *not* peer-reviewed.
- **The Pimentel et al. 2020 "MI = same for any deterministic f(X)" result** is mathematically airtight but operationally vacuous; do not over-cite it as a reason to abandon probing — V-information (Xu et al. 2020) resolves it by restricting decoder capacity, and *this is precisely why* conditional V-info is the right tool.
- **PID has no consensus measure.** Williams–Beer's I_min has been criticized (Harder et al. 2013); Bertschinger et al. 2014, Ince 2017 (I_ccs), and Kolchinsky 2022 give different decompositions that can disagree. Report results under at least two PID definitions or note this explicitly.
- **MINE variance** grows exponentially with true MI (Poole et al. 2019), so MINE-only estimates of conditional MI in high dimensions are unreliable. Use V-information / two-probe gaps as the primary estimator and MINE only as corroboration.
- **CKA and probing can both be high without functional similarity** (Davari et al. 2022, "Reliability of CKA"); never report either in isolation.
- **Amnesic probing via INLP removes linearly extractable information**, not all information; Ravfogel et al. 2022 have shown the procedure has subtle failure modes when properties are entangled.
- **Gubian et al. 2025's untrained-HuBERT result** is a single finding in one architecture; whether random networks generically produce moderate task-relevant geometric structure across modalities is itself a candidate open question under this report's framework.

---

**Coverage table (spec audit):**

| Spec item | Covered? |
|---|---|
| Input bleed-through, surface features | §1 (control tasks), §3 Findings 1–3 |
| Cui 2022 random networks / CKA | Finding 4, §2, Stage 2 |
| Gubian 2025 reverse/encoding probes | Finding 5 (Encoding Probe = Shen et al. 2025 with Gubian as co-author of the underlying speech-orthogonality work, arXiv:2506.10855) |
| MI / conditional MI for input vs model | Finding 3, §2 |
| Hewitt & Liang 2019 control tasks/selectivity | Finding 1 (with numbers) |
| Voita & Titov 2020 MDL probes | Finding 2 |
| Pimentel 2020 information-theoretic probing | Finding 2 |
| V-information / Xu 2020 | Finding 2 |
| PID redundant/unique/synergistic | Finding 6 |
| Partial correlation / mediation | §2 (covered via PID atoms and conditional probing as the operational equivalent) |
| Random label / shuffled baselines | §2, §4 controls table |
| MINE, variational MI bounds | §2 |
| nninfo software | Finding 6, §2, Stage 3 |
| NTP vs MLM vs contrastive vs diffusion | §3 (Voita 2019, Fayyaz 2021, Alajrami 2022, Park 2023, DiffMAE 2023, Clark & Jaini 2023) |
| Controlled architecture experiments | §3, Stage 2 recommendations |
| Statistical methods for the experiment | §4 + Recommendations |
| Models/tasks distinguishing sources | §4 |
| Baselines and controls | §4 controls table |
| Tools and practical applicability | §2, Stage 3 |

