### Research on probing methods and cross-architecture applications

用户:


Claude:

This is a very detailed and well-structured research request — your prompt already specifies exactly what you need across all five tasks. I'll dive straight into comprehensive research on this.

Let me kick off an in-depth research process covering probing methods, their taxonomy, cross-architecture applications, and open methodological questions.
our comprehensive survey of probing methods in deep learning is ready. It covers a taxonomy of probing techniques (classic, causal/interventional, lens-based, direction extraction, sparse dictionary, and evaluation frameworks), architecture-specific probing work across ten model families (CNNs, encoder/decoder transformers, ViTs, SSMs, hybrids, multimodal, diffusion, world models, and latent reasoning models), cross-architecture comparison studies, ten open methodological questions, and eight promising future directions. The report synthesizes findings from foundational work like Alain & Bengio's linear probes through recent developments including AxBench, Patchscopes, refusal directions, and SAE-based methods, with particular attention to the persistent "encoding ≠ use" gap and the need for universal cross-architecture methodologies.
# A Comprehensive Survey of Probing Methods in Deep Learning and Representation Analysis

## Executive Summary

Probing methods are post-hoc diagnostic tools that ask: *what information is encoded in a neural network's internal representations, and how is it used?* Since Alain & Bengio (2016) introduced the linear classifier probe, the field has expanded from simple "decodability" experiments into a sprawling methodology that now encompasses causal interventions, sparse-dictionary decompositions, training-free transcoders ("lenses"), activation-steering directions, and architecture-specific tools. This report enumerates these methods, organizes them taxonomically, summarizes what is known across ten architecture families, and articulates the open questions and the most promising directions for *universal*, methodology-level conclusions.

---

## Task 1 — Taxonomy of Probing and Related Methods

### A. Classic Probing

- **Linear probe (Alain & Bengio, 2016; arXiv:1610.01644).** A linear classifier trained on frozen intermediate activations to predict a label. Originally validated on CNNs (Inception, ResNet) on ImageNet. Core finding: linear separability increases monotonically with depth in classification CNNs; intermediate layers carry partially-formed but linearly readable class information.

- **SentEval probing suite (Conneau et al., 2018; arXiv:1805.01070).** Ten "auxiliary" probing tasks (sentence length, word content, bigram shift, tree depth, top-constituent, tense, subject number, object number, semantic odd-man-out, coordination inversion) for sentence embeddings. Validated on LSTM, GRU, BoW, InferSent, SkipThought, and early transformer encoders. Core finding: different architectures encode surface, syntactic, and semantic information at different depths; BoW carries more syntactic information than expected.

- **Edge probing (Tenney, Xia, Chen et al., 2019; arXiv:1905.06316).** A probe that classifies relations between pairs of token spans (POS, constituents, dependencies, NER, semantic-role labels, coreference, SPR, relations). Validated on ELMo, GPT, BERT, CoVe. Core finding: BERT recovers a "classical NLP pipeline" — POS in early layers, parsing mid-network, semantic roles and coref late.

- **Structural probe (Hewitt & Manning, 2019; arXiv:1905.06316/N19-1419).** A learned linear transformation B such that squared distances ‖B(h_i − h_j)‖² approximate dependency-tree distances; a sibling probe recovers tree depth from squared norms. Validated on ELMo and BERT. Core finding: a *single* linear subspace of BERT encodes the parse tree geometrically — strong evidence for the "linear representation" hypothesis.

- **Control task / selectivity (Hewitt & Liang, 2019; arXiv:1909.03368).** Pairs a real probing task with a randomized control task (e.g., random POS labels). Selectivity = real-task accuracy − control-task accuracy. Validated on ELMo. Core finding: high-capacity (MLP) probes achieve near-perfect control-task accuracy and have low selectivity, i.e., they *memorize*; linear/bilinear probes are much more selective. This launched the now-standard practice of always running control tasks.

- **MDL probe (Voita & Titov, 2020; arXiv:2003.12298).** Reframes probing as compression: a representation "contains" a label only insofar as it shortens the *minimum description length* of the label given the representation (variational or online-code variants). Validated on ELMo, BERT, XLNet, randomly initialized baselines. Core finding: MDL gives a stable ranking even when accuracies are saturated; randomly initialized baselines reveal that ordinary accuracy probes are easily misled.

- **Information-theoretic probing (Pimentel et al., 2020; arXiv:2004.03061).** Argues probing should estimate the *mutual information* I(rep; property), with the caveat that, by the data-processing inequality, deeper representations cannot strictly contain *more* information than earlier ones — so probing is really about *usability*/*accessibility*. Companion paper "Pareto Probing" (Pimentel et al., 2020; arXiv:2010.02180) formalizes the accuracy–complexity Pareto frontier.

### B. Causal / Interventional Probing

- **AMNESIC probing (Elazar, Ravfogel, Jacobi, Goldberg, 2021; arXiv:2006.00995).** Combines INLP (iterative null-space projection) with behavioral evaluation: remove a property from representations, then check whether downstream behavior changes. Validated on BERT. Core finding: many features that probes can read out are *not* used by the model — encoding ≠ use.

- **AlterRep (Ravfogel, Prasad, Linzen, Goldberg, 2021; arXiv:2105.06965).** Counterfactually intervenes on a property by projecting along a probe-found direction, then measures downstream behavioral change. Validated on BERT for syntactic number agreement. Core finding: causally manipulating the linear subspace of grammatical number shifts model agreement predictions, supporting causal use of that subspace.

- **INLP / Iterative Null-space Projection (Ravfogel et al., 2020; arXiv:2004.07667).** Iteratively trains linear classifiers for an attribute and projects activations onto each one's null space, eliminating *linear* readability. Validated on word embeddings (FastText), BERT. The "linear concept erasure" line continues with **R-LACE** (Ravfogel et al., 2022; arXiv:2201.12091) and **LEACE** (Belrose et al., 2023; arXiv:2306.03819) which give *optimal* linear erasure in closed form.

- **Causal Basis Extraction (CBE) / LEACE (Belrose et al., 2023; arXiv:2306.03819).** Closed-form least-squares "concept-scrubbing" guaranteed to remove all linear information about a concept while changing activations as little as possible. Validated on BERT, GPT-2, GPT-Neo. Core finding: LEACE causally degrades a model's ability to use the scrubbed feature; on bias benchmarks, much less damage than INLP.

- **DAS — Distributed Alignment Search (Geiger et al., 2023/2024; arXiv:2303.02536 and "Causal Abstraction" arXiv:2301.04709).** Learns an orthogonal rotation of the activation basis such that *interchange interventions* on a low-dimensional subspace implement a hypothesized high-level causal variable. **Boundless DAS** (arXiv:2305.08809) scales it to LLaMA. Validated on BERT, GPT, Alpaca/LLaMA. Core finding: DAS can localize abstract variables (e.g., a "left operand" of a comparison) to specific subspaces, providing the strongest causal probing test currently available.

### C. "Lens" Methods (Training-Free / Trained Transcoders)

- **Logit Lens (nostalgebraist, 2020; LessWrong post).** Applies the model's own unembedding (W_U) to every intermediate residual stream, reading early "guesses" of the next-token distribution. Validated on GPT-2, GPT-Neo, later applied to almost all decoder-only LMs. Core finding: predictions are progressively refined; many concepts are decodable many layers before output.

- **Tuned Lens (Belrose, Furman, Smith et al., 2023; arXiv:2303.08112).** Learns an *affine* translator per layer to map intermediate residuals into the final-layer basis, then applies W_U. More reliable than the logit lens for early layers. Validated on Pythia, GPT-Neo, OPT, BLOOM, LLaMA. Core finding: tuned lens has substantially lower perplexity and reveals smoother "prediction trajectories." Reveals "iterative inference" structure across decoder LMs of multiple families.

- **Patchscopes (Ghandeharioun, Caciularu, Pearce, Dixon, Geva, 2024; arXiv:2401.06102).** Uses the LM itself as the decoder: patch a hidden state from one forward pass into a separate "explanation prompt" and let the model verbalize. Validated on LLaMA-2, Vicuna, GPT-J, Pythia. Core finding: hidden states are decodable into natural language even when logit lens fails, esp. in early layers; supports entity resolution, multi-hop tracing, and self-correction.

- **Related**: **SelfIE** (Chen et al., 2024; arXiv:2403.10949), **Speaking Probes**, and **Superscopes** (Aiezza et al., 2025; arXiv:2503.02078) extend the same idea by amplifying weak features (CFG-style) before patching.

### D. Direction Extraction & Steering

- **Difference-in-means / ActAdd (Turner, Thiergart, Leech et al., 2023; arXiv:2308.10248).** Subtracts mean activations of negative-prompt forward passes from positive-prompt passes, yielding a steering vector added at inference. Validated on GPT-2, OPT, LLaMA. Core finding: simple mean differences can shift sentiment, topic, refusal, and other behaviors with little capability degradation.

- **Representation Engineering / RepE reading vectors (Zou, Phan, Chen et al., 2023; arXiv:2310.01405).** "LAT scan" extracts reading vectors via PCA of contrastive activations and uses them for both *detection* and *steering*. Validated on LLaMA-2, Vicuna, GPT-J. Core finding: honesty, fairness, morality, emotion, power-seeking, and many "social" attributes have low-dimensional linear readouts.

- **Contrastive Activation Addition / CAA (Rimsky, Gabrieli, Schulz et al., 2024; arXiv:2312.06681).** Diff-in-means computed per behavior over hundreds of paired examples, applied at a chosen layer. Validated on LLaMA-2 7B/13B chat. Core finding: CAA reliably steers a wide set of evaluations (sycophancy, hallucination, corrigibility); steering vectors generalize across prompts.

- **Refusal direction (Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda, 2024; arXiv:2406.11717).** Difference-in-means at the post-instruction token between harmful and harmless prompts gives a one-dimensional subspace whose ablation jailbreaks 13 open chat models up to 72B (LLaMA-2/3, Qwen, Yi, Gemma, Mistral). Core finding: refusal is mediated by a *single* direction; weight orthogonalization gives a clean, training-free jailbreak. (Note the user-provided attribution "Sharma+ 2024" appears to refer to Arditi et al.; subsequent work — Wollschläger et al., 2025, arXiv:2502.17420 — argues the refusal subspace is actually a multi-dimensional concept cone.)

- **Causal inner product (Park, Choe, Veitch, 2024; arXiv:2311.03658, "The Linear Representation Hypothesis...").** Defines a *causal* inner product on the unembedding space under which "concept directions" become orthogonal precisely when concepts are causally separable. Validated on LLaMA-2. Core finding: the right metric matters — naive cosine similarities of unembedding directions are misleading; the causal inner product makes orthogonality and projection behave correctly.

### E. Sparse Dictionary / SAE-Based Probing

- **Sparse autoencoders for features (Bricken et al., 2023, Anthropic "Monosemanticity"; Cunningham, Ewart, Riggs, Huben, Sharkey, 2023; arXiv:2309.08600).** Trains an overcomplete sparse autoencoder on residual-stream activations to recover "monosemantic" features. Validated on Pythia-70M/410M, GPT-2, Claude-3 Sonnet (Templeton et al., "Scaling Monosemanticity," 2024).

- **Sparse probing (Gurnee et al., 2023, "Finding Neurons in a Haystack"; arXiv:2305.01610).** Trains k-sparse linear probes on neuron activations (or SAE latents) to identify which neurons/features encode a concept. Validated on Pythia. Core finding: many concepts are surprisingly mono- or few-neuron-localized; superposition explains polysemanticity.

- **Variants**: Gated SAEs (Rajamanoharan et al., 2024; arXiv:2404.16014), JumpReLU SAEs, BatchTopK SAEs, Matryoshka SAEs (arXiv:2503.17547), Transcoders (Dunefsky et al., 2024) and Crosscoders (Anthropic 2024) — all are dictionary-learning probes that double as steering tools.

- **Evaluation**: **SAEBench** (Karvonen et al., 2025; arXiv:2503.09532) and the chess/Othello testbed (arXiv:2408.00113) measure SAE quality on ground-truth tasks. The **"Sanity Checks for SAEs"** preprint and **AxBench** (below) both find SAEs can be matched or beaten by simple baselines, raising real questions about whether SAEs find "real" features or merely correlate by chance at scale.

### F. Evaluation / Theoretical Frameworks

- **AxBench (Wu, Arora, Geiger et al., 2025; arXiv:2501.17148, ICML 2025 spotlight).** A large-scale synthetic benchmark of concept *detection* + model *steering* across 10+ methods on Gemma-2-2B and 9B. Core finding: for steering, *prompting* > *finetuning* > all representation-based methods; SAEs are not competitive. For detection, difference-in-means is the best representation-based method. The authors introduce **ReFT-r1** (rank-1 representation finetuning) as a competitive alternative.

- **Encoding ≠ Use dissociation (Braun et al., 2025).** Argues analytically (using deep linear networks) that the manifold of solutions to a given task admits parametrizations that have *identical behavior* yet *different* hidden representations, dissociating functional from representational similarity. Empirical follow-ups (e.g., "Dissociating Decodability and Causal Use in Bracket-Sequence Transformers," arXiv:2506.22128; the Dyck-language study) confirm that probe-decodable variables can be behaviorally irrelevant.

- **Belinkov (2022), "Probing classifiers: Promises, shortcomings, and advances"** (Computational Linguistics 48(1)). Canonical survey reviewing probing's pitfalls.

### G. Newly Discovered / Beyond-the-List Methods

- **Nonlinear / MLP / bilinear probes.** Used routinely (e.g., Conneau 2018; the Othello-GPT probe of Li, Hopkins, Bau et al., 2023, arXiv:2210.13382, found nonlinear MLPs better recovered the board than linear ones). Pimentel et al. (Pareto Probing, 2020) argue MLP probes are appropriate *if* one explicitly explores the accuracy–complexity tradeoff.

- **Kernel / Gaussian-process probes.** Kernelized probes (e.g., Voita et al., minimum description length under kernel methods) sit between linear and full MLP probes.

- **Behavioral probing.** Probes that test *output behavior* under controlled minimal-pair inputs rather than activations. Examples: **Marvin & Linzen (2018, arXiv:1808.09031)** and the **BLiMP** suite (Warstadt et al., 2020) for syntactic minimal pairs; **CheckList** (Ribeiro et al., 2020) for general capability testing. These complement representational probes by directly measuring causal use without ever opening the model.

- **Geometric / intrinsic-dimensionality probes.** Ansuini, Laio, Macke, Zoccolan (2019; arXiv:1905.12784) measure local intrinsic dimension (TWO-NN) of CNN representations and find a "hunchback" profile that predicts test accuracy. Related: **manifold capacity** (Cohen, Chung, Lee, Sompolinsky, 2020), **effective rank**, **participation ratio**, and the "Effective Encoding Dimension" used to study ViT bottlenecks (arXiv:2512.07331).

- **Probing via attribution / gradients.** Integrated Gradients (Sundararajan et al., 2017), SHAP, and **LRP** (Bach et al., 2015) plus the Mamba-specific **MambaLRP** (Jafari et al., 2024) are gradient/relevance-based probes for neuron and token attribution.

- **Adversarial probing.** "Amnesic" + adversarial input crafting (Elazar et al., 2021). The "Information-Theoretic Probing with Minimum Description Length" framework folds adversarial control conditions in.

- **Encoding Probe (Gubian et al., 2025/2026; cited at arXiv:2605.00607).** Reverses the usual decoding probe: a regression that predicts the model's *own activations* from external features (lexical, syntactic, speaker) to quantify each feature's contribution to representation construction. Demonstrated on wav2vec2 variants and text models.

- **Activation Subspace Bottlenecks (Sunku Mohan, Gupta, Das, Singh, 2026; arXiv:2602.22719).** Identifies low-dimensional bottleneck subspaces in Mamba SSMs and uses them for steering (8.27% avg improvement on 6 benchmarks across 5 SSMs).

- **LATIM (Pitorro & Treviso, 2025; arXiv:2502.15612).** Token-level decomposition for Mamba-1/Mamba-2 — an SSM analogue of attention-based attribution.

- **Probes for specific property classes.** Truthfulness (CCS — Burns, Ye, Klein, Steinhardt, 2022; arXiv:2212.03827; Marks & Tegmark, 2023, arXiv:2310.06824); world models (Othello-GPT, board games arXiv:2408.00113); knowledge (ROME / Causal Tracing — Meng et al., 2022, arXiv:2202.05262); planning (Lindner et al.; Pal et al. "Future Lens," arXiv:2311.04897); language identity (Tang et al., 2024 multilingual probes); emotion, humor, sentiment (RepE, Tigges et al. 2023, arXiv:2310.15154); time and space neurons (Gurnee & Tegmark, 2024, "Language Models Represent Space and Time," arXiv:2310.02207).

---

## Task 2 — Probing Work by Architecture

### CNNs (ResNet, VGG, ConvNeXt, etc.)
- Linear-probe origins: Alain & Bengio (2016; arXiv:1610.01644). Hierarchical-feature CAVs (TCAV — Kim et al., 2018; arXiv:1711.11279). Network Dissection (Bau et al., 2017; arXiv:1704.05796) labels individual conv units. Intrinsic-dimension profiles (Ansuini et al., 2019; arXiv:1905.12784). Cross-task linear-probe grids on medical CNNs (2021). Core findings: features become increasingly abstract and increasingly linearly separable with depth; ID rises then falls; final-layer ID predicts test accuracy.

### Encoder Transformers (BERT, RoBERTa, ELECTRA, ELMo)
- The most heavily probed family. Structural probe (Hewitt & Manning 2019), edge probing (Tenney et al. 2019), "BERT rediscovers the classical NLP pipeline" (Tenney, Das, Pavlick 2019; arXiv:1905.05950), control tasks (Hewitt & Liang 2019), MDL probing (Voita & Titov 2020), AMNESIC (Elazar et al. 2021), AlterRep (Ravfogel et al. 2021). Core finding: syntax mid-network, semantics late; many decodable features are not causally used.

### Decoder-Only Transformer LMs (GPT, LLaMA, Pythia, Mistral, Gemma, Qwen)
- All "lens" methods (logit, tuned, Patchscopes). Refusal direction (Arditi et al. 2024). CAA, ActAdd, RepE. Truthfulness (CCS, ITI — Li et al. 2023, arXiv:2306.03341). Knowledge editing/causal tracing (ROME, MEMIT). Sparse probing & SAEs (Gurnee 2023, Bricken 2023, Templeton 2024). DAS / Boundless DAS. AxBench (Gemma-2). Core finding: many high-level features (truth, refusal, sentiment, language, sycophancy) are encoded near-linearly and can be steered — but the *use* of these directions is fragile under finetuning and adversarial attack.

### Vision Transformers (ViT, DeiT, MAE, DINO, DINOv2)
- Layer-wise linear probes: "From Colors to Classes" (arXiv:2503.24071) shows early-layer color/texture features, late-layer object features. "Layer by layer, module by module" (arXiv:2506.05280) finds intermediate-layer probes are often optimal under distribution shift, and that probing inside the FFN beats probing block outputs. "What do Vision Transformers Learn?" (Ghiasi et al., 2022). EL-VIT (arXiv:2401.12666) interactive probing. ID studies of DINO (arXiv:2512.07331). Core finding: ViTs encode global features earlier than CNNs but still show CNN-like hierarchical concept progression; last-block global attention acts as a learned pooling.

### State-Space Models (Mamba, Mamba-2, RWKV, S4)
- Activation Subspace Bottlenecks (Sunku Mohan et al., 2026; arXiv:2602.22719) — steering Mamba via scalar interventions on bottleneck subspaces, validated across 5 SSMs. LATIM (Pitorro & Treviso, 2025; arXiv:2502.15612) token-level interactions. MambaLRP (Jafari et al., 2024) layer-wise relevance propagation adapted to selective SSMs. Influence-Score controllability analysis (arXiv:2511.17970). Core finding: Mamba has strong recency bias and concentrates "influence" in mid-to-late layers; selective-gate channels can be pruned with little accuracy loss; classic attention-style probes do not transfer cleanly, motivating SSM-specific tools.

### Hybrid SSM-Transformer (Jamba, RecurrentGemma, Zamba, Griffin)
- This is the *least-explored* family. To date, mainly evaluation/benchmark studies rather than dedicated probing work; SAE training on Jamba/Griffin and tuned-lens equivalents are an active but underpublished area as of mid-2026. Bottleneck-subspace techniques validated on pure SSMs are beginning to be transferred.

### Multimodal (CLIP, LLaVA, BLIP, Chameleon)
- CLIP "concept probing" (Goh et al., 2021 "Multimodal Neurons in Artificial Neural Networks," OpenAI Distill) identifies multimodal neurons via dataset-example maximization and text probes. Cross-modal alignment probing (Merullo et al., 2022, arXiv:2209.15162). LLaVA-style attention probing for visual grounding (Cai et al., 2023, ViP-LLaVA, arXiv:2312.00784). Linear probes on CLIP vision encoders are the *de facto* downstream evaluation (Radford et al., 2021). Recent SAE work on CLIP/SigLIP (Daujotas, 2024; Fry, 2024) and on LLaVA shows interpretable cross-modal features. Core finding: a substantial fraction of "visual" features in vision-language models are linearly aligned with text-token directions.

### Diffusion Models (Stable Diffusion, DiT, Flux)
- "Beyond Surface Statistics" (Chen, Viégas, Wattenberg, 2023; arXiv:2306.05720) — linear probes recover depth and saliency from latent UNets, and interventions causally change generated images. "Emergence and Evolution of Interpretable Concepts in Diffusion Models" (arXiv:2504.15473, NeurIPS 2025) trains SAEs on SD activations and shows concept emergence across denoising steps. h-space / DiffEdit / Concept Sliders are concept-direction probes that double as editors. Core finding: 3D scene attributes (depth, saliency, shading) emerge in latent representations within the first few denoising steps — much earlier than human-visible structure.

### World Models (Dreamer, V-JEPA, I-JEPA, MuZero)
- Linear-probing of latent dynamics in DreamerV2/V3 (Hafner et al.) is used to test whether latent states capture reward, agent position, object presence. V-JEPA (Bardes et al., 2024) is evaluated by frozen-feature linear probes on action recognition and physics-prediction benchmarks. Othello-GPT and Chess-GPT (Li et al., 2023; Karvonen, 2024) are *micro-world* models intentionally built so that ground-truth world states (board) can be probed; linear and nonlinear probes both recover the board, with the discovery that *modeling the board causally improves play* (Nanda et al., 2023 follow-up). Core finding: latent world states in implicit world models are largely linearly accessible and causally manipulable.

### Latent Reasoning Models (Coconut, Huginn, CODI, COLAR, SIM-CoT)
- **Coconut (Hao et al., 2024; arXiv:2412.06769)**: continuous-thought reasoning explicitly designed so the "reasoning state" is a hidden vector. Probing studies decode the continuous thought via the LM head, finding superposition of multiple candidate next steps (BFS-like). **Huginn / depth-recurrent transformers** (Geiping et al., 2025, arXiv:2502.05171) — probing studies show recurrent residual updates implement iterative refinement. **CODI** (Shen et al., 2025) uses self-distillation on last-token hidden states. Core finding: latent reasoning models encode multiple alternative reasoning paths simultaneously in a low-dimensional subspace; standard logit/tuned-lens style probes are starting to be adapted to these models.

---

## Task 3 — Cross-Architecture Comparison Studies

Few papers run a *unified probing methodology* across many architectures, but several touch the question:

- **Raghu, Unterthiner, Kornblith, Zhang, Dosovitskiy (2021), "Do Vision Transformers See Like Convolutional Neural Networks?" (arXiv:2108.08810).** Uses CKA to compare ViT vs ResNet representations layer-by-layer. Finding: ViTs preserve spatial information further into the network and have more uniform layer-to-layer similarity than CNNs; ViTs aggregate global context earlier.

- **"Measuring the Measures" (arXiv:2509.04622, 2025).** 35 vision models across CNN, ViT, Swin, ConvNeXt × supervised vs self-supervised, scored by four similarity metrics. Finding: architectural family is the *primary* determinant of representational similarity; metric choice strongly affects whether one detects family-level differences. ConvNeXt and MAE cluster together despite different design philosophies — convergent computational solutions.

- **"The Triangle of Similarity" (arXiv:2601.17093).** CNN, ViT, VLM compared under pruning. Finding: architectural family forms distinct CKA clusters; pruning sometimes regularizes representations toward a shared core.

- **AxBench (arXiv:2501.17148)** is currently the closest thing to a cross-method, cross-scale, single-architecture *probing benchmark*, but is restricted to Gemma-2.

**Sub-questions:**

1. *Are there systematic differences in linear-probe accuracy across architectures?* Yes. Linear-probe accuracy rises monotonically with depth in CNNs (Alain & Bengio); shows a non-monotonic / mid-network peak in ViTs and in many language transformers (the "intermediate-layer" phenomenon — arXiv:2506.05280); is uneven and recency-biased in Mamba SSMs (arXiv:2511.17970). Diffusion models show *very early* emergence (steps 3–5 of 50).

2. *Does the "encoding ≠ use" problem exist across all architectures?* Available evidence says yes wherever it has been tested — BERT (AMNESIC), GPT/LLaMA (control-task style failures in AxBench), and the Dyck-bracket transformer case (arXiv:2606.22128). The Braun et al. (2025) deep-linear-network analysis suggests it is generic to over-parameterized networks. No paper to date has demonstrated the *absence* of the gap in any architecture.

3. *Are probe-found directions consistent across architectures?* Partially. Refusal directions found by difference-in-means transfer in their *qualitative* role across 13 chat LMs of multiple families (Arditi et al., 2024), but not numerically. Cross-architecture concept-direction consistency under stricter metrics (Park's causal inner product) has not been systematically studied. CKA-based work suggests "computational species" cluster more than individual models converge.

---

## Task 4 — Open Methodological Questions

1. **Probe complexity vs. task complexity.** What is the *right* probe family for a given hypothesis? Pareto Probing (Pimentel et al., 2020) operationalizes this as a frontier; selectivity (Hewitt & Liang) operationalizes it as a control. But no consensus exists on which to prefer when they disagree.

2. **Decodability ≠ causal use.** Behavioral and interventional follow-ups (AMNESIC, AlterRep, DAS, Braun et al. 2025) repeatedly show this gap. Open: can decodability ever be a *sufficient* condition for use, given the right control? Bracket-language work (arXiv:2606.22128) suggests not.

3. **Confounding by probe memorization and dataset statistics.** Sparse-probing accuracies on SAEs are matched by *frozen-component random baselines* (arXiv:2602.14111, "Sanity Checks for SAEs"). How do we know a probe finds *the model's* feature rather than statistical regularities of the data?

4. **Linearity assumption.** The Linear Representation Hypothesis (Park et al., 2024) is empirically successful but not universal. "Not All Features Are Linear" (Engels et al., 2024; arXiv:2405.14860) finds explicitly nonlinear concept manifolds (e.g., days-of-week) — when should we believe linear probes?

5. **Choice of layer / module / token position.** ViT work (arXiv:2506.05280) shows that probing the FFN beats probing the block output; LM work shows that the post-instruction token can dominate (the Zhao "harmfulness vs refusal" paper). No principled rule yet exists for the right "probing site."

6. **Evaluation circularity.** Steering methods are usually evaluated by judges (LLM-as-judge) that may share biases with the steered model. AxBench documents this; the gap between SAE *interpretability* and SAE *utility* (τ_b ≈ 0.30 in arXiv:2510.03659) suggests current evaluations measure different things.

7. **Causal interpretation of subspace ablations.** Concept-erasure ablations can have *off-target* effects (e.g., TruthfulQA degradation under refusal-direction orthogonalization). When is an intervention "minimal" enough to count as causal evidence?

8. **Architecture transfer of methodology.** Logit lens depends on a residual stream that is read by a fixed unembedding — does not naturally exist for CNNs, SSMs, or diffusion UNets. Tuned-lens for SSMs and diffusion is largely undeveloped.

9. **Probing the training process vs. probing the trained model.** Tracking when a feature *becomes* decodable is qualitatively different from finding it in the final model; few methods are designed for trajectory analysis.

10. **The "hydra effect."** Ablating one feature often causes another to take over (McGrath et al., 2023). This means single-direction probes can overstate localization.

---

## Task 5 — Most Promising Cross-Architecture Probing Directions

The deepest results in this field are not "model X encodes feature Y" but methodological observations that hold across families. Five high-impact directions:

1. **A universal causal probing benchmark.** Extend AxBench-style synthetic-data steering+detection evaluation to a matrix of {CNN, ViT, encoder LM, decoder LM, SSM, hybrid, diffusion, world model}. The deliverable would be a *standardized score* (analogous to selectivity) that lets cross-architecture comparisons mean something. The Dyck-language and Othello-GPT testbeds give a model of how to do this with ground-truth features.

2. **Architecture-agnostic lens construction.** Generalize the tuned-lens recipe — "learn an affine map from intermediate state to a canonical output basis" — to architectures without a clean residual stream (CNN feature maps, SSM hidden states, diffusion latents). A "universal lens" would let interpretability findings transfer rigorously across model families.

3. **Quantitative tests of the linear representation hypothesis across families.** Use Park et al.'s causal inner product across many architectures and feature classes. The high-impact question: is linearity a property of *transformers* or of *language tasks* or of *scale*? CNN and SSM evidence is sparse.

4. **Encoding–use gap as a universal phenomenon.** Combine Braun et al.'s analytical framework with empirical probing across architectures to characterize *how big* the gap is — does it scale with parameter count, with overparameterization, with task structure? A scaling law for the encoding–use gap would be a major theoretical contribution.

5. **Probing training dynamics.** Cross-architecture studies of *when* features become decodable, when they become causal, and how this depends on data and optimization. Pythia checkpoints (decoder LM), the equivalent SSM checkpoints (e.g., Mamba training trajectories), and ViT pretraining checkpoints make this newly tractable.

6. **Behavioral probes as ground truth.** Behavioral minimal-pair suites are architecture-agnostic by construction. Tying behavioral probes to representational probes — e.g., "does the probe direction explain behavioral failure?" — gives a portable validity test.

7. **SAE / dictionary probes done right.** The frozen-random-baseline result (arXiv:2602.14111) is a warning that current SAE evaluation is unreliable. A methodology that establishes SAE features as *real* features of the model — for any architecture — would unlock the rest of mechanistic interpretability. Promising leads include identifiable SAEs under linear-representation assumptions (Cui et al., 2025) and Matching-Pursuit SAEs (arXiv:2506.05239).

8. **Bridging mechanistic interpretability and probing.** Probing tells us what is *there*; circuit analysis tells us how it is *used*. DAS (Geiger et al.) is the current bridge but is expensive. Cheaper alignment-based methods that are architecture-portable would be transformative.

---

## Closing Synthesis

After a decade of probing, the field has moved from *"can we read X out of layer N?"* to *"is X used, why, and how reliably, across architectures?"* The most reliable empirical regularities — that decoders encode high-level features near-linearly; that single directions can mediate behaviors as significant as refusal; that intermediate layers often contain the most useful features — coexist with sobering negative results: SAEs can be matched by random baselines, refusal "directions" may actually be cones, MLP probes memorize, and *decodable* almost never means *causally used*. The most promising frontier is not more clever probes on more models, but the construction of *cross-architecture probing benchmarks* and *methodological theory* (Braun et al., 2025; Pimentel et al., 2020; Geiger et al., 2023) that can yield conclusions that survive beyond any particular model family. Probing as a field is maturing from a collection of techniques into a science of representational inference — and the next major contributions will come from groups that treat the methodology, not the model, as the object of study.

