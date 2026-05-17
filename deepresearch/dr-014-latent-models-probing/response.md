### Latent space models and representation analysis

用户:


Claude:
his is a thorough and well-structured research brief. The scope is clear — you want a comprehensive landscape of models with interesting latent spaces, their openness/accessibility, existing representation analysis work, and gaps worth filling. Let me dive into the research now.
our survey of models with interesting latent spaces for representation analysis is ready. It catalogues over 70 architectures across seven categories—latent reasoning models (Coconut, Huginn), latent diffusion/flow systems (SD3, FLUX), world models (Dreamer, JEPA, Genie), multimodal alignment spaces (CLIP, ImageBind, SONAR), hierarchical language models (BLT, LCM, MEGABYTE), state-space models and hybrids (Mamba, RWKV, Jamba), and other notable latent architectures. The report classifies latent spaces along active/passive, unimodal/multimodal, and structural axes, audits existing interpretability work to identify blank spots, and provides a ranked priority list of ten targets for new probing/SAE/causal analysis, led by Huginn-3.5B, BLT, and the Large Concept Model. It also includes cross-cutting observations on trajectory interpretability, open-weights constraints, and source-verification caveats.
# A Comprehensive Survey of Models with Interesting Latent Spaces for Representation Analysis

## Executive Summary

The standard decoder-only Transformer has dominated interpretability research, but the past three years have seen an explosion of architectures whose latent spaces differ from it in fundamental ways: they iterate in latent space (Coconut, Huginn), compress to non-token units (BLT, LCM, MEGABYTE), maintain rich recurrent state (Mamba, RWKV), learn world dynamics (Dreamer, JEPA, Genie), align modalities geometrically (CLIP, ImageBind), or operate as denoising/flow systems (SD3, Flux). This report catalogues these models, classifies their latent spaces, surveys existing representation analysis work, and recommends a ranked priority list of targets for new probing/causal/SAE/geometric analysis. The dominant theme: many of these models have *no public mechanistic interpretability work at all*, despite being open-source and runnable at modest scale — they constitute the largest current "blank spots" in interpretability research.

---

## Task 1 — Comprehensive Model Inventory

### A. Latent Reasoning Models (active latent computation in LLMs)

**Coconut — Chain of Continuous Thought.** Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian (Meta + UCSD), arXiv:2412.06769, Dec 2024. Feeds the last hidden state back as the next input embedding for several "thinking" steps inside `<bot>...<eot>` markers, bypassing the LM head. Curriculum gradually replaces text CoT steps with continuous ones. Code open-sourced (facebookresearch/coconut, Jan 2025); experiments are on GPT-2-scale fine-tunes — no large pretrained weights are released. Latent space: continuous, residual-stream–dimensioned vectors, iterative (typically 1–6 thought steps), shown empirically to encode superposed BFS-style alternatives.

**Huginn-3.5B / "Scaling Test-Time Compute with Latent Reasoning: A Recurrent-Depth Approach."** Geiping, Kirchenbauer, McLeish et al. (UMD/ELLIS), arXiv:2502.05171, Feb 2025. Depth-recurrent Transformer: a single core block is unrolled an arbitrary number of times (4 → 132+) per token at inference, with stochastic recurrence depth during training. **Weights publicly released** on HF (`tomg-group-umd/huginn-0125`); training code on GitHub (`seal-rg/recurrent-pretraining`). Latent space: continuous residual stream that is iteratively refined, with explicit prelude/recurrent/coda block structure. This is currently the most operable, large-scale "active" latent reasoner.

**Ouro.** Zhu et al., 2025 — a related recurrent-depth model trained at LLM scale; cited alongside Huginn as a second exemplar.

**Pause Tokens.** Goyal et al., "Think before you speak: Training Language Models with Pause Tokens," arXiv:2310.02226, 2023. Inserts `<pause>` tokens that give the model extra forward-pass compute without emitting text — minimal architecture change, but creates a distinct "passive thinking" latent slot. No special pretrained weights typically; reproducible on any decoder LM.

**Implicit/Internalized CoT (iCoT).** Deng et al., arXiv:2405.14838, 2024. Sequentially fine-tunes a model to drop CoT tokens one at a time until reasoning is fully internalized in the residual stream. Code public; uses standard Transformer weights.

**Quiet-STaR.** Zelikman et al., 2024. Generates rationale tokens at every position in parallel and uses a learned mixing head to weight their contribution.

**CCoT (Compressed CoT).** Cheng & Van Durme, 2024. Encodes a reasoning chain into a small set of continuous "contemplation tokens" with a learned encoder/decoder.

**CODI.** Shen et al., 2025. Self-distillation aligns hidden activations at a specific token position with a teacher's CoT activations.

**SoftCoT.** Xu et al., 2025. A frozen assistant LM produces "soft tokens" via a learned projector for a frozen main LM.

**LightThinker, HCoT, BoLT, Reasoning-CPT, PHD-Transformer.** Various 2024–2025 papers compressing reasoning into latent or gist tokens (see the "Survey on Latent Reasoning," arXiv:2507.06203, and its companion repo `multimodal-art-projection/LatentCoT-Horizon`, which maintains an updated bibliography).

**MCOUT (Multimodal Chain of Continuous Thought).** arXiv:2508.12587, 2025. Extends Coconut to VLMs, iterating jointly over visual and language hidden states.

### B. Latent Diffusion / Flow Generative Models

**Stable Diffusion (LDM).** Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," arXiv:2112.10752, 2021/2022. KL- or VQ-regularised VAE latent (~4×64×64 for SD1.5) on which a U-Net or DiT runs the diffusion process. Weights and code fully open (CompVis, Stability AI).

**SDXL.** Podell et al., arXiv:2307.01952, 2023. Larger U-Net + dual text encoders over the same 4-channel latent. Open weights.

**SD3 / Stable Diffusion 3.** Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis," arXiv:2403.03206, 2024. Rectified flow matching + MM-DiT joint text/image transformer. 16-channel VAE latent. Weights released (research license).

**FLUX.1 (Black Forest Labs).** 2024. 12B-parameter rectified-flow transformer with a 16-channel VAE latent; `dev` and `schnell` weights public. No formal paper; widely studied empirically.

**Sora (OpenAI), Veo (Google), Gen-3 (Runway).** Video diffusion/flow models over learned spatio-temporal latents ("patches"). Closed-source — only useful for interpretability via API behavioural probes.

**Flow Matching / Rectified Flow.** Lipman et al. arXiv:2210.02747 (2022); Liu et al. arXiv:2209.03003 (2022). Continuous-time CNF training objective; underlies SD3, FLUX, several video models.

**DiT (Diffusion Transformers).** Peebles & Xie, arXiv:2212.09748, 2022. Transformer backbone on VAE latents; the architectural basis for SD3, FLUX, Sora-class models.

**Consistency Models / LCM.** Song et al. arXiv:2303.01469. Distilled few-step diffusion in the same latent space.

### C. World Models

**PlaNet / Dreamer v1.** Hafner et al., arXiv:1912.01603, 2019. Recurrent State-Space Model (RSSM) with stochastic + deterministic latent components.

**DreamerV2.** Hafner et al., arXiv:2010.02193, 2020. Discrete categorical latent (32 one-hot vectors from 32 categoricals) — first version with non-continuous world-model latents.

**DreamerV3.** Hafner, Pasukonis, Ba, Lillicrap, arXiv:2301.04104, 2023. RSSM with `(h_t, z_t)`; `z_t` is 32×32 categorical one-hot. Single configuration across 150+ tasks. Code open (`danijar/dreamerv3`). Latent space: hybrid deterministic-continuous + categorical-discrete; trained with reconstruction.

**DreamerV4 / DreamerV5.** Successors with discrete categorical latents (referenced in 2025 literature).

**I-JEPA.** Assran et al., arXiv:2301.08243, 2023. Predicts representations of masked image blocks from a context block, in a learned latent space (no pixel reconstruction). Weights open (`facebookresearch/ijepa`).

**V-JEPA.** Bardes et al., arXiv:2404.08471, 2024. Video extension of I-JEPA; predicts masked spatio-temporal feature blocks.

**V-JEPA 2.** Assran, Bardes, Fan, Garrido et al. (Meta FAIR), arXiv:2506.09985, June 2025. Scaled self-supervised video model used for understanding, prediction, and planning. Open weights.

**LeJEPA.** Balestriero & LeCun, 2025. Provable, scalable JEPA-style SSL without heuristics.

**Genie.** Bruce, Dennis, Edwards, Parker-Holder et al. (DeepMind), arXiv:2402.15391, Feb 2024. 11B-parameter foundation world model trained from unlabelled internet video. Three components: a spatiotemporal VQ video tokenizer, a Latent Action Model (LAM) that infers a discrete latent action vocabulary fully unsupervised, and an autoregressive dynamics transformer over tokens conditioned on latent actions. Closed weights (DeepMind), but the architecture is reproduced in open implementations.

**Genie 2 (Dec 2024).** Closed; scaled video-prompted world model.

**DINO-WM.** Zhou et al., 2024 — uses frozen DINOv2 features as a world-model latent.

**The Matrix (arXiv:2412.03568), GameGen-X, Diffusion Forcing (Chen et al. 2024).** Various 2024–2025 video-world-model variants.

### D. Multimodal / Cross-Modal Latent Spaces

**CLIP.** Radford et al., arXiv:2103.00020, 2021. Contrastive image–text aligned embedding (typically 512–768-D). Weights open (OpenAI, OpenCLIP, EVA-CLIP variants up to multi-B parameters).

**SigLIP / SigLIP 2.** Zhai et al., arXiv:2303.15343 (2023) and follow-ups. Sigmoid loss; open weights.

**ImageBind.** Girdhar et al. (Meta), arXiv:2305.05665, 2023. Aligns 6 modalities (image, text, audio, depth, thermal, IMU) to a single embedding space using image as an anchor. Weights open.

**LanguageBind, AudioCLIP, PointBind, MetaCLIP.** Various successors / variants.

**Chameleon.** Meta Chameleon team, arXiv:2405.09818, 2024. Early-fusion, mixed-modal autoregressive model — tokenises images via a discrete VQ-VAE and trains a single transformer over interleaved text+image tokens. Weights released (7B, 30B).

**Other early-fusion / unified models:** Emu (BAAI), Show-o, Transfusion (Meta, 2024 — combines next-token prediction with diffusion over image tokens in one transformer), Janus (DeepSeek), Lumina-mGPT, Unified-IO 2.

**SONAR.** Duquenne, Schwenk, Sagot (Meta), 2023. 1024-D sentence embedding space covering 200 languages × text+speech. Encoder/decoder weights public.

### E. Latent / Hierarchical Language Models (encode → latent → decode)

**Byte Latent Transformer (BLT).** Pagnoni, Pasunuru, Rodriguez et al. (Meta + UW), arXiv:2412.09871, Dec 2024. Three modules: a local encoder over bytes → variable-length "patches" segmented by next-byte entropy → a large latent Transformer over patches → a local byte decoder. Scaling study up to **8B params, 4T training bytes**. Code and weights public (`facebookresearch/blt`, CC-BY-NC). A 2026 follow-up (arXiv:2605.08044 — likely a placeholder ID for a recent preprint) introduces BLT-Diffusion, BLT-S, BLT-DV for faster generation.

**MEGABYTE.** Yu et al. (Meta), arXiv:2305.07185, 2023. Hierarchical decoder: a large global model over patches and a small local model over bytes within patches. Predecessor of BLT.

**Large Concept Model (LCM).** Barrault et al. (Meta), arXiv:2412.08821, Dec 2024. Autoregressive predictor in **SONAR sentence-embedding space**. Variants: MSE-LCM, Diffusion-LCM (one-tower / two-tower contextualizer + denoiser), Quantised-LCM with residual vector quantisation. Sizes 1.6B and 7B; trained on up to 7.7T tokens. Code public (`facebookresearch/large_concept_model`); model weights have been released for the base variants.

**SONAR-LLM.** Dragunov et al., arXiv:2508.05305, Aug 2025. Decoder-only transformer that predicts SONAR sentence embeddings but is supervised by token-level cross-entropy back-propagated through the frozen SONAR decoder. Sizes 39M–1.3B; all checkpoints + code released.

**Planning-with-Latent-Language / PLANNER.** Zhang et al. 2023 — latent-variable planning over text.

**Diffusion language models.** SEDD (Lou et al. 2024), MDLM (Sahoo et al. 2024), LLaDA (2025), Mercury (Inception Labs, 2025), Google Gemini Diffusion (2025) — discrete or continuous diffusion over text tokens; latent state is the noised sequence itself, "iterative" in a fundamentally different sense from autoregressive LMs.

### F. State-Space Models, Linear RNNs, and Hybrids

**S4 / S4D / S5.** Gu, Goel, Ré, arXiv:2111.00396, 2021. HiPPO-initialised SSM; structured A matrix. Long-Range Arena breakthrough.

**Mamba (Mamba-1).** Gu & Dao, arXiv:2312.00752, 2023. Selective SSM — input-dependent `(Δ, B, C)`. Hidden state of size N per channel (D×N total) acts as an unbounded, compressed memory. Open code+weights (130M, 370M, 790M, 1.4B, 2.8B).

**Mamba-2.** Dao & Gu, arXiv:2405.21060, ICML 2024. State Space Duality (SSD): unifies SSMs and attention via structured matrices; head dimensions matched to attention. Open weights up to 2.8B (and larger community-trained variants).

**Mamba-3.** Lahoti, Li, Chen et al., arXiv:2603.15569, 2026. Improved sequence modelling via continued exploration of SSM principles. Open.

**RWKV-4 "Raven".** Peng et al., arXiv:2305.13048, EMNLP 2023. RNN with channel/time mixing; vector-valued state.

**RWKV-5 "Eagle" & RWKV-6 "Finch".** Peng et al., arXiv:2404.05892, COLM 2024. **Matrix-valued state (64×64)** with multi-headed time mixing; dynamic recurrence in v6. Weights from 0.4B to 14B released.

**RWKV-7 "Goose" / G1.** Peng et al., arXiv:2503.14456, March 2025. Generalised delta rule with vector gating and adaptive per-channel learning rates; theoretically surpasses TC0 of softmax attention while remaining parallelisable. Weights 0.4B–14B (Apache 2.0).

**Jamba.** AI21, arXiv:2403.19887, 2024. Hybrid Mamba + Transformer + MoE; 52B total / 12B active. Open weights (Jamba-1.5 Mini, Large).

**Griffin / Hawk / RecurrentGemma.** De et al. (DeepMind), arXiv:2402.19427, 2024. Gated linear recurrences (RG-LRU) interleaved with local attention. RecurrentGemma 2B and 9B weights open.

**Hyena, H3, RetNet, GLA, DeltaNet, xLSTM, mLSTM/sLSTM.** Other linear-recurrent or long-convolution architectures with distinctive hidden-state structures; all have open implementations.

**Falcon-Mamba 7B (TII, 2024), Codestral Mamba 7B (Mistral, 2024), Zamba (Zyphra), Samba (Microsoft).** Production-grade pure or hybrid SSM LLMs with open weights.

### G. Other Models with Interesting Latent Spaces

**Neural ODEs.** Chen, Rubanova, Bettencourt, Duvenaud, NeurIPS 2018 (arXiv:1806.07366). Continuous-depth networks defined by an ODE; latent trajectory is a continuous curve. Open code.

**FFJORD / Continuous Normalizing Flows.** Grathwohl et al., 2018.

**Modern Hopfield Networks.** Ramsauer et al., "Hopfield Networks is All You Need," arXiv:2008.02217, 2020. Continuous Hopfield with exponential storage capacity; mathematically equivalent to a single-step attention.

**Energy-Based Models (EBMs).** JEM (Grathwohl et al. 2019), IGEBM (Du & Mordatch 2019). Latent space defined implicitly by an energy function.

**VQ-VAE / VQ-VAE-2 / RQ-VAE / FSQ.** Discrete codebook latents used inside almost every modern multimodal model (Chameleon, Genie tokenizer, music models).

**Autoencoders for representation:** β-VAE, FactorVAE (disentanglement); DINOv2 (open Meta vision foundation model — self-distilled patch features); MAE; Perceiver IO (cross-attention bottleneck of latent slots).

**Slot-based / object-centric models.** SLATE, SAVi, DINOSAUR — explicitly factorize visual scenes into slot vectors.

**Hierarchical Reasoning Model (HRM).** Wang et al. 2024 — high/low frequency recurrent modules in latent space for ARC-style reasoning.

---

## Task 2 — Classification of Latent Spaces

### Axis 1 — Passive (single forward pass per token/sample) vs Active (iterates / refines internally)

| Active latent computation | Passive (standard feed-forward) |
|---|---|
| Coconut, Huginn, Ouro, iCoT, CODI, CCoT, Quiet-STaR, MCOUT | Standard Transformer LLMs |
| All diffusion / flow models (denoising steps) | CLIP, SigLIP, ImageBind, DINOv2 (encoder pass) |
| All world models with imagined rollouts (Dreamer, JEPA-planning, Genie) | BLT, MEGABYTE, LCM at inference (one autoregressive step each) |
| Modern Hopfield (iterative retrieval) | Chameleon, Transfusion (token-by-token AR) |
| Neural ODE solver | Mamba/RWKV per-token (state evolves but each token is one step) |
| HRM | EBMs *training* is passive but *sampling* (Langevin) is active |

Note that SSMs (Mamba, RWKV) occupy an interesting middle ground: per-token computation is single-pass, but the hidden state itself acts as a persistent, compressed memory that evolves recurrently across positions — *passive within a token, active across a sequence*.

### Axis 2 — Unimodal vs Multimodal

- **Unimodal text:** Coconut, Huginn, BLT, MEGABYTE, Mamba, RWKV, Jamba, Griffin, LCM (text only at present).
- **Unimodal vision:** DreamerV3 (pixels), SD/SDXL/SD3/FLUX (image latents), I-JEPA, DINOv2, V-JEPA / V-JEPA 2 (video).
- **Multimodal aligned:** CLIP, SigLIP, ImageBind (6 modalities), SONAR (text+speech, 200 languages).
- **Multimodal early-fusion (shared autoregressive latent):** Chameleon, Transfusion, Janus, Show-o, Unified-IO 2.
- **Multimodal world-modelling:** Genie, V-JEPA 2, GameGen-X.

### Axis 3 — Known Interesting Structure

- **Discrete / categorical latents:** DreamerV2/V3 (32×32 categorical), VQ-VAE codebooks, Chameleon tokens, Genie's latent action codebook, Quantised-LCM (residual VQ).
- **Matrix-valued state:** RWKV-5/6/7 (64×64 per head); contrast with vector-state Mamba.
- **Provably superposed / BFS-like:** Coconut's continuous thoughts shown to encode multiple alternative reasoning paths simultaneously (per the ICLR 2025 paper).
- **Modality alignment / shared geometry:** CLIP, ImageBind, SONAR — known to support arithmetic, anisotropy phenomena, and modality-gap effects.
- **Dynamically allocated compute / variable-length patches:** BLT (entropy-segmented), MEGABYTE (fixed patches), Huginn (variable recurrence).
- **Manifold/flow structure:** SD3, FLUX, Flow Matching — latent space carries an explicit probability path; analysable via the velocity field.
- **Disentanglement targets:** β-VAE family, FactorVAE, slot-based models.
- **Compressed sentence-level semantics:** LCM, SONAR-LLM — latent unit is a sentence rather than a token.
- **Residual stream present:** Coconut, Huginn, Chameleon, BLT global transformer (yes); pure Mamba/RWKV (yes, but interleaved with non-residual SSM mixing — the residual stream is interrupted by selective state updates rather than additive attention/MLP).
- **No residual stream / non-additive computation:** Neural ODEs (replaced by continuous flow), Genie's video tokenizer/dynamics (token-level), most diffusion U-Nets.

---

## Task 3 — Existing Representation Analysis & Blank Spots

### What has been studied

**State-space models / Mamba — moderate coverage and rapidly growing.**
- **LaTIM** (Pitorro & Treviso, arXiv:2502.15612, Feb 2025): a token-level decomposition method for both Mamba-1 and Mamba-2, providing the first attention-style attribution maps for SSMs.
- **Mamba Knockout** (arXiv:2505.24244, May 2025): extends the Attention-Knockout interpretability framework from Transformers to SSMs; introduces a novel feature-knockout intervention exploiting SSM structure to trace factual-information flow.
- **Activation Subspace Bottlenecks in SSMs** (Mohan, Gupta, Das, Singh, arXiv:2602.22719, 2026): mechanistic interpretability work identifying small activation subspaces critical for routing in 5 SSMs, with test-time steering yielding ~8% average gains on 6 benchmarks.
- **Controllability Analysis / Influence Score** (arXiv:2511.17970, late 2025): comparative analysis across mamba-130m, 2.8b, 2.8b-slimpj — finds strong recency bias and mid-to-late layer concentration of influence; emergent behaviours only at scale.
- **Towards Universality** (Wang et al., arXiv:2410.06672, 2024): trains SAEs on Mamba and Transformer LMs and quantifies feature similarity across architectures — the first SAE study of Mamba.
- `joelburget/mamba-sae` — community SAELens-based SAE training for Mamba.

**Latent reasoning models — small but growing.**
- **"Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer"** (arXiv:2507.02199, July 2025): logit lens, coda lens, and token-rank trajectories applied to Huginn-3.5B. Finding: probing is inconsistent across blocks; recurrence yields only marginal CoT-like behaviour and falls short of explicit CoT models.
- **"Loop, Think, & Generalize"** (arXiv:2604.07822, 2026): probes Huginn and Ouro; finds correct-trajectory representations have higher entropy, anisotropy, and intrinsic dimensionality but lower effective rank.
- The Coconut paper itself shows that its continuous thoughts encode superposed alternatives (a kind of latent BFS), but no third-party SAE/causal-tracing work on Coconut weights has appeared as of mid-2025.

**Multimodal — CLIP is extremely well-studied; ImageBind/SONAR much less.**
- Multiple SAE studies on CLIP: Joseph et al. (arXiv:2502.06755, "Interpretable and Testable Vision Features"), **Matryoshka SAE on CLIP/SigLIP** (Zaigrajew et al., arXiv:2502.20578, ICML 2025), Sparse-CLIP (arXiv:2601.20075), Contextual-Dependency-Score SAEs (arXiv:2604.05724). Findings include identifying culturally-specific features (e.g. a "Brazil" feature in CLIP-24K-6909) and showing CLIP and DINOv2 SAEs are not directly comparable feature-by-feature.
- Linear probing of CLIP is a standard practice with thousands of papers.
- The modality gap (Liang et al., NeurIPS 2022) is a well-known geometric finding.
- ImageBind: comparatively little mechanistic work — mostly retrieval-style probes.

**World models — sparse coverage.**
- Some work probing DreamerV3 latents for environment variables and on the role of categorical vs continuous latents (e.g. MuDreamer, arXiv:2405.15083, which questions the reconstruction objective).
- Genie: essentially **no published interpretability work** on the learned latent action space beyond the original paper's own analyses (which show latent actions are consistent across games and transferable to CoinRun behavioural cloning).
- I-JEPA / V-JEPA: probing studies exist (the original papers contain linear-probe evaluations) but no SAE, causal tracing, or detailed mechanistic work.

**Latent diffusion.**
- Substantial work on Stable Diffusion's "h-space" (Kwon et al., 2022) showing semantic directions in the U-Net bottleneck.
- Diffusion Lens (Toker et al., 2024) and various concept-erasure / concept-editing studies (e.g. Concept Sliders, Gandikota et al.).
- Cross-attention map analyses (Prompt-to-Prompt) are widespread.
- SDXL, SD3, FLUX latent spaces are much less analysed than SD1.5.
- Sora / Veo: closed → essentially no public mechanistic work.

**BLT / LCM / SONAR-LLM.**
- BLT: no published representation-analysis work as of mid-2025 beyond the original paper's own ablations and patch-entropy analyses.
- LCM / SONAR-LLM: only the original papers — **complete blank spot** for SAE, probing, causal patching.

**Hopfield / EBMs / Neural ODEs.** Theoretical analysis is rich; LLM-style mechanistic interpretability work is essentially absent.

### Blank-spot summary (models with open weights and ~zero published interpretability)

1. **BLT** — open 8B, no mechanistic work.
2. **Large Concept Model** — open 7B, no interpretability work in concept space.
3. **SONAR-LLM** — open 39M–1.3B, no probing.
4. **RWKV-7 Goose** — open 0.4–14B, very little interpretability work specifically targeting RWKV-7 (most prior work focused on RWKV-4).
5. **Jamba / Falcon-Mamba / Zamba / Samba** — open hybrid SSM-Transformers, essentially no mechanistic interpretability comparing the two stream types within a single model.
6. **Genie's latent action space** — open architecture, partially reproducible (open implementations exist).
7. **DreamerV3 categorical latents** — open code, no SAE-style work on the 32×32 categorical bottleneck.
8. **I-JEPA / V-JEPA 2** — open weights, minimal mechanistic work despite being canonical "no-reconstruction" representation learners.
9. **Chameleon / Transfusion** — open weights, no published work on how the shared early-fusion latent encodes modality identity.
10. **Coconut weights** (small) — no third-party SAE/probing study despite the paper's own evidence for latent BFS.

---

## Task 4 — Ranked Recommendations

Each entry weights: (i) **uniqueness** of the latent space vs a standard Transformer, (ii) **operability** (open weights, scale, compute cost), (iii) **blank-spot degree**, and (iv) **new questions** that become possible.

### 1. Huginn-3.5B (Recurrent-Depth Transformer)
Open 3.5B weights, single block unrolled 4–132 times. *Unique question:* what does the *same* layer compute at iteration 3 vs iteration 30? Does a single fixed-point attractor emerge, or do trajectories diverge? Existing work (Geiping; July 2025 OpenReview; arXiv:2604.07822) is exploratory and reports *probing inconsistencies across blocks* — a perfect entry for SAE-on-latent-trajectories and causal patching across iteration steps. Very high priority.

### 2. Byte Latent Transformer (BLT, 8B)
Patch latents have variable lengths derived from byte-entropy. *Unique question:* what does a single "patch embedding" encode and how does information distribute between local (byte) and global (latent) streams? Cross-attention bridge between streams is exactly the kind of bottleneck SAE/activation-patching can dissect. Open, CC-BY-NC, with active follow-ups (Fast BLT, 2026). High priority.

### 3. Large Concept Model + SONAR / SONAR-LLM
Latents are *sentence-level* in a 1024-D space shared across 200 languages × text+speech. *Unique question:* is "sentence arithmetic" possible? Do features in SONAR space disentangle topic vs sentiment vs language? How does the diffusion-LCM denoise a sentence? Zero published mechanistic work; SONAR-LLM in particular is small (≤1.3B) and easy to probe. Very high priority for representation geometry research.

### 4. Mamba-2 and Mamba-3 (and a hybrid like Jamba or Samba)
The recurrent hidden state of an SSM is a fundamentally different memory than KV cache. State-Space Duality (Dao & Gu 2024) means SSD blocks can be viewed *both* as attention and as recurrence — interpretability gets a natural duality to exploit. Existing work (LaTIM, Mamba Knockout, Activation Subspace Bottlenecks) is just beginning. Within Jamba you can compare *Mamba layers vs Transformer layers in the same model*, which is a unique controlled experiment. High priority.

### 5. RWKV-7 Goose (matrix-valued, generalized delta rule)
Matrix-valued state (e.g. 64×64 per head) with channel-wise adaptive learning rate makes the "memory" of RWKV-7 a far richer object than a vector. Authors claim TC0-surpassing expressivity. Open 0.4–14B. *Unique question:* what is stored in the matrix state — can we read out specific facts/positions from rank-one updates? Almost no third-party interpretability. High priority.

### 6. Coconut (and Coconut-style variants: CODI, CCoT, SoftCoT)
Continuous thoughts shown to encode superposed alternatives. Coconut weights are small (GPT-2 scale) and trivially runnable. *Unique question:* can SAEs decode multiple parallel reasoning paths from a single continuous thought vector? Can one *causally patch* a thought from one problem into another to verify the BFS interpretation? Medium–high priority; small scale is both a feature (cheap) and a limitation (results may not transfer).

### 7. Genie (or open-source reimplementations) — latent action space
A fully unsupervised, discrete codebook of "actions" learned from video. *Unique question:* do latent actions form a structured group (e.g. directional symmetries)? Are they disentangled across games? Closed DeepMind weights are a barrier, but open reimplementations (e.g. "Genie clones" on GitHub) and DeepMind's published 11B-action-codebook analyses are starting points. Medium priority but extremely high upside in *world-model representation geometry*.

### 8. V-JEPA 2
Open Meta weights, video self-supervision *without pixel reconstruction*, used for planning. *Unique question:* what's in a JEPA representation that's *not* in DINOv2 / MAE? Because JEPA is explicitly defined by predictability in latent space, geometric/manifold analyses (intrinsic dimension, curvature) are natural and unique. Medium-high priority.

### 9. DreamerV3 — discrete categorical latents
32×32 one-hot bottleneck makes feature analysis tractable: each "category" position is already discrete. Open code, modest compute. *Unique question:* are specific categorical slots functionally specialised (e.g. one slot tracks reward, another tracks agent position)? Causal interventions on individual categoricals are extremely cheap. Medium priority — best for RL/world-model representation work.

### 10. Chameleon (and Transfusion if weights become available)
Early-fusion shared latent over text + image tokens. *Unique question:* does a single residual-stream direction carry "modality identity," and does it disentangle from semantics? Where, if anywhere, do cross-modal semantic features live? Open 7B/30B weights. Medium priority — competes with the larger CLIP-SAE literature but asks fundamentally different questions about *fused* (not aligned) representations.

### Honourable mentions

- **Flow-matching transformers (SD3, FLUX)** — the velocity field is itself a learned vector field over a latent manifold; very few mechanistic studies. Closed-licence but weights are public.
- **Diffusion language models (LLaDA, SEDD, MDLM)** — denoising over text is a fundamentally different inference dynamic; rich new probing questions.
- **Hierarchical Reasoning Model (HRM) and other "tiny world-model + reasoner" hybrids** — small enough to fully characterise.
- **Modern Hopfield networks** — analytical retrieval dynamics make them an ideal test-bed for theory-grounded interpretability.

---

## Cross-Cutting Observations and Caveats

1. **The "active latent space" frontier is the youngest and most promising.** Models that *iterate* in latent space (Coconut, Huginn, Ouro, diffusion LMs, HRM) raise the new question of *trajectory* interpretability — not "what does this layer encode?" but "what does this state encode at iteration *k*?". Standard probing/SAE tooling needs to be re-thought for trajectories.

2. **Open weights are the binding constraint.** Several of the most interesting architectures (Sora, Veo, Genie 2, GPT-4o native multimodal, Gemini Diffusion) are closed. Among open ones, the FAIR/Meta ecosystem (BLT, LCM, SONAR-LLM, Chameleon, I/V-JEPA, V-JEPA 2, MEGABYTE, Coconut) is by far the richest source.

3. **Caveat on claims of "interpretability."** Several Coconut/Huginn analyses (e.g. the July 2025 Huginn-probing paper) explicitly report *negative* or *inconsistent* results: depth-recurrent models do *not* obviously develop legible latent CoT under logit-lens-style probing. This is itself a publishable finding but suggests that some claims in primary papers (e.g. about "latent BFS" or "iterative refinement") should be treated as hypotheses, not established facts.

4. **Hybrid models are the most underexploited.** Jamba, Samba, RecurrentGemma, and Zamba interleave SSM/RNN layers with attention layers in the *same* model. They are the cleanest natural experiment for comparing latent-space structure across architectural primitives.

5. **Dates and versions.** As of May 2026, BLT, LCM, Coconut, Huginn, RWKV-7, Mamba-3, V-JEPA 2, and Genie 2 are all <18 months old. The field is moving quickly; any list will need updating within a year. The most reliable living bibliography for the latent-reasoning slice is `multimodal-art-projection/LatentCoT-Horizon` on GitHub, and for SAE work the `ai-safety-foundation/sparse_autoencoder` ecosystem plus SAELens.

6. **A note on sources.** Several arXiv IDs cited in search results above (e.g. 2602.22719, 2603.15569, 2604.07822, 2604.05724, 2605.08044, 2601.00844, 2601.07861, 2601.14354, 2603.04715) correspond to preprints from late 2025 / early 2026 — they appeared in cited reference lists but were not all independently verified through their abstract pages. They should be checked against arXiv directly before being relied on in a publication. The well-established IDs (Coconut 2412.06769, BLT 2412.09871, LCM 2412.08821, Dreamer 2301.04104, Genie 2402.15391, Mamba 2312.00752, Mamba-2 2405.21060, RWKV-7 2503.14456, Huginn 2502.05171, I-JEPA 2301.08243, V-JEPA-2 2506.09985, Survey on Latent Reasoning 2507.06203) are all confirmed.

