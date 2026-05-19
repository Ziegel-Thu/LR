### VLA representation analysis research agenda

用户:


Claude:
his is a thorough and well-structured research agenda — the questions are specific enough that I can dive right in. Let me kick off a deep research process covering all six sections: existing probing/SAE work on VLAs, modality gap geometry, action token effects on representations, memorized trajectories, representation collapse diagnostics, and promising experimental directions.
our research map on the internal representation structure of Vision-Language-Action models (2024–2026) is ready. It catalogs the emerging VLA interpretability subfield through key papers from Stanford, Berkeley, CWRU, MIPT, and Physical Intelligence, covering SAE analyses of π0.5 and OpenVLA, FFN steering, emergent world models, and catastrophic forgetting diagnostics. The report identifies three convergent empirical findings (representation collapse during fine-tuning, visual pathway dominance, and trajectory memorization in SAE features), highlights the unaddressed "modality gap" question inside VLA residual streams, and proposes seven ranked experiments with a four-stage research roadmap. It includes quantitative anchors, a diagnostics toolbox, unresolved controversies, and caveats about recent unreviewed preprints.
# The Internal Representation Structure of Vision-Language-Action Models: A 2024–2026 Research Map

## TL;DR

- **VLA interpretability is now a real subfield (2024–2026), small but rapidly growing**, anchored by Swann et al.'s SAE study of π0.5 (Dr.VLA, arXiv:2603.19183), Grant et al.'s six-model "Not All Features Are Created Equal" (arXiv:2603.19233), Häon et al.'s FFN steering at CoRL 2025 (arXiv:2509.00328), Kachaev et al.'s "Don't Blind Your VLA" (arXiv:2510.25616), Molinari et al.'s emergent world model in OpenVLA (arXiv:2509.24559), Lu et al.'s symbolic-state probes (arXiv:2502.04558), and the earlier ReVLA catastrophic-forgetting diagnosis (arXiv:2409.15250); these papers converge on three robust empirical claims — (i) VLA fine-tuning produces representation collapse and catastrophic forgetting in the VL backbone, (ii) the visual pathway dominates action generation while language tokens are often vestigial, and (iii) most SAE features in fine-tuned VLAs encode memorized trajectories rather than reusable motion primitives.
- **The "modality gap" question (CLIP-style cone effect inside a VLA residual stream) is essentially unaddressed**: no paper has computed the literal cross-modal cosine-distance metric of Liang, Zhang, Kwon, Yeung, Zou "Mind the Gap" (NeurIPS 2022, arXiv:2203.02053) for vision / language / proprioception / action tokens inside an OpenVLA, π0 or GR00T residual stream, although three indirect lines of evidence (subspace injection in Grant et al., t-SNE token collapse in Kachaev et al., separate-dim SAEs in Swann et al.) all point to highly separable modality subspaces with cross-modal fusion concentrated in middle LM layers (≈ layers 14–24 of the Llama-2 / PaliGemma backbones).
- **The single most impactful next experiment is a layer-wise modality-gap measurement combined with cross-VLA SAE feature transfer**: rerun Liang et al.'s cosine-distance protocol across all transformer layers of OpenVLA, π0, π0.5, GR00T N1.5 and SmolVLA, label each SAE feature by its memorization-vs-primitive score à la Swann et al., and test whether "general" features occupy the cross-modal region while "memorized" features cluster inside modality-specific cones. This would unify the five threads above into a single causal account of when and why VLAs generalize.

## Key Findings

### 1. The published VLA interpretability papers (exhaustive list, 2024–2026)

| Paper | Authors / Year / Venue | Method | Core Finding |
|---|---|---|---|
| **Dr.VLA — SAE on π0.5 / OpenVLA** (arXiv:2603.19183) | Aiden Swann, Lachlain McGranahan, Hugo Buurmeijer, Monroe Kennedy III, Mac Schwager (Stanford), 2026 | TopK Sparse Autoencoders on hidden activations of *both* the PaliGemma backbone and the action expert in π0.5; per-timestep and per-token; four-metric logistic classifier (onset count, episode coverage, mean activation, relative run length) | 98.43 % of active SAE features on π0.5+LIBERO and 99.55 % on OpenVLA+LIBERO Goal are episode-specific *memorized trajectories*; only 32 / 2 044 features in PaliGemma layer 5 are general motion primitives. Training on larger / more diverse data (DROID, 1 545 tasks) raises the general-feature share to ~5–11 %. Knowledge insulation (π0.5-KI) also amplifies general features. Steering individual general features causally shifts robot behavior on LIBERO. |
| **Not All Features Are Created Equal** (arXiv:2603.19233; Action Atlas https://action-atlas.com) | Bryce Grant, Xijia Zhao, Peng Wang (Case Western Reserve), 2026 | Activation injection + per-token SAEs + linear probes across 6 VLAs (π0.5, OpenVLA-OFT, X-VLA, SmolVLA, GR00T N1.5, ACT), 394 000+ rollouts, 4 benchmarks | (a) Visual pathway dominates: in π0.5, null-prompt rollouts retain cosine 0.775 to baseline, and injecting layer-0 activations alone restores 0.997. (b) Cross-task injection drives 99.8 % of X-VLA episodes to *source-task* coordinates — VLAs encode spatially bound motor programs, not abstract task vectors. (c) Mean-pooled SAEs catastrophically destroy task success (96 %→8 %) while per-token SAEs preserve it (96 %→94 %), proving modality-specific tokens occupy non-fungible subspaces. (d) Subspace injection in π0.5 / SmolVLA / GR00T confirms VLM-pathway (goal semantics) and expert-pathway (motor) activations live in *separable subspaces*, with 2× greater displacement from expert injection. Appendix E.3 reports a CKA layer-independence analysis. |
| **Mechanistic Interpretability for Steering VLAs** (arXiv:2509.00328) | Bear Häon, Kaylene Stocking, Ian Chuang, Claire Tomlin (UC Berkeley EECS), CoRL 2025 (Seoul) | Project FFN value-vectors of each transformer block to the vocabulary, cluster by semantic alignment, then activation-inject into π0 (UR5) and OpenVLA (LIBERO) | First VLA mechanistic-steering framework. The flagship finding, verbatim: *"less than 25 % of FFN neurons being rewired for action prediction and a large fraction of the remainder representing clear, semantically interpretable concepts."* Steering FFN directions for "up", "slow", "careful" modulates real-robot behavior zero-shot. Demonstrates VLAs inherit compositional semantics from their VLM. |
| **Don't Blind Your VLA** (arXiv:2510.25616) | Nikita Kachaev, Mikhail Kolosov, Daniil Zelezetsky, Alexey K. Kovalev, Aleksandr I. Panov (Cognitive AI Lab / MIPT), Oct 2025 | t-SNE of class-token embeddings, attention-map analysis, linear probes (ImageNet-100), VL-Think diagnostic suite, alignment loss to a frozen vision teacher | Documents three failure modes in OpenVLA vs PrismaticVLM / Qwen2.5-VL: (1) **attention sink** — attention maps become diffuse, noisy, decorrelated from the target object; (2) **representation collapse** — patch embeddings for cup / bottle / knife collapse into overlapping t-SNE clusters; (3) **domain forgetting** — symbol / arrow / icon understanding degrades. Linear-probe ImageNet-100 accuracy: PrismaticVLM 79.88 % → OpenVLA SFT 77.48 % → Aligned 82.13 % → teacher C-RADIOv3 87.31 %. The strongest VL-fusion attention emerges in *middle* transformer layers 14–24. |
| **Emergent World Representations in OpenVLA** (arXiv:2509.24559) | Marco Molinari (LSE.AI / LSE), Leonardo Nevali (ETH Zurich), Saharsha Navani (LSE.AI / Princeton CS), Omar Younis (Mila – Quebec AI Institute), Sep 2025 | Embedding arithmetic + linear / nonlinear probes for state-transition vectors on the residual stream at layers 7 / 15 / 22 / 30; LIBERO 4-suite | OpenVLA has a *latent state-transition function* (Koopman-style). Probes peak at middle layers 15 and 22. Probing accuracy improves with pre-training compute. Linear probes ≥ MLP probes, consistent with the Linear Representation Hypothesis. The authors sketch an SAE-based planning extension. |
| **Probing OpenVLA for Symbolic States** (arXiv:2502.04558) | Hong Lu, Hengxu Li, Prithviraj Singh Shahani, Stephanie Herbers, Matthias Scheutz (Tufts HRILab, Department of Computer Science), Feb 2025; ICRA-Safe-VLM Workshop 2025 spotlight | Linear probes on all 33 hidden layers of OpenVLA's Llama-2 backbone for object-relation and action predicates; integration with the DIARC cognitive architecture | Symbolic states are recoverable with > 0.90 accuracy across most layers; contrary to the authors' hypothesis, object states are *not* encoded earlier than action states — the encoding is broadly distributed. First demonstrated VLA-cognitive-architecture hybrid. |
| **ReVLA — Reverting Vision Backbones** (arXiv:2409.15250) | Sombit Dey et al., 2024; workshop venue | Diagnoses catastrophic forgetting in OpenVLA's DINOv2 / SigLIP via a depth-regression probe; proposes gradual backbone reversal by model merging | DINOv2 inside OpenVLA fails depth regression after VLA fine-tuning. Reverting backbones improves OOD grasping by 77 % and lifting by 66 %. First "VLA training-health" diagnostic. |
| **Knowledge Insulating VLA** (arXiv:2505.23705; pi.website blog) | Driess, Springenberg, Ichter, Yu, Li-Bell, Pertsch, Ren, Walke, Vuong, Shi, Levine (Physical Intelligence), May 2025 | Stop-gradient between action expert and VLM backbone, plus discrete-token co-training | Letting the action expert's gradients flow back into the VLM corrupts its representations; insulating the backbone preserves VLM knowledge and speeds up training while still permitting continuous-action output. Indirect evidence for representation drift during action fine-tuning. |
| **From Intention to Execution / INT-ACT** (arXiv:2506.09930) | Irving Fang, Juexiao Zhang, Shengbang Tong, Chen Feng (NYU), 2025 | 50-task generalization probe across VLA architectures | Quantifies an "Intention-Action Gap": OOD semantic understanding does not translate into reliable execution. Fragile under language variations and compounding visual-language shifts. |
| **VLA-GSE / Actions as Language (VLM2VLA)** (arXiv:2605.06175 / 2509.22195) | Princeton + collaborators, 2025 | Compare full-fine-tune vs LoRA vs generalized / specialized experts on general multimodal benchmarks | Concrete catastrophic forgetting of VL benchmark performance during full VLA fine-tune even on π0.5; LoRA / GSE retain VLM capabilities. |
| **Original VLA architecture papers** (cited as targets) | OpenVLA (Kim et al., CoRL 2024 / PMLR 270), OpenVLA-OFT (Kim, Finn, Liang et al., "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success", RSS 2025, arXiv:2502.19645 — boosts OpenVLA average success 76.5 % → 97.1 % with 26× throughput increase), π0 (Black et al., 2024), π0.5 (Physical Intelligence, 2025), π*0.6 (PI, 2025), Octo (Octo Team, RSS 2024), RT-1 (Brohan et al., 2022), RT-2 (Zitkovich et al., 2023), Diffusion Policy (Chi et al., IJRR 2024), ACT (Zhao et al., RSS 2023), Gato (Reed et al., DeepMind 2022) | – | These works do not themselves perform internal representation analysis beyond ablations and t-SNE in Octo's appendix; all mechanistic studies of their internals come from the third-party papers above. |

**RT-2, Octo, Gato, Diffusion Policy, ACT — what's known about their internals:** RT-2 has no published mechanistic study (the model is closed-source; only behavioral evaluations exist). Octo, ACT and Diffusion Policy are included as the "non-VLM control" in Grant et al. (2026): ACT is treated as a CVAE-transformer baseline showing that *without* a pretrained VLM backbone, SAE-based feature extraction is far less interpretable. A Diffusion-Policy follow-up (D²PPO, arXiv:2508.02644) independently documents *representation collapse* in diffusion-policy hidden layers — clustered, indistinguishable embeddings for semantically different observations — and fixes it with a dispersive contrastive loss. Gato has only the original DeepMind technical report and qualitative LessWrong / Alignment-Forum write-ups; no published probe / SAE analyses exist for it as far as we found.

### 2. Internal modality gap inside VLAs

There is **no published direct measurement** of the Liang et al. (NeurIPS 2022) "cone-effect" cosine-distance metric *across modalities inside a VLA residual stream*. The Liang et al. baseline finding for CLIP-style models is an average image–text cone cosine of 0.56 / 0.47 / 0.51 across three CLIP variants, confirming a narrow-cone initialization bias; this protocol has not been ported to VLAs. The closest evidence to date:

- **Subspace separability (Grant et al. 2026, §4.5):** in all three dual-pathway VLAs (π0.5, SmolVLA, GR00T N1.5) the VLM-pathway and the expert-pathway activations live in **separable subspaces**; injecting expert activations causes 2× the behavioral displacement of injecting VLM activations, and subspace injection succeeds without cross-pathway interference. The "Layer Independence via CKA" analysis (Appendix E.3) and the per-token-vs-mean-pooled SAE collapse (96 %→8 %) further imply that vision, language, proprioception and action tokens occupy *non-fungible* subspaces.
- **Modality-collapse t-SNE (Kachaev et al. 2025, Fig. 5):** the closest published visual analogue to Liang et al.'s "two cones" plot — PrismaticVLM / Qwen2.5-VL maintain three clean class clusters (cup / bottle / knife) while OpenVLA collapses them, and the strongest VL-fusion attention emerges in middle layers 14–24.
- **Fusion-layer localization (Molinari et al. 2025):** state-transition probes peak at layers 15 and 22 of the 32-layer Llama-2 backbone of OpenVLA, again middle-layer.
- **Separate-dimension SAEs (Swann et al. 2026):** the authors *had to* train SAEs separately on PaliGemma (residual dim 2048) and on the π0.5 action expert (dim 1024); the two streams are not concatenable. The per-token SAE viewer reveals that individual features fire on *either* image tokens *or* text tokens with very little overlap.
- **Token-budget imbalance:** ThinkProprio (arXiv:2602.06575) shows that as little as 15 % of vision tokens reproduces full performance and that injecting proprioception as text tokens at the *input* beats late conditioning; VLA-Cache (arXiv:2502.02175) and VLA-Pruner (arXiv:2511.16449) show that the majority of vision tokens are redundant. Combined with Grant et al.'s "language is ignored when vision uniquely specifies the task" finding (94 %→10 % under wrong prompts on libero_goal vs 60–100 % on libero_object regardless), this constitutes the strongest evidence that VLA attention budget is dominated by vision tokens.

What is *missing*: a layer-wise plot of `cos(mean(vision_tokens), mean(text_tokens))`, `cos(mean(text), mean(action))`, etc., across the full transformer stack. This is a clean, low-cost experiment that has not been done.

### 3. How action prediction changes the language / vision representation space

The evidence is now quantitative and convergent:

- **Häon et al. (Berkeley, CoRL 2025):** the flagship empirical claim is, verbatim, *"less than 25 % of FFN neurons being rewired for action prediction and a large fraction of the remainder representing clear, semantically interpretable concepts."* This is the most direct measurement of how much of the LM is repurposed for action.
- **Kachaev et al. (Oct 2025):** linear-probe accuracy on ImageNet-100 drops from 79.88 % (PrismaticVLM) → 77.48 % (OpenVLA SFT) — a small but real loss; the VL-Think suite shows larger drops in symbol / icon / arrow categories *not* present in robotics data, i.e. domain-specific forgetting. The fix is an alignment loss to a frozen vision teacher, which raises probe accuracy to 82.13 %.
- **ReVLA (Dey et al. 2024):** DINOv2 inside OpenVLA fails depth regression that the pretrained DINOv2 passes — direct evidence of vision-encoder drift. Reverting the backbone yields 77 % grasp and 66 % lift OOD improvement.
- **Grant et al. 2026 (Appendix E.4):** "spatial binding" of motor programs emerges *during fine-tuning*; pretrained VLAs become *more* visually bound and *less* abstract about goals.
- **VLA-GSE (arXiv:2605.06175) and VLM2VLA (arXiv:2509.22195):** report concrete catastrophic-forgetting numbers on general multimodal benchmarks for full-fine-tuned VLAs (including π0.5) versus LoRA / generalized-expert variants.
- **Knowledge Insulation (PI, May 2025):** the entire motivation is that letting action-expert gradients flow back corrupts the VLM; freezing or insulating preserves representations.
- **Octo and ACT controls (Grant et al. 2026, Appendix C):** ACT, having no VLM backbone, shows much less interpretable SAE features — confirming that the rich semantics in OpenVLA / π0.5 are *inherited* from the VLM, not built by action training.

There is **no published CKA layer-wise comparison between an OpenVLA and its base PrismaticVLM, or between π0.5 and its PaliGemma initialization**, though Grant et al. 2026 perform an *intra-VLA* CKA layer-independence analysis. This is another concrete gap.

### 4. Universality of "memorized trajectory" SAE features

- **Swann et al. (Stanford, 2026)** are the first to document the memorization phenomenon: 98.43 % of active SAE features on π0.5 + LIBERO and 99.55 % on OpenVLA + LIBERO Goal are episode-specific memorized trajectories. The fraction *drops* under (a) larger / more diverse training data (DROID 1 545-task → 89–95 % memorized) and (b) knowledge insulation.
- **Grant et al. (CWRU, 2026)** independently corroborate the phenomenon across five other VLAs — π0.5, OpenVLA-OFT, X-VLA, SmolVLA, GR00T N1.5 — finding "spatially bound motor programs tied to scene coordinates rather than abstract task representations" (cross-task injection routes 99.8 % of X-VLA episodes to source-scene positions). Causal ablation across 15 096+ feature pairs yields 28–92 % zero-effect rates, meaning the *fraction* of features that are actually behaviorally consequential is small.
- **Distinguishing memorized vs general features** has now been operationalized two different ways: (i) Swann et al.'s four-metric logistic-regression classifier (onset count, episode coverage, mean activation, relative run length) and (ii) Grant et al.'s contrastive concept-discovery + causal ablation pipeline that recovers 82+ named manipulation concepts.
- **Connection to broader imitation-learning literature:** VLA-Arena (arXiv:2512.22539) and the "Beyond Mimicry" analysis (arXiv:2602.19930) explicitly argue that current VLAs perform "trajectory-level memorization" rather than compositional generalization. INT-ACT (Fang et al. 2025) names this the *Intention-Action Gap*. The Diffusion-Policy collapse paper (D²PPO, arXiv:2508.02644) finds analogous "indistinguishable embeddings" in plain Diffusion Policy.
- **Open question (genuinely unresolved):** whether memorized-trajectory features are *intrinsic to behavior cloning* or *amplified by the VLM-fine-tuning regime*. Swann et al. provide the first causal manipulation (knowledge insulation, dataset size) that shifts the ratio, but cross-architecture replication on ACT / Diffusion Policy / RT-2 (closed-source) is still missing.

### 5. Diagnosing training problems via internal representations

A "VLA training health check" toolbox is now assembling itself ad hoc, but no paper unifies it. The available diagnostics:

| Diagnostic | Source | What it detects |
|---|---|---|
| t-SNE of class-token embeddings vs base VLM | Kachaev et al. 2025 (Fig. 5) | Representation collapse |
| Attention-map IoU with target object | Kachaev et al. 2025 | Attention sink |
| Linear probes on ImageNet-100 or VL-Think suite | Kachaev et al. 2025 | Domain forgetting |
| Depth-regression probe on vision backbone | Dey et al. 2024 (ReVLA) | Catastrophic forgetting of DINOv2 / SigLIP |
| State-transition embedding-arithmetic probe | Molinari et al. 2025 | Whether a world model emerged |
| Linear probes for symbolic predicates | Lu et al. 2025 | Whether high-level symbolic structure is recoverable |
| FFN-neuron vocabulary projection | Häon et al. 2025 | How much of the LM was repurposed |
| Per-token SAE on memorization-vs-primitive metrics | Swann et al. 2026 | Memorization risk |
| Cross-task / null-prompt activation injection | Grant et al. 2026 | Vision-dominance and spatial-binding |
| Dead-neuron / sparsity histograms | Voita et al. 2023 (LLM-side baseline) | Underutilized capacity (not yet ported to VLAs in a dedicated paper) |
| Dispersive contrastive loss diagnostic | D²PPO (arXiv:2508.02644) | Hidden-layer collapse in Diffusion Policy |

**Modality imbalance / vision dominance** is empirically the dominant failure: Grant et al. show language becomes essential only when the scene under-specifies the task (libero_goal); on libero_object, language is ignored (60–100 % regardless of wrong prompt). VLA-Cache and VLA-Pruner show vision tokens are also the dominant *compute* sink; Distracting Token Pruning (arXiv:2601.16065) explicitly finds a "negative correlation between task success and attention mass on task-irrelevant regions."

**Feature drift, dead neurons, attention pattern analysis** of VLAs proper has not been done as a systematic study — only individual flavours appear inside the papers above.

### 6. The most promising experiments (decision-ready)

Ranked by expected impact / cost ratio:

1. **Layer-wise modality-gap measurement (the Liang et al. cone-effect ported to VLAs).** For each of OpenVLA, π0, π0.5, GR00T N1.5, SmolVLA, compute the centroid of vision-token, text-token, proprioception-token and action-token activations at every layer; report (a) centroid cosine distance, (b) within-modality average cosine, (c) effective dimensionality. Predicted outcome: a clear "two-cone → fusion → re-separation" pattern with fusion concentrated in layers 14–24, and modality cones much narrower in OpenVLA than in its base PrismaticVLM. Cost: a few GPU-days. *This is the single highest-value missing measurement.*
2. **Cross-VLA SAE feature transfer.** Train SAEs on π0.5 (per Swann et al.) and OpenVLA (per Grant et al.), build a feature-matching dictionary on shared LIBERO episodes, and ask whether "memorized" and "general" feature *directions* are preserved across architectures. If general motion-primitive features transfer but memorized features don't, that is direct mechanistic evidence for the dichotomy.
3. **CKA(VLA, base VLM) layer-by-layer for OpenVLA vs PrismaticVLM and π0.5 vs PaliGemma.** Combined with INT-ACT's behavioral split, this would quantify *which* layers retain VLM knowledge and *which* drift; complements Häon et al.'s 25 % FFN-rewiring number with a global representational-similarity measurement.
4. **A unified "VLA training health-check" benchmark.** Package together: t-SNE collapse score, attention-sink score (Kachaev), depth-regression probe (ReVLA), VL-Think probe, FFN-rewiring fraction (Häon), per-token memorization fraction (Swann), null-prompt cosine recovery (Grant). Release as a single tool. This is the missing analogue of LLM evaluation harnesses for VLAs.
5. **Memorization-vs-primitive analysis on Diffusion Policy and ACT.** D²PPO already documents collapse; combining its dispersive-loss diagnostic with Swann et al.'s SAE-classifier would tell us whether memorization is a VLM-fine-tuning artifact or a general behavior-cloning phenomenon.
6. **Causal test of the visual-dominance hypothesis via language-stream ablation during training.** If null-prompt rollouts of π0.5 recover 73–77 % success at inference, what happens if language is *also* dropped during pre-training fine-tuning? This isolates whether the visual-motor prior is a *training-data* artifact or an *architectural* one.
7. **Investigate proprioception's role in the modality gap.** ThinkProprio shows early-fusion of proprioception (as text tokens) helps substantially. A modality-gap analysis that includes the proprioception token should be the *first* place that learned-projection vs text-tokenized proprioception look geometrically different.

## Details

### What "VLA" means in this literature

The papers above use "VLA" specifically for *transformer-based vision-language-action policies that inherit a pretrained VLM backbone*: OpenVLA (Llama-2-7B + DINOv2 + SigLIP, 970k OXE episodes), OpenVLA-OFT (Kim et al., RSS 2025, arXiv:2502.19645 — continuous L1 regression head boosting average LIBERO success from 76.5 % to 97.1 % with 26× throughput), π0 (Black et al., flow-matching action expert), π0.5 (Physical Intelligence, PaliGemma + action expert, open-world generalization), π*0.6 (PI, RL-fine-tuned), GR00T N1.5 (NVIDIA, Eagle 2.5 VLM + DiT cross-attention; precise layer-count breakdowns are not publicly disclosed and should not be cited without verification), SmolVLA (Hugging Face, dual VLM + expert, interleaved), X-VLA (soft-prompted cross-embodiment) and RT-2 (Google, closed). Octo, RT-1, ACT, Diffusion Policy and Gato are *not* technically VLAs by the inherited-VLM-backbone definition but are routinely included as controls or comparison points.

### How fusion happens architecturally

Three patterns dominate (cf. ThinkProprio, arXiv:2602.06575, Table 1):

- **In-context conditioning** (OpenVLA, OpenVLA-OFT, X-VLA): vision + language + (sometimes) proprioception are concatenated as prefix tokens; action tokens are decoded autoregressively. Cross-modal fusion is *implicit* via self-attention, and the modality-gap question reduces to "do tokens of different types live in different subspaces of the residual stream."
- **Dual-expert / action-expert** (π0, π0.5, SmolVLA, GR00T): a smaller continuous-action expert is added in parallel to the VLM; either gradients flow back (vanilla) or are stopped (Knowledge Insulation). Modality fusion happens via cross-attention between expert and VLM.
- **CVAE-transformer** (ACT): no VLM; latent-variable + chunked-action prediction. Used as the non-VLM control.

### Quantitative anchors readers should remember

- **< 25 % FFN rewiring** (Häon et al. 2025) — the fraction of OpenVLA / π0 FFN neurons that change role for action prediction.
- **97–99 % memorized features** (Swann et al. 2026) — the fraction of active SAE features that encode episode-specific trajectories on small datasets; drops to ~89–95 % on DROID.
- **99.8 % source-task alignment** (Grant et al. 2026) — cross-task activation injection drives X-VLA toward the source trajectory's *scene coordinates*, not its task semantics.
- **96 %→8 %** (Grant et al. 2026) — task-success collapse when SAEs are mean-pooled instead of per-token; quantifies non-fungibility of modality tokens.
- **77 % grasping / 66 % lifting OOD improvement** (ReVLA, Dey et al. 2024) — the cost of catastrophic forgetting in DINOv2 inside OpenVLA.
- **t-SNE collapse + middle-layer fusion** (Kachaev et al. 2025) — OpenVLA collapses cup / bottle / knife tokens; strongest VL-fusion attention emerges in layers 14–24 of the 32-layer Llama-2 backbone.
- **State-transition probes peak at layers 15 and 22** (Molinari et al. 2025) of OpenVLA, consistent with the above.
- **Symbolic-predicate probes > 0.90 across most layers** (Lu et al. 2025) — but *not* earlier for objects than actions, contradicting layer-hierarchy hypotheses.
- **CLIP modality-gap baseline** (Liang et al. NeurIPS 2022, arXiv:2203.02053) — average image–text cone cosine of 0.56 / 0.47 / 0.51 across three CLIP variants; this protocol has not been replicated inside any VLA.

### What is still controversial / unresolved

- Whether the visual-pathway dominance documented by Grant et al. is an *artifact of LIBERO-style benchmarks where scenes uniquely specify the task* or a *general property* of fine-tuned VLAs. Grant et al. themselves note that on libero_goal — where scenes are ambiguous — language regains causal effect, so the dominance is *task-distribution-dependent*.
- Whether OpenVLA's "emergent world model" (Molinari et al.) is a true counterfactual model or a correlational artifact of probing — Molinari et al. flag this explicitly.
- Whether SAE features in VLAs actually correspond to *monosemantic* concepts at the same fidelity as in LLMs; Swann et al. note that per-token SAEs are less interpretable than per-timestep SAEs and warrant further study.
- The "arXiv:2603.xxxxx" 2026 preprints are very recent (March 2026) and have not been peer-reviewed; specific eye-catching numbers (e.g. 99.8 % cross-task alignment) should be treated as not yet independently replicated.

## Recommendations

For a researcher or lab planning a VLA-interpretability project in mid-2026:

**Stage 1 (weeks 0–4 — diagnostics).** Reproduce the Kachaev et al. t-SNE collapse and attention-sink experiments on your VLA of interest. Also run ReVLA's depth-regression probe. *Threshold to proceed:* if your VLA already shows < 5 % drop on ImageNet-100 linear-probe accuracy *and* clean class-token t-SNE clusters relative to its base VLM, the rest of the interpretability investment will have lower marginal value. *Threshold to abort and fix:* if the depth-regression probe shows catastrophic DINOv2 forgetting, fix the training recipe (Knowledge Insulation / Visual Representation Alignment / LoRA) *before* doing any further mechanistic work, because the downstream features will be polluted.

**Stage 2 (weeks 4–12 — modality gap measurement).** Run the layer-wise modality-gap experiment (Recommendation #1 above) on at least two architectures (suggested: OpenVLA-OFT and π0.5). Report cosine centroid distance, within-modality cosine, and effective rank per modality per layer. If a single layer shows fusion (cosine distance shrinks then re-expands) it is a candidate intervention point for steering.

**Stage 3 (weeks 12–24 — SAE + cross-architecture transfer).** Train per-token SAEs at the fusion layer identified in Stage 2 on at least three VLAs. Apply Swann et al.'s four-metric classifier *and* Grant et al.'s contrastive concept identifier. Test whether "general" features (memorization score < threshold) transfer across architectures via Procrustes alignment.

**Stage 4 (week 24+).** Use the diagnostics from Stages 1–3 as the basis for a published "VLA training health-check" benchmark — this is currently a clean white space in the literature.

**Benchmarks that would change the recommendations:**
- If a future paper produces a Liang et al.-style modality-gap measurement before yours, deprioritize Stage 2 and move directly to the SAE-transfer experiment in Stage 3.
- If RT-2 weights are ever released, the cross-architecture analysis becomes much more interesting because RT-2's discrete action tokenization is the third major paradigm (alongside flow-matching and continuous regression) and has been mostly excluded from existing studies.
- If a *negative* result emerges — e.g. a paper showing that fixing representation collapse does *not* improve OOD success — interpretability work should pivot toward the *Intention-Action Gap* of INT-ACT, i.e. why semantic understanding fails to translate to motor execution.

## Caveats

- Several key preprints (Swann et al., Grant et al., Kachaev et al.) are < 6 months old and not peer-reviewed; specific quantitative claims may shift on revision. The CWRU "Action Atlas" paper in particular reports very striking numbers (99.8 % source-task alignment) that should be replicated.
- The "VLA" label is used inconsistently in the literature: some papers (LaST-VLA, autonomous-driving VLAs) use it for non-robotic planners, and some include diffusion-only policies. The robotics-VLA subfield is what this report covers.
- Comparisons across the papers are complicated by different benchmarks (LIBERO vs DROID vs SIMPLER vs real UR5) and different fine-tuning recipes (SFT vs LoRA vs Knowledge Insulation vs OFT). Numbers should be compared *within* a paper rather than across.
- The "arXiv:26xx" IDs in this report (e.g. 2603.19183, 2603.19233, 2605.06175, 2601.16065, 2602.06575) follow arXiv's 2026 numbering convention; they are 2026 preprints visible on arXiv at the time of writing (May 2026).
- Architectural layer-counts for GR00T N1.5 quoted in some third-party summaries (e.g. "12 Eagle LM + 4 VL-SA + 16 DiT") are not confirmed in NVIDIA's public materials and should be verified directly against the model card before being relied on.
- No direct internal-representation analyses of RT-1, RT-2, or Gato exist in the public literature; statements about them are by analogy.
- We did not find any cone-effect measurement *inside* a VLA residual stream; the closest CLIP-side reference is Restrepo et al. "On the Cone Effect and Modality Gap in Medical Vision-Language Embeddings" (arXiv:2603.17246), which is *not* a VLA paper but provides the methodological template.

