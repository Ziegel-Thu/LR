### Robot representation learning and multimodal fusion

用户:


Claude:
his is a very detailed and well-structured research brief — covering robot model architectures, multimodal fusion, cross-embodiment representations, and how representation analysis can aid training. I'll dive into a comprehensive research session on all six sections right away.
our survey on representation analysis in robot learning models (2024–2026) is ready. It covers the current VLA ecosystem (RT-1/2, OpenVLA, π0/π0.5, Octo, V-JEPA 2-AC), probing and sparse-autoencoder findings on OpenVLA and π0.5, cross-embodiment representation gaps, modality-gap geometry, and representation analysis as a training aid (ReVLA, RS-CL, D²PPO). It includes a coverage assessment of which models have been analyzed, prioritized recommendations for new research directions, and caveats around source-date hygiene and closed-weight model limitations.
# Representation Analysis in Robot Learning Models: A 2024–2026 Survey

## TL;DR
- **Robot policy representations are now being analyzed with the same mechanistic-interpretability toolkit used for LLMs** (linear probes, sparse autoencoders, activation steering), but the evidence base remains thin: only a handful of papers (Molinari et al. 2025, Lu et al. 2025, Swann et al. 2026, Häon et al. 2025, Kachaev et al. 2025) have done rigorous internal analysis, and almost all target OpenVLA or π0/π0.5.
- **The biggest open gap is cross-embodiment representation alignment.** Despite Open X-Embodiment training on 22 robots, no published paper computes CKA / Procrustes / mutual-nearest-neighbor similarity between policy internals across embodiments, and the Platonic Representation Hypothesis has only just been invoked for VLAs (Kachaev et al., Oct 2025).
- **Representation analysis is starting to pay for itself as a training aid.** Depth-regression probes diagnose catastrophic forgetting in OpenVLA's DINOv2 backbone (ReVLA: 77%/66% relative gain on OOD grasping/lifting), cosine-distillation to a frozen vision teacher delivers up to a 10% relative OOD gain on Simpler (Don't Blind Your VLA), and dispersive losses fix representation collapse in Diffusion Policy (+22.7%). Expect these to become standard regularizers in 2026.

## Key Findings

1. **The VLA ecosystem has converged on a small set of architectures** — PaLI/PaliGemma + flow-matching or Llama-2 + discretized action tokens — and on Open X-Embodiment as the dominant pretraining corpus. Most large models (RT-2, π0, π0.5, GR00T, Gemini Robotics) are closed-weight; the open-weight stack is OpenVLA → Octo → SmolVLA → MiniVLA, plus V-JEPA 2-AC for world-model-based control.
2. **Modality-gap geometry is real but unmeasured inside VLAs.** Multiple recent papers (VITA, dVLA, DUST, FocusVLA) describe a vision–action "gap" qualitatively, but none report CLIP-style cone-effect numbers. This is a tractable, thesis-scale open problem.
3. **Probing of OpenVLA shows an emergent world model in the middle layers** (Molinari et al. 2025) and >0.90 linear-probe accuracy for symbolic object/action states across most Llama layers (Lu et al. 2025). These findings make OpenVLA the de facto "BERT of robotics" for interpretability work.
4. **Sparse autoencoders on π0.5 reveal that most features encode memorized trajectories,** with a minority of generalizable motion-primitive features (Swann et al. 2026), suggesting current VLAs behave more like retrieval engines than compositional planners.
5. **Cross-embodiment "convergence" is asserted but not measured.** RT-X demonstrates positive transfer (~50–200% success-rate uplift on small-data robots), but the claim that policies learn a shared latent space rests on behavioral, not representational, evidence.
6. **Robotics offers analysis affordances absent in LLMs**: ground-truth physical states, closed causal action–environment loops, and naturally controlled embodiment differences. These are largely untapped.

## Details

### 1. Major Robot Learning Models and Training Paradigms (2023–2026)

**Vision-Language-Action (VLA) models.**

- **RT-1** (Brohan et al., RSS 2023, arXiv 2212.06817): 35M-parameter FiLM-EfficientNet + TokenLearner + Transformer; inputs RGB sequence + language, outputs 11-D discretized actions (7 arm + 3 base + 1 mode); runs at 3 Hz; trained on 130k episodes from 13 Everyday Robots over 17 months; open-source.
- **RT-2** (Brohan et al., CoRL 2023, arXiv 2307.15818): co-fine-tunes web VQA with RT-1 robot data; actions tokenized as text. Two backbones — PaLM-E (12B) and PaLI-X (5B and 55B variants); the 5B PaLI-X runs at 5 Hz, the 55B variant runs at 1–3 Hz via multi-TPU. Not open-source; >3× emergent-skill success vs. baselines.
- **RT-X / RT-1-X / RT-2-X** (Open X-Embodiment Collaboration, ICRA 2024, arXiv 2310.08864): same architectures trained on the OXE dataset — 22 robots, 527 skills, 160,266 tasks, 60 datasets from 21 institutions. RT-1-X beats per-dataset originals on 4/5 small datasets.
- **OpenVLA** (Kim et al., CoRL 2024, arXiv 2406.09246): 7B parameters; Llama-2-7B + fused DINOv2 (~300M) + SigLIP (~400M) visual encoder; trained on 970k OXE episodes; 64× A100 × 15 days; outperforms RT-2-X (55B) by 16.5% absolute success on 29 tasks; Apache-licensed, supports LoRA/FSDP fine-tuning; bfloat16 / int4 inference.
- **Octo** (Octo Model Team, RSS 2024, arXiv 2405.12213): transformer-based diffusion policy; small (~27M) and base (~93M) versions; 800k OXE trajectories; supports language and goal-image conditioning; fully open-source.
- **π0 / π0.5** (Black et al., Physical Intelligence 2024–2025): PaliGemma VLM backbone + flow-matching continuous-action expert (Transfusion-style MoE block-attention); trained on 10,000+ hours and 903M timesteps across 7 robot configurations plus 9.1% OXE/DROID; closed-weight, partnership access. π0.5 adds a hierarchical FAST-tokenized discrete-action pretraining phase.
- **Gemini Robotics / On-Device** (Google DeepMind, 2025): Gemini-2.0 VLM backbone + dexterous low-level control; closed-weight; on-device variant preserves dexterity at low latency.
- **SmolVLA** (Hugging Face, 2025): 450M open-source compact VLA aimed at democratization.
- **MiniVLA** (Stanford SAIL): Qwen-2.5 0.5B + the same ViT as OpenVLA — 7× smaller language model.

**Diffusion / chunking policies.**

- **ACT — Action Chunking with Transformers** (Zhao et al., RSS 2023): CVAE encoder + transformer decoder predicting 20–100-step action chunks; ResNet-18 vision, ~50 ms inference at 7-DoF; introduced with the ALOHA bimanual platform.
- **Diffusion Policy** (Chi et al., RSS 2023): conditional DDPM over action chunks; explicitly multimodal action distribution; 200–500 ms DDPM, ~50 ms with consistency distillation.
- **Diffusion Transformer Policy / DiT-Policy** (Hou et al., arXiv 2410.15959): replaces a small action head with a large in-context conditional DiT; trained on OXE and beats Octo and OpenVLA on Franka and LIBERO.

**Cross-embodiment generalist agents.**

- **Gato** (Reed et al., DeepMind, 2022): 1.2B parameters; multi-modal, multi-task, multi-embodiment tokenized policy.
- **RoboCat** (Bousmalis et al., TMLR 2023, arXiv 2306.11706): Gato architecture + VQ-GAN encoder; visual goal-conditioned decision transformer; self-improvement loop fine-tunes on 100–1000 demos, then generates more training data.

**World-model methods.**

- **DreamerV3** (Hafner et al., Nature 2025): RSSM-based world model with fixed hyperparameters; strong on Proprio Control Suite, Atari, DMLab, and Minecraft; underpins DayDreamer for physical robot learning.
- **UniSim** (Yang et al., ICLR 2024 Outstanding Paper, arXiv 2310.06114): action-in / video-out generative simulator built from internet, robotics, and navigation data; RL policies trained purely in UniSim transfer zero-shot to real robots.
- **V-JEPA 2 / V-JEPA 2-AC** (Assran et al., Meta FAIR, arXiv 2506.09985, Jun 2025): Joint-Embedding-Predictive video world model with up to a 1-billion-parameter ViT-g encoder pretrained on over 1 million hours of internet video plus 1 million images. The action-conditioned predictor is a ~300M-parameter, 24-layer, 16-head, 1024-hidden-dimension transformer with GELU activations, post-trained on <62 hours of DROID-style robot video while the encoder is frozen. Zero-shot pick-and-place on two unseen Franka arms achieves 60–80% manipulation success and 100% reach, beating Cosmos (0–20%) at 16 s per action vs. 4 minutes.
- **Genie / Genie 2** (Bruce et al., ICML 2024) and **NVIDIA Cosmos** (2025): related video-world-model lines.

### 2. Representations Under Different Modality Combinations

**Vision+language vs. + proprioception vs. + tactile.** Architecturally, the dominant pattern is concatenative token-level fusion: visual patches and language tokens go through a shared transformer, while proprioception is either (a) zero-padded into the state/action vector (π0 zero-pads to 18-D), (b) tokenized and prepended (Octo, RoboCat), or (c) injected via FiLM or cross-attention (some Diffusion Policy variants). Tactile signals are integrated either by tokenizing each tactile modality independently and using cross-modal attention bottlenecks (Sparsh-X, Higuera et al., arXiv 2506.14754) or by fusing into a unified 3D point cloud with visual points (3D-ViTac, Huang et al., arXiv 2410.24091). Force-guided attention that dynamically reweights vision vs. touch at different manipulation stages (Adaptive Visuo-Tactile Fusion, arXiv 2505.13982) achieves 93% success on three contact-rich tasks.

**Probing across modalities.** Lu et al. (Tufts, "Probing a VLA for Symbolic States", arXiv 2502.04558, Feb 2025) probe each of OpenVLA's 33 Llama hidden layers on LIBERO-spatial and find >0.90 linear-probe accuracy for both object and action symbolic states across most layers — and, contrary to a natural hypothesis, object states are not encoded earlier than action states. Molinari et al. ("Emergent World Representations in OpenVLA", arXiv 2509.24559, Sep 2025) show the state-transition world model is concentrated in OpenVLA's **middle** Llama layers, with linear probes outperforming MLPs (evidence the structure is genuinely linear in the residual stream) and an Allan-variance-style noise analysis on the transition vectors.

**Modality dominance / imbalance.** FocusVLA, VITA, BayesVLA, and AVA-VLA all document attention-pattern evidence of "visual dominance" or "modality imbalance" between vision and action tokens. The Contrastive Representation Regularization for VLA paper (RS-CL, Kim et al., arXiv 2510.01711, Oct 2025) explicitly closes the proprioception–VLM gap by adding a contrastive auxiliary loss using proprioceptive-state distances as soft supervision, raising RoboCasa-Kitchen pick-and-place from 30.8% → 41.5% and real-robot tasks from 45.0% → 58.3%.

**CLIP-style modality gap in robot models.** The original modality-gap result (Liang et al., NeurIPS 2022, arXiv 2203.02053) showed CLIP image/text embeddings sit in two narrow cones at arm's length even at random initialization (cone effect + contrastive loss preserves the gap). No published paper measures cone-effect or mean inter-modality cosine distance between vision/language/proprio/action tokens in a VLA, although VITA and dVLA (arXiv 2509.25681) name a "modality gap between visual observations and low-level actions" and try to close it via shared discrete latents. DUST (arXiv 2510.27607) argues against forcing such a shared space. This is one of the clearest tractable research gaps surfaced by this survey.

### 3. Cross-Embodiment Representations

RT-X demonstrates **behavioral** cross-embodiment transfer (RT-1-X beats per-dataset originals on 4/5 small datasets; ~50–200% success-rate uplift), and π0 trains across 7 robot configurations by zero-padding to an 18-D unified state/action space. But **representational** cross-embodiment evidence is sparse.

- **Wang et al., "Cross-Embodiment Robot Manipulation Skill Transfer using Latent Space Alignment"** (arXiv 2406.01968, ICRA 2024): trains encoders/decoders to project source and target robot state/action spaces into a common latent with cycle-consistency + latent-dynamics losses; demonstrates sim-to-sim and sim-to-real transfer but does not measure post-hoc CKA between independently trained policies.
- **Mattson, Tian et al., "Representation Alignment from Human Feedback for Cross-Embodiment Reward Learning"** (RLC 2024, arXiv 2408.05610): soft-nearest-neighbor alignment between video trajectories of different embodiments — the closest published thing to MNN cross-embodiment representational analysis.
- **XSkill** (Xu et al., CoRL 2023): cross-embodiment skill prototypes via a skill alignment transformer.

**The Platonic Representation Hypothesis** (Huh, Cheung, Wang, Isola, ICML 2024, arXiv 2405.07987) explicitly flags robotics as a domain where convergence has not been demonstrated. The first paper to invoke it for VLAs is **Kachaev et al., "Don't Blind Your VLA: Aligning Visual Representations for OOD Generalization"** (arXiv 2510.25616, AAMAS 2026 Oral), which measures representation drift of OpenVLA vs. its base Prismatic/Qwen2.5-VL using t-SNE, attention maps, and ImageNet-100 linear probing; they observe clear representation collapse after action SFT and propose cosine-distillation alignment to a frozen DINOv2 / SigLIP / C-RADIOv3 / Theia teacher, yielding "up to a 10% relative gain in out-of-distribution generalization with negligible computational overhead" on the Simpler benchmark.

A concurrent paper, "Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale" (arXiv 2604.18572, 2026), pushes back: mutual-kNN agreement collapses at million-scale galleries — relevant for tempering Platonic-convergence claims in big robot policies.

**Net assessment**: cross-embodiment convergence in robot policies is plausible (positive transfer is real) but **not yet shown to be representational**. CKA/Procrustes/MNN between Octo per-robot fine-tunes, or between OpenVLA-on-robot-A vs. OpenVLA-on-robot-B, is a clear thesis-scale open problem.

### 4. Representation Analysis as a Training Aid

Several 2024–2025 papers operationalize representation metrics as training signals:

- **ReVLA** (Dey et al., arXiv 2409.15250, Sep 2024) uses a depth-regression linear probe on OpenVLA's DINOv2 features to diagnose catastrophic forgetting during action SFT; gradual backbone reversal (model merging) restores the probe and "improve[s] over OpenVLA by a factor of 77% and 66% for grasping and lifting in visual OOD tasks."
- **Don't Blind Your VLA** (Kachaev et al., arXiv 2510.25616) uses ImageNet-100 linear-probe accuracy + t-SNE as a diagnostic and cosine-similarity-to-frozen-teacher as a distillation regularizer — effectively a soft CKA-style objective. Up to 10% relative OOD gain on Simpler.
- **RS-CL** (Kim et al., arXiv 2510.01711, Oct 2025) uses proprioceptive-state distances as soft supervision for a contrastive auxiliary loss, raising RoboCasa pick-and-place from 30.8% → 41.5%.
- **D²PPO: Diffusion Policy Policy Optimization with Dispersive Loss** (arXiv 2508.02644, Aug 2025) identifies **representation collapse in Diffusion Policy** via hidden-feature cluster analysis and adds a dispersive regularizer: +22.7% pretraining and +26.1% post-finetuning on RoboMimic, reaching ~94% mean success.
- **Grover et al., "Enhancing Generalization in VLA Models by Preserving Pretrained Representations"** (arXiv 2509.11417, Sep 2025) mixes frozen + finetuned visual encoders and uses a language-aligned string action tokenizer; the tokenizer alone raises OOD success from 35.03% → 50.25%; with VL co-training, 78.17% vs. 51.05%.

**Representation distillation with CKA/Procrustes objectives** is well established outside robotics (RCKA/PCKA, Zhou et al. IJCAI 2024 / arXiv 2401.11824; Saha et al. BMVC 2022) but has not been published as a teacher→student loss for robot foundation models specifically — a clear research gap.

**Sim-to-real via representations**: Ma et al., "Skill Transfer and Discovery for Sim-to-Real Learning" (arXiv 2404.05051), use spectral-decomposition skill representations transferable across same-dynamics tasks; orthogonal-skill discovery closes up to 30.2% of the sim-to-real gap on Crazyflie 2.1 quadrotors. State Representations as Incentives (arXiv 2309.11984) finds pretrained visual features systematically improve sim2real grasping over end-to-end visual learning.

### 5. Existing Representation-Analysis Work on Robot Models

Beyond Molinari et al. 2025, the corpus is small but growing:

- **Probing**:
  - Molinari et al., "Emergent World Representations in OpenVLA" (arXiv 2509.24559, Sep 2025): linear and non-linear probes on the residual stream find state-transition predictability concentrated in middle layers; Allan-variance noise analysis on transition vectors; outlines an SAE pipeline but doesn't train one.
  - Lu et al., "Probing a VLA for Symbolic States" (arXiv 2502.04558, Feb 2025): >0.90 linear-probe accuracy across most of OpenVLA's 33 Llama layers; integrated into the DIARC cognitive architecture for online symbolic belief updates.
  - Grover et al. (arXiv 2509.11417, Sep 2025): probes that demonstrate representation degradation during SFT.

- **Sparse autoencoders**:
  - Swann et al., "Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models" (Stanford; arXiv 2603.19183, 2026 / drvla.github.io): SAEs trained on π0.5 hidden-layer activations from the language backbone and the action expert. The majority of features correspond to memorized training trajectories; a minority encode interpretable motion primitives and semantics. Steering individual features causally modifies robot behavior; fine-tuning on small datasets amplifies memorization, while knowledge insulation / larger datasets shift the balance toward general features.
  - "Observing and Controlling Features in VLA Models" (arXiv 2603.05487, 2026): formalizes feature observability/controllability on π0.5 and OpenVLA via linear probes + minimal optimal-control-grounded interventions.

- **Causal / activation steering**:
  - Häon, Stocking, Chuang, Tomlin (Berkeley), "Mechanistic Interpretability for Steering VLA Models" (arXiv 2509.00328, Aug 2025; CoRL 2025): projects FFN value vectors onto the token-embedding basis without additional training data; finds sparse semantic directions ("fast", "slow", "up") and steers π0 and OpenVLA on LIBERO and a real UR5 zero-shot. ~25% of FFN neurons are rewired for action prediction; the rest stay semantically interpretable.
  - "Task Reconstruction and Extrapolation for π₀ using Text Latent" (Quanyi Li, arXiv 2505.03500, May 2025): averaging text-token hidden states yields a "text latent"; temporal interpolation of two task latents raises π₀ LIBERO-OOD success from 9% → 83% (other SOTA VLAs score <21%). Also documents **spatial overfitting** — VLAs associate object names with training locations rather than identities — and a "private instruction / backdoor" vulnerability via unreadable optimized text latents.

- **Geometric / attention analysis**: FocusVLA, FutureVLA, VITA, dVLA, AVA-VLA all use attention-map visualization to argue for cross-modal imbalance but do not report quantitative geometric measures (cone angles, inter-modality cosine).

**Coverage table** — analyzed: OpenVLA (heavily), π0/π0.5 (Swann + Häon + Li), Magma-style ConvNeXt VLAs (one SAE paper). **Unexplored**: Octo, Diffusion Policy (apart from D²PPO's cluster analysis), RT-1/RT-2 (closed weights limit access for RT-2), V-JEPA 2-AC, RoboCat, Gato, DreamerV3-for-robotics, UniSim. The V-JEPA 2 paper itself reports attentive/linear probes on action anticipation but **no third-party probing study of the action-conditioned predictor** exists yet.

### 6. Unique Opportunities Robot Models Provide

1. **Physical ground truth is directly measurable.** Unlike "did the model understand the sentence?", you can probe for end-effector pose, object position, contact force, joint angle — all with sub-mm or sub-Newton precision. Molinari (2025) and Lu (2025) already exploit this for state probing.
2. **Closed causal loops.** Action → environment change → next observation is a structural causal graph the model lives inside. Activation-steering experiments (Häon 2025; Swann 2026) obtain *physical* counterfactuals — you can verify whether a "speed" feature really controls speed by measuring the robot.
3. **Embodiment as natural controlled variation.** OXE's 22 robots with overlapping tasks is a built-in factorial design for testing the Platonic Representation Hypothesis along the embodiment dimension. **Nobody has used it for this yet.**
4. **Safety-criticality grounds interpretability in real consequences.** Steering a representation that drives a 7-kg arm next to a human is a different proposition from steering text generation; this is the strongest single argument for taking VLA interpretability seriously now (Swann et al. 2026 frame this in their motivation).
5. **Time-extended trajectories** give multi-step state evolution data that natural-language pretraining lacks — the basis for Allan-variance and Koopman-operator analyses (Molinari 2025).
6. **Sparse-reward / sparse-action structure** makes memorization-vs-generalization separability cleaner than in LLMs — Swann's SAE result ("most features = memorized trajectories") would be much harder to establish in text.

## Recommendations

For a researcher entering this field in mid-2026, in priority order:

1. **Pick OpenVLA as your model and OXE as your data,** both for tooling maturity and because every existing interpretability result targets it — your work composes with theirs. Use Octo as a smaller-scale control. *Stage gate*: revisit if a fully open π0-class model with weights drops.
2. **The single highest-leverage open problem is cross-embodiment representational alignment.** Take two OpenVLA fine-tunes on different OXE robots (e.g., Franka vs. WidowX), compute CKA, mutual-kNN, and Procrustes on residual-stream activations of paired demonstrations, and test the Platonic Representation Hypothesis along the embodiment axis. The Huh et al. mutual-kNN protocol ports directly. **Benchmark to change recommendation**: if CKA > 0.7 across embodiments, focus on what the *shared* subspace encodes; if < 0.3, focus on the *embodiment-specific* directions.
3. **Geometric measurement of the VLA modality gap.** Replicate Liang et al.'s cone-effect and gap-distance analysis on OpenVLA's vision/language/proprio/action token clouds. This is a one-month project that will be cited by every subsequent VLA-architecture paper. **Threshold**: if mean inter-modality cosine > 0.5 (à la CLIP), VLAs inherit the same gap and DUST's argument against shared latents weakens.
4. **Build a probe suite as a training diagnostic.** Following ReVLA + Kachaev: linear-probe ImageNet/depth + symbolic-state probes (Lu 2025) every k steps; treat the curves like loss curves. Use them to trigger early stopping or learning-rate decay when probe accuracy starts collapsing — likely a free 5–10% OOD success.
5. **Use CKA as a distillation loss for small VLAs.** MiniVLA/SmolVLA are obvious targets. The CKA-distillation literature (Saha BMVC 2022; RCKA IJCAI 2024) ports straightforwardly; nobody has tried it on VLAs.
6. **Train SAEs on Octo, Diffusion Policy, and V-JEPA 2-AC.** Swann et al.'s π0.5 result ("mostly memorized features") probably does not generalize — Octo and V-JEPA 2-AC have very different training distributions and inductive biases. Replication across architectures will tell us whether memorization is intrinsic to behavior-cloning VLAs or an artifact of small-data fine-tuning.

## Caveats

- **Source-date hygiene**: several papers cited in subagent results carry 2026 arXiv IDs (2603.xx, 2604.xx) that may be revised or future-indexed. The 2024–2025 results (Molinari 2509.24559, Lu 2502.04558, Häon 2509.00328, Kachaev 2510.25616, Kim 2510.01711, Grover 2509.11417, Dey 2409.15250, D²PPO 2508.02644) are confirmed.
- **Positive-transfer claims in RT-X and π0 are behavioral, not representational.** Treat "models converge in representation space" as a hypothesis, not an established finding.
- **OpenVLA's interpretability dominance is itself a sample bias.** Cross-architecture replication (Octo, Diffusion Policy, V-JEPA 2-AC) is mandatory before generalizing.
- **Sparse-autoencoder findings are model- and dataset-specific.** Swann et al.'s "mostly memorized" finding on π0.5 fine-tuned on small data may not hold for OpenVLA pretrained on 970k OXE episodes.
- **Closed-weight models (RT-2, π0, π0.5, Gemini Robotics) cannot be probed by external researchers.** Practical work is limited to OpenVLA-family, Octo, MiniVLA/SmolVLA, V-JEPA 2-AC, and open re-implementations such as open-pi-zero.
- **The "modality gap in VLAs" is currently an architectural intuition, not a measurement.** Several papers fix what they call the gap without ever measuring it; readers should treat numerical claims about it skeptically until cone-effect / cosine-distance analyses are published.

