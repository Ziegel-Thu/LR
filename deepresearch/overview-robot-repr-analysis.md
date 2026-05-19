# 表征分析用于机器人训练：方向综述

> 综合 DR-019 ~ DR-022 及 Lei 2026 论文，2026 年 6 月

---

## 1. 这个方向的主流工作在做什么？

"把表征分析工具用到机器人策略模型上"这件事，到 2026 年中已经不是一个空想，而是一个初具雏形但尚未成体系的子领域。可以按五条研究线来理解它的全貌。

### 研究线 A：VLA 的内部可解释性

这条线最集中、发展最快。核心问题是：当一个 VLM（如 PaliGemma / Llama-2）被 fine-tune 成 VLA 之后，它的内部表征发生了什么变化？

**关键发现已经收敛为三个事实：**

1. **Fine-tuning 导致视觉表征坍缩与灾难性遗忘。** ReVLA (Dey et al. 2024) 用 depth-regression probe 发现 OpenVLA 的 DINOv2 backbone 在 action SFT 之后丧失了深度回归能力。Kachaev et al. (2025) 用 t-SNE + attention map + ImageNet linear probe 进一步确认了三种失败模式：attention sink、表征坍缩、domain forgetting。Knowledge Insulation (Physical Intelligence 2025) 的整个动机就是阻止 action expert 的梯度回传污染 VLM。

2. **VLA 大部分 SAE 特征是"记忆的轨迹"，而非可复用的运动基元。** Swann et al. (Stanford 2026, Dr.VLA) 在 π0.5 上训练 SAE，发现 98.4% 的活跃特征是 episode-specific 的记忆；只有约 1.5% 是可泛化的运动基元。Grant et al. (CWRU 2026, Action Atlas) 在 6 个 VLA 上独立复现了类似现象——cross-task activation injection 将 99.8% 的 X-VLA episodes 导向源任务的场景坐标。数据规模更大时（DROID 1545 tasks）memorized 比例降至 ~89-95%。

3. **视觉通路主导 action generation，语言通路在很多场景下是摆设。** Grant et al. 发现 π0.5 在 null-prompt（无指令）下仍保持 cosine 0.775 的 action 相似度；只注入 layer-0 视觉激活就恢复到 0.997。但在场景无法唯一确定任务时（如 LIBERO-goal），语言重新变得关键——所以这是"数据分布依赖"的，不是绝对的。

**方法上的贡献：** Häon et al. (Berkeley, CoRL 2025) 建立了第一个 VLA mechanistic steering 框架——把 FFN value vector 投影到词表空间，找到 "up"/"slow" 等语义方向，zero-shot 控制真实 UR5。Molinari et al. (2025) 在 OpenVLA 中层发现了 emergent world model（linear probe 准确率高于 MLP，支持 Linear Representation Hypothesis）。Lu et al. (Tufts 2025) 对 OpenVLA 33 层做符号状态 probing，> 0.90 准确率，且反直觉地发现 object 状态并不比 action 状态更早编码。

**覆盖范围：** 几乎所有 interpretability 工作集中在 OpenVLA 和 π0/π0.5 上。Octo、Diffusion Policy、RT-1/RT-2、V-JEPA 2-AC、Gato、DreamerV3 的内部表征**基本未被研究**。

### 研究线 B：跨模态融合与模态间隙

VLA 内部存在 vision / language / proprioception / action 四种模态的 token。它们在 residual stream 中是否分居不同子空间？融合发生在哪些层？

**现状：没有人做过 Liang et al. (NeurIPS 2022) 的 cone-effect cosine distance 测量。** 这是最清晰的"everybody should do but nobody did"的空白。间接证据来自：
- Grant et al. 的 per-token vs mean-pooled SAE 实验（96%→8% 坍缩，证明不同模态 token 占据 non-fungible 子空间）
- Kachaev et al. 的 t-SNE 显示 VL-fusion attention 集中在 middle layers (14-24)
- Molinari et al. 的 world model probe 在 layer 15 和 22 峰值

多篇论文（FocusVLA, VITA, dVLA, BayesVLA, AVA-VLA）定性描述了"视觉-动作模态间隙"，但没有定量测量。RS-CL (Kim et al. 2025) 用 CKNNA 诊断出 VLM 表征与 proprioception 之间的低对齐度，并用 contrastive loss 修复（30.8% → 41.5%）。

**Proprioception 的特殊地位：** Lu et al. (2026) 发现 vision-proprioception 策略**平均比 vision-only 差 15.8%**，因为 proprio 的 loss reduction 更快，抑制了视觉学习。Think-Proprio 提出把 proprio 离散化为 VLM token 而非 MLP concat，在 CALVIN/LIBERO 长程任务上一致优于 MLP。

### 研究线 C：跨 embodiment 表征对齐

RT-X 在 22 种机器人上训练后展示了 positive transfer（小数据集 50-200% 成功率提升），π0 通过 zero-padding 到 18-D 统一动作空间覆盖 7 种机器人。但**所有跨 embodiment "convergence" 的证据都是行为层面的，不是表征层面的**。

唯一接近表征分析的工作：
- Wang et al. (ICRA 2024)：cycle-consistency latent alignment，但没做 post-hoc CKA
- Mattson & Tian (RLC 2024)：soft-nearest-neighbor alignment between video trajectories——最接近 MNN 跨 embodiment 分析的工作
- Kachaev et al. (2025)：首次在 VLA 中引用 Platonic Representation Hypothesis，但只做了 OpenVLA vs 其 base VLM 的比较，不是跨 embodiment

**OXE 的 22 个机器人在重叠任务上的数据是一个天然 factorial design，但没人用它做 CKA/Procrustes/MNN。这是一个 thesis-scale 空白。**

### 研究线 D：Sim-to-Real 的表征视角

Lei et al. (UT Austin, 2026) 是这条线最重要的新工作。他们对 diffusion policy 的 sim-real co-training 做了理论分析，发现两个内在效应：

1. **Structured Representation Alignment (SRA)：** 源域和目标域的 latent 需要同时满足 alignment（close enough for transfer）和 discernibility（separable enough for adaptation）。三种情形——disjoint（无 transfer）、structured-aligned（sweet spot）、overlapping（负 transfer，bimodal action）。用 Wasserstein distance 量化 alignment，用 MLP domain classifier 量化 discernibility。SRA 解释了 ~50% 的 loss variance；mixing ratio 只解释 ~20%。

2. **Importance Reweighting Effect：** mixing ratio $w$ 通过 softmax 调制每个域的 score function 权重。提供了闭式推荐：$w_n = N/(N+M)$ 和 $w_q = \sqrt{N/M}$。

**方法：CFG-ADDA = ADDA（adversarial alignment）+ CFG（classifier-free guidance 保持 discernibility）**。在 NutAssembly / MugHang / MugCleanup 上达到 21/30 平均成功率 vs vanilla 15.3/30。

Cheng et al. (2025) 用 unbalanced optimal transport + temporal sampling 得到 ~30% 提升。Ma et al. 用 spectral-decomposition skill representations 关闭 ~30% 的 sim-real gap。

**这条线的重要性在于：它首次给出了"表征分析何时能直接指导训练决策"的定量框架。**

### 研究线 E：数据模态与数据集层面的分析

**关键事实：** 机器人领域的"最小模态包"已经收敛为 RGB + proprioception + language，但跨数据集的模态不一致性是核心障碍。Octo 的预训练数据只有 27% 含 wrist camera、56% 含 language。OXE 85%+ 的轨迹来自 4 种机器人。

**模态的边际价值排序（加到 RGB 上的增量）：** language ≫ wrist camera ≫ depth ≈ tactile ≫ proprioception。深度主要作为 auxiliary supervision 而非运行时输入（DPR, DI²）。Tactile/audio 需要 language 做跨模态 anchor（FuSe）。

**数据集层面的统计分析几乎空白。** DROID 可视化了 3D 接触点分布，OXE-AugE 量化了不平衡，Re-Mix 用 no-regret optimization 学习混合权重——但 intrinsic dimension、mutual information、manifold structure 的正式分析都没有做过。

---

## 2. 表征分析在机器人领域的已有应用

按工具类型整理：

### Probing

| 工作 | 模型 | 方法 | 发现 |
|------|------|------|------|
| Molinari et al. 2025 | OpenVLA | Linear/MLP probes on residual stream (layer 7/15/22/30) | 中层 (15, 22) 存在 emergent world model；linear probe ≥ MLP，支持 LRH |
| Lu et al. 2025 | OpenVLA | Linear probes on 33 Llama layers for symbolic states | > 0.90 准确率；object states 和 action states 编码分布一致（非层级式） |
| ReVLA (Dey et al. 2024) | OpenVLA | Depth-regression probe on DINOv2 | 诊断出 catastrophic forgetting；backbone reversal → +77%/+66% OOD |
| Kachaev et al. 2025 | OpenVLA | t-SNE + ImageNet-100 linear probe + attention map IoU | 三种失败模式：attention sink、表征坍缩、domain forgetting |
| Grover et al. 2025 | OpenVLA | Linear probes during SFT | 表征在 fine-tuning 过程中退化 |

### Sparse Autoencoders (SAE)

| 工作 | 模型 | 发现 |
|------|------|------|
| Swann et al. 2026 (Dr.VLA) | π0.5, OpenVLA | 98.4%/99.6% 特征为 memorized trajectories；用四指标分类器区分 general/memorized；Knowledge Insulation 增加 general 比例 |
| Grant et al. 2026 (Action Atlas) | 6 VLAs (π0.5, OpenVLA-OFT, X-VLA, SmolVLA, GR00T N1.5, ACT) | Per-token SAE 保留 96%→94% 成功率，mean-pooled SAE 坍缩到 8%；82+ named manipulation concepts |
| "Observing and Controlling Features" 2026 | π0.5, OpenVLA | 形式化 feature observability/controllability |

### Activation Steering / Causal Analysis

| 工作 | 模型 | 方法 | 发现 |
|------|------|------|------|
| Häon et al. 2025 (CoRL) | π0, OpenVLA | FFN value-vector projection to vocabulary | < 25% FFN neurons rewired for action；semantic directions ("up","slow") 可 zero-shot 控制 UR5 |
| Quanyi Li 2025 | π0 | Text latent averaging + interpolation | Text latent interpolation → LIBERO-OOD 9%→83%；发现 spatial overfitting |
| Grant et al. 2026 | 6 VLAs | Null-prompt/cross-task activation injection | 视觉主导 action；cross-task injection → 99.8% 到源场景坐标 |

### CKA / Similarity Metrics

| 工作 | 用法 |
|------|------|
| RS-CL (Kim et al. 2025) | CKNNA 诊断 VLM-proprio 低对齐 → contrastive loss 修复 |
| Grant et al. 2026 | Appendix E.3 CKA layer-independence analysis |
| Kachaev et al. 2025 | 定性 t-SNE（非 CKA） |

### Clustering / Manifold Analysis

| 工作 | 模型 | 发现 |
|------|------|------|
| D²PPO (Zou et al. 2025) | Diffusion Policy | Hidden-layer cluster analysis 发现表征坍缩 → dispersive loss +22.7% |
| Lei et al. 2026 | Diffusion Policy | UMAP + Wasserstein distance → 定义 SRA 三种 regime |
| Moalla et al. 2024 | PPO agents | Feature-rank 测量 → 表征坍缩先于性能坍缩 |

**总结：几乎所有工作集中在 OpenVLA 和 π0.5 上。Octo、RT-1/2、V-JEPA 2-AC、Gato 内部未被分析。没有人做过 VLA 内部的 cone-effect 模态间隙测量。没有 CKA(VLA, base VLM) 的 layer-wise 完整比较。**

---

## 3. 表征分析辅助训练的已有案例

### 机器人领域

| 方法 | 诊断工具 | 发现的问题 | 修复方案 | 效果 | 论文 |
|------|---------|----------|---------|------|------|
| ReVLA | Depth-regression probe | DINOv2 catastrophic forgetting | Gradual backbone reversal (model merging) | +77%/+66% OOD (grasping/lifting) | Dey et al. 2024, arXiv 2409.15250 |
| Don't Blind Your VLA | t-SNE + ImageNet probe | 表征坍缩 + attention sink | Cosine-distillation to frozen vision teacher | Up to +10% relative OOD on Simpler | Kachaev et al. 2025, arXiv 2510.25616 |
| RS-CL | CKNNA alignment | VLM 对 proprio 信号不敏感 | Soft-supervised contrastive loss (proprio distance) | 30.8%→41.5% (RoboCasa), 45%→58.3% (real) | Kim et al. 2025, arXiv 2510.01711 |
| D²PPO | Clustering analysis | Diffusion Policy 表征坍缩 | Dispersive loss (InfoNCE 变体) | +22.7% pretrain / +26.1% finetune (RoboMimic ~94%) | Zou et al. 2025, arXiv 2508.02644 |
| Lei et al. CFG-ADDA | Wasserstein + domain classifier | 缺乏 structured alignment | ADDA (alignment) + CFG (discernibility) | 21/30 vs 15.3/30 vanilla (~37% relative) | Lei et al. 2026, arXiv 2604.13645 |
| Dr.VLA (steering) | SAE memorization classifier | 大部分特征为 memorized trajectories | 推荐 KI / larger data；feature steering | Causal behavior modification on LIBERO | Swann et al. 2026, arXiv 2603.19183 |
| Häon et al. (steering) | FFN vocabulary projection | < 25% neurons for action | Zero-shot activation steering | Real UR5 行为控制 | Häon et al. 2025, arXiv 2509.00328 |
| Grover et al. | Probe + mixed encoder | SFT 导致视觉退化 | Frozen + finetuned vision mix + language-aligned tokenizer | 35%→50.25%→78.17% OOD success | Grover et al. 2025, arXiv 2509.11417 |
| Knowledge Insulation | 间接（action gradient 分析） | Action expert gradient 污染 VLM | Stop-gradient between expert and VLM | 保留 VLM 知识 + 加速训练 | PI 2025, arXiv 2505.23705 |

### CV/NLP 中可迁移到机器人的案例

| 方法 | 领域 | 效果 | 可迁移性 | 论文 |
|------|------|------|---------|------|
| CKA-based knowledge distillation (RCKA/PCKA) | CV (ImageNet, CIFAR-100, COCO) | SOTA distillation 效果；CKA-ViT 4-8× 压缩 | **直接可用**——压缩 VLA 到 MiniVLA/SmolVLA，但还没人做 | Zhou et al. IJCAI 2024; CKA-ViT AccML 2024 |
| Structural probes as training diagnostics | NLP (BERT/ELMo) | 检测语法结构编码 | 思路可搬：把 syntactic probe 替换为 gripper-state/EE-pose/depth probe 做 training callback | Hewitt & Manning 2019; LPASS pruning |
| Modality gap analysis + closure | CLIP 系 | 诊断 image-text cone effect；closing gap 改善 clustering | **直接可移植**到 VLA 的 vision/language/action token | Liang et al. NeurIPS 2022; Eslami & de Melo 2024 |
| Intrinsic dimension tracking | General DNN (Atari, MuJoCo, vision) | 训练过程 manifold 降维 → generalization 信号 | 可做 ID-based early stopping 或 curriculum trigger | Mao et al. PNAS 2024; Birdal et al. |
| Feature-rank collapse detection | On-policy RL (PPO) | 表征坍缩先于性能坍缩 | **已部分迁移**（D²PPO 用了类似思路） | Moalla et al. 2024, arXiv 2405.00662 |
| SAE on DQN | RL (Atari) | Deep Q-Network 的 feature decomposition | 思路可扩展到 diffusion/flow-matching action heads | DuPlessie 2024 (MIT-PRIMES) |
| OT-based domain adaptation | Sim-to-real (robot, 但非 VLA) | Unbalanced OT + temporal sampling → +30% | **已迁移**（Cheng et al. 2025 用于机器人） | Cheng et al. 2025, arXiv 2509.18631 |

---

## 4. 最大的空白在哪里？

按重要性排序：

### 空白 1：VLA 内部模态间隙的定量测量

Liang et al. (NeurIPS 2022) 的 cone-effect cosine distance 已经是 CLIP 研究的标准分析，但**没有任何论文在 VLA 的 residual stream 中测量过 vision/language/proprio/action token 之间的 cosine centroid distance**。多篇论文定性讨论"模态间隙"并试图修复它，却从未先测量它。这是一个 one-month project，会被每篇后续 VLA 架构论文引用。

### 空白 2：跨 embodiment 的表征对齐分析

RT-X 的 22 种机器人和 π0 的 7 种机器人在重叠任务上有丰富数据。**没有任何论文计算过 CKA/Procrustes/MNN between policy internals across embodiments。** Platonic Representation Hypothesis 在 VLA 中仅被 Kachaev et al. 引用一次，且只做了 VLA vs base VLM 的比较，而非跨 embodiment。这是最自然的 thesis-scale problem。

### 空白 3：CKA(VLA, base VLM) 的 layer-wise 完整比较

Häon et al. 说 < 25% FFN neurons 被 rewire，Kachaev et al. 用 t-SNE 显示了坍缩，但**没有人做过 OpenVLA vs PrismaticVLM 或 π0.5 vs PaliGemma 的逐层 CKA 图**。这将把现有的零散发现统一成一张"fine-tuning 如何改变 VLA 表征"的完整地图。

### 空白 4：表征分析作为持续训练监控

现有工具全是 post-hoc 的。ReVLA 的 depth probe、Kachaev 的 ImageNet probe、Dr.VLA 的 memorization classifier——都在训练之后才运行。**没有人把 probe suite 做成 training callback**，像 loss curve 一样持续监控。这在 NLP 中早已成熟（structural probes），在机器人中完全缺失。

### 空白 5：CKA 作为 VLA 蒸馏损失

CKA-based distillation 在 CV 中已成熟（RCKA/PCKA, IJCAI 2024）。**没有人把它用于 VLA teacher → student 的蒸馏。** MiniVLA (0.5B) / SmolVLA (450M) 是天然目标。

### 空白 6：非 VLA 架构的 SAE 分析

Swann et al. 的"大部分特征是记忆的轨迹"这一发现只在 π0.5 小数据 fine-tune 上确认。**Octo、Diffusion Policy、V-JEPA 2-AC** 有完全不同的 inductive bias 和训练分布，memorization 行为可能截然不同。跨架构复现是确认"这是 BC 的固有问题还是小数据 VLM-fine-tuning 的 artifact"的唯一途径。

### 空白 7：数据集层面的 manifold / information-theoretic 分析

机器人数据集的 intrinsic dimension、cross-modal mutual information、manifold structure 从未被正式研究。DROID 可视化了接触点分布，但这只是个开始。

---

## 5. 我们的机会

综合以上分析，如果我们要进入这个方向，最有价值的具体方向如下（按 impact/feasibility 排序）：

### 方向 1：VLA 模态间隙的 layer-wise 定量测量

**做什么：** 在 OpenVLA（open-weight, 7B）的 32 层 residual stream 中，计算每层 vision/language/proprio/action token centroid 的 cosine distance、within-modality cosine、effective rank。复现 Liang et al. 的 cone-effect 协议。

**为什么值得做：**
- 这是目前最清晰的"大家都知道该做但没人做"的测量
- 成本低（几 GPU-day），但结果将成为 VLA 架构设计的基准参考
- 自然延伸到"fusion layer 在哪里""为什么 per-token SAE 保留性能而 mean-pooled 不行"等问题的定量解释
- 可以直接利用我们已有的表征分析工具链（CKA, probing, netrep/repsim）

**预期结果：** 一个"two-cone → fusion → re-separation"的 layer-wise pattern，fusion 集中在 layer 14-24（与 Kachaev/Molinari 的间接证据一致）。

### 方向 2：跨 embodiment 表征收敛的直接测试

**做什么：** 在 OXE 上对 OpenVLA 做 per-robot fine-tune（如 Franka vs WidowX），在 paired demonstrations 上计算 residual-stream 的 CKA、mutual-kNN (Huh et al.)、Procrustes，直接测试 Platonic Representation Hypothesis 在 embodiment 维度上是否成立。

**为什么值得做：**
- 跨 embodiment "convergence" 是这个领域最大的 claim 之一，但完全没有表征层面的证据
- 如果 CKA > 0.7，我们可以研究 shared subspace 编码了什么（task structure? physics?）
- 如果 CKA < 0.3，我们可以研究 embodiment-specific directions，以及如何 align 它们
- 与 Lei et al. 的 SRA 框架自然对接：sim-real 可推广到 cross-embodiment

### 方向 3：Probe Suite 作为训练时持续诊断

**做什么：** 构建一个 `RepresentationCallback`，每 N 步自动计算：(a) gripper-state / EE-pose / success-prediction linear probes；(b) CKA-to-frozen-reference（检测 catastrophic forgetting）；(c) effective rank / intrinsic dimension（检测表征坍缩）。接入 W&B/TensorBoard。

**为什么值得做：**
- 将 ReVLA、Kachaev、D²PPO 的 post-hoc 发现变成 online 预警
- 从 NLP/CV 搬到 robotics 的技术路线清晰（structural probes → robotics-native probes）
- 实际效用：在训练过程中自动触发 early stopping / learning rate decay，而非事后才发现 backbone 被毁
- 底层基础设施（SAE-Lens, TransformerLens v3, netrep, repsim）已成熟，只需一个 thin adapter layer

### 方向 4：SRA 框架从 sim-real 推广到 cross-modality / cross-embodiment

**做什么：** 把 Lei et al. 的 SRA 分析（Wasserstein alignment + MLP discernibility + mixing ratio formula）应用到：(a) human-video + robot-data co-training；(b) cross-embodiment data mixing。测试 $w_n = N/(N+M)$ 和 $w_q = \sqrt{N/M}$ 是否在这些设定下同样有效。

**为什么值得做：**
- Lei et al. 的框架是 domain-agnostic 的（只需要 mixing ratio、domain label、latent），作者明确列出 cross-embodiment / cross-modality 是 future work
- SRA 解释了 ~50% 的 variance，是当前最有力的理论工具
- 与方向 2 自然互补：先测量跨 embodiment 的表征对齐状况，再用 SRA 框架指导 data mixing
- 实验设计清晰：在 OXE subset 上，用不同 embodiment 作为 source/target，sweep mixing ratio，同时 log Wasserstein + discernibility

### 方向 5：跨架构的 memorization 分析

**做什么：** 在 Octo (93M, diffusion policy) 和 Diffusion Policy (纯 diffusion, 无 VLM backbone) 上训练 SAE，使用 Swann et al. 的四指标分类器 + Grant et al. 的 causal ablation，测试 "大部分特征是记忆的轨迹" 这一发现是否跨架构成立。

**为什么值得做：**
- 这回答一个根本问题：memorization 是 behavior cloning 的固有问题，还是 VLM fine-tuning 的 artifact？
- 如果是固有问题 → 需要从 BC 本身出发（如 D²PPO 的 dispersive loss）
- 如果是 VLM fine-tuning artifact → Knowledge Insulation / LoRA 就是正确方向
- Octo 是 fully open-source (93M)，Diffusion Policy 有多个开源实现，实验可行

---

## 结语：为什么是现在？

这个方向之所以值得进入，核心原因有三：

1. **工具成熟度。** TransformerLens v3 支持 9000+ 模型，SAE-Lens 是成熟的 SAE 训练框架，netrep/repsim 提供 differentiable CKA/Procrustes。LLM interpretability 社区 3 年的工具积累现在可以直接搬到 VLA 上。

2. **机器人模型提供 LLM 没有的分析优势。** 物理 ground truth 可精确测量（不像"模型是否理解了句子"）；action→environment→observation 是闭合因果环路，activation steering 可以获得真实物理 counterfactual；不同 embodiment 是天然的 controlled variation；稀疏 reward 使 memorization vs generalization 的区分比文本领域清晰得多。

3. **timing。** VLA interpretability 刚过"existence proof"阶段（2024-2025 的零散论文），还没进入"systematic methodology"阶段。现有工作几乎全部集中在 OpenVLA 和 π0.5 两个模型上，方法论碎片化。第一个提出统一诊断框架的团队将占据定义性位置。
