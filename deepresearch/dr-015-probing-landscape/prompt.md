# DR-015：Probing 方法的全景调研

## 动机

Probing 是表征分析最基础的工具族。我们想全面了解：probing 有哪些种类？各自做了什么？在不同架构上的研究现状？方法论本身有哪些 open questions？

## 任务

### 1. Probing 方法的完整分类

请列出所有已知的 probing 及相关方法，包括但不限于以下已知方法（请在此基础上**补充我们不知道的**）：

**经典 Probing**：
- Linear probe（Alain & Bengio 2016）
- Structural probe（Hewitt & Manning 2019）
- MDL probe（Voita & Titov 2020）
- Control task（Hewitt & Liang 2019）
- Edge probing（Tenney+ 2019）
- Information-theoretic probing（Pimentel+ 2020）
- SentEval probing suite（Conneau+ 2018）

**因果/干预型 Probing**：
- AMNESIC probing（Elazar+ 2021）
- AlterRep
- Causal Basis Extraction / CBE（Belrose+ 2023）
- DAS / Distributed Alignment Search（Geiger+ 2023）

**Lens 方法**：
- Logit lens（nostalgebraist 2020）
- Tuned lens（Belrose+ 2023）
- Patchscopes（Ghandeharioun+ 2024）

**方向提取/操控**：
- Difference-in-means steering vector（Turner+ 2023, ActAdd）
- INLP / Concept erasure（Ravfogel+ 2020）
- RepE reading vectors（Zou+ 2023）
- Causal inner product（Park+ 2024）
- Refusal direction probing（Sharma+ 2024）
- Contrastive Activation Addition / CAA（Rimsky+ 2024）

**评估/理论框架**：
- AxBench（Wu+ 2025，detection vs steering 评估）
- Encoding ≠ use dissociation theory（Braun+ 2025）

**不在上述类别中的方法**——请补充我们遗漏的。

对每种方法：定义、核心论文、arXiv ID、做了什么、在哪些架构上验证过、核心发现。

### 2. 各架构上的 probing 研究现状

对以下架构分别列出已有 probing 工作：
- CNN（ResNet, VGG, ConvNeXt 等）
- Encoder Transformer（BERT, RoBERTa 等）
- Decoder-only Transformer LM（GPT, LLaMA, Pythia 等）
- Vision Transformer（ViT, DeiT, MAE 等）
- SSM（Mamba, RWKV, S4 等）
- Hybrid SSM-Transformer（Jamba, RecurrentGemma 等）
- Multimodal（CLIP, LLaVA, Chameleon 等）
- Diffusion models（Stable Diffusion, DiT 等）
- World models（Dreamer, JEPA 等）
- Latent reasoning models（Coconut, Huginn 等）

### 3. Probing 方法论的跨架构比较

有没有论文**在多种架构上系统比较**过 probing 的行为？发现了什么？
- Linear probe accuracy 在不同架构上有什么系统性差异？
- Encoding ≠ use 问题是否在所有架构上都存在？
- Probe 找到的方向在不同架构间是否一致？

### 4. Probing 方法论的 open questions

目前有哪些关于 probing 本身的理论/方法论问题未解决？

### 5. 最值得做的跨架构 probing 研究方向

什么样的 probing 研究能产出**关于方法论的 universal 结论**（而不只是"模型 X 编码了特征 Y"）？
