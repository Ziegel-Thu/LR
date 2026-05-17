# DR-014: 有哪些值得做表征分析的 Latent Models？

## 动机

我们在研究神经网络的表征结构（probing、因果分析、几何/拓扑分析等）。传统研究对象主要是标准 Transformer LLM。但近年出现了大量拥有**有趣 latent space** 的模型架构——它们的隐空间可能有不同于标准 Transformer 的结构，为表征分析创造新的研究对象和机会。

我们想先了解全貌：有哪些这样的模型？

## 任务

### 1. 全面列出拥有有趣 latent space 的模型

不限于"推理"，包括但不限于：
- **Latent reasoning models**：Coconut、Pause Tokens、hidden CoT、continuous thought 等
- **Latent diffusion / 生成模型**：Stable Diffusion 的 latent space、视频生成模型等
- **World models**：Dreamer、JEPA、Genie 等学到世界模型的隐空间
- **Multimodal latent spaces**：CLIP、ImageBind、共享隐空间模型
- **Latent language models**：先编码到隐空间再解码的语言模型
- **State space models**：Mamba、RWKV 等（隐状态结构不同于 Transformer）
- **其他**：任何你认为 latent space 值得分析的模型

对每个模型：
- 论文、arXiv ID、年份
- 核心机制（一段话）
- **是否开源？权重/代码可用性？模型规模？**
- 隐空间的结构特点（维度、连续/离散、迭代/单次、是否有 residual stream 等）

### 2. 按 latent space 类型分类

- 哪些模型的隐空间是"被动的"（每层处理一次就往下传）vs "主动的"（在隐空间中迭代/推理）？
- 哪些是单模态 vs 多模态？
- 哪些的隐空间有已知的有趣结构（如纠缠/解纠缠、流形结构、对称性）？

### 3. 已有的表征分析工作

- 哪些模型已经被 probing / 因果分析 / SAE / 几何分析研究过？
- 哪些是**空白**（有模型但没人做过表征分析）？
- 已有研究发现了什么有趣的东西？

### 4. 最值得做表征分析的模型推荐

综合考虑：
- 隐空间的独特性（和标准 Transformer 有多不同？）
- 可操作性（开源？能跑？规模合理？）
- 空白程度（没人做过的优先）
- 表征分析的潜在新发现（能问出标准 Transformer 上问不了的新问题？）

推荐 top 5-10 个最值得我们做表征分析的模型。
