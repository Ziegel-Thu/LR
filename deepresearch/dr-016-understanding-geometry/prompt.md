# DR-016：模型的"理解能力"在表征几何上有什么 signature？

## 动机

很多模型声称学到了某种"理解"——V-JEPA 2 声称物理理解、DreamerV3 声称世界动力学理解、Coconut 声称推理理解、CLIP 声称跨模态语义理解、OthelloGPT 被证明有棋盘状态的 world model。但"理解"这个词缺乏精确定义。

我们想从表征几何的角度问：**这些"理解"在表征空间中有没有共同的几何 signature？** 比如：低 ID 的 semantic plateau？特定的 hunchback 形状？线性可 probe 的概念方向？如果有 → 几何 signature 可以成为"理解能力"的客观指标。

## 任务

### 1. 哪些模型声称了"理解能力"？

列出所有声称学到某种理解/world model/物理直觉/因果推理的模型。包括但不限于：
- V-JEPA 2（物理理解、时空预测）
- I-JEPA（视觉场景理解）
- DreamerV3/V4/V5（世界动力学）
- Genie（无监督动作语义）
- OthelloGPT（棋盘 world model）
- Coconut / Huginn（推理理解）
- CLIP / ImageBind（跨模态语义理解）
- 大语言模型的"世界模型"声称（如 LLM as world model 相关论文）
- 其他你认为相关的

对每个：
- 声称了什么理解能力？
- 怎么评估的？（benchmark、probe、人工评判？）
- 有没有人分析过它的**表征几何**（ID profile、流形结构、线性度、manifold capacity）？
- 开源/可跑？规模？

### 2. 已有的"理解 × 几何"研究

有没有论文研究过以下问题？
- 理解更好的模型 → ID 更低？
- World model 的 latent space → 特殊流形结构？
- 物理理解 → 对应线性可 probe 的物理量（力、速度、位置）？
- "理解"和"记忆"在几何上有什么区别？
- 有没有用几何指标（ID、curvature、topology）来**预测**或**量化**理解能力的工作？

### 3. Valeriani 的 semantic plateau 是否是 universal signature？

Valeriani 2023 在蛋白质语言模型中发现了 semantic plateau（中间层 ID 低谷 = 语义处理区域）。
- 这个现象在其他声称"理解"的模型中出现吗？
- 有没有人在 world model / vision model / latent reasoning model 中测过 ID profile？
- 没测过的话，这是一个明确的实验空白。

### 4. "理解 vs 不理解"的对照研究

有没有这样的对照实验：
- 训练好的 vs 随机初始化的同架构模型，表征几何的差异？
- 训练充分的 vs 训练不足的，几何如何演变？
- "学到了 X 理解"的 vs "没学到"的（如不同训练数据/目标），几何差异？

### 5. 最值得做的实验方向

综合考虑新颖性、可行性、影响力，推荐 5 个具体实验方向：
- 哪些模型的"理解声称"最可测试？
- 用什么几何工具（ID、curvature、manifold capacity、persistent homology）？
- 预期发现什么？
- 正面/反面结果分别意味什么？
