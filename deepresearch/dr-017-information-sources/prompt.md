# DR-017：Probe 读到的信息来自哪里——统计分解方法

## 动机

当 probe 在某层读到 90% 准确率时，这个信号可能来自：
- (a) 输入 bleed-through：词本身携带的特征
- (b) 训练目标结构：next-token 统计和概念相关
- (c) 模型内部计算：跨 token 整合产生的新信息

目前没人做过系统的来源分解。我们想知道有没有**统计/信息论方法**可以量化每个来源的贡献。

## 任务

### 1. 已有的信息来源分析方法

有没有人研究过以下问题？列出相关论文和方法：
- Probe accuracy 的来源分解（输入 vs 模型贡献）
- 随机网络也能高 CKA（Cui 2022）——后续有人跟进或解释吗？
- Encoding Probe（Gubian 2025，反向 probe）的统计框架是什么？
- 用 MI 或条件 MI 分离输入贡献和模型贡献

### 2. 统计/信息论工具

有没有可以用的统计/信息论框架来分解 probe 信号？例如（不限于此，请补充）：
- 条件互信息 I(representation; concept | input) 量化"模型贡献"？
- PID 分解：probe 信号中 redundant（从输入就有）vs unique（模型创造的）？
- 部分相关 / 中介分析
- 随机标签 / shuffled baseline 的统计检验
- 其他适用的统计或信息论方法

### 3. 不同训练目标的对比研究

同架构不同训练目标（next-token / masked LM / contrastive / denoising diffusion）下：
- Probe 行为有什么系统性差异？
- 有没有人做过这样的控制变量实验？

### 4. 实验设计建议

如果我们要做一个"probe 信号来源分解"的实验：
- 最合适的统计方法是什么？
- 用什么模型和任务最能区分三个来源？
- 需要什么 baseline / control？
- 已有的工具（如 nninfo, MINE, variational bounds）哪些可以直接用？
