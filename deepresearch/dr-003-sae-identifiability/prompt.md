# DR-003: SAE 可识别性——实验方案设计

## 背景

Sparse Autoencoders (SAE) 已成为 mechanistic interpretability 的核心工具。其基本思路是：将模型的残差流 h ∈ ℝ^d 分解为稀疏的高维特征组合 h ≈ Wdec · z，其中 z ∈ ℝ^m（m >> d）是稀疏激活向量，每个维度对应一个可解释特征。

Anthropic 的 "Scaling Monosemanticity"（Templeton et al., 2024）在 Claude 3 Sonnet 上训练了 34M-latent 的 SAE，提取出大量可解释特征。Cunningham et al. (arXiv:2309.08600) 和 Bricken et al. (Anthropic 2023) 此前在较小模型上验证了 SAE 的有效性。

然而，Paulo & Belrose (arXiv:2501.16615, 2025) 发现了一个根本性问题：**在 Llama 3 8B 上用不同随机种子训练的 131K-latent SAE，仅约 30% 的特征是跨种子共享的**。这个结果跨三个 LLM（Llama 3 8B、Gemma 2 2B、Pythia 1.4B）、两个数据集、多种 SAE 架构（standard、TopK、Gated）都稳健成立。

这意味着 SAE 发现的"特征"可能不是模型的本征对象，而部分取决于训练的随机性。如果特征不可识别，整个 mechanistic interpretability 框架需要重新审视。

相关工作：
- 经典稀疏字典学习理论（Spielman et al., 2012）在"互不相干 + 稀疏"条件下给出可识别性保证，但 LLM 的特征分布偏离这些假设
- Matryoshka SAE (arXiv:2503.17547) 尝试通过嵌套结构解决 feature splitting
- SAEBench (arXiv:2503.09532) 提供了标准化的 SAE 评估框架
- Engels et al. (arXiv:2405.14860) 发现部分特征本质上是非线性流形（如圆形），不可能被线性 SAE 唯一恢复

## 请设计的内容

请设计一组系统实验来研究 SAE 可识别性问题：

1. **复现实验**：复现 Paulo & Belrose 结果的最小设置是什么？（模型、层、SAE 配置、评估指标、计算量估算）

2. **消融实验**：哪些因素可能影响可识别性？设计对以下因素的消融：
   - 稀疏度（L1 惩罚强度 / TopK 中的 k）
   - 字典大小（expansion factor）
   - 训练数据量
   - 模型规模和层的位置
   - SAE 架构选择

3. **改进策略**：是否存在提高可识别性的训练策略？（如正则化、初始化、对比学习、共训练）设计验证实验

4. **度量方法**：如何定义和度量"特征共享率"？有哪些可选指标？各自的优缺点？

5. **成功标准**：什么结果算有意义的发现？正面/负面结果分别意味着什么？

## 实验条件

4×A100 或 8×A40 GPU 服务器，本地 M4 Max 32G Mac 可做预实验。
