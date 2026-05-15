# DR-005: 跨架构柏拉图收敛——实验方案设计

## 背景

Huh et al. (arXiv:2405.07987, ICML 2024 Position Track) 提出了"柏拉图表征假说"：随着模型规模增大，不同架构、不同模态（视觉/语言）的模型趋向学习同一个"共同的统计现实"的表征。他们综合 20+ 项证据论证视觉与语言模型的距离结构随 scale 渐近对齐。

然而，这一假说几乎只在 Transformer/ViT 系列之间测试过。2024-2025 年，结构性不同的架构已经成熟：
- **Mamba** (Gu & Dao, arXiv:2312.00752)：基于选择性状态空间模型（selective SSM），无 attention 机制
- **RWKV** (Peng et al.)：线性 attention 变体
- **Jamba**、**Zamba** 等混合架构

关键问题：这些架构有本质不同的归纳偏置（attention 的全局键值检索 vs SSM 的递归状态压缩）。如果柏拉图假说成立，即使归纳偏置不同，足够大的模型也应该收敛到相同表征。如果不成立，就能精确定位**哪种归纳偏置差异**阻止了收敛。

相关工作：
- arXiv:2409.05395 显示 Mamba-VLM 在细粒度检索任务上落后于 Transformer，但在粗粒度任务上匹配
- arXiv:2406.16722 综述了 SSM → Transformer 的架构过渡
- Tjandrasuwita et al. (arXiv:2502.16282) 发现柏拉图对齐只在低模态异质性和高冗余条件下成立
- arXiv:2602.14486 ("Aristotelian View") 部分反驳了柏拉图假说
- 表征比较工具：CKA (Kornblith, arXiv:1905.00414)、mutual k-NN、Procrustes shape metrics (Williams, arXiv:2110.14739)、ReSi benchmark (github.com/mklabunde/resi)

## 请设计的内容

请设计实验测试跨架构表征收敛：

1. **模型选择**：有哪些可直接使用的开源 checkpoint？需要覆盖：
   - 至少 2 种架构（Transformer vs Mamba/RWKV）
   - 每种架构至少 2-3 个规模（如 1B, 3B, 7B）
   - 尽量控制训练数据一致

2. **表征比较方法**：
   - 用什么度量？各自的优缺点？
   - 在哪些层比较？（如何对齐不同架构的"对应层"？）
   - 如何处理不同架构的 hidden dimension 不同？

3. **收敛速率**：如何量化"收敛程度"随 scale 的变化？能否拟合 scaling law？

4. **混杂因素控制**：训练数据不同、tokenizer 不同、训练步数不同等如何处理？

5. **成功标准**：什么结果支持/反驳假说？如何区分"部分收敛"和"完全收敛"？

## 实验条件

4×A100 或 8×A40 GPU 服务器，不训练模型，只做推理和分析。
