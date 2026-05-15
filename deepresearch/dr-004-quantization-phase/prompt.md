# DR-004: 量化临界 bit-width 与 superposition 相变——实验方案设计

## 背景

Superposition 是大型神经网络的核心现象：模型在 d 维残差流中编码远多于 d 个特征，通过近似正交方向的重叠实现。Elhage et al. (arXiv:2209.10652) 在 toy model 中完整刻画了 superposition 的几何结构，发现特征按均匀多面体排列并存在相变。

SAE 的成功依赖于 superposition 结构的可线性分离性——即高维稀疏编码可以通过线性投影恢复。但当模型被量化（降低数值精度）时，这种精细的几何结构可能被破坏。

**核心假说**：存在一个临界 bit-width b*，低于它时 superposition 结构坍塌——原本几何上可分离的特征变得不可区分，SAE 发现的特征不再可线性恢复。这个转变可能是一个相变（sharp transition）而非渐变。

相关工作：
- 量化对模型性能的影响已有广泛研究（GPTQ, arXiv:2210.17323; AWQ, arXiv:2306.00978; QuIP, arXiv:2307.13304），但几乎没有工作从表征几何/superposition 的角度分析量化
- arXiv:2307.02973 和 arXiv:2510.22058 经验性地比较了剪枝 vs 量化对表征的影响
- Michaud et al. (arXiv:2303.13506) 的 Quantization Model 将 scaling 与 emergence 统一为"按频率排序的离散 quanta"，但这里的"quantization"是概念性的，不是数值精度的量化
- 表征内蕴维数（Ansuini et al., arXiv:1905.12784）可能是一个有用的 order parameter

## 请设计的内容

请设计实验验证"量化临界 bit-width 相变"假说：

1. **模型选择**：选什么开源 LLM？考虑需要同时有 FP16 权重和良好的 SAE 训练基础设施

2. **量化方案**：如何系统地做 FP16 → INT8 → INT4 → INT3 → INT2 的逐级量化？用什么量化方法？

3. **度量 superposition 状态**：
   - 如何度量"superposition 是否坍塌"？（SAE 特征恢复率？激活空间的有效秩？内蕴维数？特征间余弦相似度分布？）
   - 如何度量"SAE 特征的线性可恢复性"？

4. **相变 vs 渐变**：如何判断转变是 sharp 还是 smooth？需要什么 order parameter 和统计检验？

5. **Baseline 和对照**：需要哪些对照实验？

6. **成功标准**：什么结果支持/反驳假说？预期的实验结果是什么？

## 实验条件

4×A100 或 8×A40 GPU 服务器。
