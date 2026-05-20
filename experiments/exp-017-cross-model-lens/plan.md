# Exp-017: 跨模型 Lens 迁移

## 目的

测试一个模型训出的 probe 能否直接用在另一个模型上——
如果可以，说明不同架构学到了相似的特征方向（支持 Platonic Representation Hypothesis）。

利用 exp-007 已有的 5 个模型的 probe weights 和 cached hidden states，
做跨模型 probe 迁移实验。

## 方法

1. 从 exp-007 的 5 个模型中提取每个特征的 best probe weight W
2. 对维度相同的模型对（Pythia-2.8B ↔ Mamba-2.8B, 都是 d=2560），
   直接用模型 A 的 probe 在模型 B 的隐状态上测准确率
3. 对维度不同的模型对，用 Procrustes 对齐后再测
4. 计算"迁移率" = 跨模型 acc / 同模型 acc

## 数据

已有（exp-007 cached hidden states on SSD）：
- Pythia-1.4B: d=2048, 24 layers
- Pythia-2.8B: d=2560, 32 layers
- Gemma-2-2B: d=2304, 26 layers
- Mamba-2.8B: d=2560, 64 layers
- RWKV-3B: d=2560, 32 layers

维度匹配对：Pythia-2.8B ↔ Mamba-2.8B ↔ RWKV-3B（都是 d=2560）

## 节点

jiagpu4（CPU 即可，数据已在 SSD）
