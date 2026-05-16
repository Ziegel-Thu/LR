# Exp-003 Phase 1: 2.8B Scale — 跨架构柏拉图收敛

## 目的

在 2.8B 规模上测量 Transformer/Mamba/RWKV 的表征相似性，与 pilot（160M）对比看相似度是否随 scale 增大。

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-2.8B, Mamba-2.8B, RWKV-4-3B |
| 训练数据 | 全部 The Pile 300B tokens |
| Tokenizer | GPT-NeoX (共享) |
| stimuli | 5000 sentences (WikiText-103 validation, len>50) |
| 序列长度 | 256 tokens |
| 表征 | 每层 residual stream, mean-pooled |
| 度量 | mutual k-NN (k=10), linear CKA |
| null calibration | 100 permutations |
| 硬件 | 1×A40 (串行 3 模型) |
| 预计时间 | ~4 GPU-hours |

## 与 pilot 的对比

| | Pilot (160M) | Phase 1 (2.8B) |
|---|---|---|
| 模型 | Pythia-160M, Mamba-130M, RWKV-4-169M | Pythia-2.8B, Mamba-2.8B, RWKV-4-3B |
| d_model | 768 | 2560 |
| Layers | 12-25 | 32-64 |

## 成功标准

- mutual-kNN 在 2.8B 上 ≥ pilot (0.4-0.75) → scale 不降低收敛
- mutual-kNN 在 2.8B 上 > pilot → 支持 PRH（收敛随 scale 增强）
- mutual-kNN 在 2.8B 上 < pilot → 反对 PRH
