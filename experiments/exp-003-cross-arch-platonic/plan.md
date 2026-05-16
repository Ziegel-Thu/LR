# Exp-003: 跨架构柏拉图收敛 Pilot

## 目的

测试 Transformer / Mamba / RWKV 三种架构在小规模下的表征相似性，为服务器上的完整实验提供预备数据。

## 假设

1. 如果 PRH 成立，三种架构的 mutual k-NN 应显著高于 permutation null
2. 相似性可能集中在中间层（语义层），早/晚层可能发散
3. 小模型上效应可能弱，但方向应一致

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-160M, Mamba-130M, RWKV-4-169M |
| 训练数据 | 全部 The Pile 300B tokens |
| Tokenizer | 全部 GPT-NeoX |
| stimuli | 5000 sentences (WikiText-103 validation) |
| 序列长度 | 256 tokens |
| 表征提取 | 每层 residual stream, mean-pooled over tokens |
| 度量 | mutual k-NN (k=10), linear CKA, 均含 permutation null |
| null permutations | 100 |
| 设备 | CPU (模型够小) |
| 预计时间 | ~15 分钟 |
| 峰值内存 | ~2 GB |
