# Exp-004 Stage 1 结果：ID 与泛化相关性 baseline

## 配置

- 模型：ResNet-18 on CIFAR-10
- 训练：SGD, lr=0.1, momentum=0.9, cosine schedule, 50 epochs
- Weight decay：0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
- Seeds：3 per weight decay，共 21 runs
- 运行方式：jiagpu5 上 4 shards 并行，每个 shard 使用 1 张 A40
- ID 估计：TwoNN on 4096 test samples, penultimate features

## 汇总结果

| Weight decay | Test acc | TwoNN ID | Param norm | Stable rank |
|---:|---:|---:|---:|---:|
| 0 | 0.9193 | 18.2 | 137.7 | 8.8 |
| 1e-5 | 0.9174 | 18.0 | 128.3 | 8.9 |
| 1e-4 | 0.9317 | 14.9 | 68.5 | 8.9 |
| 5e-4 | 0.9452 | 10.2 | 30.7 | 8.9 |
| 1e-3 | 0.9436 | 9.4 | 22.8 | 8.9 |
| 5e-3 | 0.9100 | 7.6 | 11.1 | 8.7 |
| 1e-2 | 0.8501 | 6.4 | 8.0 | 8.6 |

整体相关：

- Pearson r(TwoNN ID, test acc) = 0.365, p = 0.104
- Pearson r(stable rank, test acc) = 0.935, p = 5.67e-10

## 解读

Stage 1 没有支持原假设“最后层 ID 越低，泛化越好”。Weight decay 确实单调降低 TwoNN ID 和参数范数，但 accuracy 呈倒 U 型：适中 weight decay（5e-4 到 1e-3）最好，过强 weight decay 继续降低 ID 时 accuracy 明显下降。

这说明 ID 下降本身不是充分的泛化改进信号，至少在这个 ResNet-18/CIFAR-10/weight-decay 干预下，ID 更像是正则强度或表示退化的伴随量。相比之下，stable rank 与 accuracy 的相关性更强，后续如果继续 Stage 2，需要加入 stable rank / norm-matched 对照，避免把其它容量指标误判为 ID 因果效应。

## 文件

- Combined results: `results_stage1.json`
- Raw shard results: `results_stage1_shard0.json` ... `results_stage1_shard3.json`
- Combined figure: `figures/stage1_correlation.png`
