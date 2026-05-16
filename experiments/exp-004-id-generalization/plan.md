# Exp-004: 内蕴维数与泛化因果链 Pilot

## 目的

建立 ID（Intrinsic Dimension）与泛化性能之间的因果关系（而非仅相关性）。
本地 pilot 先做 Stage 1（相关性 baseline）和 Stage 2（TwoNN 正则化干预）。

## 背景

- Ansuini et al. (arXiv:1905.12784) 发现表征 ID 沿网络深度呈 hunchback 曲线，最后一层 ID 与测试精度相关
- 相关性 ≠ 因果性。Saxe et al. (ICLR 2018) 已证明 Information Bottleneck 的"压缩→泛化"叙事在 ReLU 下不成立
- DR-006 设计了 7 臂干预实验，核心逻辑：如果降低 ID 的同时匹配所有混杂因素（范数、秩、对齐），泛化仍提升 → ID 是因；如果 sham 对照也提升 → ID 只是果

## 假设

1. 不同 weight decay 强度训练的 ResNet-18 在 CIFAR-10 上，最后一层 ID 与测试精度负相关 (r < -0.7)
2. 在训练 loss 中加 TwoNN 正则化能降低最后一层 ID
3. ID 降低伴随泛化提升（如果因果假说成立）

## Stage 1: 相关性 Baseline

### 配置
| 参数 | 值 |
|------|-----|
| 模型 | ResNet-18 |
| 数据集 | CIFAR-10 |
| 训练 | SGD, lr=0.1, momentum=0.9, cosine schedule, 100 epochs |
| Weight decay | {0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2} (7 个级别) |
| 种子 | 3 per weight decay |
| ID 估计 | TwoNN on 4096-sample probe set, penultimate features |
| 设备 | MPS (M4 Max) |
| 预计时间 | 7×3 = 21 runs × ~5 min each ≈ 2 小时 |

### 度量
- 测试精度
- 最后一层 TwoNN ID
- ‖θ‖₂ (参数范数)
- Stable rank

### 成功标准
- Pearson r(ID, test_acc) ≤ -0.7 → 进入 Stage 2
- 如果 r > -0.5 → 相关性不够强，需要重新考虑

## Stage 2: TwoNN 正则化干预（如果 Stage 1 通过）

### 配置
| 参数 | 值 |
|------|-----|
| 基线 | ResNet-18, CIFAR-10, weight_decay=5e-4 |
| Treatment | L = L_CE + λ · TwoNN_ID(penultimate_features) |
| λ | {0, 0.01, 0.03, 0.1, 0.3, 1.0} (6 个 dose) |
| Sham 对照 | 范数匹配 weight decay（匹配 Treatment 的 ‖θ‖₂ 轨迹但无 ID 惩罚）|
| 种子 | 5 per condition |
| 预计时间 | (6+6)×5 = 60 runs × ~5 min ≈ 5 小时 |

### 成功标准
- Treatment 降低 ID 且提升泛化，而 Sham 不降低 ID → 因果链初步成立
- Treatment 和 Sham 都提升泛化 → ID 不是因，只是果
