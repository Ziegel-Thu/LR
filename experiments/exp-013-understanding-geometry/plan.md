# Exp-013: 理解的几何学（Story A）

## 研究问题

声称"理解"了某个领域的模型，其表征几何有没有共同的 signature？

## 动机

- 很多模型声称学到了"理解"（物理/世界动力学/推理/语义），但"理解"缺乏精确定义
- Valeriani 2023 在蛋白质 LM 中发现了 semantic plateau（中间层 ID 低谷），但在其他"理解型模型"上未验证
- 随机/shuffle 训练的网络失去 hunchback / low-ID terminal layer → signature 是训练产物不是架构效应
- DR-016 确认：V-JEPA 2 / DreamerV3 / Coconut / Huginn 上零几何分析数据
- **exp-013 Phase 2 初步结果：10 模型均无 hunchback（ID 单调下降）→ hunchback 可能是 CNN 特有的，需要重新思考 signature 的定义**

## DR-018 补充发现

- **抽象概念探测几乎完全空白**（守恒量/对称性/反事实/规划），仅 Leela 国际象棋有 2-7 步前瞻证据
- DreamerV3 RSSM 从未被正式 probe 过——最大的低垂果实
- Coconut 的 "BFS 叠加" 被第三方复现否定——零 latent token 仍 96.6% 准确率
- Joseph 2026 在 V-JEPA 2 发现"物理涌现区"（编码器 1/3 深度），但**守恒量从未被 probe**
- OthelloGPT 策略概念（机动性、稳定边角）和多步前瞻几乎空白

## Phase 1: OthelloGPT 标定（ground truth 已知）

| 参数 | 值 |
|------|-----|
| 模型 | OthelloGPT（Nanda 2023 版本，开源） |
| 工具 | TwoNN ID, MLE ID (K=100), stable rank, probe accuracy (每层) |

- 画 ID profile → 有没有 hunchback / semantic plateau？
- Stable rank / effective rank / manifold capacity / curvature / persistent homology 全套
- 和 probe accuracy 叠加 → 低 ID 层是不是 probe 最好的层？
- 和 ablation effect 叠加 → 低 ID 层是不是因果最重要的层？
- 训练好的 vs 随机初始化的对照

这是 signature 的"标定"——在 ground truth 完全已知的模型上确定 signature 长什么样。

## Phase 2: 跨模型 ID-profile Atlas

对每个模型画 ID profile（每层/每迭代的 ID 曲线）：

| 模型 | 声称的理解 | 开源 | 规模 |
|------|-----------|------|------|
| OthelloGPT | 棋盘 world model | ✅ | 小 |
| Pythia-1.4B | 语言（对照，不声称"理解"） | ✅ | 1.4B |
| Mamba-1.4B | 语言（SSM 对照） | ✅ | 1.4B |
| DreamerV3 | 世界动力学 | ✅ | 中 |
| V-JEPA 2 | 物理理解 | ✅ | ViT |
| Coconut | 推理 | ✅ | GPT-2 scale |
| CLIP (ViT-B/32) | 跨模态语义 | ✅ | ViT |

每个模型：TwoNN ID + MLE ID + stable rank + effective rank + manifold capacity (Cohen 2020) + curvature + persistent homology，按层（或迭代步）画曲线。目标是每个模型一个完整的**几何 portrait**。

### 核心分析

- 有没有**共同 pattern**（如所有理解型模型都有 semantic plateau，语言模型对照没有）？
- 不同类型的"理解"有不同的 signature 吗？
- ID profile 形状能否**区分**有理解的 vs 没理解的？

## Phase 3: 训练 vs 随机对照

对每个模型：训练好的 vs 随机初始化的同架构模型，ID profile 对比。
- 共同变化 = architecture effect
- 只有训练后才有 = training/understanding effect

## Phase 4:（如果 Phase 2 有信号）Probe + Ablation 叠加

在 Phase 2 中发现 signature 的模型上：
- 低 ID 层的 probe accuracy 是否更高？
- 低 ID 层的 ablation effect 是否更大？
- 如果两者都是 → semantic plateau = encoding AND use 的甜点

## 成功标准

- 找到至少一个跨模型一致的几何 signature → "理解的几何定义"
- 或发现不同理解有不同 signature → "理解"需要按几何细分
- 或发现理解型 vs 语言模型对照没有几何差异 → signature 不存在也是重要发现

## 算力估计

- Phase 1: OthelloGPT 很小，~1 GPU-hour
- Phase 2: 7 个模型 × inference + ID 计算，~20 GPU-hours
- Phase 3: 同 Phase 2（随机模型不用训练，直接 inference）
- 节点：待定（等模型资产 checklist）
