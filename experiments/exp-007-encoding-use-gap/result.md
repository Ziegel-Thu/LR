# Exp-007: Encoding ≠ Use — Full Run 结果

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-1.4B (`EleutherAI/pythia-1.4b-deduped`), 24 layers |
| 数据 | WikiText-103 validation, 176,213 tokens |
| Features | 15 个（11 有效 + 4 因不平衡跳过） |
| Probe | PyTorch linear probe (GPU), 10 epochs, lr=1e-3 |
| 干预 | AMNESIC-style: 对 probe 方向做零化投影, 测 Δloss |
| Ghost 判定 | probe acc > 0.7 AND |Δloss| < 0.01 |
| 并行 | 15 features × 8 GPU, 总耗时 ~20min |

## 结果

### Ghost Ratio

| 指标 | 值 |
|------|-----|
| **Ghost ratio** | **70.8%** (187/264) |
| Encoded (probe acc > 0.7) | 264 个 (layer, feature) 对 |
| Ghost (encoded but unused) | 187 个 |

### 逐 Feature 结果

| Feature | Best Probe Acc | Max |Δloss| | 类型 |
|---------|---------------|--------------|------|
| is_capitalized | 1.000 | 0.031 | Ghost-heavy |
| is_numeric | 1.000 | 0.087 | Mixed |
| is_punctuation | 1.000 | 0.037 | Ghost-heavy |
| is_stopword | 1.000 | 0.080 | Mixed |
| is_title_case | 1.000 | 0.058 | Mixed |
| is_subword | 1.000 | 0.018 | Ghost-heavy |
| starts_with_space | 1.000 | 0.015 | Ghost-heavy |
| is_short | 0.999 | 0.011 | Ghost-heavy |
| is_plural | 0.996 | 0.089 | Mixed |
| is_rare | 0.972 | 0.025 | Ghost-heavy |
| is_high_freq | 0.954 | 0.012 | Ghost-heavy |

### 跳过的 Features（label 不平衡 < 2%）

- ends_with_ing, ends_with_tion, has_prefix, is_non_english

## 关键发现

### 1. Ghost ratio = 71% — 远超预期 ✅

71% 的"高准确率 probe"对应的信息被 ablation 后模型表现无变化。这意味着：
- **大量 probing 研究高估了信息的因果重要性**
- Probe 能解码 ≠ 模型在用
- 验证了 Braun 2025 的理论预测（在 deep linear networks 上推导），首次在真实 LLM 上量化

### 2. 几乎所有 feature 都是完美可 probe 的

8/11 features 的 probe accuracy = 1.000，剩余 3 个也 > 0.95。Pythia-1.4B 的表征空间编码了大量 token-level 信息，但大部分是"ghost information"。

### 3. 少数 feature 有较大因果效应

- `is_plural` (0.089), `is_numeric` (0.087), `is_stopword` (0.080) 的 max |Δloss| 相对较高
- 但即使这些 feature，ghost ratio 仍然高（大多数层的 Δloss 仍然很小）
- 因果效应主要集中在少数关键层

### 4. 分层模式

从 layer profiles 图可见：
- Probe accuracy 在所有层都很高（>0.9）
- Δloss 仅在少数层有 spike，大部分层接近 0
- 模式：early layers 有轻微因果效应，middle layers 几乎无效应，late layers 略有回升

## 局限

- 当前 features 偏 token-level（形态学特征），缺少语义/句法特征
- Ghost threshold = 0.01 的选择有一定主观性
- AMNESIC 投影只 ablate 1 个方向，可能低估了 multi-direction encoding 的因果效应
- 训练数据来自 WikiText validation（域内），可能高估 probe accuracy

## 生成的图表

- `encoding_vs_use.png` — 散点图: probe accuracy vs Δloss
- `layer_profiles.png` — 逐层 encoding vs use 对比

## 下一步

- [ ] 添加语义特征（NER、情感、主题）需要 NLP 标注工具
- [ ] 用 DAS (Distributed Alignment Search) 替代单方向 ablation
- [ ] 在 Pythia-2.8B 上重复验证
- [ ] 与 exp-008 的 SAE features 做交叉分析
