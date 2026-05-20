# Exp-017: Cross-Model Lens Transfer — Results

## 配置

| 参数 | 值 |
|------|-----|
| 模型对 | Pythia-2.8B ↔ RWKV-3B（都是 d=2560, 32 layers）|
| 数据 | exp-007 cached hidden states (WikiText-103, 176K tokens) |
| Probe | sklearn LogisticRegression (max_iter=2000) |
| 特征 | 11 个（同 exp-007 有效特征） |
| 层匹配 | 使用 exp-007 best probe layer |
| 节点 | jiagpu4 CPU |

## 核心结果

### Aggregate Transfer Matrix

**Transfer Ratio (跨模型 acc / 同模型 acc)**:

|  | → Pythia-2.8B | → RWKV-3B |
|--|--------------|----------|
| Pythia-2.8B → | 1.000 | **0.545** |
| RWKV-3B → | **0.562** | 1.000 |

**Raw Accuracy**:

|  | → Pythia-2.8B | → RWKV-3B |
|--|--------------|----------|
| Pythia-2.8B → | 0.995 | 0.544 |
| RWKV-3B → | 0.560 | 0.995 |

### 逐特征迁移率

| Feature | Avg Transfer Ratio | Range | Same-model Acc |
|---------|-------------------|-------|---------------|
| starts_with_space | **0.788** | [0.71, 0.87] | 1.000 |
| is_punctuation | **0.771** | [0.71, 0.83] | 1.000 |
| is_numeric | **0.741** | [0.52, 0.97] | 1.000 |
| is_subword | **0.718** | [0.57, 0.87] | 1.000 |
| is_short | 0.625 | [0.63, 0.63] | 1.000 |
| is_plural | 0.619 | [0.37, 0.86] | 0.999 |
| is_high_freq | 0.542 | [0.39, 0.70] | 0.979 |
| is_stopword | 0.529 | [0.32, 0.74] | 1.000 |
| is_capitalized | 0.399 | [0.17, 0.63] | 1.000 |
| is_title_case | 0.198 | [0.14, 0.26] | 1.000 |
| is_rare | 0.154 | [0.03, 0.28] | 0.971 |

## 关键发现

1. **部分 Platonic 收敛**：平均迁移率 ~55%，远好于随机（~50% for binary）但远不完美
   - Pythia (Transformer) 和 RWKV (RNN) 学到了**部分相似**的特征方向

2. **特征类型决定迁移性**：
   - **高迁移** (~70-80%): 表面特征（空格、标点、数字、subword）
   - **中迁移** (~50-60%): 词汇特征（短词、复数、高频、停用词）
   - **低迁移** (~15-40%): 大小写、title case、稀有词
   
3. **表面特征跨架构更一致**：`starts_with_space` (79%) 和 `is_punctuation` (77%) 迁移最好，
   这些是最"底层"的特征，可能因为不同架构被迫用相似方式编码它们

4. **语义相关特征迁移差**：`is_rare` (15%) 和 `is_title_case` (20%) 几乎不迁移，
   说明不同架构用完全不同的方向编码"稀有性"和"标题格式"

5. **迁移的不对称性**：多数特征 RWKV→Pythia 优于 Pythia→RWKV
   - is_plural: RWKV→Pythia 0.86 vs Pythia→RWKV 0.37
   - 可能因为 RWKV 的浅层编码方向更"通用"

## 意义

- **不支持强 Platonic Representation Hypothesis**：不同架构不会收敛到相同的线性特征方向
- **支持弱 PRH**：表面/底层特征有部分跨架构一致性
- **架构特异性在语义层面最强**：越高层的概念，架构间差异越大

## 下一步

- [ ] 加入 Mamba-2.8B（需要在同节点有 cache，108GB）
- [ ] 用 Procrustes 对齐后重测（消除旋转自由度）
- [ ] 测 Pythia-1.4B ↔ Pythia-2.8B（同架构不同规模）
