# Exp-020: Steering Vector OOD Boundary Detection

## 研究问题

Activation steering（向 hidden states 添加方向向量）在小幅度时有效，但幅度过大时模型行为崩坏。这个"OOD boundary"是否可以系统化地刻画？

## 动机

- Tan 2024 发现了 steering 的 OOD 边界现象但没有系统研究
- Exp-007 已在 Pythia-2.8B 上识别出多种 probe 方向（ghost vs used features）
- 如果 ghost features 的 OOD boundary 与 used features 不同 → 说明 OOD boundary 与因果重要性有关
- 可以建立"steering 安全范围"的定量理论

## 假设

1. 所有 steering 方向都存在明确的 OOD boundary（KL 突增点）
2. Ghost features 的 OOD boundary 更窄（模型对不使用的信息的扰动更敏感）
3. OOD boundary 可能与 probe 方向的几何性质（如与主成分的夹角）相关

## 方法

### 基本流程

1. 加载 Pythia-2.8B 和 exp-007 的 cached hidden states
2. 对每个 feature，在 best probe layer 重新训练 linear probe 获取方向向量 W
3. 对 α ∈ [-5, +5]（21 个点）：
   - Hook best layer，向 residual stream 添加 α × W_norm
   - 在 WikiText validation（100 句子）上计算：
     - (a) 语言模型 loss
     - (b) 输出分布与 unsteered 的 KL 散度
4. 绘制 loss vs α 和 KL vs α 曲线
5. 自动检测 OOD boundary：KL 散度首次超过阈值的 α 值

### 对照实验

- 随机方向（从同维高斯采样正规化）作为 baseline
- 比较 ghost vs used features 的 boundary 差异

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-2.8B (`EleutherAI/pythia-2.8b-deduped`) |
| 数据 | WikiText-103 validation, 100 sentences |
| Features | exp-007 的 11 个有效 features + 3 个随机方向 |
| α 范围 | [-5, +5], 21 个等间距点 |
| 节点 | jiagpu4 GPU 2 |
| 依赖 | exp-007 cached hiddens (`/nvmessd/lifanhong/LR-env/exp007_28b/`) |

## 成功标准

- 能够为每个 feature 画出 loss-α 和 KL-α 曲线
- 能识别 OOD boundary（KL spike）
- Ghost vs used features 的 boundary 有可观测差异
