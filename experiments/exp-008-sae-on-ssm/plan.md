# Exp-008: SSM 上的 SAE

## 研究问题

Sparse Autoencoder 在非 Transformer 架构（Mamba）上能否找到 monosemantic 特征？Superposition 是 Transformer 特有的还是普遍现象？

## 动机

- 所有 SAE 工作都在 Transformer 上——Mamba/RWKV 上**零论文**
- 如果 SAE 在 SSM 上 work → superposition 是普遍的
- 如果 SAE 在 SSM 上 fail → superposition 是 Transformer 特有的
- 两种结果都回答一个根本性问题

## 假设

1. Mamba 的隐状态也存在 superposition（因为 d_model < 概念数量）
2. SAE 可以提取 monosemantic 特征，但性质可能不同于 Transformer
3. SSM 的递推结构可能产生更多"时序"特征

## Phase 1: Mamba-130M Pilot

### 配置
| 参数 | 值 |
|------|-----|
| 模型 | Mamba-130M (state-spaces/mamba-130m) |
| 训练数据 | Pile validation, 2M tokens |
| SAE 架构 | TopK, d_sae=4096, K=32 |
| 目标层 | 中间层（~layer 12/24）的 output hidden state |
| 训练 | Adam, lr=3e-4, 50K steps |
| 评估 | reconstruction MSE, dead feature %, feature activation sparsity |

### 注意事项
- Mamba hidden state 和 Transformer 不同：
  - 没有标准 residual stream
  - State 是 (B, L, D, N) 而非 (B, L, D)
  - 先做 output hidden state（最接近 Transformer residual stream）
- 需要适配 hook 提取 Mamba 中间层激活

### 输出
- Reconstruction quality vs Transformer SAE 对比
- Dead feature rate 对比
- Feature activation histogram 对比
- 手动检查 top 50 features 的语义

## Phase 2: 特征对比（Mamba SAE vs Pythia SAE）

| 分析 | 方法 |
|------|------|
| 特征重叠 | MMCS between Mamba/Pythia SAE dictionaries |
| 特征类型 | 分类: token-level / positional / semantic / syntactic / temporal |
| 因果效果 | 在 Mamba 上做 feature clamping |

### 核心问题
- Mamba 和 Pythia 的 SAE 特征有多少"共享概念"？
- Mamba 是否有 Transformer 没有的特征类型？

## Phase 3: 规模扩展（如果 Phase 1 成功）

| 模型 | d_model | SAE d_sae |
|------|---------|-----------|
| Mamba-130M | 768 | 4096 |
| Mamba-370M | 1024 | 8192 |
| Mamba-1.4B | 2048 | 16384 |

## 成功标准

- Phase 1 成功 = reconstruction quality 和 Transformer SAE 同量级
- Phase 2 有意义 = 至少一种系统性差异被发现
- SAE 在 Mamba 上完全失败 = 同样重要的结果

## 算力估计

- Phase 1: 激活缓存 ~2h + SAE 训练 ~4h = ~6 GPU-hours
- 可在本地 pilot (Mamba-130M 足够小)
