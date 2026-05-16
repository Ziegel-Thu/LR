# 研究计划

## 研究方向

神经网络表征的数学结构——度量、几何、信息论、可解释性

## 当前阶段

方向探索完成，本地 pilot 已验证信号。准备上服务器做完整实验。

## 已完成

### 文献与方向调研
- DR-001: 全景地图（8 个子方向 + 10 个开放问题）
- DR-002: 追加 15 个开放问题
- 精读 3 篇核心论文: Klabunde (similarity survey), Ehrlich (PID), Braun (function-representation dissociation)

### 本地 Pilot 实验
- **exp-001 SAE 可识别性**: 信号弱，小模型不足以下结论
- **exp-002 量化与特征恢复**: ✅ 发现频率分层效应（稀有特征先崩溃，4-bit low=42% vs high=95%）
- **exp-003 跨架构柏拉图收敛**: ✅ 信号强（Transformer/Mamba/RWKV mutual-kNN z>350），U 形深度曲线
- **exp-004 ID→泛化因果链**: 运行中

## 服务器实验计划

按优先级排序。需要 GPU 服务器（ssh jiagpu45678）。

### 优先级 1: 跨架构柏拉图收敛 (exp-003 扩展)

**基于**: DR-005 方案，exp-003 pilot 已验证信号
**核心问题**: 不同架构随 scale 增大是否收敛到同一表征？
**为什么优先**: pilot 信号最强，且有完美受控实验组（Pythia/Mamba-1/RWKV-4 共享 The Pile 数据）

| 参数 | 值 |
|------|-----|
| 模型 | Pythia {410M,1.4B,2.8B,6.9B}, Mamba-1 {370M,1.4B,2.8B}, RWKV-4 {430M,1.5B,3B,7B} |
| 度量 | mutual k-NN (primary), CKA, Procrustes + permutation null calibration |
| 分析 | scaling law 拟合 s(N) = s∞ - a·N^(-β)；逐层深度曲线 |
| 计算 | ~2000-3000 A100-hours, 纯推理 |
| 卡数 | 1×A40 per model, 可并行 |

### 优先级 2: 量化与 SAE 特征恢复 (exp-002 扩展)

**基于**: DR-004 方案，exp-002 pilot 已验证频率分层效应
**核心问题**: 权重量化下是否存在 superposition 相变？稀有特征是否先崩溃？

| 参数 | 值 |
|------|-----|
| 模型 | Gemma-2 2B/9B + Gemma Scope SAE（预训练好的，不用自己训）|
| 量化 | HQQ, FP16→INT8→INT4→INT3→INT2 |
| 分析 | 频率分层恢复率，Hill 系数拟合，有限尺寸标度（Pythia 410M→6.9B）|
| 计算 | ~510 A100-hours, ~5 天 on 4×A100 |

### 优先级 3: RLHF 表征漂移 (新实验)

**基于**: DR-007 方案
**核心问题**: RLHF 对内部表征的改变集中在哪里？能否预测安全性退化？

| 参数 | 值 |
|------|-----|
| 模型 | Tulu-3 / OLMo-2 / Zephyr (各有 base/SFT/DPO 三阶段) |
| 分析 | 逐层 metric ensemble, ΔW SVD, 漂移方向 vs refusal direction |
| 计算 | 12-14 周 on 4×A100, 纯推理+分析 |

### 优先级 4: ID→泛化因果链 (exp-004 扩展)

**基于**: DR-006 方案
**核心问题**: 内蕴维数的降低是否因果地导致泛化提升？

| 参数 | 值 |
|------|-----|
| 设计 | 7 臂干预实验（4 treatment + 3 sham control）|
| 模型 | ResNet-18 / ViT-S on CIFAR-10/100, Tiny-ImageNet |
| 计算 | ~3 天 on 4×A100（本地可做 CIFAR-10 部分）|

### 优先级 5: SAE 可识别性 (exp-001 扩展)

**基于**: DR-003 方案
**核心问题**: 不同种子训练的 SAE 能否收敛到相同特征？

| 参数 | 值 |
|------|-----|
| 模型 | Llama-3 8B, Gemma-2 2B, Pythia-1.4B |
| 计算 | ~6000-8000 A100-hours |
