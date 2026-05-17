# Exp-005: Additivity Tax — IG vs SHAP 可辨识性 Tradeoff

## 目的

验证 formalization_v3 §5.5 的核心发现：**additivity 而非 completeness 是不可能性的根源**。

## 假设

- IG（满足 completeness，不满足 additivity）能区分不同局部行为
- SHAP（满足 completeness + additivity）无法区分
- 混合 Φ^α = (1-α)·IG + α·SHAP 存在 critical α*，在此处 identifiability 出现 phase transition

## 方法

1. 生成 N 个小型 ReLU 网络（2-hidden-layer, d=10）
2. 固定观测点 x 和 baseline 分布 μ
3. 对每个网络，用不同的局部行为 h（在 x 附近的函数形状）训练
4. 计算 IG 和 SHAP attributions
5. 测量 IGA（identifiability gap）：不同局部行为的模型能否被 attribution 区分

## 配置

- Input dim: d=10
- Hidden dims: 32, 32
- 模型数: 200 per local behavior pair
- α 扫描: [0, 0.1, 0.2, ..., 1.0]
- 局部行为对数: 50 random pairs
- Device: MPS (M4 Max)
- 预计时间: 2-3 小时

## 预期结果

- α=0 (IG): IGA > 1（可区分）
- α=1 (SHAP): IGA ≈ 1（不可区分，等于随机猜测）
- 存在 critical α* ∈ (0, 1)
