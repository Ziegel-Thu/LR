# Exp-015: OthelloGPT 策略 Probing

## 目的

OthelloGPT 是世界模型探测的奠基案例。已有工作证明线性 probe 可恢复棋盘状态（>99% acc），
但**策略概念**（tile stability、frontier discs、mobility、corner/edge control）几乎未被探测。

本实验系统探测 OthelloGPT 中的战略概念表征，并用 exp-007 的 encoding≠use 框架测试
这些概念是否被因果使用。

## 方法

1. 下载/训练 OthelloGPT 模型（Li et al. 2023 的 8 层 GPT）
2. 生成 Othello 棋局数据，计算每步的真值标签：
   - 棋盘状态（64 格 × 3 类：黑/白/空）— 已知 baseline
   - Tile stability（每格是否不可翻转）
   - Frontier discs（与空格相邻的棋子数）
   - Mobility（合法走子数）
   - Corner occupancy（四角控制）
   - Edge control（边线控制）
3. 逐层线性/MLP probe 训练
4. Ablation：投影掉 probe 方向，测下一步预测准确率变化

## 假设

- 低层编码棋盘状态，高层编码策略概念
- 策略概念的 ghost ratio 应低于 exp-007 中的 token-level 特征
  （因为 OthelloGPT 的任务直接依赖策略理解）

## 节点

jiagpu4（8× A40）
