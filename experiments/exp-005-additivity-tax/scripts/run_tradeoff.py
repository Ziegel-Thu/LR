"""
Exp-005: Additivity Tax — IG vs SHAP Identifiability Tradeoff

验证 additivity（而非 completeness）是归因方法不可能性的根源。
对 Φ^α = (1-α)·IG + α·SHAP，测量 identifiability gap (IGA) 随 α 的变化。

理论预测：
  α=0 (IG): IGA > 1（可区分不同局部行为）
  α=1 (SHAP): IGA = 1（不可区分）
  存在 critical α*
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import gc

EXP_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# 1. 小型 ReLU 网络
# ============================================================

class SmallReLUNet(nn.Module):
    def __init__(self, d_in=10, d_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# 2. 生成具有指定局部行为的模型
# ============================================================

def make_model_with_local_behavior(d_in, x0, delta, behavior_type, seed):
    """
    训练一个小网络，使其在 x0 附近的行为由 behavior_type 决定。

    behavior_type:
      'linear_pos': 在 x0 附近沿 feature 0 正斜率
      'linear_neg': 在 x0 附近沿 feature 0 负斜率
      'quadratic':  在 x0 附近沿 feature 0 二次
      'flat':       在 x0 附近沿 feature 0 平坦
    """
    torch.manual_seed(seed)
    model = SmallReLUNet(d_in)

    # 生成训练数据：在 x0±delta 邻域内密集采样
    n_local = 500
    n_far = 500
    x_local = x0.unsqueeze(0).repeat(n_local, 1) + delta * (2 * torch.rand(n_local, d_in) - 1)
    x_far = torch.rand(n_far, d_in)  # 远场随机

    x_train = torch.cat([x_local, x_far])

    # 目标函数根据 behavior_type 设定（只控制 feature 0 的局部行为）
    def target_fn(x, btype):
        t = x[:, 0] - x0[0]  # 相对于 x0 的偏移
        if btype == 'linear_pos':
            local = 2.0 * t
        elif btype == 'linear_neg':
            local = -2.0 * t
        elif btype == 'quadratic':
            local = 3.0 * t ** 2
        elif btype == 'flat':
            local = torch.zeros_like(t)
        else:
            raise ValueError(f"Unknown behavior: {btype}")
        # 远场：用其余 features 的随机线性组合
        far = 0.5 * x[:, 1:].sum(dim=1)
        return local + far

    y_train = target_fn(x_train, behavior_type)

    # 快速训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(300):
        pred = model(x_train)
        loss = nn.MSELoss()(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ============================================================
# 3. Attribution 计算
# ============================================================

def integrated_gradients(model, x, baseline, steps=50):
    """Integrated Gradients: IG_j = (x_j - x'_j) * ∫_0^1 ∂f/∂x_j(x' + t(x-x')) dt"""
    x = x.detach().requires_grad_(False)
    baseline = baseline.detach()
    alphas = torch.linspace(0, 1, steps + 1).unsqueeze(1)  # (steps+1, 1)
    path = baseline.unsqueeze(0) + alphas * (x - baseline).unsqueeze(0)  # (steps+1, d)
    path.requires_grad_(True)

    outputs = model(path)  # (steps+1,)
    grads = torch.autograd.grad(outputs.sum(), path)[0]  # (steps+1, d)

    # Trapezoidal integration
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2.0
    # 修正：用简单均值
    avg_grads = grads.mean(dim=0)
    ig = (x - baseline) * avg_grads
    return ig.detach()


def shap_values(model, x, baseline_samples, n_samples=100):
    """
    Kernel SHAP 的简化版：采样式 marginal contribution。
    SHAP_j = E_S[f(x_S, z_{-S}) - f(z)] 对 feature j 的边际贡献。
    """
    d = x.shape[0]
    shap = torch.zeros(d)

    for _ in range(n_samples):
        # 随机排列
        perm = torch.randperm(d)
        # 随机选一个 baseline sample
        z = baseline_samples[torch.randint(len(baseline_samples), (1,)).item()]

        # 逐步加入 features
        x_masked = z.clone()
        prev_val = model(x_masked.unsqueeze(0)).item()
        for j in perm:
            x_masked[j] = x[j]
            curr_val = model(x_masked.unsqueeze(0)).item()
            shap[j] += (curr_val - prev_val)
            prev_val = curr_val

    shap /= n_samples
    return shap.detach()


def mixed_attribution(model, x, baseline, baseline_samples, alpha, ig_steps=50, shap_samples=100):
    """Φ^α = (1-α)·IG + α·SHAP"""
    ig = integrated_gradients(model, x, baseline, steps=ig_steps)
    if alpha < 1e-6:
        return ig
    shap = shap_values(model, x, baseline_samples, n_samples=shap_samples)
    return (1 - alpha) * ig + alpha * shap


# ============================================================
# 4. Identifiability Gap (IGA) 测量
# ============================================================

def measure_iga(models_h0, models_h1, x0, baseline, baseline_samples, alpha,
                ig_steps=50, shap_samples=100):
    """
    测量 IGA：不同局部行为的模型能否被 Φ^α 区分。

    对每个模型计算 attribution，然后用最优线性阈值分类。
    IGA = spec + sens（最优值；>1 表示可区分）。
    """
    attrs_h0 = []
    attrs_h1 = []

    for m in models_h0:
        a = mixed_attribution(m, x0, baseline, baseline_samples, alpha,
                              ig_steps=ig_steps, shap_samples=shap_samples)
        attrs_h0.append(a.numpy())

    for m in models_h1:
        a = mixed_attribution(m, x0, baseline, baseline_samples, alpha,
                              ig_steps=ig_steps, shap_samples=shap_samples)
        attrs_h1.append(a.numpy())

    attrs_h0 = np.array(attrs_h0)  # (n, d)
    attrs_h1 = np.array(attrs_h1)  # (n, d)

    # 用 feature 0 的 attribution 做 1D 分类（理论预测这是最 informative 的维度）
    vals_h0 = attrs_h0[:, 0]
    vals_h1 = attrs_h1[:, 0]

    # 最优阈值搜索
    all_vals = np.concatenate([vals_h0, vals_h1])
    thresholds = np.sort(all_vals)

    best_iga = 1.0
    for t in thresholds:
        # T(v) = 1 if v > t
        spec = np.mean(vals_h0 <= t)  # h0 被正确分类为 0
        sens = np.mean(vals_h1 > t)   # h1 被正确分类为 1
        iga = spec + sens
        best_iga = max(best_iga, iga)

        # 也试反方向
        spec_r = np.mean(vals_h0 > t)
        sens_r = np.mean(vals_h1 <= t)
        best_iga = max(best_iga, spec_r + sens_r)

    # 也用全 d 维的 attribution 做多维分类（最近邻或 LDA）
    from scipy.spatial.distance import cdist
    # 类间距离 vs 类内距离
    inter = cdist(attrs_h0, attrs_h1).mean()
    intra = (cdist(attrs_h0, attrs_h0).mean() + cdist(attrs_h1, attrs_h1).mean()) / 2
    separability = inter / (intra + 1e-10)

    return {
        'iga_1d': float(best_iga),
        'separability_nd': float(separability),
        'mean_attr0_h0': float(vals_h0.mean()),
        'mean_attr0_h1': float(vals_h1.mean()),
        'std_attr0_h0': float(vals_h0.std()),
        'std_attr0_h1': float(vals_h1.std()),
    }


# ============================================================
# 5. 主实验
# ============================================================

def main():
    print("=" * 60)
    print("Exp-005: Additivity Tax — IG vs SHAP Tradeoff")
    print("=" * 60)

    d_in = 10
    n_models_per_behavior = 30  # 每种局部行为训练多少个模型（不同种子 → 不同远场）
    behavior_pairs = [
        ('linear_pos', 'linear_neg'),
        ('linear_pos', 'quadratic'),
        ('linear_pos', 'flat'),
        ('linear_neg', 'quadratic'),
        ('quadratic', 'flat'),
    ]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    delta = 0.1

    # 固定观测点和 baseline
    torch.manual_seed(42)
    x0 = torch.rand(d_in) * 0.5 + 0.25  # 在 [0.25, 0.75] 中心区域
    baseline = torch.zeros(d_in)  # IG baseline
    baseline_samples = torch.rand(200, d_in)  # SHAP baseline samples

    results = {}

    for b0, b1 in behavior_pairs:
        print(f"\n--- Behavior pair: {b0} vs {b1} ---")

        # 训练模型
        print(f"  Training {n_models_per_behavior} models for '{b0}'...")
        models_h0 = [make_model_with_local_behavior(d_in, x0, delta, b0, seed=s)
                      for s in range(n_models_per_behavior)]

        print(f"  Training {n_models_per_behavior} models for '{b1}'...")
        models_h1 = [make_model_with_local_behavior(d_in, x0, delta, b1, seed=s + 1000)
                      for s in range(n_models_per_behavior)]

        pair_results = {}
        for alpha in alphas:
            print(f"  α={alpha:.1f}: ", end="", flush=True)
            r = measure_iga(models_h0, models_h1, x0, baseline, baseline_samples,
                            alpha, ig_steps=50, shap_samples=50)
            print(f"IGA_1D={r['iga_1d']:.3f}, sep_nD={r['separability_nd']:.3f}, "
                  f"μ0={r['mean_attr0_h0']:.3f}±{r['std_attr0_h0']:.3f}, "
                  f"μ1={r['mean_attr0_h1']:.3f}±{r['std_attr0_h1']:.3f}")
            pair_results[f"alpha_{alpha:.1f}"] = r

        results[f"{b0}_vs_{b1}"] = pair_results

        # 清理
        del models_h0, models_h1
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: IGA_1D across α")
    print("=" * 60)
    print(f"{'Pair':>25} | " + " | ".join(f"α={a:.1f}" for a in alphas))
    print("-" * (25 + 3 + len(alphas) * 8))
    for pair_name, pair_r in results.items():
        vals = [pair_r[f"alpha_{a:.1f}"]["iga_1d"] for a in alphas]
        print(f"{pair_name:>25} | " + " | ".join(f"{v:5.3f}" for v in vals))

    # Save
    out_path = EXP_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
