"""
Exp-005 Phase 2: Adversarial Bilodeau Construction

验证 Bilodeau 不可能性在 worst-case 下确实成立：
构造两个模型，局部行为不同但 SHAP/IG attributions 相同。

核心思路（来自 formalization_v3 §5.2）：
  对可加模型 m(x) = g(x_0)，SHAP_0 = g(x_0) - E_μ[g]。
  通过调节 g 在邻域外的值（远场），可以让任意两个不同局部行为的
  模型产生相同的 SHAP attribution。

Phase 1 结果：自然训练网络上 IGA≈2.0（SHAP works）
Phase 2 目标：adversarial 构造上 IGA→1.0（SHAP fails）
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import gc

EXP_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# 1. Adversarial model construction (Bilodeau-style)
# ============================================================

class AdversarialAdditiveModel(nn.Module):
    """
    可加模型 m(x) = g_j(x_j)，其中 g_j 是分段线性函数。
    在邻域内行为由 local_behavior 控制，
    邻域外由 far_field_value 控制（用来对消 SHAP）。
    """
    def __init__(self, d_in, target_feature=0, local_behavior='pos',
                 x0_j=0.5, delta=0.1, far_field_value=0.0):
        super().__init__()
        self.d_in = d_in
        self.j = target_feature
        self.local_behavior = local_behavior
        self.x0_j = x0_j
        self.delta = delta
        self.far_field_value = far_field_value

    def _g(self, t):
        """Piecewise function: local behavior in [x0-delta, x0+delta], constant far field."""
        x0 = self.x0_j
        d = self.delta
        eta = d * 0.5  # transition band width

        result = torch.zeros_like(t)

        # Local region: [x0-d, x0+d]
        local_mask = (t >= x0 - d) & (t <= x0 + d)
        t_local = t[local_mask] - x0  # centered

        if self.local_behavior == 'pos':
            result[local_mask] = 2.0 * t_local
        elif self.local_behavior == 'neg':
            result[local_mask] = -2.0 * t_local
        elif self.local_behavior == 'quad':
            result[local_mask] = 5.0 * t_local ** 2
        elif self.local_behavior == 'flat':
            result[local_mask] = 0.0

        # Far field: constant value (the adversarial knob)
        far_mask = (t < x0 - d - eta) | (t > x0 + d + eta)
        result[far_mask] = self.far_field_value

        # Transition bands: linear interpolation
        # Left transition: [x0-d-eta, x0-d]
        left_trans = (t >= x0 - d - eta) & (t < x0 - d)
        if left_trans.any():
            t_lt = t[left_trans]
            # Interpolate from far_field_value to local value at x0-d
            local_val_left = self._local_val_at_boundary('left')
            alpha = (t_lt - (x0 - d - eta)) / eta
            result[left_trans] = self.far_field_value * (1 - alpha) + local_val_left * alpha

        # Right transition: [x0+d, x0+d+eta]
        right_trans = (t > x0 + d) & (t <= x0 + d + eta)
        if right_trans.any():
            t_rt = t[right_trans]
            local_val_right = self._local_val_at_boundary('right')
            alpha = (t_rt - (x0 + d)) / eta
            result[right_trans] = local_val_right * (1 - alpha) + self.far_field_value * alpha

        return result

    def _local_val_at_boundary(self, side):
        d = self.delta
        if side == 'left':
            t_boundary = -d  # centered
        else:
            t_boundary = d

        if self.local_behavior == 'pos':
            return 2.0 * t_boundary
        elif self.local_behavior == 'neg':
            return -2.0 * t_boundary
        elif self.local_behavior == 'quad':
            return 5.0 * t_boundary ** 2
        elif self.local_behavior == 'flat':
            return 0.0

    def forward(self, x):
        # Additive: only depends on feature j
        return self._g(x[:, self.j])


def compute_adversarial_far_field(x0_j, delta, local_behavior, mu_samples, target_shap):
    """
    计算使 SHAP_j = target_shap 所需的 far_field_value。

    对可加单特征模型: SHAP_j = g(x0_j) - E_μ[g]
    E_μ[g] = E_local + far_field_value * p_far + transition_terms

    求解 far_field_value 使得 g(x0_j) - E_μ[g] = target_shap。
    """
    eta = delta * 0.5

    # Compute g(x0_j) for this local behavior
    if local_behavior == 'pos':
        g_at_x0 = 0.0  # 2.0 * (x0_j - x0_j) = 0
    elif local_behavior == 'neg':
        g_at_x0 = 0.0
    elif local_behavior == 'quad':
        g_at_x0 = 0.0  # 5.0 * 0^2 = 0
    elif local_behavior == 'flat':
        g_at_x0 = 0.0

    # Estimate E_μ[g] as function of far_field_value
    # Sample μ and compute contributions
    t = mu_samples  # (N,) samples from μ_j

    local_mask = (t >= x0_j - delta) & (t <= x0_j + delta)
    far_mask = (t < x0_j - delta - eta) | (t > x0_j + delta + eta)

    # Local contribution (independent of far_field_value)
    t_local = t[local_mask] - x0_j
    if local_behavior == 'pos':
        local_vals = 2.0 * t_local
    elif local_behavior == 'neg':
        local_vals = -2.0 * t_local
    elif local_behavior == 'quad':
        local_vals = 5.0 * t_local ** 2
    elif local_behavior == 'flat':
        local_vals = torch.zeros_like(t_local)

    E_local = local_vals.mean() * local_mask.float().mean()
    p_far = far_mask.float().mean()

    # Transition contribution (depends on far_field_value, but approximately)
    # For simplicity, treat transition as negligible (eta is small)
    # E_μ[g] ≈ E_local + far_field_value * p_far

    # Solve: g(x0_j) - (E_local + ffv * p_far) = target_shap
    # ffv = (g(x0_j) - E_local - target_shap) / p_far
    if p_far < 1e-6:
        return 0.0  # can't control

    ffv = (g_at_x0 - E_local.item() - target_shap) / p_far.item()
    return ffv


# ============================================================
# 2. Attribution computation (reuse from phase 1)
# ============================================================

def shap_values_exact(model, x, baseline_samples, n_samples=200):
    """Permutation SHAP."""
    d = x.shape[0]
    shap = torch.zeros(d)
    for _ in range(n_samples):
        perm = torch.randperm(d)
        z = baseline_samples[torch.randint(len(baseline_samples), (1,)).item()]
        x_masked = z.clone()
        prev_val = model(x_masked.unsqueeze(0)).item()
        for j in perm:
            x_masked[j] = x[j]
            curr_val = model(x_masked.unsqueeze(0)).item()
            shap[j] += (curr_val - prev_val)
            prev_val = curr_val
    shap /= n_samples
    return shap.detach()


def integrated_gradients(model, x, baseline, steps=100):
    """Integrated Gradients."""
    x = x.detach().clone()
    baseline = baseline.detach().clone()
    alphas = torch.linspace(0, 1, steps + 1).unsqueeze(1)
    path = baseline.unsqueeze(0) + alphas * (x - baseline).unsqueeze(0)
    path.requires_grad_(True)
    outputs = model(path)
    grads = torch.autograd.grad(outputs.sum(), path)[0]
    avg_grads = grads.mean(dim=0)
    ig = (x - baseline) * avg_grads
    return ig.detach()


# ============================================================
# 3. Main experiment
# ============================================================

def main():
    print("=" * 60)
    print("Exp-005 Phase 2: Adversarial Bilodeau Construction")
    print("=" * 60)

    d_in = 10
    x0 = torch.rand(d_in) * 0.5 + 0.25
    x0_j = x0[0].item()
    delta = 0.1
    target_feature = 0

    # Baseline distribution: Uniform[0,1]^d
    n_baseline = 5000
    torch.manual_seed(42)
    baseline_samples = torch.rand(n_baseline, d_in)
    mu_j_samples = baseline_samples[:, target_feature]

    behavior_pairs = [
        ('pos', 'neg'),
        ('pos', 'quad'),
        ('pos', 'flat'),
        ('neg', 'quad'),
        ('quad', 'flat'),
    ]

    alphas = [0.0, 0.2, 0.5, 0.8, 1.0]  # IG-SHAP mix
    n_models_per_behavior = 30

    results = {}

    for b0, b1 in behavior_pairs:
        print(f"\n{'='*50}")
        print(f"Adversarial pair: {b0} vs {b1}")
        print(f"{'='*50}")

        # Target SHAP: pick a common value (0.0)
        target_shap = 0.0

        # Compute far field values that make SHAP identical
        ffv0 = compute_adversarial_far_field(x0_j, delta, b0, mu_j_samples, target_shap)
        ffv1 = compute_adversarial_far_field(x0_j, delta, b1, mu_j_samples, target_shap)
        print(f"  Far field values: {b0}={ffv0:.4f}, {b1}={ffv1:.4f}")

        # Create adversarial models with different seeds for slight variation
        models_h0 = []
        models_h1 = []
        for seed in range(n_models_per_behavior):
            # Add small random noise to far field value for each seed
            torch.manual_seed(seed)
            noise = torch.randn(1).item() * 0.01
            m0 = AdversarialAdditiveModel(d_in, target_feature, b0, x0_j, delta, ffv0 + noise)
            m1 = AdversarialAdditiveModel(d_in, target_feature, b1, x0_j, delta, ffv1 + noise)
            models_h0.append(m0)
            models_h1.append(m1)

        pair_results = {}
        for alpha in alphas:
            print(f"  α={alpha:.1f}: ", end="", flush=True)

            attrs_h0 = []
            attrs_h1 = []
            baseline_pt = torch.zeros(d_in)

            for m in models_h0:
                if alpha < 1e-6:
                    a = integrated_gradients(m, x0, baseline_pt)
                elif alpha > 1 - 1e-6:
                    a = shap_values_exact(m, x0, baseline_samples, n_samples=100)
                else:
                    ig = integrated_gradients(m, x0, baseline_pt)
                    shap = shap_values_exact(m, x0, baseline_samples, n_samples=100)
                    a = (1 - alpha) * ig + alpha * shap
                attrs_h0.append(a.numpy())

            for m in models_h1:
                if alpha < 1e-6:
                    a = integrated_gradients(m, x0, baseline_pt)
                elif alpha > 1 - 1e-6:
                    a = shap_values_exact(m, x0, baseline_samples, n_samples=100)
                else:
                    ig = integrated_gradients(m, x0, baseline_pt)
                    shap = shap_values_exact(m, x0, baseline_samples, n_samples=100)
                    a = (1 - alpha) * ig + alpha * shap
                attrs_h1.append(a.numpy())

            attrs_h0 = np.array(attrs_h0)
            attrs_h1 = np.array(attrs_h1)

            # 1D IGA on feature 0
            vals_h0 = attrs_h0[:, 0]
            vals_h1 = attrs_h1[:, 0]

            all_vals = np.concatenate([vals_h0, vals_h1])
            thresholds = np.sort(np.unique(all_vals))
            best_iga = 1.0
            for t in thresholds:
                spec = np.mean(vals_h0 <= t)
                sens = np.mean(vals_h1 > t)
                best_iga = max(best_iga, spec + sens)
                best_iga = max(best_iga, np.mean(vals_h0 > t) + np.mean(vals_h1 <= t))

            print(f"IGA={best_iga:.3f}, "
                  f"μ0={vals_h0.mean():.4f}±{vals_h0.std():.4f}, "
                  f"μ1={vals_h1.mean():.4f}±{vals_h1.std():.4f}")

            pair_results[f"alpha_{alpha:.1f}"] = {
                'iga_1d': float(best_iga),
                'mean_attr0_h0': float(vals_h0.mean()),
                'mean_attr0_h1': float(vals_h1.mean()),
                'std_attr0_h0': float(vals_h0.std()),
                'std_attr0_h1': float(vals_h1.std()),
            }

        results[f"{b0}_vs_{b1}"] = pair_results

    # Summary comparison with Phase 1
    print("\n" + "=" * 60)
    print("SUMMARY: Adversarial IGA (cf. Phase 1 all ≈ 2.0)")
    print("=" * 60)
    print(f"{'Pair':>20} | " + " | ".join(f"α={a:.1f}" for a in alphas))
    print("-" * (20 + 3 + len(alphas) * 9))
    for pair_name, pair_r in results.items():
        vals = [pair_r[f"alpha_{a:.1f}"]["iga_1d"] for a in alphas]
        print(f"{pair_name:>20} | " + " | ".join(f"{v:6.3f}" for v in vals))

    # Save
    out_path = EXP_DIR / "results_phase2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
