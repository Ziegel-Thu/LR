"""
Exp-005 Phase 3: Smoothed Analysis — Adversarial → Natural 的 Phase Transition

在 adversarial 模型上加不同强度的随机扰动（smoothing），
观察 IGA 如何从 ~1.1（worst-case）回升到 ~2.0（average-case）。

类比 Spielman-Teng smoothed analysis：
  worst-case 是理论极端，smoothed = worst-case + Gaussian perturbation。
  如果 IGA 在很小的扰动下就回升，说明 adversarial failure 极其脆弱。

关键输出：IGA vs perturbation_sigma 曲线，找 critical sigma。
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

EXP_DIR = Path(__file__).resolve().parent.parent


class SmoothedAdversarialModel(nn.Module):
    """
    Adversarial 可加模型 + 随机扰动。
    远场值 = adversarial_ffv + sigma * noise。
    """
    def __init__(self, d_in, target_feature, local_behavior,
                 x0_j, delta, adversarial_ffv, sigma, seed):
        super().__init__()
        self.d_in = d_in
        self.j = target_feature
        self.local_behavior = local_behavior
        self.x0_j = x0_j
        self.delta = delta
        self.eta = delta * 0.5

        # Smoothed far field: adversarial + random perturbation
        torch.manual_seed(seed)
        self.far_field_value = adversarial_ffv + sigma * torch.randn(1).item()

    def _local_val(self, t_centered):
        if self.local_behavior == 'pos':
            return 2.0 * t_centered
        elif self.local_behavior == 'neg':
            return -2.0 * t_centered
        elif self.local_behavior == 'quad':
            return 5.0 * t_centered ** 2
        elif self.local_behavior == 'flat':
            return torch.zeros_like(t_centered)

    def _boundary_val(self, side):
        d = self.delta
        t = -d if side == 'left' else d
        if self.local_behavior == 'pos':
            return 2.0 * t
        elif self.local_behavior == 'neg':
            return -2.0 * t
        elif self.local_behavior == 'quad':
            return 5.0 * t ** 2
        elif self.local_behavior == 'flat':
            return 0.0

    def forward(self, x):
        t = x[:, self.j]
        x0 = self.x0_j
        d = self.delta
        eta = self.eta
        ffv = self.far_field_value
        result = torch.zeros_like(t)

        # Local
        local_mask = (t >= x0 - d) & (t <= x0 + d)
        if local_mask.any():
            result[local_mask] = self._local_val(t[local_mask] - x0)

        # Far field
        far_mask = (t < x0 - d - eta) | (t > x0 + d + eta)
        result[far_mask] = ffv

        # Left transition
        lt = (t >= x0 - d - eta) & (t < x0 - d)
        if lt.any():
            alpha = (t[lt] - (x0 - d - eta)) / eta
            result[lt] = ffv * (1 - alpha) + self._boundary_val('left') * alpha

        # Right transition
        rt = (t > x0 + d) & (t <= x0 + d + eta)
        if rt.any():
            alpha = (t[rt] - (x0 + d)) / eta
            result[rt] = self._boundary_val('right') * (1 - alpha) + ffv * alpha

        return result


def shap_values(model, x, baseline_samples, n_samples=100):
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
    return (shap / n_samples).detach()


def compute_adversarial_ffv(x0_j, delta, local_behavior, mu_samples):
    """Compute far-field value that makes SHAP_j = 0."""
    eta = delta * 0.5
    t = mu_samples
    local_mask = (t >= x0_j - delta) & (t <= x0_j + delta)
    far_mask = (t < x0_j - delta - eta) | (t > x0_j + delta + eta)

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

    if p_far < 1e-6:
        return 0.0
    return (0.0 - E_local.item()) / p_far.item()  # target SHAP = 0


def measure_iga(attrs_h0, attrs_h1):
    vals_h0 = attrs_h0[:, 0]
    vals_h1 = attrs_h1[:, 0]
    all_vals = np.concatenate([vals_h0, vals_h1])
    thresholds = np.sort(np.unique(all_vals))
    best_iga = 1.0
    for t in thresholds:
        best_iga = max(best_iga,
                       np.mean(vals_h0 <= t) + np.mean(vals_h1 > t),
                       np.mean(vals_h0 > t) + np.mean(vals_h1 <= t))
    return best_iga


def main():
    print("=" * 60)
    print("Exp-005 Phase 3: Smoothed Analysis")
    print("=" * 60)

    d_in = 10
    torch.manual_seed(42)
    x0 = torch.rand(d_in) * 0.5 + 0.25
    x0_j = x0[0].item()
    delta = 0.1
    baseline_samples = torch.rand(5000, d_in)
    mu_j = baseline_samples[:, 0]

    # Perturbation strengths: log-spaced from 0.001 to 10
    sigmas = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    n_models = 30
    behavior_pairs = [('pos', 'neg'), ('pos', 'flat'), ('quad', 'flat')]

    results = {}

    for b0, b1 in behavior_pairs:
        print(f"\n--- {b0} vs {b1} ---")
        ffv0 = compute_adversarial_ffv(x0_j, delta, b0, mu_j)
        ffv1 = compute_adversarial_ffv(x0_j, delta, b1, mu_j)

        pair_results = {}
        for sigma in sigmas:
            models_h0 = [SmoothedAdversarialModel(d_in, 0, b0, x0_j, delta, ffv0, sigma, s)
                         for s in range(n_models)]
            models_h1 = [SmoothedAdversarialModel(d_in, 0, b1, x0_j, delta, ffv1, sigma, s + 1000)
                         for s in range(n_models)]

            attrs_h0 = np.array([shap_values(m, x0, baseline_samples, n_samples=80).numpy()
                                 for m in models_h0])
            attrs_h1 = np.array([shap_values(m, x0, baseline_samples, n_samples=80).numpy()
                                 for m in models_h1])

            iga = measure_iga(attrs_h0, attrs_h1)
            mean_diff = abs(attrs_h0[:, 0].mean() - attrs_h1[:, 0].mean())
            print(f"  σ={sigma:>6.3f}: IGA={iga:.3f}, |Δμ|={mean_diff:.4f}")

            pair_results[f"sigma_{sigma}"] = {
                'iga': float(iga),
                'mean_diff': float(mean_diff),
                'mean_h0': float(attrs_h0[:, 0].mean()),
                'mean_h1': float(attrs_h1[:, 0].mean()),
            }

        results[f"{b0}_vs_{b1}"] = pair_results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: IGA vs σ (smoothing strength)")
    print("=" * 60)
    print(f"{'Pair':>15} | " + " | ".join(f"σ={s:.3f}" if s < 1 else f"σ={s:.0f}" for s in sigmas))
    print("-" * 120)
    for pair, pr in results.items():
        vals = [pr[f"sigma_{s}"]["iga"] for s in sigmas]
        print(f"{pair:>15} | " + " | ".join(f"{v:.3f}" for v in vals))

    out_path = EXP_DIR / "results_phase3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
