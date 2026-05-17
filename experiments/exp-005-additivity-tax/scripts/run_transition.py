"""
Exp-005 Phase 4: Natural → Adversarial Transition

从自然训练网络出发，逐步将远场行为移向 adversarial 值，
测量 IGA 从 ~2.0 崩溃到 ~1.0 的 critical point。

参数 β ∈ [0, 1]:
  β=0: 自然网络（远场由训练自然决定）
  β=1: 完全 adversarial（远场精确对消 SHAP）
  中间: 线性插值

输出: IGA vs β 曲线 → "attribution condition number"
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import gc

EXP_DIR = Path(__file__).resolve().parent.parent


class NaturalToAdversarialModel(nn.Module):
    """
    自然训练的网络 + β 比例的 adversarial 远场修正。

    natural_model: 训练好的网络
    β=0: 使用 natural_model 原样
    β=1: 输出被修正为 adversarial attribution
    中间: 线性插值 output = (1-β)*natural + β*adversarial
    """
    def __init__(self, natural_model, adversarial_model, beta):
        super().__init__()
        self.natural = natural_model
        self.adversarial = adversarial_model
        self.beta = beta

    def forward(self, x):
        with torch.no_grad():
            nat = self.natural(x)
            adv = self.adversarial(x)
        return (1 - self.beta) * nat + self.beta * adv


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


class AdversarialAdditiveModel(nn.Module):
    """分段可加模型，远场值精确对消 SHAP。"""
    def __init__(self, d_in, x0_j, delta, local_behavior, ffv):
        super().__init__()
        self.j = 0
        self.x0_j = x0_j
        self.delta = delta
        self.eta = delta * 0.5
        self.local_behavior = local_behavior
        self.ffv = ffv

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
        t = -self.delta if side == 'left' else self.delta
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
        x0, d, eta, ffv = self.x0_j, self.delta, self.eta, self.ffv
        result = torch.zeros_like(t)

        local = (t >= x0 - d) & (t <= x0 + d)
        if local.any():
            result[local] = self._local_val(t[local] - x0)

        far = (t < x0 - d - eta) | (t > x0 + d + eta)
        result[far] = ffv

        lt = (t >= x0 - d - eta) & (t < x0 - d)
        if lt.any():
            a = (t[lt] - (x0 - d - eta)) / eta
            result[lt] = ffv * (1 - a) + self._boundary_val('left') * a

        rt = (t > x0 + d) & (t <= x0 + d + eta)
        if rt.any():
            a = (t[rt] - (x0 + d)) / eta
            result[rt] = self._boundary_val('right') * (1 - a) + ffv * a

        return result


def train_natural_model(d_in, x0, delta, behavior, seed):
    torch.manual_seed(seed)
    model = SmallReLUNet(d_in)
    n_local, n_far = 500, 500
    x_local = x0.unsqueeze(0).repeat(n_local, 1) + delta * (2 * torch.rand(n_local, d_in) - 1)
    x_far = torch.rand(n_far, d_in)
    x_train = torch.cat([x_local, x_far])

    t = x_train[:, 0] - x0[0]
    if behavior == 'pos':
        local_y = 2.0 * t
    elif behavior == 'neg':
        local_y = -2.0 * t
    elif behavior == 'quad':
        local_y = 5.0 * t ** 2
    elif behavior == 'flat':
        local_y = torch.zeros_like(t)
    y_train = local_y + 0.5 * x_train[:, 1:].sum(dim=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(300):
        loss = nn.MSELoss()(model(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def compute_ffv(x0_j, delta, behavior, mu_samples):
    eta = delta * 0.5
    t = mu_samples
    local_mask = (t >= x0_j - delta) & (t <= x0_j + delta)
    far_mask = (t < x0_j - delta - eta) | (t > x0_j + delta + eta)

    t_local = t[local_mask] - x0_j
    if behavior == 'pos':
        lv = 2.0 * t_local
    elif behavior == 'neg':
        lv = -2.0 * t_local
    elif behavior == 'quad':
        lv = 5.0 * t_local ** 2
    elif behavior == 'flat':
        lv = torch.zeros_like(t_local)

    E_local = lv.mean() * local_mask.float().mean()
    p_far = far_mask.float().mean()
    if p_far < 1e-6:
        return 0.0
    return (0.0 - E_local.item()) / p_far.item()


def shap_values(model, x, baseline_samples, n_samples=100):
    d = x.shape[0]
    shap = torch.zeros(d)
    for _ in range(n_samples):
        perm = torch.randperm(d)
        z = baseline_samples[torch.randint(len(baseline_samples), (1,)).item()]
        x_m = z.clone()
        prev = model(x_m.unsqueeze(0)).item()
        for j in perm:
            x_m[j] = x[j]
            curr = model(x_m.unsqueeze(0)).item()
            shap[j] += curr - prev
            prev = curr
    return (shap / n_samples).detach()


def measure_iga(attrs_h0, attrs_h1):
    v0 = attrs_h0[:, 0]
    v1 = attrs_h1[:, 0]
    all_v = np.concatenate([v0, v1])
    best = 1.0
    for t in np.sort(np.unique(all_v)):
        best = max(best,
                   np.mean(v0 <= t) + np.mean(v1 > t),
                   np.mean(v0 > t) + np.mean(v1 <= t))
    return best


def main():
    print("=" * 60)
    print("Exp-005 Phase 4: Natural → Adversarial Transition")
    print("=" * 60)

    d_in = 10
    torch.manual_seed(42)
    x0 = torch.rand(d_in) * 0.5 + 0.25
    x0_j = x0[0].item()
    delta = 0.1
    baseline_samples = torch.rand(5000, d_in)
    mu_j = baseline_samples[:, 0]

    betas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_models = 30
    pairs = [('pos', 'neg'), ('pos', 'flat'), ('quad', 'flat')]

    results = {}

    for b0, b1 in pairs:
        print(f"\n--- {b0} vs {b1} ---")
        ffv0 = compute_ffv(x0_j, delta, b0, mu_j)
        ffv1 = compute_ffv(x0_j, delta, b1, mu_j)

        # Train natural models
        print(f"  Training {n_models} natural models per behavior...")
        nat_h0 = [train_natural_model(d_in, x0, delta, b0, s) for s in range(n_models)]
        nat_h1 = [train_natural_model(d_in, x0, delta, b1, s + 1000) for s in range(n_models)]

        # Adversarial models (one per seed, matching local behavior)
        adv_h0 = [AdversarialAdditiveModel(d_in, x0_j, delta, b0, ffv0) for _ in range(n_models)]
        adv_h1 = [AdversarialAdditiveModel(d_in, x0_j, delta, b1, ffv1) for _ in range(n_models)]

        pair_results = {}
        for beta in betas:
            # Interpolate: (1-β)*natural + β*adversarial
            mixed_h0 = [NaturalToAdversarialModel(n, a, beta) for n, a in zip(nat_h0, adv_h0)]
            mixed_h1 = [NaturalToAdversarialModel(n, a, beta) for n, a in zip(nat_h1, adv_h1)]

            attrs_h0 = np.array([shap_values(m, x0, baseline_samples, n_samples=80).numpy()
                                 for m in mixed_h0])
            attrs_h1 = np.array([shap_values(m, x0, baseline_samples, n_samples=80).numpy()
                                 for m in mixed_h1])

            iga = measure_iga(attrs_h0, attrs_h1)
            print(f"  β={beta:.2f}: IGA={iga:.3f}, "
                  f"μ0={attrs_h0[:,0].mean():.4f}±{attrs_h0[:,0].std():.4f}, "
                  f"μ1={attrs_h1[:,0].mean():.4f}±{attrs_h1[:,0].std():.4f}")

            pair_results[f"beta_{beta:.2f}"] = {
                'iga': float(iga),
                'mean_h0': float(attrs_h0[:, 0].mean()),
                'mean_h1': float(attrs_h1[:, 0].mean()),
                'std_h0': float(attrs_h0[:, 0].std()),
                'std_h1': float(attrs_h1[:, 0].std()),
            }

        results[f"{b0}_vs_{b1}"] = pair_results
        del nat_h0, nat_h1, adv_h0, adv_h1
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: IGA vs β (0=natural, 1=adversarial)")
    print("=" * 60)
    print(f"{'Pair':>15} | " + " | ".join(f"β={b:.2f}" for b in betas))
    print("-" * 120)
    for pair, pr in results.items():
        vals = [pr[f"beta_{b:.2f}"]["iga"] for b in betas]
        print(f"{pair:>15} | " + " | ".join(f"{v:.3f}" for v in vals))

    out_path = EXP_DIR / "results_phase4.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
