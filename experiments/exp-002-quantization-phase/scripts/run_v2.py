"""
Exp-002 v2: 量化临界 bit-width — 用噪声模拟量化
不需要网络，不需要 HQQ，直接对已缓存的 FP32 激活加噪声

两种方式:
1. 真量化: 对激活做 round-to-nearest 到 b-bit grid
2. 匹配 MSE 的高斯噪声: 作为对照

对应: experiments/exp-002-quantization-phase/plan.md
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "exp-001-sae-identifiability" / "scripts"))
from run_pilot import TopKSAE


def quantize_activations(acts: torch.Tensor, nbits: int) -> torch.Tensor:
    """Simulate b-bit uniform quantization on activations."""
    if nbits >= 16:
        return acts.clone()

    # Per-channel min/max quantization
    amin = acts.min(dim=0, keepdim=True).values
    amax = acts.max(dim=0, keepdim=True).values
    scale = (amax - amin) / (2**nbits - 1)
    scale = scale.clamp(min=1e-8)

    # Quantize and dequantize
    quantized = torch.round((acts - amin) / scale)
    quantized = quantized.clamp(0, 2**nbits - 1)
    dequantized = quantized * scale + amin
    return dequantized


def add_matched_noise(acts: torch.Tensor, nbits: int) -> torch.Tensor:
    """Add Gaussian noise with MSE matching b-bit quantization."""
    if nbits >= 16:
        return acts.clone()

    amin = acts.min(dim=0, keepdim=True).values
    amax = acts.max(dim=0, keepdim=True).values
    R = amax - amin
    # Uniform quantization MSE = Δ²/12 where Δ = R / 2^b
    delta = R / (2**nbits)
    noise_var = delta**2 / 12
    noise = torch.randn_like(acts) * noise_var.sqrt()
    return acts + noise


def analyze_recovery(sae, fp32_acts, degraded_acts, label):
    """Analyze feature recovery between FP32 and degraded activations."""
    sae.eval()
    n = min(len(fp32_acts), len(degraded_acts), 50000)

    with torch.no_grad():
        _, z_fp32, _ = sae(fp32_acts[:n])
        _, z_deg, _ = sae(degraded_acts[:n])
        x_hat_deg, _, _ = sae(degraded_acts[:n])
        nmse = F.mse_loss(x_hat_deg, degraded_acts[:n]).item() / degraded_acts[:n].var().item()

    z_fp32 = z_fp32.numpy()
    z_deg = z_deg.numpy()

    # Per-feature correlation
    corrs = []
    firing_rates = []
    for j in range(sae.d_sae):
        f1 = z_fp32[:, j]
        f2 = z_deg[:, j]
        fr = (f1 > 0).mean()
        firing_rates.append(fr)

        if f1.std() > 1e-8 and f2.std() > 1e-8:
            r = np.corrcoef(f1, f2)[0, 1]
            corrs.append(r if not np.isnan(r) else 0.0)
        else:
            corrs.append(0.0)

    corrs = np.array(corrs)
    firing_rates = np.array(firing_rates)

    # Alive features only
    alive = firing_rates > 1e-5
    n_alive = alive.sum()
    alive_corrs = corrs[alive]

    # Frequency stratification
    strat = {}
    if n_alive > 10:
        rates = firing_rates[alive]
        ac = corrs[alive]
        t = np.percentile(rates[rates > 0], [33, 67])
        low = rates <= t[0]
        mid = (rates > t[0]) & (rates <= t[1])
        high = rates > t[1]
        strat = {
            "low_freq_corr": float(ac[low].mean()) if low.sum() > 0 else 0,
            "mid_freq_corr": float(ac[mid].mean()) if mid.sum() > 0 else 0,
            "high_freq_corr": float(ac[high].mean()) if high.sum() > 0 else 0,
        }

    return {
        "label": label,
        "nmse": float(nmse),
        "n_alive": int(n_alive),
        "mean_corr": float(alive_corrs.mean()) if n_alive > 0 else 0,
        "frac_above_0.8": float((alive_corrs > 0.8).mean()) if n_alive > 0 else 0,
        "frac_above_0.5": float((alive_corrs > 0.5).mean()) if n_alive > 0 else 0,
        "frac_above_0.3": float((alive_corrs > 0.3).mean()) if n_alive > 0 else 0,
        "frequency_stratified": strat,
    }


def main():
    exp_dir = Path(__file__).resolve().parent.parent

    # Load cached FP32 activations from exp-001
    act_path = exp_dir.parent / "exp-001-sae-identifiability" / "data" / "activations_layer6.pt"
    print(f"Loading FP32 activations from {act_path}")
    fp32_acts = torch.load(act_path, map_location="cpu", weights_only=True)
    # Use first 2M tokens (avoid memory issues with the 14GB file)
    fp32_acts = fp32_acts[:2_000_000]
    print(f"Using {fp32_acts.shape[0]} activation vectors, shape: {fp32_acts.shape}")

    # Load SAE from exp-001
    sae_path = exp_dir.parent / "exp-001-sae-identifiability" / "checkpoints" / "sae_seed0.pt"
    print(f"Loading SAE from {sae_path}")
    ckpt = torch.load(sae_path, map_location="cpu", weights_only=True)
    sae = TopKSAE(768, 4096, 32)
    sae.load_state_dict(ckpt["model_state_dict"])

    # Test each bit-width
    bit_widths = [16, 8, 6, 4, 3, 2, 1]
    results_quant = []
    results_noise = []

    for b in bit_widths:
        print(f"\n--- {b}-bit ---")

        # Real quantization
        q_acts = quantize_activations(fp32_acts, b)
        r = analyze_recovery(sae, fp32_acts, q_acts, f"quant_{b}bit")
        r["nbits"] = b
        results_quant.append(r)
        print(f"  Quantized: mean_r={r['mean_corr']:.4f}, r>0.8={r['frac_above_0.8']:.1%}")

        # Matched-MSE Gaussian noise
        n_acts = add_matched_noise(fp32_acts, b)
        r2 = analyze_recovery(sae, fp32_acts, n_acts, f"noise_{b}bit")
        r2["nbits"] = b
        results_noise.append(r2)
        print(f"  Gaussian:  mean_r={r2['mean_corr']:.4f}, r>0.8={r2['frac_above_0.8']:.1%}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Bits':>4} | {'--- Quantized ---':^30} | {'--- Gaussian Noise ---':^30}")
    print(f"{'':>4} | {'mean_r':>8} {'r>0.8':>8} {'r>0.5':>8} | {'mean_r':>8} {'r>0.8':>8} {'r>0.5':>8}")
    print("-" * 80)
    for rq, rn in zip(results_quant, results_noise):
        print(f"{rq['nbits']:>4} | {rq['mean_corr']:>8.4f} {rq['frac_above_0.8']:>8.1%} {rq['frac_above_0.5']:>8.1%}"
              f" | {rn['mean_corr']:>8.4f} {rn['frac_above_0.8']:>8.1%} {rn['frac_above_0.5']:>8.1%}")

    # Frequency stratification summary
    print(f"\n{'='*80}")
    print("FREQUENCY STRATIFICATION (Quantized)")
    print(f"{'='*80}")
    print(f"{'Bits':>4} | {'Low freq':>10} {'Mid freq':>10} {'High freq':>10}")
    print("-" * 40)
    for r in results_quant:
        s = r.get("frequency_stratified", {})
        print(f"{r['nbits']:>4} | {s.get('low_freq_corr', 0):>10.4f} {s.get('mid_freq_corr', 0):>10.4f} {s.get('high_freq_corr', 0):>10.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    bits = [r["nbits"] for r in results_quant]
    axes[0].plot(bits, [r["mean_corr"] for r in results_quant], 'o-', label="Quantized", color="blue")
    axes[0].plot(bits, [r["mean_corr"] for r in results_noise], 's--', label="Gaussian noise", color="gray")
    axes[0].set_xlabel("Bit-width")
    axes[0].set_ylabel("Mean feature correlation")
    axes[0].set_title("Feature Recovery vs Bit-width")
    axes[0].legend()
    axes[0].set_xlim(max(bits) + 0.5, min(bits) - 0.5)

    axes[1].plot(bits, [r["frac_above_0.8"] for r in results_quant], 'o-', label="r>0.8 (quant)", color="blue")
    axes[1].plot(bits, [r["frac_above_0.5"] for r in results_quant], 's-', label="r>0.5 (quant)", color="green")
    axes[1].plot(bits, [r["frac_above_0.8"] for r in results_noise], 'o--', label="r>0.8 (noise)", color="lightblue")
    axes[1].set_xlabel("Bit-width")
    axes[1].set_ylabel("Fraction of features")
    axes[1].set_title("Feature Survival Rate")
    axes[1].legend()
    axes[1].set_xlim(max(bits) + 0.5, min(bits) - 0.5)

    # Frequency stratification
    for label, key, color in [("Low freq", "low_freq_corr", "red"),
                               ("Mid freq", "mid_freq_corr", "orange"),
                               ("High freq", "high_freq_corr", "green")]:
        vals = [r.get("frequency_stratified", {}).get(key, 0) for r in results_quant]
        axes[2].plot(bits, vals, 'o-', label=label, color=color)
    axes[2].set_xlabel("Bit-width")
    axes[2].set_ylabel("Mean correlation")
    axes[2].set_title("Recovery by Feature Frequency")
    axes[2].legend()
    axes[2].set_xlim(max(bits) + 0.5, min(bits) - 0.5)

    plt.tight_layout()
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "bitwidth_recovery.png", dpi=150)
    print(f"\nFigure saved to {fig_dir / 'bitwidth_recovery.png'}")

    # Save results
    all_results = {
        "quantized": results_quant,
        "gaussian_noise": results_noise,
        "method": "activation-level quantization simulation (not weight quantization)",
    }
    with open(exp_dir / "results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {exp_dir / 'results_v2.json'}")


if __name__ == "__main__":
    main()
