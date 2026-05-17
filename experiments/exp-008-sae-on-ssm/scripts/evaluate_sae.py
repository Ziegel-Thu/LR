#!/usr/bin/env python3
"""
Exp-008 Phase 1: Evaluate trained SAE on Mamba-130M activations.

Computes reconstruction quality, dead features, sparsity stats,
and generates top-activating examples for feature inspection.

Usage:
  python evaluate_sae.py --sae-path /nvmessd/lifanhong/LR-env/exp008_sae/sae_final.pt \
                         --acts-dir /nvmessd/lifanhong/LR-env/exp008_acts
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TopKSAE(nn.Module):
    """Same architecture as train_sae.py."""

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

    def encode(self, x):
        pre_acts = F.relu(self.encoder(x))
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        sparse_acts = torch.zeros_like(pre_acts)
        sparse_acts.scatter_(1, topk_idx, topk_vals)
        return sparse_acts

    def forward(self, x):
        sparse_acts = self.encode(x)
        x_hat = self.decoder(sparse_acts)
        return x_hat, sparse_acts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-path", required=True)
    parser.add_argument("--acts-dir", required=True)
    parser.add_argument("--out-dir", default=None, help="Output dir. Default: same as sae dir")
    parser.add_argument("--n-eval", type=int, default=100000, help="Tokens to evaluate on")
    parser.add_argument("--top-n", type=int, default=50, help="Features to inspect")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sae_path = Path(args.sae_path)
    acts_dir = Path(args.acts_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sae_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SAE
    ckpt = torch.load(sae_path, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    sae = TopKSAE(config["d_model"], config["d_sae"], config["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae = sae.to(args.device).eval()

    acts_mean = ckpt["acts_mean"].to(args.device)
    acts_std = ckpt["acts_std"].to(args.device)

    print(f"SAE: d_model={config['d_model']}, d_sae={config['d_sae']}, K={config['k']}")

    # Load eval activations
    chunks = sorted(acts_dir.glob("acts_*.pt"))
    all_acts = []
    n_loaded = 0
    for cp in chunks:
        acts = torch.load(cp, map_location="cpu", weights_only=True)
        all_acts.append(acts)
        n_loaded += acts.shape[0]
        if n_loaded >= args.n_eval:
            break

    eval_acts = torch.cat(all_acts, dim=0)[:args.n_eval]
    print(f"Eval set: {eval_acts.shape}")

    # Normalize
    eval_acts_norm = (eval_acts.to(args.device) - acts_mean) / acts_std

    # Evaluate in batches
    batch_size = 4096
    all_mse = []
    all_sparse_acts = []
    feature_fire_count = torch.zeros(config["d_sae"], device=args.device)

    with torch.no_grad():
        for i in range(0, len(eval_acts_norm), batch_size):
            batch = eval_acts_norm[i:i + batch_size]
            x_hat, sparse = sae(batch)
            mse = F.mse_loss(x_hat, batch, reduction="none").mean(dim=-1)
            all_mse.append(mse.cpu())
            all_sparse_acts.append(sparse.cpu())
            feature_fire_count += (sparse > 0).float().sum(dim=0)

    all_mse = torch.cat(all_mse)
    all_sparse_acts = torch.cat(all_sparse_acts)

    # ── Metrics ──
    mean_mse = all_mse.mean().item()

    # Variance explained
    eval_var = eval_acts_norm.var().item()
    frac_var_explained = 1 - mean_mse / (eval_var + 1e-10)

    # Dead features
    dead_mask = feature_fire_count == 0
    n_dead = dead_mask.sum().item()
    dead_pct = n_dead / config["d_sae"] * 100

    # Feature firing rates
    fire_rate = feature_fire_count / len(eval_acts_norm)
    alive_fire_rates = fire_rate[~dead_mask].cpu().numpy()

    # Sparsity: avg number of active features per token
    avg_active = (all_sparse_acts > 0).float().sum(dim=1).mean().item()

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Mean MSE (normalized):    {mean_mse:.6f}")
    print(f"  Frac variance explained:  {frac_var_explained:.4f}")
    print(f"  Dead features:            {n_dead}/{config['d_sae']} ({dead_pct:.1f}%)")
    print(f"  Avg active per token:     {avg_active:.1f} (target K={config['k']})")
    if len(alive_fire_rates) > 0:
        print(f"  Alive feature fire rate:  mean={alive_fire_rates.mean():.4f}, "
              f"median={np.median(alive_fire_rates):.4f}")

    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. MSE histogram
    axes[0, 0].hist(all_mse.numpy(), bins=100, edgecolor="none", alpha=0.7)
    axes[0, 0].set_xlabel("MSE per token")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Reconstruction MSE (mean={mean_mse:.6f})")
    axes[0, 0].axvline(mean_mse, color="red", linestyle="--", label=f"mean={mean_mse:.4f}")
    axes[0, 0].legend()

    # 2. Feature firing rate histogram
    if len(alive_fire_rates) > 0:
        axes[0, 1].hist(alive_fire_rates, bins=100, edgecolor="none", alpha=0.7)
    axes[0, 1].set_xlabel("Fire rate")
    axes[0, 1].set_ylabel("Count (alive features)")
    axes[0, 1].set_title(f"Feature Fire Rate (dead={dead_pct:.1f}%)")
    axes[0, 1].set_yscale("log")

    # 3. Active features per token
    active_per_token = (all_sparse_acts > 0).float().sum(dim=1).numpy()
    axes[1, 0].hist(active_per_token, bins=50, edgecolor="none", alpha=0.7)
    axes[1, 0].set_xlabel("Active features")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"Active Features per Token (mean={avg_active:.1f})")

    # 4. Feature activation magnitude (top features)
    feature_mean_act = all_sparse_acts.mean(dim=0).numpy()
    sorted_idx = np.argsort(feature_mean_act)[::-1][:100]
    axes[1, 1].bar(range(len(sorted_idx)), feature_mean_act[sorted_idx], alpha=0.7)
    axes[1, 1].set_xlabel("Feature rank")
    axes[1, 1].set_ylabel("Mean activation")
    axes[1, 1].set_title("Top 100 Features by Mean Activation")

    fig.suptitle("Exp-008: Mamba-130M SAE Evaluation", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_figures.png", dpi=150)
    print(f"\nFigures: {out_dir / 'eval_figures.png'}")

    # ── Save results ──
    eval_results = {
        "config": config,
        "mean_mse": mean_mse,
        "frac_variance_explained": frac_var_explained,
        "n_dead": n_dead,
        "dead_pct": dead_pct,
        "avg_active_per_token": avg_active,
        "alive_fire_rate_mean": float(alive_fire_rates.mean()) if len(alive_fire_rates) > 0 else 0,
        "alive_fire_rate_median": float(np.median(alive_fire_rates)) if len(alive_fire_rates) > 0 else 0,
        "n_eval_tokens": len(eval_acts_norm),
    }
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Results: {out_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
