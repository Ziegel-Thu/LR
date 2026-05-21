#!/usr/bin/env python3
"""Exp-021: SAE as Empirical ICA — Are SAE Features Independent?

Tests whether Sparse Autoencoder features satisfy statistical independence,
bridging SAE practice with ICA/identifiability theory.

Phases:
  1. Cache Pythia-160M layer 6 activations (or load existing)
  2. Train SAEs with varying L1 penalty
  3. Compute PCA/FastICA/Random baselines
  4. Measure independence: pairwise correlation + mutual information
  5. Generate plots and summary

Usage:
  CUDA_VISIBLE_DEVICES=1 python sae_independence.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --output-dir /nvmessd/lifanhong/LR-env/exp021_sae_ica
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════
# TopK SAE (from exp-008)
# ═══════════════════════════════════════════════

class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_acts = F.relu(self.encoder(x))
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        sparse_acts = torch.zeros_like(pre_acts)
        sparse_acts.scatter_(1, topk_idx, topk_vals)
        return sparse_acts, pre_acts

    def forward(self, x: torch.Tensor) -> dict:
        sparse_acts, pre_acts = self.encode(x)
        x_hat = self.decoder(sparse_acts)
        mse = F.mse_loss(x_hat, x)
        l1 = pre_acts.abs().mean()
        return {"x_hat": x_hat, "sparse_acts": sparse_acts, "mse": mse, "l1": l1}


# ═══════════════════════════════════════════════
# Phase 1: Cache activations
# ═══════════════════════════════════════════════

def cache_activations(cache_dir: str, output_dir: Path, device: str,
                      n_tokens: int = 500_000) -> torch.Tensor:
    """Cache Pythia-160M layer 6 activations from WikiText-103."""
    acts_path = output_dir / "pythia160m_layer6_acts.pt"

    # Try loading existing cache
    if acts_path.exists():
        print(f"  Loading cached activations: {acts_path}")
        acts = torch.load(acts_path, map_location="cpu", weights_only=True)
        if acts.shape[0] >= n_tokens:
            return acts[:n_tokens]
        print(f"  Cache has {acts.shape[0]} tokens, need {n_tokens}. Re-caching...")

    # Also check exp-008 cache
    exp008_path = Path("/nvmessd/lifanhong/LR-env/exp008_phase2/pythia_acts.pt")
    if exp008_path.exists() and n_tokens <= 235_000:
        print(f"  Loading exp-008 Pythia-160M activations: {exp008_path}")
        acts = torch.load(exp008_path, map_location="cpu", weights_only=True)
        print(f"  Shape: {acts.shape}")
        torch.save(acts[:n_tokens], acts_path)
        return acts[:n_tokens]

    # Cache fresh activations
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_id = "EleutherAI/pythia-160m-deduped"
    print(f"  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=torch.float16
    ).to(device).eval()

    n_layers = len(model.gpt_neox.layers)
    target_layer = 6
    d_model = model.config.hidden_size
    print(f"  {n_layers} layers, d_model={d_model}, hooking layer {target_layer}")

    # Register hook
    activations_buf = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        activations_buf.append(hidden.detach().cpu().float())

    hook = model.gpt_neox.layers[target_layer].register_forward_hook(hook_fn)

    # Load WikiText
    print("  Loading WikiText-103 validation...")
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1",
        split="validation",
        cache_dir=str(Path(cache_dir) / "datasets"),
    )
    texts = [t for t in ds["text"] if len(t.strip()) > 50]
    print(f"  {len(texts)} texts available")

    collected = 0
    all_acts = []
    batch_size, seq_len = 16, 256

    for i in range(0, len(texts), batch_size):
        if collected >= n_tokens:
            break
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts, return_tensors="pt",
            max_length=seq_len, truncation=True, padding="max_length",
        ).to(device)

        activations_buf.clear()
        with torch.no_grad():
            model(**tokens)

        if activations_buf:
            act = activations_buf[0]
            mask = tokens["attention_mask"].cpu()
            act_flat = act[mask.bool()]
            all_acts.append(act_flat)
            collected += act_flat.shape[0]
            if collected % 100_000 < batch_size * seq_len:
                print(f"    {collected:,} tokens...", flush=True)

    hook.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    acts = torch.cat(all_acts, dim=0)[:n_tokens]
    print(f"  Cached {acts.shape[0]:,} tokens × d={acts.shape[1]}")
    torch.save(acts, acts_path)
    return acts


# ═══════════════════════════════════════════════
# Phase 2: Train SAE
# ═══════════════════════════════════════════════

def train_sae(acts: torch.Tensor, d_sae: int, k: int, l1_coeff: float,
              n_steps: int, device: str, lr: float = 3e-4,
              batch_size: int = 4096) -> tuple:
    """Train a TopK SAE. Returns (sae, acts_mean, acts_std, final_mse)."""
    d_model = acts.shape[1]

    acts_mean = acts.mean(dim=0)
    acts_std = acts.std(dim=0) + 1e-8
    acts_norm = (acts - acts_mean) / acts_std

    sae = TopKSAE(d_model, d_sae, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = TensorDataset(acts_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=2, pin_memory=True)

    step = 0
    last_mse = 0.0
    sae.train()
    while step < n_steps:
        for (batch,) in loader:
            if step >= n_steps:
                break
            batch = batch.to(device)
            out = sae(batch)
            loss = out["mse"] + l1_coeff * out["l1"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_mse = out["mse"].item()
            step += 1
            if step % 5000 == 0:
                print(f"      step {step}/{n_steps}, MSE={last_mse:.6f}", flush=True)

    sae.eval()
    return sae, acts_mean, acts_std, last_mse


def get_sae_features(sae: TopKSAE, acts: torch.Tensor,
                     acts_mean: torch.Tensor, acts_std: torch.Tensor,
                     device: str, batch_size: int = 8192) -> np.ndarray:
    """Get SAE feature activations for all tokens."""
    acts_norm = (acts - acts_mean) / acts_std
    all_feats = []
    sae.eval()
    with torch.no_grad():
        for i in range(0, len(acts_norm), batch_size):
            batch = acts_norm[i:i + batch_size].to(device)
            sparse_acts, _ = sae.encode(batch)
            all_feats.append(sparse_acts.cpu())
    return torch.cat(all_feats, dim=0).numpy()


# ═══════════════════════════════════════════════
# Phase 3: Baselines
# ═══════════════════════════════════════════════

def compute_pca(acts_np: np.ndarray, n_components: int) -> np.ndarray:
    """PCA projection (computed via SVD on centered data)."""
    centered = acts_np - acts_np.mean(axis=0)
    # Use randomized SVD for speed
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(centered)


def compute_ica(acts_np: np.ndarray, n_components: int,
                max_samples: int = 100_000) -> np.ndarray:
    """FastICA decomposition."""
    from sklearn.decomposition import FastICA

    # Subsample for fitting (ICA is slow on large data)
    n = min(len(acts_np), max_samples)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(acts_np), n, replace=False)

    ica = FastICA(n_components=n_components, random_state=42, max_iter=500, tol=1e-3)
    ica_feats = ica.fit_transform(acts_np[idx])
    return ica_feats, idx


def compute_random_proj(acts_np: np.ndarray, n_components: int) -> np.ndarray:
    """Random Gaussian projection."""
    rng = np.random.default_rng(42)
    d = acts_np.shape[1]
    proj = rng.standard_normal((d, n_components)) / np.sqrt(d)
    centered = acts_np - acts_np.mean(axis=0)
    return centered @ proj


# ═══════════════════════════════════════════════
# Phase 4: Independence metrics
# ═══════════════════════════════════════════════

def pairwise_abs_correlation(features: np.ndarray, n_pairs: int = 2000,
                             seed: int = 42) -> tuple[float, np.ndarray]:
    """Mean |Pearson r| over subsampled pairs of alive features."""
    n_feat = features.shape[1]
    feat_std = features.std(axis=0)
    alive = np.where(feat_std > 1e-10)[0]

    if len(alive) < 2:
        return 0.0, np.array([])

    rng = np.random.default_rng(seed)
    max_pairs = len(alive) * (len(alive) - 1) // 2
    n_pairs = min(n_pairs, max_pairs)

    # Standardize alive features
    feat_z = np.zeros_like(features)
    feat_z[:, alive] = (
        (features[:, alive] - features[:, alive].mean(axis=0))
        / (feat_std[alive] + 1e-10)
    )

    # Sample pairs
    pairs = set()
    while len(pairs) < n_pairs:
        ij = rng.choice(alive, size=2, replace=False)
        pairs.add((min(ij[0], ij[1]), max(ij[0], ij[1])))
    pairs = list(pairs)

    # Pearson r via dot product on standardized data
    n = features.shape[0]
    corrs = np.array([
        np.dot(feat_z[:, i], feat_z[:, j]) / n for i, j in pairs
    ])
    abs_corrs = np.abs(corrs[np.isfinite(corrs)])
    return float(abs_corrs.mean()) if len(abs_corrs) > 0 else 0.0, abs_corrs


def mutual_info_histogram(x: np.ndarray, y: np.ndarray, bins: int = 50) -> float:
    """Estimate MI between two 1D arrays via histogram binning."""
    # Handle sparse features: put exact zeros in a separate bin
    # by using quantile-based bins on nonzero values
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    c_xy = c_xy / c_xy.sum()
    nz = c_xy > 0
    h_xy = -np.sum(c_xy[nz] * np.log(c_xy[nz]))

    c_x = c_xy.sum(axis=1)
    c_x = c_x[c_x > 0]
    h_x = -np.sum(c_x * np.log(c_x))

    c_y = c_xy.sum(axis=0)
    c_y = c_y[c_y > 0]
    h_y = -np.sum(c_y * np.log(c_y))

    return float(max(h_x + h_y - h_xy, 0.0))


def pairwise_mi(features: np.ndarray, n_pairs: int = 500,
                max_tokens: int = 50_000, bins: int = 50,
                seed: int = 42) -> tuple[float, np.ndarray]:
    """Mean pairwise MI over subsampled pairs and tokens."""
    n_feat = features.shape[1]
    feat_std = features.std(axis=0)
    alive = np.where(feat_std > 1e-10)[0]

    if len(alive) < 2:
        return 0.0, np.array([])

    rng = np.random.default_rng(seed)

    # Subsample tokens
    n_tok = min(features.shape[0], max_tokens)
    tok_idx = rng.choice(features.shape[0], n_tok, replace=False)
    feat_sub = features[tok_idx]

    # Subsample pairs
    max_pairs = len(alive) * (len(alive) - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    pairs = set()
    while len(pairs) < n_pairs:
        ij = rng.choice(alive, size=2, replace=False)
        pairs.add((min(ij[0], ij[1]), max(ij[0], ij[1])))
    pairs = list(pairs)

    mis = np.array([
        mutual_info_histogram(feat_sub[:, i], feat_sub[:, j], bins=bins)
        for i, j in pairs
    ])
    return float(mis.mean()) if len(mis) > 0 else 0.0, mis


def compute_sparsity(features: np.ndarray) -> float:
    """Fraction of zero (or near-zero) activations."""
    return float((np.abs(features) < 1e-10).mean())


def count_dead_features(features: np.ndarray) -> int:
    """Count features that never activate."""
    return int((np.abs(features).sum(axis=0) < 1e-10).sum())


# ═══════════════════════════════════════════════
# Phase 5: Plots
# ═══════════════════════════════════════════════

def make_plots(all_results: dict, l1_coeffs: list, output_dir: Path):
    """Generate all analysis plots."""

    # --- Plot 1: Main comparison (3 panels) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sae_keys = [f"SAE_L1={l1}" for l1 in l1_coeffs]
    sae_sparsities = [all_results[k]["sparsity"] for k in sae_keys]
    sae_mis = [all_results[k]["mean_mi"] for k in sae_keys]
    sae_corrs = [all_results[k]["mean_abs_corr"] for k in sae_keys]

    baseline_styles = [
        ("PCA", "red", "--"), ("FastICA", "green", "-."), ("Random", "gray", ":")
    ]

    # (a) Sparsity → MI
    ax = axes[0]
    ax.plot(sae_sparsities, sae_mis, "bo-", markersize=8, label="SAE (varying L1)", zorder=5)
    for i, l1 in enumerate(l1_coeffs):
        ax.annotate(f"L1={l1}", (sae_sparsities[i], sae_mis[i]),
                    fontsize=6, textcoords="offset points", xytext=(5, 5))
    for name, color, ls in baseline_styles:
        if name in all_results:
            ax.axhline(y=all_results[name]["mean_mi"], color=color,
                       linestyle=ls, alpha=0.7, label=name)
    ax.set_xlabel("Sparsity (fraction zeros)")
    ax.set_ylabel("Mean Pairwise MI (nats)")
    ax.set_title("Sparsity → Independence (MI)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Sparsity → |correlation|
    ax = axes[1]
    ax.plot(sae_sparsities, sae_corrs, "bo-", markersize=8, label="SAE (varying L1)", zorder=5)
    for name, color, ls in baseline_styles:
        if name in all_results:
            ax.axhline(y=all_results[name]["mean_abs_corr"], color=color,
                       linestyle=ls, alpha=0.7, label=name)
    ax.set_xlabel("Sparsity (fraction zeros)")
    ax.set_ylabel("Mean |Pearson r|")
    ax.set_title("Sparsity → Decorrelation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Bar chart comparison
    ax = axes[2]
    methods = ["PCA", "FastICA", "Random"] + [f"SAE\nL1={l1}" for l1 in l1_coeffs]
    mi_vals = [all_results.get(n, {}).get("mean_mi", 0) for n in ["PCA", "FastICA", "Random"]] + sae_mis
    colors = ["#d62728", "#2ca02c", "#7f7f7f"] + ["#1f77b4"] * len(l1_coeffs)
    ax.bar(range(len(methods)), mi_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean Pairwise MI (nats)")
    ax.set_title("Independence Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "independence_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: independence_analysis.png")

    # --- Plot 2: Correlation distributions ---
    dist_methods = ["PCA", "FastICA", "Random"]
    # Add best and worst SAE
    if sae_keys:
        dist_methods.append(sae_keys[0])  # lowest L1
        dist_methods.append(sae_keys[len(sae_keys) // 2])  # mid L1
        dist_methods.append(sae_keys[-1])  # highest L1

    n_panels = len(dist_methods)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for idx, (name, ax) in enumerate(zip(dist_methods, axes.flat)):
        r = all_results.get(name, {})
        corr_dist = r.get("corr_dist", [])
        if corr_dist:
            ax.hist(corr_dist, bins=50, alpha=0.7, density=True, color="steelblue")
            ax.axvline(x=r["mean_abs_corr"], color="red", linestyle="--",
                       label=f"mean={r['mean_abs_corr']:.4f}")
            ax.legend(fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("|Pearson r|")
        ax.set_ylabel("Density")
    for idx in range(len(dist_methods), 6):
        axes.flat[idx].set_visible(False)

    fig.suptitle("Pairwise |Correlation| Distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: correlation_distributions.png")

    # --- Plot 3: MI distributions ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for idx, (name, ax) in enumerate(zip(dist_methods, axes.flat)):
        r = all_results.get(name, {})
        mi_dist = r.get("mi_dist", [])
        if mi_dist:
            ax.hist(mi_dist, bins=50, alpha=0.7, density=True, color="darkorange")
            ax.axvline(x=r["mean_mi"], color="red", linestyle="--",
                       label=f"mean={r['mean_mi']:.4f}")
            ax.legend(fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("MI (nats)")
        ax.set_ylabel("Density")
    for idx in range(len(dist_methods), 6):
        axes.flat[idx].set_visible(False)

    fig.suptitle("Pairwise Mutual Information Distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "mi_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mi_distributions.png")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp-021: SAE as Empirical ICA")
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-tokens", type=int, default=230_000,
                        help="Number of tokens (<=235K to reuse exp-008 cache)")
    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=20_000)
    parser.add_argument("--n-ica", type=int, default=128,
                        help="Number of ICA/PCA/Random components")
    parser.add_argument("--n-corr-pairs", type=int, default=2000)
    parser.add_argument("--n-mi-pairs", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 65)
    print("  Exp-021: SAE as Empirical ICA")
    print("  Are SAE features statistically independent?")
    print("=" * 65)

    # ── Phase 1: Activations ──
    print(f"\n{'─'*65}")
    print("[Phase 1] Caching Pythia-160M layer 6 activations")
    print(f"{'─'*65}")
    acts = cache_activations(args.cache_dir, output_dir, args.device, args.n_tokens)
    acts_np = acts.numpy()
    n_tokens, d_model = acts.shape
    print(f"  → {n_tokens:,} tokens × {d_model} dims")

    # ── Phase 2: Train SAEs ──
    l1_coeffs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    sae_data = {}

    print(f"\n{'─'*65}")
    print(f"[Phase 2] Training {len(l1_coeffs)} SAEs (d_sae={args.d_sae}, K={args.k})")
    print(f"{'─'*65}")

    for l1 in l1_coeffs:
        label = f"SAE_L1={l1}"
        print(f"\n  ▸ {label} ({args.n_steps} steps)...")
        t0 = time.time()
        sae, acts_mean, acts_std, final_mse = train_sae(
            acts, args.d_sae, args.k, l1, args.n_steps, args.device
        )
        dt = time.time() - t0

        feats = get_sae_features(sae, acts, acts_mean, acts_std, args.device)

        # Quick quality check
        with torch.no_grad():
            check = ((acts[:2000] - acts_mean) / acts_std).to(args.device)
            out = sae(check)
            var_expl = 1.0 - out["mse"].item() / check.var().item()

        sparsity = compute_sparsity(feats)
        n_dead = count_dead_features(feats)

        print(f"    VarExpl={var_expl:.1%}  Sparsity={sparsity:.3f}  "
              f"Dead={n_dead}/{args.d_sae}  Time={dt:.0f}s")

        sae_data[l1] = {
            "features": feats,
            "var_explained": var_expl,
            "sparsity": sparsity,
            "n_dead": n_dead,
            "final_mse": final_mse,
        }

        # Save SAE checkpoint
        torch.save({
            "state_dict": sae.state_dict(),
            "config": {"d_model": d_model, "d_sae": args.d_sae, "k": args.k},
            "acts_mean": acts_mean, "acts_std": acts_std,
            "l1_coeff": l1,
        }, output_dir / f"sae_l1_{l1}.pt")

        del sae
        torch.cuda.empty_cache()

    # ── Phase 3: Baselines ──
    print(f"\n{'─'*65}")
    print(f"[Phase 3] Computing baselines ({args.n_ica} components)")
    print(f"{'─'*65}")

    print("  ▸ PCA...")
    pca_feats = compute_pca(acts_np, args.n_ica)
    print(f"    Shape: {pca_feats.shape}")

    print("  ▸ FastICA...")
    ica_feats, ica_idx = compute_ica(acts_np, args.n_ica)
    print(f"    Shape: {ica_feats.shape} (fitted on {len(ica_idx)} tokens)")

    print("  ▸ Random projection...")
    rand_feats = compute_random_proj(acts_np, args.n_ica)
    print(f"    Shape: {rand_feats.shape}")

    # ── Phase 4: Independence analysis ──
    print(f"\n{'─'*65}")
    print("[Phase 4] Computing independence metrics")
    print(f"{'─'*65}")

    all_results = {}

    # Baselines
    baseline_feats = {
        "PCA": pca_feats,
        "FastICA": ica_feats,
        "Random": rand_feats,
    }

    for name, feats in baseline_feats.items():
        print(f"\n  ▸ {name} ({feats.shape[1]} features, {feats.shape[0]} tokens)...")
        t0 = time.time()
        mean_corr, corr_dist = pairwise_abs_correlation(feats, n_pairs=args.n_corr_pairs)
        mean_mi, mi_dist = pairwise_mi(feats, n_pairs=args.n_mi_pairs)
        sparsity = compute_sparsity(feats)
        dt = time.time() - t0

        print(f"    Mean|r|={mean_corr:.4f}  MeanMI={mean_mi:.4f}  "
              f"Sparsity={sparsity:.4f}  ({dt:.1f}s)")

        all_results[name] = {
            "mean_abs_corr": mean_corr,
            "mean_mi": mean_mi,
            "sparsity": sparsity,
            "n_features": int(feats.shape[1]),
            "n_tokens": int(feats.shape[0]),
            "corr_dist": corr_dist.tolist(),
            "mi_dist": mi_dist.tolist(),
        }

    # SAEs
    for l1 in l1_coeffs:
        name = f"SAE_L1={l1}"
        sd = sae_data[l1]
        feats = sd["features"]
        print(f"\n  ▸ {name} ({feats.shape[1]} features, {feats.shape[0]} tokens)...")
        t0 = time.time()
        mean_corr, corr_dist = pairwise_abs_correlation(feats, n_pairs=args.n_corr_pairs)
        mean_mi, mi_dist = pairwise_mi(feats, n_pairs=args.n_mi_pairs)
        dt = time.time() - t0

        print(f"    Mean|r|={mean_corr:.4f}  MeanMI={mean_mi:.4f}  "
              f"Sparsity={sd['sparsity']:.4f}  ({dt:.1f}s)")

        all_results[name] = {
            "mean_abs_corr": mean_corr,
            "mean_mi": mean_mi,
            "sparsity": sd["sparsity"],
            "var_explained": sd["var_explained"],
            "n_dead": sd["n_dead"],
            "final_mse": sd["final_mse"],
            "n_features": int(feats.shape[1]),
            "n_tokens": int(feats.shape[0]),
            "corr_dist": corr_dist.tolist(),
            "mi_dist": mi_dist.tolist(),
        }

    # Save raw results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # ── Phase 5: Plots ──
    print(f"\n{'─'*65}")
    print("[Phase 5] Generating plots")
    print(f"{'─'*65}")
    make_plots(all_results, l1_coeffs, output_dir)

    # ── Summary ──
    total_time = time.time() - t_start
    print(f"\n{'═'*65}")
    print("  SUMMARY")
    print(f"{'═'*65}")
    header = f"{'Method':<20} {'#Feat':>6} {'Sparsity':>10} {'Mean|r|':>10} {'MeanMI':>10}"
    print(header)
    print("─" * len(header))

    for name in ["PCA", "FastICA", "Random"]:
        r = all_results[name]
        print(f"{name:<20} {r['n_features']:>6} {r['sparsity']:>10.4f} "
              f"{r['mean_abs_corr']:>10.4f} {r['mean_mi']:>10.4f}")

    print("─" * len(header))
    for l1 in l1_coeffs:
        name = f"SAE_L1={l1}"
        r = all_results[name]
        print(f"{name:<20} {r['n_features']:>6} {r['sparsity']:>10.4f} "
              f"{r['mean_abs_corr']:>10.4f} {r['mean_mi']:>10.4f}")

    print(f"\nTotal time: {total_time / 60:.1f} min")
    print(f"Output: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
