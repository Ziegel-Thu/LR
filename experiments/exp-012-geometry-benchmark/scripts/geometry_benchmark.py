#!/usr/bin/env python3
"""
Exp-012: Geometry benchmark — ID estimator comparison + participation ratio.

Extends exp-006 Phase 1 data with:
  - Multiple MLE K values (10, 50, 100, 500)
  - Participation ratio
  - Cross-estimator rank correlation
  - All metrics vs validation loss correlation

Usage:
  python geometry_benchmark.py --data-dir /nvmessd/lifanhong/LR-env/exp003_reps
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
FIG_DIR = EXP_DIR / "figures"

PYTHIA_SCALES = [
    ("Pythia-70M", 70e6, 3.64),
    ("Pythia-160M", 160e6, 3.28),
    ("Pythia-410M", 410e6, 2.94),
    ("Pythia-1B", 1e9, 2.68),
    ("Pythia-1.4B", 1.4e9, 2.56),
    ("Pythia-2.8B", 2.8e9, 2.40),
    ("Pythia-6.9B", 6.9e9, 2.21),
]


def twonn_id(X):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X)
    distances, _ = nn.kneighbors(X)
    r1, r2 = distances[:, 1], distances[:, 2]
    mask = r1 > 1e-10
    mu = r2[mask] / r1[mask]
    log_sum = np.sum(np.log(mu))
    return float(len(mu) / log_sum) if log_sum > 1e-10 else 0.0


def mle_id(X, k=100):
    from sklearn.neighbors import NearestNeighbors
    k_actual = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean").fit(X)
    distances, _ = nn.kneighbors(X)
    ids = []
    for i in range(len(X)):
        dists = distances[i, 1:]
        dists = dists[dists > 1e-10]
        if len(dists) < 2:
            continue
        r_k = dists[-1]
        if r_k < 1e-10:
            continue
        d_hat = (len(dists) - 1) / np.sum(np.log(r_k / dists[:-1]))
        ids.append(d_hat)
    return float(np.mean(ids)) if ids else 0.0


def stable_rank(X):
    sv = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    sv2 = sv ** 2
    return float(sv2.sum() / (sv2[0] + 1e-10))


def effective_rank(X):
    sv = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    sv2 = sv ** 2
    p = sv2 / (sv2.sum() + 1e-10)
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio(X):
    """PR = (Σσ²)² / Σσ⁴. Measures how many dimensions contribute."""
    sv = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    sv2 = sv ** 2
    return float(sv2.sum() ** 2 / (np.sum(sv2 ** 2) + 1e-10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load reps
    reps = {}
    available = []
    for name, params, loss in PYTHIA_SCALES:
        path = data_dir / f"{name}_reps.pt"
        if path.exists():
            r = torch.load(path, map_location="cpu", weights_only=True).numpy()
            reps[name] = r
            available.append((name, params, loss))
            print(f"Loaded {name}: {r.shape}")

    # ── Phase 1: Compute all metrics at middle layer ──
    print(f"\n{'='*70}")
    print("Phase 1: Multi-metric comparison")
    print(f"{'='*70}")

    k_values = [10, 50, 100, 500]
    all_metrics = {}

    for name, params, loss in available:
        R = reps[name]
        mid = R.shape[0] // 2
        X = R[mid]

        metrics = {
            "params": params,
            "loss": loss,
            "twonn_id": twonn_id(X),
            "stable_rank": stable_rank(X),
            "effective_rank": effective_rank(X),
            "participation_ratio": participation_ratio(X),
        }
        for k in k_values:
            metrics[f"mle_id_k{k}"] = mle_id(X, k=k)

        all_metrics[name] = metrics
        print(f"  {name}: twonn={metrics['twonn_id']:.1f}, "
              f"srank={metrics['stable_rank']:.1f}, "
              f"erank={metrics['effective_rank']:.1f}, "
              f"PR={metrics['participation_ratio']:.1f}")

    # ── Phase 2: Estimator consistency ──
    print(f"\n{'='*70}")
    print("Phase 2: ID estimator consistency")
    print(f"{'='*70}")

    metric_names = ["twonn_id"] + [f"mle_id_k{k}" for k in k_values]
    n_metrics = len(metric_names)
    models = [name for name, _, _ in available]

    # Rank correlation matrix between estimators
    values_matrix = np.zeros((n_metrics, len(models)))
    for mi, mname in enumerate(metric_names):
        for mj, model in enumerate(models):
            values_matrix[mi, mj] = all_metrics[model][mname]

    rank_corr = np.zeros((n_metrics, n_metrics))
    for i in range(n_metrics):
        for j in range(n_metrics):
            rho, _ = stats.spearmanr(values_matrix[i], values_matrix[j])
            rank_corr[i, j] = rho

    print("  Spearman rank correlation between ID estimators:")
    print(f"  {'':>12}", end="")
    for m in metric_names:
        print(f"  {m[-6:]:>8}", end="")
    print()
    for i, mi in enumerate(metric_names):
        print(f"  {mi[-12:]:>12}", end="")
        for j in range(n_metrics):
            print(f"  {rank_corr[i,j]:>8.3f}", end="")
        print()

    # ── Phase 3: Correlation with loss ──
    print(f"\n{'='*70}")
    print("Phase 3: Metrics vs validation loss")
    print(f"{'='*70}")

    all_metric_names = ["twonn_id", "stable_rank", "effective_rank",
                        "participation_ratio"] + [f"mle_id_k{k}" for k in k_values]
    losses = [all_metrics[m]["loss"] for m in models]

    loss_correlations = {}
    for mname in all_metric_names:
        vals = [all_metrics[m][mname] for m in models]
        r, p = stats.pearsonr(losses, vals)
        rho, p_rho = stats.spearmanr(losses, vals)
        loss_correlations[mname] = {"pearson_r": r, "pearson_p": p,
                                     "spearman_rho": rho, "spearman_p": p_rho}
        print(f"  {mname:>20}: Pearson r={r:>7.4f} (p={p:.4f}), Spearman ρ={rho:>7.4f}")

    # ── Figures ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. All metrics vs params
    ax = axes[0, 0]
    params_list = [all_metrics[m]["params"] for m in models]
    for mname, color in zip(["twonn_id", "stable_rank", "effective_rank", "participation_ratio"],
                             ["blue", "red", "green", "orange"]):
        vals = [all_metrics[m][mname] for m in models]
        ax.plot(params_list, vals, "o-", label=mname, color=color, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Metric value")
    ax.set_title("Geometry Metrics vs Scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. MLE ID K sensitivity
    ax = axes[0, 1]
    for name in models:
        vals = [all_metrics[name][f"mle_id_k{k}"] for k in k_values]
        ax.plot(k_values, vals, "o-", label=name.replace("Pythia-", ""), markersize=4, alpha=0.7)
    ax.set_xlabel("K (MLE neighbors)")
    ax.set_ylabel("MLE ID estimate")
    ax.set_title("MLE ID: K Sensitivity")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Estimator rank correlation heatmap
    ax = axes[0, 2]
    im = ax.imshow(rank_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_metrics))
    ax.set_yticks(range(n_metrics))
    short_names = ["2NN"] + [f"MLE-{k}" for k in k_values]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("ID Estimator Rank Correlation")

    # 4-6. Each metric vs loss
    for mi, (mname, color) in enumerate(zip(
            ["stable_rank", "participation_ratio", "twonn_id"],
            ["red", "orange", "blue"])):
        ax = axes[1, mi]
        vals = [all_metrics[m][mname] for m in models]
        ax.scatter(losses, vals, s=80, color=color, zorder=5)
        for x, y, m in zip(losses, vals, models):
            ax.annotate(m.replace("Pythia-", ""), (x, y), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))
        r = loss_correlations[mname]["pearson_r"]
        ax.set_xlabel("Validation Loss")
        ax.set_ylabel(mname)
        ax.set_title(f"{mname} vs Loss (r={r:.3f})")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp-012: Geometry Metrics Benchmark", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "geometry_benchmark.png", dpi=150)
    print(f"\nFigure: {FIG_DIR / 'geometry_benchmark.png'}")

    # Save
    results = {
        "models": models,
        "metrics": all_metrics,
        "estimator_rank_correlation": {
            "metrics": metric_names,
            "matrix": rank_corr.tolist(),
        },
        "loss_correlations": loss_correlations,
    }
    out_path = EXP_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
