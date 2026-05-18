#!/usr/bin/env python3
"""
Exp-006 Phase 3: Representation alignment vs validation loss.

Uses Pythia's known validation losses and the kNN alignment scores
from Phase 1 to analyze the relationship.

Pythia validation losses from the official tech report / model cards.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
FIG_DIR = EXP_DIR / "figures"

# Pythia validation losses (Pile validation, from Biderman et al. 2023 Table 5)
# These are final checkpoint losses
PYTHIA_LOSSES = {
    "Pythia-70M": 3.64,
    "Pythia-160M": 3.28,
    "Pythia-410M": 2.94,
    "Pythia-1B": 2.68,
    "Pythia-1.4B": 2.56,
    "Pythia-2.8B": 2.40,
    "Pythia-6.9B": 2.21,
}

PARAMS = {
    "Pythia-70M": 70e6,
    "Pythia-160M": 160e6,
    "Pythia-410M": 410e6,
    "Pythia-1B": 1e9,
    "Pythia-1.4B": 1.4e9,
    "Pythia-2.8B": 2.8e9,
    "Pythia-6.9B": 6.9e9,
}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load Phase 1 results
    p1_path = EXP_DIR / "results_phase1.json"
    if not p1_path.exists():
        print(f"ERROR: {p1_path} not found")
        return

    with open(p1_path) as f:
        p1 = json.load(f)

    # Extract intra-model metrics at middle layer
    intra = p1["intra_model"]
    cross = p1["cross_model"]

    models = sorted(intra.keys(), key=lambda m: intra[m]["params"])
    losses = [PYTHIA_LOSSES[m] for m in models]
    params = [PARAMS[m] for m in models]

    # Intra-model metrics at middle layer
    metrics_at_mid = {}
    for metric in ["stable_rank", "effective_rank", "twonn_id", "mle_id"]:
        vals = []
        for m in models:
            mid = len(intra[m]["layers"]) // 2
            vals.append(intra[m]["layers"][mid][metric])
        metrics_at_mid[metric] = vals

    # Cross-model kNN (consecutive pairs → assign to geometric mean params)
    cross_pairs = list(cross.keys())
    cross_knn = [cross[pk]["best_knn"] for pk in cross_pairs]
    cross_params = []
    cross_losses_avg = []
    for pk in cross_pairs:
        ma = cross[pk]["model_a"]
        mb = cross[pk]["model_b"]
        geo_p = np.sqrt(PARAMS[ma] * PARAMS[mb])
        avg_loss = (PYTHIA_LOSSES[ma] + PYTHIA_LOSSES[mb]) / 2
        cross_params.append(geo_p)
        cross_losses_avg.append(avg_loss)

    # ── Analysis ──

    print("=" * 60)
    print("PHASE 3: Representation Metrics vs Validation Loss")
    print("=" * 60)

    # 1. Intra metrics vs loss
    print("\n--- Intra-model metrics vs loss (Pearson r) ---")
    correlations = {}
    for metric, vals in metrics_at_mid.items():
        r, p = stats.pearsonr(losses, vals)
        rho, p_rho = stats.spearmanr(losses, vals)
        correlations[metric] = {"pearson_r": r, "pearson_p": p,
                                "spearman_rho": rho, "spearman_p": p_rho}
        print(f"  {metric}: Pearson r={r:.4f} (p={p:.4f}), Spearman ρ={rho:.4f}")

    # 2. Cross-model kNN vs average loss
    r_knn, p_knn = stats.pearsonr(cross_losses_avg, cross_knn)
    rho_knn, p_rho_knn = stats.spearmanr(cross_losses_avg, cross_knn)
    correlations["cross_knn_vs_avg_loss"] = {
        "pearson_r": r_knn, "pearson_p": p_knn,
        "spearman_rho": rho_knn, "spearman_p": p_rho_knn,
    }
    print(f"\n  cross kNN vs avg_loss: Pearson r={r_knn:.4f} (p={p_knn:.4f}), "
          f"Spearman ρ={rho_knn:.4f}")

    # 3. Loss scaling law: L(N) = a * N^{-alpha}
    log_params = np.log(params)
    slope, intercept, r_loss, p_loss, _ = stats.linregress(log_params, losses)
    print(f"\n  Loss vs log(N): slope={slope:.4f}, R²={r_loss**2:.4f}")

    # ── Figures ──

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Intra metrics vs loss
    for mi, (metric, vals) in enumerate(metrics_at_mid.items()):
        ax = axes[0, mi] if mi < 3 else axes[1, mi - 3]
        ax.scatter(losses, vals, s=80, zorder=5)
        for x, y, m in zip(losses, vals, models):
            ax.annotate(m.replace("Pythia-", ""), (x, y), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))

        # Trend line
        z = np.polyfit(losses, vals, 1)
        x_line = np.linspace(min(losses), max(losses), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="red", alpha=0.5)

        r = correlations[metric]["pearson_r"]
        ax.set_xlabel("Validation Loss")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric} vs Loss (r={r:.3f})")
        ax.grid(True, alpha=0.3)

    # Row 1, col 3: Loss scaling law
    ax = axes[0, 2]
    ax.scatter(params, losses, s=80, zorder=5)
    for x, y, m in zip(params, losses, models):
        ax.annotate(m.replace("Pythia-", ""), (x, y), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
    x_fit = np.logspace(np.log10(min(params)), np.log10(max(params)), 100)
    ax.plot(x_fit, slope * np.log(x_fit) + intercept, "--", color="red", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Validation Loss")
    ax.set_title(f"Loss Scaling (R²={r_loss**2:.3f})")
    ax.grid(True, alpha=0.3)

    # Row 2: Cross kNN vs loss
    ax = axes[1, 0]
    ax.scatter(cross_losses_avg, cross_knn, s=80, zorder=5)
    for x, y, pk in zip(cross_losses_avg, cross_knn, cross_pairs):
        label = pk.replace("Pythia-", "").replace(" ↔ ", "↔")
        ax.annotate(label, (x, y), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
    z = np.polyfit(cross_losses_avg, cross_knn, 1)
    x_line = np.linspace(min(cross_losses_avg), max(cross_losses_avg), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="red", alpha=0.5)
    ax.set_xlabel("Average Validation Loss (pair)")
    ax.set_ylabel("Best kNN Overlap")
    ax.set_title(f"kNN Alignment vs Loss (r={r_knn:.3f})")
    ax.grid(True, alpha=0.3)

    # Row 2: kNN vs params (direct)
    ax = axes[1, 1]
    ax.scatter(cross_params, cross_knn, s=80, zorder=5)
    for x, y, pk in zip(cross_params, cross_knn, cross_pairs):
        label = pk.replace("Pythia-", "").replace(" ↔ ", "↔")
        ax.annotate(label, (x, y), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xscale("log")
    ax.set_xlabel("Geometric Mean Parameters")
    ax.set_ylabel("Best kNN Overlap")
    ax.set_title("kNN Alignment vs Scale")
    ax.grid(True, alpha=0.3)

    # Row 2: repr alignment vs loss (normalized)
    ax = axes[1, 2]
    # Normalize both to [0,1]
    knn_norm = (np.array(cross_knn) - min(cross_knn)) / (max(cross_knn) - min(cross_knn) + 1e-10)
    loss_norm = (np.array(cross_losses_avg) - min(cross_losses_avg)) / (max(cross_losses_avg) - min(cross_losses_avg) + 1e-10)
    ax.scatter(1 - loss_norm, knn_norm, s=80, zorder=5)  # 1-loss so higher=better
    for x, y, pk in zip(1 - loss_norm, knn_norm, cross_pairs):
        label = pk.replace("Pythia-", "").replace(" ↔ ", "↔")
        ax.annotate(label, (x, y), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="y=x")
    ax.set_xlabel("Normalized Loss Improvement (1 - norm_loss)")
    ax.set_ylabel("Normalized kNN Alignment")
    ax.set_title("Repr Alignment vs Loss (normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Exp-006 Phase 3: Representation Metrics vs Validation Loss", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "repr_vs_loss.png", dpi=150)
    print(f"\nFigure: {FIG_DIR / 'repr_vs_loss.png'}")

    # Save
    results = {
        "pythia_losses": PYTHIA_LOSSES,
        "intra_correlations": correlations,
        "cross_knn_vs_loss": {
            "pairs": cross_pairs,
            "knn": cross_knn,
            "avg_losses": cross_losses_avg,
            "pearson_r": r_knn,
            "spearman_rho": rho_knn,
        },
        "loss_scaling": {"slope": slope, "intercept": intercept, "r2": r_loss**2},
    }
    out_path = EXP_DIR / "results_phase3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
