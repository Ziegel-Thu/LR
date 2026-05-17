#!/usr/bin/env python3
"""
Exp-006 Phase 1: Fit power laws to representation scaling metrics.

Reads results_phase1.json, fits s(N) = s∞ - a·N^{-β} for each metric,
generates scaling curve figures.

Usage:
  python fit_power_law.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
FIG_DIR = EXP_DIR / "figures"


def power_law_decay(N, s_inf, a, beta):
    """s(N) = s_inf - a * N^{-beta}"""
    return s_inf - a * np.power(N, -beta)


def power_law_growth(N, a, beta):
    """s(N) = a * N^{beta} (for metrics that grow with scale)"""
    return a * np.power(N, beta)


def fit_power_law(x, y, mode="decay"):
    """Fit power law and return params + R²."""
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    try:
        if mode == "decay":
            # s(N) = s_inf - a * N^{-beta}, decreasing towards s_inf
            popt, _ = curve_fit(
                power_law_decay, x, y,
                p0=[y[-1], y[0] - y[-1], 0.3],
                bounds=([min(y) * 0.5, 0, 0.01], [max(y) * 2, max(y) * 10, 5.0]),
                maxfev=10000,
            )
            y_pred = power_law_decay(x, *popt)
            params = {"s_inf": popt[0], "a": popt[1], "beta": popt[2]}
        else:
            # s(N) = a * N^{beta}, increasing
            popt, _ = curve_fit(
                power_law_growth, x, y,
                p0=[y[0], 0.3],
                bounds=([0, 0.001], [max(y) * 100, 5.0]),
                maxfev=10000,
            )
            y_pred = power_law_growth(x, *popt)
            params = {"a": popt[0], "beta": popt[1]}

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        params["r2"] = r2
        return params, y_pred

    except (RuntimeError, ValueError) as e:
        return {"error": str(e), "r2": 0.0}, y


def main():
    results_path = EXP_DIR / "results_phase1.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run compute_scaling_metrics.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    intra = results["intra_model"]
    cross = results.get("cross_model", {})

    # ── 1. Intra-model scaling: metric at middle layer vs params ──
    models = sorted(intra.keys(), key=lambda m: intra[m]["params"])
    params_list = [intra[m]["params"] for m in models]

    intra_metrics = ["stable_rank", "effective_rank", "twonn_id", "mle_id"]
    metric_labels = {
        "stable_rank": "Stable Rank",
        "effective_rank": "Effective Rank",
        "twonn_id": "TwoNN ID",
        "mle_id": "MLE ID",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    fit_results = {}
    for mi, metric in enumerate(intra_metrics):
        mid_values = []
        mean_values = []
        for m in models:
            layers = intra[m]["layers"]
            mid_idx = len(layers) // 2
            mid_values.append(layers[mid_idx][metric])
            mean_values.append(np.mean([l[metric] for l in layers]))

        # Fit power law (growth mode for ID/rank metrics)
        fit, y_pred = fit_power_law(params_list, mid_values, mode="growth")
        fit_results[f"mid_{metric}"] = fit

        # Plot
        ax = axes[mi]
        ax.plot(params_list, mid_values, "o-", color="blue", label="Middle layer", markersize=8)
        ax.plot(params_list, mean_values, "s--", color="gray", alpha=0.5, label="Layer mean", markersize=6)
        if "error" not in fit:
            x_fit = np.logspace(np.log10(min(params_list)), np.log10(max(params_list)), 100)
            y_fit = power_law_growth(x_fit, fit["a"], fit["beta"])
            ax.plot(x_fit, y_fit, "--", color="red", alpha=0.7,
                    label=f"Power law β={fit['beta']:.3f}, R²={fit['r2']:.3f}")

        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f"{metric_labels[metric]} vs Model Scale")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add model names
        for x, y, m in zip(params_list, mid_values, models):
            ax.annotate(m.replace("Pythia-", ""), (x, y), fontsize=7,
                        textcoords="offset points", xytext=(0, 8), ha="center")

    fig.suptitle("Exp-006: Intra-Model Representation Metrics vs Scale", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intra_scaling.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'intra_scaling.png'}")

    # ── 2. Cross-model scaling: alignment vs params ──
    if cross:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        cross_metrics = ["cka", "knn", "shape_distance"]
        cross_labels = {"cka": "Best CKA", "knn": "Best kNN Overlap", "shape_distance": "Best Shape Distance"}
        cross_keys = {"cka": "best_cka", "knn": "best_knn", "shape_distance": "best_shape"}

        for mi, metric in enumerate(cross_metrics):
            pair_names = list(cross.keys())
            # Use geometric mean of the two model sizes as x
            x_vals = []
            y_vals = []
            labels = []
            for pk in pair_names:
                cr = cross[pk]
                ma = intra.get(cr["model_a"], {})
                mb = intra.get(cr["model_b"], {})
                if ma and mb:
                    geo_mean = np.sqrt(ma["params"] * mb["params"])
                    x_vals.append(geo_mean)
                    y_vals.append(cr[cross_keys[metric]])
                    labels.append(pk.replace("Pythia-", "").replace(" ↔ ", "↔"))

            ax = axes2[mi]
            ax.plot(x_vals, y_vals, "o-", color="blue", markersize=8)
            for x, y, lab in zip(x_vals, y_vals, labels):
                ax.annotate(lab, (x, y), fontsize=7,
                            textcoords="offset points", xytext=(0, 8), ha="center")

            # Fit
            if len(x_vals) >= 3:
                mode = "decay" if metric == "shape_distance" else "growth"
                fit, _ = fit_power_law(x_vals, y_vals, mode=mode)
                fit_results[f"cross_{metric}"] = fit
                if "error" not in fit:
                    x_fit = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 100)
                    if mode == "growth":
                        y_fit = power_law_growth(x_fit, fit["a"], fit["beta"])
                    else:
                        y_fit = power_law_decay(x_fit, fit["s_inf"], fit["a"], fit["beta"])
                    ax.plot(x_fit, y_fit, "--", color="red", alpha=0.7,
                            label=f"β={fit.get('beta', 0):.3f}, R²={fit['r2']:.3f}")
                    ax.legend(fontsize=8)

            ax.set_xscale("log")
            ax.set_xlabel("Geometric mean params")
            ax.set_ylabel(cross_labels[metric])
            ax.set_title(f"{cross_labels[metric]} vs Scale")
            ax.grid(True, alpha=0.3)

        fig2.suptitle("Exp-006: Cross-Scale Alignment Metrics", fontsize=14)
        fig2.tight_layout()
        fig2.savefig(FIG_DIR / "cross_scaling.png", dpi=150)
        print(f"Saved: {FIG_DIR / 'cross_scaling.png'}")

    # ── 3. Layer-wise depth profiles for each scale ──
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    axes3 = axes3.flatten()
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    for mi, metric in enumerate(intra_metrics):
        ax = axes3[mi]
        for ci, m in enumerate(models):
            layers = intra[m]["layers"]
            depths = np.linspace(0, 1, len(layers))
            vals = [l[metric] for l in layers]
            ax.plot(depths, vals, "-", color=colors[ci], label=m.replace("Pythia-", ""), linewidth=1.5)
        ax.set_xlabel("Normalized depth")
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f"{metric_labels[metric]} Depth Profile")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig3.suptitle("Exp-006: Layer-wise Metrics Across Pythia Scales", fontsize=14)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "depth_profiles.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'depth_profiles.png'}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("POWER LAW FIT SUMMARY")
    print(f"{'='*70}")
    for key, fit in fit_results.items():
        if "error" in fit:
            print(f"  {key}: FAILED ({fit['error']})")
        else:
            beta = fit.get("beta", "N/A")
            r2 = fit.get("r2", "N/A")
            print(f"  {key}: β={beta:.4f}, R²={r2:.4f}")

    # Save
    out_path = EXP_DIR / "results_power_law.json"
    with open(out_path, "w") as f:
        json.dump(fit_results, f, indent=2, default=str)
    print(f"\nFit results saved: {out_path}")


if __name__ == "__main__":
    main()
