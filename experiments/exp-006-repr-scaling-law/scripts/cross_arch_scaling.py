#!/usr/bin/env python3
"""
Exp-006 Phase 2: Multi-architecture scaling analysis.

Computes cross-architecture alignment metrics (CKA, kNN, shape) at 4 scale tiers:
  ~160M: Pythia-160M / Mamba-130M / RWKV-169M
  ~400M: Pythia-410M / Mamba-370M / RWKV-430M
  ~1.4B: Pythia-1.4B / Mamba-1.4B / RWKV-1.5B
  ~2.8B: Pythia-2.8B / Mamba-2.8B / RWKV-3B

Separates intra-family (SSM↔SSM) vs inter-family (Transformer↔SSM) pairs
and fits power laws to each category.

Usage:
  python cross_arch_scaling.py --data-dir /nvmessd/lifanhong/LR-env/exp003_reps
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
FIG_DIR = EXP_DIR / "figures"

SCALE_TIERS = {
    "160M": {
        "params": 160e6,
        "models": {
            "Pythia": "Pythia-160M_reps.pt",
            "Mamba": "Mamba-130M_reps.pt",
            "RWKV": "RWKV-169M_reps.pt",
        },
    },
    "400M": {
        "params": 400e6,
        "models": {
            "Pythia": "Pythia-410M_reps.pt",
            "Mamba": "Mamba-370M_reps.pt",
            "RWKV": "RWKV-430M_reps.pt",
        },
    },
    "1.4B": {
        "params": 1.4e9,
        "models": {
            "Pythia": "Pythia-1.4B_reps.pt",
            "Mamba": "Mamba-1.4B_reps.pt",
            "RWKV": "RWKV-1.5B_reps.pt",
        },
    },
    "2.8B": {
        "params": 2.8e9,
        "models": {
            "Pythia": "Pythia-2.8B_reps.pt",
            "Mamba": "Mamba-2.8B_reps.pt",
            "RWKV": "RWKV-4-3B_reps.pt",
        },
    },
}

PAIR_TYPES = {
    "Pythia ↔ Mamba": "inter",
    "Pythia ↔ RWKV": "inter",
    "Mamba ↔ RWKV": "intra",
}

PAIR_COLORS = {
    "Pythia ↔ Mamba": "blue",
    "Pythia ↔ RWKV": "green",
    "Mamba ↔ RWKV": "red",
}


def mutual_knn(X, Y, k=10):
    X_t = F.normalize(torch.tensor(X, dtype=torch.float32), dim=1)
    Y_t = F.normalize(torch.tensor(Y, dtype=torch.float32), dim=1)
    sim_XX = X_t @ X_t.T
    sim_YY = Y_t @ Y_t.T
    sim_XX.fill_diagonal_(-1)
    sim_YY.fill_diagonal_(-1)
    nn_X = sim_XX.topk(k, dim=1).indices.numpy()
    nn_Y = sim_YY.topk(k, dim=1).indices.numpy()
    return float(np.mean([len(set(nn_X[i]) & set(nn_Y[i])) / k for i in range(len(X))]))


def linear_cka(X, Y):
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    hsic_xx = np.linalg.norm(Xc.T @ Xc, "fro") ** 2
    hsic_yy = np.linalg.norm(Yc.T @ Yc, "fro") ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def shape_distance(X, Y):
    from netrep.metrics import LinearMetric
    metric = LinearMetric(alpha=1.0, center_columns=True)
    metric.fit(X, Y)
    return float(metric.score(X, Y))


def compare_pair(reps_a, reps_b, k=10):
    """Compare two models at best-aligned depth. Returns dict of metrics."""
    n_la, n_lb = reps_a.shape[0], reps_b.shape[0]
    n_compare = min(n_la, n_lb)

    best = {"knn": 0, "cka": 0, "shape": float("inf"),
            "knn_alpha": 0, "cka_alpha": 0, "shape_alpha": 0}

    for ci in range(n_compare):
        la = int(ci * (n_la - 1) / max(n_compare - 1, 1))
        lb = int(ci * (n_lb - 1) / max(n_compare - 1, 1))
        alpha = ci / max(n_compare - 1, 1)

        X = reps_a[la]
        Y = reps_b[lb]

        knn = mutual_knn(X, Y, k=k)
        cka = linear_cka(X, Y)
        shp = shape_distance(X, Y)

        if knn > best["knn"]:
            best["knn"] = knn
            best["knn_alpha"] = alpha
        if cka > best["cka"]:
            best["cka"] = cka
            best["cka_alpha"] = alpha
        if shp < best["shape"]:
            best["shape"] = shp
            best["shape_alpha"] = alpha

    return best


def power_law_growth(N, a, beta):
    return a * np.power(N, beta)


def fit_power_law(x, y):
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    try:
        popt, _ = curve_fit(power_law_growth, x, y, p0=[y[0], 0.1],
                            bounds=([0, 0.001], [max(y)*100, 5.0]), maxfev=10000)
        y_pred = power_law_growth(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return {"a": popt[0], "beta": popt[1], "r2": 1 - ss_res/(ss_tot+1e-10)}
    except Exception as e:
        return {"error": str(e), "r2": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--knn-k", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load all reps
    all_reps = {}
    for tier, cfg in SCALE_TIERS.items():
        all_reps[tier] = {}
        for arch, fname in cfg["models"].items():
            path = data_dir / fname
            if not path.exists():
                print(f"MISSING: {path}")
                continue
            r = torch.load(path, map_location="cpu", weights_only=True)
            all_reps[tier][arch] = r.numpy()
            print(f"Loaded {tier}/{arch}: {r.shape}")

    # Compute pairwise metrics per tier
    results = {}
    for tier in SCALE_TIERS:
        results[tier] = {}
        archs = list(all_reps[tier].keys())
        pairs = [(archs[i], archs[j]) for i in range(len(archs)) for j in range(i+1, len(archs))]

        for a, b in pairs:
            pk = f"{a} ↔ {b}"
            print(f"\n[{tier}] {pk}")
            t0 = time.time()
            res = compare_pair(all_reps[tier][a], all_reps[tier][b], k=args.knn_k)
            elapsed = time.time() - t0
            res["elapsed_s"] = elapsed
            res["pair_type"] = PAIR_TYPES.get(pk, "unknown")
            results[tier][pk] = res
            print(f"  kNN={res['knn']:.4f}  CKA={res['cka']:.4f}  shape={res['shape']:.4f}  ({elapsed:.0f}s)")

    # Summary
    print(f"\n{'='*80}")
    print("CROSS-ARCHITECTURE SCALING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Tier':>6} {'Pair':>20} {'Type':>6} {'kNN':>8} {'CKA':>8} {'Shape':>8}")
    print("-" * 60)
    for tier in SCALE_TIERS:
        for pk, res in results[tier].items():
            print(f"{tier:>6} {pk:>20} {res['pair_type']:>6} "
                  f"{res['knn']:>8.4f} {res['cka']:>8.4f} {res['shape']:>8.4f}")

    # Fit power laws: intra vs inter
    print(f"\n{'='*80}")
    print("POWER LAW FITS: intra-family vs inter-family")
    print(f"{'='*80}")

    fit_results = {}
    for metric in ["knn", "cka"]:
        for ptype in ["intra", "inter"]:
            xs, ys = [], []
            for tier, cfg in SCALE_TIERS.items():
                for pk, res in results[tier].items():
                    if res["pair_type"] == ptype:
                        xs.append(cfg["params"])
                        ys.append(res[metric])
            if len(xs) >= 3:
                fit = fit_power_law(xs, ys)
                key = f"{metric}_{ptype}"
                fit_results[key] = fit
                print(f"  {key}: β={fit.get('beta','N/A'):.4f}, R²={fit.get('r2',0):.4f}")

    # Figure: kNN scaling, intra vs inter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for mi, metric in enumerate(["knn", "cka", "shape"]):
        ax = axes[mi]
        for pk in PAIR_TYPES:
            xs, ys = [], []
            for tier, cfg in SCALE_TIERS.items():
                if pk in results[tier]:
                    xs.append(cfg["params"])
                    ys.append(results[tier][pk][metric])
            if xs:
                ax.plot(xs, ys, "o-", label=pk, color=PAIR_COLORS.get(pk, "gray"), markersize=8)

        ax.set_xscale("log")
        ax.set_xlabel("Model scale (params)")
        labels = {"knn": "Mutual kNN Overlap", "cka": "Linear CKA", "shape": "Shape Distance"}
        ax.set_ylabel(labels[metric])
        ax.set_title(f"Cross-Architecture {labels[metric]} vs Scale")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp-006 Phase 2: Multi-Architecture Scaling", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cross_arch_scaling.png", dpi=150)
    print(f"\nFigure: {FIG_DIR / 'cross_arch_scaling.png'}")

    # Save
    output = {"results": results, "power_law_fits": fit_results}
    out_path = EXP_DIR / "results_phase2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
