"""
Exp-003 Phase 1: Cross-architecture scaling analysis
Compare representations across 3 architectures × 3 scales (410M, 1.4B, 2.8B)

Usage:
  python compare_scaling.py                     # full analysis, 20 perms
  python compare_scaling.py --n-perms 100       # full permutation null
  python compare_scaling.py --scales 2.8B       # single scale only
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
import time

PHASE_DIR = Path(__file__).resolve().parent
DATA_DIR = PHASE_DIR / "data"
FIG_DIR = PHASE_DIR / "figures"

SCALES = {
    "410M": {
        "Pythia": "Pythia-410M_reps.pt",
        "Mamba": "Mamba-370M_reps.pt",
        "RWKV": "RWKV-430M_reps.pt",
    },
    "1.4B": {
        "Pythia": "Pythia-1.4B_reps.pt",
        "Mamba": "Mamba-1.4B_reps.pt",
        "RWKV": "RWKV-1.5B_reps.pt",
    },
    "2.8B": {
        "Pythia": "Pythia-2.8B_reps.pt",
        "Mamba": "Mamba-2.8B_reps.pt",
        "RWKV": "RWKV-4-3B_reps.pt",
    },
}

PILOT_KNN_Z = {
    "Pythia ↔ Mamba": 379.4,
    "Pythia ↔ RWKV": 353.5,
    "Mamba ↔ RWKV": 436.0,
}

PAIR_COLORS = {
    "Pythia ↔ Mamba": "blue",
    "Pythia ↔ RWKV": "green",
    "Mamba ↔ RWKV": "red",
}


def mutual_knn(X, Y, k=10):
    """Mutual k-NN overlap using cosine similarity."""
    X_norm = F.normalize(torch.tensor(X, dtype=torch.float32), dim=1)
    Y_norm = F.normalize(torch.tensor(Y, dtype=torch.float32), dim=1)

    sim_XX = X_norm @ X_norm.T
    sim_YY = Y_norm @ Y_norm.T
    sim_XX.fill_diagonal_(-1)
    sim_YY.fill_diagonal_(-1)

    nn_X = sim_XX.topk(k, dim=1).indices
    nn_Y = sim_YY.topk(k, dim=1).indices

    overlaps = []
    for i in range(len(X)):
        set_x = set(nn_X[i].numpy())
        set_y = set(nn_Y[i].numpy())
        overlaps.append(len(set_x & set_y) / k)
    return np.mean(overlaps)


def linear_cka(X, Y):
    """Linear CKA."""
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
    return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt())).item()


def compute_null(X, Y, metric_fn, n_perms=20, **kwargs):
    """Permutation null distribution."""
    nulls = []
    for _ in range(n_perms):
        perm = np.random.permutation(len(Y))
        nulls.append(metric_fn(X, Y[perm], **kwargs))
    return np.array(nulls)


def compare_pair(reps_a, reps_b, n_perms=20):
    """Compare two models across normalized depth, return per-layer metrics."""
    n_layers_a = reps_a.shape[0]
    n_layers_b = reps_b.shape[0]
    n_points = min(n_layers_a, n_layers_b)
    alphas = np.linspace(0, 1, n_points)

    knn_scores = []
    cka_scores = []

    for alpha in tqdm(alphas, desc="  layers", leave=False):
        la = min(int(alpha * (n_layers_a - 1)), n_layers_a - 1)
        lb = min(int(alpha * (n_layers_b - 1)), n_layers_b - 1)

        X = reps_a[la].numpy()
        Y = reps_b[lb].numpy()

        knn = mutual_knn(X, Y, k=10)
        knn_null = compute_null(X, Y, mutual_knn, n_perms=n_perms, k=10)
        knn_z = float((knn - knn_null.mean()) / (knn_null.std() + 1e-8))

        cka = linear_cka(X, Y)
        cka_null = compute_null(X, Y, linear_cka, n_perms=n_perms)
        cka_z = float((cka - cka_null.mean()) / (cka_null.std() + 1e-8))

        knn_scores.append({
            "alpha": float(alpha), "layer_a": la, "layer_b": lb,
            "raw": float(knn), "null_mean": float(knn_null.mean()),
            "null_std": float(knn_null.std()), "z": knn_z,
        })
        cka_scores.append({
            "alpha": float(alpha), "layer_a": la, "layer_b": lb,
            "raw": float(cka), "null_mean": float(cka_null.mean()),
            "null_std": float(cka_null.std()), "z": cka_z,
        })

    best_knn = max(knn_scores, key=lambda x: x["z"])
    best_cka = max(cka_scores, key=lambda x: x["z"])

    return {
        "knn": knn_scores,
        "cka": cka_scores,
        "best_knn_z": best_knn["z"],
        "best_knn_raw": best_knn["raw"],
        "best_knn_alpha": best_knn["alpha"],
        "best_cka_z": best_cka["z"],
        "best_cka_raw": best_cka["raw"],
        "best_cka_alpha": best_cka["alpha"],
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-architecture scaling comparison")
    parser.add_argument("--n-perms", type=int, default=20)
    parser.add_argument("--scales", nargs="+", default=list(SCALES.keys()),
                        choices=list(SCALES.keys()))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load all reps
    all_reps = {}
    for scale in args.scales:
        all_reps[scale] = {}
        for arch, fname in SCALES[scale].items():
            path = data_dir / fname
            if not path.exists():
                print(f"MISSING: {path}")
                return
            reps = torch.load(path, map_location="cpu", weights_only=True)
            all_reps[scale][arch] = reps
            print(f"Loaded {scale}/{arch}: {reps.shape}")

    # Pairwise comparison per scale
    archs = ["Pythia", "Mamba", "RWKV"]
    pairs = [(archs[i], archs[j]) for i in range(3) for j in range(i + 1, 3)]
    results = {}

    for scale in args.scales:
        results[scale] = {}
        for a, b in pairs:
            pair_key = f"{a} ↔ {b}"
            print(f"\n{'=' * 60}")
            print(f"[{scale}] {pair_key}")
            print(f"{'=' * 60}")
            t0 = time.time()
            res = compare_pair(
                all_reps[scale][a], all_reps[scale][b], n_perms=args.n_perms
            )
            elapsed = time.time() - t0
            print(f"  Best kNN: z={res['best_knn_z']:.1f}  raw={res['best_knn_raw']:.4f}  α={res['best_knn_alpha']:.2f}")
            print(f"  Best CKA: z={res['best_cka_z']:.1f}  raw={res['best_cka_raw']:.4f}  α={res['best_cka_alpha']:.2f}")
            print(f"  Elapsed: {elapsed:.0f}s")
            results[scale][pair_key] = res

    # === Summary table ===
    print(f"\n{'=' * 80}")
    print("SCALING SUMMARY")
    print(f"{'=' * 80}")
    header = f"{'Scale':>8} {'Pair':>20} {'kNN z':>10} {'kNN raw':>10} {'CKA z':>10} {'CKA raw':>10}"
    print(header)
    print("-" * len(header))

    print(f"{'160M':>8} {'(pilot)':>20}")
    for pk, z in PILOT_KNN_Z.items():
        print(f"{'':>8} {pk:>20} {z:>10.1f} {'—':>10} {'—':>10} {'—':>10}")

    for scale in args.scales:
        for pk, res in results[scale].items():
            print(f"{scale:>8} {pk:>20} {res['best_knn_z']:>10.1f} {res['best_knn_raw']:>10.4f} "
                  f"{res['best_cka_z']:>10.1f} {res['best_cka_raw']:>10.4f}")

    # === Figure 1: depth curves per scale ===
    n_scales = len(args.scales)
    fig, axes = plt.subplots(n_scales, 2, figsize=(14, 4 * n_scales), squeeze=False)

    for si, scale in enumerate(args.scales):
        for pk, res in results[scale].items():
            al_k = [s["alpha"] for s in res["knn"]]
            axes[si, 0].plot(al_k, [s["raw"] for s in res["knn"]], 'o-',
                             label=pk, color=PAIR_COLORS[pk], markersize=3)
            al_c = [s["alpha"] for s in res["cka"]]
            axes[si, 1].plot(al_c, [s["raw"] for s in res["cka"]], 'o-',
                             label=pk, color=PAIR_COLORS[pk], markersize=3)

        axes[si, 0].set_title(f"{scale} — Mutual k-NN (k=10)")
        axes[si, 0].set_xlabel("Normalized depth α")
        axes[si, 0].set_ylabel("Mutual k-NN overlap")
        axes[si, 0].legend()

        axes[si, 1].set_title(f"{scale} — Linear CKA")
        axes[si, 1].set_xlabel("Normalized depth α")
        axes[si, 1].set_ylabel("CKA")
        axes[si, 1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "scaling_depth_curves.png", dpi=150)
    print(f"\nFig 1 saved: {FIG_DIR / 'scaling_depth_curves.png'}")

    # === Figure 2: scaling trend ===
    if len(args.scales) >= 2:
        scale_sizes = {"410M": 0.41, "1.4B": 1.4, "2.8B": 2.8}
        pilot_size = 0.16

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

        for pk in PILOT_KNN_Z:
            sizes = [pilot_size]
            knn_zs = [PILOT_KNN_Z[pk]]
            cka_sizes = []
            cka_zs = []

            for scale in args.scales:
                if pk in results[scale]:
                    sizes.append(scale_sizes[scale])
                    knn_zs.append(results[scale][pk]["best_knn_z"])
                    cka_sizes.append(scale_sizes[scale])
                    cka_zs.append(results[scale][pk]["best_cka_z"])

            axes2[0].plot(sizes, knn_zs, 'o-', label=pk, color=PAIR_COLORS[pk])
            if cka_sizes:
                axes2[1].plot(cka_sizes, cka_zs, 'o-', label=pk, color=PAIR_COLORS[pk])

        axes2[0].set_xlabel("Model size (B params)")
        axes2[0].set_ylabel("Best mutual k-NN z-score")
        axes2[0].set_title("Cross-arch similarity vs Scale (kNN)")
        axes2[0].legend()
        axes2[0].set_xscale("log")

        axes2[1].set_xlabel("Model size (B params)")
        axes2[1].set_ylabel("Best CKA z-score")
        axes2[1].set_title("Cross-arch similarity vs Scale (CKA)")
        axes2[1].legend()
        axes2[1].set_xscale("log")

        fig2.tight_layout()
        fig2.savefig(FIG_DIR / "scaling_trend.png", dpi=150)
        print(f"Fig 2 saved: {FIG_DIR / 'scaling_trend.png'}")

    # Save JSON
    output = {
        "n_perms": args.n_perms,
        "scales": args.scales,
        "pilot_knn_z": PILOT_KNN_Z,
        "results": results,
    }
    with open(PHASE_DIR / "results_scaling.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {PHASE_DIR / 'results_scaling.json'}")


if __name__ == "__main__":
    main()
