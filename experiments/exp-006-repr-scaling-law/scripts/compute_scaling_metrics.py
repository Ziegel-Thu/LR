#!/usr/bin/env python3
"""
Exp-006 Phase 1: Compute representation quality metrics across Pythia scaling ladder.

For each Pythia model (70M → 6.9B), computes per-layer:
  - Intra-model: TwoNN ID, MLE ID, stable rank, effective rank
  - Cross-model (consecutive scale pairs): CKA, shape metric, mutual kNN

Reuses reps extracted by exp-003/phase1/extract_reps.py.

Usage:
  python compute_scaling_metrics.py --data-dir /nvmessd/lifanhong/LR-env/exp003_reps
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent

PYTHIA_SCALES = [
    ("Pythia-70M",  70e6),
    ("Pythia-160M", 160e6),
    ("Pythia-410M", 410e6),
    ("Pythia-1B",   1e9),
    ("Pythia-1.4B", 1.4e9),
    ("Pythia-2.8B", 2.8e9),
    ("Pythia-6.9B", 6.9e9),
]


# ── Intra-model metrics ─────────────────────────────────────────────────────

def stable_rank(X: np.ndarray) -> float:
    """Stable rank: ||A||_F² / ||A||_2² = Σσᵢ² / σ₁²."""
    sv = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    sv2 = sv ** 2
    return float(sv2.sum() / (sv2[0] + 1e-10))


def effective_rank(X: np.ndarray) -> float:
    """Effective rank: exp(H(p)) where p = σᵢ² / Σσᵢ²."""
    sv = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    sv2 = sv ** 2
    p = sv2 / (sv2.sum() + 1e-10)
    p = p[p > 1e-10]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def twonn_id(X: np.ndarray) -> float:
    """TwoNN intrinsic dimension estimator (Facco et al. 2017)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X)
    distances, _ = nn.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]

    mask = r1 > 1e-10
    mu = r2[mask] / r1[mask]
    n = len(mu)
    log_mu_sum = np.sum(np.log(mu))
    if log_mu_sum < 1e-10:
        return 0.0
    return float(n / log_mu_sum)


def mle_id(X: np.ndarray, k: int = 100) -> float:
    """MLE intrinsic dimension (Levina & Bickel 2004)."""
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
        log_ratios = np.log(r_k / dists[:-1])
        d_hat = (len(dists) - 1) / np.sum(log_ratios)
        ids.append(d_hat)

    return float(np.mean(ids)) if ids else 0.0


# ── Cross-model metrics ─────────────────────────────────────────────────────

def mutual_knn(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Mutual k-NN overlap using cosine similarity."""
    X_t = F.normalize(torch.tensor(X, dtype=torch.float32), dim=1)
    Y_t = F.normalize(torch.tensor(Y, dtype=torch.float32), dim=1)

    sim_XX = X_t @ X_t.T
    sim_YY = Y_t @ Y_t.T
    sim_XX.fill_diagonal_(-1)
    sim_YY.fill_diagonal_(-1)

    nn_X = sim_XX.topk(k, dim=1).indices.numpy()
    nn_Y = sim_YY.topk(k, dim=1).indices.numpy()

    overlaps = [len(set(nn_X[i]) & set(nn_Y[i])) / k for i in range(len(X))]
    return float(np.mean(overlaps))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    hsic_xx = np.linalg.norm(Xc.T @ Xc, "fro") ** 2
    hsic_yy = np.linalg.norm(Yc.T @ Yc, "fro") ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def shape_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Williams shape metric (Procrustes distance)."""
    from netrep.metrics import LinearMetric
    metric = LinearMetric(alpha=1.0, center_columns=True)
    metric.fit(X, Y)
    return float(metric.score(X, Y))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with *_reps.pt files")
    parser.add_argument("--id-k", type=int, default=100, help="K for MLE ID")
    parser.add_argument("--knn-k", type=int, default=10, help="k for mutual kNN")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load reps
    reps = {}
    available = []
    for name, params in PYTHIA_SCALES:
        path = data_dir / f"{name}_reps.pt"
        if path.exists():
            r = torch.load(path, map_location="cpu", weights_only=True)
            reps[name] = r.numpy()
            available.append((name, params))
            print(f"Loaded {name}: {r.shape}")
        else:
            print(f"SKIP {name}: {path} not found")

    if len(available) < 2:
        print("Need at least 2 models. Exiting.")
        return

    # ── Compute intra-model metrics ──
    print(f"\n{'='*70}")
    print("INTRA-MODEL METRICS (per layer)")
    print(f"{'='*70}")

    intra_results = {}
    for name, params in available:
        R = reps[name]  # (n_layers, n_stimuli, d_model)
        n_layers = R.shape[0]
        print(f"\n--- {name} ({n_layers} layers, d={R.shape[2]}) ---")

        layers_data = []
        for li in tqdm(range(n_layers), desc=f"  {name}"):
            X = R[li]
            sr = stable_rank(X)
            er = effective_rank(X)
            id_2nn = twonn_id(X)
            id_mle = mle_id(X, k=args.id_k)

            layers_data.append({
                "layer": li,
                "stable_rank": sr,
                "effective_rank": er,
                "twonn_id": id_2nn,
                "mle_id": id_mle,
            })

            if li % max(1, n_layers // 5) == 0 or li == n_layers - 1:
                print(f"    L{li:2d}: srank={sr:.1f}, erank={er:.1f}, "
                      f"2nn_id={id_2nn:.1f}, mle_id={id_mle:.1f}")

        intra_results[name] = {
            "params": params,
            "n_layers": n_layers,
            "d_model": int(R.shape[2]),
            "layers": layers_data,
        }

    # ── Compute cross-model metrics (consecutive scale pairs) ──
    print(f"\n{'='*70}")
    print("CROSS-MODEL METRICS (consecutive scale pairs)")
    print(f"{'='*70}")

    cross_results = {}
    for idx in range(len(available) - 1):
        name_a, _ = available[idx]
        name_b, _ = available[idx + 1]
        pair_key = f"{name_a} ↔ {name_b}"

        R_a = reps[name_a]
        R_b = reps[name_b]
        n_la, n_lb = R_a.shape[0], R_b.shape[0]
        n_compare = min(n_la, n_lb)

        print(f"\n--- {pair_key} ({n_la} vs {n_lb} layers) ---")

        layers_data = []
        t0 = time.time()
        for ci in tqdm(range(n_compare), desc=f"  {pair_key}"):
            la = int(ci * (n_la - 1) / max(n_compare - 1, 1))
            lb = int(ci * (n_lb - 1) / max(n_compare - 1, 1))
            alpha = ci / max(n_compare - 1, 1)

            X = R_a[la]
            Y = R_b[lb]

            cka = linear_cka(X, Y)
            knn = mutual_knn(X, Y, k=args.knn_k)
            shape = shape_distance(X, Y)

            layers_data.append({
                "alpha": float(alpha),
                "layer_a": la,
                "layer_b": lb,
                "cka": cka,
                "knn": knn,
                "shape_distance": shape,
            })

            if ci % max(1, n_compare // 5) == 0 or ci == n_compare - 1:
                print(f"    α={alpha:.2f} (L{la}/{lb}): CKA={cka:.4f}, "
                      f"kNN={knn:.4f}, shape={shape:.4f}")

        elapsed = time.time() - t0

        best_cka = max(layers_data, key=lambda x: x["cka"])
        best_knn = max(layers_data, key=lambda x: x["knn"])
        best_shape = min(layers_data, key=lambda x: x["shape_distance"])

        cross_results[pair_key] = {
            "model_a": name_a,
            "model_b": name_b,
            "n_compare": n_compare,
            "elapsed_s": elapsed,
            "best_cka": best_cka["cka"],
            "best_cka_alpha": best_cka["alpha"],
            "best_knn": best_knn["knn"],
            "best_knn_alpha": best_knn["alpha"],
            "best_shape": best_shape["shape_distance"],
            "best_shape_alpha": best_shape["alpha"],
            "layers": layers_data,
        }

        print(f"  Best CKA={best_cka['cka']:.4f} @ α={best_cka['alpha']:.2f}")
        print(f"  Best kNN={best_knn['knn']:.4f} @ α={best_knn['alpha']:.2f}")
        print(f"  Best shape={best_shape['shape_distance']:.4f} @ α={best_shape['alpha']:.2f}")
        print(f"  Elapsed: {elapsed:.0f}s")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY: Intra-model metrics (middle layer)")
    print(f"{'='*70}")
    print(f"{'Model':>15} {'Params':>10} {'sRank':>8} {'eRank':>8} {'2NN_ID':>8} {'MLE_ID':>8}")
    print("-" * 60)
    for name, params in available:
        mid = len(intra_results[name]["layers"]) // 2
        d = intra_results[name]["layers"][mid]
        print(f"{name:>15} {params/1e6:>8.0f}M {d['stable_rank']:>8.1f} "
              f"{d['effective_rank']:>8.1f} {d['twonn_id']:>8.1f} {d['mle_id']:>8.1f}")

    print(f"\n{'='*70}")
    print("SUMMARY: Cross-model metrics (best layer)")
    print(f"{'='*70}")
    print(f"{'Pair':>30} {'CKA':>8} {'kNN':>8} {'Shape':>8}")
    print("-" * 58)
    for pk, cr in cross_results.items():
        print(f"{pk:>30} {cr['best_cka']:>8.4f} {cr['best_knn']:>8.4f} {cr['best_shape']:>8.4f}")

    # Save
    output = {
        "config": {"id_k": args.id_k, "knn_k": args.knn_k},
        "intra_model": intra_results,
        "cross_model": cross_results,
    }

    out_path = EXP_DIR / "results_phase1.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
