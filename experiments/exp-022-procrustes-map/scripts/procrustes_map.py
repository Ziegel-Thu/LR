#!/usr/bin/env python3
"""exp-022: Full cross-architecture Procrustes similarity map.

Computes pairwise Procrustes distance and linear CKA between all 16 models
using middle-layer representations from exp-003.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA


# Model registry: name -> (filename, n_layers, d_model)
MODELS = {
    "Pythia-70M":   ("Pythia-70M_reps.pt",   7,  512),
    "Pythia-160M":  ("Pythia-160M_reps.pt",  13,  768),
    "Pythia-410M":  ("Pythia-410M_reps.pt",  25, 1024),
    "Pythia-1B":    ("Pythia-1B_reps.pt",    17, 2048),
    "Pythia-1.4B":  ("Pythia-1.4B_reps.pt",  25, 2048),
    "Pythia-2.8B":  ("Pythia-2.8B_reps.pt",  33, 2560),
    "Pythia-6.9B":  ("Pythia-6.9B_reps.pt",  33, 4096),
    "Mamba-130M":   ("Mamba-130M_reps.pt",   25,  768),
    "Mamba-370M":   ("Mamba-370M_reps.pt",   49, 1024),
    "Mamba-1.4B":   ("Mamba-1.4B_reps.pt",   49, 2048),
    "Mamba-2.8B":   ("Mamba-2.8B_reps.pt",   65, 2560),
    "RWKV-169M":    ("RWKV-169M_reps.pt",   13,  768),
    "RWKV-430M":    ("RWKV-430M_reps.pt",   25, 1024),
    "RWKV-1.5B":    ("RWKV-1.5B_reps.pt",   25, 2048),
    "RWKV-3B":      ("RWKV-4-3B_reps.pt",   33, 2560),
    "RWKV-7B":      ("RWKV-7B_reps.pt",     33, 4096),
}

MODEL_ORDER = list(MODELS.keys())


def load_middle_layer(reps_dir: str, name: str, n_samples: int, seed: int = 42) -> np.ndarray:
    """Load middle layer representations, subsample sentences."""
    fname, n_layers, d_model = MODELS[name]
    path = os.path.join(reps_dir, fname)
    reps = torch.load(path, map_location="cpu", weights_only=True)  # (n_layers, 1728, d)
    mid = n_layers // 2
    X = reps[mid].numpy()  # (1728, d)

    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)
    idx.sort()
    return X[idx].astype(np.float64)


def align_dims(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Project to min(d1, d2, n_samples) via PCA if dimensions differ."""
    if X.shape[1] == Y.shape[1]:
        return X, Y
    # Cap at n_samples - 1 to stay within PCA limits
    target_d = min(X.shape[1], Y.shape[1], X.shape[0] - 1)
    if X.shape[1] > target_d:
        pca = PCA(n_components=target_d, random_state=42)
        X = pca.fit_transform(X)
    if Y.shape[1] > target_d:
        pca = PCA(n_components=target_d, random_state=42)
        Y = pca.fit_transform(Y)
    return X, Y


def procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Orthogonal Procrustes distance after centering and scaling."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    X = X / (np.linalg.norm(X) + 1e-12)
    Y = Y / (np.linalg.norm(Y) + 1e-12)
    R, _ = orthogonal_procrustes(X, Y)
    return float(np.linalg.norm(X @ R - Y))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA similarity."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-8))


def family(name: str) -> str:
    if name.startswith("Pythia"):
        return "Pythia"
    if name.startswith("Mamba"):
        return "Mamba"
    return "RWKV"


def main():
    parser = argparse.ArgumentParser(description="exp-022: Procrustes similarity map")
    parser.add_argument("--reps-dir", required=True, help="Directory with *_reps.pt files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--n-samples", type=int, default=500, help="Sentences to subsample")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    N = len(MODEL_ORDER)

    # Pre-load all representations
    print(f"Loading {N} models from {args.reps_dir} ...")
    reps = {}
    for name in MODEL_ORDER:
        t0 = time.time()
        reps[name] = load_middle_layer(args.reps_dir, name, args.n_samples)
        print(f"  {name}: shape {reps[name].shape}, {time.time()-t0:.1f}s")

    # Compute pairwise matrices
    proc_mat = np.zeros((N, N))
    cka_mat = np.zeros((N, N))
    total_pairs = N * (N - 1) // 2
    done = 0
    t_start = time.time()

    for i in range(N):
        for j in range(i, N):
            Xi, Yj = align_dims(reps[MODEL_ORDER[i]].copy(), reps[MODEL_ORDER[j]].copy())
            proc_mat[i, j] = procrustes_distance(Xi, Yj)
            proc_mat[j, i] = proc_mat[i, j]
            cka_mat[i, j] = linear_cka(Xi, Yj)
            cka_mat[j, i] = cka_mat[i, j]
            if i != j:
                done += 1
                if done % 20 == 0:
                    elapsed = time.time() - t_start
                    print(f"  {done}/{total_pairs} pairs, {elapsed:.1f}s elapsed")

    elapsed = time.time() - t_start
    print(f"All {total_pairs} pairs computed in {elapsed:.1f}s")

    # Save matrices
    np.save(os.path.join(args.output_dir, "procrustes_matrix.npy"), proc_mat)
    np.save(os.path.join(args.output_dir, "cka_matrix.npy"), cka_mat)
    with open(os.path.join(args.output_dir, "model_names.json"), "w") as f:
        json.dump(MODEL_ORDER, f, indent=2)

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("PROCRUSTES DISTANCE MATRIX")
    print("=" * 70)
    header = f"{'':>14s}" + "".join(f"{n:>14s}" for n in MODEL_ORDER)
    print(header)
    for i, ni in enumerate(MODEL_ORDER):
        row = f"{ni:>14s}" + "".join(f"{proc_mat[i,j]:14.4f}" for j in range(N))
        print(row)

    print("\n" + "=" * 70)
    print("CKA SIMILARITY MATRIX")
    print("=" * 70)
    print(header)
    for i, ni in enumerate(MODEL_ORDER):
        row = f"{ni:>14s}" + "".join(f"{cka_mat[i,j]:14.4f}" for j in range(N))
        print(row)

    # Within-family vs cross-family stats
    families = [family(n) for n in MODEL_ORDER]
    within_proc, cross_proc = [], []
    within_cka, cross_cka = [], []
    for i in range(N):
        for j in range(i + 1, N):
            if families[i] == families[j]:
                within_proc.append(proc_mat[i, j])
                within_cka.append(cka_mat[i, j])
            else:
                cross_proc.append(proc_mat[i, j])
                cross_cka.append(cka_mat[i, j])

    print("\n" + "=" * 70)
    print("ARCHITECTURE CLUSTERING ANALYSIS")
    print("=" * 70)
    print(f"Within-family  Procrustes: mean={np.mean(within_proc):.4f}, std={np.std(within_proc):.4f}")
    print(f"Cross-family   Procrustes: mean={np.mean(cross_proc):.4f}, std={np.std(cross_proc):.4f}")
    print(f"Within-family  CKA:        mean={np.mean(within_cka):.4f}, std={np.std(within_cka):.4f}")
    print(f"Cross-family   CKA:        mean={np.mean(cross_cka):.4f}, std={np.std(cross_cka):.4f}")

    # Per-family breakdown
    for fam in ["Pythia", "Mamba", "RWKV"]:
        idxs = [k for k, f in enumerate(families) if f == fam]
        dists = [proc_mat[i, j] for i in idxs for j in idxs if i < j]
        ckas = [cka_mat[i, j] for i in idxs for j in idxs if i < j]
        if dists:
            print(f"\n  {fam} internal ({len(dists)} pairs):")
            print(f"    Procrustes: mean={np.mean(dists):.4f}, min={np.min(dists):.4f}, max={np.max(dists):.4f}")
            print(f"    CKA:        mean={np.mean(ckas):.4f}, min={np.min(ckas):.4f}, max={np.max(ckas):.4f}")

    # Cross-family breakdown
    for f1, f2 in [("Pythia", "Mamba"), ("Pythia", "RWKV"), ("Mamba", "RWKV")]:
        idxs1 = [k for k, f in enumerate(families) if f == f1]
        idxs2 = [k for k, f in enumerate(families) if f == f2]
        dists = [proc_mat[i, j] for i in idxs1 for j in idxs2]
        ckas = [cka_mat[i, j] for i in idxs1 for j in idxs2]
        print(f"\n  {f1} x {f2} ({len(dists)} pairs):")
        print(f"    Procrustes: mean={np.mean(dists):.4f}, min={np.min(dists):.4f}, max={np.max(dists):.4f}")
        print(f"    CKA:        mean={np.mean(ckas):.4f}, min={np.min(ckas):.4f}, max={np.max(ckas):.4f}")

    # Procrustes vs CKA correlation
    all_proc = [proc_mat[i, j] for i in range(N) for j in range(i + 1, N)]
    all_cka = [cka_mat[i, j] for i in range(N) for j in range(i + 1, N)]
    corr = np.corrcoef(all_proc, all_cka)[0, 1]
    print(f"\n{'=' * 70}")
    print(f"Procrustes vs CKA correlation (Pearson r): {corr:.4f}")
    print(f"{'=' * 70}")

    # Scale effect: within each family, compare pairs at similar vs different scales
    print(f"\n{'=' * 70}")
    print("SCALE EFFECT (within-family, sorted by distance)")
    print("=" * 70)
    for fam in ["Pythia", "Mamba", "RWKV"]:
        idxs = [k for k, f in enumerate(families) if f == fam]
        pairs = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                pairs.append((MODEL_ORDER[i], MODEL_ORDER[j], proc_mat[i, j], cka_mat[i, j]))
        pairs.sort(key=lambda x: x[2])
        print(f"\n  {fam}:")
        for m1, m2, pd, ck in pairs:
            print(f"    {m1:>14s} - {m2:<14s}  Procrustes={pd:.4f}  CKA={ck:.4f}")

    # Nearest neighbor for each model
    print(f"\n{'=' * 70}")
    print("NEAREST NEIGHBOR (by Procrustes distance)")
    print("=" * 70)
    for i, ni in enumerate(MODEL_ORDER):
        dists = [(proc_mat[i, j], MODEL_ORDER[j]) for j in range(N) if i != j]
        dists.sort()
        top3 = dists[:3]
        fam_i = families[i]
        nn_str = ", ".join(f"{n} ({d:.4f}, {'same' if family(n)==fam_i else 'CROSS'})" for d, n in top3)
        print(f"  {ni:>14s} -> {nn_str}")

    # Save summary
    summary = {
        "n_models": N,
        "n_samples": args.n_samples,
        "within_family_procrustes_mean": float(np.mean(within_proc)),
        "cross_family_procrustes_mean": float(np.mean(cross_proc)),
        "within_family_cka_mean": float(np.mean(within_cka)),
        "cross_family_cka_mean": float(np.mean(cross_cka)),
        "procrustes_cka_correlation": float(corr),
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(args.output_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
