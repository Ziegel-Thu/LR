#!/usr/bin/env python3
"""
exp-019: U-shape depth curve mechanism analysis.

For each model's per-layer representations, compute:
  1. Intrinsic dimensionality (TwoNN)
  2. Effective rank (stable rank)
  3. Consecutive-layer CKA
  4. Input similarity: CKA(layer_i, layer_0)
  5. Output similarity: CKA(layer_i, layer_last)
  6. Isotropy (mean cosine similarity of random pairs)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


# ── metrics ──────────────────────────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n, d) matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-8))


def twonn_id(X: np.ndarray, k: int = 2) -> float:
    """TwoNN intrinsic-dimensionality estimator."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    dists, _ = nn.kneighbors(X)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    # filter zero distances
    mask = r1 > 1e-12
    if mask.sum() < 10:
        return float("nan")
    mu = r2[mask] / r1[mask]
    mu = mu[mu > 1.0]  # TwoNN requires mu > 1
    if len(mu) < 10:
        return float("nan")
    n = len(mu)
    # MLE estimator: d = n / sum(log(mu))
    return float(n / np.sum(np.log(mu)))


def stable_rank(X: np.ndarray) -> float:
    """Stable rank = ||X||_F^2 / ||X||_2^2."""
    s = np.linalg.svd(X, compute_uv=False)
    return float((s ** 2).sum() / (s[0] ** 2 + 1e-10))


def isotropy(X: np.ndarray, n_pairs: int = 5000, seed: int = 42) -> float:
    """Mean cosine similarity between random representation pairs."""
    rng = np.random.RandomState(seed)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / (norms + 1e-10)
    idx = rng.randint(0, len(X), (n_pairs, 2))
    # avoid self-pairs
    mask = idx[:, 0] != idx[:, 1]
    idx = idx[mask]
    cos = (X_norm[idx[:, 0]] * X_norm[idx[:, 1]]).sum(axis=1)
    return float(np.mean(cos))


# ── main ─────────────────────────────────────────────────────────────────────

def analyze_model(reps: np.ndarray, model_name: str) -> dict:
    """Analyze a single model's representations.

    Args:
        reps: (n_layers, n_samples, d_model)
        model_name: identifier for logging
    Returns:
        dict with per-layer metrics
    """
    n_layers = reps.shape[0]
    print(f"\n{'='*60}")
    print(f"  {model_name}: {n_layers} layers, {reps.shape[1]} samples, d={reps.shape[2]}")
    print(f"{'='*60}")

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "n_samples": reps.shape[1],
        "d_model": reps.shape[2],
        "intrinsic_dim": [],
        "stable_rank": [],
        "consecutive_cka": [],
        "input_similarity": [],
        "output_similarity": [],
        "isotropy": [],
    }

    layer_0 = reps[0]
    layer_last = reps[-1]

    for i in range(n_layers):
        t0 = time.time()
        X = reps[i]

        id_val = twonn_id(X)
        sr_val = stable_rank(X)
        iso_val = isotropy(X)
        in_sim = linear_cka(X, layer_0)
        out_sim = linear_cka(X, layer_last)

        if i < n_layers - 1:
            cons_cka = linear_cka(X, reps[i + 1])
        else:
            cons_cka = float("nan")

        results["intrinsic_dim"].append(id_val)
        results["stable_rank"].append(sr_val)
        results["consecutive_cka"].append(cons_cka)
        results["input_similarity"].append(in_sim)
        results["output_similarity"].append(out_sim)
        results["isotropy"].append(iso_val)

        dt = time.time() - t0
        print(
            f"  Layer {i:3d}/{n_layers-1}: "
            f"ID={id_val:7.1f}  SR={sr_val:7.1f}  "
            f"iso={iso_val:+.4f}  "
            f"in_sim={in_sim:.4f}  out_sim={out_sim:.4f}  "
            f"cons_cka={cons_cka:.4f}  "
            f"({dt:.1f}s)"
        )

    return results


def print_summary(all_results: list[dict]):
    """Print cross-model summary to identify U-shape patterns."""
    print("\n" + "=" * 80)
    print("  SUMMARY: U-shape pattern analysis")
    print("=" * 80)

    for r in all_results:
        model = r["model"]
        n = r["n_layers"]
        if n < 3:
            continue

        # normalize layer indices to [0, 1]
        third = max(1, n // 3)

        def mean_seg(vals, start, end):
            seg = [v for v in vals[start:end] if not (isinstance(v, float) and np.isnan(v))]
            return np.mean(seg) if seg else float("nan")

        early = slice(0, third)
        mid = slice(third, 2 * third)
        late = slice(2 * third, n)

        print(f"\n  {model} ({n} layers):")
        for metric in ["intrinsic_dim", "stable_rank", "isotropy", "input_similarity", "output_similarity"]:
            vals = r[metric]
            e = mean_seg(vals, 0, third)
            m = mean_seg(vals, third, 2 * third)
            la = mean_seg(vals, 2 * third, n)
            # U-shape: early high, mid low, late high (or inverse)
            if not np.isnan(e) and not np.isnan(m) and not np.isnan(la):
                is_u = (e > m and la > m)
                is_inv_u = (e < m and la < m)
                shape = "U-shape" if is_u else ("∩-shape" if is_inv_u else "other")
            else:
                shape = "N/A"
            print(
                f"    {metric:20s}: early={e:8.3f}  mid={m:8.3f}  late={la:8.3f}  → {shape}"
            )

    # Hypothesis evaluation
    print("\n" + "=" * 80)
    print("  HYPOTHESIS EVALUATION")
    print("=" * 80)

    h1_support = 0  # bottleneck: ID/rank U-shape
    h2_support = 0  # specialization: isotropy ∩-shape
    h3_support = 0  # residual: input_sim + output_sim U-shape

    for r in all_results:
        n = r["n_layers"]
        if n < 3:
            continue
        third = max(1, n // 3)

        def check_u(vals):
            e = np.mean([v for v in vals[:third] if not np.isnan(v)] or [0])
            m = np.mean([v for v in vals[third:2*third] if not np.isnan(v)] or [0])
            la = np.mean([v for v in vals[2*third:] if not np.isnan(v)] or [0])
            return e > m and la > m

        def check_inv_u(vals):
            e = np.mean([v for v in vals[:third] if not np.isnan(v)] or [0])
            m = np.mean([v for v in vals[third:2*third] if not np.isnan(v)] or [0])
            la = np.mean([v for v in vals[2*third:] if not np.isnan(v)] or [0])
            return e < m and la < m

        if check_u(r["intrinsic_dim"]) or check_u(r["stable_rank"]):
            h1_support += 1
        if check_inv_u(r["isotropy"]):
            h2_support += 1
        if check_u(r["input_similarity"]) or check_u(r["output_similarity"]):
            h3_support += 1

    total = sum(1 for r in all_results if r["n_layers"] >= 3)
    print(f"\n  H1 (bottleneck compression): {h1_support}/{total} models show ID/rank U-shape")
    print(f"  H2 (task specialization):    {h2_support}/{total} models show isotropy ∩-shape")
    print(f"  H3 (residual dominance):     {h3_support}/{total} models show input/output sim pattern")


def main():
    parser = argparse.ArgumentParser(description="exp-019: U-shape mechanism analysis")
    parser.add_argument("--reps-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    reps_dir = Path(args.reps_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rep_files = sorted(reps_dir.glob("*_reps.pt"))
    if not rep_files:
        print(f"No *_reps.pt files found in {reps_dir}")
        return

    print(f"Found {len(rep_files)} representation files")
    print(f"Using {args.n_samples} samples per model")

    all_results = []
    np.random.seed(42)
    sample_idx = None

    for fpath in rep_files:
        model_name = fpath.stem.replace("_reps", "")
        print(f"\nLoading {fpath.name} ...")
        reps = torch.load(fpath, map_location="cpu", weights_only=True).numpy()

        # subsample sentences
        n_total = reps.shape[1]
        n_use = min(args.n_samples, n_total)
        if sample_idx is None or len(sample_idx) != n_use:
            sample_idx = np.random.choice(n_total, n_use, replace=False)
            sample_idx.sort()
        reps = reps[:, sample_idx, :]

        result = analyze_model(reps, model_name)
        all_results.append(result)

        # save per-model
        out_path = output_dir / f"{model_name}_ushape.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    # save combined
    combined_path = output_dir / "all_ushape_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results to {combined_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
