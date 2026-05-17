"""
Exp-003 Pilot 追加分析：Williams Shape Metric

在已有的 CKA + mutual kNN 基础上，加 Williams 2021 的 shape metric。
这是唯一满足三角不等式的表征相似度量，能构成真正的度量空间。

比较三种度量对 Pythia/Mamba/RWKV 三架构的排序是否一致。
"""

import torch
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXP_DIR = Path(__file__).resolve().parent.parent


def load_reps():
    """Load pilot representations: (n_layers, n_stimuli, d_model)"""
    models = {}
    for f in DATA_DIR.glob("*_reps.pt"):
        name = f.stem.replace("_reps", "")
        reps = torch.load(f, map_location="cpu", weights_only=True)
        models[name] = reps.numpy()  # (n_layers, n_stimuli, d_model)
        print(f"  {name}: {reps.shape}")
    return models


def compute_shape_distance(X, Y):
    """
    Williams shape metric (Generalized Shape Metrics, NeurIPS 2021).
    Uses Procrustes distance after centering and normalizing.
    """
    from netrep.metrics import LinearMetric
    metric = LinearMetric(alpha=1.0, center_columns=True)
    metric.fit(X, Y)
    return metric.score(X, Y)


def compute_cka(X, Y):
    """Linear CKA."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def compute_mutual_knn(X, Y, k=10):
    """Mutual k-NN overlap."""
    from scipy.spatial import KDTree
    tree_x = KDTree(X)
    tree_y = KDTree(Y)
    _, idx_x = tree_x.query(X, k=k+1)
    _, idx_y = tree_y.query(Y, k=k+1)
    idx_x = idx_x[:, 1:]  # exclude self
    idx_y = idx_y[:, 1:]

    overlaps = []
    for i in range(len(X)):
        overlap = len(set(idx_x[i]) & set(idx_y[i])) / k
        overlaps.append(overlap)
    return np.mean(overlaps)


def main():
    print("=" * 60)
    print("Exp-003 Pilot: Williams Shape Metric Analysis")
    print("=" * 60)

    print("\nLoading representations...")
    models = load_reps()
    model_names = sorted(models.keys())
    print(f"Models: {model_names}")

    # Determine min layers for comparison
    n_layers_list = {name: reps.shape[0] for name, reps in models.items()}
    print(f"Layers: {n_layers_list}")

    pairs = []
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            pairs.append((model_names[i], model_names[j]))

    results = {}

    for name_a, name_b in pairs:
        reps_a = models[name_a]
        reps_b = models[name_b]
        n_layers_a = reps_a.shape[0]
        n_layers_b = reps_b.shape[0]

        print(f"\n--- {name_a} ({n_layers_a} layers) vs {name_b} ({n_layers_b} layers) ---")

        pair_results = []

        # Compare at matched relative depths
        n_compare = min(n_layers_a, n_layers_b)
        for li in range(n_compare):
            # Map to relative depth
            la = int(li * (n_layers_a - 1) / (n_compare - 1)) if n_compare > 1 else 0
            lb = int(li * (n_layers_b - 1) / (n_compare - 1)) if n_compare > 1 else 0

            X = reps_a[la]  # (n_stimuli, d_model_a)
            Y = reps_b[lb]  # (n_stimuli, d_model_b)

            cka = compute_cka(X, Y)
            knn = compute_mutual_knn(X, Y, k=10)
            shape = compute_shape_distance(X, Y)

            pair_results.append({
                'layer_a': int(la),
                'layer_b': int(lb),
                'relative_depth': float(li / max(n_compare - 1, 1)),
                'cka': float(cka),
                'mutual_knn': float(knn),
                'shape_distance': float(shape),
            })

            if li % 3 == 0 or li == n_compare - 1:
                print(f"  L{la}/{lb} (d={li/(n_compare-1):.2f}): "
                      f"CKA={cka:.4f}, kNN={knn:.4f}, shape={shape:.4f}")

        results[f"{name_a}_vs_{name_b}"] = pair_results

    # Summary: compare rankings
    print("\n" + "=" * 60)
    print("SUMMARY: Mean values across layers")
    print("=" * 60)
    print(f"{'Pair':>30} | {'CKA':>8} | {'kNN':>8} | {'Shape':>8}")
    print("-" * 62)
    for pair_name, pr in results.items():
        mean_cka = np.mean([r['cka'] for r in pr])
        mean_knn = np.mean([r['mutual_knn'] for r in pr])
        mean_shape = np.mean([r['shape_distance'] for r in pr])
        print(f"{pair_name:>30} | {mean_cka:8.4f} | {mean_knn:8.4f} | {mean_shape:8.4f}")

    # Check ranking consistency
    print("\n--- Ranking Consistency ---")
    pair_names = list(results.keys())
    for metric_name, key in [("CKA", "cka"), ("kNN", "mutual_knn"), ("Shape", "shape_distance")]:
        means = {p: np.mean([r[key] for r in results[p]]) for p in pair_names}
        if key == "shape_distance":
            ranked = sorted(means, key=means.get)  # lower = more similar
        else:
            ranked = sorted(means, key=means.get, reverse=True)  # higher = more similar
        print(f"  {metric_name} ranking (most similar first): {' > '.join(ranked)}")

    # Save
    out_path = EXP_DIR / "results_shape_metric.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
