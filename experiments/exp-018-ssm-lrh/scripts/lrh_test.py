#!/usr/bin/env python3
"""Exp-018: Linear Representation Hypothesis on SSM architectures.

Tests three aspects of LRH on Mamba-2.8B, RWKV-3B, and Pythia-2.8B:
1. Cross-layer direction consistency
2. Concept orthogonality
3. Direction stability under data perturbation

Uses cached hidden states from exp-007.
"""

import torch
import numpy as np
import json
import argparse
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from itertools import combinations

# ── Feature labelers ──────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "only", "own", "same", "than", "too", "very", "just", "because",
    "if", "when", "while", "where", "how", "what", "which", "who",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "my", "his",
    "our", "your", "their",
}


def label_features(token_strs, token_counts=None):
    """Generate binary labels for all features. Returns dict of arrays."""
    n = len(token_strs)
    labels = {}

    # Precompute stripped tokens
    stripped = [t.strip() for t in token_strs]
    lower = [s.lower() for s in stripped]

    # Frequency ranks (if token_counts available)
    if token_counts is not None:
        sorted_by_freq = sorted(token_counts.items(), key=lambda x: -x[1])
        rank_map = {tid: i for i, (tid, _) in enumerate(sorted_by_freq)}
    else:
        rank_map = None

    # is_capitalized: first non-space char is uppercase
    labels["is_capitalized"] = np.array([
        1 if s and s[0].isupper() else 0 for s in stripped
    ])

    # is_plural: ends with s/es/ies, length > 2
    labels["is_plural"] = np.array([
        1 if len(s) > 2 and s.lower().endswith(("s", "es", "ies"))
        and not s.lower().endswith(("ss", "us", "is"))
        else 0 for s in stripped
    ])

    # is_numeric: contains a digit
    labels["is_numeric"] = np.array([
        1 if any(c.isdigit() for c in s) else 0 for s in stripped
    ])

    # is_punctuation: all non-alphanumeric (and non-empty)
    labels["is_punctuation"] = np.array([
        1 if s and all(not c.isalnum() for c in s) else 0 for s in stripped
    ])

    # is_short: ≤2 chars after strip
    labels["is_short"] = np.array([
        1 if 0 < len(s) <= 2 else 0 for s in stripped
    ])

    # is_stopword
    labels["is_stopword"] = np.array([
        1 if lo in STOPWORDS else 0 for lo in lower
    ])

    # is_title_case: first char upper, rest lower
    labels["is_title_case"] = np.array([
        1 if len(s) > 1 and s[0].isupper() and s[1:].islower()
        else 0 for s in stripped
    ])

    # starts_with_space: original token starts with space
    labels["starts_with_space"] = np.array([
        1 if t and t[0] == " " else 0 for t in token_strs
    ])

    # is_subword: does NOT start with space (for BPE tokenizers)
    labels["is_subword"] = np.array([
        1 if t and t[0] != " " and t[0] != "Ġ" else 0 for t in token_strs
    ])

    # is_high_freq and is_rare (need token_counts)
    if token_counts is not None:
        # Use actual token occurrence in the corpus
        total = sum(token_counts.values())
        freq_threshold_high = total * 0.001  # top 0.1%
        freq_threshold_rare = total * 0.00001  # bottom
        # Simpler: use rank
        labels["is_high_freq"] = np.zeros(n, dtype=np.int64)
        labels["is_rare"] = np.zeros(n, dtype=np.int64)
        # We don't have per-position token_ids easily, so use string frequency
        from collections import Counter
        str_counts = Counter(token_strs)
        total_tokens = len(token_strs)
        sorted_strs = sorted(str_counts.items(), key=lambda x: -x[1])
        str_rank = {s: i for i, (s, _) in enumerate(sorted_strs)}
        vocab_size = len(str_rank)
        for i, t in enumerate(token_strs):
            r = str_rank.get(t, vocab_size)
            if r < 1000:
                labels["is_high_freq"][i] = 1
            if r > max(vocab_size - 1000, vocab_size * 0.8):
                labels["is_rare"][i] = 1
    else:
        # Fallback: use string frequency in corpus
        from collections import Counter
        str_counts = Counter(token_strs)
        sorted_strs = sorted(str_counts.items(), key=lambda x: -x[1])
        str_rank = {s: i for i, (s, _) in enumerate(sorted_strs)}
        vocab_size = len(str_rank)
        labels["is_high_freq"] = np.array([
            1 if str_rank.get(t, vocab_size) < 1000 else 0 for t in token_strs
        ])
        labels["is_rare"] = np.array([
            1 if str_rank.get(t, vocab_size) > max(vocab_size - 1000, int(vocab_size * 0.8))
            else 0 for t in token_strs
        ])

    return labels


# ── Probe training ────────────────────────────────────────────────

def get_probe_direction(X, y):
    """Train logistic regression probe, return normalized weight vector and accuracy."""
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X, y)
    w = clf.coef_[0]
    acc = clf.score(X, y)
    w_norm = w / (np.linalg.norm(w) + 1e-10)
    return w_norm, acc


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp-018: LRH tests on SSM architectures")
    parser.add_argument("--model-name", required=True, help="Model name for labeling output")
    parser.add_argument("--cache-dir", required=True, help="Path to exp-007 cache directory")
    parser.add_argument("--output-dir", required=True, help="Where to save results")
    parser.add_argument("--n-layers-sample", type=int, default=8,
                        help="Number of evenly-spaced layers to sample")
    parser.add_argument("--max-tokens", type=int, default=50000,
                        help="Max tokens to use (subsample for speed)")
    args = parser.parse_args()

    t0 = time.time()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exp-018 LRH Test: {args.model_name} ===")
    print(f"Cache: {cache_dir}")
    print(f"Output: {out_dir}")

    # Load meta
    meta = json.load(open(cache_dir / "meta.json"))
    n_layers = meta["n_layers"]
    d_model = meta["d_model"]
    print(f"Model: {meta.get('model', args.model_name)}, layers={n_layers}, d={d_model}")

    # Select layers
    if n_layers <= args.n_layers_sample:
        layers = list(range(n_layers))
    else:
        layers = [int(i * (n_layers - 1) / (args.n_layers_sample - 1))
                  for i in range(args.n_layers_sample)]
    print(f"Sampled layers ({len(layers)}): {layers}")

    # Load token info
    token_info = torch.load(cache_dir / "token_info.pt", weights_only=False)
    if isinstance(token_info, dict):
        token_strs = token_info["token_strs"]
        token_counts = token_info.get("token_counts", None)
    else:
        token_strs = token_info
        token_counts = None

    n_tokens_total = len(token_strs)
    print(f"Total tokens: {n_tokens_total}")

    # Subsample for speed
    if n_tokens_total > args.max_tokens:
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(n_tokens_total, args.max_tokens, replace=False))
    else:
        idx = np.arange(n_tokens_total)
    n_tokens = len(idx)
    print(f"Using {n_tokens} tokens")

    # Subsample token_strs
    if isinstance(token_strs, list):
        sub_token_strs = [token_strs[i] for i in idx]
    else:
        sub_token_strs = [token_strs[int(i)] for i in idx]

    # Generate labels
    all_labels = label_features(sub_token_strs, token_counts)

    # Filter features with sufficient balance (at least 5% minority class)
    min_frac = 0.05
    features_to_use = []
    for feat, y in all_labels.items():
        pos_frac = y.mean()
        if min_frac <= pos_frac <= (1 - min_frac):
            features_to_use.append(feat)
            print(f"  Feature '{feat}': {pos_frac:.1%} positive ({int(y.sum())}/{n_tokens})")
        else:
            print(f"  Feature '{feat}': SKIPPED ({pos_frac:.1%} positive, too imbalanced)")

    print(f"\nUsing {len(features_to_use)} features: {features_to_use}")

    # ── Train probes at each layer ────────────────────────────────
    directions = {}   # {feat: {layer: w_norm}}
    accuracies = {}   # {feat: {layer: acc}}

    for li, layer in enumerate(layers):
        print(f"\n--- Layer {layer} ({li+1}/{len(layers)}) ---")
        h_path = cache_dir / "hiddens" / f"layer_{layer}.pt"
        X_full = torch.load(h_path, map_location="cpu")

        # Subsample
        X = X_full[idx].numpy().astype(np.float32)
        del X_full  # free memory

        for feat in features_to_use:
            y = all_labels[feat]
            w, acc = get_probe_direction(X, y)
            if feat not in directions:
                directions[feat] = {}
                accuracies[feat] = {}
            directions[feat][layer] = w
            accuracies[feat][layer] = acc
            print(f"  {feat}: acc={acc:.4f}")

        del X

    # ── Test 1: Cross-layer direction consistency ─────────────────
    print("\n\n=== TEST 1: Cross-Layer Direction Consistency ===")
    cross_layer = {}
    for feat in features_to_use:
        layer_keys = sorted(directions[feat].keys())
        sims = []
        for l1, l2 in combinations(layer_keys, 2):
            sim = cosine_sim(directions[feat][l1], directions[feat][l2])
            sims.append(sim)
        mean_sim = float(np.mean(sims)) if sims else 0.0
        # Also compute adjacent-layer similarity
        adj_sims = []
        for i in range(len(layer_keys) - 1):
            sim = cosine_sim(directions[feat][layer_keys[i]],
                             directions[feat][layer_keys[i + 1]])
            adj_sims.append(sim)
        adj_mean = float(np.mean(adj_sims)) if adj_sims else 0.0

        cross_layer[feat] = {
            "mean_pairwise_cos": mean_sim,
            "mean_adjacent_cos": adj_mean,
            "all_pairwise": sims,
            "all_adjacent": adj_sims,
        }
        print(f"  {feat}: pairwise={mean_sim:.4f}, adjacent={adj_mean:.4f}")

    all_pairwise = [v["mean_pairwise_cos"] for v in cross_layer.values()]
    all_adjacent = [v["mean_adjacent_cos"] for v in cross_layer.values()]
    print(f"\n  OVERALL: pairwise={np.mean(all_pairwise):.4f} ± {np.std(all_pairwise):.4f}")
    print(f"  OVERALL: adjacent={np.mean(all_adjacent):.4f} ± {np.std(all_adjacent):.4f}")

    # ── Test 2: Concept orthogonality ─────────────────────────────
    print("\n\n=== TEST 2: Concept Orthogonality ===")
    orthogonality = {}
    for layer in sorted(layers):
        pair_sims = []
        pair_names = []
        for f1, f2 in combinations(features_to_use, 2):
            if layer in directions[f1] and layer in directions[f2]:
                sim = abs(cosine_sim(directions[f1][layer], directions[f2][layer]))
                pair_sims.append(sim)
                pair_names.append(f"{f1}_vs_{f2}")
        if pair_sims:
            orthogonality[layer] = {
                "mean_abs_cos": float(np.mean(pair_sims)),
                "max_abs_cos": float(np.max(pair_sims)),
                "min_abs_cos": float(np.min(pair_sims)),
                "n_near_orthogonal": int(sum(1 for s in pair_sims if s < 0.1)),
                "n_pairs": len(pair_sims),
            }
            print(f"  Layer {layer}: mean|cos|={np.mean(pair_sims):.4f}, "
                  f"max={np.max(pair_sims):.4f}, "
                  f"near-orthogonal(<0.1)={orthogonality[layer]['n_near_orthogonal']}/{len(pair_sims)}")

    # Overall orthogonality
    all_means = [v["mean_abs_cos"] for v in orthogonality.values()]
    if all_means:
        print(f"\n  OVERALL: mean|cos|={np.mean(all_means):.4f} ± {np.std(all_means):.4f}")

    # ── Test 3: Direction stability under perturbation ────────────
    print("\n\n=== TEST 3: Direction Stability (50/50 split) ===")
    rng = np.random.RandomState(123)
    perm = rng.permutation(n_tokens)
    half = n_tokens // 2
    idx_a = np.sort(perm[:half])
    idx_b = np.sort(perm[half:])

    # Pick middle layer for stability test
    mid_layer = layers[len(layers) // 2]
    print(f"  Testing at layer {mid_layer}")

    h_path = cache_dir / "hiddens" / f"layer_{mid_layer}.pt"
    X_full = torch.load(h_path, map_location="cpu")
    X_all = X_full[idx].numpy().astype(np.float32)
    del X_full

    stability = {}
    for feat in features_to_use:
        y = all_labels[feat]
        # Split A
        Xa, ya = X_all[idx_a], y[idx_a]
        # Split B
        Xb, yb = X_all[idx_b], y[idx_b]

        # Check balance in each split
        if ya.mean() < min_frac or ya.mean() > (1 - min_frac):
            print(f"  {feat}: SKIPPED (imbalanced in split A)")
            continue
        if yb.mean() < min_frac or yb.mean() > (1 - min_frac):
            print(f"  {feat}: SKIPPED (imbalanced in split B)")
            continue

        wa, acc_a = get_probe_direction(Xa, ya)
        wb, acc_b = get_probe_direction(Xb, yb)
        sim = cosine_sim(wa, wb)
        stability[feat] = {
            "cos_similarity": sim,
            "acc_split_a": acc_a,
            "acc_split_b": acc_b,
        }
        print(f"  {feat}: cos={sim:.4f} (acc_a={acc_a:.4f}, acc_b={acc_b:.4f})")

    del X_all

    if stability:
        all_stab = [v["cos_similarity"] for v in stability.values()]
        print(f"\n  OVERALL: stability={np.mean(all_stab):.4f} ± {np.std(all_stab):.4f}")

    # ── Save results ──────────────────────────────────────────────
    results = {
        "model_name": args.model_name,
        "meta": meta,
        "layers_sampled": layers,
        "n_tokens_used": n_tokens,
        "features_used": features_to_use,
        "accuracies": {f: {str(l): float(a) for l, a in accs.items()}
                       for f, accs in accuracies.items()},
        "test1_cross_layer": {
            f: {"mean_pairwise_cos": v["mean_pairwise_cos"],
                "mean_adjacent_cos": v["mean_adjacent_cos"]}
            for f, v in cross_layer.items()
        },
        "test2_orthogonality": {
            str(l): v for l, v in orthogonality.items()
        },
        "test3_stability": stability,
        "summary": {},
    }

    # Compute summary
    results["summary"]["cross_layer_pairwise_mean"] = float(np.mean(all_pairwise))
    results["summary"]["cross_layer_pairwise_std"] = float(np.std(all_pairwise))
    results["summary"]["cross_layer_adjacent_mean"] = float(np.mean(all_adjacent))
    results["summary"]["cross_layer_adjacent_std"] = float(np.std(all_adjacent))
    if all_means:
        results["summary"]["orthogonality_mean_abs_cos"] = float(np.mean(all_means))
        results["summary"]["orthogonality_std"] = float(np.std(all_means))
    if stability:
        results["summary"]["stability_mean"] = float(np.mean(all_stab))
        results["summary"]["stability_std"] = float(np.std(all_stab))

    # Mean probe accuracy
    all_accs = [a for f_accs in accuracies.values() for a in f_accs.values()]
    results["summary"]["mean_probe_accuracy"] = float(np.mean(all_accs))

    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed

    out_file = out_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_file}")
    print(f"Total time: {elapsed:.1f}s")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model_name}")
    print(f"{'='*60}")
    print(f"  Probe accuracy (mean):     {results['summary']['mean_probe_accuracy']:.4f}")
    print(f"  Cross-layer (pairwise):    {results['summary']['cross_layer_pairwise_mean']:.4f} ± {results['summary']['cross_layer_pairwise_std']:.4f}")
    print(f"  Cross-layer (adjacent):    {results['summary']['cross_layer_adjacent_mean']:.4f} ± {results['summary']['cross_layer_adjacent_std']:.4f}")
    if "orthogonality_mean_abs_cos" in results["summary"]:
        print(f"  Orthogonality |cos|:       {results['summary']['orthogonality_mean_abs_cos']:.4f} ± {results['summary']['orthogonality_std']:.4f}")
    if "stability_mean" in results["summary"]:
        print(f"  Direction stability:       {results['summary']['stability_mean']:.4f} ± {results['summary']['stability_std']:.4f}")


if __name__ == "__main__":
    main()
